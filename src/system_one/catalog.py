"""Extract version-pinned definitions, including Open5GS C macro entries."""

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def provenance(root: Path, repository: str) -> dict:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--exact-match"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return {"repository": repository, "tag": tag, "commit": commit}


def extract_open5gs(text: str) -> list[dict]:
    # Only preprocess local macro definitions; no build or upstream includes are needed.
    stripped = re.sub(r"^\s*#\s*include[^\n]*", "", text, flags=re.MULTILINE)
    expanded = subprocess.run(["cc", "-E", "-P", "-x", "c", "-"], input=stripped,
                              capture_output=True, text=True, check=True).stdout
    arrays = {name: re.findall(r'"([^"\\]*)"', body)
              for name, body in re.findall(r"const\s+char\s*\*\s*(\w+)\s*\[\s*\]\s*=\s*\{(.*?)\};",
                                          expanded, flags=re.DOTALL)}
    entries = []
    for match in re.finditer(r"\[([A-Z][A-Za-z0-9_]+)\]\s*=\s*\{", expanded):
        depth, cursor, quoted, escaped = 1, match.end(), False, False
        while depth and cursor < len(expanded):
            char = expanded[cursor]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        body = expanded[match.end():cursor - 1]
        kind = re.search(r"\.type\s*=\s*OGS_METRICS_METRIC_TYPE_(\w+)", body)
        name = re.search(r'\.name\s*=\s*("(?:[^"\\]|\\.)*")', body)
        if not kind or not name:
            continue
        description = re.search(r'\.description\s*=\s*("(?:[^"\\]|\\.)*")', body)
        labels = re.search(r"\.labels\s*=\s*(\w+)", body)
        label_names = arrays[labels.group(1)] if labels else []
        histogram = re.search(r"\.histogram_params\s*=\s*\{(.*)\}", body, flags=re.DOTALL)
        metric_name = ast.literal_eval(name.group(1))
        line = next((i for i, line in enumerate(text.splitlines(), 1) if f'"{metric_name}"' in line), None)
        entries.append({"symbol": match.group(1), "name": metric_name,
                        "type": kind.group(1).lower(), "labels": label_names,
                        "description": ast.literal_eval(description.group(1)) if description else "",
                        "histogram_definition": histogram.group(1).strip() if histogram else None,
                        "line": line})
    declared = len(re.findall(r'\.name\s*=\s*"', expanded))
    if len(entries) != declared or len({e["name"] for e in entries}) != len(entries):
        raise ValueError(f"unrecognized/duplicate Open5GS declarations: extracted {len(entries)} of {declared}; "
                         "extraction must be reviewed")
    return entries


def markdown_metrics(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if "|" not in line:
            continue
        # Escaped pipes occur in label-value documentation.
        columns = [part.strip() for part in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        columns[0] = columns[0].strip("`")
        if len(columns) < 3 or not re.fullmatch(r"(?:container|machine|kube)_[A-Za-z0-9_:]+", columns[0]):
            continue
        rows.append({"name": columns[0], "type": columns[1].lower(),
                     "description": columns[2], "documentation_columns": columns[3:], "line": line_number})
    return rows


def build_catalog(open5gs: Path, cadvisor: Path, kube_state_metrics: Path, output: Path) -> dict:
    origins = {
        "open5gs": provenance(open5gs, "https://github.com/open5gs/open5gs"),
        "cadvisor": provenance(cadvisor, "https://github.com/google/cadvisor"),
        "kube_state_metrics": provenance(kube_state_metrics, "https://github.com/kubernetes/kube-state-metrics"),
    }
    entries, files, nfs = [], {}, {}
    for config in sorted((open5gs / "configs/open5gs").glob("*.yaml.in")):
        nf = config.name.removesuffix(".yaml.in")
        path = open5gs / "src" / nf / "metrics.c"
        if not path.exists():
            nfs[nf] = {"metrics_c": False, "defined_families": 0}
            continue
        metrics = extract_open5gs(path.read_text())
        nfs[nf] = {"metrics_c": True, "defined_families": len(metrics)}
        relative = str(path.relative_to(open5gs))
        files["open5gs/" + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for metric in metrics:
            entries.append({"source": "open5gs", "nf": nf, "file": relative,
                            "source_url": f"{origins['open5gs']['repository']}/blob/{origins['open5gs']['commit']}/{relative}",
                            **metric})
    for origin, root, paths in (
        ("cadvisor", cadvisor, [cadvisor / "docs/storage/prometheus.md"]),
        ("kube_state_metrics", kube_state_metrics, sorted((kube_state_metrics / "docs/metrics").rglob("*.md"))),
    ):
        for path in paths:
            relative = str(path.relative_to(root))
            files[origin + "/" + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            for metric in markdown_metrics(path):
                entries.append({"source": origin, "file": relative,
                                "source_url": f"{origins[origin]['repository']}/blob/{origins[origin]['commit']}/{relative}",
                                **metric})
    counts = {origin: sum(e["source"] == origin for e in entries) for origin in origins}
    if any(count == 0 for count in counts.values()):
        raise ValueError("empty source inventory; upstream documentation layout may have changed")
    catalog = {"version": 1, "origins": origins, "counts": counts, "open5gs_nfs": nfs,
               "source_sha256": files, "metrics": entries,
               "scope": "source-defined/documented families, not a claim of runtime availability; "
                        "collect inventories actual label series and all additional exposed metrics"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(catalog, stream, indent=2)
        stream.write("\n")
    return catalog
