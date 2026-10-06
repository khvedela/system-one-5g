"""Join captured telemetry with explicit external labels without inventing truth."""

import csv
import hashlib
import json
from pathlib import Path

from .actions import ACTION_NAMES
from .features import snapshot_features, schema_hash


def assemble_table(annotation_path: Path, schema_path: Path, output: Path) -> dict:
    annotations = json.loads(annotation_path.read_text())
    schema = json.loads(schema_path.read_text())
    names = schema["feature_names"]
    if schema["sha256"] != schema_hash(names) or not annotations.get("label_origin"):
        raise ValueError("a valid frozen schema and explicit label_origin are required")
    records = annotations.get("records", [])
    if not records:
        raise ValueError("no labeled snapshots")
    rows, metadata, seen = [], [], set()
    for record in records:
        path = (annotation_path.parent / record["snapshot"]).resolve()
        if path in seen:
            raise ValueError("duplicate snapshot annotations")
        seen.add(path)
        if record["action"] not in ACTION_NAMES or not record["group_id"]:
            raise ValueError("each snapshot needs a valid action and independent group_id")
        snapshot = json.loads(path.read_text())
        if not snapshot.get("sources") or any(not source["ok"] for source in snapshot["sources"]):
            raise ValueError("failed telemetry cannot be labeled as a complete observation")
        previous_path = (annotation_path.parent / record["previous"]).resolve() if record.get("previous") else None
        previous = json.loads(previous_path.read_text()) if previous_path else None
        values = snapshot_features(snapshot, previous)
        unknown = set(values) - set(names)
        if unknown:
            raise ValueError("snapshot has features outside the frozen schema; review schema drift")
        rows.append([record["action"], record["group_id"], *(values.get(name) for name in names)])
        metadata.append({"snapshot": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "previous_snapshot": str(previous_path) if previous_path else None,
                         "previous_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest() if previous_path else None,
                         "action": record["action"], "group_id": record["group_id"]})
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("dataset output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["action", "group_id", *names])
        writer.writerows(rows)
    result = {"schema": schema, "label_origin": annotations["label_origin"],
              "annotations_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
              "records": metadata}
    with sidecar.open("x") as stream:
        json.dump(result, stream, indent=2)
    return {"rows": len(rows), "metric_features": len(names), "label_origin": annotations["label_origin"]}
