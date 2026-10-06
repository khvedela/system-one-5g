"""Discover every exposed numeric series; retain raw telemetry for replay."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.openmetrics.parser import text_string_to_metric_families as openmetrics_parser


def series_key(source: str, name: str, labels: dict) -> str:
    return json.dumps([source, name, sorted(labels.items())], separators=(",", ":"))


def parse_metrics(text: str, source: str, openmetrics: bool = False) -> list[dict]:
    parser = openmetrics_parser if openmetrics else text_string_to_metric_families
    samples, seen = [], set()
    exposed_names = set(re.findall(r"^([A-Za-z_:][A-Za-z0-9_:]*)[\s{]", text, flags=re.MULTILINE))
    for family in parser(text):
        for sample in family.samples:
            name = sample.name
            if name not in exposed_names and name.endswith("_total") and name[:-6] in exposed_names:
                name = name[:-6]
            key = series_key(source, name, sample.labels)
            if key in seen:
                raise ValueError(f"duplicate metric series: {sample.name}")
            seen.add(key)
            cumulative = ((family.type == "counter" and not sample.name.endswith("_created"))
                          or (family.type in ("histogram", "summary")
                              and sample.name.endswith(("_bucket", "_sum", "_count"))))
            value = float(sample.value)
            samples.append({"key": key, "source": source, "name": name, "parser_name": sample.name,
                            "family": family.name, "type": family.type,
                            "help": family.documentation, "unit": getattr(family, "unit", ""),
                            "labels": dict(sample.labels), "cumulative": cumulative,
                            "value": value if math.isfinite(value) else None,
                            "raw_value": str(sample.value),
                            "sample_timestamp": float(sample.timestamp) if sample.timestamp is not None else None})
    return samples


def flatten_numeric(value, prefix: str = "") -> dict[str, float]:
    result = {}
    if isinstance(value, dict):
        for name, item in sorted(value.items()):
            result.update(flatten_numeric(item, f"{prefix}/{name}"))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            identity = json.dumps([item["major"], item["minor"], item["op"]], separators=(",", ":")) \
                if isinstance(item, dict) and {"major", "minor", "op"} <= set(item) else str(i)
            result.update(flatten_numeric(item, f"{prefix}/{identity}"))
    elif isinstance(value, (int, float, bool)) and math.isfinite(value):
        result[prefix] = float(value)
    return result


def docker_snapshot(options: dict, timeout: float) -> tuple[list[dict], dict]:
    import docker
    if options.get("base_url"):
        client = docker.DockerClient(base_url=options["base_url"], timeout=timeout)
    elif os.environ.get("DOCKER_HOST"):
        client = docker.from_env(timeout=timeout)
    else:
        context = subprocess.run(["docker", "context", "inspect", "--format", "{{json .Endpoints.docker}}"],
                                 capture_output=True, text=True, check=True, timeout=timeout)
        endpoint = json.loads(context.stdout)
        client = docker.DockerClient(base_url=endpoint["Host"], timeout=timeout)
    samples, raw = [], {}
    try:
        containers = client.containers.list(all=True, filters=options.get("filters", {}))
        for container in containers:
            labels = {"container_id": container.id, "container_name": container.name}
            attrs = container.attrs
            # Container configuration can contain credentials; retain only resource/lifecycle fields.
            info = {"state": {k: v for k, v in attrs.get("State", {}).items()
                              if k in ("Running", "Paused", "Restarting", "OOMKilled", "Dead", "Pid", "ExitCode",
                                       "StartedAt", "FinishedAt")},
                    "restart_count": attrs.get("RestartCount", 0),
                    "limits": {k: v for k, v in attrs.get("HostConfig", {}).items()
                               if k in ("Memory", "MemorySwap", "MemoryReservation", "NanoCpus", "CpuQuota",
                                        "CpuPeriod", "CpuShares", "PidsLimit", "OomKillDisable")}}
            if attrs.get("State", {}).get("Running"):
                try:
                    info["stats"] = container.stats(stream=False)
                except docker.errors.DockerException as error:
                    info["stats_error"] = type(error).__name__
            raw[container.id] = info
            for name, value in flatten_numeric(info).items():
                cumulative = (name.startswith("/stats/networks/")
                              or (name.startswith("/stats/cpu_stats/")
                                  and ("/cpu_usage/" in name or "/throttling_data/" in name))
                              or (name.startswith("/stats/blkio_stats/") and name.endswith("/value")))
                samples.append({"key": series_key("docker", name, labels), "source": "docker",
                                "name": name, "type": "counter" if cumulative else "untyped", "labels": labels,
                                "cumulative": cumulative, "value": value,
                                "help": "Docker Engine numeric field; see retained raw stats"})
            stats = info.get("stats", {})
            cpu = stats.get("cpu_stats", {})
            previous = stats.get("precpu_stats", {})
            cpu_delta = cpu.get("cpu_usage", {}).get("total_usage", 0) - previous.get("cpu_usage", {}).get("total_usage", 0)
            system_delta = cpu.get("system_cpu_usage", 0) - previous.get("system_cpu_usage", 0)
            online = cpu.get("online_cpus") or len(cpu.get("cpu_usage", {}).get("percpu_usage", []))
            if ("cpu_usage" in previous and "system_cpu_usage" in previous
                    and cpu_delta >= 0 and system_delta > 0 and online):
                name = "derived/cpu_percent_of_one_core"
                samples.append({"key": series_key("docker", name, labels), "source": "docker", "name": name,
                                "type": "gauge", "labels": labels, "cumulative": False,
                                "value": cpu_delta / system_delta * online * 100,
                                "help": "Docker CPU percent; may exceed 100 on multicore workloads"})
    finally:
        client.close()
    return samples, raw


def capture(config: dict, output: Path) -> dict:
    timeout = float(config.get("timeout_seconds", 5))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    targets = config.get("targets", [])
    ids = [target["id"] for target in targets]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        raise ValueError("target IDs must be unique safe names")
    if not targets and not config.get("docker", {}).get("enabled"):
        raise ValueError("configure at least one metrics endpoint or enable Docker")
    for target in targets:
        url = urlsplit(target["url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query:
            raise ValueError("use an HTTP(S) URL without embedded credentials/query parameters")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {"schema_version": 1, "collected_at": time.time(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "samples": [], "sources": []}
    for target in targets:
        started = time.time()
        status = {"id": target["id"], "url": target["url"], "ok": False, "collected_at": started}
        try:
            headers = {"Accept": "text/plain;version=0.0.4,application/openmetrics-text;version=1.0.0;q=0.8"}
            if target.get("bearer_token_env"):
                headers["Authorization"] = "Bearer " + os.environ[target["bearer_token_env"]]
            context = ssl.create_default_context(cafile=target.get("ca_file"))
            with urlopen(Request(target["url"], headers=headers), timeout=timeout, context=context) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type", "text/plain")
            (output / f"{target['id']}.metrics").write_bytes(payload)
            samples = parse_metrics(payload.decode("utf-8"), target["id"], "openmetrics" in content_type)
            for sample in samples:
                sample["collected_at"] = started
            snapshot["samples"].extend(samples)
            status.update(ok=True, sample_count=len(samples), content_type=content_type,
                          sha256=hashlib.sha256(payload).hexdigest())
        except Exception as error:
            # Error strings from HTTP libraries may include authentication material.
            status["error_type"] = type(error).__name__
        snapshot["sources"].append(status)
    if config.get("docker", {}).get("enabled"):
        started = time.time()
        status = {"id": "docker", "ok": False, "collected_at": started}
        try:
            samples, raw = docker_snapshot(config["docker"], timeout)
            (output / "docker-stats.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
            for sample in samples:
                sample["collected_at"] = started
            snapshot["samples"].extend(samples)
            errors = [key for key, item in raw.items() if "stats_error" in item]
            status.update(ok=not errors, sample_count=len(samples), containers=len(raw), failed_containers=errors)
        except Exception as error:
            status["error_type"] = type(error).__name__
        snapshot["sources"].append(status)
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    inventory = [{k: v for k, v in sample.items() if k not in ("value", "raw_value", "sample_timestamp", "collected_at")}
                 for sample in snapshot["samples"]]
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return snapshot
