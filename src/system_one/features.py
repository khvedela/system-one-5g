"""Freeze all discovered series and derive reset-aware counter rates."""

import hashlib
import json
import math


def snapshot_features(snapshot: dict, previous: dict | None = None) -> dict[str, float | None]:
    current = {sample["key"]: sample for sample in snapshot["samples"]}
    if len(current) != len(snapshot["samples"]):
        raise ValueError("duplicate series keys")
    old = {sample["key"]: sample for sample in previous["samples"]} if previous else {}
    features = {}
    for key, sample in current.items():
        value = sample["value"]
        features[key + "/value"] = value
        if sample.get("cumulative"):
            before = old.get(key)
            dt = (sample.get("collected_at", snapshot["collected_at"])
                  - before.get("collected_at", previous["collected_at"])) if before else 0
            valid = (before is not None and before["value"] is not None and value is not None and dt > 0)
            reset = valid and value < before["value"]
            features[key + "/rate_per_second"] = (value - before["value"]) / dt if valid and not reset else None
            features[key + "/counter_reset"] = float(reset) if valid else None
    return features


def schema_hash(names: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(names), separators=(",", ":")).encode()).hexdigest()


def discover_schema(snapshots: list[dict]) -> dict:
    if not snapshots:
        raise ValueError("at least one training snapshot is required")
    ordered = sorted(snapshots, key=lambda s: s["collected_at"])
    names = set()
    previous = None
    for snapshot in ordered:
        if any(not source["ok"] for source in snapshot["sources"]):
            raise ValueError("cannot discover a complete schema from failed sources")
        names.update(snapshot_features(snapshot, previous))
        previous = snapshot
    names = sorted(names)
    if not names:
        raise ValueError("no numeric metrics were discovered")
    return {"version": 1, "feature_names": names, "sha256": schema_hash(names),
            "derived_model_names": [*names, *(name + "/missing" for name in names)],
            "policy": "all observed series; values plus missing indicators; schema changes require retraining"}


def encode_features(values: dict, schema: dict, imputation: list[float]) -> tuple[list[float], list[str], list[str]]:
    names = schema["feature_names"]
    if schema.get("sha256") != schema_hash(names) or len(imputation) != len(names):
        raise ValueError("feature schema or imputation does not match")
    if any(not math.isfinite(value) for value in imputation):
        raise ValueError("imputation must be fitted finite training values")
    missing = [name for name in names if values.get(name) is None or not math.isfinite(values[name])]
    missing_set = set(missing)
    extras = sorted(set(values) - set(names))
    encoded = [float(values[name]) if name not in missing_set else imputation[i] for i, name in enumerate(names)]
    return encoded + [float(name in missing_set) for name in names], missing, extras
