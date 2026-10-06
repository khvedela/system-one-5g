"""Read-only decisions with freshness checks and append-only audit records."""

import hashlib
from copy import deepcopy
import json
from pathlib import Path
import time

from .actions import ACTION_NAMES
from .decision import DecisionModel
from .features import snapshot_features


def shadow_decision(model_path: Path, snapshot: dict, previous: dict | None,
                    log_path: Path, max_age_seconds: float = 30, now: float | None = None) -> dict:
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")
    now = time.time() if now is None else now
    reason = None
    if not snapshot.get("sources") or any(not source["ok"] for source in snapshot["sources"]):
        reason = "source_unavailable"
    elif any(not 0 <= now - source.get("collected_at", snapshot["collected_at"]) <= max_age_seconds
             for source in snapshot["sources"]):
        reason = "stale_or_invalid_timestamp"
    if reason:
        decision = {"action": "ESCALATE", "confidence": None,
                    "probabilities": {name: None for name in ACTION_NAMES}, "escalation_reason": reason}
    else:
        model = DecisionModel.load(model_path)
        if not model.telemetry_schema:
            raise ValueError("shadow telemetry requires a model trained with train-metrics")
        decision = model.decide(snapshot_features(snapshot, previous)).as_dict()
    record = {"mode": "shadow", "action_executed": False, "timestamp": now,
              "snapshot_timestamp": snapshot["collected_at"], "sources": deepcopy(snapshot["sources"]),
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "decision": decision}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    return record
