import json

from system_one.shadow import shadow_decision


def test_stale_and_failed_sources_escalate_without_fabricating_probabilities(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"not loaded when sources are stale")
    snapshot = {"collected_at": 10, "samples": [], "sources": [{"id": "amf", "ok": True}]}
    log = tmp_path / "audit.jsonl"
    first = shadow_decision(model, snapshot, None, log, now=100)
    assert first["decision"]["action"] == "ESCALATE"
    assert first["decision"]["confidence"] is None
    assert all(v is None for v in first["decision"]["probabilities"].values())
    assert not first["action_executed"]
    snapshot["sources"][0]["ok"] = False
    second = shadow_decision(model, snapshot, None, log, now=11)
    assert second["decision"]["escalation_reason"] == "source_unavailable"
    assert len(log.read_text().splitlines()) == 2
    assert json.loads(log.read_text().splitlines()[0]) == first
