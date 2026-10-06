import json
from pathlib import Path

import numpy as np

from system_one.robustness import broader_states, assess_readiness


def test_broader_probe_generation_is_seeded_and_has_boundaries():
    first, kinds = broader_states(42, 100)
    second, _ = broader_states(42, 100)
    np.testing.assert_array_equal(first, second)
    assert kinds.count("policy_boundary") == 50
    assert kinds.count("broad_combination") == 50


def test_readiness_never_authorizes_control_and_records_failure():
    criteria = json.loads(Path("experiments/readiness.json").read_text())
    reference = {"calibration": {"after": {"ece": .01}},
                 "high_confidence_errors": {"error_rate_among_confident": .01}}
    guarded = {"test_count": 100, "test": {"classification": {"accuracy": .99,
                   "classification_report": {"macro avg": {"f1-score": .99}}},
                   "coverage": .8, "accepted_accuracy": 1., "reasons": {}},
               "broader_distribution": {"accepted_accuracy": .8, "coverage": .6},
               "latency": {"p95_ms": .1}, "ood": {"one": {"escalation_rate": 1}},
               "simulator": {"scenarios": {"one": {"model": {"mean_cost": 1},
                 "rule": {"mean_cost": 1}, "no_action": {"mean_cost": 2}}}}}
    result = assess_readiness(reference, guarded, criteria)
    assert not result["passed"]
    assert not result["checks"]["broader_accepted_accuracy"]["passed"]
    assert not result["checks"]["real_labeled_telemetry"]["passed"]
    assert not result["automatic_network_control_allowed"]
