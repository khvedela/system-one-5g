import csv

import numpy as np
import pytest

from system_one.decision import DecisionModel
from system_one.table_training import grouped_splits, train_table
from system_one.training import TrainingConfig


def test_groups_are_disjoint_and_splits_reproducible():
    groups = [f"episode-{i // 10}" for i in range(300)]
    first = grouped_splits(groups, 42)
    again = grouped_splits(groups, 42)
    seen = set()
    for name, ids in first.items():
        np.testing.assert_array_equal(ids, again[name])
        selected = {groups[i] for i in ids}
        assert not selected & seen
        seen.update(selected)
    assert len(seen) == 30


def test_train_more_than_six_metrics_and_escalate_missing_or_drift(tmp_path):
    rng = np.random.default_rng(4)
    names = [f"metric_{i:02d}" for i in range(24)]
    x = rng.uniform(1, 2, size=(300, len(names)))
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["action", "group_id", *names])
        for i, row in enumerate(x):
            writer.writerow(["NO_ACTION" if row[0] < 1.5 else "SCALE_OUT", f"episode-{i // 10}", *row])
    result = train_table(path, tmp_path / "model", TrainingConfig(epochs=2, hidden_sizes=(8, 4)))
    assert result["metric_count"] == 24
    assert result["input_width_including_missing_indicators"] == 48
    model = DecisionModel.load(tmp_path / "model/model.pt")
    state = dict(zip(names, [1.5] * len(names), strict=True))
    assert len(model.decide(state).probabilities) == 5
    assert model.decide({**state, "metric_new": 1}).escalation_reason == "schema_drift"
    state.pop(names[0])
    assert model.decide(state).escalation_reason == "missing_metrics"
    assert (tmp_path / "model/group_splits.json").is_file()


def test_support_guard_escalates_extremes_preserving_probabilities():
    import torch
    from dataclasses import asdict
    from system_one.actions import ACTION_NAMES
    from system_one.model import MLP
    from system_one.state import FEATURE_NAMES, State
    checkpoint = {"format_version": 1, "state_dict": MLP().state_dict(),
                  "feature_names": list(FEATURE_NAMES), "action_names": list(ACTION_NAMES),
                  "mean": [50, 50, 500, 100, .1, 2], "std": [20, 20, 300, 100, .1, 1],
                  "training_config": asdict(TrainingConfig()), "temperature": 1., "threshold": .8,
                  "support_guard": {"margin_std": .1},
                  "support_min": [0, 0, 0, 0, 0, 1], "support_max": [100, 100, 6000, 1100, 1, 10]}
    model = DecisionModel(checkpoint)
    result = model.decide(State(50, 50, 500, 9000, .01, 2))
    assert result.action == "ESCALATE"
    assert result.escalation_reason == "outside_training_support"
    assert sum(result.probabilities.values()) == pytest.approx(1)
