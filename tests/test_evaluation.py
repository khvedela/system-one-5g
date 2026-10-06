import numpy as np
import pytest

from system_one.data import DatasetConfig, generate_clean, label_states
from system_one.evaluation import baseline_metrics, classification_metrics
from system_one.evaluation import probability_metrics
from system_one.evaluation import high_confidence_errors


def test_clean_oracle_baseline_is_perfect():
    clean, _ = generate_clean(DatasetConfig(samples=100))
    assert baseline_metrics(clean, label_states(clean))["accuracy"] == 1


def test_metrics_keep_absent_classes_and_known_confusions():
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert result["accuracy"] == .75
    assert result["confusion_matrix"][0][:2] == [1, 1]
    assert len(result["confusion_matrix"]) == 5
    assert result["classification_report"]["SCALE_IN"]["recall"] == 1


def test_calibration_metrics_on_known_distributions():
    labels = np.array([0, 1, 2, 3, 4])
    perfect = probability_metrics(labels, np.eye(5))
    assert perfect["brier_score"] == 0
    assert perfect["nll"] == 0
    assert perfect["ece"] == 0
    uniform = probability_metrics(labels, np.full((5, 5), .2))
    assert uniform["brier_score"] == pytest.approx(.8)
    assert uniform["nll"] == pytest.approx(np.log(5))
    assert uniform["ece"] == pytest.approx(0)
    assert sum(row["count"] for row in perfect["reliability_bins"]) == 5


def test_confident_failure_records_label_state_and_forced_escalation():
    states = np.array([[50, 50, 500, 100, .01, 2]])
    probabilities = np.array([[.01, .01, .95, .01, .02]])
    failures = high_confidence_errors(states, states, np.array([0]), probabilities,
                                     np.array([123]), threshold=.99)
    assert failures["error_count"] == 1
    assert failures["error_rate_among_confident"] == 1
    case = failures["cases"][0]
    assert case["row_id"] == 123
    assert case["predicted_action"] == "SCALE_OUT"
    assert case["action"] == "ESCALATE"
    assert case["correct_action"] == "NO_ACTION"
    assert case["failure_type"] == "model_policy_mismatch"
