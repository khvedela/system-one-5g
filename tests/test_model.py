import torch
import pytest

from system_one.model import MLP
from system_one.decision import DecisionModel, select_decision
from system_one.state import State, FEATURE_NAMES
from system_one.actions import ACTION_NAMES
from system_one.training import TrainingConfig
from dataclasses import asdict


def test_mlp_forward_returns_five_finite_logits():
    logits = MLP()(torch.zeros(4, 6))
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_probability_contract_and_threshold_boundary():
    p = [.1, .05, .8, .03, .02]
    decision = select_decision(p, .8)
    assert decision.action == "SCALE_OUT"
    assert decision.confidence == .8
    assert sum(decision.probabilities.values()) == pytest.approx(1)
    uncertain = select_decision(p, .81)
    assert uncertain.action == "ESCALATE"
    assert uncertain.escalation_reason == "low_confidence"
    assert uncertain.probabilities == decision.probabilities
    assert uncertain.confidence == .8
    assert select_decision([0, 0, 0, 0, 1], .9).escalation_reason == "predicted_escalate"


@pytest.mark.parametrize("p,threshold", [([1, 0], .8), ([.2] * 5, 2),
    ([.1] * 5, .8), ([float("nan"), 0, 0, 0, 1], .8)])
def test_invalid_probabilities_and_threshold(p, threshold):
    with pytest.raises(ValueError):
        select_decision(p, threshold)


def test_saved_model_public_api_matches_loaded_model(tmp_path):
    checkpoint = {
        "format_version": 1, "state_dict": MLP().state_dict(),
        "mean": [0.] * 6, "std": [1.] * 6,
        "feature_names": list(FEATURE_NAMES), "action_names": list(ACTION_NAMES),
        "training_config": asdict(TrainingConfig()), "temperature": 1.3, "threshold": .8,
    }
    state = State(50, 60, 500, 100, .01, 2)
    before = DecisionModel(checkpoint).decide(state)
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    assert DecisionModel.load(path).decide(state.as_dict()) == before
