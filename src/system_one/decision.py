"""Stable public API independent of the neural architecture."""

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from collections.abc import Mapping

import torch

from .actions import Action, ACTION_NAMES
from .state import State, FEATURE_NAMES
from .training import network_from_checkpoint


@dataclass(frozen=True)
class Decision:
    action: str
    confidence: float
    probabilities: dict[str, float]
    escalation_reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def select_decision(probabilities: list[float], threshold: float) -> Decision:
    if len(probabilities) != len(ACTION_NAMES) or any(not math.isfinite(p) or p < 0 or p > 1
                                                   for p in probabilities):
        raise ValueError("invalid action probabilities")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = probabilities[best]
    uncertain = confidence < threshold
    action = Action.ESCALATE if uncertain else Action(best)
    reason = "low_confidence" if uncertain else ("predicted_escalate" if action == Action.ESCALATE else None)
    return Decision(action.name, confidence, dict(zip(ACTION_NAMES, probabilities, strict=True)), reason)


class DecisionModel:
    def __init__(self, checkpoint: dict):
        if checkpoint.get("format_version") not in (1, 2):
            raise ValueError("unsupported model format")
        self.feature_names = tuple(checkpoint["feature_names"])
        if ((checkpoint["format_version"] == 1 and self.feature_names != FEATURE_NAMES)
                or not self.feature_names or len(set(self.feature_names)) != len(self.feature_names)
                or checkpoint["action_names"] != list(ACTION_NAMES)):
            raise ValueError("model feature/action contract does not match")
        self.temperature = float(checkpoint["temperature"])
        self.threshold = float(checkpoint["threshold"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.mean = torch.tensor(checkpoint["mean"], dtype=torch.float32)
        self.std = torch.tensor(checkpoint["std"], dtype=torch.float32)
        if (self.mean.shape != (len(self.feature_names),) or self.std.shape != self.mean.shape or not torch.isfinite(self.mean).all()
                or not torch.isfinite(self.std).all() or (self.std <= 0).any()):
            raise ValueError("invalid normalization parameters")
        self.network = network_from_checkpoint(checkpoint)
        self.telemetry_schema = checkpoint.get("telemetry_schema")
        self.imputation = checkpoint.get("imputation")
        if self.telemetry_schema and self.feature_names != tuple([
                *self.telemetry_schema["feature_names"],
                *(name + "/missing" for name in self.telemetry_schema["feature_names"])]):
            raise ValueError("telemetry schema does not match model input order")
        self.support = checkpoint.get("support_guard")
        if self.support:
            self.lower = torch.tensor(checkpoint["support_min"], dtype=torch.float32)
            self.upper = torch.tensor(checkpoint["support_max"], dtype=torch.float32)
            if (self.lower.shape != self.mean.shape or self.upper.shape != self.mean.shape
                    or not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all()
                    or (self.lower > self.upper).any() or not math.isfinite(self.support["margin_std"])
                    or self.support["margin_std"] < 0):
                raise ValueError("support bounds must match features")
        torch.set_num_threads(checkpoint["training_config"]["threads"])

    @classmethod
    def load(cls, path: str | Path) -> "DecisionModel":
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def decide(self, state: State | Mapping) -> Decision:
        missing, extra = [], []
        if self.telemetry_schema:
            if not isinstance(state, Mapping):
                raise ValueError("telemetry state must be a metric mapping")
            from .features import encode_features
            encoded, missing, extra = encode_features(state, self.telemetry_schema, self.imputation)
            state = dict(zip(self.feature_names, encoded, strict=True))
        if self.feature_names == FEATURE_NAMES:
            if not isinstance(state, State):
                state = State(**state)
            values = torch.tensor(state.values(), dtype=torch.float32)
        else:
            if not isinstance(state, Mapping) or set(state) != set(self.feature_names):
                raise ValueError("state must exactly match the saved feature schema")
            items = [state[name] for name in self.feature_names]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in items):
                raise ValueError("features must be finite numeric observations")
            values = torch.tensor(items, dtype=torch.float32)
        with torch.inference_mode():
            logits = self.network((values - self.mean) / self.std)
            probabilities = torch.softmax(logits / self.temperature, dim=-1).tolist()
        decision = select_decision(probabilities, self.threshold)
        if missing or extra:
            return Decision("ESCALATE", decision.confidence, decision.probabilities,
                            "missing_metrics" if missing else "schema_drift")
        if self.support:
            margin = self.support["margin_std"] * self.std
            if ((values < self.lower - margin) | (values > self.upper + margin)).any():
                return Decision("ESCALATE", decision.confidence, decision.probabilities, "outside_training_support")
        return decision
