"""Threshold comparisons distinguish policy accuracy from selective risk."""

import numpy as np

from .actions import Action
from .evaluation import classification_metrics


DEFAULT_THRESHOLDS = (0.0, .5, .6, .7, .8, .9, .95, .99, 1.0)


def uncertainty_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    confidence = probabilities.max(axis=1)
    raw = probabilities.argmax(axis=1)
    forced = confidence < threshold
    decisions = np.where(forced, int(Action.ESCALATE), raw)
    accepted = decisions != int(Action.ESCALATE)
    return {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "accepted_count": int(accepted.sum()),
        "accepted_accuracy": float((decisions[accepted] == labels[accepted]).mean()) if accepted.any() else None,
        "escalation_rate": float((~accepted).mean()),
        "forced_escalation_rate": float(forced.mean()),
        "classification": classification_metrics(labels, decisions),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray,
                     target_accuracy: float = .98, min_accepted: int = 30,
                     thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
                     minimum_threshold: float = .8) -> tuple[float, list[dict], bool]:
    if not 0 < target_accuracy <= 1 or min_accepted < 1:
        raise ValueError("invalid threshold selection target")
    if not thresholds or any(not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError("threshold candidates must be in [0, 1]")
    if not 0 <= minimum_threshold <= max(thresholds):
        raise ValueError("minimum_threshold must be covered by candidates")
    results = [uncertainty_metrics(labels, probabilities, t) for t in thresholds]
    eligible = [row for row in results if row["threshold"] >= minimum_threshold
                and row["accepted_count"] >= min_accepted
                and row["accepted_accuracy"] >= target_accuracy]
    if not eligible:
        return max(thresholds), results, False
    chosen = max(eligible, key=lambda row: (row["coverage"], -row["threshold"]))
    return chosen["threshold"], results, True
