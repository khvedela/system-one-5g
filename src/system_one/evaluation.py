"""Metrics always use a fixed five-action class order."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .actions import ACTION_NAMES
from .data import label_states


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    classes = np.arange(len(ACTION_NAMES))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "classification_report": classification_report(
            labels, predictions, labels=classes, target_names=ACTION_NAMES,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "class_order": ACTION_NAMES,
    }


def baseline_metrics(observed: np.ndarray, labels: np.ndarray) -> dict:
    return classification_metrics(labels, label_states(observed))


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> list[dict]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    results = []
    for i in range(bins):
        mask = assignments == i
        count = int(mask.sum())
        results.append({"lower": i / bins, "upper": (i + 1) / bins, "count": count,
                        "confidence": float(confidence[mask].mean()) if count else None,
                        "accuracy": float(correct[mask].mean()) if count else None})
    return results


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    targets = np.eye(len(ACTION_NAMES))[labels]
    reliability = reliability_bins(labels, probabilities)
    ece = sum(row["count"] / len(labels) * abs(row["accuracy"] - row["confidence"])
              for row in reliability if row["count"])
    return {
        "brier_score": float(np.square(probabilities - targets).sum(axis=1).mean()),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
        "ece": float(ece), "reliability_bins": reliability,
    }


def save_reliability_diagram(path, before: dict, after: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, (name, metrics) in zip(axes, (("Before temperature scaling", before),
                                           ("After temperature scaling", after)), strict=True):
        rows = [row for row in metrics["reliability_bins"] if row["count"]]
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        axis.scatter([r["confidence"] for r in rows], [r["accuracy"] for r in rows],
                     s=[20 + 100 * r["count"] / sum(t["count"] for t in rows) for r in rows])
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy",
                 title=f"{name}\nECE = {metrics['ece']:.4f}")
        axis.grid(alpha=.2)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def high_confidence_errors(features: np.ndarray, clean: np.ndarray, labels: np.ndarray,
                           probabilities: np.ndarray, row_ids: np.ndarray, threshold: float,
                           confidence_cutoff: float = .9) -> dict:
    from .data import state_from_row
    from .decision import select_decision
    from .policy import ground_truth

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    confident = confidence >= confidence_cutoff
    wrong = confident & (predicted != labels)
    errors = []
    for i in np.flatnonzero(wrong):
        state = state_from_row(features[i])
        decision = select_decision(probabilities[i].tolist(), threshold)
        observed_policy = ground_truth(state)
        errors.append({"row_id": int(row_ids[i]), "state": state.as_dict(),
                       "clean_state": state_from_row(clean[i]).as_dict(),
                       "predicted_action": ACTION_NAMES[predicted[i]],
                       "correct_action": ACTION_NAMES[labels[i]],
                       "observed_policy_action": observed_policy.name,
                       "failure_type": "measurement_noise_boundary" if int(observed_policy) != labels[i]
                                       else "model_policy_mismatch",
                       **decision.as_dict()})
    return {"confidence_cutoff": confidence_cutoff, "error_count": len(errors),
            "confident_count": int(confident.sum()),
            "error_rate_among_confident": float(wrong.sum() / confident.sum()) if confident.any() else None,
            "error_rate_overall": float(wrong.mean()), "cases": errors}
