"""One reproducible, local, end-to-end experiment; no action execution."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .actions import ACTION_NAMES
from .calibration import fit_temperature
from .data import (DatasetConfig, generate_clean, add_noise, label_states,
                   split_indices, save_dataset, save_splits, state_from_row)
from .decision import DecisionModel
from .evaluation import (baseline_metrics, classification_metrics, probability_metrics,
                         save_reliability_diagram, high_confidence_errors)
from .ood import ood_states
from .policy import ground_truth
from .training import TrainingConfig, train, checkpoint_logits
from .uncertainty import select_threshold, uncertainty_metrics, DEFAULT_THRESHOLDS


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment_metadata() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    root = Path(__file__).resolve().parents[2]
    files = [*sorted(Path(__file__).parent.glob("*.py")), root / "pyproject.toml", root / "uv.lock"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "device": "cpu", "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "torch", "scikit-learn", "matplotlib")},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in files if p.exists()},
    }


def benchmark(model: DecisionModel, states: list[dict], iterations: int = 1000) -> dict:
    for i in range(100):
        model.decide(states[i % len(states)])
    durations = []
    for i in range(iterations):
        start = time.perf_counter_ns()
        model.decide(states[i % len(states)])
        durations.append((time.perf_counter_ns() - start) / 1e6)
    return {"iterations": iterations, "warmup": 100, "batch_size": 1,
            "includes": "state validation, normalization, forward pass, softmax and decision selection",
            "excludes": "model loading and JSON/file I/O", "device": "cpu",
            "mean_ms": float(np.mean(durations)), "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95)), "p99_ms": float(np.percentile(durations, 99))}


def evaluate_ood(model: DecisionModel, seed: int, confidence_cutoff: float) -> dict:
    results = {}
    for name, states in ood_states(seed).items():
        decisions = [model.decide(state) for state in states]
        cases = [{"state": state.as_dict(), "correct_action": ground_truth(state).name,
                  "raw_action": max(decision.probabilities, key=decision.probabilities.get),
                  **decision.as_dict()} for state, decision in zip(states, decisions, strict=True)]
        results[name] = {
            "count": len(cases),
            "mean_confidence": float(np.mean([d.confidence for d in decisions])),
            "escalation_rate": sum(d.action == "ESCALATE" for d in decisions) / len(decisions),
            "policy_agreement": sum(c["action"] == c["correct_action"] for c in cases) / len(cases),
            "raw_high_confidence_error_count": sum(c["confidence"] >= confidence_cutoff
                                                  and c["raw_action"] != c["correct_action"] for c in cases),
            "cases": cases,
        }
    return results


def run_experiment(output: Path, config: dict) -> dict:
    allowed = {"dataset", "training", "uncertainty", "evaluation"}
    if set(config) - allowed:
        raise ValueError(f"unknown config sections: {sorted(set(config) - allowed)}")
    data_config = DatasetConfig(**config.get("dataset", {}))
    training_values = dict(config.get("training", {}))
    training_values.setdefault("seed", data_config.seed)
    if "hidden_sizes" in training_values:
        training_values["hidden_sizes"] = tuple(training_values["hidden_sizes"])
    training_config = TrainingConfig(**training_values)
    uncertainty_config = {"target_accuracy": .98, "min_accepted": 30,
                          "thresholds": DEFAULT_THRESHOLDS, "minimum_threshold": .8,
                          **config.get("uncertainty", {})}
    evaluation_config = {"confidence_cutoff": .9, "latency_iterations": 1000,
                         **config.get("evaluation", {})}
    if set(evaluation_config) != {"confidence_cutoff", "latency_iterations"}:
        raise ValueError("unknown evaluation configuration")
    if not 0 <= evaluation_config["confidence_cutoff"] <= 1:
        raise ValueError("confidence_cutoff must be in [0, 1]")
    iterations = evaluation_config["latency_iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("latency_iterations must be a positive integer")
    clean, scenarios = generate_clean(data_config)
    labels = label_states(clean)
    observed = add_noise(clean, data_config)
    splits = split_indices(labels, data_config.seed)
    output.mkdir(parents=True, exist_ok=False)
    resolved_config = {"dataset": asdict(data_config), "training": asdict(training_config),
                       "uncertainty": uncertainty_config, "evaluation": evaluation_config}
    write_json(output / "config.json", resolved_config)
    write_json(output / "environment.json", environment_metadata())
    save_dataset(output / "dataset", observed, clean, labels, scenarios, data_config)
    save_splits(output / "dataset", splits)
    print(f"Generated {len(labels)} states; training on {len(splits['train'])} CPU samples", flush=True)
    checkpoint, history = train(observed[splits["train"]], labels[splits["train"]],
                                observed[splits["selection"]], labels[splits["selection"]], training_config)
    write_json(output / "training.json", history)
    print(f"Training stopped at epoch {len(history)}; selected epoch {checkpoint['best_epoch']}", flush=True)
    calibration_logits = checkpoint_logits(checkpoint, observed[splits["calibration"]])
    calibration_labels = labels[splits["calibration"]]
    temperature = fit_temperature(calibration_logits, torch.tensor(calibration_labels))
    calibration_probabilities = torch.softmax(calibration_logits / temperature, dim=1).numpy()
    threshold, calibration_sweep, met = select_threshold(calibration_labels, calibration_probabilities,
                                                         **uncertainty_config)
    checkpoint.update(temperature=temperature, threshold=threshold,
                      dataset_config=asdict(data_config))
    torch.save(checkpoint, output / "model.pt")
    model = DecisionModel.load(output / "model.pt")
    test_ids = splits["test"]
    test_labels = labels[test_ids]
    test_logits = checkpoint_logits(checkpoint, observed[test_ids])
    raw_probabilities = torch.softmax(test_logits, dim=1).numpy()
    probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    before = probability_metrics(test_labels, raw_probabilities)
    after = probability_metrics(test_labels, probabilities)
    failures = high_confidence_errors(observed[test_ids], clean[test_ids], test_labels,
                                      probabilities, test_ids, threshold,
                                      evaluation_config["confidence_cutoff"])
    write_json(output / "high-confidence-errors.json", failures)
    ood = evaluate_ood(model, data_config.seed, evaluation_config["confidence_cutoff"])
    write_json(output / "ood.json", ood)
    results = {
        "samples": len(labels), "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "class_counts": {name: int((labels == i).sum()) for i, name in enumerate(ACTION_NAMES)},
        "architecture": [6, *training_config.hidden_sizes, 5],
        "parameter_count": sum(p.numel() for p in model.network.parameters()),
        "model_bytes": (output / "model.pt").stat().st_size,
        "best_epoch": checkpoint["best_epoch"], "epochs_run": len(history),
        "rule_baseline_noisy": baseline_metrics(observed[test_ids], test_labels),
        "rule_baseline_clean": baseline_metrics(clean[test_ids], test_labels),
        "model_classification": classification_metrics(test_labels, probabilities.argmax(axis=1)),
        "calibration": {"temperature": temperature, "before": before, "after": after,
                        "validation_before": probability_metrics(calibration_labels,
                                                                  torch.softmax(calibration_logits, dim=1).numpy()),
                        "validation_after": probability_metrics(calibration_labels, calibration_probabilities)},
        "uncertainty": {"threshold": threshold, "validation_target_met": met,
                        "validation_sweep": calibration_sweep,
                        "test": uncertainty_metrics(test_labels, probabilities, threshold)},
        "high_confidence_errors": {k: v for k, v in failures.items() if k != "cases"},
        "ood": {name: {k: v for k, v in row.items() if k != "cases"} for name, row in ood.items()},
        "latency": benchmark(model, [state_from_row(row).as_dict() for row in observed[test_ids[:100]]],
                             iterations),
    }
    write_json(output / "metrics.json", results)
    save_reliability_diagram(output / "reliability.png", before, after)
    print(f"Test accuracy: {results['model_classification']['accuracy']:.4f}; "
          f"threshold: {threshold:.2f}; artifacts: {output}", flush=True)
    return results
