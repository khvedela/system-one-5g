"""Preserve the MVP, evaluate support checks and broader cases, and gate readiness."""

from collections import Counter
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from .actions import ACTION_NAMES
from .data import DatasetConfig, generate_clean, add_noise, label_states, split_indices, state_from_row
from .decision import DecisionModel
from .evaluation import classification_metrics
from .experiment import benchmark, environment_metadata, evaluate_ood, write_json
from .policy import ground_truth
from .simulator import compare_controllers


def broader_states(seed: int, samples: int = 1000) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    states, kinds = [], []
    for i in range(samples):
        replicas = int(rng.integers(1, 11))
        cpu, memory = rng.uniform(0, 100, size=2)
        traffic, latency, error = rng.uniform(0, 600), rng.uniform(0, 1200), rng.uniform(0, .5)
        kind = "broad_combination"
        if i % 2:
            kind = "policy_boundary"
            field = i % 5
            if field == 0:
                cpu = rng.choice([25, 75, 85]) + rng.uniform(-.5, .5)
            elif field == 1:
                memory = rng.choice([40, 85]) + rng.uniform(-.5, .5)
            elif field == 2:
                traffic = rng.choice([100, 300]) + rng.uniform(-2, 2)
            elif field == 3:
                latency = rng.choice([100, 200]) + rng.uniform(-2, 2)
            else:
                error = rng.choice([.01, .1]) + rng.uniform(-.001, .001)
        states.append(state_from_row(np.array([cpu, memory, traffic * replicas, latency, error, replicas])).values())
        kinds.append(kind)
    return np.array(states), kinds


def outcomes(model: DecisionModel, states: np.ndarray, labels: np.ndarray) -> dict:
    decisions = [model.decide(state_from_row(row)) for row in states]
    predictions = np.array([ACTION_NAMES.index(d.action) for d in decisions])
    accepted = predictions != 4
    return {"classification": classification_metrics(labels, predictions),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float((predictions[accepted] == labels[accepted]).mean()) if accepted.any() else None,
            "reasons": dict(Counter(d.escalation_reason or "normal_action" for d in decisions))}


def assess_readiness(reference: dict, guarded: dict, criteria: dict) -> dict:
    checks = {}
    def check(name, measured, limit, minimum=False):
        passed = measured is not None and (measured >= limit if minimum else measured <= limit)
        checks[name] = {"measured": measured, "limit": limit, "passed": bool(passed)}
    check("accuracy", guarded["test"]["classification"]["accuracy"], criteria["accuracy_min"], True)
    check("macro_f1", guarded["test"]["classification"]["classification_report"]["macro avg"]["f1-score"],
          criteria["macro_f1_min"], True)
    check("ece", reference["calibration"]["after"]["ece"], criteria["ece_max"])
    check("accepted_accuracy", guarded["test"]["accepted_accuracy"], criteria["accepted_accuracy_min"], True)
    check("coverage", guarded["test"]["coverage"], criteria["coverage_min"], True)
    check("broader_accepted_accuracy", guarded["broader_distribution"]["accepted_accuracy"],
          criteria["broader_accepted_accuracy_min"], True)
    check("broader_coverage", guarded["broader_distribution"]["coverage"], criteria["broader_coverage_min"], True)
    check("high_confidence_errors", reference["high_confidence_errors"]["error_rate_among_confident"],
          criteria["high_confidence_error_rate_max"])
    check("latency_p95_ms", guarded["latency"]["p95_ms"], criteria["latency_p95_ms_max"])
    check("ood_escalation", min(row["escalation_rate"] for row in guarded["ood"].values()),
          criteria["ood_escalation_rate_min"], True)
    check("additional_guard_escalation", guarded["test"]["reasons"].get("outside_training_support", 0)
          / guarded["test_count"], criteria["new_in_distribution_guard_escalation_max"])
    scenarios = guarded["simulator"]["scenarios"].values()
    model_cost = sum(row["model"]["mean_cost"] for row in scenarios)
    rule_cost = sum(row["rule"]["mean_cost"] for row in scenarios)
    no_action_cost = sum(row["no_action"]["mean_cost"] for row in scenarios)
    check("simulator_cost_vs_rule", model_cost / rule_cost, criteria["simulator_cost_vs_rule_max"])
    check("simulator_cost_vs_no_action", model_cost / no_action_cost, criteria["simulator_cost_vs_no_action_max"])
    checks["real_labeled_telemetry"] = {"measured": False, "required": criteria["real_labeled_telemetry_required"],
                                         "passed": not criteria["real_labeled_telemetry_required"]}
    passed = all(row["passed"] for row in checks.values())
    return {"criteria_version": criteria["version"], "checks": checks, "passed": passed,
            "next_step": "shadow_evaluation_only" if passed else "improve_standalone_first",
            "automatic_network_control_allowed": False}


def run_robustness(reference_run: Path, output: Path, criteria: dict, seed: int = 42) -> dict:
    import json
    checkpoint = torch.load(reference_run / "model.pt", weights_only=True)
    config = DatasetConfig(**checkpoint["dataset_config"])
    clean, _ = generate_clean(config)
    observed = add_noise(clean, config)
    labels = label_states(clean)
    splits = split_indices(labels, config.seed)
    training = observed[splits["train"]]
    # Bounds use only the frozen reference training split, never probes or held-out data.
    checkpoint.update(support_min=training.min(axis=0).tolist(), support_max=training.max(axis=0).tolist(),
                      support_guard={"margin_std": .1})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "criteria.json", criteria)
    write_json(output / "environment.json", environment_metadata())
    torch.save(checkpoint, output / "model.pt")
    model = DecisionModel.load(output / "model.pt")
    ids = splits["test"]
    test = outcomes(model, observed[ids], labels[ids])
    ood = evaluate_ood(model, seed, .9)
    write_json(output / "ood.json", ood)
    broader, kinds = broader_states(seed)
    broader_labels = label_states(broader)
    broader_result = outcomes(model, broader, broader_labels)
    broad_cases = [{"kind": kind, "state": state_from_row(row).as_dict(),
                    "policy_action": ground_truth(state_from_row(row)).name,
                    **model.decide(state_from_row(row)).as_dict()}
                   for row, kind in zip(broader, kinds, strict=True)]
    write_json(output / "broader-cases.json", broad_cases)
    simulator = compare_controllers(model)
    start = time.perf_counter()
    latency = benchmark(model, [state_from_row(row).as_dict() for row in observed[ids[:100]]])
    # getrusage is process peak RSS, not model-only allocation.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"reference_run": str(reference_run), "seed": seed, "test_count": len(ids), "test": test,
              "broader_distribution": broader_result,
              "ood": {name: {k: v for k, v in row.items() if k != "cases"} for name, row in ood.items()},
              "latency": latency, "parameter_count": sum(p.numel() for p in model.network.parameters()),
              "model_bytes": (output / "model.pt").stat().st_size,
              "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
              "rss_scope": "whole experiment process peak including libraries/data; not model-only RAM",
              "benchmark_wall_seconds": time.perf_counter() - start, "simulator": simulator}
    reference = json.loads((reference_run / "metrics.json").read_text())
    result["readiness"] = assess_readiness(reference, result, criteria)
    write_json(output / "metrics.json", result)
    return result
