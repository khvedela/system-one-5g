"""Train on a frozen, arbitrary-width metric table with grouped holdouts."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .actions import ACTION_NAMES
from .calibration import fit_temperature
from .decision import DecisionModel
from .evaluation import classification_metrics, probability_metrics, save_reliability_diagram
from .experiment import environment_metadata, write_json
from .features import schema_hash
from .training import TrainingConfig, train, checkpoint_logits
from .uncertainty import select_threshold, uncertainty_metrics


def grouped_splits(groups: list[str], seed: int) -> dict[str, np.ndarray]:
    unique = sorted(set(groups))
    if len(unique) < 20:
        raise ValueError("at least 20 independent groups/episodes are required")
    ordered = np.random.default_rng(seed).permutation(unique)
    a, b = int(.7 * len(ordered)), int(.85 * len(ordered))
    validation = ordered[a:b]
    buckets = {"train": ordered[:a], "selection": validation[:max(1, len(validation) // 2)],
               "calibration": validation[max(1, len(validation) // 2):], "test": ordered[b:]}
    return {name: np.array([i for i, group in enumerate(groups) if group in set(bucket)], dtype=np.int64)
            for name, bucket in buckets.items()}


def train_table(data: Path, output: Path, config: TrainingConfig = TrainingConfig()) -> dict:
    with data.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = tuple(sorted(set(reader.fieldnames or []) - {"action", "group_id"}))
        if not names or not {"action", "group_id"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires action, group_id and numeric metric columns")
        rows = list(reader)
    groups = [row["group_id"] for row in rows]
    if any(not group for group in groups):
        raise ValueError("group_id must identify independent collection runs or episodes")
    labels = np.array([ACTION_NAMES.index(row["action"]) for row in rows], dtype=np.int64)
    x = np.array([[float(row[name]) if row[name] else np.nan for name in names] for row in rows])
    if np.isinf(x).any():
        raise ValueError("infinite observations must be marked missing")
    splits = grouped_splits(groups, config.seed)
    imputation = []
    for j, name in enumerate(names):
        available = x[splits["train"], j]
        available = available[np.isfinite(available)]
        if not len(available):
            raise ValueError(f"no training observation for metric {name}")
        imputation.append(float(np.median(available)))
    missing = ~np.isfinite(x)
    encoded = np.concatenate([np.where(missing, imputation, x), missing.astype(float)], axis=1)
    model_names = (*names, *(name + "/missing" for name in names))
    schema = {"version": 1, "feature_names": list(names), "sha256": schema_hash(names)}
    output.mkdir(parents=True, exist_ok=False)
    checkpoint, history = train(encoded[splits["train"]], labels[splits["train"]],
                                encoded[splits["selection"]], labels[splits["selection"]], config, model_names)
    logits = checkpoint_logits(checkpoint, encoded[splits["calibration"]])
    temperature = fit_temperature(logits, torch.tensor(labels[splits["calibration"]]))
    probability = torch.softmax(logits / temperature, dim=1).numpy()
    threshold, sweep, target_met = select_threshold(labels[splits["calibration"]], probability)
    checkpoint.update(temperature=temperature, threshold=threshold, telemetry_schema=schema,
                      imputation=imputation, support_guard={"margin_std": .1})
    torch.save(checkpoint, output / "model.pt")
    model = DecisionModel.load(output / "model.pt")
    ids = splits["test"]
    logits = checkpoint_logits(checkpoint, encoded[ids])
    before = probability_metrics(labels[ids], torch.softmax(logits, dim=1).numpy())
    p = torch.softmax(logits / temperature, dim=1).numpy()
    after = probability_metrics(labels[ids], p)
    decisions = [model.decide({name: (float(x[i, j]) if np.isfinite(x[i, j]) else None)
                              for j, name in enumerate(names)}) for i in ids]
    result = {"metric_count": len(names), "input_width_including_missing_indicators": len(model_names),
              "schema": schema, "split_sizes": {k: len(v) for k, v in splits.items()},
              "raw_classification": classification_metrics(labels[ids], p.argmax(axis=1)),
              "final_classification": classification_metrics(labels[ids], np.array([ACTION_NAMES.index(d.action) for d in decisions])),
              "calibration_before": before, "calibration_after": after,
              "threshold": threshold, "validation_target_met": target_met,
              "validation_threshold_sweep": sweep,
              "uncertainty_without_support_guard": uncertainty_metrics(labels[ids], p, threshold),
              "final_coverage": sum(d.action != "ESCALATE" for d in decisions) / len(decisions)}
    write_json(output / "metrics.json", result)
    write_json(output / "training.json", history)
    write_json(output / "environment.json", environment_metadata())
    write_json(output / "schema.json", schema)
    write_json(output / "config.json", asdict(config))
    write_json(output / "splits.json", {k: v.tolist() for k, v in splits.items()})
    write_json(output / "group_splits.json", {k: sorted({groups[i] for i in v}) for k, v in splits.items()})
    (output / "data.csv").write_bytes(data.read_bytes())
    save_reliability_diagram(output / "reliability.png", before, after)
    return result
