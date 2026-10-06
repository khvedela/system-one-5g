"""Seeded synthetic states with explicit scenario and latent-state metadata."""

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .actions import ACTION_NAMES
from .policy import POLICY_VERSION, POLICY_THRESHOLDS, ground_truth
from .state import FEATURE_NAMES, RANGES, State


DATASET_VERSION = "server-data-v1"
SCENARIOS = ("normal", "low_load", "overload", "high_errors", "mixed")


@dataclass(frozen=True)
class DatasetConfig:
    samples: int = 20_000
    seed: int = 42
    cpu_noise_std: float = 3.0
    latency_noise_std: float = 12.0
    request_rate_noise_std: float = 30.0
    error_rate_noise_std: float = 0.002
    broad_fraction: float = 0.0
    boundary_fraction: float = 0.0

    def __post_init__(self):
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 100:
            raise ValueError("samples must be an integer >= 100")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name, value in asdict(self).items():
            if name.endswith("_std") and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (not 0 <= self.broad_fraction <= 1 or not 0 <= self.boundary_fraction <= 1
                or self.broad_fraction + self.boundary_fraction > 1):
            raise ValueError("broad and boundary fractions must be nonnegative and sum to <= 1")


def generate_clean(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    scenario_ids = rng.permutation(np.arange(config.samples) % len(SCENARIOS))
    states = np.empty((config.samples, len(FEATURE_NAMES)), dtype=np.float64)
    # cpu, memory, traffic per replica, latency, error rate bounds, per scenario.
    bounds = (
        ((20, 70), (20, 80), (100, 280), (20, 180), (0, .04)),
        ((5, 24), (10, 39), (10, 95), (5, 95), (0, .009)),
        ((78, 99), (45, 95), (320, 600), (220, 1000), (0, .06)),
        ((10, 90), (20, 90), (80, 450), (100, 800), (.11, .35)),
        ((72, 95), (30, 95), (20, 200), (20, 180), (0, .12)),
    )
    for i, scenario in enumerate(scenario_ids):
        cpu, memory, traffic, latency, error = [rng.uniform(*b) for b in bounds[scenario]]
        replicas = int(rng.integers(1, 11))
        states[i] = cpu, memory, traffic * replicas, latency, error, replicas
    if config.broad_fraction or config.boundary_fraction:
        extra_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 4]))
        indices = extra_rng.permutation(config.samples)
        broad_count = int(config.samples * config.broad_fraction)
        boundary_count = int(config.samples * config.boundary_fraction)
        for j, i in enumerate(indices[:broad_count + boundary_count]):
            replicas = int(extra_rng.integers(1, 11))
            cpu, memory = extra_rng.uniform(0, 100, 2)
            traffic, latency = extra_rng.uniform(0, 600), extra_rng.uniform(0, 1200)
            error = extra_rng.uniform(0, .03 if extra_rng.random() < .5 else .5)
            scenario_ids[i] = 5 if j < broad_count else 6
            if j >= broad_count:
                field = j % 5
                if field == 0:
                    cpu = extra_rng.choice([25, 75, 85]) + extra_rng.uniform(-.5, .5)
                elif field == 1:
                    memory = extra_rng.choice([40, 85]) + extra_rng.uniform(-.5, .5)
                elif field == 2:
                    traffic = extra_rng.choice([100, 300]) + extra_rng.uniform(-2, 2)
                elif field == 3:
                    latency = extra_rng.choice([100, 200]) + extra_rng.uniform(-2, 2)
                else:
                    error = extra_rng.choice([.01, .1]) + extra_rng.uniform(-.001, .001)
            states[i] = cpu, memory, traffic * replicas, latency, error, replicas
    return states, scenario_ids


def label_states(states: np.ndarray) -> np.ndarray:
    return np.array([int(ground_truth(state_from_row(row))) for row in states], dtype=np.int64)


def add_noise(clean: np.ndarray, config: DatasetConfig) -> np.ndarray:
    # Independent stream prevents changing clean states when noise settings change.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    observed = clean.copy()
    scales = [config.cpu_noise_std, 0, config.request_rate_noise_std,
              config.latency_noise_std, config.error_rate_noise_std, 0]
    observed += rng.normal(size=observed.shape) * np.array(scales)
    for j, name in enumerate(FEATURE_NAMES):
        observed[:, j] = np.clip(observed[:, j], *RANGES[name])
    return observed


def split_indices(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels))
    train, remainder = train_test_split(indices, test_size=.30, random_state=seed, stratify=labels)
    validation, test = train_test_split(remainder, test_size=.50, random_state=seed + 1,
                                      stratify=labels[remainder])
    selection, calibration = train_test_split(validation, test_size=.50, random_state=seed + 2,
                                             stratify=labels[validation])
    return {"train": train, "validation": validation, "test": test,
            "selection": selection, "calibration": calibration}


def save_splits(path: Path, splits: dict[str, np.ndarray]) -> None:
    (path / "splits.json").write_text(json.dumps({k: v.tolist() for k, v in splits.items()}, indent=2) + "\n")


def state_from_row(row: np.ndarray) -> State:
    values = dict(zip(FEATURE_NAMES, row.tolist(), strict=True))
    if not float(values["replicas"]).is_integer():
        raise ValueError("replicas must be an integer")
    values["replicas"] = int(values["replicas"])
    return State(**values)


def save_dataset(path: Path, observed: np.ndarray, clean: np.ndarray,
                 labels: np.ndarray, scenarios: np.ndarray, config: DatasetConfig) -> None:
    path.mkdir(parents=True, exist_ok=False)
    with (path / "states.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "scenario", *FEATURE_NAMES,
                         *(f"clean_{name}" for name in FEATURE_NAMES), "action"])
        for i, (x, latent, y, scenario) in enumerate(zip(observed, clean, labels, scenarios, strict=True)):
            names = (*SCENARIOS, "broad_combination", "policy_boundary")
            writer.writerow([i, names[scenario], *x, *latent, ACTION_NAMES[y]])
    metadata = {
        "dataset_version": "server-data-v2" if config.broad_fraction or config.boundary_fraction else DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "policy_thresholds": POLICY_THRESHOLDS, "config": asdict(config),
        "feature_names": FEATURE_NAMES, "action_names": ACTION_NAMES,
        "numpy_version": np.__version__,
        "label_source": "ground_truth(clean_state); observations include measurement noise",
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
