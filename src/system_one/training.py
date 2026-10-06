"""CPU training with train-only normalization and validation early stopping."""

from dataclasses import dataclass, asdict
import math
import random

import numpy as np
import torch
from torch import nn

from .actions import ACTION_NAMES
from .model import MLP
from .state import FEATURE_NAMES


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 12
    min_delta: float = 0.0001
    threads: int = 1
    hidden_sizes: tuple[int, int] = (256, 128)

    def __post_init__(self):
        for name in ("epochs", "batch_size", "patience", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("learning_rate", "weight_decay", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"invalid {name}")
        if len(self.hidden_sizes) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                                           for v in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive integers")


def train(x_train: np.ndarray, y_train: np.ndarray, x_selection: np.ndarray,
          y_selection: np.ndarray, config: TrainingConfig,
          feature_names: tuple[str, ...] = FEATURE_NAMES) -> tuple[dict, list[dict]]:
    if (not feature_names or len(set(feature_names)) != len(feature_names)
            or x_train.ndim != 2 or x_selection.ndim != 2
            or x_train.shape[1] != len(feature_names) or x_selection.shape[1] != len(feature_names)
            or len(x_train) != len(y_train) or len(x_selection) != len(y_selection)
            or not len(y_train) or not len(y_selection)
            or not np.isfinite(x_train).all() or not np.isfinite(x_selection).all()):
        raise ValueError("training arrays must be finite and match the feature contract")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    features = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)
    selection = torch.tensor((x_selection - mean) / std, dtype=torch.float32)
    selection_labels = torch.tensor(y_selection, dtype=torch.long)
    network = MLP(config.hidden_sizes, len(feature_names))
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_weights = None
    history = []
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(1, config.epochs + 1):
        network.train()
        indices = torch.randperm(len(labels), generator=generator)
        total_loss = 0.0
        for batch in indices.split(config.batch_size):
            optimizer.zero_grad()
            loss = loss_fn(network(features[batch]), labels[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        network.eval()
        with torch.inference_mode():
            logits = network(selection)
            validation_loss = float(loss_fn(logits, selection_labels))
            validation_accuracy = float((logits.argmax(dim=1) == selection_labels).float().mean())
        history.append({"epoch": epoch, "train_loss": total_loss / len(labels),
                        "validation_loss": validation_loss, "validation_accuracy": validation_accuracy})
        if validation_loss < best_loss - config.min_delta:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_weights = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    checkpoint = {
        "format_version": 1 if feature_names == FEATURE_NAMES else 2, "state_dict": best_weights,
        "mean": mean.tolist(), "std": std.tolist(), "feature_names": list(feature_names),
        "action_names": list(ACTION_NAMES), "training_config": asdict(config),
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "temperature": 1.0, "threshold": 0.0,
        "support_min": x_train.min(axis=0).tolist(), "support_max": x_train.max(axis=0).tolist(),
    }
    return checkpoint, history


def network_from_checkpoint(checkpoint: dict) -> MLP:
    network = MLP(tuple(checkpoint["training_config"]["hidden_sizes"]), len(checkpoint["feature_names"]))
    network.load_state_dict(checkpoint["state_dict"])
    return network.eval()


def checkpoint_logits(checkpoint: dict, features: np.ndarray) -> torch.Tensor:
    values = torch.tensor((features - np.array(checkpoint["mean"])) / np.array(checkpoint["std"]),
                          dtype=torch.float32)
    with torch.inference_mode():
        return network_from_checkpoint(checkpoint)(values)
