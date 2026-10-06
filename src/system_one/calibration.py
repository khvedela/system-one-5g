"""Post-hoc temperature scaling fitted exclusively on calibration data."""

import math

import torch
from torch import nn


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    values = logits.detach().to(torch.float64)
    labels = labels.detach().long()
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=100,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(.1), math.log(10)).exp()
        loss = nn.functional.cross_entropy(values / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(math.log(.1), math.log(10)).exp())
    original = float(nn.functional.cross_entropy(values, labels))
    calibrated = float(nn.functional.cross_entropy(values / temperature, labels))
    return temperature if calibrated <= original else 1.0
