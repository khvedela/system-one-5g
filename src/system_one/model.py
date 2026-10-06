"""A small MLP; callers will use the decision wrapper rather than logits."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, int] = (256, 128), input_size: int = 6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], 5),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)
