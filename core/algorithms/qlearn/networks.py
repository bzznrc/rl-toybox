"""Simple LinearQNet used by Snake q-learning baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = int(input_size)

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, int(hidden_size)))
            layers.append(nn.ReLU())
            in_size = int(hidden_size)

        layers.append(nn.Linear(in_size, int(output_size)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)
