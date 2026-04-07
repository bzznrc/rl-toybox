"""Network modules for DQN agents."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class MLPQNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(input_size)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden)))
            layers.append(nn.ReLU())
            in_features = int(hidden)
        layers.append(nn.Linear(in_features, int(output_size)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

    def copy(self) -> "MLPQNetwork":
        return copy.deepcopy(self)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(input_size)
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, int(hidden)), nn.GELU()])
            in_features = int(hidden)

        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_features, 1)
        self.advantage_head = nn.Linear(in_features, int(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_extractor(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def copy(self) -> "DuelingQNetwork":
        return copy.deepcopy(self)


def build_q_network(
    *,
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    dueling: bool,
) -> nn.Module:
    if dueling:
        return DuelingQNetwork(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    return MLPQNetwork(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
