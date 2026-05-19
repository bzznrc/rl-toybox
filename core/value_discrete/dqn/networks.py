"""Network modules for DQN agents."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from core.algorithms.base import build_mlp


class MLPQNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.network, _ = build_mlp(
            int(input_size),
            [int(size) for size in hidden_sizes],
            activation=nn.ReLU,
            output_dim=int(output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

    def copy(self) -> "MLPQNetwork":
        return copy.deepcopy(self)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.feature_extractor, in_features = build_mlp(
            int(input_size),
            [int(size) for size in hidden_sizes],
            activation=nn.GELU,
        )
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
