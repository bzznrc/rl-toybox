"""Policy/value MLPs for compact search-play games."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.algorithms.base import build_mlp


class PolicyValueMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], policy_size: int) -> None:
        super().__init__()
        self.trunk, in_features = build_mlp(
            int(input_size),
            [int(size) for size in hidden_sizes],
            activation=nn.ReLU,
        )
        self.policy_head = nn.Linear(in_features, int(policy_size))
        self.value_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        values = torch.tanh(self.value_head(features)).squeeze(-1)
        return policy_logits, values


def build_policy_value_network(*, input_size: int, hidden_sizes: list[int], policy_size: int) -> nn.Module:
    return PolicyValueMLP(input_size=int(input_size), hidden_sizes=list(hidden_sizes), policy_size=int(policy_size))
