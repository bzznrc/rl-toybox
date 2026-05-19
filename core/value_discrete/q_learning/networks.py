"""Simple LinearQNet used by Snake q-learning baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.algorithms.base import build_mlp


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
        super().__init__()
        self.network, _ = build_mlp(
            int(input_size),
            [int(size) for size in hidden_layers],
            activation=nn.ReLU,
            output_dim=int(output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)
