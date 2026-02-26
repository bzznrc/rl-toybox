"""SAC network stubs."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Stub actor for future SAC implementation."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Linear(int(obs_dim), int(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)


class CriticNetwork(nn.Module):
    """Stub critic for future SAC implementation."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Linear(int(obs_dim + action_dim), 1)

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs_action)
