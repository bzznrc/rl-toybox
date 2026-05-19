"""Network modules for soft actor-critic."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.algorithms.base import build_mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ActorNetwork(nn.Module):
    """Gaussian actor with tanh squashing handled in the algorithm layer."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.backbone, feature_dim = build_mlp(
            int(obs_dim),
            [int(size) for size in hidden_sizes],
            activation=nn.ReLU,
        )
        self.mean_head = nn.Linear(int(feature_dim), int(action_dim))
        self.log_std_head = nn.Linear(int(feature_dim), int(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class CriticNetwork(nn.Module):
    """Q-network over observation-action pairs."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.backbone, _ = build_mlp(
            int(obs_dim + action_dim),
            [int(size) for size in hidden_sizes],
            activation=nn.ReLU,
            output_dim=1,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat((obs, action), dim=-1)
        return self.backbone(obs_action).squeeze(-1)
