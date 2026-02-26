"""Actor-critic networks for PPO."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(obs_dim)
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, int(hidden)), nn.Tanh()])
            in_features = int(hidden)
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_features, int(action_dim))
        self.value_head = nn.Linear(in_features, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
