"""Network modules for soft actor-critic."""

from __future__ import annotations

import torch
import torch.nn as nn


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *[int(size) for size in hidden_sizes], int(output_dim)]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(int(in_dim), int(out_dim)))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(int(dims[-2]), int(dims[-1])))
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """Gaussian actor with tanh squashing handled in the algorithm layer."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        hidden = [int(size) for size in hidden_sizes]
        if hidden:
            layers: list[nn.Module] = []
            input_dim = int(obs_dim)
            for hidden_dim in hidden:
                layers.append(nn.Linear(int(input_dim), int(hidden_dim)))
                layers.append(nn.ReLU())
                input_dim = int(hidden_dim)
            self.backbone = nn.Sequential(*layers)
            feature_dim = int(input_dim)
        else:
            self.backbone = nn.Identity()
            feature_dim = int(obs_dim)
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
        self.backbone = _build_mlp(int(obs_dim + action_dim), [int(size) for size in hidden_sizes], 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat((obs, action), dim=-1)
        return self.backbone(obs_action).squeeze(-1)
