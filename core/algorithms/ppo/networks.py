"""Actor-critic networks for PPO and MAPPO-style CTDE."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        *,
        critic_obs_dim: int | None = None,
        critic_hidden_sizes: list[int] | None = None,
        action_type: str = "discrete",
        init_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.action_type = str(action_type).strip().lower()
        if self.action_type not in {"discrete", "continuous"}:
            raise ValueError("PPO ActorCritic action_type must be 'discrete' or 'continuous'.")

        actor_backbone, actor_out_dim = self._build_mlp(int(obs_dim), list(hidden_sizes))
        self.actor_backbone = actor_backbone
        self.policy_head = nn.Linear(actor_out_dim, int(action_dim))
        self.log_std: nn.Parameter | None
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.full((int(action_dim),), float(init_log_std), dtype=torch.float32))
        else:
            self.log_std = None
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        critic_input_dim = int(obs_dim) if critic_obs_dim is None else int(critic_obs_dim)
        critic_sizes = list(hidden_sizes) if critic_hidden_sizes is None else list(critic_hidden_sizes)
        critic_backbone, critic_out_dim = self._build_mlp(critic_input_dim, critic_sizes)
        self.critic_backbone = critic_backbone
        self.value_head = nn.Linear(critic_out_dim, 1)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_sizes: list[int]) -> tuple[nn.Sequential, int]:
        layers: list[nn.Module] = []
        in_features = int(input_dim)
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, int(hidden)), nn.Tanh()])
            in_features = int(hidden)
        return nn.Sequential(*layers), in_features

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.actor_backbone(obs)
        return self.policy_head(features)

    def policy_log_std(self) -> torch.Tensor:
        if self.log_std is None:
            raise RuntimeError("PPO policy_log_std is only available for continuous action policies.")
        return torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        features = self.critic_backbone(critic_obs)
        return self.value_head(features).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        value_input = obs if critic_obs is None else critic_obs
        value = self.value(value_input)
        return logits, value
