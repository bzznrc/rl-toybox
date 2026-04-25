"""Actor-critic networks for PPO and MAPPO-style CTDE."""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn as nn


RecurrentState: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _init_linear(layer: nn.Linear, *, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=float(gain))
    nn.init.constant_(layer.bias, 0.0)


def _init_mlp(module: nn.Module, *, gain: float = 2**0.5) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            _init_linear(child, gain=float(gain))


def _init_recurrent(module: nn.GRU | nn.LSTM) -> None:
    for name, param in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


def build_mlp(input_dim: int, hidden_sizes: list[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    in_features = int(input_dim)
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(in_features, int(hidden)), nn.Tanh()])
        in_features = int(hidden)
    return nn.Sequential(*layers), in_features


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        *,
        critic_obs_dim: int | None = None,
        critic_hidden_sizes: list[int] | None = None,
        share_backbone: bool = False,
        action_type: str = "discrete",
        init_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        policy_head_feature_groups: list[list[int]] | None = None,
    ):
        super().__init__()
        self.action_type = str(action_type).strip().lower()
        if self.action_type not in {"discrete", "continuous"}:
            raise ValueError("PPO ActorCritic action_type must be 'discrete' or 'continuous'.")

        critic_input_dim = int(obs_dim) if critic_obs_dim is None else int(critic_obs_dim)
        self.share_backbone = bool(share_backbone)
        self.policy_head_feature_groups = [
            [int(index) for index in group]
            for group in ([] if policy_head_feature_groups is None else policy_head_feature_groups)
            if group
        ]
        if self.share_backbone and int(critic_input_dim) != int(obs_dim):
            raise ValueError("Shared-backbone PPO requires critic_obs_dim to match obs_dim.")

        self.shared_backbone: nn.Sequential | None
        if self.share_backbone:
            shared_backbone, shared_out_dim = build_mlp(int(obs_dim), list(hidden_sizes))
            self.shared_backbone = shared_backbone
            self.actor_backbone = nn.Identity()
            self.critic_backbone = nn.Identity()
            self.policy_head = nn.Linear(shared_out_dim, int(action_dim))
            self.policy_heads = nn.ModuleList(
                [nn.Linear(shared_out_dim, int(action_dim)) for _ in self.policy_head_feature_groups]
            )
            self.value_head = nn.Linear(shared_out_dim, 1)
        else:
            actor_backbone, actor_out_dim = build_mlp(int(obs_dim), list(hidden_sizes))
            self.shared_backbone = None
            self.actor_backbone = actor_backbone
            self.policy_head = nn.Linear(actor_out_dim, int(action_dim))
            self.policy_heads = nn.ModuleList(
                [nn.Linear(actor_out_dim, int(action_dim)) for _ in self.policy_head_feature_groups]
            )
            critic_sizes = list(hidden_sizes) if critic_hidden_sizes is None else list(critic_hidden_sizes)
            critic_backbone, critic_out_dim = build_mlp(critic_input_dim, critic_sizes)
            self.critic_backbone = critic_backbone
            self.value_head = nn.Linear(critic_out_dim, 1)
        self.log_std: nn.Parameter | None
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.full((int(action_dim),), float(init_log_std), dtype=torch.float32))
        else:
            self.log_std = None
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.shared_backbone is not None:
            _init_mlp(self.shared_backbone)
        else:
            _init_mlp(self.actor_backbone)
            _init_mlp(self.critic_backbone)

        _init_linear(self.policy_head, gain=0.01)
        for policy_head in self.policy_heads:
            _init_linear(policy_head, gain=0.01)
        _init_linear(self.value_head, gain=1.0)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if self.shared_backbone is not None:
            features = self.shared_backbone(obs)
        else:
            features = self.actor_backbone(obs)
        if self.policy_heads:
            logits = torch.stack([head(features) for head in self.policy_heads], dim=1)
            group_scores: list[torch.Tensor] = []
            for group in self.policy_head_feature_groups:
                valid_indices = [index for index in group if 0 <= int(index) < int(obs.shape[1])]
                if valid_indices:
                    group_scores.append(obs[:, valid_indices].amax(dim=1))
                else:
                    group_scores.append(torch.zeros((int(obs.shape[0]),), dtype=obs.dtype, device=obs.device))
            head_indices = torch.stack(group_scores, dim=1).argmax(dim=1)
            return logits[torch.arange(int(obs.shape[0]), device=obs.device), head_indices]
        return self.policy_head(features)

    def policy_log_std(self) -> torch.Tensor:
        if self.log_std is None:
            raise RuntimeError("PPO policy_log_std is only available for continuous action policies.")
        return torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        if self.shared_backbone is not None:
            features = self.shared_backbone(critic_obs)
        else:
            features = self.critic_backbone(critic_obs)
        return self.value_head(features).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_backbone is not None:
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            shared_features = self.shared_backbone(obs)
            if self.policy_heads:
                logits = self.policy(obs)
            else:
                logits = self.policy_head(shared_features)
            value = self.value_head(shared_features).squeeze(-1)
            return logits, value
        logits = self.policy(obs)
        value_input = obs if critic_obs is None else critic_obs
        value = self.value(value_input)
        return logits, value


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_hidden_sizes: list[int],
        *,
        recurrent_hidden_size: int,
        recurrent_type: str = "lstm",
        actor_head_hidden_sizes: list[int] | None = None,
        critic_head_hidden_sizes: list[int] | None = None,
        action_type: str = "discrete",
        init_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.action_type = str(action_type).strip().lower()
        if self.action_type not in {"discrete", "continuous"}:
            raise ValueError("PPO RecurrentActorCritic action_type must be 'discrete' or 'continuous'.")

        self.recurrent_type = str(recurrent_type).strip().lower()
        if self.recurrent_type not in {"lstm", "gru"}:
            raise ValueError("PPO RecurrentActorCritic recurrent_type must be 'lstm' or 'gru'.")

        self.recurrent_hidden_size = max(1, int(recurrent_hidden_size))
        self.recurrent_num_layers = 1

        self.encoder, encoder_out_dim = build_mlp(int(obs_dim), list(encoder_hidden_sizes))
        if self.recurrent_type == "gru":
            self.recurrent: nn.GRU | nn.LSTM = nn.GRU(
                input_size=int(encoder_out_dim),
                hidden_size=int(self.recurrent_hidden_size),
                num_layers=int(self.recurrent_num_layers),
                batch_first=True,
            )
        else:
            self.recurrent = nn.LSTM(
                input_size=int(encoder_out_dim),
                hidden_size=int(self.recurrent_hidden_size),
                num_layers=int(self.recurrent_num_layers),
                batch_first=True,
            )

        actor_head_sizes = [] if actor_head_hidden_sizes is None else list(actor_head_hidden_sizes)
        critic_head_sizes = [] if critic_head_hidden_sizes is None else list(critic_head_hidden_sizes)
        self.actor_backbone, actor_out_dim = build_mlp(int(self.recurrent_hidden_size), actor_head_sizes)
        self.critic_backbone, critic_out_dim = build_mlp(int(self.recurrent_hidden_size), critic_head_sizes)
        self.policy_head = nn.Linear(actor_out_dim, int(action_dim))
        self.value_head = nn.Linear(critic_out_dim, 1)

        self.log_std: nn.Parameter | None
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.full((int(action_dim),), float(init_log_std), dtype=torch.float32))
        else:
            self.log_std = None
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _init_mlp(self.encoder)
        _init_recurrent(self.recurrent)
        _init_mlp(self.actor_backbone)
        _init_mlp(self.critic_backbone)
        _init_linear(self.policy_head, gain=0.01)
        _init_linear(self.value_head, gain=1.0)

    def zero_state(self, batch_size: int, device: torch.device) -> RecurrentState:
        base = torch.zeros(
            int(self.recurrent_num_layers),
            int(batch_size),
            int(self.recurrent_hidden_size),
            dtype=torch.float32,
            device=device,
        )
        if self.recurrent_type == "gru":
            return base
        return base.clone(), base

    @staticmethod
    def detach_state(state: RecurrentState) -> RecurrentState:
        if isinstance(state, tuple):
            return tuple(part.detach() for part in state)
        return state.detach()

    def policy_log_std(self) -> torch.Tensor:
        if self.log_std is None:
            raise RuntimeError("PPO policy_log_std is only available for continuous action policies.")
        return torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)

    def forward_sequence(
        self,
        obs: torch.Tensor,
        *,
        state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if obs.dim() != 3:
            raise ValueError(f"Recurrent PPO expected obs ndim 2 or 3, got {obs.dim()}.")

        batch_size, seq_len, obs_dim = obs.shape
        encoded = self.encoder(obs.reshape(-1, int(obs_dim))).reshape(batch_size, seq_len, -1)
        initial_state = self.zero_state(int(batch_size), obs.device) if state is None else state
        recurrent_out, next_state = self.recurrent(encoded, initial_state)
        actor_features = self.actor_backbone(recurrent_out.reshape(-1, int(self.recurrent_hidden_size))).reshape(
            batch_size,
            seq_len,
            -1,
        )
        critic_features = self.critic_backbone(recurrent_out.reshape(-1, int(self.recurrent_hidden_size))).reshape(
            batch_size,
            seq_len,
            -1,
        )
        policy_output = self.policy_head(actor_features)
        value_output = self.value_head(critic_features).squeeze(-1)
        return policy_output, value_output, next_state

    def forward_step(
        self,
        obs: torch.Tensor,
        *,
        state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        policy_output, value_output, next_state = self.forward_sequence(obs, state=state)
        return policy_output[:, 0, :], value_output[:, 0], next_state
