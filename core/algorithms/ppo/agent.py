"""Minimal PPO implementation with optional MAPPO-style centralized critic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from core.algorithms.base import Algorithm
from core.algorithms.ppo.networks import ActorCritic
from core.algorithms.ppo.rollout import RolloutBuffer
from core.io.checkpoint import load_torch_checkpoint, save_torch_checkpoint


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: list[int]
    critic_hidden_sizes: list[int] | None = None
    critic_obs_dim: int | None = None
    centralized_critic: bool = False
    critic_condition_on_agent_obs: bool = True
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_gpu: bool = False
    action_type: str = "discrete"
    action_low: float = -1.0
    action_high: float = 1.0
    init_log_std: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 2.0


class PPOAlgorithm(Algorithm):
    algo_id = "ppo"

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

        self._action_type = str(config.action_type).strip().lower()
        if self._action_type not in {"discrete", "continuous"}:
            raise ValueError("PPO action_type must be 'discrete' or 'continuous'.")
        self._is_discrete = bool(self._action_type == "discrete")
        self._action_low = float(config.action_low)
        self._action_high = float(config.action_high)
        if (not self._is_discrete) and self._action_low >= self._action_high:
            raise ValueError("PPO continuous action bounds require action_low < action_high.")

        self._use_centralized_critic = bool(config.centralized_critic)
        if self._use_centralized_critic:
            if config.critic_obs_dim is None:
                raise ValueError("PPO centralized critic requires critic_obs_dim.")
            self._central_obs_dim = int(config.critic_obs_dim)
        else:
            self._central_obs_dim = int(config.obs_dim)
        self._critic_condition_on_agent_obs = bool(
            self._use_centralized_critic and bool(config.critic_condition_on_agent_obs)
        )
        self._critic_obs_dim = int(self._central_obs_dim) + (
            int(config.obs_dim) if self._critic_condition_on_agent_obs else 0
        )

        critic_hidden_sizes = (
            list(config.hidden_sizes)
            if config.critic_hidden_sizes is None
            else list(config.critic_hidden_sizes)
        )

        self.model = ActorCritic(
            config.obs_dim,
            config.action_dim,
            config.hidden_sizes,
            critic_obs_dim=int(self._critic_obs_dim),
            critic_hidden_sizes=critic_hidden_sizes,
            action_type=str(self._action_type),
            init_log_std=float(config.init_log_std),
            min_log_std=float(config.min_log_std),
            max_log_std=float(config.max_log_std),
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config.learning_rate))
        self.rollout = RolloutBuffer()

        self._last_log_prob: float | np.ndarray = 0.0
        self._last_value: float | np.ndarray = 0.0
        self._last_action_for_storage: int | np.ndarray | None = None

    def _normalize_action_mask(
        self,
        action_mask: object,
        *,
        batch_size: int,
    ) -> np.ndarray:
        if not self._is_discrete:
            raise ValueError("PPO action masks are only supported for discrete action policies.")

        mask_array = np.asarray(action_mask, dtype=np.bool_)
        action_dim = int(self.config.action_dim)

        if mask_array.ndim == 1:
            if int(mask_array.size) == 1 and int(batch_size) > 1:
                mask_array = np.full((int(batch_size), action_dim), bool(mask_array.item()), dtype=np.bool_)
            else:
                mask_array = mask_array.reshape(1, -1)
        if mask_array.ndim != 2:
            raise ValueError(f"PPO action mask expected ndim 1 or 2, got {mask_array.ndim}.")

        if int(mask_array.shape[0]) == 1 and int(batch_size) > 1:
            mask_array = np.repeat(mask_array, int(batch_size), axis=0)
        if int(mask_array.shape[0]) != int(batch_size):
            raise ValueError(
                f"PPO action mask expected batch size {int(batch_size)}, got {int(mask_array.shape[0])}."
            )
        if int(mask_array.shape[1]) != int(action_dim):
            raise ValueError(
                f"PPO action mask expected action dim {int(action_dim)}, got {int(mask_array.shape[1])}."
            )

        # Guarantee a valid categorical distribution for every row.
        valid_counts = mask_array.sum(axis=1)
        if np.any(valid_counts <= 0):
            mask_array = mask_array.copy()
            mask_array[valid_counts <= 0, :] = True
        return mask_array.astype(np.bool_, copy=False)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, action_mask: np.ndarray | None) -> torch.Tensor:
        if action_mask is None:
            return logits
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        return logits.masked_fill(~mask_tensor, float(-1e9))

    @staticmethod
    def _as_batch_obs(obs: object) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        if obs_array.ndim == 1:
            return obs_array.reshape(1, -1)
        if obs_array.ndim == 2:
            return obs_array
        raise ValueError(f"PPO expected obs ndim 1 or 2, got {obs_array.ndim}.")

    def _fallback_central_obs(self, obs_batch: np.ndarray) -> np.ndarray:
        if not self._use_centralized_critic:
            return np.asarray(obs_batch, dtype=np.float32).reshape(-1, int(self._central_obs_dim))

        batch_size = int(obs_batch.shape[0])
        state = np.zeros((int(self._central_obs_dim),), dtype=np.float32)
        flat_obs = np.asarray(obs_batch, dtype=np.float32).reshape(-1)
        copy_count = min(int(state.size), int(flat_obs.size))
        if copy_count > 0:
            state[:copy_count] = flat_obs[:copy_count]
        return np.repeat(state.reshape(1, -1), batch_size, axis=0)

    def _as_batch_central_obs(
        self,
        central_obs: object | None,
        *,
        batch_size: int,
        fallback_obs_batch: np.ndarray,
    ) -> np.ndarray:
        if central_obs is None:
            return self._fallback_central_obs(fallback_obs_batch)

        central_array = np.asarray(central_obs, dtype=np.float32)
        if central_array.ndim == 1:
            central_batch = central_array.reshape(1, -1)
        elif central_array.ndim == 2:
            central_batch = central_array
        else:
            raise ValueError(f"PPO centralized obs expected ndim 1 or 2, got {central_array.ndim}.")

        if int(central_batch.shape[0]) == 1 and int(batch_size) > 1:
            central_batch = np.repeat(central_batch, int(batch_size), axis=0)
        if int(central_batch.shape[0]) != int(batch_size):
            raise ValueError(
                f"PPO centralized obs expected batch size {int(batch_size)}, got {int(central_batch.shape[0])}."
            )
        if int(central_batch.shape[1]) != int(self._central_obs_dim):
            raise ValueError(
                f"PPO centralized obs expected dim {int(self._central_obs_dim)}, got {int(central_batch.shape[1])}."
            )
        return central_batch.astype(np.float32, copy=False)

    def _build_critic_input(
        self,
        *,
        obs_batch: np.ndarray,
        central_obs_batch: np.ndarray,
    ) -> np.ndarray:
        if int(np.asarray(obs_batch).shape[0]) != int(np.asarray(central_obs_batch).shape[0]):
            raise ValueError(
                "PPO critic input build expected matching batch rows for obs and centralized obs."
            )
        if not self._critic_condition_on_agent_obs:
            return np.asarray(central_obs_batch, dtype=np.float32)
        return np.concatenate(
            (
                np.asarray(central_obs_batch, dtype=np.float32),
                np.asarray(obs_batch, dtype=np.float32),
            ),
            axis=1,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _broadcast_batch(values: object, batch_size: int, *, dtype: np.dtype) -> np.ndarray:
        value_array = np.asarray(values, dtype=dtype).reshape(-1)
        if value_array.size == 1:
            return np.full((int(batch_size),), value_array.item(), dtype=dtype)
        if int(value_array.size) != int(batch_size):
            raise ValueError(
                f"PPO expected batch size {int(batch_size)}, got {int(value_array.size)}."
            )
        return value_array.astype(dtype, copy=False)

    def _normalize_actions_for_storage(self, values: object, *, batch_size: int) -> np.ndarray:
        if self._is_discrete:
            return self._broadcast_batch(values, batch_size, dtype=np.int64)

        action_dim = int(self.config.action_dim)
        action_array = np.asarray(values, dtype=np.float32)
        if action_array.ndim == 0:
            action_array = action_array.reshape(1, 1)
        elif action_array.ndim == 1:
            if int(action_array.size) == int(action_dim):
                action_array = action_array.reshape(1, action_dim)
            elif int(action_dim) == 1 and int(action_array.size) == int(batch_size):
                action_array = action_array.reshape(batch_size, 1)
            else:
                raise ValueError(
                    "PPO continuous action batch expected shape (action_dim,) or (batch, action_dim)."
                )
        elif action_array.ndim != 2:
            raise ValueError(
                f"PPO continuous action batch expected ndim 1 or 2, got {action_array.ndim}."
            )

        if int(action_array.shape[0]) == 1 and int(batch_size) > 1:
            action_array = np.repeat(action_array, int(batch_size), axis=0)
        if int(action_array.shape[0]) != int(batch_size):
            raise ValueError(
                f"PPO expected action batch size {int(batch_size)}, got {int(action_array.shape[0])}."
            )
        if int(action_array.shape[1]) != int(action_dim):
            raise ValueError(
                f"PPO expected action dim {int(action_dim)}, got {int(action_array.shape[1])}."
            )
        return action_array.astype(np.float32, copy=False)

    def _clip_continuous_actions(self, action_batch: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(action_batch, dtype=np.float32), self._action_low, self._action_high)
        return clipped.astype(np.float32, copy=False)

    def _continuous_distribution(self, means: torch.Tensor) -> Normal:
        log_std = self.model.policy_log_std()
        std = torch.exp(log_std).unsqueeze(0).expand_as(means)
        return Normal(means, std)

    def act(
        self,
        obs: np.ndarray,
        explore: bool,
        action_mask: np.ndarray | None = None,
        central_obs: np.ndarray | None = None,
    ) -> int | np.ndarray:
        obs_batch = self._as_batch_obs(obs)
        batch_size = int(obs_batch.shape[0])

        mask_array = None
        if action_mask is not None and self._is_discrete:
            mask_array = self._normalize_action_mask(action_mask, batch_size=batch_size)

        central_batch = self._as_batch_central_obs(
            central_obs,
            batch_size=batch_size,
            fallback_obs_batch=obs_batch,
        )
        critic_input_batch = self._build_critic_input(
            obs_batch=obs_batch,
            central_obs_batch=central_batch,
        )

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        critic_obs_tensor = torch.as_tensor(critic_input_batch, dtype=torch.float32, device=self.device)
        policy_output, value = self.model(obs_tensor, critic_obs=critic_obs_tensor)

        if self._is_discrete:
            masked_logits = self._masked_logits(policy_output, mask_array)
            dist = Categorical(logits=masked_logits)
            if explore:
                action_tensor = dist.sample()
            else:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            log_prob = dist.log_prob(action_tensor)
            action_np = action_tensor.detach().cpu().numpy().astype(np.int64).reshape(-1)
            env_action_np: np.ndarray | int = action_np
            storage_action_np: np.ndarray | int = action_np
        else:
            dist = self._continuous_distribution(policy_output)
            if explore:
                raw_action_tensor = dist.sample()
            else:
                raw_action_tensor = policy_output
            log_prob = dist.log_prob(raw_action_tensor).sum(dim=-1)
            raw_action_np = raw_action_tensor.detach().cpu().numpy().astype(np.float32)
            env_action_np = self._clip_continuous_actions(raw_action_np)
            storage_action_np = raw_action_np

        log_prob_np = log_prob.detach().cpu().numpy().astype(np.float32).reshape(-1)
        value_np = value.detach().cpu().numpy().astype(np.float32).reshape(-1)

        if np.asarray(obs, dtype=np.float32).ndim == 1:
            self._last_log_prob = float(log_prob_np[0])
            self._last_value = float(value_np[0])
            if self._is_discrete:
                action_scalar = int(np.asarray(env_action_np, dtype=np.int64).reshape(-1)[0])
                self._last_action_for_storage = int(np.asarray(storage_action_np, dtype=np.int64).reshape(-1)[0])
                return action_scalar
            action_vec = np.asarray(env_action_np, dtype=np.float32).reshape(-1, int(self.config.action_dim))[0]
            storage_vec = np.asarray(storage_action_np, dtype=np.float32).reshape(-1, int(self.config.action_dim))[0]
            self._last_action_for_storage = storage_vec.astype(np.float32, copy=False)
            return action_vec.astype(np.float32, copy=False)

        self._last_log_prob = log_prob_np
        self._last_value = value_np
        if self._is_discrete:
            self._last_action_for_storage = np.asarray(storage_action_np, dtype=np.int64).reshape(-1)
            return np.asarray(env_action_np, dtype=np.int64).reshape(-1)

        self._last_action_for_storage = np.asarray(storage_action_np, dtype=np.float32).reshape(-1, int(self.config.action_dim))
        return np.asarray(env_action_np, dtype=np.float32).reshape(-1, int(self.config.action_dim))

    def observe(self, transition: dict[str, Any]) -> None:
        obs_batch = self._as_batch_obs(transition["obs"])
        next_obs_batch = self._as_batch_obs(transition["next_obs"])
        batch_size = int(obs_batch.shape[0])

        action_source = self._last_action_for_storage
        if action_source is None:
            action_source = transition["action"]
        actions = self._normalize_actions_for_storage(action_source, batch_size=batch_size)

        action_mask_raw = transition.get("action_mask")
        if self._is_discrete and action_mask_raw is not None:
            action_masks = self._normalize_action_mask(action_mask_raw, batch_size=batch_size)
        else:
            action_masks = np.ones((batch_size, int(self.config.action_dim)), dtype=np.bool_)

        rewards = self._broadcast_batch(transition["reward"], batch_size, dtype=np.float32)
        dones = self._broadcast_batch(transition["done"], batch_size, dtype=np.bool_)
        log_probs = self._broadcast_batch(self._last_log_prob, batch_size, dtype=np.float32)
        values = self._broadcast_batch(self._last_value, batch_size, dtype=np.float32)

        central_obs_batch = self._as_batch_central_obs(
            transition.get("central_obs"),
            batch_size=batch_size,
            fallback_obs_batch=obs_batch,
        )
        next_central_obs_batch = self._as_batch_central_obs(
            transition.get("next_central_obs"),
            batch_size=batch_size,
            fallback_obs_batch=next_obs_batch,
        )

        self.rollout.observations.append(obs_batch)
        self.rollout.centralized_observations.append(central_obs_batch)
        self.rollout.actions.append(actions)
        self.rollout.action_masks.append(action_masks)
        self.rollout.rewards.append(rewards)
        self.rollout.dones.append(dones)
        self.rollout.log_probs.append(log_probs)
        self.rollout.values.append(values)
        self.rollout.last_next_observation = next_obs_batch
        self.rollout.last_next_centralized_observation = next_central_obs_batch
        self.rollout.last_done = np.asarray(dones, dtype=np.bool_).reshape(-1)
        self._last_action_for_storage = None

    def _rollout_bootstrap_value(self, expected_batch_size: int) -> np.ndarray:
        if expected_batch_size <= 0:
            return np.zeros((0,), dtype=np.float32)

        if self.rollout.last_done is None or self.rollout.last_next_observation is None:
            return np.zeros((expected_batch_size,), dtype=np.float32)

        done_last = np.asarray(self.rollout.last_done, dtype=np.float32).reshape(-1)
        if int(done_last.size) != int(expected_batch_size):
            if int(done_last.size) == 1:
                done_last = np.full((expected_batch_size,), done_last.item(), dtype=np.float32)
            else:
                return np.zeros((expected_batch_size,), dtype=np.float32)

        if bool(np.all(done_last > 0.5)):
            return np.zeros((expected_batch_size,), dtype=np.float32)

        next_obs_batch = np.asarray(self.rollout.last_next_observation, dtype=np.float32)
        next_central_obs_raw = self.rollout.last_next_centralized_observation
        next_central_obs_batch = self._as_batch_central_obs(
            next_central_obs_raw,
            batch_size=expected_batch_size,
            fallback_obs_batch=next_obs_batch,
        )
        next_critic_input_batch = self._build_critic_input(
            obs_batch=next_obs_batch,
            central_obs_batch=next_central_obs_batch,
        )

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
            next_critic_obs_tensor = torch.as_tensor(next_critic_input_batch, dtype=torch.float32, device=self.device)
            _, next_values_tensor = self.model(next_obs_tensor, critic_obs=next_critic_obs_tensor)
        next_values = next_values_tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)

        if int(next_values.size) != int(expected_batch_size):
            if int(next_values.size) == 1:
                next_values = np.full((expected_batch_size,), next_values.item(), dtype=np.float32)
            else:
                return np.zeros((expected_batch_size,), dtype=np.float32)

        return next_values * (1.0 - done_last)

    def _compute_gae(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rewards_seq = [np.asarray(batch, dtype=np.float32).reshape(-1) for batch in self.rollout.rewards]
        dones_seq = [np.asarray(batch, dtype=np.float32).reshape(-1) for batch in self.rollout.dones]
        values_seq = [np.asarray(batch, dtype=np.float32).reshape(-1) for batch in self.rollout.values]

        advantages_rev: list[np.ndarray] = []
        returns_rev: list[np.ndarray] = []
        if rewards_seq:
            last_agent_count = int(rewards_seq[-1].shape[0])
        else:
            last_agent_count = 0
        next_value = self._rollout_bootstrap_value(last_agent_count)
        next_advantage = np.zeros_like(next_value, dtype=np.float32)

        for rewards_t, dones_t, values_t in zip(reversed(rewards_seq), reversed(dones_seq), reversed(values_seq)):
            if rewards_t.shape != dones_t.shape or rewards_t.shape != values_t.shape:
                raise RuntimeError(
                    "PPO rollout batch shapes are inconsistent: "
                    f"rewards={rewards_t.shape}, dones={dones_t.shape}, values={values_t.shape}"
                )

            agent_count = int(rewards_t.shape[0])
            if next_value.shape[0] != agent_count:
                next_value = np.zeros((agent_count,), dtype=np.float32)
                next_advantage = np.zeros((agent_count,), dtype=np.float32)

            mask = 1.0 - dones_t
            delta = rewards_t + float(self.config.gamma) * next_value * mask - values_t
            advantage_t = delta + float(self.config.gamma) * float(self.config.gae_lambda) * mask * next_advantage
            return_t = advantage_t + values_t

            advantages_rev.append(advantage_t.astype(np.float32, copy=False))
            returns_rev.append(return_t.astype(np.float32, copy=False))

            next_value = values_t
            next_advantage = advantage_t

        advantages = list(reversed(advantages_rev))
        returns = list(reversed(returns_rev))

        if advantages:
            flat_advantages = np.concatenate(advantages, axis=0)
            if flat_advantages.size > 1:
                mean = float(flat_advantages.mean())
                std = float(flat_advantages.std())
                advantages = [(adv - mean) / (std + 1e-8) for adv in advantages]
        return advantages, returns

    def update(self) -> dict[str, float]:
        if len(self.rollout) == 0:
            return {}

        advantages_batches, returns_batches = self._compute_gae()

        obs_flat = np.concatenate(
            [
                np.asarray(batch, dtype=np.float32).reshape(-1, int(self.config.obs_dim))
                for batch in self.rollout.observations
            ],
            axis=0,
        )
        central_obs_flat = np.concatenate(
            [
                np.asarray(batch, dtype=np.float32).reshape(-1, int(self._central_obs_dim))
                for batch in self.rollout.centralized_observations
            ],
            axis=0,
        )
        if self._is_discrete:
            actions_flat = np.concatenate(
                [np.asarray(batch, dtype=np.int64).reshape(-1) for batch in self.rollout.actions],
                axis=0,
            )
            action_masks_flat = np.concatenate(
                [
                    np.asarray(batch, dtype=np.bool_).reshape(-1, int(self.config.action_dim))
                    for batch in self.rollout.action_masks
                ],
                axis=0,
            )
        else:
            actions_flat = np.concatenate(
                [
                    np.asarray(batch, dtype=np.float32).reshape(-1, int(self.config.action_dim))
                    for batch in self.rollout.actions
                ],
                axis=0,
            )
            action_masks_flat = None

        old_log_probs_flat = np.concatenate(
            [np.asarray(batch, dtype=np.float32).reshape(-1) for batch in self.rollout.log_probs],
            axis=0,
        )
        advantages_flat = np.concatenate(
            [np.asarray(batch, dtype=np.float32).reshape(-1) for batch in advantages_batches],
            axis=0,
        )
        returns_flat = np.concatenate(
            [np.asarray(batch, dtype=np.float32).reshape(-1) for batch in returns_batches],
            axis=0,
        )

        critic_input_flat = self._build_critic_input(
            obs_batch=obs_flat,
            central_obs_batch=central_obs_flat,
        )
        obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)
        critic_obs_tensor = torch.as_tensor(critic_input_flat, dtype=torch.float32, device=self.device)
        if self._is_discrete:
            action_tensor = torch.as_tensor(actions_flat, dtype=torch.long, device=self.device)
            action_mask_tensor = torch.as_tensor(action_masks_flat, dtype=torch.bool, device=self.device)
        else:
            action_tensor = torch.as_tensor(actions_flat, dtype=torch.float32, device=self.device)
            action_mask_tensor = None
        old_log_probs = torch.as_tensor(old_log_probs_flat, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages_flat, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns_flat, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clip_frac_sum = 0.0
        update_steps = 0
        sample_count = int(obs_flat.shape[0])

        for _ in range(int(self.config.update_epochs)):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, int(self.config.minibatch_size)):
                end = min(sample_count, start + int(self.config.minibatch_size))
                batch_idx = permutation[start:end]

                batch_obs = obs_tensor[batch_idx]
                batch_critic_obs = critic_obs_tensor[batch_idx]
                batch_actions = action_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                policy_output, values = self.model(batch_obs, critic_obs=batch_critic_obs)
                if self._is_discrete:
                    if action_mask_tensor is None:
                        raise RuntimeError("PPO discrete update requires action masks tensor.")
                    batch_action_masks = action_mask_tensor[batch_idx]
                    masked_logits = policy_output.masked_fill(~batch_action_masks, float(-1e9))
                    dist = Categorical(logits=masked_logits)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                else:
                    dist = self._continuous_distribution(policy_output)
                    log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.config.clip_ratio),
                    1.0 + float(self.config.clip_ratio),
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)
                loss = (
                    policy_loss
                    + float(self.config.value_coef) * value_loss
                    - float(self.config.entropy_coef) * entropy
                )
                with torch.no_grad():
                    approx_kl = torch.mean(batch_old_log_probs - log_probs)
                    clip_frac = torch.mean(
                        (torch.abs(ratio - 1.0) > float(self.config.clip_ratio)).float()
                    )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
                self.optimizer.step()

                total_loss += float(loss.item())
                policy_loss_sum += float(policy_loss.item())
                value_loss_sum += float(value_loss.item())
                entropy_sum += float(entropy.item())
                approx_kl_sum += float(approx_kl.item())
                clip_frac_sum += float(clip_frac.item())
                update_steps += 1

        denom = max(1, int(update_steps))
        mean_loss = total_loss / float(denom)
        mean_policy_loss = policy_loss_sum / float(denom)
        mean_value_loss = value_loss_sum / float(denom)
        mean_entropy = entropy_sum / float(denom)
        mean_approx_kl = approx_kl_sum / float(denom)
        mean_clip_frac = clip_frac_sum / float(denom)
        self.rollout.clear()
        return {
            "loss": mean_loss,
            "policy_loss": mean_policy_loss,
            "value_loss": mean_value_loss,
            "entropy": mean_entropy,
            "approx_kl": mean_approx_kl,
            "clip_frac": mean_clip_frac,
        }

    def save(self, path: str) -> None:
        save_torch_checkpoint(
            path,
            {
                "algo_id": self.algo_id,
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
        )

    def load(self, path: str) -> None:
        checkpoint = load_torch_checkpoint(path, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            return

        self.model.load_state_dict(checkpoint)
