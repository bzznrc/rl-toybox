"""Minimal PPO implementation for toy environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core.algorithms.base import Algorithm
from core.algorithms.ppo.networks import ActorCritic
from core.algorithms.ppo.rollout import RolloutBuffer
from core.io.checkpoint import load_torch_checkpoint, save_torch_checkpoint


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: list[int]
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


class PPOAlgorithm(Algorithm):
    algo_id = "ppo"

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(config.obs_dim, config.action_dim, config.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config.learning_rate))
        self.rollout = RolloutBuffer()

        self._last_log_prob: float | np.ndarray = 0.0
        self._last_value: float | np.ndarray = 0.0

    def _normalize_action_mask(
        self,
        action_mask: object,
        *,
        batch_size: int,
    ) -> np.ndarray:
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

    def act(
        self,
        obs: np.ndarray,
        explore: bool,
        action_mask: np.ndarray | None = None,
    ) -> int | np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        if obs_array.ndim not in {1, 2}:
            raise ValueError(f"PPO expected obs ndim 1 or 2, got {obs_array.ndim}.")

        batch_size = 1 if obs_array.ndim == 1 else int(obs_array.shape[0])
        mask_array = None
        if action_mask is not None:
            mask_array = self._normalize_action_mask(action_mask, batch_size=batch_size)

        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
        logits, value = self.model(obs_tensor)
        masked_logits = self._masked_logits(logits, mask_array)
        dist = Categorical(logits=masked_logits)

        if explore:
            action_tensor = dist.sample()
        else:
            action_tensor = torch.argmax(logits, dim=-1)

        log_prob = dist.log_prob(action_tensor)
        action_np = action_tensor.detach().cpu().numpy().astype(np.int64).reshape(-1)
        log_prob_np = log_prob.detach().cpu().numpy().astype(np.float32).reshape(-1)
        value_np = value.detach().cpu().numpy().astype(np.float32).reshape(-1)

        if obs_array.ndim == 1:
            self._last_log_prob = float(log_prob_np[0])
            self._last_value = float(value_np[0])
            return int(action_np[0])

        self._last_log_prob = log_prob_np
        self._last_value = value_np
        return action_np

    @staticmethod
    def _as_batch_obs(obs: object) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        if obs_array.ndim == 1:
            return obs_array.reshape(1, -1)
        if obs_array.ndim == 2:
            return obs_array
        raise ValueError(f"PPO observe expected obs ndim 1 or 2, got {obs_array.ndim}.")

    @staticmethod
    def _broadcast_batch(values: object, batch_size: int, *, dtype: np.dtype) -> np.ndarray:
        value_array = np.asarray(values, dtype=dtype).reshape(-1)
        if value_array.size == 1:
            return np.full((int(batch_size),), value_array.item(), dtype=dtype)
        if int(value_array.size) != int(batch_size):
            raise ValueError(
                f"PPO observe expected batch size {int(batch_size)}, got {int(value_array.size)}."
            )
        return value_array.astype(dtype, copy=False)

    def observe(self, transition: dict[str, Any]) -> None:
        obs_batch = self._as_batch_obs(transition["obs"])
        batch_size = int(obs_batch.shape[0])

        actions = self._broadcast_batch(transition["action"], batch_size, dtype=np.int64)
        action_mask_raw = transition.get("action_mask")
        if action_mask_raw is None:
            action_masks = np.ones((batch_size, int(self.config.action_dim)), dtype=np.bool_)
        else:
            action_masks = self._normalize_action_mask(action_mask_raw, batch_size=batch_size)
        rewards = self._broadcast_batch(transition["reward"], batch_size, dtype=np.float32)
        dones = self._broadcast_batch(transition["done"], batch_size, dtype=np.bool_)
        log_probs = self._broadcast_batch(self._last_log_prob, batch_size, dtype=np.float32)
        values = self._broadcast_batch(self._last_value, batch_size, dtype=np.float32)

        self.rollout.observations.append(obs_batch)
        self.rollout.actions.append(actions)
        self.rollout.action_masks.append(action_masks)
        self.rollout.rewards.append(rewards)
        self.rollout.dones.append(dones)
        self.rollout.log_probs.append(log_probs)
        self.rollout.values.append(values)
        self.rollout.last_next_observation = self._as_batch_obs(transition["next_obs"])
        self.rollout.last_done = np.asarray(dones, dtype=np.bool_).reshape(-1)

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
        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
            _, next_values_tensor = self.model(next_obs_tensor)
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
            [np.asarray(batch, dtype=np.float32).reshape(-1, int(self.config.obs_dim)) for batch in self.rollout.observations],
            axis=0,
        )
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

        obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions_flat, dtype=torch.long, device=self.device)
        action_mask_tensor = torch.as_tensor(action_masks_flat, dtype=torch.bool, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs_flat, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages_flat, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns_flat, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        sample_count = int(obs_flat.shape[0])

        for _ in range(int(self.config.update_epochs)):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, int(self.config.minibatch_size)):
                end = min(sample_count, start + int(self.config.minibatch_size))
                batch_idx = permutation[start:end]

                batch_obs = obs_tensor[batch_idx]
                batch_actions = action_tensor[batch_idx]
                batch_action_masks = action_mask_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                logits, values = self.model(batch_obs)
                masked_logits = logits.masked_fill(~batch_action_masks, float(-1e9))
                dist = Categorical(logits=masked_logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

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

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
                self.optimizer.step()

                total_loss += float(loss.item())

        mean_loss = total_loss / max(1, int(self.config.update_epochs))
        self.rollout.clear()
        return {"loss": mean_loss}

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
