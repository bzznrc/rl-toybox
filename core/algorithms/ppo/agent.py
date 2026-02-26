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

        self._last_log_prob = 0.0
        self._last_value = 0.0

    def act(self, obs: np.ndarray, explore: bool) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.model(obs_tensor)
        dist = Categorical(logits=logits)

        if explore:
            action_tensor = dist.sample()
        else:
            action_tensor = torch.argmax(logits, dim=-1)

        log_prob = dist.log_prob(action_tensor)
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())
        return int(action_tensor.item())

    def observe(self, transition: dict[str, Any]) -> None:
        self.rollout.observations.append(np.asarray(transition["obs"], dtype=np.float32))
        self.rollout.actions.append(int(transition["action"]))
        self.rollout.rewards.append(float(transition["reward"]))
        self.rollout.dones.append(bool(transition["done"]))
        self.rollout.log_probs.append(self._last_log_prob)
        self.rollout.values.append(self._last_value)

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.rollout.rewards, dtype=np.float32)
        dones = np.asarray(self.rollout.dones, dtype=np.float32)
        values = np.asarray(self.rollout.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        last_advantage = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + float(self.config.gamma) * next_value * mask - values[t]
            last_advantage = delta + float(self.config.gamma) * float(self.config.gae_lambda) * mask * last_advantage
            advantages[t] = last_advantage
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self) -> dict[str, float]:
        if len(self.rollout) == 0:
            return {}

        advantages, returns = self._compute_gae()

        obs_tensor = torch.as_tensor(np.asarray(self.rollout.observations), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(self.rollout.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.rollout.log_probs), dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        sample_count = len(self.rollout)

        for _ in range(int(self.config.update_epochs)):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, int(self.config.minibatch_size)):
                end = min(sample_count, start + int(self.config.minibatch_size))
                batch_idx = permutation[start:end]

                batch_obs = obs_tensor[batch_idx]
                batch_actions = action_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                logits, values = self.model(batch_obs)
                dist = Categorical(logits=logits)
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
