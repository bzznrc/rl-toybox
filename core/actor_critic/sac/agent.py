"""Soft actor-critic for compact continuous-control games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from core.algorithms.base import Algorithm
from core.actor_critic.sac.networks import ActorNetwork, CriticNetwork
from core.actor_critic.sac.replay import SACReplayBuffer
from core.io.checkpoint import load_torch_checkpoint, save_torch_checkpoint


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: list[int] | None = None
    action_low: float | list[float] | np.ndarray = -1.0
    action_high: float | list[float] | np.ndarray = 1.0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    replay_size: int = 200_000
    tau: float = 0.005
    grad_clip_norm: float = 10.0
    init_alpha: float = 0.2
    target_entropy: float | None = None
    use_gpu: bool = False


class SACAlgorithm(Algorithm):
    algo_id = "sac"
    LOG_PROB_EPS = 1e-6

    def __init__(self, config: SACConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        self.hidden_sizes = [int(size) for size in (config.hidden_sizes or [128, 128])]
        self.action_low = self._resolve_action_bound(config.action_low, fill_value=-1.0)
        self.action_high = self._resolve_action_bound(config.action_high, fill_value=1.0)
        if np.any(self.action_low >= self.action_high):
            raise ValueError("SAC action bounds require low < high for every action dimension.")

        self.action_scale = torch.as_tensor(
            (self.action_high - self.action_low) * 0.5,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_bias = torch.as_tensor(
            (self.action_high + self.action_low) * 0.5,
            dtype=torch.float32,
            device=self.device,
        )
        self._log_action_scale = torch.log(torch.clamp(self.action_scale, min=self.LOG_PROB_EPS))

        self.actor = ActorNetwork(config.obs_dim, config.action_dim, self.hidden_sizes).to(self.device)
        self.critic_1 = CriticNetwork(config.obs_dim, config.action_dim, self.hidden_sizes).to(self.device)
        self.critic_2 = CriticNetwork(config.obs_dim, config.action_dim, self.hidden_sizes).to(self.device)
        self.target_critic_1 = CriticNetwork(config.obs_dim, config.action_dim, self.hidden_sizes).to(self.device)
        self.target_critic_2 = CriticNetwork(config.obs_dim, config.action_dim, self.hidden_sizes).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(config.learning_rate))
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=float(config.learning_rate),
        )
        self.log_alpha = torch.tensor(
            float(np.log(max(float(config.init_alpha), 1e-6))),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(config.learning_rate))
        target_entropy = config.target_entropy
        if target_entropy is None:
            target_entropy = -float(config.action_dim)
        self.target_entropy = float(target_entropy)
        self.replay = SACReplayBuffer(
            int(config.replay_size),
            obs_dim=int(config.obs_dim),
            action_dim=int(config.action_dim),
        )
        self.total_env_steps = 0
        self.training_steps = 0

    def _resolve_action_bound(
        self,
        raw_value: float | list[float] | np.ndarray,
        *,
        fill_value: float,
    ) -> np.ndarray:
        bound = np.asarray(raw_value, dtype=np.float32).reshape(-1)
        if bound.size == 0:
            bound = np.asarray([float(fill_value)], dtype=np.float32)
        if bound.size == 1:
            return np.full((int(self.config.action_dim),), float(bound.item()), dtype=np.float32)
        if bound.shape != (int(self.config.action_dim),):
            raise ValueError("SAC action bounds must match action_dim.")
        return bound.astype(np.float32, copy=True)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().detach().cpu().item())

    def _sample_action_and_log_prob(
        self,
        obs_tensor: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(obs_tensor)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        if deterministic:
            log_prob = torch.zeros((obs_tensor.shape[0],), dtype=torch.float32, device=self.device)
        else:
            correction = torch.log(torch.clamp(1.0 - squashed.pow(2), min=self.LOG_PROB_EPS))
            log_prob = dist.log_prob(pre_tanh) - correction - self._log_action_scale
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def _soft_update_targets(self) -> None:
        tau = float(self.config.tau)
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def act(self, obs: np.ndarray, explore: bool) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32).reshape(1, int(self.config.obs_dim))
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_tensor, _ = self._sample_action_and_log_prob(obs_tensor, deterministic=not bool(explore))
        return action_tensor.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def observe(self, transition: dict[str, Any]) -> None:
        self.replay.add(
            (
                np.asarray(transition["obs"], dtype=np.float32),
                np.asarray(transition["action"], dtype=np.float32),
                float(transition["reward"]),
                np.asarray(transition["next_obs"], dtype=np.float32),
                bool(transition["done"]),
            )
        )
        self.total_env_steps += 1

    def update(self) -> dict[str, float]:
        if len(self.replay) < int(self.config.batch_size):
            return {}

        batch = self.replay.sample(int(self.config.batch_size))
        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_prob = self._sample_action_and_log_prob(next_obs, deterministic=False)
            next_q1 = self.target_critic_1(next_obs, next_actions)
            next_q2 = self.target_critic_2(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.log_alpha.exp() * next_log_prob
            q_target = rewards + (1.0 - dones) * float(self.config.gamma) * next_q

        q1 = self.critic_1(obs, actions)
        q2 = self.critic_2(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            float(self.config.grad_clip_norm),
        )
        self.critic_optimizer.step()

        policy_actions, log_prob = self._sample_action_and_log_prob(obs, deterministic=False)
        q1_pi = self.critic_1(obs, policy_actions)
        q2_pi = self.critic_2(obs, policy_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - q_pi).mean()
        policy_entropy = float((-log_prob.detach()).mean().item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config.grad_clip_norm))
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + float(self.target_entropy))).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update_targets()
        self.training_steps += 1

        return {
            "loss": float(critic_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(policy_entropy),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha),
        }

    def save(self, path: str) -> None:
        save_torch_checkpoint(
            path,
            {
                "algo_id": self.algo_id,
                "config": {
                    "obs_dim": int(self.config.obs_dim),
                    "action_dim": int(self.config.action_dim),
                    "hidden_sizes": list(self.hidden_sizes),
                    "action_low": self.action_low.tolist(),
                    "action_high": self.action_high.tolist(),
                    "learning_rate": float(self.config.learning_rate),
                    "gamma": float(self.config.gamma),
                    "batch_size": int(self.config.batch_size),
                    "replay_size": int(self.config.replay_size),
                    "tau": float(self.config.tau),
                    "grad_clip_norm": float(self.config.grad_clip_norm),
                    "init_alpha": float(self.config.init_alpha),
                    "target_entropy": float(self.target_entropy),
                },
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": float(self.log_alpha.detach().cpu().item()),
                "total_env_steps": int(self.total_env_steps),
                "training_steps": int(self.training_steps),
            },
        )

    def load(self, path: str) -> None:
        checkpoint = load_torch_checkpoint(path, map_location=self.device)
        actor_state = checkpoint.get("actor")
        critic_1_state = checkpoint.get("critic_1")
        critic_2_state = checkpoint.get("critic_2")
        if actor_state is None or critic_1_state is None or critic_2_state is None:
            raise RuntimeError("SAC checkpoint missing actor/critic state.")
        self.actor.load_state_dict(actor_state)
        self.critic_1.load_state_dict(critic_1_state)
        self.critic_2.load_state_dict(critic_2_state)
        self.target_critic_1.load_state_dict(checkpoint.get("target_critic_1", critic_1_state))
        self.target_critic_2.load_state_dict(checkpoint.get("target_critic_2", critic_2_state))

        actor_optim = checkpoint.get("actor_optimizer")
        critic_optim = checkpoint.get("critic_optimizer")
        alpha_optim = checkpoint.get("alpha_optimizer")
        if actor_optim is not None:
            self.actor_optimizer.load_state_dict(actor_optim)
        if critic_optim is not None:
            self.critic_optimizer.load_state_dict(critic_optim)
        if alpha_optim is not None:
            self.alpha_optimizer.load_state_dict(alpha_optim)

        self.log_alpha.data.fill_(float(checkpoint.get("log_alpha", self.log_alpha.item())))
        self.total_env_steps = int(checkpoint.get("total_env_steps", self.total_env_steps))
        self.training_steps = int(checkpoint.get("training_steps", self.training_steps))
