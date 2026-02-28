"""DQN algorithm wrapper with vanilla and enhanced modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.algorithms.base import Algorithm
from core.algorithms.exploration import EpsilonController, ExplorationConfig, resolve_exploration_config
from core.algorithms.dqn.networks import build_q_network
from core.algorithms.dqn.replay import (
    PrioritizedReplayBuffer,
    PrioritizedReplayConfig,
    UniformReplayBuffer,
)
from core.io.checkpoint import load_torch_checkpoint, save_torch_checkpoint


@dataclass
class DQNConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: list[int]
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gamma: float = 0.98
    batch_size: int = 128
    replay_size: int = 150_000
    target_sync_every: int = 500
    grad_clip_norm: float = 10.0
    learn_start_steps: int = 5_000
    train_every_steps: int = 4
    exploration: ExplorationConfig | dict[str, object] = field(default_factory=ExplorationConfig)
    use_gpu: bool = False
    dueling: bool = True
    double_dqn: bool = True
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 40_000_000
    per_epsilon: float = 1e-4


class DQNAlgorithm(Algorithm):
    algo_id = "dqn"

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

        self.online_model = build_q_network(
            input_size=config.obs_dim,
            hidden_sizes=config.hidden_sizes,
            output_size=config.action_dim,
            dueling=config.dueling,
        ).to(self.device)
        self.target_model = build_q_network(
            input_size=config.obs_dim,
            hidden_sizes=config.hidden_sizes,
            output_size=config.action_dim,
            dueling=config.dueling,
        ).to(self.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.AdamW(
            self.online_model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        if config.prioritized_replay:
            self.replay = PrioritizedReplayBuffer(
                capacity=config.replay_size,
                config=PrioritizedReplayConfig(
                    alpha=config.per_alpha,
                    beta_start=config.per_beta_start,
                    beta_frames=config.per_beta_frames,
                    epsilon=config.per_epsilon,
                ),
            )
        else:
            self.replay = UniformReplayBuffer(capacity=config.replay_size)

        self.total_env_steps = 0
        self.training_steps = 0
        self._exploration = EpsilonController(resolve_exploration_config(config.exploration))
        self.epsilon = float(self._exploration.epsilon)

    def act(self, obs: np.ndarray, explore: bool) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, int(self.config.action_dim) - 1)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q_values = self.online_model(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def observe(self, transition: dict[str, Any]) -> None:
        self.replay.add(
            (
                np.asarray(transition["obs"], dtype=np.float32),
                int(transition["action"]),
                float(transition["reward"]),
                np.asarray(transition["next_obs"], dtype=np.float32),
                bool(transition["done"]),
            )
        )
        self.total_env_steps += 1
        self.epsilon = float(self._exploration.advance_step())

    def on_episode_end(self, avg_reward: float) -> dict[str, float | int | str] | None:
        bump = self._exploration.on_episode_end(avg_reward)
        self.epsilon = float(self._exploration.epsilon)
        if bump is None:
            return None
        return {
            "bump": "on",
            "epsilon": float(bump.epsilon),
            "cooldown_steps": int(bump.cooldown_steps),
            "reason": str(bump.reason),
        }

    def exploration_avg_window(self) -> int | None:
        return int(self._exploration.config.avg_window_episodes)

    def update(self) -> dict[str, float]:
        if self.total_env_steps < int(self.config.learn_start_steps):
            return {}
        if self.total_env_steps % int(self.config.train_every_steps) != 0:
            return {}
        if len(self.replay) < int(self.config.batch_size):
            return {}

        batch, indices, is_weights = self.replay.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        next_state_tensor = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(np.asarray(dones), dtype=torch.bool, device=self.device)
        weight_tensor = torch.as_tensor(np.asarray(is_weights), dtype=torch.float32, device=self.device)

        current_q = self.online_model(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online_model(next_state_tensor).argmax(dim=1)
            else:
                next_actions = self.target_model(next_state_tensor).argmax(dim=1)
            next_q = self.target_model(next_state_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = reward_tensor + (~done_tensor).float() * float(self.config.gamma) * next_q

        td_errors = target_q - current_q
        loss_values = self.loss_fn(current_q, target_q)
        loss = (loss_values * weight_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), float(self.config.grad_clip_norm))
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors.detach().abs().cpu().tolist())

        self.training_steps += 1
        if self.training_steps % int(self.config.target_sync_every) == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())

        return {
            "loss": float(loss.item()),
            "epsilon": float(self.epsilon),
        }

    def save(self, path: str) -> None:
        save_torch_checkpoint(
            path,
            {
                "algo_id": self.algo_id,
                "config": asdict(self.config),
                "online_model": self.online_model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "exploration_state": self._exploration.state_dict(),
                "total_env_steps": self.total_env_steps,
                "training_steps": self.training_steps,
            },
        )

    def load(self, path: str) -> None:
        checkpoint = load_torch_checkpoint(path, map_location=self.device)
        if "online_model" in checkpoint:
            self.online_model.load_state_dict(checkpoint["online_model"])
            target_state = checkpoint.get("target_model", checkpoint["online_model"])
            self.target_model.load_state_dict(target_state)
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            exploration_state = checkpoint.get("exploration_state")
            if isinstance(exploration_state, dict):
                self._exploration.load_state_dict(exploration_state)
            else:
                self._exploration.set_epsilon(float(checkpoint.get("epsilon", self.epsilon)))
            self.epsilon = float(self._exploration.epsilon)
            self.total_env_steps = int(checkpoint.get("total_env_steps", self.total_env_steps))
            self.training_steps = int(checkpoint.get("training_steps", self.training_steps))
            return

        # Compatibility path: raw model state dict.
        self.online_model.load_state_dict(checkpoint)
        self.target_model.load_state_dict(self.online_model.state_dict())
