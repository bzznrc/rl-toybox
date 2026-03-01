"""Q-learning trainer/agent wrapper preserving Snake LinearQNet behavior."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.algorithms.base import Algorithm
from core.algorithms.exploration import EpsilonController, ExplorationConfig, resolve_exploration_config
from core.algorithms.qlearn.networks import LinearQNet
from core.io.checkpoint import load_torch_checkpoint, save_torch_checkpoint


@dataclass
class QLearnConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: list[int]
    learning_rate: float = 1e-3
    gamma: float = 0.9
    max_memory: int = 100_000
    batch_size: int = 1_000
    exploration: ExplorationConfig | dict[str, object] | None = None
    use_gpu: bool = False


class QTrainer:
    def __init__(self, model: LinearQNet, lr: float, gamma: float):
        self.model = model
        self.device = next(model.parameters()).device
        self.gamma = float(gamma)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> float:
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        next_state_tensor = torch.as_tensor(
            np.asarray(next_state, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        action_tensor = torch.as_tensor(
            np.asarray(action, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        reward_tensor = torch.as_tensor(
            np.asarray(reward, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        done_tensor = torch.as_tensor(np.asarray(done, dtype=np.bool_), dtype=torch.bool, device=self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            next_state_tensor = next_state_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
            reward_tensor = reward_tensor.unsqueeze(0)
            done_tensor = done_tensor.unsqueeze(0)

        pred = self.model(state_tensor)
        target = pred.clone().detach()

        for idx in range(len(done_tensor)):
            q_new = reward_tensor[idx]
            if not bool(done_tensor[idx].item()):
                q_new = reward_tensor[idx] + self.gamma * torch.max(self.model(next_state_tensor[idx]))
            target[idx][torch.argmax(action_tensor[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


class QLearnAlgorithm(Algorithm):
    algo_id = "qlearn"

    def __init__(self, config: QLearnConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

        self.model = LinearQNet(
            input_size=config.obs_dim,
            hidden_layers=config.hidden_sizes,
            output_size=config.action_dim,
        ).to(self.device)
        self.trainer = QTrainer(self.model, lr=config.learning_rate, gamma=config.gamma)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=int(config.max_memory))

        self._exploration = EpsilonController(resolve_exploration_config(config.exploration))
        self.epsilon = float(self._exploration.epsilon)
        self.n_games = 0
        self._episode_done = False
        self._last_short_loss = 0.0

    def _one_hot(self, action_idx: int) -> list[int]:
        action = [0] * int(self.config.action_dim)
        action[int(action_idx)] = 1
        return action

    def act(self, obs: np.ndarray, explore: bool) -> int:
        if explore:
            if random.random() < self.epsilon:
                return random.randint(0, int(self.config.action_dim) - 1)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.model(obs_tensor)
        return int(torch.argmax(prediction, dim=1).item())

    def observe(self, transition: dict[str, Any]) -> None:
        obs = np.asarray(transition["obs"], dtype=np.float32)
        action = int(transition["action"])
        reward = float(transition["reward"])
        next_obs = np.asarray(transition["next_obs"], dtype=np.float32)
        done = bool(transition["done"])

        self.memory.append((obs, action, reward, next_obs, done))
        self._last_short_loss = self.trainer.train_step(
            obs,
            self._one_hot(action),
            reward,
            next_obs,
            done,
        )
        self.epsilon = float(self._exploration.advance_step())
        self._episode_done = done

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
        if not self._episode_done:
            return {"loss": float(self._last_short_loss), "epsilon": float(self.epsilon)}

        if len(self.memory) > int(self.config.batch_size):
            mini_batch = random.sample(self.memory, int(self.config.batch_size))
        else:
            mini_batch = list(self.memory)

        if not mini_batch:
            self._episode_done = False
            return {}

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        action_one_hot = [self._one_hot(action) for action in actions]
        long_loss = self.trainer.train_step(states, action_one_hot, rewards, next_states, dones)

        self.n_games += 1
        self._episode_done = False
        return {
            "loss": float(long_loss),
            "epsilon": float(self.epsilon),
            "episodes": float(self.n_games),
        }

    def save(self, path: str) -> None:
        save_torch_checkpoint(
            path,
            {
                "algo_id": self.algo_id,
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "exploration_state": self._exploration.state_dict(),
                "n_games": self.n_games,
            },
        )

    def load(self, path: str) -> None:
        checkpoint = load_torch_checkpoint(path, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                self.trainer.optimizer.load_state_dict(optimizer_state)
            exploration_state = checkpoint.get("exploration_state")
            if isinstance(exploration_state, dict):
                self._exploration.load_state_dict(exploration_state)
            else:
                self._exploration.set_epsilon(float(checkpoint.get("epsilon", self.epsilon)))
            self.epsilon = float(self._exploration.epsilon)
            self.n_games = int(checkpoint.get("n_games", self.n_games))
            return

        self.model.load_state_dict(checkpoint)
