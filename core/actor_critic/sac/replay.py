"""Replay buffer for soft actor-critic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SACReplayBuffer:
    def __init__(self, capacity: int, *, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._size = 0
        self._index = 0

    def __len__(self) -> int:
        return int(self._size)

    def add(self, transition) -> None:
        obs, action, reward, next_obs, done = transition
        self._obs[self._index] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self._actions[self._index] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self._rewards[self._index] = float(reward)
        self._next_obs[self._index] = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        self._dones[self._index] = 1.0 if bool(done) else 0.0
        self._index = (self._index + 1) % max(1, self.capacity)
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int):
        if self._size <= 0:
            raise RuntimeError("SAC replay buffer is empty.")
        indices = np.random.randint(0, self._size, size=int(batch_size))
        return ReplayBatch(
            observations=self._obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_obs[indices],
            dones=self._dones[indices],
        )


@dataclass(frozen=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
