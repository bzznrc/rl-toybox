"""SAC algorithm stub (continuous control) for future Walk game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.algorithms.base import Algorithm
from core.io.checkpoint import load_torch_checkpoint, save_torch_checkpoint


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: list[int] | None = None  # Future network sizes (SAC still stubbed).
    action_low: float = -1.0
    action_high: float = 1.0


class SACAlgorithm(Algorithm):
    algo_id = "sac"

    def __init__(self, config: SACConfig):
        self.config = config

    def act(self, obs: np.ndarray, explore: bool) -> np.ndarray:
        del obs, explore
        # TODO: implement stochastic Gaussian actor and tanh-squashed sampling.
        return np.zeros((self.config.action_dim,), dtype=np.float32)

    def observe(self, transition: dict[str, Any]) -> None:
        del transition
        # TODO: store transitions in replay buffer.

    def update(self) -> dict[str, float]:
        # TODO: implement SAC update logic (actor + critics + alpha tuning).
        return {}

    def save(self, path: str) -> None:
        save_torch_checkpoint(
            path,
            {
                "algo_id": self.algo_id,
                "config": {
                    "obs_dim": self.config.obs_dim,
                    "action_dim": self.config.action_dim,
                    "hidden_sizes": list(self.config.hidden_sizes or []),
                    "action_low": self.config.action_low,
                    "action_high": self.config.action_high,
                },
            },
        )

    def load(self, path: str) -> None:
        _ = load_torch_checkpoint(path)
