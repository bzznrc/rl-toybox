"""Rollout storage for PPO."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    centralized_observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    dones: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    last_next_observation: np.ndarray | None = None
    last_next_centralized_observation: np.ndarray | None = None
    last_done: np.ndarray | None = None

    def clear(self) -> None:
        self.observations.clear()
        self.centralized_observations.clear()
        self.actions.clear()
        self.action_masks.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.last_next_observation = None
        self.last_next_centralized_observation = None
        self.last_done = None

    def __len__(self) -> int:
        return len(self.observations)
