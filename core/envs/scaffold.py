"""Lightweight placeholder environments for scaffold-first game entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.envs.base import Env


@dataclass
class ScaffoldEnv(Env):
    """Minimal env that keeps scaffolded games runnable without fake gameplay systems."""

    game_id: str
    obs_dim: int
    mode: str
    render: bool
    level: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        self._current_level = max(1, int(self.level or 1))
        self._done = False

    def reset(self) -> np.ndarray:
        self._done = False
        return np.zeros((int(self.obs_dim),), dtype=np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        del action
        obs = np.zeros((int(self.obs_dim),), dtype=np.float32)
        done = not self._done
        self._done = True
        return (
            obs,
            0.0,
            done,
            {
                "level": int(self._current_level),
                "success": 0,
                "scaffold": True,
                "game_id": self.game_id,
                "note": self.note,
            },
        )
