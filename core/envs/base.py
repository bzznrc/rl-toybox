"""Minimal environment interface for toy RL environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Env(ABC):
    """Common environment interface for runners and scripts."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation vector."""

    @abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Step environment with an action and return transition tuple."""

    def render(self) -> None:
        """Render one frame. Envs that auto-render can keep this as a no-op."""

    def close(self) -> None:
        """Release any external resources."""
