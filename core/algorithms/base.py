"""Algorithm interface used by shared runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Algorithm(ABC):
    algo_id: str

    @abstractmethod
    def act(self, obs: np.ndarray, explore: bool) -> int | np.ndarray:
        """Return action for the given observation."""

    @abstractmethod
    def observe(self, transition: dict[str, Any]) -> None:
        """Consume a transition (for replay buffers or rollout storage)."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Run one algorithm update step and return metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist state to checkpoint path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load state from checkpoint path."""
