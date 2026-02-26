"""Small action/observation space helpers (gym-free)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(frozen=True)
class Discrete:
    n: int

    def __post_init__(self) -> None:
        if int(self.n) <= 0:
            raise ValueError("Discrete.n must be > 0")

    def contains(self, value: object) -> bool:
        if isinstance(value, (np.integer, int)):
            return 0 <= int(value) < int(self.n)
        return False

    def sample(self) -> int:
        return int(np.random.randint(0, int(self.n)))


@dataclass(frozen=True)
class Box:
    shape: tuple[int, ...]
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("Box.shape must be non-empty")
        if float(self.low) >= float(self.high):
            raise ValueError("Box.low must be < Box.high")

    def contains(self, value: object) -> bool:
        if not isinstance(value, np.ndarray):
            return False
        if value.shape != self.shape:
            return False
        return bool(np.all(value >= self.low) and np.all(value <= self.high))

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self.shape).astype(np.float32)


Space: TypeAlias = Discrete | Box
