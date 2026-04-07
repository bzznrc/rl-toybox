"""Small action/observation space helpers (gym-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    low: float | list[float] | tuple[float, ...] | np.ndarray
    high: float | list[float] | tuple[float, ...] | np.ndarray
    low_array: np.ndarray = field(init=False, repr=False)
    high_array: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("Box.shape must be non-empty")
        shape = tuple(int(axis) for axis in self.shape)
        object.__setattr__(self, "shape", shape)
        low_array = np.asarray(self.low, dtype=np.float32)
        high_array = np.asarray(self.high, dtype=np.float32)
        if low_array.ndim == 0:
            low_array = np.full(shape, float(low_array.item()), dtype=np.float32)
        else:
            low_array = np.broadcast_to(low_array, shape).astype(np.float32, copy=False)
        if high_array.ndim == 0:
            high_array = np.full(shape, float(high_array.item()), dtype=np.float32)
        else:
            high_array = np.broadcast_to(high_array, shape).astype(np.float32, copy=False)
        if np.any(low_array >= high_array):
            raise ValueError("Box.low must be < Box.high for every axis.")
        object.__setattr__(self, "low_array", low_array.copy())
        object.__setattr__(self, "high_array", high_array.copy())

    def contains(self, value: object) -> bool:
        try:
            value_array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return False
        if value_array.shape != self.shape:
            return False
        return bool(np.all(value_array >= self.low_array) and np.all(value_array <= self.high_array))

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low_array, self.high_array, size=self.shape).astype(np.float32)


Space: TypeAlias = Discrete | Box
