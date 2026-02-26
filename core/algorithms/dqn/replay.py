"""Replay buffers for DQN variants."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any


Transition = tuple[Any, int, float, Any, bool]


class UniformReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: deque[Transition] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> tuple[list[Transition], list[int], list[float]]:
        batch = random.sample(self._buffer, int(batch_size))
        return batch, [], [1.0] * len(batch)

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        del indices, td_errors


class SumTree:
    """Binary sum tree for prioritized replay."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = [0.0] * (2 * self.capacity)
        self.data: list[Transition | None] = [None] * self.capacity
        self.write = 0
        self.size = 0

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def add(self, priority: float, data: Transition) -> int:
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float) -> None:
        change = float(priority) - float(self.tree[idx])
        self.tree[idx] = float(priority)
        while idx > 1:
            idx //= 2
            self.tree[idx] += change

    def get(self, value: float) -> tuple[int, float, Transition]:
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        item = self.data[data_idx]
        if item is None:
            raise RuntimeError("Prioritized replay sampled an empty slot.")
        return idx, float(self.tree[idx]), item


@dataclass
class PrioritizedReplayConfig:
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 1_000_000
    epsilon: float = 1e-4


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, config: PrioritizedReplayConfig):
        self.tree = SumTree(capacity)
        self.config = config
        self.frame = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def add(self, transition: Transition) -> None:
        priority = self.max_priority ** self.config.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> tuple[list[Transition], list[int], list[float]]:
        batch: list[Transition] = []
        indices: list[int] = []
        priorities: list[float] = []

        if self.tree.total <= 0:
            raise RuntimeError("Cannot sample from empty prioritized replay tree.")

        self.frame += 1
        segment = self.tree.total / float(batch_size)
        beta = min(
            1.0,
            self.config.beta_start
            + self.frame * (1.0 - self.config.beta_start) / max(1, int(self.config.beta_frames)),
        )

        for i in range(int(batch_size)):
            start = segment * i
            end = segment * (i + 1)
            value = random.uniform(start, end)
            idx, priority, transition = self.tree.get(value)
            indices.append(idx)
            priorities.append(priority)
            batch.append(transition)

        probabilities = [priority / self.tree.total for priority in priorities]
        weights = [(self.tree.size * probability) ** (-beta) for probability in probabilities]
        max_weight = max(weights) if weights else 1.0
        weights = [weight / max_weight for weight in weights]
        return batch, indices, weights

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.config.epsilon) ** self.config.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)
