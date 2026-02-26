"""SAC replay buffer stub."""

from __future__ import annotations


class SACReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)

    def __len__(self) -> int:
        return 0

    def add(self, transition) -> None:
        del transition

    def sample(self, batch_size: int):
        raise NotImplementedError("SAC replay sampling is not implemented yet.")
