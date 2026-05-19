"""Algorithm interface and small shared helpers used by runners and agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch.nn as nn


ActivationFactory = type[nn.Module] | Callable[[], nn.Module]


def build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    *,
    activation: ActivationFactory = nn.ReLU,
    output_dim: int | None = None,
) -> tuple[nn.Sequential, int]:
    """Build a simple feed-forward trunk and return its output feature size."""

    layers: list[nn.Module] = []
    in_features = int(input_dim)
    for hidden_size in hidden_sizes:
        hidden_features = int(hidden_size)
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(activation())
        in_features = hidden_features
    if output_dim is not None:
        layers.append(nn.Linear(in_features, int(output_dim)))
        in_features = int(output_dim)
    return nn.Sequential(*layers), int(in_features)


def normalize_action_mask(
    action_mask: object | None,
    action_dim: int,
    *,
    fallback_mask: object | None = None,
    empty_is_all_valid: bool = True,
) -> np.ndarray:
    """Return a valid one-dimensional boolean action mask."""

    dim = int(action_dim)
    if dim <= 0:
        raise ValueError("action_dim must be positive.")

    source = fallback_mask if action_mask is None else action_mask
    if source is None:
        return np.ones((dim,), dtype=np.bool_)

    mask = np.asarray(source, dtype=np.bool_).reshape(-1)
    if int(mask.size) != dim:
        raise ValueError(f"Action mask expected {dim} values, got {int(mask.size)}.")
    if int(mask.sum()) <= 0:
        if fallback_mask is not None and source is not fallback_mask:
            return normalize_action_mask(
                fallback_mask,
                dim,
                empty_is_all_valid=bool(empty_is_all_valid),
            )
        if bool(empty_is_all_valid):
            return np.ones((dim,), dtype=np.bool_)
    return mask.astype(np.bool_, copy=False)


def normalize_action_mask_batch(
    action_mask: object | None,
    action_dim: int,
    *,
    batch_size: int,
    allow_scalar_broadcast: bool = True,
) -> np.ndarray:
    """Return a valid two-dimensional boolean action-mask batch."""

    rows = int(batch_size)
    dim = int(action_dim)
    if rows < 0:
        raise ValueError("batch_size must not be negative.")
    if action_mask is None:
        return np.ones((rows, dim), dtype=np.bool_)

    mask_array = np.asarray(action_mask, dtype=np.bool_)
    if mask_array.ndim == 1:
        if bool(allow_scalar_broadcast) and int(mask_array.size) == 1 and rows > 1:
            mask_array = np.full((rows, dim), bool(mask_array.item()), dtype=np.bool_)
        else:
            mask_array = mask_array.reshape(1, -1)
    if mask_array.ndim != 2:
        raise ValueError(f"Action mask expected ndim 1 or 2, got {mask_array.ndim}.")

    if int(mask_array.shape[0]) == 1 and rows > 1:
        mask_array = np.repeat(mask_array, rows, axis=0)
    if int(mask_array.shape[0]) != rows:
        raise ValueError(f"Action mask expected batch size {rows}, got {int(mask_array.shape[0])}.")
    if int(mask_array.shape[1]) != dim:
        raise ValueError(f"Action mask expected action dim {dim}, got {int(mask_array.shape[1])}.")

    valid_counts = mask_array.sum(axis=1)
    if np.any(valid_counts <= 0):
        mask_array = mask_array.copy()
        mask_array[valid_counts <= 0, :] = True
    return mask_array.astype(np.bool_, copy=False)


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

    def reset_policy_state(self) -> None:
        """Optional hook for clearing recurrent policy state between episodes."""

    def on_episode_end(self, avg_reward: float) -> dict[str, float | int | str] | None:
        """Optional hook for episode-level bookkeeping in runners."""
        del avg_reward
        return None

    def exploration_avg_window(self) -> int | None:
        """Optional rolling episode window request for exploration control."""
        return None
