"""Shared CTDE data containers for multi-agent experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentralizedCriticBatch:
    actor_obs: np.ndarray
    central_obs: np.ndarray
    action_mask: np.ndarray | None = None
