"""Optional tiny helpers for multi-agent environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SharedPolicyTransition:
    agent_id: str
    obs: np.ndarray
    action: int | np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    info: dict[str, Any]


def flatten_agent_obs(obs_by_agent: dict[str, np.ndarray]) -> np.ndarray:
    """Stable concat helper for simple parameter-sharing setups."""
    if not obs_by_agent:
        return np.zeros(0, dtype=np.float32)
    stacked = [obs_by_agent[key].astype(np.float32, copy=False) for key in sorted(obs_by_agent.keys())]
    return np.concatenate(stacked, axis=0).astype(np.float32, copy=False)
