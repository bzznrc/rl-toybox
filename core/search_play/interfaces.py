"""Small search/self-play interfaces used by scaffolded planning games."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyValueBatch:
    observations: np.ndarray
    policy_logits: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class MCTSConfig:
    simulations_per_move: int
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
