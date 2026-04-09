"""Small search/self-play interfaces used by Osero."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyValueBatch:
    observations: np.ndarray
    policy_targets: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class MCTSConfig:
    simulations_per_move: int
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float


@dataclass(frozen=True)
class ReplaySample:
    observation: np.ndarray
    policy_target: np.ndarray
    value_target: float


@dataclass(frozen=True)
class SearchPlayTrainConfig:
    max_games: int
    train_after_games: int
    updates_per_game: int
    checkpoint_every_games: int
