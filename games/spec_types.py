"""Shared GameSpec dataclass for registry entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.envs.base import Env
from core.envs.spaces import Space


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    default_algo: str
    make_env: Callable[[str, bool], Env]
    obs_dim: int
    action_space: Space
    run_name: str
    train_config: dict[str, object] = field(default_factory=dict)
    algo_config: dict[str, object] = field(default_factory=dict)
