"""Vroom game spec."""

from __future__ import annotations

import numpy as np

from core.envs.spaces import Box
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.vroom import config
from games.vroom.env import VroomEnv


SPEC = GameSpec(
    game_id="vroom",
    default_algo="sac",
    make_env=build_env_factory(VroomEnv),
    obs_dim=config.OBS_DIM,
    action_space=Box(
        shape=(config.ACT_DIM,),
        low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    ),
    capabilities=GameCapabilities(),
    device="cuda" if config.USE_GPU else "cpu",
)
