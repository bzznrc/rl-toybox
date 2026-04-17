"""Frogger game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.frogger import config
from games.frogger.env import FroggerEnv


SPEC = GameSpec(
    game_id="frogger",
    default_algo="recurrent_ppo",
    make_env=build_env_factory(FroggerEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(
        recurrent_friendly=True,
    ),
    device="cuda" if config.USE_GPU else "cpu",
)
