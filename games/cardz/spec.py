"""Cardz game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.cardz import config
from games.cardz.env import CardzEnv


SPEC = GameSpec(
    game_id="cardz",
    default_algo="a2c",
    make_env=build_env_factory(CardzEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(
        masked_actions=True,
        recurrent_friendly=True,
    ),
    device="cuda" if config.USE_GPU else "cpu",
)
