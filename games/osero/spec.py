"""Osero game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec
from games.osero import config
from games.osero.env import OseroEnv


SPEC = GameSpec(
    game_id="osero",
    default_algo="search_play",
    make_env=lambda mode, render, level=None: OseroEnv(mode=mode, render=render, level=level),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(
        masked_actions=True,
        self_play=True,
    ),
    device="cuda" if config.USE_GPU else "cpu",
    env_metadata={"board_size": int(config.BOARD_SIZE)},
)
