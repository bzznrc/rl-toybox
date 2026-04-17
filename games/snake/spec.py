"""Snake game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.snake import config
from games.snake.env import SnakeEnv

SPEC = GameSpec(
    game_id="snake",
    default_algo="qlearn",
    make_env=build_env_factory(SnakeEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(),
    device="cuda" if config.USE_GPU else "cpu",
)
