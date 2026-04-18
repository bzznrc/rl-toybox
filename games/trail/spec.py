"""Trail game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.trail import config
from games.trail.env import TrailEnv


SPEC = GameSpec(
    game_id="trail",
    default_algo="ppo",
    make_env=build_env_factory(TrailEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(),
    device="cuda" if config.USE_GPU else "cpu",
)
