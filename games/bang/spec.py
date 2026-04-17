"""Bang game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.bang import config
from games.bang.env import BangEnv

SPEC = GameSpec(
    game_id="bang",
    default_algo="dqn",
    make_env=build_env_factory(BangEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(),
    device="cuda" if config.USE_GPU else "cpu",
)
