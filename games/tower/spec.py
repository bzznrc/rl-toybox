"""Tower game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.tower import config
from games.tower.env import TowerEnv


SPEC = GameSpec(
    game_id="tower",
    default_algo="dqn",
    make_env=build_env_factory(TowerEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(masked_actions=True),
    device="cuda" if config.USE_GPU else "cpu",
)
