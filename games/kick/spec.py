"""Kick game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameCapabilities, GameSpec, build_env_factory
from games.kick import config
from games.kick.env import KickEnv


SPEC = GameSpec(
    game_id="kick",
    default_algo="ppo",
    make_env=build_env_factory(KickEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    capabilities=GameCapabilities(
        centralized_critic_required=True,
    ),
    device="cuda" if config.USE_GPU else "cpu",
    env_metadata={"central_obs_dim": int(config.CENTRAL_OBS_DIM)},
)
