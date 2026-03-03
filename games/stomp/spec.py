"""Stomp game spec (SAC stub)."""

from __future__ import annotations

from core.envs.spaces import Box
from games.spec_types import GameSpec
from games.stomp.env import StompEnv


def make_env(mode: str, render: bool, level: int | None = None):
    del level
    return StompEnv(mode=mode, render=render)


HIDDEN_DIMENSIONS = [128, 128]  # Future SAC nets (not implemented yet).
RUN_NAME = "_".join(str(size) for size in HIDDEN_DIMENSIONS)


SPEC = GameSpec(
    game_id="stomp",
    default_algo="sac",
    make_env=make_env,
    obs_dim=6,
    action_space=Box(shape=(2,), low=-1.0, high=1.0),
    run_name=RUN_NAME,
    algo_config={
        "hidden_sizes": list(HIDDEN_DIMENSIONS),
        "action_low": -1.0,
        "action_high": 1.0,
    },
    train_config={
        "max_steps": 100_000,
    },
)
