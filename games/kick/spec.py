"""Kick game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from games.kick import config
from games.kick.env import KickEnv
from games.spec_types import GameSpec


def make_env(mode: str, render: bool):
    return KickEnv(mode=mode, render=render)


HIDDEN_DIMENSIONS = [96, 96]
RUN_NAME = "_".join(str(size) for size in HIDDEN_DIMENSIONS)


SPEC = GameSpec(
    game_id="kick",
    default_algo="ppo",
    make_env=make_env,
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=RUN_NAME,
    algo_config={
        "hidden_sizes": list(HIDDEN_DIMENSIONS),
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "update_epochs": 4,
        "minibatch_size": 256,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    train_config={
        "max_iterations": 500,
        "rollout_steps": 1024,
        "checkpoint_every_iterations": 10,
        "reward_window": 100,
    },
)
