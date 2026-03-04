"""Kick game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from games.kick import config
from games.kick.env import KickEnv
from games.spec_types import GameSpec


def make_env(mode: str, render: bool, level: int | None = None):
    return KickEnv(mode=mode, render=render, level=level)


RUN_NAME = "_".join(str(size) for size in config.HIDDEN_DIMENSIONS)


SPEC = GameSpec(
    game_id="kick",
    default_algo="ppo",
    make_env=make_env,
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=RUN_NAME,
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "learning_rate": config.LEARNING_RATE,
        "gamma": config.GAMMA,
        "gae_lambda": config.GAE_LAMBDA,
        "clip_ratio": config.CLIP_RATIO,
        "update_epochs": config.UPDATE_EPOCHS,
        "minibatch_size": config.MINIBATCH_SIZE,
        "entropy_coef": config.ENTROPY_COEF,
        "value_coef": config.VALUE_COEF,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "use_gpu": config.USE_GPU,
    },
    train_config={
        "max_iterations": config.MAX_TRAINING_ITERATIONS,
        "rollout_steps": config.ROLLOUT_STEPS,
        "checkpoint_every_iterations": config.CHECKPOINT_EVERY_ITERATIONS,
        "reward_window": config.REWARD_ROLLING_WINDOW,
        "min_episodes_for_stats": config.MIN_EPISODES_FOR_STATS,
    },
)
