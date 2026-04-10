"""Cardz game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import (
    GameSpec,
    build_env_factory,
    build_hidden_run_name,
    build_on_policy_train_config,
)
from games.cardz import config
from games.cardz.env import CardzEnv


SPEC = GameSpec(
    game_id="cardz",
    display_name="Cardz",
    default_algo="a2c",
    make_env=build_env_factory(CardzEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=build_hidden_run_name(config.HIDDEN_DIMENSIONS),
    family="actor_critic",
    role="Simple stochastic hidden-information actor-critic showcase.",
    summary="Tiny 2-player 3-lane card duel with AB/BA turn order, ATK/BAN lane play, masked actions, and A2C-oriented defaults.",
    primary_algo_label="A2C",
    implementation_stage="implemented",
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "share_backbone": bool(config.SHARE_BACKBONE),
        "learning_rate": config.LEARNING_RATE,
        "gamma": config.GAMMA,
        "gae_lambda": config.GAE_LAMBDA,
        "clip_ratio": config.CLIP_RATIO,
        "update_epochs": config.UPDATE_EPOCHS,
        "minibatch_size": config.MINIBATCH_SIZE,
        "entropy_coef": float(config.LEVEL_SETTINGS[int(config.MIN_LEVEL)]["entropy_coef"]),
        "value_coef": config.VALUE_COEF,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "use_gpu": config.USE_GPU,
    },
    train_config=build_on_policy_train_config(
        max_iterations=config.MAX_TRAINING_ITERATIONS,
        rollout_steps=config.ROLLOUT_STEPS,
        checkpoint_every_iterations=config.CHECKPOINT_EVERY_ITERATIONS,
        reward_window=config.REWARD_ROLLING_WINDOW,
        min_episodes_for_stats=config.MIN_EPISODES_FOR_STATS,
    ),
)
