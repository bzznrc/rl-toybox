"""Peek game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import (
    GameSpec,
    build_env_factory,
    build_on_policy_train_config,
    build_recurrent_run_name,
)
from games.peek import config
from games.peek.env import PeekEnv


SPEC = GameSpec(
    game_id="peek",
    default_algo="ppo",
    make_env=build_env_factory(PeekEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=build_recurrent_run_name(
        config.ENCODER_HIDDEN_DIMENSIONS,
        recurrent_type=config.RECURRENT_TYPE,
        recurrent_hidden_size=config.RECURRENT_HIDDEN_SIZE,
        actor_head_hidden_sizes=config.ACTOR_HEAD_HIDDEN_DIMENSIONS,
        critic_head_hidden_sizes=config.CRITIC_HEAD_HIDDEN_DIMENSIONS,
    ),
    algo_config={
        "hidden_sizes": list(config.ENCODER_HIDDEN_DIMENSIONS),
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
        "recurrent_type": str(config.RECURRENT_TYPE),
        "recurrent_hidden_size": int(config.RECURRENT_HIDDEN_SIZE),
        "actor_head_hidden_sizes": list(config.ACTOR_HEAD_HIDDEN_DIMENSIONS),
        "critic_head_hidden_sizes": list(config.CRITIC_HEAD_HIDDEN_DIMENSIONS),
        "recurrent_seq_len": int(config.RECURRENT_SEQ_LEN),
    },
    train_config=build_on_policy_train_config(
        max_iterations=config.MAX_TRAINING_ITERATIONS,
        rollout_steps=config.ROLLOUT_STEPS,
        checkpoint_every_iterations=config.CHECKPOINT_EVERY_ITERATIONS,
        reward_window=config.REWARD_ROLLING_WINDOW,
        min_episodes_for_stats=config.MIN_EPISODES_FOR_STATS,
    ),
)
