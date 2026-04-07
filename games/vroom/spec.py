"""Vroom game spec."""

from __future__ import annotations

import numpy as np

from core.envs.spaces import Box
from core.game import (
    GameSpec,
    build_env_factory,
    build_hidden_run_name,
    build_off_policy_train_config,
)
from games.vroom import config
from games.vroom.env import VroomEnv


SPEC = GameSpec(
    game_id="vroom",
    display_name="Vroom",
    default_algo="sac",
    make_env=build_env_factory(VroomEnv),
    obs_dim=config.OBS_DIM,
    action_space=Box(
        shape=(config.ACT_DIM,),
        low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    ),
    run_name=build_hidden_run_name(config.HIDDEN_DIMENSIONS),
    family="actor_critic",
    role="Continuous-control showcase.",
    summary="Top-down one-lap racer with continuous steer/throttle/brake control and compact vector observations.",
    primary_algo_label="SAC",
    implementation_stage="implemented",
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "learning_rate": config.LEARNING_RATE,
        "gamma": config.GAMMA,
        "batch_size": config.BATCH_SIZE,
        "replay_size": config.REPLAY_BUFFER_SIZE,
        "tau": config.TAU,
        "grad_clip_norm": config.GRAD_CLIP_NORM,
        "init_alpha": config.INIT_ALPHA,
        "use_gpu": config.USE_GPU,
    },
    train_config=build_off_policy_train_config(
        max_steps=config.MAX_TRAINING_STEPS,
        train_after_steps=config.LEARN_START_STEPS,
        update_every_steps=config.TRAIN_EVERY_STEPS,
        updates_per_step=config.UPDATES_PER_TRAIN,
        checkpoint_every_steps=config.CHECKPOINT_EVERY_STEPS,
        reward_window=config.REWARD_ROLLING_WINDOW,
    ),
)
