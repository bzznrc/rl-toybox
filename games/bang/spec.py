"""Bang game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from games.bang import config
from games.bang.env import BangEnv
from games.off_policy_defaults import OFF_POLICY_TRAIN_DEFAULTS, make_exploration_config
from games.spec_types import GameSpec


def make_env(mode: str, render: bool):
    return BangEnv(mode=mode, render=render)


RUN_NAME = config.MODEL_SUBDIR

SPEC = GameSpec(
    game_id="bang",
    default_algo="dqn",
    make_env=make_env,
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=RUN_NAME,
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "gamma": config.GAMMA,
        "batch_size": config.BATCH_SIZE,
        "replay_size": config.REPLAY_BUFFER_SIZE,
        "target_sync_every": config.TARGET_SYNC_EVERY,
        "grad_clip_norm": config.GRAD_CLIP_NORM,
        "learn_start_steps": config.LEARN_START_STEPS,
        "train_every_steps": config.TRAIN_EVERY_STEPS,
        "exploration": make_exploration_config(config.EPSILON_DECAY),
        "use_gpu": config.USE_GPU,
        "dueling": True,
        "double_dqn": True,
        "prioritized_replay": True,
        "per_alpha": config.PER_ALPHA,
        "per_beta_start": config.PER_BETA_START,
        "per_beta_frames": config.PER_BETA_FRAMES,
        "per_epsilon": config.PER_EPSILON,
    },
    train_config={
        **OFF_POLICY_TRAIN_DEFAULTS,
        "max_steps": config.TOTAL_TRAINING_STEPS,
        "train_after_steps": config.LEARN_START_STEPS,
        "update_every_steps": config.TRAIN_EVERY_STEPS,
        "updates_per_step": config.GRADIENT_STEPS_PER_UPDATE,
        "checkpoint_every_steps": config.CHECKPOINT_EVERY_STEPS,
    },
)
