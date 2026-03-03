"""Vroom game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from games.off_policy_defaults import OFF_POLICY_TRAIN_DEFAULTS, make_exploration_config
from games.vroom import config
from games.spec_types import GameSpec
from games.vroom.env import VroomEnv


def make_env(mode: str, render: bool, level: int | None = None):
    return VroomEnv(mode=mode, render=render, level=level)

RUN_NAME = "_".join(str(size) for size in config.HIDDEN_DIMENSIONS)


SPEC = GameSpec(
    game_id="vroom",
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
        "use_gpu": config.USE_GPU,
        "exploration": make_exploration_config(
            config.EPSILON_START,
            config.EPSILON_MIN,
            config.EPSILON_DECAY_STEPS,
            patience_episodes=config.EPS_BUMP_PATIENCE_EPISODES,
            min_improvement=config.EPS_BUMP_MIN_IMPROVEMENT,
            eps_bump_cap=config.EPS_BUMP_CAP,
            bump_cooldown_steps=config.EPS_BUMP_COOLDOWN_STEPS,
        ),
        "dueling": False,
        "double_dqn": False,
        "prioritized_replay": False,
    },
    train_config={
        **OFF_POLICY_TRAIN_DEFAULTS,
        "max_steps": config.MAX_TRAINING_STEPS,
        "train_after_steps": config.LEARN_START_STEPS,
        "update_every_steps": config.TRAIN_EVERY_STEPS,
        "updates_per_step": config.UPDATES_PER_TRAIN,
        "checkpoint_every_steps": config.CHECKPOINT_EVERY_STEPS,
        "reward_window": config.REWARD_ROLLING_WINDOW,
        "min_episodes_for_stats": config.REWARD_ROLLING_WINDOW,
    },
)
