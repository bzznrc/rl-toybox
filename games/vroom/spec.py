"""Vroom game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from games.vroom import config
from games.spec_types import GameSpec
from games.vroom.env import VroomEnv


def make_env(mode: str, render: bool):
    return VroomEnv(mode=mode, render=render)


HIDDEN_DIMENSIONS = [48, 48]
RUN_NAME = "_".join(str(size) for size in HIDDEN_DIMENSIONS)


SPEC = GameSpec(
    game_id="vroom",
    default_algo="dqn",
    make_env=make_env,
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=RUN_NAME,
    algo_config={
        "hidden_sizes": list(HIDDEN_DIMENSIONS),
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_size": 100_000,
        "target_sync_every": 500,
        "learn_start_steps": 1_000,
        "train_every_steps": 1,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay_steps": 100_000,
        "dueling": False,
        "double_dqn": False,
        "prioritized_replay": False,
    },
    train_config={
        "max_steps": 300_000,
        "train_after_steps": 1_000,
        "update_every_steps": 1,
        "updates_per_step": 1,
        "checkpoint_every_steps": 25_000,
        "reward_window": 100,
    },
)
