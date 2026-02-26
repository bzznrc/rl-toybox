"""Snake game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from games.snake import config
from games.snake.env import SnakeEnv
from games.spec_types import GameSpec


def make_env(mode: str, render: bool):
    return SnakeEnv(mode=mode, render=render)


RUN_NAME = config.MODEL_SUBDIR

SPEC = GameSpec(
    game_id="snake",
    default_algo="qlearn",
    make_env=make_env,
    obs_dim=config.NUM_INPUT_FEATURES,
    action_space=Discrete(config.NUM_ACTIONS),
    run_name=RUN_NAME,
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "learning_rate": config.LEARNING_RATE,
        "gamma": config.GAMMA,
        "max_memory": config.MAX_MEMORY,
        "batch_size": config.BATCH_SIZE,
        "epsilon_start": config.EPSILON_START,
        "epsilon_decay": config.EPSILON_DECAY,
        "epsilon_end": config.EPSILON_END,
        "use_gpu": False,
    },
    train_config={
        "max_steps": 1_000_000,
        "train_after_steps": 0,
        "update_every_steps": 1,
        "updates_per_step": 1,
        "checkpoint_every_steps": 50_000,
        "reward_window": config.AVG100_WINDOW,
    },
)
