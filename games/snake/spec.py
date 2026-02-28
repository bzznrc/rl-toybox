"""Snake game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from games.off_policy_defaults import OFF_POLICY_TRAIN_DEFAULTS, make_exploration_config
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
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=RUN_NAME,
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "learning_rate": config.LEARNING_RATE,
        "gamma": config.GAMMA,
        "max_memory": config.MAX_MEMORY,
        "batch_size": config.BATCH_SIZE,
        "exploration": make_exploration_config(
            config.EPSILON_START,
            config.EPSILON_MIN,
            config.EPSILON_DECAY_STEPS,
            patience_episodes=config.EPS_BUMP_PATIENCE_EPISODES,
            min_improvement=config.EPS_BUMP_MIN_IMPROVEMENT,
            eps_bump_cap=config.EPS_BUMP_CAP,
            bump_cooldown_steps=config.EPS_BUMP_COOLDOWN_STEPS,
        ),
        "use_gpu": False,
    },
    train_config={
        **OFF_POLICY_TRAIN_DEFAULTS,
        "max_steps": config.MAX_TRAINING_STEPS,
        "checkpoint_every_steps": config.CHECKPOINT_EVERY_STEPS,
    },
)
