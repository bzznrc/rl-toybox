"""Snake game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import (
    GameSpec,
    build_env_factory,
    build_exploration_config,
    build_hidden_run_name,
    build_off_policy_train_config,
)
from games.snake import config
from games.snake.env import SnakeEnv

SPEC = GameSpec(
    game_id="snake",
    default_algo="qlearn",
    make_env=build_env_factory(SnakeEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=build_hidden_run_name(config.HIDDEN_DIMENSIONS),
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "learning_rate": config.LEARNING_RATE,
        "gamma": config.GAMMA,
        "max_memory": config.MAX_MEMORY,
        "batch_size": config.BATCH_SIZE,
        "exploration": build_exploration_config(
            config.EPSILON_START,
            config.EPSILON_MIN,
            config.EPSILON_DECAY_STEPS,
            patience_episodes=config.EPS_BUMP_PATIENCE_EPISODES,
            min_improvement=config.EPS_BUMP_MIN_IMPROVEMENT,
            eps_bump_cap=config.EPS_BUMP_CAP,
            bump_cooldown_steps=config.EPS_BUMP_COOLDOWN_STEPS,
        ),
        "use_gpu": config.USE_GPU,
    },
    train_config=build_off_policy_train_config(
        max_steps=config.MAX_TRAINING_STEPS,
        checkpoint_every_steps=config.CHECKPOINT_EVERY_STEPS,
        reward_window=config.REWARD_ROLLING_WINDOW,
    ),
)
