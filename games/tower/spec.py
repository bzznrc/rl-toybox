"""Tower game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import (
    GameSpec,
    build_env_factory,
    build_exploration_config,
    build_hidden_run_name,
    build_off_policy_train_config,
)
from games.tower import config
from games.tower.env import TowerEnv


SPEC = GameSpec(
    game_id="tower",
    display_name="Tower",
    default_algo="dqn",
    make_env=build_env_factory(TowerEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=build_hidden_run_name(config.HIDDEN_DIMENSIONS),
    family="value_discrete",
    role="Delayed reward + combinatorial decisions + action masking showcase.",
    summary="Tiny wave-based tower defense with Fast, Heavy, and Area towers for masked DQN.",
    primary_algo_label="Masked Double DQN",
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
        "exploration": build_exploration_config(
            config.EPSILON_START,
            config.EPSILON_MIN,
            config.EPSILON_DECAY_STEPS,
            patience_episodes=config.EPS_BUMP_PATIENCE_EPISODES,
            min_improvement=config.EPS_BUMP_MIN_IMPROVEMENT,
            eps_bump_cap=config.EPS_BUMP_CAP,
            bump_cooldown_steps=config.EPS_BUMP_COOLDOWN_STEPS,
        ),
        "dueling": True,
        "double_dqn": True,
        "prioritized_replay": False,
    },
    train_config=build_off_policy_train_config(
        max_steps=config.TOTAL_TRAINING_STEPS,
        train_after_steps=config.LEARN_START_STEPS,
        update_every_steps=config.TRAIN_EVERY_STEPS,
        updates_per_step=config.UPDATES_PER_TRAIN,
        checkpoint_every_steps=config.CHECKPOINT_EVERY_STEPS,
        reward_window=config.REWARD_ROLLING_WINDOW,
    ),
)
