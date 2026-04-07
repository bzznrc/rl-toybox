"""Bang game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import (
    GameSpec,
    build_env_factory,
    build_exploration_config,
    build_hidden_run_name,
    build_off_policy_train_config,
)
from games.bang import config
from games.bang.env import BangEnv

SPEC = GameSpec(
    game_id="bang",
    display_name="Bang",
    default_algo="dqn",
    make_env=build_env_factory(BangEnv),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=build_hidden_run_name(config.HIDDEN_DIMENSIONS),
    family="value_discrete",
    role="Flagship discrete RL game.",
    summary="Arena shooter used to showcase the repo's advanced value-based stack.",
    primary_algo_label="Rainbow-lite DQN",
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "gamma": config.GAMMA,
        "batch_size": config.BATCH_SIZE,
        "replay_size": config.REPLAY_BUFFER_SIZE,
        "target_sync_every": config.TARGET_SYNC_EVERY,
        "grad_clip_norm": config.GRAD_CLIP_NORM,
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
        "dueling": True,
        "double_dqn": True,
        "prioritized_replay": True,
        "per_alpha": config.PER_ALPHA,
        "per_beta_start": config.PER_BETA_START,
        "per_beta_frames": config.PER_BETA_FRAMES,
        "per_epsilon": config.PER_EPSILON,
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
