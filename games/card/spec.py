"""Card scaffold spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import (
    GameSpec,
    build_hidden_run_name,
    build_on_policy_train_config,
    build_scaffold_env_factory,
)
from games.card import config


SPEC = GameSpec(
    game_id="card",
    display_name="Card",
    default_algo="a2c",
    make_env=build_scaffold_env_factory(
        game_id="card",
        obs_dim=config.OBS_DIM,
        note="Scaffold entry for the future stochastic hidden-information card game.",
    ),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=build_hidden_run_name(config.HIDDEN_DIMENSIONS),
    family="actor_critic",
    role="Simple stochastic hidden-information actor-critic game.",
    summary="New scaffold for an A2C-style compact card game.",
    primary_algo_label="A2C",
    implementation_stage="scaffold",
    algo_config={
        "hidden_sizes": list(config.HIDDEN_DIMENSIONS),
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
    },
    train_config=build_on_policy_train_config(
        max_iterations=config.MAX_TRAINING_ITERATIONS,
        rollout_steps=config.ROLLOUT_STEPS,
        checkpoint_every_iterations=config.CHECKPOINT_EVERY_ITERATIONS,
        reward_window=config.REWARD_ROLLING_WINDOW,
        min_episodes_for_stats=config.MIN_EPISODES_FOR_STATS,
    ),
)
