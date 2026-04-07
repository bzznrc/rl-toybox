"""Othello scaffold spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import (
    GameSpec,
    build_hidden_run_name,
    build_off_policy_train_config,
    build_scaffold_env_factory,
)
from games.othello import config


SPEC = GameSpec(
    game_id="othello",
    display_name="Othello",
    default_algo="search_play",
    make_env=build_scaffold_env_factory(
        game_id="othello",
        obs_dim=config.OBS_DIM,
        note="Scaffold entry for the future AlphaZero-lite self-play implementation.",
    ),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=build_hidden_run_name(config.POLICY_VALUE_HIDDEN_DIMENSIONS),
    family="search_play",
    role="AlphaZero-lite capstone with planning and self-play.",
    summary="Intentional outlier scaffold for policy/value training plus MCTS.",
    primary_algo_label="MCTS + policy/value net + self-play",
    implementation_stage="scaffold",
    algo_config={},
    train_config=build_off_policy_train_config(
        max_steps=1_000,
        checkpoint_every_steps=250,
        reward_window=10,
        min_episodes_for_stats=0,
    ),
)
