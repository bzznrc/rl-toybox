"""Osero game spec."""

from __future__ import annotations

from core.envs.spaces import Discrete
from core.game import GameSpec
from games.osero import config
from games.osero.env import OseroEnv


SPEC = GameSpec(
    game_id="osero",
    display_name="Osero",
    default_algo="search_play",
    make_env=lambda mode, render, level=None: OseroEnv(mode=mode, render=render, level=level),
    obs_dim=config.OBS_DIM,
    action_space=Discrete(config.ACT_DIM),
    run_name=f"b{int(config.BOARD_SIZE)}_{'_'.join(str(size) for size in config.POLICY_VALUE_HIDDEN_DIMENSIONS)}",
    family="search_play",
    role="Planning + self-play showcase.",
    summary="Compact AlphaZero-lite Osero with flattened board IO, MCTS, and a small shared policy/value MLP.",
    primary_algo_label="MCTS + policy/value net + self-play",
    implementation_stage="implemented",
    algo_config={
        "board_size": int(config.BOARD_SIZE),
        "hidden_sizes": list(config.POLICY_VALUE_HIDDEN_DIMENSIONS),
        "learning_rate": float(config.LEARNING_RATE),
        "weight_decay": float(config.WEIGHT_DECAY),
        "batch_size": int(config.BATCH_SIZE),
        "replay_size": int(config.REPLAY_BUFFER_SIZE),
        "min_replay_to_train": int(config.MIN_REPLAY_TO_TRAIN),
        "value_loss_weight": float(config.VALUE_LOSS_WEIGHT),
        "grad_clip_norm": float(config.GRAD_CLIP_NORM),
        "use_gpu": bool(config.USE_GPU),
        "simulations_per_move": int(config.SIMULATIONS_PER_MOVE),
        "c_puct": float(config.CPUCT),
        "dirichlet_alpha": float(config.DIRICHLET_ALPHA),
        "dirichlet_epsilon": float(config.DIRICHLET_EPSILON),
        "temperature_sample_moves": int(config.TEMPERATURE_SAMPLE_MOVES),
    },
    train_config={
        "max_games": int(config.MAX_GAMES),
        "train_after_games": int(config.TRAIN_AFTER_GAMES),
        "updates_per_game": int(config.UPDATES_PER_GAME),
        "checkpoint_every_games": int(config.CHECKPOINT_EVERY_GAMES),
    },
)
