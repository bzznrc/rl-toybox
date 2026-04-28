"""Declarative configuration for Four."""

from __future__ import annotations

from core.io_schema import row_major_grid_feature_names
from core.utils import env_flag


# RUNTIME
GAME_ID = "four"
WINDOW_TITLE = "Four"
USE_GPU = env_flag("FOUR_USE_GPU", False)
AI_STEP_DELAY_SECONDS = 0.20


# ENV
BOARD_ROWS = 6
BOARD_COLS = 7
CONNECT_N = 4
BOARD_TOP_MARGIN = 42.0
BOARD_SIDE_MARGIN = 132.0
BOARD_BOTTOM_MARGIN = 56.0
BOARD_FRAME_PADDING = 8.0
CELL_INSET_RATIO = 0.10
STONE_INSET_RATIO = 0.16
LEGAL_HINT_RATIO = 0.22
HOVER_OUTLINE_WIDTH = 3.0


# IO
INPUT_FEATURE_NAMES = row_major_grid_feature_names(BOARD_ROWS, BOARD_COLS, prefix="board")
ACTION_NAMES = [f"drop_c{col}" for col in range(BOARD_COLS)]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
if OBS_DIM != 42:
    raise RuntimeError(f"Four INPUT_FEATURE_NAMES expected 42 entries, got {OBS_DIM}.")
if ACT_DIM != 7:
    raise RuntimeError(f"Four ACTION_NAMES expected 7 entries, got {ACT_DIM}.")


# GAME
DEFAULT_ALGO = "search_play"
GAME_CAPABILITIES = {
    "masked_actions": True,
    "self_play": True,
}
ENV_METADATA = {
    "board_rows": int(BOARD_ROWS),
    "board_cols": int(BOARD_COLS),
    "connect_n": int(CONNECT_N),
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSS = -5.0
REWARD_DRAW = 0.0
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_loss": PENALTY_LOSS,
    "outcome.reward_draw": REWARD_DRAW,
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [48, 48],
}
ALGO_CONFIG_OVERRIDES = {
    "search_play": {
        "board_rows": int(BOARD_ROWS),
        "board_cols": int(BOARD_COLS),
        "action_dim": int(ACT_DIM),
        "simulations_per_move": 48,
        "dirichlet_alpha": 0.35,
    }
}
DEFAULT_TRAIN_CONFIG = {
    "budget": 5_000,
    "train_after_games": 8,
    "updates_per_game": 3,
    "checkpoint_every": 25,
    "arena_every_games": 25,
    "arena_games_per_opponent": 3,
}
