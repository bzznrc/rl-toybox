"""Declarative configuration for Flip."""

from __future__ import annotations

from core.io_schema import row_major_grid_action_names, row_major_grid_feature_names
from core.shared_config import CELL_INSET
from core.utils import env_flag


# RUNTIME
GAME_ID = "flip"
WINDOW_TITLE = "Flip"
USE_GPU = env_flag("FLIP_USE_GPU", False)
AI_STEP_DELAY_SECONDS = 0.20


# ENV
BOARD_ROWS = 6
BOARD_COLS = 6
BOARD_TOP_MARGIN = 42.0
BOARD_BOTTOM_MARGIN = 56.0
BOARD_CELL_TILES = 3
PIECE_TILES = 2
LEGAL_HINT_TILES = 1
BOARD_FRAME_PADDING = float(CELL_INSET * BOARD_CELL_TILES)


# IO
INPUT_FEATURE_NAMES = row_major_grid_feature_names(BOARD_ROWS, BOARD_COLS, prefix="board")
ACTION_NAMES = row_major_grid_action_names(BOARD_ROWS, BOARD_COLS, prefix="place")
OBS_DIM = 36
ACT_DIM = 36
assert len(INPUT_FEATURE_NAMES) == 36
assert len(ACTION_NAMES) == 36
assert OBS_DIM == 36
assert ACT_DIM == 36


# GAME
DEFAULT_ALGO = "search_play"
GAME_CAPABILITIES = {
    "masked_actions": True,
    "self_play": True,
}
ENV_METADATA = {
    "board_rows": int(BOARD_ROWS),
    "board_cols": int(BOARD_COLS),
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
