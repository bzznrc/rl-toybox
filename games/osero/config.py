"""Central configuration for Osero."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.io_schema import row_major_grid_action_names, row_major_grid_feature_names
from core.utils import env_flag, env_int


SUPPORTED_BOARD_SIZES = (6, 8)
DEFAULT_BOARD_SIZE = 6


def _resolve_board_size(raw_value: int) -> int:
    try:
        board_size = int(raw_value)
    except (TypeError, ValueError):
        return int(DEFAULT_BOARD_SIZE)
    if board_size not in SUPPORTED_BOARD_SIZES:
        return int(DEFAULT_BOARD_SIZE)
    return int(board_size)


# RUNTIME
WINDOW_TITLE = "Osero"
FPS = 18
TRAINING_FPS = 0
USE_GPU = env_flag("OSERO_USE_GPU", env_flag("OTHELLO_USE_GPU", False))
BOARD_SIZE = _resolve_board_size(env_int("OSERO_BOARD_SIZE", env_int("OTHELLO_BOARD_SIZE", DEFAULT_BOARD_SIZE)))


# ENV
SCREEN_WIDTH = screen_width(DEFAULT_GRID_COLUMNS, DEFAULT_TILE_SIZE)
SCREEN_HEIGHT = screen_height(DEFAULT_GRID_ROWS, DEFAULT_TILE_SIZE, BB_HEIGHT)
WORLD_WIDTH = int(SCREEN_WIDTH)
WORLD_HEIGHT = int(SCREEN_HEIGHT - BB_HEIGHT)
BOARD_TOP_MARGIN = 50.0
BOARD_SIDE_MARGIN = 150.0
BOARD_BOTTOM_MARGIN = 56.0
BOARD_FRAME_PADDING = 8.0
STONE_INSET_RATIO = 0.18
LEGAL_HINT_RATIO = 0.24
HOVER_OUTLINE_WIDTH = 3.0


# IO
INPUT_FEATURE_NAMES_BY_SIZE = {
    6: row_major_grid_feature_names(6),
    8: row_major_grid_feature_names(8),
}
ACTION_NAMES_BY_SIZE = {
    6: row_major_grid_action_names(6, include_pass=True),
    8: row_major_grid_action_names(8, include_pass=True),
}
INPUT_FEATURE_NAMES = list(INPUT_FEATURE_NAMES_BY_SIZE[int(BOARD_SIZE)])
ACTION_NAMES = list(ACTION_NAMES_BY_SIZE[int(BOARD_SIZE)])
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# MODEL / SEARCH
POLICY_VALUE_HIDDEN_DIMENSIONS_BY_SIZE = {
    6: (128, 128),
    8: (128, 128, 128),
}
POLICY_VALUE_HIDDEN_DIMENSIONS = tuple(POLICY_VALUE_HIDDEN_DIMENSIONS_BY_SIZE[int(BOARD_SIZE)])
SIMULATIONS_PER_MOVE_BY_SIZE = {
    6: 48,
    8: 64,
}
SIMULATIONS_PER_MOVE = int(SIMULATIONS_PER_MOVE_BY_SIZE[int(BOARD_SIZE)])
CPUCT = 1.25
DIRICHLET_ALPHA_BY_SIZE = {
    6: 0.35,
    8: 0.25,
}
DIRICHLET_ALPHA = float(DIRICHLET_ALPHA_BY_SIZE[int(BOARD_SIZE)])
DIRICHLET_EPSILON = 0.25
TEMPERATURE_SAMPLE_MOVES = 10


# TRAINING
MAX_GAMES = 300
TRAIN_AFTER_GAMES = 8
UPDATES_PER_GAME = 2
CHECKPOINT_EVERY_GAMES = 25
MIN_REPLAY_TO_TRAIN = 128
REPLAY_BUFFER_SIZE = 20_000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
VALUE_LOSS_WEIGHT = 1.0
