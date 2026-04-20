"""Central configuration for Osero."""

from __future__ import annotations

import os

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.io_schema import row_major_grid_action_names, row_major_grid_feature_names
from core.utils import env_flag


SUPPORTED_BOARD_SIZES = (4, 6, 8)
DEFAULT_BOARD_SIZE = 6


def _resolve_board_size(raw_value: object) -> int:
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if "x" in normalized:
            normalized = normalized.split("x", 1)[0].strip()
        raw_value = normalized
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
USE_GPU = env_flag("OSERO_USE_GPU", False)
BOARD_SIZE = _resolve_board_size(os.getenv("OSERO_BOARD_SIZE", str(DEFAULT_BOARD_SIZE)))


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
    4: row_major_grid_feature_names(4, prefix="board"),
    6: row_major_grid_feature_names(6, prefix="board"),
    8: row_major_grid_feature_names(8, prefix="board"),
}
ACTION_NAMES_BY_SIZE = {
    4: row_major_grid_action_names(4, include_pass=True),
    6: row_major_grid_action_names(6, include_pass=True),
    8: row_major_grid_action_names(8, include_pass=True),
}
INPUT_FEATURE_NAMES = list(INPUT_FEATURE_NAMES_BY_SIZE[int(BOARD_SIZE)])
ACTION_NAMES = list(ACTION_NAMES_BY_SIZE[int(BOARD_SIZE)])
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# GAME
DEFAULT_ALGO = "search_play"
GAME_CAPABILITIES = {
    "masked_actions": True,
    "self_play": True,
}
ENV_METADATA = {
    "board_size": int(BOARD_SIZE),
}


# MODEL / SEARCH
DEFAULT_MODEL_CONFIG_BY_SIZE = {
    4: {
        "hidden_sizes": [48, 48],
    },
    6: {
        "hidden_sizes": [64, 64],
    },
    8: {
        "hidden_sizes": [96, 96],
    },
}
ALGO_CONFIG_OVERRIDES_BY_SIZE = {
    4: {
        "search_play": {
        "simulations_per_move": 32,
        "dirichlet_alpha": 0.5,
        },
    },
    6: {
        "search_play": {
        "simulations_per_move": 48,
        "dirichlet_alpha": 0.35,
        },
    },
    8: {
        "search_play": {
        "simulations_per_move": 64,
        "dirichlet_alpha": 0.25,
        },
    },
}
CPUCT = 1.25
DIRICHLET_EPSILON = 0.25
TEMPERATURE_SAMPLE_MOVES = 10


# TRAINING
DEFAULT_MODEL_CONFIG = dict(DEFAULT_MODEL_CONFIG_BY_SIZE[int(BOARD_SIZE)])
ALGO_CONFIG_OVERRIDES = dict(ALGO_CONFIG_OVERRIDES_BY_SIZE[int(BOARD_SIZE)])
DEFAULT_TRAIN_CONFIG = {
    "budget": 12_000_000,
    "checkpoint_every": 25,
}
