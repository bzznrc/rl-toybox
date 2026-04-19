"""Central configuration for Trail."""

from __future__ import annotations

import os

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_CELL_INSET as CELL_INSET,
    DEFAULT_GRID_COLUMNS as GRID_WIDTH_TILES,
    DEFAULT_GRID_ROWS as GRID_HEIGHT_TILES,
    DEFAULT_TILE_SIZE as TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_flag


def _resolve_observation_preset(raw_value: object) -> str:
    normalized = str(raw_value).strip().lower()
    if normalized in {"tiny", "small", "16"}:
        return "tiny"
    return "default"


# RUNTIME
WINDOW_TITLE = "Trail"
FPS = 18
TRAINING_FPS = 0
USE_GPU = env_flag("TRAIL_USE_GPU", False)


# ENV
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
MAX_EPISODE_STEPS = 320
TRAINING_TOTAL_GAMES = 1
PLAY_TOTAL_GAMES = 10
START_MARGIN_TILES = 6
START_OFFSET_CHOICES = (-4, -2, 0, 2, 4)
START_HORIZONTAL_SEPARATION_TILES = 12
OPPONENT_COMMIT_MIN_TICKS = 4
OPPONENT_COMMIT_MAX_TICKS = 6
OPPONENT_NEAR_TIE_EPSILON = 0.03
OPPONENT_OPENING_TOTAL_TICKS = 24
OPPONENT_OPENING_SHIFT_CHOICES = (-4, 4, 6)
GRID_LINE_ALPHA = 18
ARENA_OUTLINE_ALPHA = 64


# IO
DEFAULT_INPUT_FEATURE_NAMES = [
    # SELF
    "self_dir_x",
    "self_dir_y",
    # SENS
    "sens_fwd",
    "sens_left",
    "sens_right",
    "sens_back",
    "sens_fwd_left",
    "sens_fwd_right",
    # OPP
    "opp_dx",
    "opp_dy",
    "opp_dir_x",
    "opp_dir_y",
    "opp_dist_norm",
    "opp_fwd_align",
    # MAP
    "map_area_left_norm",
    "map_area_straight_norm",
    "map_area_right_norm",
    "map_area_adv_norm",
    "map_fill_ratio_norm",
    # FLAG
    "flag_time_norm",
]
TINY_INPUT_FEATURE_NAMES = [
    name
    for name in DEFAULT_INPUT_FEATURE_NAMES
    if name not in {"sens_fwd_left", "sens_fwd_right", "map_area_adv_norm", "flag_time_norm"}
]
OBSERVATION_PRESET = _resolve_observation_preset(os.getenv("TRAIL_OBS_PRESET", "default"))
INPUT_FEATURE_NAMES_BY_PRESET = {
    "default": DEFAULT_INPUT_FEATURE_NAMES,
    "tiny": TINY_INPUT_FEATURE_NAMES,
}
INPUT_FEATURE_NAMES = list(INPUT_FEATURE_NAMES_BY_PRESET[str(OBSERVATION_PRESET)])
ACTION_NAMES = [
    "turn_left",
    "go_straight",
    "turn_right",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
if OBS_DIM not in {16, 20}:
    raise RuntimeError(f"Trail INPUT_FEATURE_NAMES expected 16 or 20 entries, got {OBS_DIM}.")


# MODEL
DEFAULT_HIDDEN_SIZES = (32, 32)
TINY_HIDDEN_SIZES = (24, 24)


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 200,
    "check_window": 20,
    "success_threshold": 0.60,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "max_episode_steps": 260,
        "opponent_strength": 0.15,
        "entropy_coef": 0.020,
    },
    2: {
        "max_episode_steps": 300,
        "opponent_strength": 0.55,
        "entropy_coef": 0.015,
    },
    3: {
        "max_episode_steps": 340,
        "opponent_strength": 1.00,
        "entropy_coef": 0.010,
    },
}


# REWARDS
REWARD_WIN = 1.0
PENALTY_LOSE = -1.0
REWARD_DRAW = 0.0
REWARD_SPACE_CONTROL_SCALE = 0.03
REWARD_SPACE_CONTROL_CLIP = 0.01
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "shape.reward_space_control": REWARD_SPACE_CONTROL_CLIP,
}
