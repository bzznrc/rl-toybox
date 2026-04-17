"""Central configuration for Snake AI."""

from __future__ import annotations

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


# RUNTIME
WINDOW_TITLE = "Snake AI"
FPS = 20
TRAINING_FPS = 0
USE_GPU = env_flag("SNAKE_USE_GPU", False)


# ENV
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
NN_CONTROL_MARKER_SIZE_PX = max(4, TILE_SIZE // 3)

WRAP_AROUND = True
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5


# IO
INPUT_FEATURE_NAMES = [
    "self_heading_sin",
    "self_heading_cos",
    "self_len_norm",
    "self_last_act_norm",
    "self_hunger_norm",
    "sens_fwd",
    "sens_left",
    "sens_right",
    "tgt_rel_angle_sin",
    "tgt_rel_angle_cos",
    "tgt_manhattan_norm",
    "tgt_dist_delta",
]
ACTION_NAMES = [
    "straight",
    "turn_right",
    "turn_left",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
SUCCESS_FOODS_REQUIRED = 5

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 250,
    "check_window": 25,
    "success_threshold": 0.80,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "num_obstacles": 0,
        "timeout_steps_per_length": 120,
    },
    2: {
        "num_obstacles": 6,
        "timeout_steps_per_length": 100,
    },
    3: {
        "num_obstacles": 12,
        "timeout_steps_per_length": 80,
    },
}


# REWARDS
PENALTY_LOSE = -5.0
REWARD_FOOD = 1.0
PENALTY_STEP = -0.005
PROGRESS_SCALE = 1.0
PROGRESS_CLIP = 0.05
REWARD_COMPONENTS = {
    "outcome.penalty_lose": PENALTY_LOSE,
    "event.reward_food": REWARD_FOOD,
    "progress.scale": PROGRESS_SCALE,
    "step.penalty_step": PENALTY_STEP,
}
