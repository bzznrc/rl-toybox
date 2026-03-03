"""Central configuration for Snake AI."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_CELL_INSET,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
)
from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Snake AI"
FPS = 20
TRAINING_FPS = 0
USE_GPU = env_flag("SNAKE_USE_GPU", False)


# ENV
GRID_WIDTH_TILES = DEFAULT_GRID_COLUMNS
GRID_HEIGHT_TILES = DEFAULT_GRID_ROWS
TILE_SIZE = DEFAULT_TILE_SIZE
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
SCREEN_WIDTH = GRID_WIDTH_TILES * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT_TILES * TILE_SIZE + BB_HEIGHT
CELL_INSET = DEFAULT_CELL_INSET
NN_CONTROL_MARKER_SIZE_PX = max(4, TILE_SIZE // 3)

WRAP_AROUND = True
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5


# IO
INPUT_FEATURE_NAMES = [
    "self_heading_sin",
    "self_heading_cos",
    "self_length",
    "self_last_action",
    "ray_fwd",
    "ray_left",
    "ray_right",
    "tgt_rel_angle_sin",
    "tgt_rel_angle_cos",
    "tgt_manhattan_dist",
    "tgt_dist_delta",
    "self_steps_since_food",
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


# TRAINING
HIDDEN_DIMENSIONS = [32]

MAX_TRAINING_STEPS = 3_000_000
CHECKPOINT_EVERY_STEPS = 100_000

MAX_MEMORY = 100_000
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 300_000
EPS_BUMP_PATIENCE_EPISODES = 200
EPS_BUMP_MIN_IMPROVEMENT = 0.10
EPS_BUMP_CAP = 0.25
EPS_BUMP_COOLDOWN_STEPS = EPSILON_DECAY_STEPS // 2
