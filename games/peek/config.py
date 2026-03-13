"""Central configuration for Peek."""

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
from core.utils import env_flag, env_int


# RUNTIME
WINDOW_TITLE = "Peek"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("PEEK_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)

BASE_SEED = env_int("PEEK_BASE_SEED", 2026)
LAYOUT_ATTEMPTS = 32
ROOM_PLACE_ATTEMPTS = 96
VISIBILITY_RANGE = 7
RAY_RANGE = 6
GUARD_VISION_RANGE = 4
GUARD_MOVE_PERIOD = 2
DRAW_GUARD_VISION = env_flag("PEEK_DRAW_GUARD_VISION", True)

ROOM_SIZE_MIN_TILES = max(5, GRID_HEIGHT_TILES // 5)
ROOM_SIZE_MAX_TILES = min(10, ROOM_SIZE_MIN_TILES + 4)
ROOM_SIZE_RANGE = (ROOM_SIZE_MIN_TILES, ROOM_SIZE_MAX_TILES)
MAP_PATH_SPAN_TILES = GRID_WIDTH_TILES + GRID_HEIGHT_TILES
START_KEY_DISTANCE_RATIO_MIN = 0.15
START_KEY_DISTANCE_RATIO_MAX = 0.30
KEY_DOOR_DISTANCE_RATIO = 0.55
# Episode length is derived from the expected route length and guard pressure.
STEP_BUDGET_PER_ROUTE_TILE = 20.0
STEP_BUDGET_PER_GUARD = 20.0


# IO
INPUT_FEATURE_NAMES = [
    "self_has_key",
    "self_time_left",
    "self_here_revisited",
    "ray_wall_up",
    "ray_wall_down",
    "ray_wall_left",
    "ray_wall_right",
    "obj1_dx",
    "obj1_dy",
    "obj1_type",
    "mem_visited_up",
    "mem_visited_down",
    "mem_visited_left",
    "mem_visited_right",
    "opp1_dx",
    "opp1_dy",
    "opp1_facing_dx",
    "opp1_facing_dy",
]
ACTION_NAMES = [
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "wait",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)

ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_WAIT = 4

OBJECT_EMPTY = 0.0
OBJECT_KEY = 1.0
OBJECT_DOOR = 2.0


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 200,
    "check_window": 25,
    "success_threshold": 0.60,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "room_count": 2,
        "extra_connection": False,
        "guard_count": 0,
        "entropy_coef": 0.04,
    },
    2: {
        "room_count": 6,
        "extra_connection": True,
        "guard_count": 2,
        "entropy_coef": 0.02,
    },
    3: {
        "room_count": 8,
        "extra_connection": True,
        "guard_count": 4,
        "entropy_coef": 0.015,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
REWARD_KEY = 2.5
REWARD_EXPLORE_NEW_TILE = 0.02
PENALTY_BLOCKED_MOVE = -0.01

REWARD_COMPONENTS = {
    "W": REWARD_WIN,
    "L": PENALTY_LOSE,
    "K": REWARD_KEY,
    "P": REWARD_EXPLORE_NEW_TILE,
    "B": PENALTY_BLOCKED_MOVE,
}


# TRAINING
ENCODER_HIDDEN_DIMENSIONS = [32]
ACTOR_HEAD_HIDDEN_DIMENSIONS = [32]
CRITIC_HEAD_HIDDEN_DIMENSIONS = [32]
RECURRENT_TYPE = "lstm"
RECURRENT_HIDDEN_SIZE = 64
RECURRENT_SEQ_LEN = 64

MAX_TRAINING_ITERATIONS = 12000
ROLLOUT_STEPS = 1024
CHECKPOINT_EVERY_ITERATIONS = 10

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
