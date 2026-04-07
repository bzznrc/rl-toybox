"""Central configuration for Stealth."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_CELL_INSET as CELL_INSET,
    screen_height,
    screen_width,
)
from core.utils import env_flag, env_int


# RUNTIME
WINDOW_TITLE = "Stealth"
FPS = 15
TRAINING_FPS = 0
USE_GPU = env_flag("STEALTH_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
GRID_WIDTH_TILES = 24
GRID_HEIGHT_TILES = 18
TILE_SIZE = 28
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)

BASE_SEED = env_int("STEALTH_BASE_SEED", 2407)
LAYOUT_ATTEMPTS = 48
ROOM_PLACE_ATTEMPTS = 128
EXIT_VIEW_RANGE = 8
GUARD_VISION_RANGE = 5
GUARD_MOVE_PERIOD = 2
DRAW_GUARD_VISION = env_flag("STEALTH_DRAW_GUARD_VISION", True)

ROOM_SIZE_RANGE = (5, 8)
STEP_BUDGET_PER_ROUTE_TILE = 12.0
STEP_BUDGET_MIN = 70
PATCH_RADIUS = 2
LOCAL_VIEW_SIZE = 5
LOCAL_PATCH_VALUES = LOCAL_VIEW_SIZE * LOCAL_VIEW_SIZE

TOKEN_EMPTY = 0.0
TOKEN_WALL = 1.0
TOKEN_COVER = 2.0
TOKEN_EXIT = 3.0
TOKEN_GUARD = 4.0
TOKEN_DANGER = 5.0


# IO
PATCH_FEATURE_NAMES = [f"local_{index:02d}" for index in range(LOCAL_PATCH_VALUES)]
SCALAR_FEATURE_NAMES = [
    "has_exit_in_view",
    "exit_dx_if_seen",
    "exit_dy_if_seen",
    "on_cover",
    "danger_now",
    "steps_remaining_norm",
    "patrol_phase_norm",
]
INPUT_FEATURE_NAMES = [*PATCH_FEATURE_NAMES, *SCALAR_FEATURE_NAMES]
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


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 160,
    "check_window": 20,
    "success_threshold": 0.60,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "room_count": 3,
        "extra_connection": False,
        "guard_count": 1,
        "cover_count": 2,
        "min_start_exit_dist": 12,
        "entropy_coef": 0.040,
    },
    2: {
        "room_count": 4,
        "extra_connection": True,
        "guard_count": 1,
        "cover_count": 3,
        "min_start_exit_dist": 15,
        "entropy_coef": 0.030,
    },
    3: {
        "room_count": 5,
        "extra_connection": True,
        "guard_count": 2,
        "cover_count": 4,
        "min_start_exit_dist": 18,
        "entropy_coef": 0.020,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
PENALTY_STEP = -0.01
PENALTY_BLOCKED_MOVE = -0.015
REWARD_PROGRESS_SCALE = 0.03
REWARD_PROGRESS_CLIP = 1.0

REWARD_COMPONENTS = {
    "W": REWARD_WIN,
    "L": PENALTY_LOSE,
    "P": REWARD_PROGRESS_SCALE,
    "S": PENALTY_STEP,
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
