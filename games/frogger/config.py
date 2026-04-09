"""Central configuration for Frogger."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_CELL_INSET as CELL_INSET,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_flag, env_int


# RUNTIME
WINDOW_TITLE = "Frogger"
FPS = 14
TRAINING_FPS = 0
USE_GPU = env_flag("FROGGER_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
WORLD_WIDTH = screen_width(DEFAULT_GRID_COLUMNS, DEFAULT_TILE_SIZE)
WORLD_HEIGHT = DEFAULT_GRID_ROWS * DEFAULT_TILE_SIZE
SCREEN_WIDTH = WORLD_WIDTH
SCREEN_HEIGHT = screen_height(DEFAULT_GRID_ROWS, DEFAULT_TILE_SIZE, BB_HEIGHT)

BOARD_WIDTH_TILES = 21
MAX_LANE_COUNT = 5
MAX_BOARD_ROWS = MAX_LANE_COUNT + 2
COLUMN_WIDTH_PX = 40
ROW_PITCH_PX = 64
ROW_BAND_HEIGHT_RATIO = 0.72
FROG_SIZE_PX = 40
CAR_LENGTH_TILES = 2.10
CAR_HEIGHT_RATIO = 0.50
CAR_FRONT_STRIP_RATIO = 0.18
LANE_DASH_WIDTH_RATIO = 0.58
LANE_DASH_HEIGHT_RATIO = 0.06
LANE_DASH_GAP_RATIO = 0.42
GOAL_MARKER_HEIGHT_RATIO = 0.16
POINT_ICON_PACK_SIZE = 5
MIN_CAR_GAP_TILES = 4.8

BASE_SEED = env_int("FROGGER_BASE_SEED", 4813)
MAX_LANE_SPEED = 0.80
PATCH_RADIUS = 2
LOCAL_VIEW_SIZE = 5
LOCAL_PATCH_VALUES = LOCAL_VIEW_SIZE * LOCAL_VIEW_SIZE

TOKEN_EMPTY = 0.0
TOKEN_BOUNDARY = 1.0
TOKEN_SAFE = 2.0
TOKEN_ROAD = 3.0
TOKEN_CAR = 4.0
TOKEN_GOAL = 5.0


# IO
PATCH_FEATURE_NAMES = [f"local_{index:02d}" for index in range(LOCAL_PATCH_VALUES)]
SCALAR_FEATURE_NAMES = [
    "run_steps_remaining_norm",
    "frog_lane_id_norm",
    "frog_x_norm",
    "goal_dy_norm",
    "lane_dir_here",
    "lane_speed_here_norm",
    "flag_danger_now",
]
INPUT_FEATURE_NAMES = [*PATCH_FEATURE_NAMES, *SCALAR_FEATURE_NAMES]
ACTION_NAMES = [
    "up",
    "down",
    "left",
    "right",
    "wait",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
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
        "lane_count": 1,
        "style_indices": [0, 1],
        "car_count_choices": [1],
        "max_steps": 160,
        "entropy_coef": 0.040,
    },
    2: {
        "lane_count": 3,
        "style_indices": [0, 1, 2],
        "car_count_choices": [1, 2],
        "max_steps": 240,
        "entropy_coef": 0.030,
    },
    3: {
        "lane_count": 5,
        "style_indices": [0, 1, 2],
        "car_count_choices": [1, 2],
        "max_steps": 320,
        "entropy_coef": 0.020,
    },
}


# REWARDS
REWARD_PROGRESS_FORWARD = 0.05
REWARD_EVENT_GOAL = 1.0
REWARD_EVENT_HIT = -1.0
REWARD_COST_STEP = -0.01
REWARD_TERMINAL_WIN = 2.0
REWARD_TERMINAL_LOSS = -2.0
REWARD_COMPONENT_NAMES = (
    "reward_progress_forward",
    "reward_event_goal",
    "reward_event_hit",
    "reward_cost_step",
    "reward_terminal_win",
    "reward_terminal_loss",
)


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
