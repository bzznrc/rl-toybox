"""Central configuration for Vroom."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Vroom"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("VROOM_USE_GPU", False)


# ENV
GRID_WIDTH_TILES = DEFAULT_GRID_COLUMNS
GRID_HEIGHT_TILES = DEFAULT_GRID_ROWS
TILE_SIZE = DEFAULT_TILE_SIZE
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)


# IO
INPUT_FEATURE_NAMES = [
    "self_lat_offset",
    "self_lat_offset_delta",
    "self_fwd_speed",
    "self_fwd_speed_delta",
    "self_heading_sin",
    "self_heading_cos",
    "self_in_contact",
    "self_last_action",
    "ray_fwd_near",
    "ray_fwd_far",
    "ray_fwd_left",
    "ray_fwd_right",
    "tgt_dx",
    "tgt_dy",
    "tgt_dvx",
    "tgt_dvy",
    "trk_lookahead_sin",
    "trk_lookahead_cos",
    "trk_lookahead_dist",
    "trk_curvature_ahead",
]
ACTION_NAMES = [
    "coast",
    "throttle",
    "left_coast",
    "right_coast",
    "left_throttle",
    "right_throttle",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 200,
    "check_window": 100,
    "success_threshold": 0.60,
    "consecutive_checks_required": 3,
}

LEVEL_SETTINGS = {
    1: {
        "num_cars": 1,
        "opponent_speed_cap": 0.0,
        "obstacle_clusters": 2,
    },
    2: {
        "num_cars": 2,
        "opponent_speed_cap": 0.75,
        "obstacle_clusters": 4,
    },
    3: {
        "num_cars": 4,
        "opponent_speed_cap": 1.0,
        "obstacle_clusters": 4,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
PENALTY_STEP = -0.01
PROGRESS_SCALE = 1.0
PROGRESS_CLIP = 0.2
PENALTY_COLLISION = -1.0
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "progress.scale": PROGRESS_SCALE,
    "progress.clip": PROGRESS_CLIP,
    "event.penalty_collision": PENALTY_COLLISION,
    "step.penalty_step": PENALTY_STEP,
}


# TRAINING
HIDDEN_DIMENSIONS = [48, 48]

MAX_TRAINING_STEPS = 10_000_000
CHECKPOINT_EVERY_STEPS = 100_000

REPLAY_BUFFER_SIZE = 200_000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
GAMMA = 0.99
TARGET_SYNC_EVERY = 2_000
GRAD_CLIP_NORM = 10.0

LEARN_START_STEPS = 20_000
TRAIN_EVERY_STEPS = 4
UPDATES_PER_TRAIN = 1
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 1_500_000
EPS_BUMP_PATIENCE_EPISODES = 100
EPS_BUMP_MIN_IMPROVEMENT = 0.10
EPS_BUMP_CAP = 0.35
EPS_BUMP_COOLDOWN_STEPS = EPSILON_DECAY_STEPS // 2
