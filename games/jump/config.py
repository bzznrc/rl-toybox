"""Declarative configuration for Jump."""

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
WINDOW_TITLE = "Jump"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("JUMP_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
PLAYFIELD_HEIGHT = SCREEN_HEIGHT - BB_HEIGHT
PHYSICS_DT = 1.0 / FPS

PLAYER_TILES = 1
ENEMY_TILES = 1
PLAYER_SIZE = TILE_SIZE * PLAYER_TILES
ENEMY_SIZE = TILE_SIZE * ENEMY_TILES
PLATFORM_THICKNESS_TILES = 1
PLATFORM_THICKNESS_PX = TILE_SIZE * PLATFORM_THICKNESS_TILES
PLAYER_RUN_SPEED_PX_PER_SEC = 11.0 * TILE_SIZE
ENEMY_RUN_SPEED_PX_PER_SEC = 5.5 * TILE_SIZE
JUMP_VELOCITY_PX_PER_SEC = 23.0 * TILE_SIZE
GRAVITY_PX_PER_SEC2 = 57.5 * TILE_SIZE
MAX_FALL_SPEED_PX_PER_SEC = 28.0 * TILE_SIZE
COYOTE_TIME_STEPS = 4
GROUND_SNAP_PX = 4.0
ENEMY_STOMP_TOP_WINDOW_PX = 0.75 * TILE_SIZE
ENEMY_STOMP_MIN_OVERLAP_PX = 1.0
ENEMY_STOMP_BOUNCE_VELOCITY_PX_PER_SEC = 11.0 * TILE_SIZE
GENERATION_RUNWAY_TILES = 2
GOAL_FLAG_HEIGHT_TILES = 5
GOAL_FLAG_WIDTH_TILES = 2
BASE_SURFACE_ROW = 24
LANE_VERTICAL_SPACING_TILES = 4
LANE_SURFACE_ROWS = tuple(
    int(BASE_SURFACE_ROW - lane_idx * LANE_VERTICAL_SPACING_TILES)
    for lane_idx in range(3)
)
LANE_COUNT = len(LANE_SURFACE_ROWS)
STATUS_HISTORY_LIMIT = 12
CAMERA_LOOKAHEAD_PX = 0.18 * SCREEN_WIDTH

LOCAL_DX_NORM_PX = 0.5 * SCREEN_WIDTH
LOCAL_DY_NORM_PX = 0.5 * PLAYFIELD_HEIGHT
FLOOR_PROBE_F1_OFFSET_PX = 2.0 * TILE_SIZE
FLOOR_PROBE_F2_OFFSET_PX = 5.0 * TILE_SIZE
ARC_PROBE_F1_OFFSET_PX = 3.0 * TILE_SIZE
ARC_PROBE_F2_OFFSET_PX = 6.0 * TILE_SIZE
FLOOR_PROBE_STEP_UP_PX = 1.0 * TILE_SIZE
FLOOR_PROBE_DROP_PX = 1.5 * TILE_SIZE
ARC_PROBE_RISE_PX = (JUMP_VELOCITY_PX_PER_SEC * JUMP_VELOCITY_PX_PER_SEC) / (2.0 * GRAVITY_PX_PER_SEC2)
ARC_PROBE_DROP_PX = ARC_PROBE_RISE_PX + (2.0 * TILE_SIZE)
ARC_PROBE_PEAK_EXTRA_PX = 2.0 * TILE_SIZE
ARC_PROBE_SAMPLES = 5
SENS_DOWN_RANGE_PX = 10.0 * TILE_SIZE
SENS_UP_CLEAR_RANGE_PX = 16.0 * TILE_SIZE

LEVEL_GENERATION_ATTEMPTS = 64
STANDARD_PLATFORM_SIZE_TILES = [6, 9, 12]
STANDARD_START_PLATFORM_TILES = 9
STANDARD_GOAL_STRETCH_TILES = 12
SEGMENT_TARGET_SPACING_TILES = 12
EPISODE_STEPS_PER_TILE = 20.0
ENEMY_SPACING_TILES = 20.0
MIN_LEVEL_LENGTH_TILES = 48
BASE_GAP_MIN_TILES = 2
LEVEL3_EXTRA_GAP_MIN_TILES = 1
BASE_GAP_EXTRA_TILES = 1
LEVEL2_EXTRA_GAP_MAX_TILES = 1
DEFAULT_LANE_DELTA_CHOICES = [0, 0, 1, -1, -1]
ADVANCED_LANE_DELTA_CHOICES = [0, 1, 1, -1, -1]


# IO
INPUT_FEATURE_NAMES = [
    "self_vx_norm",
    "self_vy_norm",
    "self_grounded",
    "sens_floor_f1_norm",
    "sens_floor_f2_norm",
    "sens_floor_b1_norm",
    "sens_floor_b2_norm",
    "sens_arc_f1_norm",
    "sens_arc_f2_norm",
    "sens_up_clear_norm",
    "sens_down_ground_norm",
    "land_next_dx",
    "land_next_dy",
    "land_next2_dx",
    "land_next2_dy",
    "opp1_dx",
    "opp1_dy",
    "opp1_vx_norm",
    "opp2_dx",
    "opp2_dy",
    "opp2_vx_norm",
    "flag_goal_dx",
    "flag_goal_dy",
    "flag_progress_norm",
]
ACTION_NAMES = [
    "move_left",
    "move_right",
    "jump",
    "move_stop",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
if OBS_DIM != 24:
    raise RuntimeError(f"Jump INPUT_FEATURE_NAMES expected 24 entries, got {OBS_DIM}.")

ACTION_MOVE_LEFT = 0
ACTION_MOVE_RIGHT = 1
ACTION_JUMP = 2
ACTION_MOVE_STOP = 3


# GAME
DEFAULT_ALGO = "ppo"


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 250,
    "check_window": 25,
    "success_threshold": 0.65,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "length_tiles": 64,
        "lane_count": 1,
        "enemy_frequency": 0.25,
    },
    2: {
        "length_tiles": 96,
        "lane_count": 2,
        "enemy_frequency": 0.50,
    },
    3: {
        "length_tiles": 128,
        "lane_count": 3,
        "enemy_frequency": 1.00,
    },
}


# REWARDS
REWARD_FINISH = 10.0
PENALTY_FAIL = -5.0
REWARD_STOMP = 1.00
REWARD_STOMP_MAX = 5.00
FORWARD_PROGRESS_SCALE = 2.5
FORWARD_PROGRESS_CLIP = 0.10
BACKTRACK_PENALTY_SCALE = 2.5
BACKTRACK_PENALTY_CLIP = 0.10
PENALTY_STEP = -0.001
REWARD_COMPONENTS = {
    "outcome.reward_finish": REWARD_FINISH,
    "outcome.penalty_fail": PENALTY_FAIL,
    "combat.reward_stomp": REWARD_STOMP,
    "progress.forward_scale": FORWARD_PROGRESS_SCALE,
    "step.penalty_step": PENALTY_STEP,
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [32, 32],
}
ALGO_CONFIG_OVERRIDES = {
    "ppo": {
    "minibatch_size": 256,
    "entropy_coef": 0.005,
    }
}
DEFAULT_TRAIN_CONFIG = {
    "budget": 6_000_000,
    "rollout_steps": 1_024,
    "checkpoint_every": 10,
}
