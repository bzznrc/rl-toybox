"""Declarative configuration for Jump."""

from __future__ import annotations

from core.shared_config import (
    BB_HEIGHT as BB_HEIGHT,
    CELL_INSET as CELL_INSET,
    FPS as FPS,
    PHYSICS_DT as PHYSICS_DT,
    PLAYFIELD_HEIGHT,
    SCREEN_HEIGHT as SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TRAINING_FPS as TRAINING_FPS,
)
from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Jump"
USE_GPU = env_flag("JUMP_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False
SHOW_SENS_PATCH_GRID = False
SHOW_GHOST_OVERLAY = env_flag("JUMP_SHOW_GHOST_OVERLAY", SHOW_SENS_PATCH_GRID)
GHOST_OVERLAY_ALPHA = 128


# ENV
PLAYER_TILES = 1
ENEMY_TILES = 1
PLATFORM_BLOCK_TILES = 1
PLAYER_SIZE = TILE_SIZE * PLAYER_TILES
ENEMY_SIZE = TILE_SIZE * ENEMY_TILES
PLATFORM_THICKNESS_TILES = PLATFORM_BLOCK_TILES
PLATFORM_THICKNESS_PX = TILE_SIZE * PLATFORM_THICKNESS_TILES
STANDARD_PLATFORM_MATERIAL_ROWS = 2
MOVING_PLATFORM_MATERIAL_ROWS = 1
STANDARD_PLATFORM_HEIGHT_PX = TILE_SIZE * STANDARD_PLATFORM_MATERIAL_ROWS
MOVING_PLATFORM_HEIGHT_PX = TILE_SIZE * MOVING_PLATFORM_MATERIAL_ROWS
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
MOVING_PLATFORM_BLOCK_COUNTS = [6, 9]
MOVING_PLATFORM_SIZE_TILES = [
    int(block_count) * int(PLATFORM_BLOCK_TILES)
    for block_count in MOVING_PLATFORM_BLOCK_COUNTS
]
MOVING_PLATFORM_SPEED_PX_PER_SEC = 4.0 * TILE_SIZE
MOVING_PLATFORM_ENTRY_GAP_MIN_TILES = 1
MOVING_PLATFORM_ENTRY_GAP_MAX_TILES = 3
MOVING_PLATFORM_EXIT_GAP_MIN_TILES = 1
MOVING_PLATFORM_EXIT_GAP_MAX_TILES = 3
MOVING_PLATFORM_TRAVEL_MIN_TILES = 4
MOVING_PLATFORM_TRAVEL_MAX_TILES = 8

LEVEL_GENERATION_ATTEMPTS = 64
STANDARD_PLATFORM_BLOCK_COUNTS = [6, 9, 12]
STANDARD_PLATFORM_SIZE_TILES = [
    int(block_count) * int(PLATFORM_BLOCK_TILES)
    for block_count in STANDARD_PLATFORM_BLOCK_COUNTS
]
STANDARD_START_PLATFORM_TILES = 9 * PLATFORM_BLOCK_TILES
STANDARD_GOAL_STRETCH_TILES = 12 * PLATFORM_BLOCK_TILES
SEGMENT_TARGET_SPACING_TILES = 12
EPISODE_STEPS_PER_TILE = 20.0
ENEMY_SPACING_TILES = 20.0
MIN_LEVEL_LENGTH_TILES = 48
BASE_GAP_MIN_TILES = 2
TOP_LEVEL_EXTRA_GAP_MIN_TILES = 1
BASE_GAP_EXTRA_TILES = 1
ADVANCED_GAP_MAX_BONUS_TILES = 1
DEFAULT_LANE_DELTA_CHOICES = [0, 0, 1, -1, -1]
ADVANCED_LANE_DELTA_CHOICES = [0, 1, 1, -1, -1]


# IO
# Observation order: SELF (4) + SENS (8) + LAND (10) + OPP (6) + HAZ (4) + FLAG (4) = 36.
INPUT_FEATURE_NAMES = [
    "self_vx_norm",
    "self_vy_norm",
    "self_grounded",
    "self_lane_norm",
    "sens_ground_l2",
    "sens_ground_l1",
    "sens_ground_c0",
    "sens_ground_r1",
    "sens_ground_r2",
    "sens_gap_f1",
    "sens_gap_f2",
    "sens_gap_f3",
    "land_next_dx",
    "land_next_dy",
    "land_next_width",
    "land_next_lane_delta",
    "land_gap_dx",
    "land_gap_width",
    "land_move_dx",
    "land_move_dy",
    "land_move_vx_norm",
    "land_move_phase",
    "opp1_dx",
    "opp1_dy",
    "opp1_vx_norm",
    "opp1_tti",
    "opp2_dx",
    "opp2_dy",
    "haz_route_dx",
    "haz_route_tti",
    "haz_lane_dx",
    "haz_lane_tti",
    "flag_goal_dx",
    "flag_goal_dy",
    "flag_progress_norm",
    "flag_time_left",
]
ACTION_NAMES = [
    "move_left",
    "move_right",
    "jump",
    "move_stop",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
if OBS_DIM != 36:
    raise RuntimeError(f"Jump INPUT_FEATURE_NAMES expected 36 entries, got {OBS_DIM}.")

ACTION_MOVE_LEFT = 0
ACTION_MOVE_RIGHT = 1
ACTION_JUMP = 2
ACTION_MOVE_STOP = 3


# GAME
DEFAULT_ALGO = "ppo"


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 5
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 100,
    "success_threshold": 0.60,
}

LEVEL_SETTINGS = {
    1: {
        "length_tiles": 48,
        "lane_count": 1,
        "enemy_frequency": 0.0,
        "moving_platform_frequency": 0.0,
    },
    2: {
        "length_tiles": 64,
        "lane_count": 1,
        "enemy_frequency": 0.25,
        "moving_platform_frequency": 0.0,
    },
    3: {
        "length_tiles": 80,
        "lane_count": 2,
        "enemy_frequency": 0.25,
        "moving_platform_frequency": 0.0,
    },
    4: {
        "length_tiles": 104,
        "lane_count": 2,
        "enemy_frequency": 0.50,
        "moving_platform_frequency": 0.35,
    },
    5: {
        "length_tiles": 128,
        "lane_count": 3,
        "enemy_frequency": 0.75,
        "moving_platform_frequency": 0.75,
    },
}


# REWARDS
REWARD_FINISH = 10.0
PENALTY_FAIL = -5.0
REWARD_STOMP = 1.00
REWARD_STOMP_MAX = 5.00
PROGRESS_SCALE = 2.5
PROGRESS_CLIP = 0.10
STALL_PROGRESS_EPS = 0.0005
PENALTY_STALL = -0.005
REWARD_COMPONENTS = {
    "outcome.reward_finish": REWARD_FINISH,
    "outcome.penalty_fail": PENALTY_FAIL,
    "combat.reward_stomp": REWARD_STOMP,
    "progress.scale": PROGRESS_SCALE,
    "progress.penalty_stall": PENALTY_STALL,
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
