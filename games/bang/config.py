"""Central configuration for Bang AI."""

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
WINDOW_TITLE = "Bang AI"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("BANG_USE_GPU", False)


# ENV
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
NN_CONTROL_MARKER_SIZE_PX = max(4, TILE_SIZE // 3)

PLAYER_MOVE_SPEED = 5
AIM_RATE_PER_STEP = 5
PROJECTILE_SPEED = 10
SHOOT_COOLDOWN_FRAMES = 30
AIM_TOLERANCE_DEGREES = 10
MAX_EPISODE_STEPS = 1200
EVENT_TIMER_NORMALIZATION_FRAMES = MAX_EPISODE_STEPS
PLAYER_SPAWN_X_RATIO = 1 / 8
ENEMY_SPAWN_X_RATIO = 7 / 8

ENEMY_SHOT_ERROR_CHOICES = [-20, -10, 0, 10, 20]
ENEMY_MOVE_COMMIT_FRAMES = 10
ENEMY_RECENT_POSITION_MEMORY = 8
ENEMY_HIDDEN_URGENCY_FRAMES = 24
ENEMY_RECENT_POSITION_PENALTY = 0.40
SPAWN_Y_OFFSET = 180
SAFE_RADIUS = 100
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5
OBSTACLE_START_ATTEMPTS = 100

PROJECTILE_TRAJECTORY_DOT_THRESHOLD = 0.98
PROJECTILE_HITBOX_SIZE = 10


# IO
INPUT_FEATURE_NAMES = [
    "self_ang_sin",
    "self_ang_cos",
    "self_move_x",
    "self_move_y",
    "self_shot_cd_norm",
    "sens_fwd",
    "sens_left",
    "sens_right",
    "sens_back",
    "opp1_dx",
    "opp1_dy",
    "opp1_los",
    "opp1_ang_sin",
    "opp1_ang_cos",
    "opp2_dx",
    "opp2_dy",
    "opp2_los",
    "opp2_ang_sin",
    "opp2_ang_cos",
    "opp3_dx",
    "opp3_dy",
    "opp3_los",
    "opp3_ang_sin",
    "opp3_ang_cos",
    "opp_near_dist_norm",
    "haz_tti_norm",
    "haz_miss_norm",
    "haz_in_traj",
]
ACTION_NAMES = [
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_stop",
    "aim_left",
    "aim_right",
    "shoot",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)

ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_STOP_MOVE = 4
ACTION_AIM_LEFT = 5
ACTION_AIM_RIGHT = 6
ACTION_SHOOT = 7


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 250,
    "check_window": 25,
    "success_threshold": 0.40,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "num_players": 2,
        "num_obstacles": 4,
        "enemy_reposition_bias": 0.25,
        "enemy_shoot_probability": 0.025,
    },
    2: {
        "num_players": 2,
        "num_obstacles": 8,
        "enemy_reposition_bias": 0.60,
        "enemy_shoot_probability": 0.05,
    },
    3: {
        "num_players": 4,
        "num_obstacles": 12,
        "enemy_reposition_bias": 1.00,
        "enemy_shoot_probability": 0.10,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
REWARD_KILL = 2.0
PENALTY_STEP = -0.005
ENGAGEMENT_SCALE = 0.5
ENGAGEMENT_CLIP = 0.25
HAZARD_SCALE = 0.5
HAZARD_CLIP = 0.25
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "event.reward_kill": REWARD_KILL,
    "progress.engagement_scale": ENGAGEMENT_SCALE,
    "progress.hazard_scale": HAZARD_SCALE,
    "step.penalty_step": PENALTY_STEP,
}
