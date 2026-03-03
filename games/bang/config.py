"""Central configuration for Bang AI."""

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
WINDOW_TITLE = "Bang AI"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("BANG_USE_GPU", False)


# ENV
BOARD_COLUMNS = DEFAULT_GRID_COLUMNS
BOARD_ROWS = DEFAULT_GRID_ROWS
BOARD_CELL_SIZE_PX = DEFAULT_TILE_SIZE
BOARD_BOTTOM_BAR_HEIGHT_PX = DEFAULT_BOTTOM_BAR_HEIGHT
BOARD_CELL_INSET_PX = DEFAULT_CELL_INSET

GRID_WIDTH_TILES = BOARD_COLUMNS
GRID_HEIGHT_TILES = BOARD_ROWS
TILE_SIZE = BOARD_CELL_SIZE_PX
BB_HEIGHT = BOARD_BOTTOM_BAR_HEIGHT_PX
SCREEN_WIDTH = GRID_WIDTH_TILES * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT_TILES * TILE_SIZE + BB_HEIGHT
CELL_INSET = BOARD_CELL_INSET_PX
NN_CONTROL_MARKER_SIZE_PX = max(4, BOARD_CELL_SIZE_PX // 3)

PLAYER_MOVE_SPEED = 5
AIM_RATE_PER_STEP = 5
PROJECTILE_SPEED = 10
SHOOT_COOLDOWN_FRAMES = 30
AIM_TOLERANCE_DEGREES = 10
MAX_EPISODE_STEPS = 1200
EVENT_TIMER_NORMALIZATION_FRAMES = MAX_EPISODE_STEPS
PLAYER_SPAWN_X_RATIO = 1 / 8
ENEMY_SPAWN_X_RATIO = 7 / 8

ENEMY_STUCK_MOVE_ATTEMPTS = 2
ENEMY_ESCAPE_FOLLOW_FRAMES = 12
ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES = (90, -90, 180)
ENEMY_SHOT_ERROR_CHOICES = [-20, -10, 0, 10, 20]
SPAWN_Y_OFFSET = 180
SAFE_RADIUS = 100
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5
OBSTACLE_START_ATTEMPTS = 100

PROJECTILE_TRAJECTORY_DOT_THRESHOLD = 0.98
PROJECTILE_HITBOX_SIZE = 10


# IO
INPUT_FEATURE_NAMES = [
    "self_angle_sin",
    "self_angle_cos",
    "self_move_intent_x",
    "self_move_intent_y",
    "self_shot_cd_norm",
    "self_tgt_seen_norm",
    "ray_fwd",
    "ray_left",
    "ray_right",
    "ray_back",
    "tgt_dx",
    "tgt_dy",
    "tgt_dvx",
    "tgt_dvy",
    "tgt_dist",
    "tgt_in_los",
    "tgt_rel_angle_sin",
    "tgt_rel_angle_cos",
    "haz_dx",
    "haz_dy",
    "haz_dvx",
    "haz_dvy",
    "haz_dist",
    "haz_in_trajectory",
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
    "min_episodes_per_level": 200,
    "check_window": 100,
    "success_threshold": 0.40,
    "consecutive_checks_required": 3,
}

LEVEL_SETTINGS = {
    1: {
        "num_players": 2,
        "num_obstacles": 4,
        "enemy_move_probability": 0.00,
        "enemy_shoot_probability": 0.02,
    },
    2: {
        "num_players": 2,
        "num_obstacles": 8,
        "enemy_move_probability": 0.10,
        "enemy_shoot_probability": 0.06,
    },
    3: {
        "num_players": 4,
        "num_obstacles": 12,
        "enemy_move_probability": 0.20,
        "enemy_shoot_probability": 0.10,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
REWARD_KILL = 2.0
PENALTY_STEP = -0.005
ENGAGEMENT_SCALE = 0.2
ENGAGEMENT_CLIP = 0.1
HAZARD_SCALE = 0.2
HAZARD_CLIP = 0.1
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "event.reward_kill": REWARD_KILL,
    "progress.engagement_scale": ENGAGEMENT_SCALE,
    "progress.hazard_scale": HAZARD_SCALE,
    "step.penalty_step": PENALTY_STEP,
}


# TRAINING
HIDDEN_DIMENSIONS = [64, 64]

TOTAL_TRAINING_STEPS = 10_000_000
CHECKPOINT_EVERY_STEPS = 200_000

REPLAY_BUFFER_SIZE = 500_000
BATCH_SIZE = 256
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 1e-5
GAMMA = 0.99
TARGET_SYNC_EVERY = 10_000
GRAD_CLIP_NORM = 10.0

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = TOTAL_TRAINING_STEPS
PER_EPSILON = 1e-4

LEARN_START_STEPS = 50_000
TRAIN_EVERY_STEPS = 4
UPDATES_PER_TRAIN = 1
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 2_500_000
EPS_BUMP_PATIENCE_EPISODES = 150
EPS_BUMP_MIN_IMPROVEMENT = 0.10
EPS_BUMP_CAP = 0.35
EPS_BUMP_COOLDOWN_STEPS = EPSILON_DECAY_STEPS // 2
