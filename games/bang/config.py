"""Central configuration for Bang AI."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_CELL_INSET,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
)
from core.utils import PROJECT_ROOT, env_flag

BOARD_COLUMNS = DEFAULT_GRID_COLUMNS
BOARD_ROWS = DEFAULT_GRID_ROWS
BOARD_CELL_SIZE_PX = DEFAULT_TILE_SIZE
BOARD_BOTTOM_BAR_HEIGHT_PX = DEFAULT_BOTTOM_BAR_HEIGHT
BOARD_CELL_INSET_PX = DEFAULT_CELL_INSET

# Quick toggles
SHOW_GAME_OVERRIDE: bool | None = None
USE_GPU = env_flag("BANG_USE_GPU", False)
LOAD_MODEL = "B"  # False, "B" (best), or "L" (last)
# Set to None to start training from MIN_LEVEL, or set an explicit level in [MIN_LEVEL, MAX_LEVEL].
RESUME_LEVEL = 3
# Used by both `play_ai` and `play_user`.
PLAY_OPPONENT_LEVEL = 5

# Runtime
FPS = 60
TRAINING_FPS = 0
WINDOW_TITLE = "Bang AI"

# Arena dimensions
GRID_WIDTH_TILES = BOARD_COLUMNS
GRID_HEIGHT_TILES = BOARD_ROWS
TILE_SIZE = BOARD_CELL_SIZE_PX
BB_HEIGHT = BOARD_BOTTOM_BAR_HEIGHT_PX
SCREEN_WIDTH = GRID_WIDTH_TILES * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT_TILES * TILE_SIZE + BB_HEIGHT
CELL_INSET = BOARD_CELL_INSET_PX
CELL_INSET_DOUBLE = CELL_INSET * 2

# Rendering
NN_CONTROL_MARKER_SIZE_PX = max(4, BOARD_CELL_SIZE_PX // 3)

# Input/output spaces
INPUT_FEATURE_NAMES = [
    # SELF (8)
    "self_angle_sin",
    "self_angle_cos",
    "self_move_intent_x",
    "self_move_intent_y",
    "self_aim_intent",
    "self_last_action",
    "self_time_since_shot",
    "self_time_since_tgt_seen",
    # RAYS (4)
    "ray_fwd",
    "ray_left",
    "ray_right",
    "ray_back",
    # TGT (6)
    "tgt_dx",
    "tgt_dy",
    "tgt_dvx",
    "tgt_dvy",
    "tgt_dist",
    "tgt_in_los",
    # HAZ (6)
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
NUM_INPUT_FEATURES = OBS_DIM
NUM_ACTIONS = ACT_DIM
STATE_SIZE = OBS_DIM
ACTION_SIZE = ACT_DIM
MODEL_INPUT_SIZE = OBS_DIM
MODEL_OUTPUT_SIZE = ACT_DIM

ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_STOP_MOVE = 4
ACTION_AIM_LEFT = 5
ACTION_AIM_RIGHT = 6
ACTION_SHOOT = 7

# Gameplay tuning
PLAYER_MOVE_SPEED = 5
AIM_RATE_PER_STEP = 5
PROJECTILE_SPEED = 10
SHOOT_COOLDOWN_FRAMES = 30
AIM_TOLERANCE_DEGREES = 10
MAX_EPISODE_STEPS = 1200
EVENT_TIMER_NORMALIZATION_FRAMES = MAX_EPISODE_STEPS
PLAYER_SPAWN_X_RATIO = 1 / 8
ENEMY_SPAWN_X_RATIO = 7 / 8

# Enemy behavior / curriculum
MIN_LEVEL = 1
MAX_LEVEL = 5
REWARD_ROLLING_WINDOW = 100
CURRICULUM_REWARD_THRESHOLDS = [8.0, 8.0, 6.0, 6.0]
CURRICULUM_CONSECUTIVE_CHECKS = 5
CURRICULUM_MIN_EPISODES_PER_LEVEL = 100
LEVEL_SETTINGS = {
    1: {
        "num_players": 2,
        "num_obstacles": 4,
        "enemy_move_probability": 0.00,
        "enemy_shot_error_choices": [-24, -12, 0, 12, 24],
        "enemy_shoot_probability": 0.04,
    },
    2: {
        "num_players": 2,
        "num_obstacles": 6,
        "enemy_move_probability": 0.10,
        "enemy_shot_error_choices": [-20, -10, 0, 10, 20],
        "enemy_shoot_probability": 0.06,
    },
    3: {
        "num_players": 2,
        "num_obstacles": 8,
        "enemy_move_probability": 0.20,
        "enemy_shot_error_choices": [-16, -8, 0, 8, 16],
        "enemy_shoot_probability": 0.08,
    },
    4: {
        "num_players": 3,
        "num_obstacles": 10,
        "enemy_move_probability": 0.20,
        "enemy_shot_error_choices": [-12, -6, 0, 6, 12],
        "enemy_shoot_probability": 0.10,
    },
    5: {
        "num_players": 4,
        "num_obstacles": 12,
        "enemy_move_probability": 0.20,
        "enemy_shot_error_choices": [-8, -4, 0, 4, 8],
        "enemy_shoot_probability": 0.12,
    },
}

ENEMY_STUCK_MOVE_ATTEMPTS = 2
ENEMY_ESCAPE_FOLLOW_FRAMES = 16
ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES = (90, -90, 180)
SPAWN_Y_OFFSET = 180
SAFE_RADIUS = 100
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5

# Collision / sensing
PROJECTILE_TRAJECTORY_DOT_THRESHOLD = 0.98
OBSTACLE_START_ATTEMPTS = 100
PROJECTILE_HITBOX_SIZE = 10
PROJECTILE_HITBOX_HALF = PROJECTILE_HITBOX_SIZE // 2

# Model and training
HIDDEN_DIMENSIONS = [64, 64]
MODEL_SUBDIR = "_".join(str(size) for size in HIDDEN_DIMENSIONS)
MODEL_DIR = PROJECT_ROOT / "model" / MODEL_SUBDIR
MODEL_NAME = f"bang_{MODEL_SUBDIR}"
MODEL_CHECKPOINT_PATH = str(MODEL_DIR / f"{MODEL_NAME}.pth")
MODEL_BEST_PATH = str(MODEL_DIR / f"{MODEL_NAME}_best.pth")
MODEL_SAVE_RETRIES = 5
MODEL_SAVE_RETRY_DELAY_SECONDS = 0.2

TOTAL_TRAINING_STEPS = 10_000_000
CHECKPOINT_EVERY_STEPS = 200_000

REPLAY_BUFFER_SIZE = 500_000
BATCH_SIZE = 256
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 1e-5
GAMMA = 0.99
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = TOTAL_TRAINING_STEPS
PER_EPSILON = 1e-4

BEST_MODEL_MIN_EPISODES = REWARD_ROLLING_WINDOW
TARGET_SYNC_EVERY = 10_000
GRAD_CLIP_NORM = 10.0

# Replay-first training cadence
LEARN_START_STEPS = 50_000
TRAIN_EVERY_STEPS = 4
GRADIENT_STEPS_PER_UPDATE = 1
EPSILON_DECAY = 0.9999995007

# Reward shaping
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
REWARD_KILL = 2.0
ENGAGEMENT_SCALE = 0.2
ENGAGEMENT_CLIP = 0.1
HAZARD_SCALE = 0.2
HAZARD_CLIP = 0.1
PENALTY_STEP = -0.005
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "event.reward_kill": REWARD_KILL,
    "progress.engagement_scale": ENGAGEMENT_SCALE,
    "progress.engagement_clip": ENGAGEMENT_CLIP,
    "progress.hazard_scale": HAZARD_SCALE,
    "progress.hazard_clip": HAZARD_CLIP,
    "step.penalty_step": PENALTY_STEP,
}

