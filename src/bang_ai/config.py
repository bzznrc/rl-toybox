"""Central configuration for Bang AI."""

from __future__ import annotations

from bang_ai.utils import PROJECT_ROOT, env_flag

BOARD_COLUMNS = 48
BOARD_ROWS = 32
BOARD_CELL_SIZE_PX = 20
BOARD_BOTTOM_BAR_HEIGHT_PX = 30
BOARD_CELL_INSET_PX = 4

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

# Colors
COLOR_AQUA = (102, 212, 200)
COLOR_DEEP_TEAL = (38, 110, 105)
COLOR_CORAL = (244, 137, 120)
COLOR_BRICK_RED = (150, 62, 54)
COLOR_SLATE_GRAY = (97, 101, 107)
COLOR_FOG_GRAY = (230, 231, 235)
COLOR_CHARCOAL = (28, 30, 36)
COLOR_NEAR_BLACK = (18, 18, 22)
COLOR_SOFT_WHITE = (238, 238, 242)
COLOR_AMBER = (255, 224, 130)

# P3 and P4
COLOR_P3_BLUE = (66, 133, 244)        # Google Blue 500
COLOR_P3_NAVY = (26, 92, 173)         # Deep Blue
COLOR_P4_PURPLE = (171, 71, 188)      # Material Purple 500
COLOR_P4_DEEP_PURPLE = (123, 31, 162) # Deep Purple 700

# Input/output spaces
INPUT_FEATURE_NAMES = [
    "enemy_distance",
    "enemy_in_los",
    "enemy_relative_angle_sin",
    "enemy_relative_angle_cos",
    "delta_enemy_distance",
    "delta_enemy_relative_angle",
    "nearest_projectile_distance",
    "nearest_projectile_relative_angle_sin",
    "nearest_projectile_relative_angle_cos",
    "delta_projectile_distance",
    "in_projectile_trajectory",
    "time_since_last_shot",
    "time_since_last_seen_enemy",
    "time_since_last_projectile_seen",
    "up_blocked",
    "down_blocked",
    "left_blocked",
    "right_blocked",
    "player_angle_sin",
    "player_angle_cos",
    "move_intent_x",
    "move_intent_y",
    "aim_intent",
    "last_action_index",
]
ACTION_NAMES = [
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "stop_move",
    "aim_left",
    "aim_right",
    "shoot",
]
NUM_INPUT_FEATURES = len(INPUT_FEATURE_NAMES)
NUM_ACTIONS = len(ACTION_NAMES)
STATE_SIZE = NUM_INPUT_FEATURES
ACTION_SIZE = NUM_ACTIONS
MODEL_INPUT_SIZE = NUM_INPUT_FEATURES
MODEL_OUTPUT_SIZE = NUM_ACTIONS

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
PROJECTILE_DISTANCE_MISSING = -1.0

# Model and training
HIDDEN_DIMENSIONS = [64, 48]
MODEL_SUBDIR = "_".join(str(size) for size in HIDDEN_DIMENSIONS)
MODEL_DIR = PROJECT_ROOT / "model" / MODEL_SUBDIR
MODEL_NAME = f"bang_{MODEL_SUBDIR}"
MODEL_CHECKPOINT_PATH = str(MODEL_DIR / f"{MODEL_NAME}.pth")
MODEL_BEST_PATH = str(MODEL_DIR / f"{MODEL_NAME}_best.pth")
MODEL_SAVE_RETRIES = 5
MODEL_SAVE_RETRY_DELAY_SECONDS = 0.2

TOTAL_TRAINING_STEPS = 40_000_000
CHECKPOINT_EVERY_STEPS = 100_000

REPLAY_BUFFER_SIZE = 150_000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GAMMA = 0.98
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = TOTAL_TRAINING_STEPS
PER_EPSILON = 1e-4

EPSILON_START_SCRATCH = 1.0
EPSILON_START_RESUME = 0.25
EPSILON_MIN = 0.05
EPSILON_DECAY_EPISODES = 10_000
EPSILON_STAGNATION_BOOST = 0.05
EPSILON_EXPLORATION_CAP = 0.25
EPSILON_LEVEL_UP_RESET = 0.25
STAGNATION_WINDOW = REWARD_ROLLING_WINDOW
PATIENCE = 25
STAGNATION_IMPROVEMENT_THRESHOLD = 0.05
EPISODE_CHECKPOINT_EVERY = 50
BEST_MODEL_MIN_EPISODES = REWARD_ROLLING_WINDOW
TARGET_SYNC_EVERY = 500
GRAD_CLIP_NORM = 10.0

# Replay-first training cadence
LEARN_START_STEPS = 5_000
TRAIN_EVERY_STEPS = 4
GRADIENT_STEPS_PER_UPDATE = 1

# Reward shaping
REWARD_WIN = 20.0
PENALTY_LOSE = -10.0
REWARD_HIT_ENEMY = 5.0
PENALTY_TIME_STEP = -0.005
PENALTY_BAD_SHOT = -0.1
PENALTY_BLOCKED_MOVE = -0.1
REWARD_COMPONENTS = {
    "time_step": PENALTY_TIME_STEP,
    "bad_shot": PENALTY_BAD_SHOT,
    "blocked_move": PENALTY_BLOCKED_MOVE,
    "hit_enemy": REWARD_HIT_ENEMY,
    "win": REWARD_WIN,
    "lose": PENALTY_LOSE,
}
