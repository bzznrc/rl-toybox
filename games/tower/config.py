"""Central configuration for Tower."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_flag, env_int


# RUNTIME
WINDOW_TITLE = "Tower"
FPS = 30
TRAINING_FPS = 0
USE_GPU = env_flag("TOWER_USE_GPU", False)
BASE_SEED = env_int("TOWER_BASE_SEED", 1703)
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT


# ENV
WORLD_WIDTH = screen_width(DEFAULT_GRID_COLUMNS, DEFAULT_TILE_SIZE)
WORLD_HEIGHT = DEFAULT_GRID_ROWS * DEFAULT_TILE_SIZE
SCREEN_WIDTH = WORLD_WIDTH
SCREEN_HEIGHT = screen_height(DEFAULT_GRID_ROWS, DEFAULT_TILE_SIZE, BB_HEIGHT)

MAX_CREDITS_NORMALIZER = 60.0
MAX_LIVES_NORMALIZER = 10.0
MAX_WAVE_COUNT_NORMALIZER = 20.0
DECISION_BUDGET_NORMALIZER = 6.0
WAVE_CLEAR_CREDIT_BONUS = 6
ENEMY_LEAK_DAMAGE = 1

SLOT_NAMES = ("left", "upper", "mid", "lower", "right")
TOWER_KINDS = ("fast", "heavy", "area")
ENEMY_KINDS = ("light", "armored", "flying")
ENTRY_MODES = ("left", "right", "both")


# IO
INPUT_FEATURE_NAMES = [
    "run_gold_norm",
    "run_lives_norm",
    "run_wave_norm",
    "run_actions_left_norm",
    "wave_entry_left",
    "wave_entry_right",
    "wave_count_light_norm",
    "wave_count_armored_norm",
    "wave_count_flying_norm",
    "map_layout_id_norm",
    "slot_left_tower_kind",
    "slot_left_tower_level_norm",
    "slot_upper_tower_kind",
    "slot_upper_tower_level_norm",
    "slot_mid_tower_kind",
    "slot_mid_tower_level_norm",
    "slot_lower_tower_kind",
    "slot_lower_tower_level_norm",
    "slot_right_tower_kind",
    "slot_right_tower_level_norm",
]
ACTION_NAMES = [
    "start_wave",
    "build_fast_left",
    "build_fast_upper",
    "build_fast_mid",
    "build_fast_lower",
    "build_fast_right",
    "build_heavy_left",
    "build_heavy_upper",
    "build_heavy_mid",
    "build_heavy_lower",
    "build_heavy_right",
    "build_area_left",
    "build_area_upper",
    "build_area_mid",
    "build_area_lower",
    "build_area_right",
    "upgrade_left",
    "upgrade_upper",
    "upgrade_mid",
    "upgrade_lower",
    "upgrade_right",
    "sell_left",
    "sell_upper",
    "sell_mid",
    "sell_lower",
    "sell_right",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 60

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 120,
    "check_window": 20,
    "success_threshold": 0.65,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "start_credits": 12,
        "start_lives": 10,
        "num_waves": 6,
        "wave_scale": 1.00,
        "decision_budget": 6,
    },
    2: {
        "start_credits": 12,
        "start_lives": 10,
        "num_waves": 7,
        "wave_scale": 1.10,
        "decision_budget": 6,
    },
    3: {
        "start_credits": 12,
        "start_lives": 10,
        "num_waves": 8,
        "wave_scale": 1.20,
        "decision_budget": 6,
    },
}


# REWARDS
REWARD_PROGRESS_KILL = 0.05
REWARD_EVENT_LEAK = -0.25
REWARD_PROGRESS_WAVE_CLEAR = 0.50
REWARD_TERMINAL_WIN = 2.00
REWARD_TERMINAL_LOSS = -2.00

REWARD_COMPONENT_ORDER = (
    "reward_progress_kill",
    "reward_event_leak",
    "reward_progress_wave_clear",
    "reward_terminal_win",
    "reward_terminal_loss",
)


# TRAINING
HIDDEN_DIMENSIONS = [64, 64]

TOTAL_TRAINING_STEPS = 250_000
LEARN_START_STEPS = 2_000
TRAIN_EVERY_STEPS = 1
UPDATES_PER_TRAIN = 1
CHECKPOINT_EVERY_STEPS = 20_000

LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 0.0
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 80_000
TARGET_SYNC_EVERY = 500
GRAD_CLIP_NORM = 10.0

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 80_000
EPS_BUMP_PATIENCE_EPISODES = 40
EPS_BUMP_MIN_IMPROVEMENT = 0.08
EPS_BUMP_CAP = 0.30
EPS_BUMP_COOLDOWN_STEPS = 4_000
