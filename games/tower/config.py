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
MAX_LIVES_NORMALIZER = 12.0
MAX_WAVE_COUNT_NORMALIZER = 12.0
WAVE_CLEAR_CREDIT_BONUS = 5
ENEMY_LEAK_DAMAGE = 1

SLOT_NAMES = ("slot_0", "slot_1", "slot_2", "slot_3", "slot_4")
TOWER_KINDS = ("fast", "heavy", "area")
ENEMY_KINDS = ("light", "armored", "flying")


# IO
INPUT_FEATURE_NAMES = [
    "glob_gold_norm",
    "glob_lives_norm",
    "glob_wave_norm",
    "glob_acts_left_norm",
    "wave_n_light_norm",
    "wave_n_armored_norm",
    "wave_n_flying_norm",
    "route_shortcut_upper_active",
    "route_shortcut_lower_active",
    "slot_0_kind_id",
    "slot_0_lvl_norm",
    "slot_0_exposure_norm",
    "slot_1_kind_id",
    "slot_1_lvl_norm",
    "slot_1_exposure_norm",
    "slot_2_kind_id",
    "slot_2_lvl_norm",
    "slot_2_exposure_norm",
    "slot_3_kind_id",
    "slot_3_lvl_norm",
    "slot_3_exposure_norm",
    "slot_4_kind_id",
    "slot_4_lvl_norm",
    "slot_4_exposure_norm",
]
ACTION_NAMES = [
    "start_wave",
    "build_fast_0",
    "build_fast_1",
    "build_fast_2",
    "build_fast_3",
    "build_fast_4",
    "build_heavy_0",
    "build_heavy_1",
    "build_heavy_2",
    "build_heavy_3",
    "build_heavy_4",
    "build_area_0",
    "build_area_1",
    "build_area_2",
    "build_area_3",
    "build_area_4",
    "upgrade_0",
    "upgrade_1",
    "upgrade_2",
    "upgrade_3",
    "upgrade_4",
    "sell_0",
    "sell_1",
    "sell_2",
    "sell_3",
    "sell_4",
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


# BALANCE
BUILD_COST = 5
UPGRADE_COST_BY_LEVEL = {1: 5, 2: 5}
SELL_VALUE_BY_LEVEL = {1: 4, 2: 8, 3: 12}

TOWER_LEVEL_STATS = {
    "fast": (
        {"damage": 1.00, "cooldown_ticks": 12, "attack_range": 136.0},
        {"damage": 1.20, "cooldown_ticks": 10,  "attack_range": 142.0},
        {"damage": 1.40, "cooldown_ticks": 8,  "attack_range": 148.0},
    ),
    "heavy": (
        {"damage": 3.60, "cooldown_ticks": 32, "attack_range": 168.0},
        {"damage": 4.80, "cooldown_ticks": 30, "attack_range": 176.0},
        {"damage": 6.00, "cooldown_ticks": 28, "attack_range": 184.0},
    ),
    "area": (
        {"damage": 0.60, "cooldown_ticks": 22, "attack_range": 150.0, "splash_radius": 24.0},
        {"damage": 0.80, "cooldown_ticks": 20, "attack_range": 156.0, "splash_radius": 32.0},
        {"damage": 1.00, "cooldown_ticks": 18, "attack_range": 162.0, "splash_radius": 40.0},
    ),
}

TOWER_MATCHUP_MULTIPLIERS = {
    "fast":  {"light": 1.00, "armored": 0.60, "flying": 1.60},
    "heavy": {"light": 0.80, "armored": 1.40, "flying": 0.40},
    "area":  {"light": 1.20, "armored": 0.80, "flying": 0.00},  # cannot hit flying
}

ENEMY_STATS = {
    "light":   {"max_hp": 2.8, "speed": 3.60, "armor": 0.00, "bounty": 0, "radius": 8.0,  "spawn_gap": 8},
    "armored": {"max_hp": 9.0, "speed": 1.80, "armor": 0.40, "bounty": 0, "radius": 10.0, "spawn_gap": 12},
    "flying":  {"max_hp": 3.6, "speed": 4.00, "armor": 0.20, "bounty": 0, "radius": 9.0,  "spawn_gap": 10},
}

AUTHORED_WAVE_TEMPLATES = (
    {"light": 6,  "armored": 0, "flying": 0},
    {"light": 6,  "armored": 0, "flying": 2},
    {"light": 6,  "armored": 2, "flying": 2},
    {"light": 8,  "armored": 2, "flying": 4},
    {"light": 8,  "armored": 4, "flying": 4},
    {"light": 10, "armored": 4, "flying": 6},
    {"light": 10, "armored": 6, "flying": 6},
    {"light": 12, "armored": 6, "flying": 8},
    {"light": 12, "armored": 8, "flying": 10},
)

WAVE_ROUTE_MODES = (
    "none",
    "upper",
    "lower",
    "none",
    "upper",
    "lower",
    "none",
    "upper",
    "lower",
)

SPAWN_TYPE_ORDER = ("light", "flying", "armored")

LEVEL_SETTINGS = {
    1: {
        "start_credits": 12,
        "start_lives": 12,
        "num_waves": 5,
    },
    2: {
        "start_credits": 12,
        "start_lives": 12,
        "num_waves": 7,
    },
    3: {
        "start_credits": 12,
        "start_lives": 12,
        "num_waves": 9,
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
