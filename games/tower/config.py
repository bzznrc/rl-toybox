"""Central configuration for Tower."""

from __future__ import annotations

from core.arcade_style import DEFAULT_BOTTOM_BAR_HEIGHT
from core.utils import env_flag, env_int


# RUNTIME
WINDOW_TITLE = "Tower"
FPS = 30
TRAINING_FPS = 0
USE_GPU = env_flag("TOWER_USE_GPU", False)
BASE_SEED = env_int("TOWER_BASE_SEED", 1703)
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT


# ENV
WORLD_WIDTH = 840
WORLD_HEIGHT = 420
SCREEN_WIDTH = WORLD_WIDTH
SCREEN_HEIGHT = WORLD_HEIGHT + BB_HEIGHT

MAX_GOLD_NORMALIZER = 60.0
MAX_LIVES_NORMALIZER = 10.0
MAX_WAVE_COUNT_NORMALIZER = 20.0
DECISION_BUDGET_NORMALIZER = 6.0
WAVE_CLEAR_GOLD_BONUS = 2
ENEMY_LEAK_DAMAGE = 1

SLOT_NAMES = ("left", "upper", "mid", "lower", "right")
TOWER_KINDS = ("arrow", "cannon", "tesla")
ENEMY_KINDS = ("swarm", "armored", "flying")
ENTRY_MODES = ("left", "right", "both")


# IO
INPUT_FEATURE_NAMES = [
    "run_gold_norm",
    "run_lives_norm",
    "run_wave_norm",
    "run_actions_left_norm",
    "wave_entry_left",
    "wave_entry_right",
    "wave_count_swarm_norm",
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
    "build_arrow_left",
    "build_arrow_upper",
    "build_arrow_mid",
    "build_arrow_lower",
    "build_arrow_right",
    "build_cannon_left",
    "build_cannon_upper",
    "build_cannon_mid",
    "build_cannon_lower",
    "build_cannon_right",
    "build_tesla_left",
    "build_tesla_upper",
    "build_tesla_mid",
    "build_tesla_lower",
    "build_tesla_right",
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
        "start_gold": 18,
        "start_lives": 10,
        "num_waves": 6,
        "wave_scale": 0.98,
        "decision_budget": 6,
    },
    2: {
        "start_gold": 17,
        "start_lives": 10,
        "num_waves": 7,
        "wave_scale": 1.08,
        "decision_budget": 6,
    },
    3: {
        "start_gold": 16,
        "start_lives": 10,
        "num_waves": 8,
        "wave_scale": 1.18,
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
