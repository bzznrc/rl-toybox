"""Central configuration for Cardz."""

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
WINDOW_TITLE = "Cardz"
FPS = 18
TRAINING_FPS = 0
USE_GPU = env_flag("CARDZ_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = True
BASE_SEED = env_int("CARDZ_BASE_SEED", 2903)
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT + 14


# ENV
WORLD_WIDTH = screen_width(DEFAULT_GRID_COLUMNS, DEFAULT_TILE_SIZE)
WORLD_HEIGHT = DEFAULT_GRID_ROWS * DEFAULT_TILE_SIZE
SCREEN_WIDTH = WORLD_WIDTH
SCREEN_HEIGHT = screen_height(DEFAULT_GRID_ROWS, DEFAULT_TILE_SIZE, BB_HEIGHT)

PLAYER_COUNT = 2
NUM_LANES = 3
MAX_TURNS = 5
MAX_HAND_SIZE = 5
STARTING_HAND_SIZE = 5
MAX_UNITS_PER_LANE = 2

CARD_DRAW_ORDER = ("U1", "U2", "U3", "Atk", "Ban")
CARD_DRAW_WEIGHTS = (0.30, 0.20, 0.15, 0.175, 0.175)
CARD_TYPE_IDS = {
    "U1": 1.0,
    "U2": 2.0,
    "U3": 3.0,
    "Atk": 4.0,
    "Ban": 5.0,
}
CARD_COSTS = {
    "U1": 1,
    "U2": 2,
    "U3": 3,
    "Atk": 1,
    "Ban": 1,
}
CARD_POWERS = {
    "U1": 2,
    "U2": 3,
    "U3": 4,
    "Atk": 0,
    "Ban": 1,
}
CARD_VALUES = {
    "U1": 2,
    "U2": 3,
    "U3": 4,
    "Atk": 2,
    "Ban": 1,
}
CARD_KINDS = {
    "U1": "unit",
    "U2": "unit",
    "U3": "unit",
    "Atk": "tactic",
    "Ban": "banner",
}
ATK_DELTA = 2
BAN_POWER = 1

TURN_NORMALIZER = float(MAX_TURNS)
ENERGY_NORMALIZER = float(MAX_TURNS)
MATCH_SCORE_NORMALIZER = float(NUM_LANES * MAX_TURNS)
LANE_POWER_NORMALIZER = 18.0
LANE_COUNT_NORMALIZER = float(MAX_UNITS_PER_LANE)
CARD_COST_NORMALIZER = 3.0
CARD_VALUE_NORMALIZER = 4.0

SCORE_TRACK_SLOTS = 4
SCORE_TRACK_GAP = 6.0
SCORE_TRACK_HEIGHT = 24.0
LANE_SCORE_ICON_SIZE = 18.0
LANE_SCORE_ICON_INSET = 4.0

LANE_TOP = 48.0
LANE_WIDTH = 280.0
LANE_HEIGHT = 420.0
LANE_GAP = 18.0
LANE_INSET = 4.0

HAND_CARD_WIDTH = 116.0
HAND_CARD_HEIGHT = 78.0
HAND_CARD_GAP = 12.0

UNIT_CARD_WIDTH = HAND_CARD_WIDTH
UNIT_CARD_HEIGHT = HAND_CARD_HEIGHT
UNIT_CARD_GAP = HAND_CARD_GAP
BANNER_MARKER_SIZE = 18.0
BANNER_MARKER_GAP = 10.0
HAND_TOP = WORLD_HEIGHT - 118.0

UI_FONT_NAME = ("Roboto-Light", "Roboto Light", "Roboto", "Arial", "sans-serif")
TITLE_FONT_NAME = ("Roboto-Bold", "Roboto Bold", "Roboto", "Arial", "sans-serif")


# IO
INPUT_FEATURE_NAMES = [
    "turn_norm",
    "energy_self_norm",
    "energy_opp_norm",
    "score_self_norm",
    "score_opp_norm",
    "lane_0_power_self_norm",
    "lane_0_power_opp_norm",
    "lane_0_count_self_norm",
    "lane_0_count_opp_norm",
    "lane_1_power_self_norm",
    "lane_1_power_opp_norm",
    "lane_1_count_self_norm",
    "lane_1_count_opp_norm",
    "lane_2_power_self_norm",
    "lane_2_power_opp_norm",
    "lane_2_count_self_norm",
    "lane_2_count_opp_norm",
    "hand_0_type",
    "hand_0_cost_norm",
    "hand_0_value_norm",
    "hand_1_type",
    "hand_1_cost_norm",
    "hand_1_value_norm",
    "hand_2_type",
    "hand_2_cost_norm",
    "hand_2_value_norm",
    "hand_3_type",
    "hand_3_cost_norm",
    "hand_3_value_norm",
    "hand_4_type",
    "hand_4_cost_norm",
    "hand_4_value_norm",
]
ACTION_NAMES = [
    f"play_hand_{slot}_lane_{lane}"
    for slot in range(MAX_HAND_SIZE)
    for lane in range(NUM_LANES)
] + ["pass"]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
PASS_ACTION_INDEX = ACT_DIM - 1


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 1
REWARD_ROLLING_WINDOW = 60
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW
LEVEL_SETTINGS = {
    1: {
        "entropy_coef": 0.015,
    },
}


# REWARDS
REWARD_PROGRESS_TURN_POINTS_PER_LANE = 0.2
REWARD_TERMINAL_MATCH_WIN = 1.0
REWARD_TERMINAL_MATCH_DRAW = 0.0
REWARD_TERMINAL_MATCH_LOSS = -1.0
REWARD_COMPONENT_NAMES = (
    "reward_progress_turn_points",
    "reward_terminal_match_win",
    "reward_terminal_match_draw",
    "reward_terminal_match_loss",
)


# TRAINING
HIDDEN_DIMENSIONS = [96, 96]
SHARE_BACKBONE = True

MAX_TRAINING_ITERATIONS = 8000
ROLLOUT_STEPS = 1024
CHECKPOINT_EVERY_ITERATIONS = 10

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
