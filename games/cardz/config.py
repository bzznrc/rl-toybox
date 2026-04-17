"""Central configuration for Cardz."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT,
    GAME_TITLE_FONT_NAME,
    GAME_UI_FONT_NAME,
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
TURN_RESOLUTION_PAUSE_SECONDS = 2.5
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT


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
CARD_IDS = {
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
HAND_COUNT_NORMALIZER = float(MAX_HAND_SIZE)

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

UI_FONT_NAME = GAME_UI_FONT_NAME
TITLE_FONT_NAME = GAME_TITLE_FONT_NAME


# IO
# Structured public-information P1 view:
# - global match state
# - explicit one-hot phase flags
# - per-lane public board state for both players
# - explicit one-hot P1 hand slots only
PHASE_FEATURE_NAMES = (
    "phase_open",
    "phase_resp",
    "phase_free",
)
HAND_CARD_FLAG_SUFFIXES = (
    "empty",
    "u1",
    "u2",
    "u3",
    "atk",
    "ban",
)
HAND_CARD_FLAG_BY_KEY = {
    None: "empty",
    "U1": "u1",
    "U2": "u2",
    "U3": "u3",
    "Atk": "atk",
    "Ban": "ban",
}
INPUT_FEATURE_NAMES = [
    "glob_turn_norm",
    "glob_energy_p1_norm",
    "glob_energy_p2_norm",
    "glob_score_p1_norm",
    "glob_score_p2_norm",
    "glob_hand_count_p1_norm",
    "glob_hand_count_p2_norm",
    *PHASE_FEATURE_NAMES,
    *[
        feature_name
        for lane in range(NUM_LANES)
        for feature_name in (
            f"lane_{lane}_power_p1_norm",
            f"lane_{lane}_power_p2_norm",
            f"lane_{lane}_unit_count_p1_norm",
            f"lane_{lane}_unit_count_p2_norm",
        )
    ],
    *[
        feature_name
        for lane in range(NUM_LANES)
        for feature_name in (
            f"lane_{lane}_p1_has_ban",
            f"lane_{lane}_p1_has_atk",
            f"lane_{lane}_p2_has_ban",
            f"lane_{lane}_p2_has_atk",
        )
    ],
    *[
        f"hand_{slot}_{suffix}"
        for slot in range(MAX_HAND_SIZE)
        for suffix in HAND_CARD_FLAG_SUFFIXES
    ],
]
ACTION_NAMES = [
    f"play_hand_{slot}_lane_{lane}"
    for slot in range(MAX_HAND_SIZE)
    for lane in range(NUM_LANES)
] + ["pass"]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
PASS_ACTION_INDEX = ACT_DIM - 1
if OBS_DIM != 64:
    raise RuntimeError(f"Cardz INPUT_FEATURE_NAMES expected 64 entries, got {OBS_DIM}.")


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW
CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 200,
    "check_window": 25,
    "success_threshold": 0.60,
    "consecutive_checks_required": 2,
}
LEVEL_SETTINGS = {
    1: {
        "entropy_coef": 0.015,
        "opp_max_hand": 3,
        "opp_random_move_prob": 0.75,
    },
    2: {
        "entropy_coef": 0.010,
        "opp_max_hand": 4,
        "opp_random_move_prob": 0.35,
    },
    3: {
        "entropy_coef": 0.005,
        "opp_max_hand": 5,
        "opp_random_move_prob": 0.10,
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
