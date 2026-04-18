"""Central configuration for Fuse."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_CELL_INSET as CELL_INSET,
    DEFAULT_GRID_COLUMNS as BOARD_WIDTH_TILES,
    DEFAULT_GRID_ROWS as BOARD_HEIGHT_TILES,
    DEFAULT_TILE_SIZE as TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Fuse"
FPS = 15
TRAINING_FPS = 0
USE_GPU = env_flag("FUSE_USE_GPU", False)


# ENV
SCREEN_WIDTH = screen_width(BOARD_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(BOARD_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
MAX_BOMBS_PER_PLAYER = 1
BOMB_FUSE_STEPS = FPS * 2
BLAST_RADIUS_TILES = 3
EXPLOSION_LIFETIME_STEPS = 1
MAX_EPISODE_STEPS = 260
SENSE_RANGE_TILES = 4
SAFE_SEARCH_HORIZON_STEPS = 7
SAFE_SPACE_NORMALIZER = 10.0
ESCAPE_ROUTE_NORMALIZER = 4.0
WIN_HISTORY_LIMIT = 12


# IO
INPUT_FEATURE_NAMES = [
    # SELF
    "self_bombs_norm",
    "self_bomb_cd_norm",
    "self_on_bomb",
    "self_can_place_bomb",
    # SENS
    "sens_free_up_norm",
    "sens_free_down_norm",
    "sens_free_left_norm",
    "sens_free_right_norm",
    "sens_box_up_norm",
    "sens_box_down_norm",
    "sens_box_left_norm",
    "sens_box_right_norm",
    # OPP
    "opp1_dx",
    "opp1_dy",
    "opp1_same_row",
    "opp1_same_col",
    "opp2_dx",
    "opp2_dy",
    "opp2_same_row",
    "opp2_same_col",
    "opp3_dx",
    "opp3_dy",
    "opp3_same_row",
    "opp3_same_col",
    # MAP / MEM
    "map_safe_up_norm",
    "map_safe_down_norm",
    "map_safe_left_norm",
    "map_safe_right_norm",
    # HAZ
    "haz_here_tti_norm",
    "haz_post_bomb_escape_norm",
    # FLAG
    "flag_bomb_value_norm",
    "flag_can_hit_opp_now",
    "flag_crates_left_norm",
    "flag_time_norm",
]
ACTION_NAMES = [
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_stop",
    "place_bomb",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
if OBS_DIM != 34:
    raise RuntimeError(f"Fuse INPUT_FEATURE_NAMES expected 34 entries, got {OBS_DIM}.")

ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_MOVE_STOP = 4
ACTION_PLACE_BOMB = 5


# MODEL
DEFAULT_HIDDEN_SIZES = (64, 64)


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 250,
    "check_window": 25,
    "success_threshold": 0.45,
    "consecutive_checks_required": 2,
}
LEVEL_SETTINGS = {
    1: {
        "num_players": 2,
        "crate_density": 0.04,
        "max_episode_steps": 320,
        "bot_safety_weight": 1.15,
        "bot_bomb_weight": 0.65,
        "bot_trap_weight": 0.20,
        "bot_chase_weight": 0.15,
        "bot_random_action_prob": 0.0,
    },
    2: {
        "num_players": 3,
        "crate_density": 0.055,
        "max_episode_steps": 360,
        "bot_safety_weight": 1.35,
        "bot_bomb_weight": 0.85,
        "bot_trap_weight": 0.45,
        "bot_chase_weight": 0.30,
        "bot_random_action_prob": 0.0,
    },
    3: {
        "num_players": 4,
        "crate_density": 0.07,
        "max_episode_steps": 400,
        "bot_safety_weight": 1.55,
        "bot_bomb_weight": 1.00,
        "bot_trap_weight": 0.75,
        "bot_chase_weight": 0.45,
        "bot_random_action_prob": 0.0,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -8.0
REWARD_ELIM = 1.5
REWARD_CRATE = 0.05
PENALTY_STEP = -0.0025
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "event.reward_elim": REWARD_ELIM,
    "event.reward_crate": REWARD_CRATE,
    "step.penalty_step": PENALTY_STEP,
}
