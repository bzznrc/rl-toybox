"""Central configuration for Kick."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_CELL_INSET,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_float, env_flag


# RUNTIME
WINDOW_TITLE = "Kick"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("KICK_USE_GPU", False)


# ENV
GRID_WIDTH_TILES = DEFAULT_GRID_COLUMNS
GRID_HEIGHT_TILES = DEFAULT_GRID_ROWS
TILE_SIZE = DEFAULT_TILE_SIZE
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
CELL_INSET = DEFAULT_CELL_INSET

GAME_SPEED_SCALE = max(0.2, env_float("KICK_SPEED_SCALE", 0.5))
PHYSICS_DT = 1.0 / FPS
BALL_RADIUS_SCALE = 1.8
PLAYER_V_MAX_PX_PER_SEC = 3.8 * FPS * GAME_SPEED_SCALE
PLAYER_A_MAX_PX_PER_SEC2 = PLAYER_V_MAX_PX_PER_SEC * 4.0

PITCH_LINE_WIDTH = 3
PENALTY_AREA_DEPTH_RATIO = 16.5 / 105.0
PENALTY_AREA_WIDTH_RATIO = 40.3 / 68.0

STAMINA_MIN = 0.5
STAMINA_MAX = 1.0
STAMINA_DRAIN_SECONDS = 5.0
STAMINA_RECOVER_SECONDS = 1.0
GK_HIGH_BYPASS_PROB_DEFAULT = 0.25


# IO
INPUT_FEATURE_NAMES = [
    "self_vx",
    "self_vy",
    "self_theta_cos",
    "self_theta_sin",
    "self_has_ball",
    "self_role",
    "self_stamina",
    "self_stamina_delta",
    "tgt_dx",
    "tgt_dy",
    "tgt_rel_angle_sin",
    "tgt_rel_angle_cos",
    "tgt_dvx",
    "tgt_dvy",
    "tgt_is_free",
    "tgt_owner_team",
    "goal_opp_dx",
    "goal_opp_dy",
    "goal_own_dx",
    "goal_own_dy",
    "ally1_dx",
    "ally1_dy",
    "ally1_dvx",
    "ally1_dvy",
    "ally2_dx",
    "ally2_dy",
    "ally2_dvx",
    "ally2_dvy",
    "foe1_dx",
    "foe1_dy",
    "foe1_dvx",
    "foe1_dvy",
    "foe2_dx",
    "foe2_dy",
    "foe2_dvx",
    "foe2_dvy",
]
ACTION_NAMES = [
    "stay",
    "move_n",
    "move_ne",
    "move_e",
    "move_se",
    "move_s",
    "move_sw",
    "move_w",
    "move_nw",
    "kick_low",
    "kick_mid",
    "kick_high",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 250,
    "check_window": 25,
    "success_threshold": 0.40,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "players_per_team": 3,
        "active_roles": ["GK", "LCM", "ST1"],
        "goals_size_scale": 2.0,
        "enemy_stamina_scale": 0.50,
        "enemy_shot_error_choices": [-30, 0, 30],
        "entropy_coef": 0.25,
        "kickoff": "GK",
    },
    2: {
        "players_per_team": 7,
        "active_roles": ["GK", "LB", "RB", "LCM", "RCM", "ST1", "ST2"],
        "goals_size_scale": 1.5,
        "enemy_stamina_scale": 0.75,
        "enemy_shot_error_choices": [-20, 0, 20],
        "entropy_coef": 0.015,
        "kickoff": "GK",
    },
    3: {
        "players_per_team": 11,
        "active_roles": ["GK", "LB", "LCB", "RCB", "RB", "LM", "LCM", "RCM", "RM", "ST1", "ST2"],
        "goals_size_scale": 1.0,
        "enemy_stamina_scale": 1.0,
        "enemy_shot_error_choices": [-10, 0, 10],
        "entropy_coef": 0.01,
        "kickoff": "CC",
    },
}


# REWARDS
REWARD_SCORE = 10.0
PENALTY_CONCEDE = -5.0
PENALTY_TURNOVER = -0.25
REWARD_PROGRESS = 5.0
PENALTY_ZONE = -0.0005

REWARD_COMPONENTS = {
    "outcome.reward_score": REWARD_SCORE,
    "outcome.penalty_concede": PENALTY_CONCEDE,
    "event.penalty_turnover": PENALTY_TURNOVER,
    "progress.reward_progress": REWARD_PROGRESS,
    "event.penalty_zone": PENALTY_ZONE,
}


# TRAINING
HIDDEN_DIMENSIONS = [96, 96]

MAX_TRAINING_ITERATIONS = 4500
ROLLOUT_STEPS = 2048
CHECKPOINT_EVERY_ITERATIONS = 10

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 512
ENTROPY_COEF = 0.025
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
