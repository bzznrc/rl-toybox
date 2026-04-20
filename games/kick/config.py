"""Central configuration for Kick."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_CELL_INSET as CELL_INSET,
    DEFAULT_GRID_COLUMNS as GRID_WIDTH_TILES,
    DEFAULT_GRID_ROWS as GRID_HEIGHT_TILES,
    DEFAULT_TILE_SIZE as TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_float, env_flag


# RUNTIME
WINDOW_TITLE = "Kick"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("KICK_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)

GAME_SPEED_SCALE = max(0.2, env_float("KICK_SPEED_SCALE", 0.5))
PHYSICS_DT = 1.0 / FPS
BALL_RADIUS_SCALE = 1.8
PLAYER_V_MAX_PX_PER_SEC = 3.8 * FPS * GAME_SPEED_SCALE
PLAYER_A_MAX_PX_PER_SEC2 = PLAYER_V_MAX_PX_PER_SEC * 4.0
SHOW_ZONE_TARGET_CLONES = False
SHOW_BOTTOM_REWARD_BREAKDOWN = True
ZONE_TARGET_CLONE_ALPHA = 128
DEBUG_SANITY_CHECKS = env_flag("KICK_DEBUG_SANITY", False)

PITCH_LINE_WIDTH = 3
PENALTY_AREA_DEPTH_RATIO = 16.5 / 105.0
PENALTY_AREA_WIDTH_RATIO = 40.3 / 68.0

STAMINA_MIN = 0.5
STAMINA_MAX = 1.0
STAMINA_DRAIN_SECONDS = 5.0
STAMINA_RECOVER_SECONDS = 1.0


# ROLES
ROLE_NAMES = ["GK", "LB", "RB", "LM", "CM", "RM", "CS"]
ROLE_FEATURE_NAME_BY_ROLE = {
    "GK": "self_role_gk",
    "LB": "self_role_lb",
    "RB": "self_role_rb",
    "LM": "self_role_lm",
    "CM": "self_role_lcm",
    "RM": "self_role_rm",
    "CS": "self_role_lcs",
}
OBS_NEAREST_OUTFIELD_PLAYERS = 3
BASE_ROLE_ATTACK_SHIFT_TILES_BY_ROLE = {
    "GK": 0.5,
    "LB": 2.0,
    "RB": 2.0,
    "LM": 3.0,
    "CM": 3.0,
    "RM": 3.0,
    "CS": 4.0,
}
# Single tweak point for how far the attacking shape pushes beyond the
# defensive home positions. The derived per-role map below is used by both
# the scripted team logic and the RL anchor targets.
ATTACKING_ANCHOR_SHIFT_SCALE = 2.0
ROLE_ATTACK_SHIFT_TILES_BY_ROLE = {
    role: float(base_shift) * float(ATTACKING_ANCHOR_SHIFT_SCALE)
    for role, base_shift in BASE_ROLE_ATTACK_SHIFT_TILES_BY_ROLE.items()
}


# IO
INPUT_FEATURE_NAMES = [
    "self_x_norm",
    "self_y_norm",
    "self_vx",
    "self_vy",
    "self_theta_cos",
    "self_theta_sin",
    "self_has_ball",
    "self_stamina",
    "self_stamina_delta",
    *[ROLE_FEATURE_NAME_BY_ROLE[role] for role in ROLE_NAMES],
    "tgt_dx",
    "tgt_dy",
    "tgt_dist_norm",
    "tgt_rel_ang_sin",
    "tgt_rel_ang_cos",
    "tgt_dvx",
    "land_opp_goal_dx",
    "land_opp_goal_dy",
    "land_own_goal_dx",
    "land_own_goal_dy",
    "land_gk_dx",
    "land_gk_dy",
    "land_gk_dvy",
    "ally1_dx",
    "ally1_dy",
    "ally1_dvx",
    "ally1_dvy",
    "ally2_dx",
    "ally2_dy",
    "ally2_dvx",
    "ally2_dvy",
    "ally3_dx",
    "ally3_dy",
    "ally3_dvx",
    "ally3_dvy",
    "opp1_dx",
    "opp1_dy",
    "opp1_dvx",
    "opp1_dvy",
    "opp2_dx",
    "opp2_dy",
    "opp2_dvx",
    "opp2_dvy",
    "opp3_dx",
    "opp3_dy",
    "opp3_dvx",
    "opp3_dvy",
    "map_anchor_dx",
    "map_anchor_dy",
    "flag_shoot_mode",
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
if OBS_DIM != 56:
    raise RuntimeError(f"Kick INPUT_FEATURE_NAMES expected 56 entries, got {OBS_DIM}.")


# GAME
DEFAULT_ALGO = "ppo"


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 5
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
        "players_opponent": 1,
        "opponent_roles": ["GK"],
        "start_possession": "RND_LEFT",
        "goals_size_scale": 2.5,
        "enemy_stamina_scale": 0.50,
        "entropy_coef": 0.02,
    },
    2: {
        "players_opponent": 3,
        "opponent_roles": ["GK", "LM", "RM"],
        "start_possession": "RND_LEFT",
        "goals_size_scale": 2.0,
        "enemy_stamina_scale": 0.50,
        "entropy_coef": 0.015,
    },
    3: {
        "players_opponent": 5,
        "opponent_roles": ["GK", "LB", "RB", "CM", "CS"],
        "start_possession": "CEN",
        "goals_size_scale": 1.5,
        "enemy_stamina_scale": 0.75,
        "entropy_coef": 0.01,
    },
    4: {
        "players_opponent": 7,
        "opponent_roles": list(ROLE_NAMES),
        "start_possession": "CEN",
        "goals_size_scale": 1.25,
        "enemy_stamina_scale": 0.75,
        "entropy_coef": 0.01,
    },
    5: {
        "players_opponent": 7,
        "opponent_roles": list(ROLE_NAMES),
        "start_possession": "CEN",
        "goals_size_scale": 1.0,
        "enemy_stamina_scale": 1.00,
        "entropy_coef": 0.005,
    },
}


# REWARDS
REWARD_SCORE = 10.0
PENALTY_CONCEDE = -5.0
PENALTY_TURNOVER = -0.25
REWARD_PASS = 0.25
REWARD_PROGRESS = 2.0
B_SCALE = 0.02
B_CLIP = 0.05
# Keep one shared safe-zone ellipse for all outfield players so tuning stays
# simple and affects both scripted behavior and RL shaping consistently.
ZONE_TOL_X = 0.32
ZONE_TOL_Y = 0.16
ZONE_TOL_X_GK = 0.05
ZONE_TOL_Y_GK = 0.05
ZONE_PENALTY_LINEAR_COEF = 0.0005
ZONE_PENALTY_QUADRATIC_COEF = 0.004

REWARD_COMPONENTS = {
    "outcome.reward_score": REWARD_SCORE,
    "outcome.penalty_concede": PENALTY_CONCEDE,
    "event.penalty_turnover": PENALTY_TURNOVER,
    "event.reward_pass": REWARD_PASS,
    "progress.reward_progress": REWARD_PROGRESS,
    "event.reward_ball_approach": B_SCALE * B_CLIP,
    "event.penalty_zone": -(ZONE_PENALTY_LINEAR_COEF + ZONE_PENALTY_QUADRATIC_COEF),
}


# CENTRAL OBS
MAX_LEFT_PLAYERS = len(ROLE_NAMES)
CENTRAL_OBS_MASK_DIM = MAX_LEFT_PLAYERS
CENTRAL_OBS_BALL_FEATURES = 6
CENTRAL_OBS_DIM = (MAX_LEFT_PLAYERS * OBS_DIM) + CENTRAL_OBS_MASK_DIM + CENTRAL_OBS_BALL_FEATURES
if MAX_LEFT_PLAYERS != 7:
    raise RuntimeError(f"Kick MAX_LEFT_PLAYERS expected 7, got {MAX_LEFT_PLAYERS}.")
if CENTRAL_OBS_DIM != 405:
    raise RuntimeError(f"Kick CENTRAL_OBS_DIM expected 405, got {CENTRAL_OBS_DIM}.")

GAME_CAPABILITIES = {
    "centralized_critic_required": True,
}
ENV_METADATA = {
    "central_obs_dim": int(CENTRAL_OBS_DIM),
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [96, 96],
    "critic_hidden_sizes": [192, 192],
}
ALGO_CONFIG_OVERRIDES = {
    "ppo": {
    "minibatch_size": 512,
    "entropy_coef": float(LEVEL_SETTINGS[int(MIN_LEVEL)]["entropy_coef"]),
    }
}
DEFAULT_TRAIN_CONFIG = {
    "budget": 12_000_000,
    "rollout_steps": 2_048,
    "checkpoint_every": 10,
}
