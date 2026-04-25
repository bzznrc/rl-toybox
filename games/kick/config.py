"""Central configuration for Kick."""

from __future__ import annotations

from core.shared_config import (
    FPS,
    TILE_SIZE,
)
from core.utils import env_float, env_flag


# RUNTIME
WINDOW_TITLE = "Kick"
USE_GPU = env_flag("KICK_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
GAME_SPEED_SCALE = max(0.2, env_float("KICK_SPEED_SCALE", 0.5))
BALL_RADIUS_SCALE = 1.8
PLAYER_V_MAX_PX_PER_SEC = 3.8 * FPS * GAME_SPEED_SCALE
PLAYER_A_MAX_PX_PER_SEC2 = PLAYER_V_MAX_PX_PER_SEC * 4.0
SHOW_ZONE_TARGET_CLONES = False
SHOW_GHOST_OVERLAY = SHOW_ZONE_TARGET_CLONES
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
# Keep the goalkeeper's ghost/role-zone target fixed just outside the goal
# itself. A zero margin makes the keeper zone graze the goal edge.
GOALKEEPER_STATIC_ANCHOR_OUTSIDE_GOAL_MARGIN_TILES = 0.0


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
    "self_last_move_x",
    "self_last_move_y",
    "self_action_changed",
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
    "land_own_gk_dx",
    "land_own_gk_dy",
    "land_shot_line_dy",
    "land_shot_tti",
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
    "flag_shot_quality",
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
    "pass",
    "shoot",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)
if OBS_DIM != 63:
    raise RuntimeError(f"Kick INPUT_FEATURE_NAMES expected 63 entries, got {OBS_DIM}.")
if ACT_DIM != 11:
    raise RuntimeError(f"Kick ACTION_NAMES expected 11 entries, got {ACT_DIM}.")


# GAME
DEFAULT_ALGO = "ppo"


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 5
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 100,
    "success_threshold": 0.60,
}

LEVEL_SETTINGS = {
    1: {
        "players_opponent": 1,
        "opponent_roles": ["GK"],
        "start_possession": "RND_LEFT",
        "goals_size_scale": 3.0,
        "enemy_stamina_scale": 0.25,
        "entropy_coef": 0.02,
    },
    2: {
        "players_opponent": 3,
        "opponent_roles": ["GK", "LM", "RM"],
        "start_possession": "RND_LEFT",
        "goals_size_scale": 2.25,
        "enemy_stamina_scale": 0.50,
        "entropy_coef": 0.015,
    },
    3: {
        "players_opponent": 5,
        "opponent_roles": ["GK", "LB", "RB", "CM", "CS"],
        "start_possession": "CEN",
        "goals_size_scale": 1.75,
        "enemy_stamina_scale": 0.625,
        "entropy_coef": 0.01,
    },
    4: {
        "players_opponent": 7,
        "opponent_roles": list(ROLE_NAMES),
        "start_possession": "CEN",
        "goals_size_scale": 1.25,
        "enemy_stamina_scale": 0.75,
        "entropy_coef": 0.0075,
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
REWARD_PROGRESS_CONTROLLED = 1.0
CONTROLLED_PROGRESS_STABLE_STEPS = 3
CONTROLLED_PROGRESS_STEP_CLIP = 0.010

BALL_SUPPORT_SCALE = 0.01
BALL_SUPPORT_CLIP = 0.25
BALL_SUPPORT_TARGET_DIST_TILES = 2.5

TEAM_SHAPE_MIN_DIST_NORM = 0.065
TEAM_SHAPE_LINEAR_COEF = 0.001
TEAM_SHAPE_QUADRATIC_COEF = 0.01
TEAM_SHAPE_CLIP = 0.00006

# Role-zone uses the existing anchor ellipse as a weak positional prior.
ROLE_ZONE_TOL_X = 0.30
ROLE_ZONE_TOL_Y = 0.15
ROLE_ZONE_TOL_X_GK = 0.005
ROLE_ZONE_TOL_Y_GK = 0.075
ROLE_ZONE_LINEAR_COEF = 0.000015
ROLE_ZONE_QUADRATIC_COEF = 0.000015

# RL commands are held for this many physics frames. At 60 FPS, 12 frames
# means five policy decisions per rendered second.
RL_ACTION_REPEAT_FRAMES = 12

REWARD_COMPONENTS = {
    "outcome.reward_score": REWARD_SCORE,
    "outcome.penalty_concede": PENALTY_CONCEDE,
    "progress.reward_controlled": REWARD_PROGRESS_CONTROLLED * CONTROLLED_PROGRESS_STEP_CLIP,
    "support.reward_ball_support": BALL_SUPPORT_SCALE * BALL_SUPPORT_CLIP,
    "shape.penalty_team_shape": -TEAM_SHAPE_CLIP,
    "shape.penalty_role_zone": -(ROLE_ZONE_LINEAR_COEF + ROLE_ZONE_QUADRATIC_COEF),
}


# CENTRAL OBS
MAX_LEFT_PLAYERS = len(ROLE_NAMES)
CENTRAL_OBS_MASK_DIM = MAX_LEFT_PLAYERS
CENTRAL_OBS_BALL_FEATURES = 6
CENTRAL_OBS_DIM = (MAX_LEFT_PLAYERS * OBS_DIM) + CENTRAL_OBS_MASK_DIM + CENTRAL_OBS_BALL_FEATURES
if MAX_LEFT_PLAYERS != 7:
    raise RuntimeError(f"Kick MAX_LEFT_PLAYERS expected 7, got {MAX_LEFT_PLAYERS}.")
if CENTRAL_OBS_DIM != 454:
    raise RuntimeError(f"Kick CENTRAL_OBS_DIM expected 454, got {CENTRAL_OBS_DIM}.")

GAME_CAPABILITIES = {
    "centralized_critic_required": True,
    "multi_agent": True,
}
ENV_METADATA = {
    "central_obs_dim": int(CENTRAL_OBS_DIM),
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [64, 64],
    "critic_hidden_sizes": [128, 128],
    "policy_head_feature_groups": [
        [INPUT_FEATURE_NAMES.index("self_role_gk")],
        [INPUT_FEATURE_NAMES.index("self_role_lb"), INPUT_FEATURE_NAMES.index("self_role_rb")],
        [
            INPUT_FEATURE_NAMES.index("self_role_lm"),
            INPUT_FEATURE_NAMES.index("self_role_lcm"),
            INPUT_FEATURE_NAMES.index("self_role_rm"),
        ],
        [INPUT_FEATURE_NAMES.index("self_role_lcs")],
    ],
}
ALGO_CONFIG_OVERRIDES = {
    "ppo": {
        "minibatch_size": 512,
        "entropy_coef": float(LEVEL_SETTINGS[int(MIN_LEVEL)]["entropy_coef"]),
    }
}
DEFAULT_TRAIN_CONFIG = {
    "budget": 24_000_000,
    "rollout_steps": 2_048,
    "checkpoint_every": 10,
}
