"""Central configuration for Kick."""

from __future__ import annotations

from core.shared_config import FPS
from core.utils import env_float, env_flag


# RUNTIME
GAME_ID = "kick"
WINDOW_TITLE = "Kick"
USE_GPU = env_flag("KICK_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
GAME_SPEED_SCALE = max(0.2, env_float("KICK_SPEED_SCALE", 0.55))
BALL_RADIUS_SCALE = 1.8
PLAYER_V_MAX_PX_PER_SEC = 3.9 * FPS * GAME_SPEED_SCALE
PLAYER_A_MAX_PX_PER_SEC2 = PLAYER_V_MAX_PX_PER_SEC * 4.0
DEBUG_SANITY_CHECKS = env_flag("KICK_DEBUG_SANITY", False)

PITCH_LINE_WIDTH = 3
PENALTY_AREA_DEPTH_RATIO = 16.5 / 105.0
PENALTY_AREA_WIDTH_RATIO = 40.3 / 68.0


# IO
INPUT_FEATURE_NAMES = [
    # SELF: 7
    "self_x_norm",
    "self_y_norm",
    "self_vx",
    "self_vy",
    "self_theta_cos",
    "self_theta_sin",
    "self_has_ball",

    # TGT / BALL: 9
    "tgt_dx",
    "tgt_dy",
    "tgt_dist_norm",
    "tgt_rel_ang_sin",
    "tgt_rel_ang_cos",
    "tgt_dvx",
    "tgt_dvy",
    "tgt_owner_left",
    "tgt_owner_right",

    # LAND: 4
    "land_opp_goal_dx",
    "land_opp_goal_dy",
    "land_own_goal_dx",
    "land_own_goal_dy",

    # ALLY: 2 x 2 = 4
    "ally1_dx",
    "ally1_dy",
    "ally2_dx",
    "ally2_dy",

    # OPP: 3 x 4 = 12
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
]

OBS_DIM = 36

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
    "kick",
]

ACT_DIM = 10


# GAME
DEFAULT_ALGO = "ppo"
TEAM_SIZE_CHOICES = (3, 5, 7)
DEFAULT_TEAM_SIZE = 3
MAX_TEAM_PLAYERS = 7
TEAM_SIZE_LABELS = {
    3: "3 vs. 3",
    5: "5 vs. 5",
    7: "7 vs. 7",
}


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 5
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 100,
    "success_threshold": 0.60,
}

LEVEL_SCRIPTED_SETTINGS: dict[int, dict[str, object]] = {
    1: {
        "right_players": {3: 1, 5: 2, 7: 3},
        "reaction_frames": 18,
        "pass_probability": 0.20,
        "mistake_probability": 0.20,
        "scripted_player_speed": 0.25,
    },
    2: {
        "right_players": {3: 1, 5: 3, 7: 4},
        "reaction_frames": 12,
        "pass_probability": 0.35,
        "mistake_probability": 0.12,
        "scripted_player_speed": 0.55,
    },
    3: {
        "right_players": {3: 2, 5: 4, 7: 5},
        "reaction_frames": 10,
        "pass_probability": 0.45,
        "mistake_probability": 0.08,
        "scripted_player_speed": 0.75,
    },
    4: {
        "right_players": {3: 3, 5: 5, 7: 6},
        "reaction_frames": 8,
        "pass_probability": 0.55,
        "mistake_probability": 0.05,
        "scripted_player_speed": 0.90,
    },
    5: {
        "right_players": {3: 3, 5: 5, 7: 7},
        "reaction_frames": 5,
        "pass_probability": 0.65,
        "mistake_probability": 0.03,
        "scripted_player_speed": 1.00,
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

RL_ACTION_REPEAT_FRAMES = 12

REWARD_COMPONENTS = {
    "outcome.reward_score": REWARD_SCORE,
    "outcome.penalty_concede": PENALTY_CONCEDE,
    "progress.reward_controlled": REWARD_PROGRESS_CONTROLLED * CONTROLLED_PROGRESS_STEP_CLIP,
    "support.reward_ball_support": BALL_SUPPORT_SCALE * BALL_SUPPORT_CLIP,
    "shape.penalty_team_shape": -TEAM_SHAPE_CLIP,
}


# CENTRAL OBS
MAX_LEFT_PLAYERS = MAX_TEAM_PLAYERS
CENTRAL_OBS_DIM = 128

GAME_CAPABILITIES = {
    "centralized_critic_required": True,
    "multi_agent": True,
}
ENV_METADATA = {
    "central_obs_dim": CENTRAL_OBS_DIM,
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [64, 64],
    "critic_hidden_sizes": [128, 128],
}
ALGO_CONFIG_OVERRIDES = {
    "a2c": {
        "critic_condition_on_agent_obs": False,
    },
    "mappo": {
        "critic_condition_on_agent_obs": False,
    },
    "ppo": {
        "minibatch_size": 256,
        "entropy_coef": 0.01,
        "critic_condition_on_agent_obs": False,
    }
}
DEFAULT_TRAIN_CONFIG = {
    "budget": 8_000_000,
    "rollout_steps": 1_024,
    "checkpoint_every": 10,
}
