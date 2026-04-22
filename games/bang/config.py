"""Central configuration for Bang AI."""

from __future__ import annotations

from core.game import build_exploration_config
from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Bang AI"
USE_GPU = env_flag("BANG_USE_GPU", False)


# ENV
PLAYER_MOVE_SPEED = 5
AIM_RATE_PER_STEP = 5
PROJECTILE_SPEED = 10
SHOOT_COOLDOWN_FRAMES = 30
AIM_TOLERANCE_DEGREES = 10
MAX_EPISODE_STEPS = 1200
EVENT_TIMER_NORMALIZATION_FRAMES = MAX_EPISODE_STEPS
PLAYER_SPAWN_X_RATIO = 1 / 8
ENEMY_SPAWN_X_RATIO = 7 / 8

ENEMY_SHOT_ERROR_CHOICES = [-20, -10, 0, 10, 20]
ENEMY_MOVE_COMMIT_FRAMES = 10
ENEMY_RECENT_POSITION_MEMORY = 8
ENEMY_HIDDEN_URGENCY_FRAMES = 24
ENEMY_RECENT_POSITION_PENALTY = 0.40
SPAWN_Y_OFFSET = 180
SAFE_RADIUS = 100
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5
OBSTACLE_START_ATTEMPTS = 100

PROJECTILE_TRAJECTORY_DOT_THRESHOLD = 0.98
PROJECTILE_HITBOX_SIZE = 10


# IO
INPUT_FEATURE_NAMES = [
    "self_ang_sin",
    "self_ang_cos",
    "self_move_x",
    "self_move_y",
    "self_shot_cd_norm",
    "sens_fwd",
    "sens_left",
    "sens_right",
    "sens_back",
    "opp1_dx",
    "opp1_dy",
    "opp1_los",
    "opp1_ang_sin",
    "opp1_ang_cos",
    "opp2_dx",
    "opp2_dy",
    "opp2_los",
    "opp2_ang_sin",
    "opp2_ang_cos",
    "opp3_dx",
    "opp3_dy",
    "opp3_los",
    "opp3_ang_sin",
    "opp3_ang_cos",
    "opp_near_dist_norm",
    "haz_tti_norm",
    "haz_miss_norm",
    "haz_in_traj",
]
ACTION_NAMES = [
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_stop",
    "aim_left",
    "aim_right",
    "shoot",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)

ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_STOP_MOVE = 4
ACTION_AIM_LEFT = 5
ACTION_AIM_RIGHT = 6
ACTION_SHOOT = 7


# GAME
DEFAULT_ALGO = "dqn"


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 5
REWARD_ROLLING_WINDOW = 100

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 100,
    "success_threshold": 0.60,
}

LEVEL_SETTINGS = {
    1: {
        "num_players": 2,
        "num_obstacles": 0,
        "enemy_reposition_bias": 0.0,
        "enemy_shoot_probability": 0.0,
    },
    2: {
        "num_players": 2,
        "num_obstacles": 4,
        "enemy_reposition_bias": 0.25,
        "enemy_shoot_probability": 0.025,
    },
    3: {
        "num_players": 2,
        "num_obstacles": 8,
        "enemy_reposition_bias": 0.50,
        "enemy_shoot_probability": 0.05,
    },
    4: {
        "num_players": 3,
        "num_obstacles": 10,
        "enemy_reposition_bias": 0.75,
        "enemy_shoot_probability": 0.075,
    },
    5: {
        "num_players": 4,
        "num_obstacles": 12,
        "enemy_reposition_bias": 1.00,
        "enemy_shoot_probability": 0.10,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
REWARD_KILL = 2.0
PENALTY_STEP = -0.005
ENGAGEMENT_SCALE = 0.5
ENGAGEMENT_CLIP = 0.25
HAZARD_SCALE = 0.5
HAZARD_CLIP = 0.25
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "event.reward_kill": REWARD_KILL,
    "progress.engagement_scale": ENGAGEMENT_SCALE,
    "progress.hazard_scale": HAZARD_SCALE,
    "step.penalty_step": PENALTY_STEP,
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [64, 64],
}
ALGO_CONFIG_OVERRIDES = {
    "dqn": {
    "batch_size": 256,
    "replay_size": 500_000,
    "target_sync_every": 10_000,
    "weight_decay": 1e-5,
    "exploration": build_exploration_config(
        1.0,
        0.05,
        2_500_000,
        patience_episodes=150,
        min_improvement=0.10,
        eps_bump_cap=0.35,
        bump_cooldown_steps=1_250_000,
    ),
    "prioritized_replay": True,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_frames": 10_000_000,
    "per_epsilon": 1e-4,
    }
}
DEFAULT_TRAIN_CONFIG = {
    "budget": 9_000_000,
    "checkpoint_every": 200_000,
    "train_after_steps": 50_000,
    "update_every_steps": 4,
    "updates_per_step": 1,
}
