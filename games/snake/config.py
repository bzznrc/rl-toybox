"""Central configuration for Snake AI."""

from __future__ import annotations

from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Snake AI"
USE_GPU = env_flag("SNAKE_USE_GPU", False)


# ENV
WRAP_AROUND = True
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5


# IO
INPUT_FEATURE_NAMES = [
    "self_heading_sin",
    "self_heading_cos",
    "self_len_norm",
    "self_last_act_norm",
    "self_hunger_norm",
    "sens_fwd",
    "sens_left",
    "sens_right",
    "tgt_rel_angle_sin",
    "tgt_rel_angle_cos",
    "tgt_manhattan_norm",
    "tgt_dist_delta",
]
ACTION_NAMES = [
    "straight",
    "turn_right",
    "turn_left",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# GAME
DEFAULT_ALGO = "qlearn"


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 5
REWARD_ROLLING_WINDOW = 100
SUCCESS_FOODS_REQUIRED = 5

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 100,
    "success_threshold": 0.60,
}

LEVEL_SETTINGS = {
    1: {
        "num_obstacles": 0,
        "timeout_steps_per_length": 120,
    },
    2: {
        "num_obstacles": 3,
        "timeout_steps_per_length": 110,
    },
    3: {
        "num_obstacles": 6,
        "timeout_steps_per_length": 100,
    },
    4: {
        "num_obstacles": 9,
        "timeout_steps_per_length": 90,
    },
    5: {
        "num_obstacles": 12,
        "timeout_steps_per_length": 80,
    },
}


# REWARDS
PENALTY_LOSE = -5.0
REWARD_FOOD = 1.0
PENALTY_STEP = -0.005
PROGRESS_SCALE = 1.0
PROGRESS_CLIP = 0.05
REWARD_COMPONENTS = {
    "outcome.penalty_lose": PENALTY_LOSE,
    "event.reward_food": REWARD_FOOD,
    "progress.scale": PROGRESS_SCALE,
    "step.penalty_step": PENALTY_STEP,
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [32],
}
ALGO_CONFIG_OVERRIDES = {}
DEFAULT_TRAIN_CONFIG = {
    "budget": 3_000_000,
    "checkpoint_every": 100_000,
}
