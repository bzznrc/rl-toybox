"""Central configuration for Vroom."""

from __future__ import annotations

from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Vroom"
USE_GPU = env_flag("VROOM_USE_GPU", False)
DRAW_RAYS = env_flag("VROOM_DRAW_RAYS", False)


# ENV
MAX_EPISODE_STEPS = 1_200

# OFF-TRACK HANDLING
# Max forward speed multiplier when fully off track.
OFF_TRACK_MAX_SPEED_FACTOR = 0.25
# Seconds to blend between on-track speed (1.0) and off-track speed factor.
OFF_TRACK_SPEED_TRANSITION_SECONDS = 0.5

# TRACK GENERATION
TRACK_WIDTH_PX = 90.0
TRACK_PADDING_PX = 8.0
TRACK_FOOTPRINT_SCALE = 1.0
TRACK_CORNER_RADIUS_PX = 120.0
TRACK_SAMPLE_SPACING_PX = 6.0
TRACK_START_STRAIGHT_LEN_PX = 180.0
TRACK_LONG_SIDE_TEMPLATE_CHOICES = ("straight", "bell", "s_curve")
TRACK_LONG_SIDE_BELL_AMPLITUDE_MIN_PX = 50.0
TRACK_LONG_SIDE_BELL_AMPLITUDE_MAX_PX = 64.0
TRACK_LONG_SIDE_S_AMPLITUDE_MIN_PX = 40.0
TRACK_LONG_SIDE_S_AMPLITUDE_MAX_PX = 60.0
TRACK_LONG_SIDE_INSET_WIDTH_CAP_RATIO = 1.35
TRACK_LONG_SIDE_INSET_LENGTH_CAP_RATIO = 0.24


# IO
INPUT_FEATURE_NAMES = [
    "self_lat_off",
    "self_spd_lat",
    "self_spd_fwd",
    "self_spd_delta",
    "self_yaw_rate",
    "self_head_err_sin",
    "self_head_err_cos",
    "sens_look_near_sin",
    "sens_look_near_cos",
    "sens_look_far_sin",
    "sens_look_far_cos",
    "sens_curve_near",
    "sens_curve_far",
    "sens_fwd",
    "sens_left_front",
    "sens_right_front",
    "sens_left",
    "sens_right",
    "flag_contact",
    "flag_off_track",
]
ACTION_NAMES = [
    "steer",
    "throttle",
    "brake",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# GAME
DEFAULT_ALGO = "sac"
ACTION_SPACE_BOUNDS = {
    "low": [-1.0, 0.0, 0.0],
    "high": [1.0, 1.0, 1.0],
}


# VEHICLE MECHANICS
# Surface grip multiplier when fully off-track (1.0 on-track).
OFF_TRACK_SURFACE_GRIP = 0.50
# Steering authority smooth speed decay strength.
STEER_SPEED_DECAY = 1.35
# Throttle effectiveness loss at full steering.
TURN_THROTTLE_LOSS = 0.30
# Lateral velocity retention (closer to 1.0 = more slip).
LATERAL_DAMPING_ON_TRACK = 0.90
LATERAL_DAMPING_OFF_TRACK = 0.975
# Probe distance for road-edge sensing.
EDGE_PROBE_MAX_DISTANCE_PX = 140.0
FORWARD_RAY_MAX_DISTANCE_PX = 180.0


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
        "num_cars": 1,
        "opponent_speed_cap": 0.0,
        # Negative values coast early, positive values coast late.
        "opponent_coast_error_choices": [0.0],
    },
    2: {
        "num_cars": 2,
        "opponent_speed_cap": 0.25,
        "opponent_coast_error_choices": [-40.0, 0.0, 40.0],
    },
    3: {
        "num_cars": 2,
        "opponent_speed_cap": 0.5,
        "opponent_coast_error_choices": [-30.0, 0.0, 30.0],
    },
    4: {
        "num_cars": 3,
        "opponent_speed_cap": 0.75,
        "opponent_coast_error_choices": [-20.0, 0.0, 20.0],
    },
    5: {
        "num_cars": 4,
        "opponent_speed_cap": 1.0,
        "opponent_coast_error_choices": [-10.0, 0.0, 10.0],
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
PENALTY_STEP = -0.005
PROGRESS_SCALE = 5.0
PROGRESS_CLIP = 0.25
PENALTY_COLLISION = -0.5
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "progress.scale": PROGRESS_SCALE,
    "event.penalty_collision": PENALTY_COLLISION,
    "step.penalty_step": PENALTY_STEP,
}


# TRAINING
DEFAULT_MODEL_CONFIG = {
    "hidden_sizes": [64, 64],
}
ALGO_CONFIG_OVERRIDES = {}
DEFAULT_TRAIN_CONFIG = {
    "budget": 9_000_000,
    "checkpoint_every": 100_000,
}
