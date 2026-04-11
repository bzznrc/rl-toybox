"""Central configuration for Vroom."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_GRID_COLUMNS as GRID_WIDTH_TILES,
    DEFAULT_GRID_ROWS as GRID_HEIGHT_TILES,
    DEFAULT_TILE_SIZE as TILE_SIZE,
    screen_height,
    screen_width,
)
from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Vroom"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("VROOM_USE_GPU", False)
DRAW_RAYS = env_flag("VROOM_DRAW_RAYS", False)


# ENV
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
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
TRACK_LONG_SIDE_BELL_AMPLITUDE_MIN_PX = 72.0
TRACK_LONG_SIDE_BELL_AMPLITUDE_MAX_PX = 92.0
TRACK_LONG_SIDE_S_AMPLITUDE_MIN_PX = 40.0
TRACK_LONG_SIDE_S_AMPLITUDE_MAX_PX = 58.0
TRACK_LONG_SIDE_INSET_WIDTH_CAP_RATIO = 0.72
TRACK_LONG_SIDE_INSET_LENGTH_CAP_RATIO = 0.20


# IO
INPUT_FEATURE_NAMES = [
    "track_lat_off",
    "ego_spd_lat",
    "ego_spd_fwd",
    "ego_spd_delta",
    "ego_yaw_rate",
    "track_heading_err_sin",
    "track_heading_err_cos",
    "track_look_near_sin",
    "track_look_near_cos",
    "track_look_far_sin",
    "track_look_far_cos",
    "track_curve_near",
    "track_curve_far",
    "ray_f",
    "ray_fl",
    "ray_fr",
    "ray_l",
    "ray_r",
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
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 250,
    "check_window": 25,
    "success_threshold": 0.60,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "num_cars": 1,
        "opponent_speed_cap": 0.0,
        # Negative values coast early, positive values coast late.
        "opponent_coast_error_choices": [-40.0, 0.0, 40.0],
    },
    2: {
        "num_cars": 2,
        "opponent_speed_cap": 0.75,
        "opponent_coast_error_choices": [-20.0, 0.0, 20.0],
    },
    3: {
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
# SAC defaults:
# - Actor: 20 -> 64 -> 64 -> 3
# - Critic Q1/Q2: (20 + 3) -> 64 -> 64 -> 1
HIDDEN_DIMENSIONS = [64, 64]

MAX_TRAINING_STEPS = 10_000_000
CHECKPOINT_EVERY_STEPS = 100_000

REPLAY_BUFFER_SIZE = 200_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
GRAD_CLIP_NORM = 10.0
INIT_ALPHA = 0.20

LEARN_START_STEPS = 10_000
TRAIN_EVERY_STEPS = 1
UPDATES_PER_TRAIN = 1
