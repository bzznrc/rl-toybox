"""Central configuration for Vroom."""

from __future__ import annotations

from core.utils import env_flag


# RUNTIME
WINDOW_TITLE = "Vroom"
USE_GPU = env_flag("VROOM_USE_GPU", False)
DRAW_RAYS = env_flag("VROOM_DRAW_RAYS", False)
SHOW_GHOST_OVERLAY = DRAW_RAYS


# ENV
MAX_EPISODE_STEPS = 1_200

# OFF-TRACK HANDLING
# Max forward speed multiplier when fully off track.
OFF_TRACK_MAX_SPEED_FACTOR = 0.50
# Seconds to blend between on-track speed (1.0) and off-track speed factor.
OFF_TRACK_SPEED_TRANSITION_SECONDS = 0.5
# Virtual training-validity margin; this does not widen the road geometry or mask.
TRACK_VALID_MARGIN_PX = 0.0
TRACK_VALID_HYSTERESIS_PX = 4.0

# TRACK GENERATION
TRACK_WIDTH_PX = 90.0  # Supported: 80.0 or 90.0.
TRACK_PADDING_PX = 8.0
TRACK_FOOTPRINT_SCALE = 1.0
TRACK_CORNER_RADIUS_PX = 120.0
TRACK_SAMPLE_SPACING_PX = 6.0
TRACK_START_STRAIGHT_LEN_PX = 180.0
TRACK_LONG_SIDE_TEMPLATE_CHOICES = ("straight", "bell", "s_curve", "fold")
TRACK_SHORT_SIDE_TEMPLATE_CHOICES = ("straight", "bell")
TRACK_LONG_SIDE_BELL_AMPLITUDE_MIN_PX = 50.0
TRACK_LONG_SIDE_BELL_AMPLITUDE_MAX_PX = 64.0
TRACK_LONG_SIDE_S_AMPLITUDE_MIN_PX = 40.0
TRACK_LONG_SIDE_S_AMPLITUDE_MAX_PX = 60.0
TRACK_LONG_SIDE_INSET_WIDTH_CAP_RATIO = 1.35
TRACK_LONG_SIDE_INSET_LENGTH_CAP_RATIO = 0.24
TRACK_FOLD_GAP_PX = 16.0
TRACK_BEND_SMOOTHING_PASSES = 1
TRACK_GENERATION_MAX_ATTEMPTS = 50
TRACK_COMPLEXITY_HARD_SAMPLE_RATE = 0.50


# IO
INPUT_FEATURE_NAMES = [
    # SELF: 7
    "self_lat_off",
    "self_spd_lat",
    "self_spd_fwd",
    "self_spd_delta",
    "self_yaw_rate",
    "self_head_err_sin",
    "self_head_err_cos",

    # SENS / ROUTE: 3 probes x 5 = 15
    "sens_route1_fwd",
    "sens_route1_lat",
    "sens_route1_tan_sin",
    "sens_route1_tan_cos",
    "sens_route1_bend",

    "sens_route2_fwd",
    "sens_route2_lat",
    "sens_route2_tan_sin",
    "sens_route2_tan_cos",
    "sens_route2_bend",

    "sens_route3_fwd",
    "sens_route3_lat",
    "sens_route3_tan_sin",
    "sens_route3_tan_cos",
    "sens_route3_bend",

    # SENS / EDGE: 5
    "sens_edge_fwd",
    "sens_edge_left_front",
    "sens_edge_right_front",
    "sens_edge_left",
    "sens_edge_right",

    # SENS / CAR-PATH: 3
    "sens_car_left",
    "sens_car_fwd",
    "sens_car_right",

    # FLAG: 2
    "flag_contact",
    "flag_off_track",
]
ACTION_NAMES = [
    "steer",
    "throttle",
    "brake",
]
OBS_DIM = 32
ACT_DIM = 3
assert len(INPUT_FEATURE_NAMES) == 32
assert len(ACTION_NAMES) == 3
assert OBS_DIM == 32
assert ACT_DIM == 3


# GAME
DEFAULT_ALGO = "sac"
ACTION_SPACE_BOUNDS = {
    "low": [-1.0, 0.0, 0.0],
    "high": [1.0, 1.0, 1.0],
}


# VEHICLE MECHANICS
# Surface grip multiplier when fully off-track (1.0 on-track).
OFF_TRACK_SURFACE_GRIP = 0.75
# Steering authority smooth speed decay strength.
STEER_SPEED_DECAY = 1.35
# Low-speed steering remains possible, but less spin-prone.
STEER_FULL_SPEED_NORM = 0.75
STEER_MIN_SPEED_FACTOR = 0.25
# Throttle effectiveness loss at full steering.
TURN_THROTTLE_LOSS = 0.30
# Lateral velocity retention (closer to 1.0 = more slip).
LATERAL_DAMPING_ON_TRACK = 0.90
LATERAL_DAMPING_OFF_TRACK = 0.975
# Probe distance for road-edge sensing.
EDGE_PROBE_MAX_DISTANCE_PX = 140.0
FORWARD_RAY_MAX_DISTANCE_PX = 180.0
ROUTE_LOOKAHEAD_RANGES_PX = ((45.0, 75.0), (90.0, 135.0), (180.0, 270.0))
SENS_CAR_RANGE_PX = 140.0
SENS_CAR_SIDE_RANGE_PX = 100.0
OPPONENT_SPEED_MULT_RANGE = (0.95, 1.05)
OPPONENT_BEND_CAUTION_MULT_RANGE = (0.90, 1.10)
OPPONENT_MIN_BEND_SPEED_FACTOR = 0.40
OPPONENT_BRAKE_RESPONSE = 0.35


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 5
REWARD_ROLLING_WINDOW = 100

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 100,
    "success_threshold": 0.80,
}

LEVEL_SETTINGS = {
    1: {
        "num_cars": 1,
        "opponent_speed_cap": 0.0,
        "track_complexity_range": (0.00, 0.30),
        "random_start_prob": 0.80,
    },
    2: {
        "num_cars": 1,
        "opponent_speed_cap": 0.0,
        "track_complexity_range": (0.20, 0.50),
        "random_start_prob": 0.60,
    },
    3: {
        "num_cars": 2,
        "opponent_speed_cap": 0.50,
        "track_complexity_range": (0.40, 0.70),
        "random_start_prob": 0.40,
    },
    4: {
        "num_cars": 3,
        "opponent_speed_cap": 0.75,
        "track_complexity_range": (0.50, 0.80),
        "random_start_prob": 0.20,
    },
    5: {
        "num_cars": 4,
        "opponent_speed_cap": 1.00,
        "track_complexity_range": (0.60, 0.90),
        "random_start_prob": 0.00,
    },
}


# REWARDS
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
PENALTY_STEP = -0.0075
PROGRESS_SCALE = 7.5
PROGRESS_CLIP = 0.20
PENALTY_TRACK_COVERAGE = 0.005
PENALTY_COLLISION = -0.02
# End stuck/jiggle episodes using the existing loss outcome.
NO_PROGRESS_TIMEOUT_STEPS = 240
NO_PROGRESS_EPS_NORM = 0.01
REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "progress.scale": PROGRESS_SCALE,
    "track.penalty_coverage": PENALTY_TRACK_COVERAGE,
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
