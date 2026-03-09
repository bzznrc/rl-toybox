"""Central configuration for Walk (continuous-control biped walker)."""

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
from core.utils import env_flag, env_int


# RUNTIME
WINDOW_TITLE = "Walk"
FPS = 60
TRAINING_FPS = 0
USE_GPU = env_flag("WALK_USE_GPU", False)
PPO_METRICS_LOG_ENABLED = False


# ENV
GRID_WIDTH_TILES = DEFAULT_GRID_COLUMNS
GRID_HEIGHT_TILES = DEFAULT_GRID_ROWS
TILE_SIZE = DEFAULT_TILE_SIZE
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)

WORLD_PIXELS_PER_METER = 42.0
GROUND_BASE_Y_PX = BB_HEIGHT + int(TILE_SIZE * 5)
CAMERA_LEAD_METERS = 4.0
PHYSICS_DT = 1.0 / FPS
FALL_FAIL_ANIMATION_SECONDS = 2.0

BASE_SEED = env_int("WALK_BASE_SEED", 2026)
SHOW_PLAYER_RAYS = env_flag("WALK_SHOW_PLAYER_RAYS", True)
DRAW_RAYS = SHOW_PLAYER_RAYS

GRAVITY = -9.81
TORSO_MASS = 1.0
TORSO_INERTIA = 0.36
AIR_DRAG = 0.30
TORSO_ANGULAR_DAMP = 2.4
TORSO_UPRIGHT_SPRING = 5.5

TORSO_WIDTH = 0.50
TORSO_HEIGHT = 0.80
HIP_SPACING = 0.34
THIGH_LENGTH = 0.54
SHIN_LENGTH = 0.54
FOOT_LENGTH = 0.34
FOOT_ANGLE_OFFSET_DEG = 90.0

CONTACT_SLOP = 0.03
CONTACT_SPRING = 1500.0
CONTACT_DAMP = 62.0
FRICTION_STIFFNESS = 24.0
FRICTION_COEF = 0.95
MAX_NORMAL_FORCE = 170.0
MAX_FRICTION_FORCE = 110.0

HIP_TORQUE_ACCEL = 34.0
KNEE_TORQUE_ACCEL = 38.0
JOINT_DAMPING = 3.6
HIP_ANGLE_MIN = -1.00
HIP_ANGLE_MAX = 1.00
KNEE_ANGLE_MIN = 0.05
KNEE_ANGLE_MAX = 1.55

START_X = 2.0
START_TORSO_CLEARANCE = 1.10
START_LEFT_HIP_ANGLE = 0.14
START_LEFT_KNEE_ANGLE = 0.78
START_RIGHT_HIP_ANGLE = -0.14
START_RIGHT_KNEE_ANGLE = 0.78

TERMINAL_TILT_RAD = 1.10
TERMINAL_TORSO_CLEARANCE = 0.34
FALL_TILT_GROUND_CLEARANCE = 0.12
MAX_TORSO_SPEED_X = 4.0
MAX_TORSO_SPEED_Y = 5.5
MAX_TORSO_ANG_SPEED = 8.0

RAY_ANGLES_DEG = (15.0, 30.0, 45.0, 60.0)
RAY_MAX_DISTANCE = 3.2
RAY_STEP_SIZE = 0.05
RAY_ORIGIN_LOCAL_X = 0.05
RAY_ORIGIN_LOCAL_Y = 0.42

TERRAIN_SAMPLE_DX = 0.12
TERRAIN_RENDER_MARGIN_METERS = 2.0
TERRAIN_VARIANT_FLAT = "flat"
TERRAIN_VARIANT_STAIRS = "stairs"
TERRAIN_VARIANT_ROLLERS = "rollers"
TERRAIN_FEATURE_START_OFFSET = 2.6
TERRAIN_FEATURE_SPACING = 4.4
TERRAIN_STAIR_TREAD_LENGTH = 0.52
TERRAIN_STAIR_PATTERNS_SYMMETRIC = ((1, 2, 1), (1, 2, 3, 2, 1))
TERRAIN_STAIR_PATTERNS_CUTOFF = ((1, 2), (1, 2, 3))
TERRAIN_STAIR_STEP_HEIGHT_BASE = 0.22
TERRAIN_ROLLER_AMPLITUDE_BASE = 0.24
TERRAIN_ROLLER_HALF_WIDTH_BASE = 0.84
EPISODE_STEP_BUDGET_BASE = 540
EPISODE_STEP_BUDGET_PER_METER = 34.0

BIPED_RENDER_SCALE = 1.30
TORSO_RENDER_SIDE_SCALE = 1.20
TORSO_BORDER_INSET_PX = float(DEFAULT_CELL_INSET)
LEG_INNER_WIDTH_PX = max(2.0, float(TILE_SIZE) * 0.20)
LEG_OUTER_WIDTH_PX = float(LEG_INNER_WIDTH_PX + 2.0 * float(DEFAULT_CELL_INSET))
FOOT_INNER_WIDTH_PX = max(2.0, float(TILE_SIZE) * 0.22)
FOOT_OUTER_WIDTH_PX = float(FOOT_INNER_WIDTH_PX + 2.0 * float(DEFAULT_CELL_INSET))

DISTANCE_MARKER_SPACING_METERS = 1.0
DISTANCE_MARKER_MAJOR_EVERY = 10
DISTANCE_MARKER_HEIGHT_FRACTION = 0.25
DISTANCE_MARKER_INNER_SIZE_PX = max(2.0, float(TILE_SIZE - 2 * DEFAULT_CELL_INSET))
DISTANCE_MARKER_OUTLINE_INSET_PX = float(DEFAULT_CELL_INSET)


# IO
# NOTE: Walk intentionally uses compact normalized scalar angles (not sin/cos pairs)
# to keep OBS_DIM fixed at exactly 18 for this demo.
INPUT_FEATURE_NAMES = [
    "self_torso_tilt",
    "self_torso_ang_vel",
    "self_vx",
    "self_vy",
    "self_left_hip_angle",
    "self_left_hip_speed",
    "self_left_knee_angle",
    "self_left_knee_speed",
    "self_right_hip_angle",
    "self_right_hip_speed",
    "self_right_knee_angle",
    "self_right_knee_speed",
    "self_left_foot_contact",
    "self_right_foot_contact",
    "ray_ground_10",
    "ray_ground_25",
    "ray_ground_40",
    "ray_ground_55",
]
ACTION_NAMES = [
    "left_hip_torque",
    "left_knee_torque",
    "right_hip_torque",
    "right_knee_torque",
]
OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)


# CURRICULUM
MIN_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
MIN_EPISODES_FOR_STATS = REWARD_ROLLING_WINDOW

CURRICULUM_PROMOTION = {
    "min_episodes_per_level": 200,
    "check_window": 25,
    "success_threshold": 0.80,
    "consecutive_checks_required": 2,
}

LEVEL_SETTINGS = {
    1: {
        "terrain_variants": (TERRAIN_VARIANT_FLAT,),
        "goal_distance": 2.5,
        "terrain_scale": 1.0,
        "stairs_cutoff": False,
        "entropy_coef": 0.02,
    },
    2: {
        "terrain_variants": (TERRAIN_VARIANT_STAIRS, TERRAIN_VARIANT_ROLLERS),
        "goal_distance": 12.0,
        "terrain_scale": 1.5,
        "stairs_cutoff": False,
        "entropy_coef": 0.01,
    },
    3: {
        "terrain_variants": (TERRAIN_VARIANT_STAIRS, TERRAIN_VARIANT_ROLLERS),
        "goal_distance": 25.0,
        "terrain_scale": 2.0,
        "stairs_cutoff": True,
        "entropy_coef": 0.005,
    },
}


# REWARDS
PENALTY_LOSE = -5.0

REWARD_COMPONENTS = {
    "P": 1.0,
    "L": PENALTY_LOSE,
}


# TRAINING
HIDDEN_DIMENSIONS = [32, 32]

MAX_TRAINING_ITERATIONS = 6000
ROLLOUT_STEPS = 2048
CHECKPOINT_EVERY_ITERATIONS = 10

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
UPDATE_EPOCHS = 8
MINIBATCH_SIZE = 256
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
INIT_LOG_STD = -0.70
MIN_LOG_STD = -4.50
MAX_LOG_STD = 1.25
