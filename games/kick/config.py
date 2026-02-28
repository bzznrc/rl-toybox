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
from core.utils import env_float

# Arena layout
GRID_WIDTH_TILES = DEFAULT_GRID_COLUMNS
GRID_HEIGHT_TILES = DEFAULT_GRID_ROWS
TILE_SIZE = DEFAULT_TILE_SIZE
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
CELL_INSET = DEFAULT_CELL_INSET
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)

# Runtime
FPS = 60
TRAINING_FPS = 0
WINDOW_TITLE = "Kick"
PHYSICS_DT = 1.0 / FPS

# Input/output spaces
INPUT_FEATURE_NAMES = [
    # SELF (10)
    "self_vx",
    "self_vy",
    "self_theta_cos",
    "self_theta_sin",
    "self_has_ball",
    "self_role",
    "self_stamina",
    "self_stamina_delta",
    "self_in_contact",
    "self_last_action",
    # TGT (6)
    "tgt_dx",
    "tgt_dy",
    "tgt_dvx",
    "tgt_dvy",
    "tgt_is_free",
    "tgt_owner_team",
    # GOALS (4)
    "goal_opp_dx",
    "goal_opp_dy",
    "goal_own_dx",
    "goal_own_dy",
    # ALLIES (8)
    "ally1_dx",
    "ally1_dy",
    "ally1_dvx",
    "ally1_dvy",
    "ally2_dx",
    "ally2_dy",
    "ally2_dvx",
    "ally2_dvy",
    # FOES (8)
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

# Reward shaping
REWARD_SCORE = 10.0
PENALTY_CONCEDE = -5.0
PROGRESS_SCALE = 1.0
PROGRESS_CLIP = 0.2
REWARD_POSSESSION_GAIN = 0.5
PENALTY_POSSESSION_LOSS = -0.5
PENALTY_KICK_COST = -0.01
PENALTY_STEP = -0.001
REWARD_COMPONENTS = {
    "outcome.reward_score": REWARD_SCORE,
    "outcome.penalty_concede": PENALTY_CONCEDE,
    "progress.scale": PROGRESS_SCALE,
    "progress.clip": PROGRESS_CLIP,
    "event.reward_possession_gain": REWARD_POSSESSION_GAIN,
    "event.penalty_possession_loss": PENALTY_POSSESSION_LOSS,
    "event.penalty_kick_cost": PENALTY_KICK_COST,
    "step.penalty_step": PENALTY_STEP,
}

# Gameplay tuning
# Global pace scaler (set env var KICK_SPEED_SCALE to override).
GAME_SPEED_SCALE = max(0.2, env_float("KICK_SPEED_SCALE", 0.5))
BALL_RADIUS_SCALE = 1.8
PLAYER_V_MAX_PX_PER_SEC = 3.8 * FPS * GAME_SPEED_SCALE
PLAYER_A_MAX_PX_PER_SEC2 = PLAYER_V_MAX_PX_PER_SEC * 4.0

# Pitch line styling
PITCH_LINE_WIDTH = 3

# Real-world inspired proportions:
# - Penalty area depth: 16.5m on a 105m length
# - Penalty area width: 40.32m on a 68m width
PENALTY_AREA_DEPTH_RATIO = 16.5 / 105.0
PENALTY_AREA_WIDTH_RATIO = 40.3 / 68.0

# Stamina model
STAMINA_MIN = 0.5
STAMINA_MAX = 1.0
STAMINA_DRAIN_SECONDS = 5.0
STAMINA_RECOVER_SECONDS = 1.0
