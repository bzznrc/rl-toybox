"""Central configuration for Kick."""

from __future__ import annotations

from core.arcade_style import (
    COLOR_CHARCOAL,
    COLOR_NEAR_BLACK,
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

# Background styling aligned with Bang palette.
PITCH_BACKGROUND_COLOR = COLOR_CHARCOAL
PITCH_BACKGROUND_ACCENT_COLOR = COLOR_NEAR_BLACK
