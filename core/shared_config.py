"""Shared runtime and window defaults used across games."""

from __future__ import annotations

from core.arcade_style import (
    DEFAULT_BOTTOM_BAR_HEIGHT as BB_HEIGHT,
    DEFAULT_CELL_INSET as CELL_INSET,
    DEFAULT_GRID_COLUMNS as GRID_WIDTH_TILES,
    DEFAULT_GRID_ROWS as GRID_HEIGHT_TILES,
    DEFAULT_TILE_SIZE as TILE_SIZE,
    screen_height,
    screen_width,
)


# Shared runtime defaults.
SHOW_GAME_FPS = 60
TRAINING_FPS = 0

# Compatibility aliases for existing game configs and envs.
FPS = SHOW_GAME_FPS
FIXED_STEP_FPS = SHOW_GAME_FPS
PHYSICS_DT = 1.0 / float(FIXED_STEP_FPS)


# Shared window geometry.
SCREEN_WIDTH = screen_width(GRID_WIDTH_TILES, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT_TILES, TILE_SIZE, BB_HEIGHT)
PLAYFIELD_HEIGHT = SCREEN_HEIGHT - BB_HEIGHT
WORLD_WIDTH = SCREEN_WIDTH
WORLD_HEIGHT = PLAYFIELD_HEIGHT


# Shared visual sizing.
NN_CONTROL_MARKER_SIZE_PX = max(4, TILE_SIZE // 3)


__all__ = [
    "BB_HEIGHT",
    "CELL_INSET",
    "FIXED_STEP_FPS",
    "FPS",
    "GRID_HEIGHT_TILES",
    "GRID_WIDTH_TILES",
    "NN_CONTROL_MARKER_SIZE_PX",
    "PHYSICS_DT",
    "PLAYFIELD_HEIGHT",
    "SCREEN_HEIGHT",
    "SCREEN_WIDTH",
    "SHOW_GAME_FPS",
    "TILE_SIZE",
    "TRAINING_FPS",
    "WORLD_HEIGHT",
    "WORLD_WIDTH",
]
