"""Shared Arcade visual constants used across games."""

from __future__ import annotations


# Shared board geometry used by existing games.
DEFAULT_GRID_COLUMNS = 48
DEFAULT_GRID_ROWS = 32
DEFAULT_TILE_SIZE = 20
DEFAULT_BOTTOM_BAR_HEIGHT = 30
DEFAULT_CELL_INSET = 4


def screen_width(columns: int = DEFAULT_GRID_COLUMNS, tile_size: int = DEFAULT_TILE_SIZE) -> int:
    return int(columns) * int(tile_size)


def screen_height(
    rows: int = DEFAULT_GRID_ROWS,
    tile_size: int = DEFAULT_TILE_SIZE,
    bottom_bar_height: int = DEFAULT_BOTTOM_BAR_HEIGHT,
) -> int:
    return int(rows) * int(tile_size) + int(bottom_bar_height)


# Shared color palette.
COLOR_AQUA = (102, 212, 200)
COLOR_DEEP_TEAL = (38, 110, 105)
COLOR_CORAL = (244, 137, 120)
COLOR_BRICK_RED = (150, 62, 54)
COLOR_SLATE_GRAY = (97, 101, 107)
COLOR_FOG_GRAY = (230, 231, 235)
COLOR_CHARCOAL = (28, 30, 36)
COLOR_NEAR_BLACK = (18, 18, 22)
COLOR_SOFT_WHITE = (238, 238, 242)
COLOR_AMBER = (255, 224, 130)


# Extended colors used by Bang teams.
COLOR_P3_BLUE = (66, 133, 244)
COLOR_P3_NAVY = (26, 92, 173)
COLOR_P4_PURPLE = (171, 71, 188)
COLOR_P4_DEEP_PURPLE = (123, 31, 162)
