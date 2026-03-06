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


# Shared Color Palette.
# Core Neutrals
COLOR_SLATE_GRAY = (103, 107, 114)
COLOR_FOG_GRAY = (232, 234, 237)
COLOR_LIGHT_NEUTRAL = (245, 246, 248)
COLOR_DARK_NEUTRAL = (29, 32, 36)

# Green Pair
COLOR_LEAF_GREEN = (102, 187, 106)
COLOR_FOREST_GREEN = (56, 142, 60)

# Sand Pair
COLOR_SAND = (214, 188, 133)
COLOR_OCHRE = (166, 133, 82)

# Brown Pair
COLOR_WALNUT = (141, 110, 99)
COLOR_BARK = (93, 64, 55)

# P1
COLOR_AQUA = (102, 204, 193)
COLOR_DEEP_TEAL = (38, 110, 105)

# P2
COLOR_CORAL = (240, 128, 112)
COLOR_BRICK_RED = (150, 62, 54)

# P3
COLOR_BLUE = (66, 133, 244)
COLOR_NAVY = (26, 92, 173)

# P4
COLOR_PURPLE = (171, 71, 188)
COLOR_DEEP_PURPLE = (123, 31, 162)
