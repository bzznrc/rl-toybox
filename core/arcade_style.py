"""Shared Arcade geometry, fonts, and twelve-color palette."""

from __future__ import annotations


# Shared board geometry used by existing games.
DEFAULT_GRID_COLUMNS = 48
DEFAULT_GRID_ROWS = 32
DEFAULT_TILE_SIZE = 20
DEFAULT_BOTTOM_BAR_HEIGHT = 30
DEFAULT_CELL_INSET = 4
DEFAULT_STATUS_BAR_FONT_SIZE = 12

INTER_FONT_FILE = "Inter.ttf"
INTER_ITALIC_FONT_FILE = "Inter-Italic.ttf"
INTER_FONT_NAME = ("Inter", "Inter Regular", "Arial", "sans-serif")
GAME_UI_FONT_NAME = INTER_FONT_NAME
GAME_TITLE_FONT_NAME = INTER_FONT_NAME


def screen_width(columns: int = DEFAULT_GRID_COLUMNS, tile_size: int = DEFAULT_TILE_SIZE) -> int:
    return int(columns) * int(tile_size)


def screen_height(
    rows: int = DEFAULT_GRID_ROWS,
    tile_size: int = DEFAULT_TILE_SIZE,
    bottom_bar_height: int = DEFAULT_BOTTOM_BAR_HEIGHT,
) -> int:
    return int(rows) * int(tile_size) + int(bottom_bar_height)


# Four neutrals and four two-tone accents. Keep this palette deliberately small:
# games may vary alpha, but should not introduce additional RGB colors.
COLOR_SLATE_GRAY = (103, 107, 114)
COLOR_FOG_GRAY = (232, 234, 237)
COLOR_LIGHT_NEUTRAL = (245, 246, 248)
COLOR_DARK_NEUTRAL = (29, 32, 36)

COLOR_AQUA = (102, 204, 193)
COLOR_DEEP_TEAL = (38, 110, 105)
COLOR_CORAL = (240, 128, 112)
COLOR_BRICK_RED = (150, 62, 54)
COLOR_BLUE = (66, 133, 244)
COLOR_NAVY = (26, 92, 173)
COLOR_PURPLE = (171, 71, 188)
COLOR_DEEP_PURPLE = (123, 31, 162)

NEUTRAL_COLORS = (
    COLOR_SLATE_GRAY,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_DARK_NEUTRAL,
)
ACCENT_PAIRS = (
    (COLOR_AQUA, COLOR_DEEP_TEAL),
    (COLOR_CORAL, COLOR_BRICK_RED),
    (COLOR_BLUE, COLOR_NAVY),
    (COLOR_PURPLE, COLOR_DEEP_PURPLE),
)
