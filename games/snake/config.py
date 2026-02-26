"""Central configuration for Snake AI."""

from __future__ import annotations

from dataclasses import dataclass

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_NEAR_BLACK,
    COLOR_SLATE_GRAY,
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_CELL_INSET,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
)
from core.utils import PROJECT_ROOT


@dataclass(frozen=True)
class BoardConfig:
    columns: int
    rows: int
    cell_size_px: int
    bottom_bar_height_px: int
    cell_inset_px: int

    @property
    def screen_width_px(self) -> int:
        return self.columns * self.cell_size_px

    @property
    def screen_height_px(self) -> int:
        return self.rows * self.cell_size_px + self.bottom_bar_height_px


BOARD = BoardConfig(
    columns=DEFAULT_GRID_COLUMNS,
    rows=DEFAULT_GRID_ROWS,
    cell_size_px=DEFAULT_TILE_SIZE,
    bottom_bar_height_px=DEFAULT_BOTTOM_BAR_HEIGHT,
    cell_inset_px=DEFAULT_CELL_INSET,
)

# Runtime
SHOW_GAME_OVERRIDE: bool | None = None


def resolve_show_game(default_value: bool) -> bool:
    if SHOW_GAME_OVERRIDE is None:
        return bool(default_value)
    return SHOW_GAME_OVERRIDE


LOAD_MODEL = "B"  # False, "B" (best), or "L" (last)
FPS = 20
TRAINING_FPS = 0
WINDOW_TITLE = "Snake AI"

# Arena
GRID_WIDTH_TILES = BOARD.columns
GRID_HEIGHT_TILES = BOARD.rows
TILE_SIZE = BOARD.cell_size_px
BB_HEIGHT = BOARD.bottom_bar_height_px
SCREEN_WIDTH = BOARD.screen_width_px
SCREEN_HEIGHT = BOARD.screen_height_px
CELL_INSET = BOARD.cell_inset_px
CELL_INSET_DOUBLE = CELL_INSET * 2
NN_CONTROL_MARKER_SIZE_PX = max(4, BOARD.cell_size_px // 3)

# Gameplay
NUM_OBSTACLES = 10
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5
WRAP_AROUND = True

# Input/output spaces
INPUT_FEATURE_NAMES = [
    "danger_straight",
    "danger_right",
    "danger_left",
    "dir_left",
    "dir_right",
    "dir_up",
    "dir_down",
    "food_left",
    "food_right",
    "food_up",
    "food_down",
    "snake_length",
]
ACTION_NAMES = [
    "straight",
    "turn_right",
    "turn_left",
]
NUM_INPUT_FEATURES = len(INPUT_FEATURE_NAMES)
NUM_ACTIONS = len(ACTION_NAMES)
MODEL_INPUT_SIZE = NUM_INPUT_FEATURES
MODEL_OUTPUT_SIZE = NUM_ACTIONS

# Model and training
MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 1e-3
HIDDEN_DIMENSIONS = [32]
MODEL_SUBDIR = "_".join(str(size) for size in HIDDEN_DIMENSIONS)
MODEL_DIR = PROJECT_ROOT / "model" / MODEL_SUBDIR
MODEL_NAME = f"snake_{MODEL_SUBDIR}"
MODEL_CHECKPOINT_PATH = str(MODEL_DIR / f"{MODEL_NAME}.pth")
MODEL_BEST_PATH = str(MODEL_DIR / f"{MODEL_NAME}_best.pth")
MODEL_SAVE_RETRIES = 5
MODEL_SAVE_RETRY_DELAY_SECONDS = 0.2
AVG100_WINDOW = 100
BEST_MODEL_MIN_EPISODES = AVG100_WINDOW
GAMMA = 0.9
EPSILON_START = 0.5
EPSILON_DECAY = 0.999995
EPSILON_END = 0.05

# Reward shaping
REWARD_FOOD = 10.0
REWARD_DEATH_OR_TIMEOUT = -5.0
REWARD_STEP = -0.01
REWARD_COMPONENTS = {
    "food": REWARD_FOOD,
    "death_or_timeout": REWARD_DEATH_OR_TIMEOUT,
    "step": REWARD_STEP,
}

# Colors
# Re-exported shared palette constants:
# COLOR_AQUA, COLOR_DEEP_TEAL, COLOR_CORAL, COLOR_BRICK_RED,
# COLOR_SLATE_GRAY, COLOR_FOG_GRAY, COLOR_CHARCOAL, COLOR_NEAR_BLACK
