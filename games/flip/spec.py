"""Config-derived public spec for Flip."""

from __future__ import annotations

from games.flip import config


GAME_ID = config.GAME_ID
OBSERVATION_FAMILY = "BOARD"
BOARD_ROWS = int(config.BOARD_ROWS)
BOARD_COLS = int(config.BOARD_COLS)
INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
ACTION_NAMES = tuple(config.ACTION_NAMES)
OBS_DIM = int(config.OBS_DIM)
ACT_DIM = int(config.ACT_DIM)
DEFAULT_ALGO = config.DEFAULT_ALGO
DEFAULT_MODEL_CONFIG = dict(config.DEFAULT_MODEL_CONFIG)
DEFAULT_TRAIN_CONFIG = dict(config.DEFAULT_TRAIN_CONFIG)
