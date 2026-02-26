"""Shared utility helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_show_game(show_game_override: bool | None, default_value: bool) -> bool:
    if show_game_override is None:
        return bool(default_value)
    return bool(show_game_override)


def validate_level_settings(
    *,
    min_level: int,
    max_level: int,
    level_settings: Mapping[int, Mapping[str, object]],
    valid_player_counts: Sequence[int] = (2, 3, 4),
) -> None:
    expected_levels = set(range(int(min_level), int(max_level) + 1))
    configured_levels = set(level_settings.keys())
    missing_levels = sorted(expected_levels - configured_levels)
    extra_levels = sorted(configured_levels - expected_levels)
    if missing_levels or extra_levels:
        raise ValueError(
            "LEVEL_SETTINGS must cover exactly MIN_LEVEL..MAX_LEVEL. "
            f"missing={missing_levels}, extra={extra_levels}"
        )

    allowed_counts = {int(count) for count in valid_player_counts}
    for level, settings in level_settings.items():
        if "num_players" not in settings:
            raise ValueError(f"LEVEL_SETTINGS[{level}] is missing required key 'num_players'")
        players = int(settings["num_players"])
        if players not in allowed_counts:
            raise ValueError(
                f"LEVEL_SETTINGS[{level}]['num_players'] must be one of {sorted(allowed_counts)}, got {players}"
            )
