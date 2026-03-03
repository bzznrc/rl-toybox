"""Shared three-level curriculum progression."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Mapping

@dataclass(frozen=True)
class CurriculumConfig:
    min_level: int = 1
    max_level: int = 3
    min_episodes_per_level: int = 200
    check_window: int = 100
    success_threshold: float = 0.70
    consecutive_checks_required: int = 3


def build_curriculum_config(
    *,
    min_level: int,
    max_level: int,
    promotion_settings: Mapping[str, object] | None = None,
) -> CurriculumConfig:
    """Construct a validated curriculum config from config-file mappings."""

    defaults = CurriculumConfig(min_level=int(min_level), max_level=int(max_level))
    settings = dict(promotion_settings or {})

    min_episodes = max(1, int(settings.get("min_episodes_per_level", defaults.min_episodes_per_level)))
    check_window = max(1, int(settings.get("check_window", defaults.check_window)))
    threshold = float(settings.get("success_threshold", defaults.success_threshold))
    threshold = float(max(0.0, min(1.0, threshold)))
    consecutive_checks = max(
        1,
        int(settings.get("consecutive_checks_required", defaults.consecutive_checks_required)),
    )

    return CurriculumConfig(
        min_level=int(min_level),
        max_level=int(max_level),
        min_episodes_per_level=int(min_episodes),
        check_window=int(check_window),
        success_threshold=float(threshold),
        consecutive_checks_required=int(consecutive_checks),
    )


class ThreeLevelCurriculum:
    def __init__(
        self,
        config: CurriculumConfig | None = None,
        *,
        level_settings: Mapping[int, Mapping[str, object]] | None = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self._level = int(self.config.min_level)
        self._level_settings: dict[int, dict[str, object]] = {}
        if level_settings is not None:
            for level, settings in level_settings.items():
                self._level_settings[int(level)] = dict(settings)
        self._episodes_per_level = {
            level: 0 for level in range(int(self.config.min_level), int(self.config.max_level) + 1)
        }
        self._success_buffers = {
            level: deque(maxlen=int(self.config.check_window))
            for level in range(int(self.config.min_level), int(self.config.max_level) + 1)
        }
        self._consecutive_passes = 0

    def reset(self) -> None:
        self._level = int(self.config.min_level)
        for level in self._episodes_per_level:
            self._episodes_per_level[level] = 0
            self._success_buffers[level].clear()
        self._consecutive_passes = 0

    def get_level(self) -> int:
        return int(self._level)

    def level_settings_for(self, level: int | None = None) -> dict[str, object]:
        target_level = int(self._level if level is None else level)
        return dict(self._level_settings.get(target_level, {}))

    def apply_level_settings(self, apply_fn: Callable[[int, Mapping[str, object]], None]) -> None:
        apply_fn(self.get_level(), self.level_settings_for())

    def episodes_in_level(self, level: int | None = None) -> int:
        target_level = int(self._level if level is None else level)
        return int(self._episodes_per_level.get(target_level, 0))

    def avg_success_in_level(self, level: int | None = None) -> float | None:
        target_level = int(self._level if level is None else level)
        buffer = self._success_buffers.get(target_level)
        if buffer is None or len(buffer) <= 0:
            return None
        return float(mean(buffer))

    def on_episode_end(self, success: int) -> bool:
        success_int = 1 if int(success) > 0 else 0
        level = int(self._level)
        self._episodes_per_level[level] += 1
        self._success_buffers[level].append(int(success_int))

        if level >= int(self.config.max_level):
            return False

        episodes_here = int(self._episodes_per_level[level])
        if episodes_here < int(self.config.min_episodes_per_level):
            return False
        if episodes_here % int(self.config.check_window) != 0:
            return False

        recent = self._success_buffers[level]
        if len(recent) < int(self.config.check_window):
            return False

        avg_success = float(mean(recent))
        if avg_success >= float(self.config.success_threshold):
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        if self._consecutive_passes < int(self.config.consecutive_checks_required):
            return False

        self._level = min(int(self.config.max_level), int(self._level) + 1)
        self._consecutive_passes = 0
        return True


def advance_curriculum(
    curriculum: ThreeLevelCurriculum | None,
    *,
    success: int,
    current_level: int,
    apply_level: Callable[[int], None] | None = None,
) -> tuple[int, bool]:
    if curriculum is None:
        return int(current_level), False
    level_changed = bool(curriculum.on_episode_end(int(success)))
    next_level = int(curriculum.get_level())
    if level_changed and apply_level is not None:
        apply_level(int(next_level))
    return int(next_level), bool(level_changed)


def validate_curriculum_level_settings(
    *,
    min_level: int,
    max_level: int,
    level_settings: Mapping[int, Mapping[str, object]],
) -> None:
    expected_levels = set(range(int(min_level), int(max_level) + 1))
    configured_levels = set(int(level) for level in level_settings.keys())
    missing_levels = sorted(expected_levels - configured_levels)
    extra_levels = sorted(configured_levels - expected_levels)
    if missing_levels or extra_levels:
        raise ValueError(
            "LEVEL_SETTINGS must cover exactly MIN_LEVEL..MAX_LEVEL. "
            f"missing={missing_levels}, extra={extra_levels}"
        )
