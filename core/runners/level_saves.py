"""Shared model-save helpers for curriculum-aware runners."""

from __future__ import annotations

from core.algorithms.base import Algorithm
from core.io.runs import RunPaths
from core.logging_utils import log_save_line


def save_level_checkpoint(
    *,
    algorithm: Algorithm,
    run_paths: RunPaths,
    level: int,
    at: str,
) -> None:
    level_value = int(level)
    checkpoint_path = run_paths.model_path(level=level_value, kind="check")
    algorithm.save(str(checkpoint_path))
    log_save_line(
        kind="check",
        level=level_value,
        at=str(at),
        path=checkpoint_path,
    )


def save_best_if_improved(
    *,
    algorithm: Algorithm,
    run_paths: RunPaths,
    level: int,
    avg_reward: float,
    avg_success: float | None,
    best_avg_reward_by_level: dict[int, float],
    best_avg_success_by_level: dict[int, float],
    at: str,
) -> bool:
    level_value = int(level)
    avg_reward_value = float(avg_reward)
    avg_success_value = 0.0 if avg_success is None else float(avg_success)
    best_success = best_avg_success_by_level.get(level_value, float("-inf"))
    best_reward = best_avg_reward_by_level.get(level_value, float("-inf"))
    if not (
        avg_success_value > float(best_success)
        or (avg_success_value == float(best_success) and avg_reward_value > float(best_reward))
    ):
        return False

    best_avg_success_by_level[level_value] = float(avg_success_value)
    best_avg_reward_by_level[level_value] = float(avg_reward_value)
    best_path = run_paths.model_path(level=level_value, kind="best")
    algorithm.save(str(best_path))
    log_save_line(
        kind="best",
        level=level_value,
        at=str(at),
        avg_reward=float(avg_reward_value),
        path=best_path,
    )
    return True
