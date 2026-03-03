"""Shared helpers used by CLI scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.algorithms.base import Algorithm
from core.algorithms.factory import build_algorithm
from core.io.runs import RunPaths, normalize_model_kind, resolve_run_paths
from games.registry import get_game_spec
from games.spec_types import GameSpec


@dataclass(frozen=True)
class PreparedRun:
    spec: GameSpec
    algo_id: str
    run_paths: RunPaths
    algorithm: Algorithm


def prepare_run(game_id: str, algo_override: str | None = None) -> PreparedRun:
    spec = get_game_spec(game_id)
    algo_id = str(algo_override or spec.default_algo).strip().lower()
    run_paths = resolve_run_paths(spec.game_id, algo_id, spec.run_name, create=True)
    algorithm = build_algorithm(
        algo_id=algo_id,
        obs_dim=spec.obs_dim,
        action_space=spec.action_space,
        algo_config=spec.algo_config,
    )
    return PreparedRun(spec=spec, algo_id=algo_id, run_paths=run_paths, algorithm=algorithm)


def resolve_current_level(env: object, *, default: int = 1) -> int:
    level_value = getattr(env, "_current_level", None)
    if level_value is None:
        game = getattr(env, "game", None)
        level_value = getattr(game, "level", None)
    try:
        return max(1, int(level_value))
    except (TypeError, ValueError):
        return max(1, int(default))


def resolve_resume_path(mode: str, run_paths: RunPaths, level: int) -> Path | None:
    mode_key = str(mode).strip().lower()
    level_value = max(1, int(level))
    best_path = run_paths.model_path(level_value, "best")
    check_path = run_paths.model_path(level_value, "check")
    if mode_key == "none":
        return None
    if mode_key in {"check", "checkpoint"}:
        return check_path if check_path.exists() else None
    if mode_key == "best":
        return best_path if best_path.exists() else None
    if mode_key == "auto":
        if best_path.exists():
            return best_path
        if check_path.exists():
            return check_path
        return None
    raise ValueError(f"Unsupported resume mode: {mode}")


def resolve_play_model_path(run_paths: RunPaths, model_choice: str, level: int) -> Path:
    level_value = max(1, int(level))
    model_kind = normalize_model_kind(str(model_choice))
    if model_kind == "check":
        path = run_paths.model_path(level_value, "check")
    else:
        path = run_paths.model_path(level_value, "best")
        fallback_check = run_paths.model_path(level_value, "check")
        if not path.exists() and fallback_check.exists():
            path = fallback_check
    if not path.exists():
        raise FileNotFoundError(f"No model found at '{path}'.")
    return path
