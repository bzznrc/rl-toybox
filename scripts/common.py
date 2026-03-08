"""Shared helpers used by CLI scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

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


def normalize_resume_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key == "new":
        return "none"
    if mode_key == "checkpoint":
        return "check"
    return str(mode_key)


def resume_level_bounds(env: object) -> tuple[int, int]:
    curriculum = getattr(env, "_curriculum", None)
    if curriculum is None:
        return 1, 3
    curriculum_config = getattr(curriculum, "config", None)
    min_level = max(1, int(getattr(curriculum_config, "min_level", 1)))
    max_level = max(min_level, int(getattr(curriculum_config, "max_level", 3)))
    return int(min_level), int(max_level)


def resolve_best_resume_level(
    env: object,
    *,
    explicit_level: int | None = None,
    allow_prompt: bool = True,
) -> int:
    min_level, max_level = resume_level_bounds(env)
    if explicit_level is not None:
        return max(int(min_level), min(int(max_level), int(explicit_level)))
    if not bool(allow_prompt) or not bool(sys.stdin.isatty()):
        raise ValueError(
            "Resume mode 'best' requires --resume-level in non-interactive mode. "
            f"Expected {int(min_level)}..{int(max_level)}."
        )

    prompt = f"Resume from BEST model level ({int(min_level)}-{int(max_level)}): "
    while True:
        try:
            raw_value = input(prompt).strip()
        except EOFError as exc:
            raise ValueError(
                "Resume mode 'best' requires --resume-level when interactive input is unavailable."
            ) from exc
        if raw_value == "":
            print("Level is required.")
            continue
        try:
            parsed_level = int(raw_value)
        except ValueError:
            print("Invalid input. Enter a numeric level.")
            continue
        if int(min_level) <= int(parsed_level) <= int(max_level):
            return int(parsed_level)
        print(f"Invalid level. Choose a value between {int(min_level)} and {int(max_level)}.")


def apply_training_start_level(env: object, level: int) -> int:
    level_value = max(1, int(level))
    curriculum = getattr(env, "_curriculum", None)
    if curriculum is not None:
        curriculum_config = getattr(curriculum, "config", None)
        min_level = int(getattr(curriculum_config, "min_level", 1))
        max_level = int(getattr(curriculum_config, "max_level", max(min_level, level_value)))
        level_value = max(min_level, min(level_value, max_level))
        set_level = getattr(curriculum, "set_level", None)
        if callable(set_level):
            set_level(int(level_value), reset_progress=True)
        else:
            reset_curriculum = getattr(curriculum, "reset", None)
            if callable(reset_curriculum):
                reset_curriculum()
            if hasattr(curriculum, "_level"):
                setattr(curriculum, "_level", int(level_value))
            if hasattr(curriculum, "_consecutive_passes"):
                setattr(curriculum, "_consecutive_passes", 0)

    apply_level = getattr(env, "_apply_level_change", None)
    if not callable(apply_level):
        apply_level = getattr(env, "_apply_level_settings", None)
    if callable(apply_level):
        apply_level(int(level_value))
    else:
        if hasattr(env, "_current_level"):
            setattr(env, "_current_level", int(level_value))
        game = getattr(env, "game", None)
        if game is not None and hasattr(game, "level"):
            setattr(game, "level", int(level_value))
            configure_level = getattr(game, "configure_level", None)
            if callable(configure_level):
                configure_level()
    return int(level_value)


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
