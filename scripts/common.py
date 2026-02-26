"""Shared helpers used by CLI scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.algorithms.base import Algorithm
from core.algorithms.factory import build_algorithm
from core.io.runs import RunPaths, resolve_run_paths
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


def resolve_resume_path(mode: str, checkpoint_path: Path, best_path: Path) -> Path | None:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return None
    if mode_key == "checkpoint":
        return checkpoint_path if checkpoint_path.exists() else None
    if mode_key == "best":
        return best_path if best_path.exists() else None
    if mode_key == "auto":
        if best_path.exists():
            return best_path
        if checkpoint_path.exists():
            return checkpoint_path
        return None
    raise ValueError(f"Unsupported resume mode: {mode}")


def resolve_play_model_path(run_paths: RunPaths, model_choice: str) -> Path:
    if str(model_choice).lower() == "checkpoint":
        path = run_paths.checkpoint_path
    else:
        path = run_paths.best_path
        if not path.exists() and run_paths.checkpoint_path.exists():
            path = run_paths.checkpoint_path
    if not path.exists():
        raise FileNotFoundError(f"No model found at '{path}'.")
    return path
