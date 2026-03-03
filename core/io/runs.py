"""Run directory resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from core.utils import PROJECT_ROOT


RUNS_ROOT = PROJECT_ROOT / "runs"
MODEL_KIND_BEST = "best"
MODEL_KIND_CHECK = "check"
_MODEL_KIND_ALIASES = {
    "best": MODEL_KIND_BEST,
    "check": MODEL_KIND_CHECK,
    "checkpoint": MODEL_KIND_CHECK,
}


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    game_id: str
    algo_tag: str
    net_tag: str
    metrics_path: Path

    def model_path(self, level: int, kind: str) -> Path:
        return build_model_path(
            game=self.game_id,
            algo_tag=self.algo_tag,
            net_tag=self.net_tag,
            level=level,
            kind=kind,
            create=True,
        )


def normalize_model_kind(kind: str) -> str:
    key = str(kind).strip().lower()
    if key not in _MODEL_KIND_ALIASES:
        valid = ", ".join(sorted(_MODEL_KIND_ALIASES.keys()))
        raise ValueError(f"Unsupported model kind '{kind}'. Valid: {valid}")
    return _MODEL_KIND_ALIASES[key]


def build_model_filename(*, algo_tag: str, net_tag: str, level: int, kind: str) -> str:
    kind_value = normalize_model_kind(kind)
    level_value = max(1, int(level))
    return f"{str(algo_tag)}_{str(net_tag)}_L{level_value}_{kind_value}.pth"


def build_model_path(
    *,
    game: str,
    algo_tag: str,
    net_tag: str,
    level: int,
    kind: str,
    create: bool = True,
) -> Path:
    game_dir = RUNS_ROOT / str(game)
    if create:
        game_dir.mkdir(parents=True, exist_ok=True)
    return game_dir / build_model_filename(
        algo_tag=str(algo_tag),
        net_tag=str(net_tag),
        level=int(level),
        kind=str(kind),
    )


def build_metrics_path(*, game: str, algo_tag: str, net_tag: str, create: bool = True) -> Path:
    game_dir = RUNS_ROOT / str(game)
    if create:
        game_dir.mkdir(parents=True, exist_ok=True)
    return game_dir / f"{str(algo_tag)}_{str(net_tag)}_metrics.json"


def resolve_run_paths(game_id: str, algo_id: str, run_name: str, *, create: bool = True) -> RunPaths:
    run_dir = RUNS_ROOT / str(game_id)
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        game_id=str(game_id),
        algo_tag=str(algo_id),
        net_tag=str(run_name),
        metrics_path=build_metrics_path(game=str(game_id), algo_tag=str(algo_id), net_tag=str(run_name), create=create),
    )


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(metrics_path)
