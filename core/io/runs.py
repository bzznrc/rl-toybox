"""Run directory resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from core.utils import PROJECT_ROOT


RUNS_ROOT = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint_path: Path
    best_path: Path
    metrics_path: Path


def resolve_run_paths(game_id: str, algo_id: str, run_name: str, *, create: bool = True) -> RunPaths:
    run_dir = RUNS_ROOT / str(game_id) / str(algo_id) / str(run_name)
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoint.pth",
        best_path=run_dir / "best.pth",
        metrics_path=run_dir / "metrics.json",
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
