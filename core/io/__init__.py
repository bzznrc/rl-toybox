"""Lazy re-exports for shared IO helpers."""

from __future__ import annotations

import importlib
from typing import Any


_RUN_EXPORTS = {
    "RUNS_ROOT",
    "RunPaths",
    "build_metrics_path",
    "build_model_path",
    "load_metrics",
    "normalize_model_kind",
    "resolve_run_paths",
    "write_metrics",
}
_CHECKPOINT_EXPORTS = {"save_torch_checkpoint", "load_torch_checkpoint"}

__all__ = sorted(_RUN_EXPORTS | _CHECKPOINT_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _RUN_EXPORTS:
        module = importlib.import_module("core.io.runs")
        return getattr(module, name)
    if name in _CHECKPOINT_EXPORTS:
        module = importlib.import_module("core.io.checkpoint")
        return getattr(module, name)
    raise AttributeError(f"module 'core.io' has no attribute {name!r}")
