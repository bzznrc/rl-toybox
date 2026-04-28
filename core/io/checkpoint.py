"""Atomic torch checkpoint helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any

import torch

from core.logging_utils import format_display_path, PERIODIC_EVENT_PREFIX

_CHECKPOINT_LOGGER = logging.getLogger("rl_toybox.train")


def save_torch_checkpoint(
    path: str | Path,
    state: dict[str, Any],
    *,
    retries: int = 6,
    retry_delay_seconds: float = 0.25,
) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    max_attempts = max(1, int(retries))
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        temp_path = destination.with_name(
            f"{destination.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
        )
        try:
            torch.save(state, temp_path)
            os.replace(temp_path, destination)
            return True
        except (OSError, RuntimeError) as error:
            last_error = error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            if attempt < max_attempts - 1:
                # Windows sync/indexing tools can transiently lock checkpoint files.
                delay = float(retry_delay_seconds) * (2**attempt)
                time.sleep(max(0.0, delay))

    _CHECKPOINT_LOGGER.warning(
        "%s Warn:\tSave: Skipped After %s Attempts\tPath: %s\tReason: %s",
        PERIODIC_EVENT_PREFIX,
        max_attempts,
        format_display_path(destination),
        last_error,
    )
    return False


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))
    raw_state = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"Expected dict checkpoint at '{checkpoint_path}', got {type(raw_state)!r}.")
    return raw_state
