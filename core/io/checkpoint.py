"""Atomic torch checkpoint helpers."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import torch


def save_torch_checkpoint(
    path: str | Path,
    state: dict[str, Any],
    *,
    retries: int = 5,
    retry_delay_seconds: float = 0.2,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    max_attempts = max(1, int(retries))
    temp_path = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            torch.save(state, temp_path)
            os.replace(temp_path, destination)
            return
        except (OSError, RuntimeError) as error:
            last_error = error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            if attempt < max_attempts - 1:
                delay = float(retry_delay_seconds) * (attempt + 1)
                time.sleep(max(0.0, delay))

    raise RuntimeError(
        f"Failed to save checkpoint to '{destination}' after {max_attempts} attempts."
    ) from last_error


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))
    raw_state = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"Expected dict checkpoint at '{checkpoint_path}', got {type(raw_state)!r}.")
    return raw_state
