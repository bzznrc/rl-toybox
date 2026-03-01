"""Logging and runtime utility helpers."""

from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path
from typing import Any

from core.utils import PROJECT_ROOT


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        force=True,
    )


def get_torch_device(prefer_gpu: bool = False):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_display_path(path_value: str | Path) -> str:
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        return str(path_obj)

    for base in (Path.cwd(), PROJECT_ROOT):
        try:
            return str(path_obj.relative_to(base))
        except ValueError:
            continue
    return str(path_obj)


def _format_context_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith("missing:"):
            return f"missing:{format_display_path(text[len('missing:'):])}"
        try:
            return format_display_path(text)
        except (TypeError, ValueError):
            return text
    return str(value)


def _format_mode_label(mode: str) -> str:
    words = mode.replace("-", " ").split()
    formatted: list[str] = []
    for word in words:
        formatted.append("AI" if word.lower() == "ai" else word.title())
    return " ".join(formatted)


def log_key_values(
    logger_name: str,
    values: dict[str, Any],
    *,
    prefix: str | None = None,
    key_value_separator: str = "=",
) -> None:
    ordered = OrderedDict((key, value) for key, value in values.items() if value is not None)
    segments: list[str] = []
    if prefix:
        segments.append(str(prefix))

    for key, value in ordered.items():
        value_text = _format_context_value(value)
        if key_value_separator == ":":
            segments.append(f"{key}: {value_text}")
        else:
            segments.append(f"{key}{key_value_separator}{value_text}")

    logging.getLogger(logger_name).info("\t".join(segments))


def log_run_context(mode: str, context: dict[str, Any]) -> None:
    mode_label = _format_mode_label(mode)
    titled_context = OrderedDict(
        (key.replace("_", " ").title(), value) for key, value in context.items() if value is not None
    )
    log_key_values(
        "bang_ai.run",
        dict(titled_context),
        prefix=mode_label,
        key_value_separator=":",
    )


def log_episode_line(
    *,
    episode: int,
    ep_len: int,
    reward: float,
    avg_reward: float | None,
    best_avg: float | None,
    epsilon: float | None,
) -> None:
    avg_text = "n/a" if avg_reward is None else f"{float(avg_reward):.2f}"
    best_text = "n/a" if best_avg is None else f"{float(best_avg):.2f}"
    log_key_values(
        "rl_toybox.train",
        {
            "Episode": int(episode),
            "Length": int(ep_len),
            "Reward": f"{float(reward):.2f}",
            "Average": avg_text,
            "Best": best_text,
            "Epsilon": "n/a" if epsilon is None else f"{float(epsilon):.3f}",
        },
        key_value_separator=":",
    )


def log_iteration_line(
    *,
    iteration: int,
    steps: int,
    episodes: int,
    avg_reward: float,
    best_avg: float,
) -> None:
    log_key_values(
        "rl_toybox.train",
        {
            "Iter": int(iteration),
            "Steps": int(steps),
            "Episodes": int(episodes),
            "Avg Reward": float(avg_reward),
            "Best Avg": float(best_avg),
        },
        key_value_separator=":",
    )


def log_save_line(
    *,
    kind: str,
    at: str,
    path: str | Path,
    avg_reward: float | None = None,
) -> None:
    values: OrderedDict[str, Any] = OrderedDict()
    values["Save"] = str(kind)
    values["At"] = str(at)
    if avg_reward is not None:
        values["Avg Reward"] = float(avg_reward)
    values["Path"] = format_display_path(path)
    log_key_values("rl_toybox.train", dict(values), key_value_separator=":")

