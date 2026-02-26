"""Small logging helpers for consistent console output."""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_mode(mode: str) -> str:
    words = mode.replace("-", " ").split()
    return " ".join("AI" if word.lower() == "ai" else word.title() for word in words)


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
        value_text = _format_value(value)
        if key_value_separator == ":":
            segments.append(f"{key}: {value_text}")
        else:
            segments.append(f"{key}{key_value_separator}{value_text}")

    logging.getLogger(logger_name).info("\t".join(segments))


def log_run_context(mode: str, context: dict[str, Any]) -> None:
    mode_label = _format_mode(mode)
    ordered = OrderedDict((key, value) for key, value in context.items() if value is not None)
    titled = {key.replace("_", " ").title(): value for key, value in ordered.items()}
    log_key_values(
        "snake_ai.run",
        titled,
        prefix=mode_label,
        key_value_separator=":",
    )
