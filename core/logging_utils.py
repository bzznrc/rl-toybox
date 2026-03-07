"""Logging and runtime utility helpers."""

from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path
from typing import Any, Mapping

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


def _format_component_value(value: float) -> str:
    rounded = 0.0 if abs(float(value)) < 5e-7 else float(value)
    text = f"{rounded:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", "-0.00"}:
        return "0"
    return text


def format_reward_components(components: Mapping[str, object] | None) -> str | None:
    if not isinstance(components, Mapping) or not components:
        return None

    parts: list[str] = []
    for code, raw_value in components.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        parts.append(f"{str(code)}:{_format_component_value(value)}")

    if not parts:
        return None
    return " ".join(parts)


def log_episode_line(
    *,
    episode: int,
    level: int,
    ep_len: int,
    reward: float,
    avg_reward: float | None,
    best_avg: float | None,
    epsilon: float | None,
    success: int,
    avg_success: float | None,
    best_avg_label: str = "BR",
    reward_components: str | None = None,
) -> None:
    avg_reward_text = "n/a" if avg_reward is None else f"{float(avg_reward):.2f}"
    best_reward_text = "n/a" if best_avg is None else f"{float(best_avg):.2f}"
    avg_success_text = "n/a" if avg_success is None else f"{float(avg_success):.2f}"
    epsilon_text = "n/a" if epsilon is None else f"{float(epsilon):.3f}"
    success_value = 1 if int(success) > 0 else 0
    segments = [
        f"Ep:{int(episode):>5}",
        f"Lv:{int(level):>1}",
        f"Len:{int(ep_len):>5}",
        f"R:{float(reward):>8.2f}",
        f"AR:{avg_reward_text:>8}",
        f"{str(best_avg_label)}:{best_reward_text:>8}",
        f"E:{epsilon_text:>5}",
        f"S:{success_value:>1}",
        f"AS:{avg_success_text:>5}",
    ]
    line = "\t".join(segments) + "\t"
    if reward_components:
        line += str(reward_components)
    logging.getLogger("rl_toybox.train").info(line)


def _format_ppo_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def log_ppo_metrics_line(
    *,
    policy_loss: float | None,
    value_loss: float | None,
    entropy: float | None,
    approx_kl: float | None,
    clip_frac: float | None,
) -> None:
    line = "\t".join(
        [
            "PPO",
            f"PolicyLoss: {_format_ppo_metric(policy_loss)}",
            f"ValueLoss: {_format_ppo_metric(value_loss)}",
            f"Entropy: {_format_ppo_metric(entropy)}",
            f"ApproxKl: {_format_ppo_metric(approx_kl)}",
            f"ClipFrac: {_format_ppo_metric(clip_frac)}",
        ]
    )
    logging.getLogger("rl_toybox.train").info(line)


def log_iteration_line(
    *,
    iteration: int,
    steps: int,
    avg_reward: float,
    best_avg: float,
    best_avg_label: str = "BR",
) -> None:
    log_key_values(
        "rl_toybox.train",
        {
            "Iter": int(iteration),
            "Steps": int(steps),
            "AR": float(avg_reward),
            str(best_avg_label): float(best_avg),
        },
        key_value_separator=":",
    )


def log_save_line(
    *,
    kind: str,
    level: int,
    at: str,
    path: str | Path,
    avg_reward: float | None = None,
) -> None:
    kind_value = str(kind).strip().lower()
    if kind_value == "checkpoint":
        kind_value = "check"
    # Keep signature stable for callers, but save logs are intentionally minimal.
    _ = (level, at, avg_reward)
    kind_label = {"best": "Best", "check": "Check"}.get(kind_value, kind_value.title())
    logging.getLogger("rl_toybox.train").info(
        f">>> Save: {kind_label}\tPath: {format_display_path(path)}"
    )

