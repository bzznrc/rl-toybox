"""Logging and runtime utility helpers."""

from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Mapping

from core.utils import PROJECT_ROOT

TRAIN_PROGRESS_LOG_INTERVAL_SECONDS = 0.5
PERIODIC_EVENT_PREFIX = ">>>"
_TRAIN_PROGRESS_LOG_LAST_TS: dict[str, float] = {}
_TRAIN_PROGRESS_LOG_LOCK = Lock()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        force=True,
    )
    reset_train_progress_log_throttle()


def reset_train_progress_log_throttle() -> None:
    with _TRAIN_PROGRESS_LOG_LOCK:
        _TRAIN_PROGRESS_LOG_LAST_TS.clear()


def should_emit_train_progress_log(stream_key: str) -> bool:
    key = str(stream_key).strip().lower() or "default"
    now = time.perf_counter()
    interval = float(TRAIN_PROGRESS_LOG_INTERVAL_SECONDS)

    with _TRAIN_PROGRESS_LOG_LOCK:
        previous = _TRAIN_PROGRESS_LOG_LAST_TS.get(key)
        if previous is not None and (now - float(previous)) < interval:
            return False
        _TRAIN_PROGRESS_LOG_LAST_TS[key] = float(now)
    return True


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


def _looks_like_display_path(text: str) -> bool:
    return (
        "/" in text
        or "\\" in text
        or text.startswith(".")
        or bool(Path(text).suffix)
        or bool(re.match(r"^[A-Za-z]:", text))
    )


def _format_log_word(word: str) -> str:
    lower = word.lower()
    acronyms = {"a2c", "ai", "ctde", "dqn", "kl", "mappo", "ppo", "sac"}
    if lower in acronyms:
        return lower.upper()
    if re.fullmatch(r"[A-Za-z]+\d+", word):
        return word.upper()
    return word[:1].upper() + word[1:].lower()


def format_log_indicator(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if text.upper() == "N/A":
        return "N/A"
    if re.fullmatch(r"[A-Za-z]+\d+", text):
        return text.upper()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]*", text):
        tokens = [token for token in re.split(r"[\s_-]+", text) if token]
        return " ".join(_format_log_word(token) for token in tokens)
    return text


def _format_log_label(label: object) -> str:
    text = str(label).strip()
    if not text:
        return ""
    if text.replace("_", "").isalnum() and "_" in text:
        return format_log_indicator(text)
    return text


def _format_context_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, Mapping):
        parts = [
            f"{_format_log_label(key)}: {_format_context_value(nested_value)}"
            for key, nested_value in value.items()
        ]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_context_value(item) for item in value) + "]"
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith("missing:"):
            return f"Missing:{format_display_path(text[len('missing:'):])}"
        if isinstance(value, Path) or _looks_like_display_path(text):
            try:
                return format_display_path(text)
            except (TypeError, ValueError):
                return text
        return format_log_indicator(text)
    return str(value)


def _format_mode_label(mode: str) -> str:
    words = mode.replace("-", " ").split()
    formatted: list[str] = []
    for word in words:
        formatted.append("AI" if word.lower() == "ai" else word.title())
    return " ".join(formatted)


def _format_metric_value(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{int(precision)}f}"


def _format_log_field(label: str, value: object, *, width: int | None = None) -> str:
    label_text = _format_log_label(label)
    text = str(value)
    if width is not None:
        text = f"{text:>{int(width)}}"
    return f"{label_text}: {text}"


def _join_progress_segments(segments: list[str], reward_components: str | None = None) -> str:
    line = "\t".join(segments)
    if reward_components:
        line += "\t" + str(reward_components)
    return line


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
        segments.append(format_log_indicator(prefix))

    for key, value in ordered.items():
        value_text = _format_context_value(value)
        key_text = _format_log_label(key)
        if key_value_separator == ":":
            segments.append(_format_log_field(key_text, value_text))
        else:
            segments.append(f"{key_text}{key_value_separator}{value_text}")

    logging.getLogger(logger_name).info("\t".join(segments))


def log_periodic_event_line(
    logger_name: str,
    event: str,
    values: Mapping[str, object] | None = None,
) -> None:
    segments = [f"{PERIODIC_EVENT_PREFIX} {format_log_indicator(str(event).strip().rstrip(':'))}:"]
    if isinstance(values, Mapping):
        for key, value in values.items():
            if value is None:
                continue
            segments.append(_format_log_field(_format_log_label(key), _format_context_value(value)))
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


def _compact_reward_component_code(code: object) -> str:
    code_text = str(code).strip()
    if not code_text:
        return "?"

    if len(code_text) <= 2 and code_text.upper() == code_text and code_text.replace("_", "").isalnum():
        return code_text

    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", code_text) if token]
    if not tokens:
        compact = "".join(ch for ch in code_text if ch.isalnum()).upper()
        return compact[:2] or "?"

    if len(tokens) > 1 and tokens[0].lower() == "reward":
        tokens = tokens[1:]

    if not tokens:
        return "R"
    if len(tokens) == 1:
        return tokens[0][:2].upper()
    return f"{tokens[0][0]}{tokens[-1][0]}".upper()


def format_reward_components(components: Mapping[str, object] | None) -> str | None:
    if not isinstance(components, Mapping) or not components:
        return None

    parts: list[str] = []
    for code, raw_value in components.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        parts.append(f"{_compact_reward_component_code(code)}:{_format_component_value(value)}")

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
    avg_reward_text = _format_metric_value(avg_reward, 2)
    best_reward_text = _format_metric_value(best_avg, 2)
    avg_success_text = _format_metric_value(avg_success, 2)
    epsilon_text = _format_metric_value(epsilon, 3)
    success_value = 1 if int(success) > 0 else 0
    segments = [
        _format_log_field("Ep", int(episode), width=5),
        _format_log_field("Lv", int(level), width=1),
        _format_log_field("Len", int(ep_len), width=5),
        _format_log_field("R", f"{float(reward):.2f}", width=8),
        _format_log_field("AR", avg_reward_text, width=8),
        _format_log_field(str(best_avg_label), best_reward_text, width=8),
        _format_log_field("E", epsilon_text, width=5),
        _format_log_field("S", success_value, width=1),
        _format_log_field("AS", avg_success_text, width=5),
    ]
    logging.getLogger("rl_toybox.train").info(_join_progress_segments(segments, reward_components))


def log_on_policy_episode_line(
    *,
    episode: int,
    level: int,
    ep_len: int,
    reward: float,
    avg_reward: float | None,
    best_avg: float | None,
    success: int,
    avg_success: float | None,
    best_avg_label: str = "BR",
    policy_loss: float | None = None,
    value_loss: float | None = None,
    entropy: float | None = None,
    approx_kl: float | None = None,
    clip_frac: float | None = None,
    reward_components: str | None = None,
) -> None:
    avg_reward_text = _format_metric_value(avg_reward, 2)
    best_reward_text = _format_metric_value(best_avg, 2)
    avg_success_text = _format_metric_value(avg_success, 2)
    success_value = 1 if int(success) > 0 else 0
    segments = [
        _format_log_field("Ep", int(episode), width=5),
        _format_log_field("Lv", int(level), width=1),
        _format_log_field("Len", int(ep_len), width=5),
        _format_log_field("R", f"{float(reward):.2f}", width=8),
        _format_log_field("AR", avg_reward_text, width=8),
        _format_log_field(str(best_avg_label), best_reward_text, width=8),
        _format_log_field("S", success_value, width=1),
        _format_log_field("AS", avg_success_text, width=5),
    ]
    if policy_loss is not None:
        segments.append(_format_log_field("Policy L.", _format_metric_value(policy_loss, 3), width=7))
    if value_loss is not None:
        segments.append(_format_log_field("Value L.", _format_metric_value(value_loss, 3), width=7))
    if entropy is not None:
        segments.append(_format_log_field("Ent", _format_metric_value(entropy, 3), width=7))
    if approx_kl is not None:
        segments.append(_format_log_field("KL", _format_metric_value(approx_kl, 3), width=7))
    if clip_frac is not None:
        segments.append(_format_log_field("Clip F.", _format_metric_value(clip_frac, 3), width=7))

    logging.getLogger("rl_toybox.train").info(_join_progress_segments(segments, reward_components))


def _format_ppo_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}"


def log_ppo_update_line(
    *,
    update: int,
    level: int,
    steps: int,
    policy_loss: float | None,
    value_loss: float | None,
    explained_variance: float | None,
    entropy: float | None,
    approx_kl: float | None,
) -> None:
    segments = [
        _format_log_field("Up", int(update), width=5),
        _format_log_field("Lv", int(level), width=1),
        _format_log_field("Steps", int(steps), width=6),
        _format_log_field("Pi", _format_metric_value(policy_loss, 3), width=7),
        _format_log_field("V", _format_metric_value(value_loss, 3), width=7),
        _format_log_field("EV", _format_metric_value(explained_variance, 2), width=5),
        _format_log_field("Ent", _format_metric_value(entropy, 2), width=5),
        _format_log_field("KL", _format_metric_value(approx_kl, 3), width=6),
    ]
    logging.getLogger("rl_toybox.train").info(_join_progress_segments(segments))


def log_sac_update_line(
    *,
    update: int,
    level: int,
    steps: int,
    actor_loss: float | None,
    critic_loss: float | None,
    entropy: float | None,
    alpha: float | None,
) -> None:
    segments = [
        _format_log_field("Up", int(update), width=5),
        _format_log_field("Lv", int(level), width=1),
        _format_log_field("Steps", int(steps), width=6),
        _format_log_field("Pi", _format_metric_value(actor_loss, 3), width=7),
        _format_log_field("Q", _format_metric_value(critic_loss, 3), width=7),
        _format_log_field("Ent", _format_metric_value(entropy, 2), width=5),
        _format_log_field("Alpha", _format_metric_value(alpha, 2), width=5),
    ]
    logging.getLogger("rl_toybox.train").info(_join_progress_segments(segments))


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
            ">>> PPO:",
            _format_log_field("Policy L.", _format_ppo_metric(policy_loss), width=7),
            _format_log_field("Value L.", _format_ppo_metric(value_loss), width=7),
            _format_log_field("Ent", _format_ppo_metric(entropy), width=7),
            _format_log_field("KL", _format_ppo_metric(approx_kl), width=7),
            _format_log_field("Clip F.", _format_ppo_metric(clip_frac), width=7),
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


def log_search_play_game_line(
    *,
    game: int,
    moves: int,
    winner: str,
    first_player_win_rate: float | None,
    draw_rate: float | None,
    avg_length: float | None,
    loss: float | None,
    policy_loss: float | None = None,
    value_loss: float | None = None,
    reward_components: str | None = None,
) -> None:
    segments = [
        _format_log_field("Game", int(game), width=5),
        _format_log_field("Len", int(moves), width=5),
        _format_log_field("Win", format_log_indicator(winner), width=5),
        _format_log_field("P1 Win", _format_metric_value(first_player_win_rate, 2), width=5),
        _format_log_field("Draw", _format_metric_value(draw_rate, 2), width=5),
        _format_log_field("Avg. Len", _format_metric_value(avg_length, 1), width=6),
        _format_log_field("Loss", _format_metric_value(loss, 3), width=7),
    ]
    if policy_loss is not None:
        segments.append(_format_log_field("Policy L.", _format_metric_value(policy_loss, 3), width=7))
    if value_loss is not None:
        segments.append(_format_log_field("Value L.", _format_metric_value(value_loss, 3), width=7))
    logging.getLogger("rl_toybox.train").info(_join_progress_segments(segments, reward_components))


def log_arena_line(
    *,
    score: float,
    metrics: Mapping[str, object] | None = None,
) -> None:
    values: OrderedDict[str, object] = OrderedDict()
    values["Score"] = f"{float(score):.2f}"
    if isinstance(metrics, Mapping):
        if "random_score" in metrics:
            values["Random"] = f"{float(metrics['random_score']):.2f}"
        if "greedy_score" in metrics:
            values["Greedy"] = f"{float(metrics['greedy_score']):.2f}"
    log_periodic_event_line("rl_toybox.train", "Arena", values)


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
        f"{PERIODIC_EVENT_PREFIX} Save: {kind_label}\tPath: {format_display_path(path)}"
    )

