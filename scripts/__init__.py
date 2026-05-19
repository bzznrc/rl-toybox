"""Shared CLI helpers for rl-toybox entrypoints."""

from __future__ import annotations

from pathlib import Path

from core.game import (
    normalize_bang_mode,
    normalize_kick_team_size,
    parse_override_assignments,
    set_nested_override,
)


def normalize_choice(value: str) -> str:
    return str(value).strip().lower()


def build_common_overrides(
    args: object,
    *,
    mode: str,
    render: bool,
    headless: bool,
    episodes: int | None = None,
    include_checkpoint: bool = True,
) -> dict[str, object]:
    overrides = parse_override_assignments(getattr(args, "set_values", []))
    set_nested_override(overrides, "common.mode", str(mode))
    set_nested_override(overrides, "common.render", bool(render))
    set_nested_override(overrides, "common.headless", bool(headless))
    if episodes is not None:
        set_nested_override(overrides, "common.episodes", int(episodes))

    seed = getattr(args, "seed", None)
    if seed is not None:
        set_nested_override(overrides, "common.seed", int(seed))

    bang_mode = getattr(args, "mode", None)
    if bang_mode is not None:
        set_nested_override(overrides, "common.bang_mode", str(bang_mode))

    team_size = getattr(args, "team_size", None)
    if team_size is not None:
        set_nested_override(overrides, "common.team_size", int(team_size))

    checkpoint = getattr(args, "checkpoint", None) if bool(include_checkpoint) else None
    if checkpoint:
        set_nested_override(overrides, "common.checkpoint_path", str(Path(checkpoint)))

    return overrides


def missing_model_message(
    *,
    game_id: str,
    algo_id: str,
    run_name: str,
    level: int,
    bang_mode: object | None,
    team_size: object | None,
    original_error: Exception,
) -> str:
    game_key = str(game_id).strip().lower()
    if game_key == "bang":
        mode = normalize_bang_mode(bang_mode)
        mode_label = str(mode).replace("_", " ").title()
        return (
            f"No trained Bang model was found for level {int(level)} "
            f"({algo_id}_{run_name}_L{int(level)}). The selected mode is {mode_label}. "
            f"Train it first with: python -m scripts.train --game bang --mode {mode} --level {int(level)}"
        )
    if game_key != "kick":
        return str(original_error)
    size = normalize_kick_team_size(team_size)
    return (
        f"No trained Kick model was found for level {int(level)} "
        f"({algo_id}_{run_name}_L{int(level)}). The selected mode is {size} vs. {size}. "
        f"Train the shared Kick model first with: "
        f"python -m scripts.train --game kick --team-size {size} --level {int(level)}"
    )
