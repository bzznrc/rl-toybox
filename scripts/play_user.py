"""Generic human-play entrypoint for supported games."""

from __future__ import annotations

import argparse

from core.game import (
    apply_generic_launch_level,
    apply_seed_from_config,
    build_env_from_config,
    compose_run_config,
    parse_override_assignments,
    set_nested_override,
)
from core.logging_utils import configure_logging, log_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a game in human-control mode")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--level", type=int, default=3, help="Play level selector (defaults to 3)")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument("--headless", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Generic override in dotted.path=value form, e.g. --set common.render=false",
    )
    return parser.parse_args()


def _build_play_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides = parse_override_assignments(args.set_values)
    set_nested_override(overrides, "common.mode", "play")
    set_nested_override(overrides, "common.render", not bool(args.headless))
    set_nested_override(overrides, "common.headless", bool(args.headless))
    if args.seed is not None:
        set_nested_override(overrides, "common.seed", int(args.seed))
    return overrides


def main() -> None:
    args = parse_args()
    configure_logging()
    level = int(apply_generic_launch_level(args.game, args.level))

    composed_config = compose_run_config(args.game, mode="play", user_overrides=_build_play_overrides(args))
    apply_seed_from_config(composed_config)
    render = bool(dict(composed_config.get("common", {})).get("render", not bool(args.headless)))
    env = build_env_from_config(composed_config, mode="human", render=render, level=int(level))

    log_run_context(
        "play-user",
        {
            "game": str(dict(composed_config.get("game", {})).get("id", args.game)),
            "level": int(level),
            "render": render,
        },
    )

    obs = env.reset()
    del obs
    try:
        while True:
            _, _, done, info = env.step(0)
            if done:
                races_total = info.get("races_total")
                races_finished = info.get("races_finished")
                try:
                    if races_total is not None and races_finished is not None:
                        if int(races_finished) >= int(races_total):
                            break
                except (TypeError, ValueError):
                    pass
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
