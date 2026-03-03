"""Generic human-play entrypoint for supported games."""

from __future__ import annotations

import argparse

from core.logging_utils import configure_logging, log_run_context
from games.registry import get_game_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a game in human-control mode")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--level", type=int, default=3, help="Play level selector (defaults to 3)")
    parser.add_argument("--headless", action="store_true", help="Disable rendering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    spec = get_game_spec(args.game)
    render = not bool(args.headless)
    env = spec.make_env(mode="human", render=render, level=int(args.level))

    log_run_context(
        "play-user",
        {
            "game": spec.game_id,
            "level": int(args.level),
            "render": render,
        },
    )

    obs = env.reset()
    del obs
    try:
        while True:
            _, _, done, _ = env.step(0)
            if done:
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
