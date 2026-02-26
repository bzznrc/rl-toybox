"""Human-play entrypoint for Bang AI."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bang_ai.config import FPS, SHOW_GAME_OVERRIDE
from bang_ai.game import HumanGame
from bang_ai.logging_utils import configure_logging, log_run_context
from bang_ai.utils import resolve_show_game


def run_user() -> None:
    configure_logging()
    show_game = resolve_show_game(SHOW_GAME_OVERRIDE, default_value=True)
    game = HumanGame(show_game=show_game)
    log_run_context(
        "play-user",
        {
            "render": show_game,
            "fps": FPS if show_game else "unlocked",
            "level": game.level,
            "controls": "WASD absolute + Q/E aim + LMB/Space shoot",
        },
    )
    try:
        while True:
            game.play_step()
    finally:
        game.close()


if __name__ == "__main__":
    run_user()
