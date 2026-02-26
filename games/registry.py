"""Game registry mapping game IDs to specs."""

from __future__ import annotations

from games.bang.spec import SPEC as BANG_SPEC
from games.kick.spec import SPEC as KICK_SPEC
from games.snake.spec import SPEC as SNAKE_SPEC
from games.spec_types import GameSpec
from games.stomp.spec import SPEC as STOMP_SPEC
from games.vroom.spec import SPEC as VROOM_SPEC


GAME_SPECS: dict[str, GameSpec] = {
    BANG_SPEC.game_id: BANG_SPEC,
    SNAKE_SPEC.game_id: SNAKE_SPEC,
    VROOM_SPEC.game_id: VROOM_SPEC,
    KICK_SPEC.game_id: KICK_SPEC,
    STOMP_SPEC.game_id: STOMP_SPEC,
}


def get_game_spec(game_id: str) -> GameSpec:
    game_key = str(game_id).strip().lower()
    if game_key not in GAME_SPECS:
        valid = ", ".join(sorted(GAME_SPECS.keys()))
        raise KeyError(f"Unknown game '{game_id}'. Valid options: {valid}")
    return GAME_SPECS[game_key]
