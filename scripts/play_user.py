"""Generic human-play entrypoint for supported games."""

from __future__ import annotations

import argparse

from core.game import (
    apply_generic_launch_level,
    apply_seed_from_config,
    build_algo_from_config,
    build_env_from_config,
    compose_run_config,
    normalize_bang_mode,
    normalize_kick_team_size,
)
from core.io.runs import resolve_run_paths
from core.logging_utils import configure_logging, log_run_context
from scripts import build_common_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a game in human-control mode")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Difficulty selector (defaults to L5 for curriculum games; fixed-mode games use L1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument(
        "--mode",
        type=normalize_bang_mode,
        choices=["duel", "arena", "team_arena"],
        default=None,
        help="Bang mode: Duel, Arena, or Team Arena",
    )
    parser.add_argument(
        "--team-size",
        type=normalize_kick_team_size,
        choices=[3, 5, 7],
        default=None,
        help="Kick team size mode: 3 vs. 3, 5 vs. 5, or 7 vs. 7",
    )
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
    return build_common_overrides(
        args,
        mode="play",
        render=not bool(args.headless),
        headless=bool(args.headless),
        include_checkpoint=False,
    )


def _attach_play_user_ai_opponent(env: object, composed_config: dict[str, object], *, level: int) -> str | None:
    scripted_opponent = getattr(env, "PLAY_USER_OPPONENT", None)
    if str(scripted_opponent).strip().lower() == "scripted":
        return "scripted"

    attach_ai_opponent = getattr(env, "set_ai_opponent", None)
    if not callable(attach_ai_opponent):
        return None

    run_block = dict(composed_config.get("run", {}))
    game_id = str(dict(composed_config.get("game", {})).get("id", "")).strip()
    algo_id = str(dict(composed_config.get("algo", {})).get("id", "")).strip()
    run_name = str(run_block.get("name", "")).strip()
    run_paths = resolve_run_paths(game_id, algo_id, run_name, create=True)
    model_path = run_paths.model_path(int(level), "best")
    if not model_path.exists():
        if game_id == "bang":
            bang_mode = normalize_bang_mode(dict(composed_config.get("common", {})).get("bang_mode"))
            bang_mode_label = str(bang_mode).replace("_", " ").title()
            raise FileNotFoundError(
                f"No trained Bang opponent model was found for level {int(level)} at '{model_path}'. "
                f"The selected mode is {bang_mode_label}. Train it first with: "
                f"python -m scripts.train --game bang --mode {bang_mode} --level {int(level)}"
            )
        if game_id == "kick":
            team_size = normalize_kick_team_size(dict(composed_config.get("common", {})).get("team_size"))
            raise FileNotFoundError(
                f"No trained Kick opponent model was found for level {int(level)} at '{model_path}'. "
                f"The selected mode is {team_size} vs. {team_size}. Train the shared Kick model first with: "
                f"python -m scripts.train --game kick --team-size {team_size} --level {int(level)}"
            )
        raise FileNotFoundError(f"No BEST model found for play-user opponent at '{model_path}'.")

    algorithm = build_algo_from_config(composed_config)
    algorithm.load(str(model_path))
    attach_ai_opponent(algorithm)
    return str(model_path)


def main() -> None:
    args = parse_args()
    configure_logging()
    level = int(apply_generic_launch_level(args.game, args.level, mode="play"))

    composed_config = compose_run_config(args.game, mode="play", user_overrides=_build_play_overrides(args))
    apply_seed_from_config(composed_config)
    render = bool(dict(composed_config.get("common", {})).get("render", not bool(args.headless)))
    env = build_env_from_config(composed_config, mode="human", render=render, level=int(level))
    try:
        opponent_model = _attach_play_user_ai_opponent(env, composed_config, level=int(level))

        game_id = str(dict(composed_config.get("game", {})).get("id", args.game))
        run_context = {
            "game": game_id,
            "level": int(level),
            "render": render,
            "opponent": opponent_model,
        }
        if game_id == "bang":
            bang_mode = dict(composed_config.get("common", {})).get("bang_mode")
            run_context["bang_mode"] = str(bang_mode).replace("_", " ").title()
        if game_id == "kick":
            run_context["team_size"] = dict(composed_config.get("common", {})).get("team_size")
        log_run_context("play-user", run_context)

        obs = env.reset()
        del obs
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
