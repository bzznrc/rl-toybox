"""Generic AI-play entrypoint for all registered games."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.game import (
    apply_generic_launch_level,
    apply_seed_from_config,
    build_algo_from_config,
    build_env_from_config,
    normalize_bang_mode,
    normalize_kick_team_size,
    prepare_run,
    resolve_play_model_path,
)
from core.logging_utils import configure_logging, log_key_values, log_run_context
from core.runners.eval import run_eval
from scripts import build_common_overrides, missing_model_message, normalize_choice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play with a trained RL agent")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--algo", default=None, help="Override algorithm id; use auto/default to keep the game's default")
    parser.add_argument(
        "--model",
        default="best",
        type=normalize_choice,
        choices=["best", "check"],
        help="Model artifact to load",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
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
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Difficulty selector (defaults to L5 for curriculum games; fixed-mode games use L1)",
    )
    parser.add_argument("--render", action="store_true", help="Show Arcade window")
    parser.add_argument("--headless", action="store_true", help="Force headless evaluation")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path to load")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Generic override in dotted.path=value form, e.g. --set algo.config.learning_rate=0.0003",
    )
    return parser.parse_args()


def _build_eval_overrides(args: argparse.Namespace) -> dict[str, object]:
    render_enabled = bool(args.render) and not bool(args.headless)
    return build_common_overrides(
        args,
        mode="eval",
        render=bool(render_enabled),
        headless=bool(args.headless or not render_enabled),
        episodes=int(args.episodes),
        include_checkpoint=True,
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    level = int(apply_generic_launch_level(args.game, args.level, mode="eval"))

    prepared = prepare_run(args.game, args.algo, mode="eval", user_overrides=_build_eval_overrides(args))
    run_paths = prepared.run_paths
    composed_config = prepared.config
    game_id = str(dict(composed_config.get("game", {})).get("id", args.game))
    algo_id = str(dict(composed_config.get("algo", {})).get("id", args.algo or ""))
    apply_seed_from_config(composed_config)

    explicit_checkpoint = dict(composed_config.get("common", {})).get("checkpoint_path")
    if explicit_checkpoint:
        model_path = Path(str(explicit_checkpoint))
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{model_path}'.")
    else:
        try:
            model_path = resolve_play_model_path(run_paths, str(args.model).strip().lower(), int(level))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                missing_model_message(
                    game_id=game_id,
                    algo_id=algo_id,
                    run_name=str(dict(composed_config.get("run", {})).get("name", "")),
                    level=int(level),
                    bang_mode=dict(composed_config.get("common", {})).get("bang_mode"),
                    team_size=dict(composed_config.get("common", {})).get("team_size"),
                    original_error=exc,
                )
            ) from None
    composed_config["common"]["checkpoint_path"] = str(model_path)
    algorithm = build_algo_from_config(composed_config)
    algorithm.load(str(model_path))

    env = build_env_from_config(composed_config, mode="eval", level=int(level))
    try:
        run_context = {
            "game": game_id,
            "algo": algo_id,
            "model": model_path,
            "episodes": int(dict(composed_config.get("common", {})).get("episodes", args.episodes)),
            "level": int(level),
            "render": bool(dict(composed_config.get("common", {})).get("render", False)),
        }
        if game_id == "bang":
            bang_mode = dict(composed_config.get("common", {})).get("bang_mode")
            run_context["bang_mode"] = str(bang_mode).replace("_", " ").title()
        if game_id == "kick":
            run_context["team_size"] = dict(composed_config.get("common", {})).get("team_size")
        log_run_context("play-ai", run_context)
        result = run_eval(
            env,
            algorithm,
            episodes=int(dict(composed_config.get("common", {})).get("episodes", args.episodes)),
        )
        log_key_values(
            "rl_toybox.play_ai",
            {
                "Episodes": result.episodes,
                "Avg Reward": result.avg_reward,
                "Avg Length": result.avg_length,
                "Wins": result.wins,
            },
            prefix="Play AI Summary",
            key_value_separator=":",
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
