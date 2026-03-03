"""Generic AI-play entrypoint for all registered games."""

from __future__ import annotations

import argparse

from core.logging_utils import configure_logging, log_key_values, log_run_context
from core.runners.eval import run_eval
from scripts.common import prepare_run, resolve_play_model_path


def _normalize_choice(value: str) -> str:
    return str(value).strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play with a trained RL agent")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--algo", default=None, help="Override algorithm id")
    parser.add_argument(
        "--model",
        default="best",
        type=_normalize_choice,
        choices=["best", "check", "checkpoint"],
        help="Model artifact to load",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--level", type=int, default=3, help="Play level selector (defaults to 3)")
    parser.add_argument("--render", action="store_true", help="Show Arcade window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    prepared = prepare_run(args.game, args.algo)
    spec = prepared.spec
    algo_id = prepared.algo_id
    run_paths = prepared.run_paths
    algorithm = prepared.algorithm

    model_choice = "check" if str(args.model).strip().lower() == "checkpoint" else str(args.model).strip().lower()
    model_path = resolve_play_model_path(run_paths, model_choice, int(args.level))
    algorithm.load(str(model_path))

    env = spec.make_env(mode="eval", render=bool(args.render), level=int(args.level))
    try:
        log_run_context(
            "play-ai",
            {
                "game": spec.game_id,
                "algo": algo_id,
                "model": model_path,
                "episodes": int(args.episodes),
                "level": int(args.level),
                "render": bool(args.render),
            },
        )
        result = run_eval(env, algorithm, episodes=int(args.episodes))
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
