"""Generic training entrypoint for all registered games."""

from __future__ import annotations

import argparse
import sys

from core.logging_utils import configure_logging, log_key_values, log_run_context
from core.runners.off_policy import OffPolicyConfig, run_off_policy_training
from core.runners.on_policy import OnPolicyConfig, run_on_policy_training
from scripts.common import prepare_run, resolve_resume_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL toybox game")
    parser.add_argument("--game", required=True, help="Game id (bang, snake, vroom, kick, stomp)")
    parser.add_argument("--algo", default=None, help="Override algorithm id")
    parser.add_argument("--render", action="store_true", help="Show Arcade window during training")
    parser.add_argument(
        "--resume",
        default="ask",
        choices=["ask", "auto", "none", "new", "checkpoint", "best"],
        help=(
            "Resume source for model weights. "
            "'ask' prompts New vs Best; when best is loaded, epsilon resets to eps_bump_cap for epsilon-based algos"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override off-policy max training steps")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override on-policy iteration count")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Override checkpoint cadence")
    parser.add_argument("--log-every-episodes", type=int, default=1, help="Episode log cadence for off-policy games")
    parser.add_argument("--log-every-iterations", type=int, default=1, help="Iteration log cadence for on-policy games")
    parser.add_argument(
        "--log-heartbeat-steps",
        type=int,
        default=0,
        help="Optional heartbeat cadence in env steps (0 disables heartbeat)",
    )
    return parser.parse_args()


def _set_resume_best_epsilon_to_bump_cap(algorithm: object) -> float | None:
    exploration = getattr(algorithm, "_exploration", None)
    if exploration is None:
        return None

    config = getattr(exploration, "config", None)
    bump_cap = getattr(config, "eps_bump_cap", None)
    set_epsilon = getattr(exploration, "set_epsilon", None)
    if bump_cap is None or not callable(set_epsilon):
        return None

    updated_epsilon = float(set_epsilon(float(bump_cap)))
    if hasattr(algorithm, "epsilon"):
        setattr(algorithm, "epsilon", updated_epsilon)
    return updated_epsilon


def _prompt_resume_mode() -> str:
    while True:
        print("Train Start")
        print("  [1] New (start from scratch)")
        print("  [2] Best (resume from best model)")
        selection = input("Choose start mode [1/2]: ").strip().lower()

        if selection in {"1", "new", "n"}:
            return "none"
        if selection in {"2", "best", "b"}:
            return "best"

        print("Invalid selection. Please choose 1 (New) or 2 (Best).")


def main() -> None:
    args = parse_args()
    configure_logging()

    prepared = prepare_run(args.game, args.algo)
    spec = prepared.spec
    algo_id = prepared.algo_id
    run_paths = prepared.run_paths
    algorithm = prepared.algorithm

    resume_mode = str(args.resume).strip().lower()
    if resume_mode == "new":
        resume_mode = "none"
    if resume_mode == "ask":
        if sys.stdin.isatty():
            resume_mode = _prompt_resume_mode()
        else:
            resume_mode = "auto"

    resume_path = resolve_resume_path(resume_mode, run_paths.checkpoint_path, run_paths.best_path)
    if resume_mode == "best" and resume_path is None:
        log_key_values(
            "rl_toybox.train",
            {"Resume": "best_missing", "Fallback": "scratch"},
            prefix="Train",
            key_value_separator=":",
        )
    if resume_path is not None:
        algorithm.load(str(resume_path))
        resumed_from_best = resume_path == run_paths.best_path
        if resumed_from_best:
            bumped_epsilon = _set_resume_best_epsilon_to_bump_cap(algorithm)
            if bumped_epsilon is not None:
                log_key_values(
                    "rl_toybox.train",
                    {
                        "Bump": "resume_best",
                        "Epsilon": f"{float(bumped_epsilon):.3f}",
                    },
                    prefix="Explore",
                    key_value_separator=":",
                )

    env = spec.make_env(mode="train", render=bool(args.render))
    try:
        log_run_context(
            "train",
            {
                "game": spec.game_id,
                "algo": algo_id,
                "run": run_paths.run_dir,
                "resume": resume_path if resume_path is not None else "scratch",
                "render": bool(args.render),
            },
        )

        if algo_id == "ppo":
            train_config = dict(spec.train_config)
            if args.max_iterations is not None:
                train_config["max_iterations"] = int(args.max_iterations)
            if args.checkpoint_every is not None:
                train_config["checkpoint_every_iterations"] = int(args.checkpoint_every)
            train_config["log_every_iterations"] = int(args.log_every_iterations)
            train_config["log_heartbeat_steps"] = int(args.log_heartbeat_steps)
            metrics = run_on_policy_training(
                env,
                algorithm,
                run_paths,
                OnPolicyConfig(**train_config),
            )
        else:
            train_config = dict(spec.train_config)
            if args.max_steps is not None:
                train_config["max_steps"] = int(args.max_steps)
            if args.checkpoint_every is not None:
                train_config["checkpoint_every_steps"] = int(args.checkpoint_every)
            train_config["log_every_episodes"] = int(args.log_every_episodes)
            train_config["log_heartbeat_steps"] = int(args.log_heartbeat_steps)
            metrics = run_off_policy_training(
                env,
                algorithm,
                run_paths,
                OffPolicyConfig(**train_config),
            )

        log_key_values("rl_toybox.train", metrics, prefix="Train Summary")
    finally:
        env.close()


if __name__ == "__main__":
    main()
