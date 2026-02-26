"""Generic training entrypoint for all registered games."""

from __future__ import annotations

import argparse

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
        default="auto",
        choices=["auto", "none", "checkpoint", "best"],
        help="Resume source for model weights",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override off-policy max training steps")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override on-policy iteration count")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Override checkpoint cadence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    prepared = prepare_run(args.game, args.algo)
    spec = prepared.spec
    algo_id = prepared.algo_id
    run_paths = prepared.run_paths
    algorithm = prepared.algorithm

    resume_path = resolve_resume_path(args.resume, run_paths.checkpoint_path, run_paths.best_path)
    if resume_path is not None:
        algorithm.load(str(resume_path))

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
