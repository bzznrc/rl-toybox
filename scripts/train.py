"""Generic training entrypoint for all registered games."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.game import (
    ACTIVE_GAME_ORDER,
    apply_generic_launch_level,
    apply_seed_from_config,
    apply_training_start_level,
    build_algo_from_config,
    build_env_from_config,
    build_runner_from_config,
    parse_override_assignments,
    prepare_run,
    resolve_best_resume_level,
    resolve_resume_path,
    set_nested_override,
)
from core.logging_utils import configure_logging, log_key_values, log_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL toybox game")
    parser.add_argument("--game", required=True, help=f"Game id ({', '.join(ACTIVE_GAME_ORDER)})")
    parser.add_argument("--algo", default=None, help="Override algorithm id; use auto/default to keep the game's default")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Shared difficulty selector (defaults to L1 for curriculum games and Osero 6x6)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Normalized training step budget")
    parser.add_argument("--episodes", type=int, default=None, help="Normalized episode/game budget")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument("--render", action="store_true", help="Show Arcade window during training")
    parser.add_argument("--headless", action="store_true", help="Force headless training mode")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path to load before training")
    parser.add_argument("--save-every", type=int, default=None, help="Normalized checkpoint cadence override")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Generic override in dotted.path=value form, e.g. --set algo.config.learning_rate=0.0003",
    )
    parser.add_argument(
        "--resume",
        default="none",
        choices=["auto", "none", "check", "best"],
        help=(
            "Resume source for model weights. "
            "Use 'best', 'check', or 'auto' to reuse saved weights. "
            "When best is loaded, epsilon resets to eps_bump_cap for epsilon-based algos."
        ),
    )
    parser.add_argument(
        "--resume-level",
        type=int,
        default=None,
        help=(
            "Curriculum level to resume from when --resume best is selected. "
            "Loads L<level>_best and starts training from that level."
        ),
    )
    return parser.parse_args()


def _build_train_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides = parse_override_assignments(args.set_values)
    render_enabled = bool(args.render) and not bool(args.headless)
    set_nested_override(overrides, "common.mode", "train")
    set_nested_override(overrides, "common.render", bool(render_enabled))
    set_nested_override(overrides, "common.headless", bool(args.headless or not render_enabled))
    if args.steps is not None:
        set_nested_override(overrides, "common.total_steps", int(args.steps))
    if args.episodes is not None:
        set_nested_override(overrides, "common.episodes", int(args.episodes))
    if args.seed is not None:
        set_nested_override(overrides, "common.seed", int(args.seed))
    if args.save_every is not None:
        set_nested_override(overrides, "common.save_every", int(args.save_every))
    if args.checkpoint:
        set_nested_override(overrides, "common.checkpoint_path", str(Path(args.checkpoint)))
    return overrides


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


def main() -> None:
    args = parse_args()
    configure_logging()
    launch_level = int(apply_generic_launch_level(args.game, args.level, mode="train"))

    prepared = prepare_run(args.game, args.algo, mode="train", user_overrides=_build_train_overrides(args))
    run_paths = prepared.run_paths
    composed_config = prepared.config
    game_id = str(dict(composed_config.get("game", {})).get("id", args.game))
    algo_id = str(dict(composed_config.get("algo", {})).get("id", args.algo or ""))
    apply_seed_from_config(composed_config)
    algorithm = build_algo_from_config(composed_config)
    runner = build_runner_from_config(composed_config)

    resume_mode = args.resume

    env = build_env_from_config(composed_config, mode="train")
    try:
        if resume_mode == "best":
            target_level = int(
                resolve_best_resume_level(
                    env,
                    explicit_level=args.resume_level if args.resume_level is not None else int(launch_level),
                    allow_prompt=False,
                )
            )
            current_level = apply_training_start_level(env, target_level)
        else:
            current_level = apply_training_start_level(env, int(launch_level))
        best_path_for_level = run_paths.model_path(current_level, "best")
        explicit_checkpoint = dict(composed_config.get("common", {})).get("checkpoint_path")
        resume_path = Path(str(explicit_checkpoint)) if explicit_checkpoint else resolve_resume_path(
            resume_mode,
            run_paths,
            current_level,
        )
        if resume_mode == "best" and resume_path is None:
            log_key_values(
                "rl_toybox.train",
                {"Resume": "best_missing", "Level": current_level, "Fallback": "scratch"},
                prefix="Train",
                key_value_separator=":",
            )
        if resume_mode == "check" and resume_path is None:
            log_key_values(
                "rl_toybox.train",
                {"Resume": "check_missing", "Level": current_level, "Fallback": "scratch"},
                prefix="Train",
                key_value_separator=":",
            )
        if resume_path is not None:
            if not Path(resume_path).exists():
                raise FileNotFoundError(f"Checkpoint not found at '{resume_path}'.")
            algorithm.load(str(resume_path))
            composed_config["common"]["checkpoint_path"] = str(resume_path)
            resumed_from_best = resume_path == best_path_for_level
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

        log_run_context(
            "train",
            {
                "game": game_id,
                "algo": algo_id,
                "net": dict(dict(composed_config.get("algo", {})).get("config", {})).get("hidden_sizes"),
                "critic_net": dict(dict(composed_config.get("algo", {})).get("config", {})).get("critic_hidden_sizes"),
                "budget": dict(dict(composed_config.get("run", {})).get("train", {})).get("budget"),
                "run": run_paths.run_dir,
                "level": int(current_level),
                "resume": resume_path if resume_path is not None else "scratch",
                "render": bool(dict(composed_config.get("common", {})).get("render", False)),
            },
        )

        metrics = runner(env, algorithm, run_paths)
        log_key_values("rl_toybox.train", metrics, prefix="Train Summary")
    finally:
        env.close()


if __name__ == "__main__":
    main()
