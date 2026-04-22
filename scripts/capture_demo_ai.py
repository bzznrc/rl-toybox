"""Capture a short rendered AI demo directly from the Arcade game window."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import arcade
from PIL import Image

from core.shared_config import FPS as SHOW_GAME_FPS
from core.game import (
    apply_generic_launch_level,
    apply_seed_from_config,
    build_algo_from_config,
    build_env_from_config,
    parse_override_assignments,
    prepare_run,
    resolve_latest_play_model_path,
    set_nested_override,
)
from core.logging_utils import configure_logging, log_key_values, log_run_context
from core.runners.eval import reset_eval_policy_state, select_eval_action
from core.utils import PROJECT_ROOT


CAPTURE_DURATION_SECONDS = 15
CAPTURE_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a short rendered AI demo")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--algo", default=None, help="Override algorithm id; use auto/default to keep the game's default")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path to load")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Difficulty selector (defaults to L5 for curriculum games and Osero 6x6)",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Generic override in dotted.path=value form, e.g. --set algo.config.learning_rate=0.0003",
    )
    return parser.parse_args()


def _build_eval_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides = parse_override_assignments(args.set_values)
    set_nested_override(overrides, "common.mode", "eval")
    set_nested_override(overrides, "common.render", True)
    set_nested_override(overrides, "common.headless", False)
    if args.seed is not None:
        set_nested_override(overrides, "common.seed", int(args.seed))
    if args.checkpoint:
        set_nested_override(overrides, "common.checkpoint_path", str(Path(args.checkpoint)))
    return overrides
def _draw_current_frame(env: object) -> None:
    draw_frame = getattr(env, "draw_frame", None)
    if callable(draw_frame):
        draw_frame()


def _capture_current_frame(env: object) -> Image.Image:
    get_render_window = getattr(env, "get_render_window", None)
    window = get_render_window() if callable(get_render_window) else None
    if window is None:
        raise RuntimeError("Rendered demo capture requires an active Arcade window.")
    return arcade.get_image(
        x=0,
        y=0,
        width=int(window.width),
        height=int(window.height),
        components=3,
    ).convert("P", palette=Image.ADAPTIVE, colors=255)


def _build_output_path(game_id: str, level: int) -> Path:
    del level
    target_dir = PROJECT_ROOT / "media"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{str(game_id).strip().lower()}-demo.gif"


def _save_gif(frames: list[Image.Image], output_path: Path, capture_fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames were captured.")
    frame_duration_ms = max(1, round(1000.0 / float(capture_fps)))
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
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
    algorithm = build_algo_from_config(composed_config)

    explicit_checkpoint = dict(composed_config.get("common", {})).get("checkpoint_path")
    if explicit_checkpoint:
        model_path = Path(str(explicit_checkpoint))
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{model_path}'.")
    else:
        model_path = resolve_latest_play_model_path(run_paths, level)
    composed_config["common"]["checkpoint_path"] = str(model_path)
    algorithm.load(str(model_path))

    native_fps = int(SHOW_GAME_FPS)
    capture_fps = int(CAPTURE_FPS)
    capture_period_frames = float(native_fps) / float(capture_fps)
    target_frame_count = max(1, int(CAPTURE_DURATION_SECONDS) * int(capture_fps))
    output_path = _build_output_path(game_id, level)

    os.environ["RL_TOYBOX_RENDER_VISIBLE"] = "0"
    env = build_env_from_config(composed_config, mode="eval", render=True, level=int(level))
    frames: list[Image.Image] = []
    step_index = 0
    next_capture_step = 0.0

    def capture_if_due() -> None:
        nonlocal next_capture_step
        while len(frames) < target_frame_count and float(step_index) >= float(next_capture_step) - 1e-9:
            frames.append(_capture_current_frame(env))
            next_capture_step += float(capture_period_frames)

    try:
        log_run_context(
            "capture-demo-ai",
            {
                "game": game_id,
                "algo": algo_id,
                "level": int(level),
                "native_fps": int(native_fps),
                "capture_fps": int(capture_fps),
                "duration_seconds": int(CAPTURE_DURATION_SECONDS),
                "model": model_path,
                "output": output_path,
            },
        )
        reset_eval_policy_state(algorithm)
        obs = env.reset()
        _draw_current_frame(env)
        capture_if_due()

        while len(frames) < target_frame_count:
            action = select_eval_action(env, algorithm, obs)
            obs, _reward, done, _info = env.step(action)
            step_index += 1
            capture_if_due()
            if done:
                reset_eval_policy_state(algorithm)
                obs = env.reset()
                _draw_current_frame(env)

        _save_gif(frames, output_path, capture_fps)
        log_key_values(
            "rl_toybox.capture_demo_ai",
            {
                "Frames": len(frames),
                "Native FPS": int(native_fps),
                "Capture FPS": int(capture_fps),
                "Duration Seconds": int(CAPTURE_DURATION_SECONDS),
                "Model": model_path,
                "Output": output_path,
            },
            prefix="Capture Demo Summary",
            key_value_separator=":",
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
