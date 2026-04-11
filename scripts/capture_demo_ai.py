"""Capture a short rendered AI demo directly from the Arcade game window."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import arcade
from PIL import Image

from core.game import prepare_run, resolve_latest_play_model_path
from core.logging_utils import configure_logging, log_key_values, log_run_context
from core.runners.eval import reset_eval_policy_state, select_eval_action
from core.utils import PROJECT_ROOT


CAPTURE_DURATION_SECONDS = 15
CAPTURE_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a short rendered AI demo")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--algo", default=None, help="Override algorithm id")
    parser.add_argument("--level", type=int, default=3, help="Curriculum level selector")
    return parser.parse_args()


def _resolve_level(spec_family: str, requested_level: int) -> int:
    return 1 if str(spec_family).strip().lower() == "search_play" else max(1, int(requested_level))


def _resolve_game_fps(game_id: str) -> int:
    config_module = importlib.import_module(f"games.{str(game_id).strip().lower()}.config")
    return max(1, int(getattr(config_module, "FPS", 12)))


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

    prepared = prepare_run(args.game, args.algo)
    spec = prepared.spec
    algo_id = prepared.algo_id
    run_paths = prepared.run_paths
    algorithm = prepared.algorithm

    level = _resolve_level(spec.family, int(args.level))
    model_path = resolve_latest_play_model_path(run_paths, level)
    algorithm.load(str(model_path))

    native_fps = _resolve_game_fps(spec.game_id)
    capture_fps = int(CAPTURE_FPS)
    capture_period_frames = float(native_fps) / float(capture_fps)
    target_frame_count = max(1, int(CAPTURE_DURATION_SECONDS) * int(capture_fps))
    output_path = _build_output_path(spec.game_id, level)

    os.environ["RL_TOYBOX_RENDER_VISIBLE"] = "0"
    env = spec.make_env(mode="eval", render=True, level=int(level))
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
                "game": spec.game_id,
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
