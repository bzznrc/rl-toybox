"""Capture a short rendered AI demo directly from the Arcade game window."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import arcade
from PIL import Image

from core import arcade_style
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
from core.runners.eval import reset_eval_policy_state, select_eval_action
from core.utils import PROJECT_ROOT
from scripts import build_common_overrides, missing_model_message, normalize_choice


CAPTURE_DURATION_SECONDS = 15
CAPTURE_FPS = 30
CAPTURE_GIF_COLORS = 32
CAPTURE_GIF_OPTIMIZE = True


def _unique_rgb_colors(
    colors: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    unique: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for color in colors:
        rgb = tuple(max(0, min(255, int(channel))) for channel in color[:3])
        if rgb in seen:
            continue
        seen.add(rgb)
        unique.append(rgb)
    return tuple(unique)


def _shared_arcade_palette_colors() -> tuple[tuple[int, int, int], ...]:
    """Return every shared arcade COLOR_* value so demo GIFs preserve app colors."""

    colors: list[tuple[int, int, int]] = []
    for name in sorted(dir(arcade_style)):
        if not name.startswith("COLOR_"):
            continue
        value = getattr(arcade_style, name)
        if isinstance(value, tuple) and len(value) >= 3:
            colors.append((int(value[0]), int(value[1]), int(value[2])))
    return _unique_rgb_colors(tuple(colors))


SHARED_ARCADE_PALETTE_COLORS = _shared_arcade_palette_colors()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a short rendered AI demo")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--algo", default=None, help="Override algorithm id; use auto/default to keep the game's default")
    parser.add_argument(
        "--model",
        default="best",
        type=normalize_choice,
        choices=["best", "check"],
        help="Model artifact to load",
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
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path to load")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Difficulty selector (defaults to L5 for curriculum games; fixed-mode games use L1)",
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
    return build_common_overrides(
        args,
        mode="eval",
        render=True,
        headless=False,
        include_checkpoint=True,
    )


def _draw_current_frame(env: object) -> None:
    draw_frame = getattr(env, "draw_frame", None)
    if callable(draw_frame):
        draw_frame()


def _adaptive_palette_colors(image: Image.Image, color_count: int) -> tuple[tuple[int, int, int], ...]:
    count = max(0, min(256, int(color_count)))
    if count <= 0:
        return ()
    quantized = image.convert("RGB").quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    colors: list[tuple[int, int, int]] = []
    for idx in range(min(count, len(palette) // 3)):
        offset = idx * 3
        colors.append((int(palette[offset]), int(palette[offset + 1]), int(palette[offset + 2])))
    return _unique_rgb_colors(tuple(colors))


def _palette_image(colors: tuple[tuple[int, int, int], ...]) -> Image.Image:
    palette_values: list[int] = []
    for rgb in colors[:256]:
        palette_values.extend([int(rgb[0]), int(rgb[1]), int(rgb[2])])
    palette_values.extend([0] * max(0, 768 - len(palette_values)))
    image = Image.new("P", (1, 1))
    image.putpalette(palette_values[:768])
    return image


def _quantize_capture_frame(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    color_limit = max(2, min(256, int(CAPTURE_GIF_COLORS)))
    protected_colors = SHARED_ARCADE_PALETTE_COLORS[:color_limit]
    adaptive_count = max(0, color_limit - len(protected_colors))
    palette_colors = _unique_rgb_colors(
        tuple(protected_colors) + _adaptive_palette_colors(rgb, adaptive_count)
    )
    return rgb.quantize(
        palette=_palette_image(palette_colors),
        dither=Image.Dither.NONE,
    )


def _capture_current_frame(env: object) -> Image.Image:
    get_render_window = getattr(env, "get_render_window", None)
    window = get_render_window() if callable(get_render_window) else None
    if window is None:
        raise RuntimeError("Rendered demo capture requires an active Arcade window.")
    return _quantize_capture_frame(
        arcade.get_image(
            x=0,
            y=0,
            width=int(window.width),
            height=int(window.height),
            components=3,
        )
    )


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
        optimize=bool(CAPTURE_GIF_OPTIMIZE),
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

    capture_fps = int(CAPTURE_FPS)
    capture_period_seconds = 1.0 / float(capture_fps)
    target_frame_count = max(1, int(CAPTURE_DURATION_SECONDS) * int(capture_fps))
    output_path = _build_output_path(game_id, level)

    os.environ["RL_TOYBOX_RENDER_VISIBLE"] = "0"
    env = build_env_from_config(composed_config, mode="eval", render=True, level=int(level))
    play_action_repeat_frames = int(env.capture_action_repeat_frames())
    original_action_repeat_frames = getattr(env, "rl_action_repeat_frames", None)
    if original_action_repeat_frames is not None:
        setattr(env, "rl_action_repeat_frames", 1)
    frames: list[Image.Image] = []
    play_elapsed_seconds = 0.0
    next_capture_second = 0.0

    def capture_if_due() -> None:
        nonlocal next_capture_second
        while (
            len(frames) < target_frame_count
            and float(play_elapsed_seconds) >= float(next_capture_second) - 1e-9
        ):
            frames.append(_capture_current_frame(env))
            next_capture_second += float(capture_period_seconds)

    try:
        run_context = {
            "game": game_id,
            "algo": algo_id,
            "level": int(level),
            "render_fps": float(env.capture_render_fps()),
            "capture_fps": int(capture_fps),
            "duration_seconds": int(CAPTURE_DURATION_SECONDS),
            "action_repeat_frames": int(play_action_repeat_frames),
            "model": model_path,
            "output": output_path,
        }
        if game_id == "bang":
            bang_mode = dict(composed_config.get("common", {})).get("bang_mode")
            run_context["bang_mode"] = str(bang_mode).replace("_", " ").title()
        if game_id == "kick":
            run_context["team_size"] = dict(composed_config.get("common", {})).get("team_size")
        log_run_context("capture-demo-ai", run_context)
        reset_eval_policy_state(algorithm)
        obs = env.reset()
        _draw_current_frame(env)
        capture_if_due()

        while len(frames) < target_frame_count:
            action = select_eval_action(env, algorithm, obs)
            done = False
            for _repeat_index in range(int(play_action_repeat_frames)):
                obs, _reward, done, _info = env.step(action)
                play_elapsed_seconds += float(env.capture_step_seconds())
                capture_if_due()
                if done or len(frames) >= target_frame_count:
                    break
            if done:
                reset_eval_policy_state(algorithm)
                obs = env.reset()
                _draw_current_frame(env)

        _save_gif(frames, output_path, capture_fps)
        log_key_values(
            "rl_toybox.capture_demo_ai",
            {
                "Frames": len(frames),
                "Render FPS": float(env.capture_render_fps()),
                "Capture FPS": int(capture_fps),
                "Duration Seconds": int(CAPTURE_DURATION_SECONDS),
                "Action Repeat Frames": int(play_action_repeat_frames),
                "Model": model_path,
                "Output": output_path,
            },
            prefix="Capture Demo Summary",
            key_value_separator=":",
        )
    finally:
        if original_action_repeat_frames is not None:
            setattr(env, "rl_action_repeat_frames", original_action_repeat_frames)
        env.close()


if __name__ == "__main__":
    main()
