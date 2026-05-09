"""Capture a short rendered AI demo directly from the Arcade game window."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import arcade
import numpy as np
from PIL import Image

from core import arcade_style
from core.shared_config import FPS as SHOW_GAME_FPS
from core.game import (
    apply_generic_launch_level,
    apply_seed_from_config,
    build_algo_from_config,
    build_env_from_config,
    normalize_kick_team_size,
    parse_override_assignments,
    prepare_run,
    resolve_latest_play_model_path,
    set_nested_override,
)
from core.logging_utils import configure_logging, log_key_values, log_run_context
from core.runners.env_access import act_with_optional_signals, extract_action_mask, extract_centralized_state
from core.runners.eval import reset_eval_policy_state, select_eval_action
from core.utils import PROJECT_ROOT


CAPTURE_DURATION_SECONDS = 15
CAPTURE_FPS = 30
CAPTURE_SEARCH_PLAY_RANDOM_OPENING_PLIES = 2
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
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
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
    overrides = parse_override_assignments(args.set_values)
    set_nested_override(overrides, "common.mode", "eval")
    set_nested_override(overrides, "common.render", True)
    set_nested_override(overrides, "common.headless", False)
    if args.seed is not None:
        set_nested_override(overrides, "common.seed", int(args.seed))
    if args.team_size is not None:
        set_nested_override(overrides, "common.team_size", int(args.team_size))
    if args.checkpoint:
        set_nested_override(overrides, "common.checkpoint_path", str(Path(args.checkpoint)))
    return overrides


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


def _capture_pre_action_delay_seconds(env: object) -> float:
    delay_fn = getattr(env, "capture_pre_action_delay_seconds", None)
    if not callable(delay_fn):
        return 0.0
    try:
        return max(0.0, float(delay_fn()))
    except (TypeError, ValueError):
        return 0.0


def _capture_repeated_current_frames(
    env: object,
    frames: list[Image.Image],
    *,
    target_frame_count: int,
    frame_count: int,
) -> None:
    repeats = max(0, int(frame_count))
    for _idx in range(repeats):
        if len(frames) >= int(target_frame_count):
            return
        frames.append(_capture_current_frame(env))


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


def _should_use_exploratory_capture(composed_config: dict[str, object]) -> bool:
    algo_id = str(dict(composed_config.get("algo", {})).get("id", "")).strip().lower()
    return algo_id == "search_play"


def _sample_masked_action(action_mask: object | None) -> int | None:
    if action_mask is None:
        return None
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    legal_actions = np.flatnonzero(mask)
    if int(legal_actions.size) <= 0:
        return None
    return int(np.random.choice(legal_actions))


def _select_capture_action(
    env: object,
    algorithm: object,
    obs: object,
    *,
    explore: bool,
    random_opening: bool,
):
    if not bool(explore):
        return select_eval_action(env, algorithm, obs)
    action_mask = extract_action_mask(env, obs)
    if bool(random_opening):
        sampled_action = _sample_masked_action(action_mask)
        if sampled_action is not None:
            return int(sampled_action)
    central_obs = extract_centralized_state(env, obs)
    return act_with_optional_signals(
        algorithm,
        obs,
        explore=True,
        action_mask=action_mask,
        central_obs=central_obs,
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
            model_path = resolve_latest_play_model_path(run_paths, level)
        except FileNotFoundError as exc:
            if game_id == "kick":
                team_size = normalize_kick_team_size(dict(composed_config.get("common", {})).get("team_size"))
                raise FileNotFoundError(
                    f"No trained Kick model was found for level {int(level)}. "
                    f"The selected mode is {team_size} vs. {team_size}. Train the shared Kick model first with: "
                    f"python -m scripts.train --game kick --team-size {team_size} --level {int(level)}"
                ) from None
            raise
    composed_config["common"]["checkpoint_path"] = str(model_path)
    algorithm = build_algo_from_config(composed_config)
    algorithm.load(str(model_path))

    native_fps = int(SHOW_GAME_FPS)
    capture_fps = int(CAPTURE_FPS)
    capture_period_frames = float(native_fps) / float(capture_fps)
    target_frame_count = max(1, int(CAPTURE_DURATION_SECONDS) * int(capture_fps))
    output_path = _build_output_path(game_id, level)
    exploratory_capture = bool(_should_use_exploratory_capture(composed_config))

    os.environ["RL_TOYBOX_RENDER_VISIBLE"] = "0"
    env = build_env_from_config(composed_config, mode="eval", render=True, level=int(level))
    frames: list[Image.Image] = []
    step_index = 0
    episode_step_index = 0
    next_capture_step = 0.0

    def capture_if_due() -> None:
        nonlocal next_capture_step
        while len(frames) < target_frame_count and float(step_index) >= float(next_capture_step) - 1e-9:
            frames.append(_capture_current_frame(env))
            next_capture_step += float(capture_period_frames)

    try:
        run_context = {
            "game": game_id,
            "algo": algo_id,
            "level": int(level),
            "native_fps": int(native_fps),
            "capture_fps": int(capture_fps),
            "duration_seconds": int(CAPTURE_DURATION_SECONDS),
            "explore": bool(exploratory_capture),
            "random_opening_plies": (
                int(CAPTURE_SEARCH_PLAY_RANDOM_OPENING_PLIES) if bool(exploratory_capture) else 0
            ),
            "model": model_path,
            "output": output_path,
        }
        if game_id == "kick":
            run_context["team_size"] = dict(composed_config.get("common", {})).get("team_size")
        log_run_context("capture-demo-ai", run_context)
        reset_eval_policy_state(algorithm)
        obs = env.reset()
        _draw_current_frame(env)
        capture_if_due()

        while len(frames) < target_frame_count:
            pre_action_delay_seconds = _capture_pre_action_delay_seconds(env)
            _capture_repeated_current_frames(
                env,
                frames,
                target_frame_count=int(target_frame_count),
                frame_count=round(float(pre_action_delay_seconds) * float(capture_fps)),
            )
            if len(frames) >= target_frame_count:
                break

            random_opening = bool(
                exploratory_capture
                and int(episode_step_index) < int(CAPTURE_SEARCH_PLAY_RANDOM_OPENING_PLIES)
            )
            action = _select_capture_action(
                env,
                algorithm,
                obs,
                explore=bool(exploratory_capture),
                random_opening=bool(random_opening),
            )
            obs, _reward, done, _info = env.step(action)
            step_index += 1
            episode_step_index += 1
            capture_if_due()
            if done:
                reset_eval_policy_state(algorithm)
                obs = env.reset()
                episode_step_index = 0
                _draw_current_frame(env)

        _save_gif(frames, output_path, capture_fps)
        log_key_values(
            "rl_toybox.capture_demo_ai",
            {
                "Frames": len(frames),
                "Native FPS": int(native_fps),
                "Capture FPS": int(capture_fps),
                "Duration Seconds": int(CAPTURE_DURATION_SECONDS),
                "Explore": bool(exploratory_capture),
                "Random Opening Plies": (
                    int(CAPTURE_SEARCH_PLAY_RANDOM_OPENING_PLIES) if bool(exploratory_capture) else 0
                ),
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
