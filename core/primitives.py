"""Shared gameplay and rendering primitives used by multiple games."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, TypeVar

import arcade

from core.runtime import ArcadeWindowController

T = TypeVar("T")


def resolve_circle_collisions(
    positions: list[tuple[float, float]],
    velocities: list[tuple[float, float]],
    radii: list[float],
    *,
    sep_strength: float,
    overlap_cap: float,
    contact_damp: float,
    eps: float = 1e-6,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[bool]]:
    """Resolve deterministic pairwise circle overlaps and apply mild contact damping."""

    count = len(positions)
    if len(velocities) != count or len(radii) != count:
        raise ValueError("positions, velocities, and radii must have the same length")

    pos = [(float(x), float(y)) for x, y in positions]
    vel = [(float(vx), float(vy)) for vx, vy in velocities]
    contact_flags = [False] * count

    strength = max(0.0, float(sep_strength))
    cap = max(0.0, float(overlap_cap))

    for i in range(count):
        for j in range(i + 1, count):
            pix, piy = pos[i]
            pjx, pjy = pos[j]
            dx = pix - pjx
            dy = piy - pjy
            dist = math.hypot(dx, dy)
            overlap = (float(radii[i]) + float(radii[j])) - dist
            if overlap <= 0.0:
                continue

            contact_flags[i] = True
            contact_flags[j] = True

            if dist > eps:
                nx = dx / dist
                ny = dy / dist
            else:
                nx = 1.0
                ny = 0.0

            correction = min(overlap, cap) * strength
            shift = 0.5 * correction
            pos[i] = (pix + shift * nx, piy + shift * ny)
            pos[j] = (pjx - shift * nx, pjy - shift * ny)

    damp_scale = 1.0 - max(0.0, min(1.0, float(contact_damp)))
    for idx in range(count):
        if not contact_flags[idx]:
            continue
        vx, vy = vel[idx]
        vel[idx] = (vx * damp_scale, vy * damp_scale)

    return pos, vel, contact_flags


def _grow_connected_random_walk_shape(
    start: T,
    min_sections: int,
    max_sections: int,
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
) -> list[T]:
    target_sections = random.randint(int(min_sections), int(max_sections))
    shape = [start]
    current = start

    for _ in range(target_sections - 1):
        candidates = list(neighbor_candidates_fn(current))
        random.shuffle(candidates)
        for candidate in candidates:
            if is_candidate_valid_fn(candidate, shape):
                shape.append(candidate)
                current = candidate
                break
        else:
            break
    return shape


def spawn_connected_random_walk_shapes(
    shape_count: int,
    min_sections: int,
    max_sections: int,
    sample_start_fn: Callable[[], T | None],
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
) -> list[list[T]]:
    shapes: list[list[T]] = []
    for _ in range(int(shape_count)):
        start = sample_start_fn()
        if start is None:
            continue
        shape = _grow_connected_random_walk_shape(
            start=start,
            min_sections=min_sections,
            max_sections=max_sections,
            neighbor_candidates_fn=neighbor_candidates_fn,
            is_candidate_valid_fn=is_candidate_valid_fn,
        )
        if shape:
            shapes.append(shape)
    return shapes


def draw_two_tone_tile(
    window_controller: ArcadeWindowController,
    *,
    top_left_x: float,
    top_left_y: float,
    size: float,
    outer_color: tuple[int, int, int] | tuple[int, int, int, int],
    inner_color: tuple[int, int, int] | tuple[int, int, int, int],
    inset: float,
) -> None:
    bottom = window_controller.top_left_to_bottom(top_left_y, size)
    arcade.draw_lbwh_rectangle_filled(top_left_x, bottom, size, size, outer_color)
    inner_size = size - 2.0 * float(inset)
    if inner_size <= 0:
        return
    arcade.draw_lbwh_rectangle_filled(
        top_left_x + inset,
        bottom + inset,
        inner_size,
        inner_size,
        inner_color,
    )


def draw_staggered_square_pattern(
    window_controller: ArcadeWindowController,
    *,
    top_left_x: float,
    top_left_y: float,
    width: float,
    height: float,
    square_size: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    gap_x: float = 0.0,
    gap_y: float | None = None,
    row_offset_ratio: float = 0.5,
) -> None:
    """Draw a staggered tiny-square fill pattern in top-left coordinate space."""

    pattern_left = float(top_left_x)
    pattern_top = float(top_left_y)
    pattern_right = pattern_left + max(0.0, float(width))
    pattern_bottom = pattern_top + max(0.0, float(height))
    if pattern_right <= pattern_left or pattern_bottom <= pattern_top:
        return

    tile_size = max(1.0, float(square_size))
    spacing_x = max(0.0, float(gap_x))
    spacing_y = spacing_x if gap_y is None else max(0.0, float(gap_y))
    step_x = tile_size + spacing_x
    step_y = tile_size + spacing_y
    if step_x <= 0.0 or step_y <= 0.0:
        return

    row_offset = step_x * float(row_offset_ratio)
    row_idx = 0
    y = pattern_top
    while y < pattern_bottom:
        x = pattern_left - (row_offset if (row_idx % 2) else 0.0)
        while x < pattern_right:
            if (x + tile_size) > pattern_left:
                bottom = window_controller.top_left_to_bottom(y, tile_size)
                arcade.draw_lbwh_rectangle_filled(x, bottom, tile_size, tile_size, color)
            x += step_x
        y += step_y
        row_idx += 1


def build_staggered_square_pattern_texture(
    *,
    width: int,
    height: int,
    square_size: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    gap_x: float = 0.0,
    gap_y: float | None = None,
    row_offset_ratio: float = 0.5,
    texture_name: str | None = None,
) -> arcade.Texture:
    """Build a reusable transparent texture with a staggered tiny-square pattern."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Staggered square pattern textures require numpy and Pillow.") from exc

    tex_w = max(1, int(width))
    tex_h = max(1, int(height))
    tile_size = max(1, int(round(float(square_size))))
    spacing_x = max(0, int(round(float(gap_x))))
    spacing_y = spacing_x if gap_y is None else max(0, int(round(float(gap_y))))
    step_x = max(1, tile_size + spacing_x)
    step_y = max(1, tile_size + spacing_y)
    row_offset = int(round(float(step_x) * float(row_offset_ratio)))

    color_rgba = (
        int(color[0]),
        int(color[1]),
        int(color[2]),
        int(color[3]) if len(color) == 4 else 255,
    )

    rgba = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
    row_idx = 0
    y0 = 0
    while y0 < tex_h:
        y1 = min(tex_h, y0 + tile_size)
        x = -row_offset if (row_idx % 2) else 0
        while x < tex_w:
            x0 = max(0, x)
            x1 = min(tex_w, x + tile_size)
            if x1 > x0 and y1 > y0:
                rgba[y0:y1, x0:x1] = color_rgba
            x += step_x
        y0 += step_y
        row_idx += 1

    texture_hash = (
        str(texture_name)
        if texture_name
        else (
            f"staggered_squares_{tex_w}x{tex_h}_s{tile_size}_gx{spacing_x}_gy{spacing_y}_"
            f"o{row_offset}_{color_rgba[0]}_{color_rgba[1]}_{color_rgba[2]}_{color_rgba[3]}"
        )
    )
    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    return arcade.Texture(image=image, hash=texture_hash)


def draw_control_marker(
    window_controller: ArcadeWindowController,
    *,
    center_x: float,
    center_y_top_left: float,
    marker_size: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
) -> None:
    arcade.draw_lbwh_rectangle_filled(
        center_x - marker_size / 2.0,
        window_controller.to_arcade_y(center_y_top_left) - marker_size / 2.0,
        marker_size,
        marker_size,
        color,
    )


def draw_facing_indicator(
    window_controller: ArcadeWindowController,
    *,
    center_x: float,
    center_y_top_left: float,
    angle_degrees: float,
    length: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    line_width: float = 2.0,
) -> None:
    radians = math.radians(float(angle_degrees))
    end_x = center_x + math.cos(radians) * float(length)
    end_y = center_y_top_left + math.sin(radians) * float(length)
    arcade.draw_line(
        center_x,
        window_controller.to_arcade_y(center_y_top_left),
        end_x,
        window_controller.to_arcade_y(end_y),
        color,
        float(line_width),
    )


def status_icon_size(bottom_bar_height: float, tile_size: float) -> float:
    return max(12.0, min(float(bottom_bar_height - 8.0), float(tile_size)))


@dataclass(frozen=True)
class StatusBarLayout:
    center_y: float
    left_panel_left: float
    left_panel_right: float
    score_left: float
    score_right: float
    clock_center_x: float | None
    clock_radius: float
    clock_border_width: float


def status_bar_layout(
    *,
    width: float,
    bottom_bar_height: float,
    tile_size: float,
    cell_inset: float,
    left_padding: float = 8.0,
    right_padding: float = 10.0,
    center_gap: float = 14.0,
    left_panel_width: float = 0.0,
    include_clock: bool = True,
) -> StatusBarLayout:
    """Compute left/center/right regions for a bottom status bar.

    The layout reserves:
    - Left region (`left_panel_left`..`left_panel_right`) for optional text/logs.
    - Center region (`score_left`..`score_right`) for score/history icons.
    - Right region (clock center/radius) for a time indicator, if enabled.
    """

    bar_width = max(1.0, float(width))
    bar_height = max(1.0, float(bottom_bar_height))
    left_pad = max(0.0, float(left_padding))
    right_pad = max(0.0, float(right_padding))
    mid_gap = max(0.0, float(center_gap))
    panel_width = max(0.0, float(left_panel_width))

    icon_size = status_icon_size(bar_height, float(tile_size))
    indicator_diameter = icon_size * math.sqrt(2.0) * 0.8
    indicator_radius = indicator_diameter * 0.5
    indicator_border = max(1.0, round(float(cell_inset) * 0.5))

    left_panel_left = left_pad
    left_panel_right = left_panel_left + panel_width
    score_left = max(left_pad, left_panel_right + (mid_gap if panel_width > 0.0 else 0.0))

    if bool(include_clock):
        clock_center_x = bar_width - right_pad - indicator_radius
        score_right = max(score_left, clock_center_x - indicator_radius - mid_gap)
    else:
        clock_center_x = None
        score_right = max(score_left, bar_width - right_pad)

    return StatusBarLayout(
        center_y=bar_height * 0.5,
        left_panel_left=left_panel_left,
        left_panel_right=left_panel_right,
        score_left=score_left,
        score_right=score_right,
        clock_center_x=clock_center_x,
        clock_radius=indicator_radius if bool(include_clock) else 0.0,
        clock_border_width=indicator_border,
    )


def draw_status_square_icon(
    *,
    center_x: float,
    center_y: float,
    size: float,
    outer_color: tuple[int, int, int] | tuple[int, int, int, int],
    inner_color: tuple[int, int, int] | tuple[int, int, int, int],
    inset: float,
) -> None:
    bottom = center_y - size / 2.0
    left = center_x - size / 2.0
    arcade.draw_lbwh_rectangle_filled(left, bottom, size, size, outer_color)
    inner_size = max(1.0, size - 2.0 * float(inset))
    arcade.draw_lbwh_rectangle_filled(
        left + inset,
        bottom + inset,
        inner_size,
        inner_size,
        inner_color,
    )


def draw_time_pie_indicator(
    *,
    center_x: float,
    center_y: float,
    radius: float,
    border_width: float,
    remaining_ratio: float,
    base_color: tuple[int, int, int] | tuple[int, int, int, int],
    fill_color: tuple[int, int, int] | tuple[int, int, int, int],
    outline_color: tuple[int, int, int] | tuple[int, int, int, int],
    num_segments: int = 96,
) -> None:
    circle_segments = max(16, int(num_segments))
    ratio = max(0.0, min(1.0, float(remaining_ratio)))
    arcade.draw_circle_filled(center_x, center_y, radius, base_color, num_segments=circle_segments)
    inner_radius = max(1.0, radius - border_width)

    if ratio <= 0.0:
        arcade.draw_circle_outline(
            center_x,
            center_y,
            radius,
            outline_color,
            border_width,
            num_segments=circle_segments,
        )
        return

    if ratio >= 1.0:
        arcade.draw_circle_filled(
            center_x,
            center_y,
            inner_radius,
            fill_color,
            num_segments=circle_segments,
        )
        arcade.draw_circle_outline(
            center_x,
            center_y,
            radius,
            outline_color,
            border_width,
            num_segments=circle_segments,
        )
        return

    start_angle = 90.0
    end_angle = start_angle + 360.0 * ratio
    arcade.draw_arc_filled(
        center_x=center_x,
        center_y=center_y,
        width=inner_radius * 2.0,
        height=inner_radius * 2.0,
        color=fill_color,
        start_angle=start_angle,
        end_angle=end_angle,
        num_segments=circle_segments,
    )
    arcade.draw_circle_outline(
        center_x,
        center_y,
        radius,
        outline_color,
        border_width,
        num_segments=circle_segments,
    )
