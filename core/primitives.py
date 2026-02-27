"""Shared gameplay and rendering primitives used by multiple games."""

from __future__ import annotations

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
