"""Shared helpers for optional player-ray visualization overlays."""

from __future__ import annotations

from typing import Callable, Sequence

import arcade

from core.arcade_style import COLOR_AQUA, COLOR_FOG_GRAY


def draw_player_rays(
    *,
    origin_x: float,
    origin_y: float,
    ray_dirs: Sequence[tuple[float, float]],
    ray_values: Sequence[float],
    ray_max_distances: Sequence[float],
    to_screen: Callable[[float, float], tuple[float, float]],
    line_width: float = 1.0,
) -> None:
    """Draw player sensor rays using normalized hit fractions."""

    count = min(len(ray_dirs), len(ray_values), len(ray_max_distances))
    if count <= 0:
        return

    sx0, sy0 = to_screen(float(origin_x), float(origin_y))
    width = max(0.5, float(line_width))
    for idx in range(count):
        dir_x, dir_y = ray_dirs[idx]
        value = float(ray_values[idx])
        max_distance = max(0.0, float(ray_max_distances[idx]))
        distance = max(0.0, min(1.0, value)) * max_distance
        hit_x = float(origin_x) + float(dir_x) * distance
        hit_y = float(origin_y) + float(dir_y) * distance
        sx1, sy1 = to_screen(hit_x, hit_y)
        color = COLOR_FOG_GRAY if value >= 1.0 else COLOR_AQUA
        arcade.draw_line(float(sx0), float(sy0), float(sx1), float(sy1), color, width)
