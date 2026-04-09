"""Shared IO helpers for ordered feature vectors and normalized sensing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math


def clip(value: float, low: float, high: float) -> float:
    return float(max(float(low), min(float(high), float(value))))


def clip_signed(value: float) -> float:
    return clip(float(value), -1.0, 1.0)


def clip_unit(value: float) -> float:
    return clip(float(value), 0.0, 1.0)


def signed_potential_shaping(phi_prev: float, phi_next: float, *, scale: float, clip_abs: float) -> float:
    return clip(float(scale) * (float(phi_next) - float(phi_prev)), -abs(float(clip_abs)), abs(float(clip_abs)))


def normalize_last_action(action_index: int, action_count: int) -> float:
    count = max(1, int(action_count))
    if count <= 1:
        return 0.0
    action = int(max(0, min(int(action_index), count - 1)))
    return float(action) / float(count - 1)


def ordered_feature_vector(feature_names: Sequence[str], feature_values: Mapping[str, float]) -> list[float]:
    names = list(feature_names)
    name_set = set(names)
    missing = [name for name in names if name not in feature_values]
    if missing:
        raise KeyError(f"Missing feature values: {missing}")

    extras = [name for name in feature_values.keys() if name not in name_set]
    if extras:
        raise KeyError(f"Unknown feature values: {extras}")

    return [float(feature_values[name]) for name in names]


def row_major_grid_feature_names(rows: int, cols: int | None = None, *, prefix: str = "cell") -> list[str]:
    row_count = max(0, int(rows))
    col_count = row_count if cols is None else max(0, int(cols))
    return [f"{prefix}_r{row}_c{col}" for row in range(row_count) for col in range(col_count)]


def row_major_grid_action_names(
    rows: int,
    cols: int | None = None,
    *,
    prefix: str = "move",
    include_pass: bool = False,
    pass_name: str = "pass",
) -> list[str]:
    names = row_major_grid_feature_names(rows, cols, prefix=prefix)
    if bool(include_pass):
        names.append(str(pass_name))
    return names


def normalized_ray_first_hit(
    *,
    origin_x: float,
    origin_y: float,
    dir_x: float,
    dir_y: float,
    max_distance: float,
    is_blocked: Callable[[float, float], bool],
    step_size: float = 2.0,
    start_offset: float = 0.0,
    refine_iterations: int = 5,
) -> float:
    ray_length = max(1e-6, float(max_distance))
    ux = float(dir_x)
    uy = float(dir_y)
    mag = math.hypot(ux, uy)
    if mag <= 1e-9:
        ux, uy = 1.0, 0.0
    else:
        ux /= mag
        uy /= mag

    step = max(0.25, float(step_size))
    min_distance = min(ray_length, max(0.0, float(start_offset)))
    distance = min_distance
    prev_distance = min_distance
    prev_blocked = False
    started = False

    while distance <= ray_length:
        px = float(origin_x) + ux * distance
        py = float(origin_y) + uy * distance
        blocked = bool(is_blocked(px, py))
        if blocked:
            if (not started) or prev_blocked:
                return clip_unit(max(0.0, float(distance) - float(min_distance)) / ray_length)

            low = float(prev_distance)
            high = float(distance)
            for _ in range(max(0, int(refine_iterations))):
                mid = 0.5 * (low + high)
                mx = float(origin_x) + ux * mid
                my = float(origin_y) + uy * mid
                if bool(is_blocked(mx, my)):
                    high = mid
                else:
                    low = mid
            hit_distance = 0.5 * (low + high)
            return clip_unit(max(0.0, float(hit_distance) - float(min_distance)) / ray_length)

        prev_distance = float(distance)
        prev_blocked = bool(blocked)
        started = True
        distance += step

    return 1.0
