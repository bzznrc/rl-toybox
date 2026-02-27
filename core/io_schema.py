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
    distance = max(0.0, float(start_offset))
    while distance <= ray_length:
        px = float(origin_x) + ux * distance
        py = float(origin_y) + uy * distance
        if bool(is_blocked(px, py)):
            return clip_unit(distance / ray_length)
        distance += step

    return 1.0
