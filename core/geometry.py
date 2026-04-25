"""Small geometry helpers shared by arcade-style environments."""

from __future__ import annotations

import math


def signed_norm_delta(target: float, source: float, scale: float, *, clip: bool = True) -> float:
    denom = max(1e-6, float(scale))
    value = (float(target) - float(source)) / denom
    if not bool(clip):
        return float(value)
    return float(max(-1.0, min(1.0, value)))


def unit_interval_position(value: float, low: float, high: float) -> float:
    lo = float(low)
    hi = float(high)
    denom = max(1e-6, hi - lo)
    return float(max(0.0, min(1.0, (float(value) - lo) / denom)))


def distance_to_interval(value: float, low: float, high: float) -> float:
    val = float(value)
    lo = min(float(low), float(high))
    hi = max(float(low), float(high))
    if lo <= val <= hi:
        return 0.0
    return float(min(abs(val - lo), abs(val - hi)))


def normalized_time_to_contact(
    distance: float,
    relative_speed: float,
    *,
    max_time: float,
) -> float:
    """Return 1.0 for imminent contact and 0.0 for no closing contact."""

    dist = max(0.0, float(distance))
    closing = max(0.0, float(relative_speed))
    horizon = max(1e-6, float(max_time))
    if closing <= 1e-6:
        return 0.0
    time_to_contact = dist / closing
    return float(max(0.0, min(1.0, 1.0 - (time_to_contact / horizon))))


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = float(bx) - float(ax)
    aby = float(by) - float(ay)
    apx = float(px) - float(ax)
    apy = float(py) - float(ay)
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return float(math.hypot(float(px) - float(ax), float(py) - float(ay)))
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    closest_x = float(ax) + t * abx
    closest_y = float(ay) + t * aby
    return float(math.hypot(float(px) - closest_x, float(py) - closest_y))

