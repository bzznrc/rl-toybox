"""Canonical geometry-first track model for Vroom."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SIDE_ORDER = ("top", "right", "bottom", "left")
LONG_SIDE_ORDER = ("top", "bottom")
LONG_SIDE_TEMPLATE_CHOICES = ("straight", "bell", "s_curve", "fold")
SHORT_SIDE_TEMPLATE_CHOICES = ("straight", "bell")
FOLD_TEMPLATE_CHOICES = ("fold",)


@dataclass(frozen=True)
class TrackProjection:
    seg_index: int
    segment_t: float
    point: tuple[float, float]
    distance: float
    s: float
    lateral_offset: float
    tangent: tuple[float, float]
    normal: tuple[float, float]


@dataclass(frozen=True)
class TrackGeometry:
    centerline: np.ndarray
    arc_s: np.ndarray
    length: float
    tangents: np.ndarray
    normals: np.ndarray
    half_width: float
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    road_polygon: np.ndarray
    start_s: float
    start_pos: tuple[float, float]
    start_tangent: tuple[float, float]
    start_normal: tuple[float, float]
    start_side: str
    start_index: int
    start_line: tuple[tuple[float, float], tuple[float, float]]
    start_side_delta_min: float
    start_side_delta_max: float
    start_strip_s_min: float
    start_strip_s_max: float
    start_straight_len_px: float
    main_corner_s: tuple[float, float, float, float]
    template_family: str
    side_templates: tuple[tuple[str, str], ...]
    curved_sides: tuple[str, ...]
    _segment_vectors: np.ndarray
    _segment_lengths: np.ndarray
    _segment_s: np.ndarray
    _road_edge_starts: np.ndarray
    _road_edge_vectors: np.ndarray


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    mag = math.hypot(float(dx), float(dy))
    if mag <= 1e-12:
        return 1.0, 0.0
    return float(dx) / mag, float(dy) / mag


def _smoothstep(value: float) -> float:
    t = max(0.0, min(1.0, float(value)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _wrap_s(s: float, length: float) -> float:
    if length <= 1e-12:
        return 0.0
    return float(s) % float(length)


def _wrap_delta(target_s: float, reference_s: float, length: float) -> float:
    if length <= 1e-12:
        return 0.0
    return (float(target_s) - float(reference_s) + 0.5 * float(length)) % float(length) - 0.5 * float(length)


def _append_no_dup(target: list[tuple[float, float]], source: list[tuple[float, float]]) -> tuple[int, int]:
    if not source:
        return len(target), max(0, len(target) - 1)
    if not target:
        start = 0
        target.extend(source)
        return start, len(target) - 1

    sx, sy = source[0]
    tx, ty = target[-1]
    if abs(float(sx) - float(tx)) < 1e-9 and abs(float(sy) - float(ty)) < 1e-9:
        start = len(target) - 1
        target.extend(source[1:])
        return start, len(target) - 1

    start = len(target)
    target.extend(source)
    return start, len(target) - 1


def _sample_t_values(count: int, extra_values: tuple[float, ...] = ()) -> list[float]:
    values = [float(i) / float(max(1, int(count) - 1)) for i in range(max(2, int(count)))]
    values.extend(_clamp(float(value), 0.0, 1.0) for value in extra_values)
    values = sorted(values)

    out: list[float] = []
    for value in values:
        if not out or abs(float(value) - float(out[-1])) > 1e-6:
            out.append(float(value))
    return out


def _line_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    count: int,
    *,
    template_kind: str,
    template_amp: float,
    template_normal: tuple[float, float],
) -> list[tuple[float, float]]:
    point_count = max(3, int(count))
    nx, ny = float(template_normal[0]), float(template_normal[1])
    out: list[tuple[float, float]] = []
    for t in _sample_t_values(point_count):
        x = float(p0[0]) + (float(p1[0]) - float(p0[0])) * t
        y = float(p0[1]) + (float(p1[1]) - float(p0[1])) * t
        disp = 0.0
        if float(template_amp) > 0.0:
            kind = str(template_kind).strip().lower()
            fade = math.sin(math.pi * t) ** 2
            if kind == "bell":
                # Smooth inward single-lobe indentation.
                disp = -float(template_amp) * fade
            elif kind == "s_curve":
                # A regular, analytic chicane. The envelope keeps side joins flat
                # while avoiding the kinkier piecewise midpoint.
                disp = 0.72 * float(template_amp) * math.sin(2.0 * math.pi * t) * fade
            elif kind != "straight":
                raise ValueError(f"Unsupported long-side template '{template_kind}'.")
        if abs(float(disp)) > 1e-9:
            x += nx * disp
            y += ny * disp
        out.append((float(x), float(y)))
    return out


def _smooth_window(t: float, start_in: float, end_in: float, start_out: float, end_out: float) -> float:
    if float(t) <= float(start_in) or float(t) >= float(end_out):
        return 0.0
    if float(t) < float(end_in):
        return _smoothstep((float(t) - float(start_in)) / max(1e-9, float(end_in) - float(start_in)))
    if float(t) <= float(start_out):
        return 1.0
    return 1.0 - _smoothstep((float(t) - float(start_out)) / max(1e-9, float(end_out) - float(start_out)))


def _fold_displacement(template_kind: str, t: float, depth: float) -> float:
    kind = str(template_kind).strip().lower()
    if kind == "fold":
        return -float(depth) * _smooth_window(float(t), 0.04, 0.36, 0.64, 0.96)
    raise ValueError(f"Unsupported fold template '{template_kind}'.")


def _fold_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    template_kind: str,
    template_depth: float,
    template_normal: tuple[float, float],
    sample_spacing_px: float,
) -> list[tuple[float, float]]:
    depth = max(0.0, float(template_depth))
    if depth <= 1.0:
        side_len = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
        return _line_points(
            p0,
            p1,
            max(5, int(side_len / max(1.0, float(sample_spacing_px))) + 1),
            template_kind="straight",
            template_amp=0.0,
            template_normal=template_normal,
        )

    tx, ty = _normalize(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
    nx, ny = float(template_normal[0]), float(template_normal[1])
    side_len = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
    kind = str(template_kind).strip().lower()
    if kind not in FOLD_TEMPLATE_CHOICES:
        raise ValueError(f"Unsupported fold template '{template_kind}'.")

    point_count = max(16, int((float(side_len) + 2.0 * float(depth)) / max(1.0, float(sample_spacing_px) * 0.45)) + 1)
    out: list[tuple[float, float]] = []
    for t in _sample_t_values(point_count):
        disp = _fold_displacement(kind, t, depth)
        x = float(p0[0]) + float(tx) * float(side_len) * t + float(nx) * float(disp)
        y = float(p0[1]) + float(ty) * float(side_len) * t + float(ny) * float(disp)
        out.append((float(x), float(y)))
    return out


def _side_template_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    count: int,
    *,
    template_kind: str,
    template_amp: float,
    template_normal: tuple[float, float],
    sample_spacing_px: float,
) -> list[tuple[float, float]]:
    kind = str(template_kind).strip().lower()
    if kind in FOLD_TEMPLATE_CHOICES:
        return _fold_points(
            p0,
            p1,
            template_kind=kind,
            template_depth=float(template_amp),
            template_normal=template_normal,
            sample_spacing_px=float(sample_spacing_px),
        )
    return _line_points(
        p0,
        p1,
        count,
        template_kind=kind,
        template_amp=float(template_amp),
        template_normal=template_normal,
    )


def _arc_points(
    center_x: float,
    center_y: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    count: int,
) -> list[tuple[float, float]]:
    point_count = max(6, int(count))
    out: list[tuple[float, float]] = []
    for i in range(point_count):
        t = float(i) / float(max(1, point_count - 1))
        angle = float(start_angle) + (float(end_angle) - float(start_angle)) * t
        out.append(
            (
                float(center_x) + math.cos(angle) * float(radius),
                float(center_y) + math.sin(angle) * float(radius),
            )
        )
    return out


def _closed_segment_lengths(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if points.shape[0] <= 1:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64), 0.0
    seg_vec = np.roll(points, -1, axis=0) - points
    seg_len = np.linalg.norm(seg_vec, axis=1)
    return seg_vec, seg_len, float(np.sum(seg_len))


def _resample_closed_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    count = int(points.shape[0])
    if count <= 2:
        return np.asarray(points, dtype=np.float64)

    seg_vec, seg_len, total_len = _closed_segment_lengths(points)
    if total_len <= 1e-9:
        return np.asarray(points, dtype=np.float64)

    seg_s = np.zeros((count,), dtype=np.float64)
    if count > 1:
        seg_s[1:] = np.cumsum(seg_len[:-1], dtype=np.float64)

    target_count = max(24, int(sample_count))
    sample_s = np.linspace(0.0, float(total_len), num=target_count, endpoint=False, dtype=np.float64)
    seg_idx = np.searchsorted(seg_s, sample_s, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, count - 1)
    local_s = sample_s - seg_s[seg_idx]
    local_den = np.maximum(seg_len[seg_idx], 1e-9)
    local_t = np.clip(local_s / local_den, 0.0, 1.0)
    sampled = points[seg_idx] + seg_vec[seg_idx] * local_t[:, None]
    return np.asarray(sampled, dtype=np.float64)


def _smooth_closed_centerline(
    points: np.ndarray,
    *,
    fixed_mask: np.ndarray | None = None,
    iterations: int = 2,
    strength: float = 0.35,
) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64).copy()
    count = int(out.shape[0])
    if count < 6:
        return out

    fixed = np.zeros((count,), dtype=bool)
    if fixed_mask is not None:
        mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if int(mask.shape[0]) == count:
            fixed = mask
    locked = out.copy()
    blend = _clamp(float(strength), 0.0, 0.95)
    for _ in range(max(0, int(iterations))):
        previous = np.roll(out, 1, axis=0)
        next_point = np.roll(out, -1, axis=0)
        target = 0.5 * (previous + next_point)
        smoothed = out + (target - out) * float(blend)
        if bool(np.any(fixed)):
            smoothed[fixed] = locked[fixed]
        out = smoothed
    return np.asarray(out, dtype=np.float64)


def _segments_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> bool:
    def _orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (
            (float(q[0]) - float(p[0])) * (float(r[1]) - float(p[1]))
            - (float(q[1]) - float(p[1])) * (float(r[0]) - float(p[0]))
        )

    def _on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return (
            min(float(p[0]), float(r[0])) - 1e-9 <= float(q[0]) <= max(float(p[0]), float(r[0])) + 1e-9
            and min(float(p[1]), float(r[1])) - 1e-9 <= float(q[1]) <= max(float(p[1]), float(r[1])) + 1e-9
        )

    o1 = _orient(a0, a1, b0)
    o2 = _orient(a0, a1, b1)
    o3 = _orient(b0, b1, a0)
    o4 = _orient(b0, b1, a1)

    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return True
    if abs(o1) <= 1e-9 and _on_segment(a0, b0, a1):
        return True
    if abs(o2) <= 1e-9 and _on_segment(a0, b1, a1):
        return True
    if abs(o3) <= 1e-9 and _on_segment(b0, a0, b1):
        return True
    if abs(o4) <= 1e-9 and _on_segment(b0, a1, b1):
        return True
    return False


def _segments_are_neighbors(i: int, j: int, count: int) -> bool:
    if int(i) == int(j):
        return True
    gap = abs(int(i) - int(j))
    return bool(gap <= 1 or gap >= int(count) - 1)


def _segment_bbox_arrays(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_points = np.roll(points, -1, axis=0)
    min_x = np.minimum(points[:, 0], next_points[:, 0])
    max_x = np.maximum(points[:, 0], next_points[:, 0])
    min_y = np.minimum(points[:, 1], next_points[:, 1])
    max_y = np.maximum(points[:, 1], next_points[:, 1])
    return min_x, max_x, min_y, max_y


def _segment_grid(
    min_x: np.ndarray,
    max_x: np.ndarray,
    min_y: np.ndarray,
    max_y: np.ndarray,
    *,
    cell_size: float,
    padding: float = 0.0,
) -> tuple[dict[tuple[int, int], list[int]], list[tuple[int, int, int, int]]]:
    cell = max(1.0, float(cell_size))
    pad = max(0.0, float(padding))
    grid: dict[tuple[int, int], list[int]] = {}
    ranges: list[tuple[int, int, int, int]] = []
    for idx in range(int(min_x.shape[0])):
        ix0 = int(math.floor((float(min_x[idx]) - pad) / cell))
        ix1 = int(math.floor((float(max_x[idx]) + pad) / cell))
        iy0 = int(math.floor((float(min_y[idx]) - pad) / cell))
        iy1 = int(math.floor((float(max_y[idx]) + pad) / cell))
        ranges.append((ix0, ix1, iy0, iy1))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                grid.setdefault((ix, iy), []).append(idx)
    return grid, ranges


def _has_self_intersection(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    if count < 4:
        return False
    arr = np.asarray(points, dtype=np.float64)
    min_x_arr, max_x_arr, min_y_arr, max_y_arr = _segment_bbox_arrays(arr)
    grid, ranges = _segment_grid(
        min_x_arr,
        max_x_arr,
        min_y_arr,
        max_y_arr,
        cell_size=64.0,
    )
    seen_pairs: set[tuple[int, int]] = set()
    for i in range(count):
        a0 = points[i]
        a1 = points[(i + 1) % count]
        ix0, ix1, iy0, iy1 = ranges[i]
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for j in grid.get((ix, iy), []):
                    if j <= i:
                        continue
                    pair = (int(i), int(j))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if _segments_are_neighbors(i, j, count):
                        continue
                    if max_x_arr[i] + 1e-9 < min_x_arr[j]:
                        continue
                    if max_x_arr[j] + 1e-9 < min_x_arr[i]:
                        continue
                    if max_y_arr[i] + 1e-9 < min_y_arr[j]:
                        continue
                    if max_y_arr[j] + 1e-9 < min_y_arr[i]:
                        continue
                    b0 = points[j]
                    b1 = points[(j + 1) % count]
                    if _segments_intersect(a0, a1, b0, b1):
                        return True
    return False

def has_self_intersection(points: list[tuple[float, float]] | np.ndarray) -> bool:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return False
    clean = [(float(row[0]), float(row[1])) for row in arr]
    return _has_self_intersection(clean)


def _point_segment_distance_sq(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return (px - ax) * (px - ax) + (py - ay) * (py - ay)
    t = _clamp(((px - ax) * dx + (py - ay) * dy) / den, 0.0, 1.0)
    qx = ax + dx * t
    qy = ay + dy * t
    return (px - qx) * (px - qx) + (py - qy) * (py - qy)


def _segment_distance(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> float:
    if _segments_intersect(a0, a1, b0, b1):
        return 0.0
    min_sq = min(
        _point_segment_distance_sq(a0, b0, b1),
        _point_segment_distance_sq(a1, b0, b1),
        _point_segment_distance_sq(b0, a0, a1),
        _point_segment_distance_sq(b1, a0, a1),
    )
    return float(math.sqrt(max(0.0, float(min_sq))))


def min_non_neighbor_segment_distance(
    points: list[tuple[float, float]] | np.ndarray,
    *,
    neighbor_window: int = 20,
    early_stop_distance: float | None = None,
) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] != 2:
        return float("inf")

    count = int(arr.shape[0])
    best = float("inf")
    threshold = None if early_stop_distance is None else max(0.0, float(early_stop_distance))
    min_x_arr, max_x_arr, min_y_arr, max_y_arr = _segment_bbox_arrays(arr)

    if threshold is not None:
        grid, ranges = _segment_grid(
            min_x_arr,
            max_x_arr,
            min_y_arr,
            max_y_arr,
            cell_size=max(16.0, float(threshold)),
            padding=float(threshold),
        )
        seen_pairs: set[tuple[int, int]] = set()
        for i in range(count):
            a0 = (float(arr[i, 0]), float(arr[i, 1]))
            a1 = (float(arr[(i + 1) % count, 0]), float(arr[(i + 1) % count, 1]))
            ix0, ix1, iy0, iy1 = ranges[i]
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    for j in grid.get((ix, iy), []):
                        if j <= i:
                            continue
                        gap = min(j - i, count - (j - i))
                        if gap <= int(neighbor_window):
                            continue
                        pair = (int(i), int(j))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        limit = min(best, threshold)
                        if min_x_arr[i] - limit > max_x_arr[j]:
                            continue
                        if min_x_arr[j] - limit > max_x_arr[i]:
                            continue
                        if min_y_arr[i] - limit > max_y_arr[j]:
                            continue
                        if min_y_arr[j] - limit > max_y_arr[i]:
                            continue
                        b0 = (float(arr[j, 0]), float(arr[j, 1]))
                        b1 = (float(arr[(j + 1) % count, 0]), float(arr[(j + 1) % count, 1]))
                        dist = _segment_distance(a0, a1, b0, b1)
                        best = min(best, float(dist))
                        if best < threshold:
                            return float(best)
        return float(best if best < float("inf") else threshold)

    for i in range(count):
        a0 = (float(arr[i, 0]), float(arr[i, 1]))
        a1 = (float(arr[(i + 1) % count, 0]), float(arr[(i + 1) % count, 1]))
        for j in range(i + 1, count):
            gap = min(j - i, count - (j - i))
            if gap <= int(neighbor_window):
                continue
            b0 = (float(arr[j, 0]), float(arr[j, 1]))
            b1 = (float(arr[(j + 1) % count, 0]), float(arr[(j + 1) % count, 1]))
            limit = best
            if min_x_arr[i] - limit > max_x_arr[j]:
                continue
            if min_x_arr[j] - limit > max_x_arr[i]:
                continue
            if min_y_arr[i] - limit > max_y_arr[j]:
                continue
            if min_y_arr[j] - limit > max_y_arr[i]:
                continue
            dist = _segment_distance(a0, a1, b0, b1)
            best = min(best, float(dist))
    return float(best)


def _build_projection(
    centerline: np.ndarray,
    seg_vec: np.ndarray,
    seg_len: np.ndarray,
    seg_s: np.ndarray,
    length: float,
    x: float,
    y: float,
) -> TrackProjection:
    count = int(centerline.shape[0])
    if count <= 0:
        return TrackProjection(
            seg_index=0,
            segment_t=0.0,
            point=(0.0, 0.0),
            distance=0.0,
            s=0.0,
            lateral_offset=0.0,
            tangent=(1.0, 0.0),
            normal=(0.0, 1.0),
        )

    point_dtype = centerline.dtype if np.issubdtype(centerline.dtype, np.floating) else np.float64
    point = np.asarray([float(x), float(y)], dtype=point_dtype)
    seg_len_sq = np.maximum(seg_len * seg_len, 1e-12)
    rel = point[None, :] - centerline
    dot = np.sum(rel * seg_vec, axis=1)
    t = np.clip(dot / seg_len_sq, 0.0, 1.0)
    proj = centerline + seg_vec * t[:, None]
    delta = point[None, :] - proj
    dist_sq = np.sum(delta * delta, axis=1)
    best_idx = int(np.argmin(dist_sq))
    best_len = float(seg_len[best_idx])
    if best_len <= 1e-12:
        tx, ty = 1.0, 0.0
    else:
        tx = float(seg_vec[best_idx, 0] / best_len)
        ty = float(seg_vec[best_idx, 1] / best_len)
    nx, ny = -ty, tx
    px = float(proj[best_idx, 0])
    py = float(proj[best_idx, 1])
    s = float(seg_s[best_idx] + float(t[best_idx]) * best_len)
    if s >= float(length):
        s -= float(length)
    dx = float(x) - px
    dy = float(y) - py
    lateral = dx * nx + dy * ny
    return TrackProjection(
        seg_index=int(best_idx),
        segment_t=float(t[best_idx]),
        point=(px, py),
        distance=float(math.sqrt(float(dist_sq[best_idx]))),
        s=float(s),
        lateral_offset=float(lateral),
        tangent=(float(tx), float(ty)),
        normal=(float(nx), float(ny)),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(float(low), min(float(high), float(value))))


def _templates_for_complexity(complexity: float) -> tuple[str, ...]:
    value = _clamp(float(complexity), 0.0, 1.0)
    if value < 0.20:
        return ("straight",)
    if value < 0.45:
        return ("straight", "bell", "s_curve")
    if value < 0.70:
        return ("straight", "bell", "s_curve", "fold")
    return ("straight", "bell", "s_curve", "fold")


def _fold_min_depth(template_kind: str, min_centerline_clearance: float) -> float:
    kind = str(template_kind).strip().lower()
    clearance = max(1.0, float(min_centerline_clearance))
    if kind == "fold":
        return float(clearance)
    return 0.0


def _fold_depth_for_template(
    template_kind: str,
    side_len: float,
    available_inner_height: float,
    min_centerline_clearance: float,
) -> float:
    kind = str(template_kind).strip().lower()
    min_depth = _fold_min_depth(kind, float(min_centerline_clearance))
    if min_depth <= 0.0:
        return 0.0

    target = min(0.32 * float(available_inner_height), 0.26 * float(side_len))
    return max(float(min_depth) * 1.15, float(target))


def _downgrade_fold_template(template_kind: str) -> str:
    kind = str(template_kind).strip().lower()
    if kind == "fold":
        return "s_curve"
    return kind


def _resolve_fold_depths(
    top_template: str,
    bottom_template: str,
    top_depth: float,
    bottom_depth: float,
    available_inner_height: float,
    min_centerline_clearance: float,
) -> tuple[tuple[str, float], tuple[str, float]]:
    top_kind = str(top_template).strip().lower()
    bottom_kind = str(bottom_template).strip().lower()
    top_value = max(0.0, float(top_depth))
    bottom_value = max(0.0, float(bottom_depth))
    clearance = max(1.0, float(min_centerline_clearance))
    max_total_depth = max(0.0, float(available_inner_height) - clearance)

    for _ in range(8):
        changed = False
        for side_name in ("top", "bottom"):
            kind = top_kind if side_name == "top" else bottom_kind
            value = top_value if side_name == "top" else bottom_value
            if kind not in FOLD_TEMPLATE_CHOICES:
                continue
            min_depth = _fold_min_depth(kind, clearance)
            if float(value) >= float(min_depth):
                continue
            new_kind = _downgrade_fold_template(kind)
            new_value = 0.0 if new_kind not in FOLD_TEMPLATE_CHOICES else max(clearance * 1.15, value * 0.62)
            if side_name == "top":
                top_kind, top_value = new_kind, new_value
            else:
                bottom_kind, bottom_value = new_kind, new_value
            changed = True
        if changed:
            continue

        total_depth = (
            (top_value if top_kind in FOLD_TEMPLATE_CHOICES else 0.0)
            + (bottom_value if bottom_kind in FOLD_TEMPLATE_CHOICES else 0.0)
        )
        if float(total_depth) <= float(max_total_depth):
            break

        if float(max_total_depth) > 0.0 and float(total_depth) > 1e-6:
            scaled_top = top_value
            scaled_bottom = bottom_value
            scale = float(max_total_depth) / float(total_depth)
            if top_kind in FOLD_TEMPLATE_CHOICES:
                scaled_top = float(top_value) * float(scale)
            if bottom_kind in FOLD_TEMPLATE_CHOICES:
                scaled_bottom = float(bottom_value) * float(scale)
            top_ok = top_kind not in FOLD_TEMPLATE_CHOICES or scaled_top >= _fold_min_depth(top_kind, clearance)
            bottom_ok = bottom_kind not in FOLD_TEMPLATE_CHOICES or scaled_bottom >= _fold_min_depth(bottom_kind, clearance)
            if bool(top_ok and bottom_ok):
                top_value = float(scaled_top)
                bottom_value = float(scaled_bottom)
                break

        fold_candidates = []
        if top_kind in FOLD_TEMPLATE_CHOICES:
            fold_candidates.append(("top", top_kind, top_value))
        if bottom_kind in FOLD_TEMPLATE_CHOICES:
            fold_candidates.append(("bottom", bottom_kind, bottom_value))
        if not fold_candidates:
            break

        fold_candidates.sort(key=lambda item: float(item[2]), reverse=True)
        side_name, kind, value = fold_candidates[0]
        new_kind = _downgrade_fold_template(kind)
        new_value = 0.0 if new_kind not in FOLD_TEMPLATE_CHOICES else max(clearance * 1.15, float(value) * 0.62)
        if side_name == "top":
            top_kind, top_value = new_kind, new_value
        else:
            bottom_kind, bottom_value = new_kind, new_value

    if top_kind not in FOLD_TEMPLATE_CHOICES:
        top_value = 0.0
    if bottom_kind not in FOLD_TEMPLATE_CHOICES:
        bottom_value = 0.0
    return (str(top_kind), float(top_value)), (str(bottom_kind), float(bottom_value))


def _sample_line_template_amplitude(
    rng: np.random.Generator,
    template_kind: str,
    side_len: float,
    *,
    half_width: float,
    bell_amplitude_min_px: float,
    bell_amplitude_max_px: float,
    s_amplitude_min_px: float,
    s_amplitude_max_px: float,
    inset_width_cap_ratio: float,
    inset_length_cap_ratio: float,
) -> tuple[str, float]:
    kind = str(template_kind).strip().lower()
    if kind not in {"bell", "s_curve"}:
        return str(kind), 0.0

    amp_cap = min(
        max(0.0, float(inset_width_cap_ratio) * float(half_width)),
        max(0.0, float(inset_length_cap_ratio) * float(side_len)),
    )
    if kind == "bell":
        amp_low = min(max(0.0, float(bell_amplitude_min_px)), amp_cap)
        amp_high = min(max(float(amp_low), float(bell_amplitude_max_px)), amp_cap)
    else:
        amp_low = min(max(0.0, float(s_amplitude_min_px)), amp_cap)
        amp_high = min(max(float(amp_low), float(s_amplitude_max_px)), amp_cap)
    if amp_high <= 1.0:
        return "straight", 0.0
    amp = float(rng.uniform(amp_low, amp_high)) if amp_high > amp_low + 1e-6 else float(amp_high)
    return str(kind), float(amp)


def _sample_arrays(
    centerline: np.ndarray,
    seg_vec: np.ndarray,
    seg_len: np.ndarray,
    seg_s: np.ndarray,
    length: float,
    s: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    count = int(centerline.shape[0])
    if count <= 0 or float(length) <= 1e-9:
        return (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)

    s_wrapped = _wrap_s(float(s), float(length))
    seg_idx = int(np.searchsorted(seg_s, s_wrapped, side="right") - 1)
    seg_idx = int(max(0, min(count - 1, seg_idx)))
    this_len = float(seg_len[seg_idx])
    seg_start_s = float(seg_s[seg_idx])
    local_t = 0.0 if this_len <= 1e-9 else _clamp((s_wrapped - seg_start_s) / this_len, 0.0, 1.0)

    p0 = centerline[seg_idx]
    vec = seg_vec[seg_idx]
    px = float(p0[0] + vec[0] * local_t)
    py = float(p0[1] + vec[1] * local_t)
    tx, ty = _normalize(float(vec[0]), float(vec[1]))
    nx, ny = -ty, tx
    return (float(px), float(py)), (float(tx), float(ty)), (float(nx), float(ny))


def clean_polygon_vertices(points: list[tuple[float, float]], eps: float = 1e-4) -> list[tuple[float, float]]:
    if not points:
        return []

    threshold = max(1e-8, float(eps))
    deduped: list[tuple[float, float]] = []
    for px, py in points:
        if not deduped:
            deduped.append((float(px), float(py)))
            continue
        qx, qy = deduped[-1]
        if math.hypot(float(px) - float(qx), float(py) - float(qy)) > threshold:
            deduped.append((float(px), float(py)))

    if len(deduped) > 1:
        if math.hypot(float(deduped[0][0]) - float(deduped[-1][0]), float(deduped[0][1]) - float(deduped[-1][1])) <= threshold:
            deduped.pop()

    if len(deduped) <= 2:
        return deduped

    changed = True
    out = deduped
    while changed and len(out) > 2:
        changed = False
        next_out: list[tuple[float, float]] = []
        n = len(out)
        for i in range(n):
            prev = out[i - 1]
            cur = out[i]
            nxt = out[(i + 1) % n]
            if math.hypot(float(cur[0]) - float(prev[0]), float(cur[1]) - float(prev[1])) <= threshold:
                changed = True
                continue
            if math.hypot(float(nxt[0]) - float(cur[0]), float(nxt[1]) - float(cur[1])) <= threshold:
                changed = True
                continue
            next_out.append(cur)
        out = next_out

    return out


def build_road_polygon(track: TrackGeometry, seam_index: int | None = None) -> list[tuple[float, float]]:
    left = np.asarray(track.left_boundary, dtype=np.float64)
    right = np.asarray(track.right_boundary, dtype=np.float64)
    if left.shape[0] <= 2 or right.shape[0] <= 2:
        return []
    n = int(min(left.shape[0], right.shape[0]))
    left = left[:n]
    right = right[:n]
    if n <= 2:
        return []

    seam = int(track.start_index) if seam_index is None else int(seam_index)
    seam %= n
    if seam != 0:
        left = np.vstack((left[seam:], left[:seam]))
        right = np.vstack((right[seam:], right[:seam]))

    raw = np.vstack((left, right[::-1]))
    points = [(float(row[0]), float(row[1])) for row in raw]
    return clean_polygon_vertices(points, eps=1e-4)


def build_boundary_loops(
    track: TrackGeometry,
    seam_index: int | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    left = np.asarray(track.left_boundary, dtype=np.float64)
    right = np.asarray(track.right_boundary, dtype=np.float64)
    if left.shape[0] <= 2 or right.shape[0] <= 2:
        return [], []
    n = int(min(left.shape[0], right.shape[0]))
    left = left[:n]
    right = right[:n]
    seam = int(track.start_index) if seam_index is None else int(seam_index)
    seam %= n
    if seam != 0:
        left = np.vstack((left[seam:], left[:seam]))
        right = np.vstack((right[seam:], right[:seam]))
    left_pts = clean_polygon_vertices([(float(row[0]), float(row[1])) for row in left], eps=1e-4)
    right_pts = clean_polygon_vertices([(float(row[0]), float(row[1])) for row in right], eps=1e-4)
    return left_pts, right_pts


def raycast_polygon_edges(
    origin: tuple[float, float],
    direction: tuple[float, float],
    edge_starts: np.ndarray,
    edge_vectors: np.ndarray,
    max_dist: float,
    *,
    origin_epsilon: float = 1e-4,
) -> float | None:
    ux, uy = _normalize(float(direction[0]), float(direction[1]))
    max_distance = max(0.0, float(max_dist))
    if max_distance <= 1e-9:
        return None

    starts = np.asarray(edge_starts)
    vectors = np.asarray(edge_vectors)
    if starts.ndim != 2 or vectors.ndim != 2 or starts.shape[0] == 0 or starts.shape != vectors.shape:
        return None

    qx = starts[:, 0] - float(origin[0])
    qy = starts[:, 1] - float(origin[1])
    sx = vectors[:, 0]
    sy = vectors[:, 1]
    den = float(ux) * sy - float(uy) * sx
    valid_den = np.abs(den) > 1e-10
    if not bool(np.any(valid_den)):
        return None

    ray_t = np.full((starts.shape[0],), np.inf, dtype=np.float32)
    seg_u = np.full((starts.shape[0],), np.inf, dtype=np.float32)
    ray_t[valid_den] = (qx[valid_den] * sy[valid_den] - qy[valid_den] * sx[valid_den]) / den[valid_den]
    seg_u[valid_den] = (qx[valid_den] * float(uy) - qy[valid_den] * float(ux)) / den[valid_den]
    valid = (
        valid_den
        & (ray_t > float(origin_epsilon))
        & (ray_t <= float(max_distance))
        & (seg_u >= 0.0)
        & (seg_u <= 1.0)
    )
    if not bool(np.any(valid)):
        return None
    return float(np.min(ray_t[valid]))


def validate_track_geometry(track: TrackGeometry, *, min_centerline_clearance: float | None = None) -> None:
    polygon = np.asarray(track.road_polygon, dtype=np.float64)
    if polygon.shape[0] < 6:
        raise RuntimeError("Road polygon is invalid: not enough vertices.")

    road_points = [(float(row[0]), float(row[1])) for row in polygon]
    if _has_self_intersection(road_points):
        raise RuntimeError("Road polygon self-intersects.")

    left_loop, right_loop = build_boundary_loops(track, seam_index=int(track.start_index))
    if left_loop and _has_self_intersection(left_loop):
        raise RuntimeError("Left road boundary self-intersects.")
    if right_loop and _has_self_intersection(right_loop):
        raise RuntimeError("Right road boundary self-intersects.")

    if min_centerline_clearance is not None:
        clearance = max(0.0, float(min_centerline_clearance))
        if clearance > 0.0:
            min_distance = min_non_neighbor_segment_distance(
                np.asarray(track.centerline, dtype=np.float64),
                neighbor_window=20,
                early_stop_distance=float(clearance),
            )
            if float(min_distance) < float(clearance) - 1e-6:
                raise RuntimeError("Track centerline violates non-neighbor clearance.")

    for i in range(int(polygon.shape[0])):
        p = polygon[i]
        q = polygon[(i + 1) % int(polygon.shape[0])]
        if float(math.hypot(float(q[0]) - float(p[0]), float(q[1]) - float(p[1]))) <= 1e-5:
            raise RuntimeError("Road polygon has zero-length edge after cleanup.")

    mid_x = 0.5 * (float(track.start_line[0][0]) + float(track.start_line[1][0]))
    mid_y = 0.5 * (float(track.start_line[0][1]) + float(track.start_line[1][1]))
    proj = project_point_to_track(track, (float(mid_x), float(mid_y)))
    if abs(float(proj.lateral_offset)) > max(1.0, 0.04 * float(track.half_width)):
        raise RuntimeError("Start-line midpoint is not centered on track geometry.")

    ray_span = max(8.0, float(track.half_width) * 2.5)
    hit_left = raycast_track_edge(track, track.start_pos, track.start_normal, ray_span)
    hit_right = raycast_track_edge(track, track.start_pos, (-float(track.start_normal[0]), -float(track.start_normal[1])), ray_span)
    if hit_left is None or hit_right is None:
        raise RuntimeError("Start-strip normal ray did not hit road boundary.")
    tol = max(2.0, 0.08 * float(track.half_width))
    if abs(float(hit_left) - float(track.half_width)) > tol or abs(float(hit_right) - float(track.half_width)) > tol:
        raise RuntimeError("Start-strip normal ray distance does not match track half width.")


def _build_track_geometry_once(
    seed: int,
    width: int,
    height: int,
    *,
    track_width_px: float,
    padding_px: float,
    footprint_scale: float,
    corner_radius_px: float,
    sample_spacing_px: float,
    start_straight_len_px: float,
    long_side_template_choices: tuple[str, ...] | list[str],
    short_side_template_choices: tuple[str, ...] | list[str],
    bell_amplitude_min_px: float,
    bell_amplitude_max_px: float,
    s_amplitude_min_px: float,
    s_amplitude_max_px: float,
    inset_width_cap_ratio: float,
    inset_length_cap_ratio: float,
    fold_gap_px: float,
    complexity_min: float,
    complexity_max: float,
    use_complexity_filter: bool,
) -> TrackGeometry:
    rng = np.random.default_rng(int(seed))
    half_width = max(8.0, 0.5 * float(track_width_px))
    sample_spacing = max(3.0, float(sample_spacing_px))
    configured_template_choices = [
        str(value).strip().lower()
        for value in long_side_template_choices
        if str(value).strip().lower() in LONG_SIDE_TEMPLATE_CHOICES
    ]
    if not configured_template_choices:
        configured_template_choices = list(LONG_SIDE_TEMPLATE_CHOICES)
    configured_short_template_choices = [
        str(value).strip().lower()
        for value in short_side_template_choices
        if str(value).strip().lower() in SHORT_SIDE_TEMPLATE_CHOICES
    ]
    if not configured_short_template_choices:
        configured_short_template_choices = ["straight"]
    complexity_low = _clamp(float(complexity_min), 0.0, 1.0)
    complexity_high = _clamp(float(complexity_max), 0.0, 1.0)
    if complexity_high < complexity_low:
        complexity_low, complexity_high = complexity_high, complexity_low
    track_complexity = (
        float(rng.uniform(complexity_low, complexity_high))
        if complexity_high > complexity_low + 1e-9
        else float(complexity_high)
    )
    complexity_templates = _templates_for_complexity(track_complexity)
    if bool(use_complexity_filter):
        allowed_templates = set(complexity_templates)
        template_choices = [value for value in configured_template_choices if value in allowed_templates]
        allowed_short_templates = {value for value in complexity_templates if value in SHORT_SIDE_TEMPLATE_CHOICES}
        short_template_choices = [
            value for value in configured_short_template_choices if value in allowed_short_templates
        ]
    else:
        template_choices = list(configured_template_choices)
        short_template_choices = list(configured_short_template_choices)
    if not template_choices:
        if "straight" in configured_template_choices:
            template_choices = ["straight"]
        else:
            template_choices = [str(configured_template_choices[0])]
    if not short_template_choices:
        short_template_choices = ["straight"]
    long_side_templates = {
        "top": str(template_choices[int(rng.integers(0, len(template_choices)))]),
        "bottom": str(template_choices[int(rng.integers(0, len(template_choices)))]),
    }
    short_side_templates = {
        "right": str(short_template_choices[int(rng.integers(0, len(short_template_choices)))]),
        "left": str(short_template_choices[int(rng.integers(0, len(short_template_choices)))]),
    }
    effective_side_templates = {
        "top": str(long_side_templates.get("top", "straight")),
        "right": str(short_side_templates.get("right", "straight")),
        "bottom": str(long_side_templates.get("bottom", "straight")),
        "left": str(short_side_templates.get("left", "straight")),
    }
    straight_start_candidates = [
        str(side)
        for side in SIDE_ORDER
        if str(effective_side_templates.get(str(side), "straight")) == "straight"
    ]
    if not straight_start_candidates:
        forced_side = str(SIDE_ORDER[int(rng.integers(0, len(SIDE_ORDER)))])
        effective_side_templates[forced_side] = "straight"
        straight_start_candidates = [forced_side]
    start_side = str(straight_start_candidates[int(rng.integers(0, len(straight_start_candidates)))])

    # S-curve templates can bow outward, so reserve margin only for the sides
    # that selected an outward-capable template.
    left_guard = float(s_amplitude_max_px) if str(effective_side_templates.get("left", "straight")) == "s_curve" else 0.0
    right_guard = float(s_amplitude_max_px) if str(effective_side_templates.get("right", "straight")) == "s_curve" else 0.0
    top_guard = float(s_amplitude_max_px) if str(effective_side_templates.get("top", "straight")) == "s_curve" else 0.0
    bottom_guard = float(s_amplitude_max_px) if str(effective_side_templates.get("bottom", "straight")) == "s_curve" else 0.0
    margin_x = float(padding_px) + float(half_width) + 2.0
    left = float(margin_x) + float(left_guard)
    right = float(width) - (float(margin_x) + float(right_guard))
    top = float(padding_px) + float(half_width) + float(top_guard) + 2.0
    bottom = float(height) - (float(padding_px) + float(half_width) + float(bottom_guard) + 2.0)
    if right <= left + 120.0 or bottom <= top + 120.0:
        raise RuntimeError("Track bounds too small after padding and width constraints.")

    scale = max(0.55, min(1.0, float(footprint_scale)))
    if scale < 0.999:
        cx = 0.5 * (left + right)
        cy = 0.5 * (top + bottom)
        half_w = 0.5 * (right - left) * scale
        half_h = 0.5 * (bottom - top) * scale
        left = cx - half_w
        right = cx + half_w
        top = cy - half_h
        bottom = cy + half_h

    corner_r = min(
        max(16.0, float(corner_radius_px)),
        max(24.0, 0.5 * (right - left) - 10.0),
        max(24.0, 0.5 * (bottom - top) - 10.0),
    )
    if corner_r <= 10.0:
        raise RuntimeError("Invalid corner radius for Vroom track generation.")

    side_defs: dict[str, dict[str, object]] = {
        "top": {
            "p0": (left + corner_r, top),
            "p1": (right - corner_r, top),
            "outward": (0.0, -1.0),
            "tangent": (1.0, 0.0),
        },
        "right": {
            "p0": (right, top + corner_r),
            "p1": (right, bottom - corner_r),
            "outward": (1.0, 0.0),
            "tangent": (0.0, 1.0),
        },
        "bottom": {
            "p0": (right - corner_r, bottom),
            "p1": (left + corner_r, bottom),
            "outward": (0.0, 1.0),
            "tangent": (-1.0, 0.0),
        },
        "left": {
            "p0": (left, bottom - corner_r),
            "p1": (left, top + corner_r),
            "outward": (-1.0, 0.0),
            "tangent": (0.0, -1.0),
        },
    }

    start_side_info = side_defs[start_side]
    start_side_p0 = start_side_info["p0"]  # type: ignore[assignment]
    start_side_p1 = start_side_info["p1"]  # type: ignore[assignment]
    start_side_len = float(
        math.hypot(
            float(start_side_p1[0]) - float(start_side_p0[0]),  # type: ignore[index]
            float(start_side_p1[1]) - float(start_side_p0[1]),  # type: ignore[index]
        )
    )
    start_dir_x, start_dir_y = _normalize(
        float(start_side_p1[0]) - float(start_side_p0[0]),  # type: ignore[index]
        float(start_side_p1[1]) - float(start_side_p0[1]),  # type: ignore[index]
    )
    min_strip_len = max(48.0, 1.35 * float(track_width_px))
    transition_margin = max(10.0, 0.85 * float(half_width))
    max_strip_len = max(16.0, float(start_side_len) - 2.0 * float(transition_margin))
    requested_strip_len = max(float(min_strip_len), float(start_straight_len_px))
    if max_strip_len <= 20.0:
        strip_len = max(10.0, 0.55 * float(start_side_len))
    else:
        strip_len = _clamp(float(requested_strip_len), float(min_strip_len), float(max_strip_len))
    strip_len = min(float(strip_len), max(10.0, float(start_side_len) - 2.0))
    if strip_len <= 8.0:
        raise RuntimeError("Start straight strip is too short for valid placement.")
    start_mid = (
        0.5 * (float(start_side_p0[0]) + float(start_side_p1[0])),  # type: ignore[index]
        0.5 * (float(start_side_p0[1]) + float(start_side_p1[1])),  # type: ignore[index]
    )
    half_strip = 0.5 * float(strip_len)
    start_strip_a = (
        float(start_mid[0]) - float(start_dir_x) * half_strip,
        float(start_mid[1]) - float(start_dir_y) * half_strip,
    )
    start_strip_b = (
        float(start_mid[0]) + float(start_dir_x) * half_strip,
        float(start_mid[1]) + float(start_dir_y) * half_strip,
    )

    side_templates: dict[str, str] = {}
    side_lengths: dict[str, float] = {}
    for side in SIDE_ORDER:
        side_info = side_defs[side]
        p0 = side_info["p0"]  # type: ignore[assignment]
        p1 = side_info["p1"]  # type: ignore[assignment]
        side_len = float(math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))  # type: ignore[index]
        side_lengths[side] = float(side_len)
        side_templates[side] = str(effective_side_templates.get(side, "straight"))

    available_inner_height = max(0.0, float(bottom) - float(top))
    min_centerline_clearance = max(1.0, float(track_width_px) + max(0.0, float(fold_gap_px)))
    desired_fold_depths = {
        side: _fold_depth_for_template(
            side_templates[side],
            float(side_lengths[side]),
            float(available_inner_height),
            float(min_centerline_clearance),
        )
        for side in LONG_SIDE_ORDER
    }
    (top_template, top_depth), (bottom_template, bottom_depth) = _resolve_fold_depths(
        str(side_templates["top"]),
        str(side_templates["bottom"]),
        float(desired_fold_depths["top"]),
        float(desired_fold_depths["bottom"]),
        float(available_inner_height),
        float(min_centerline_clearance),
    )
    side_templates["top"] = str(top_template)
    side_templates["bottom"] = str(bottom_template)
    desired_fold_depths["top"] = float(top_depth)
    desired_fold_depths["bottom"] = float(bottom_depth)

    side_params: dict[str, tuple[str, float]] = {}
    for side in SIDE_ORDER:
        template_kind = str(side_templates[side])
        if template_kind in FOLD_TEMPLATE_CHOICES:
            side_params[side] = (str(template_kind), float(desired_fold_depths.get(side, 0.0)))
            continue
        resolved_kind, amp = _sample_line_template_amplitude(
            rng,
            template_kind,
            float(side_lengths[side]),
            half_width=float(half_width),
            bell_amplitude_min_px=float(bell_amplitude_min_px),
            bell_amplitude_max_px=float(bell_amplitude_max_px),
            s_amplitude_min_px=float(s_amplitude_min_px),
            s_amplitude_max_px=float(s_amplitude_max_px),
            inset_width_cap_ratio=float(inset_width_cap_ratio),
            inset_length_cap_ratio=float(inset_length_cap_ratio),
        )
        side_params[side] = (str(resolved_kind), float(amp))

    def _compose_loop(local_side_params: dict[str, tuple[str, float]]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        arc_count = max(8, int((0.5 * math.pi * corner_r) / max(1.0, sample_spacing * 0.6)) + 1)
        for side in SIDE_ORDER:
            side_info = side_defs[side]
            p0 = side_info["p0"]  # type: ignore[assignment]
            p1 = side_info["p1"]  # type: ignore[assignment]
            outward = side_info["outward"]  # type: ignore[assignment]
            side_len = float(math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))  # type: ignore[index]
            side_count = max(5, int(side_len / max(1.0, sample_spacing * 0.65)) + 1)
            template_kind, amp = local_side_params[side]
            side_pts = _side_template_points(
                p0,  # type: ignore[arg-type]
                p1,  # type: ignore[arg-type]
                side_count,
                template_kind=str(template_kind),
                template_amp=float(amp),
                template_normal=outward,  # type: ignore[arg-type]
                sample_spacing_px=float(sample_spacing),
            )
            _append_no_dup(points, side_pts)

            if side == "top":
                arc_pts = _arc_points(right - corner_r, top + corner_r, corner_r, -math.pi / 2.0, 0.0, arc_count)
            elif side == "right":
                arc_pts = _arc_points(right - corner_r, bottom - corner_r, corner_r, 0.0, math.pi / 2.0, arc_count)
            elif side == "bottom":
                arc_pts = _arc_points(left + corner_r, bottom - corner_r, corner_r, math.pi / 2.0, math.pi, arc_count)
            else:
                arc_pts = _arc_points(left + corner_r, top + corner_r, corner_r, math.pi, 1.5 * math.pi, arc_count)
            _append_no_dup(points, arc_pts)

        if len(points) > 1:
            x0, y0 = points[0]
            x1, y1 = points[-1]
            if abs(float(x0) - float(x1)) < 1e-6 and abs(float(y0) - float(y1)) < 1e-6:
                points.pop()
        return points

    dense_points = _compose_loop(side_params)
    if _has_self_intersection(dense_points):
        for side in SIDE_ORDER:
            template_kind, amp = side_params[side]
            if amp > 0.0:
                side_params[side] = (str(template_kind), float(amp) * 0.55)
        dense_points = _compose_loop(side_params)
        if _has_self_intersection(dense_points):
            for side in SIDE_ORDER:
                side_params[side] = ("straight", 0.0)
            dense_points = _compose_loop(side_params)
    effective_side_templates = tuple(
        (str(side), str(side_params[side][0]))
        for side in SIDE_ORDER
    )
    curved_sides = tuple(
        str(side)
        for side in SIDE_ORDER
        if str(side_params[side][0]) != "straight"
    )

    dense_np = np.asarray(dense_points, dtype=np.float64)
    if dense_np.shape[0] < 24:
        raise RuntimeError("Track generation failed to build a valid closed loop.")

    _, _, dense_len = _closed_segment_lengths(dense_np)
    sample_count = max(96, int(round(dense_len / max(1.0, sample_spacing))))
    sample_count = min(2400, sample_count)
    centerline = _resample_closed_polyline(dense_np, sample_count)
    has_fold_template = any(str(template_kind) in FOLD_TEMPLATE_CHOICES for template_kind, _amp in side_params.values())
    has_curved_template = any(str(template_kind) != "straight" for template_kind, _amp in side_params.values())
    smooth_iterations = 7 if has_fold_template else 3 if has_curved_template else 0
    if smooth_iterations > 0:
        pre_seg_vec, pre_seg_len, pre_track_len = _closed_segment_lengths(centerline)
        fixed_mask = None
        if pre_track_len > 1e-6:
            pre_seg_s = np.zeros((centerline.shape[0],), dtype=np.float64)
            if centerline.shape[0] > 1:
                pre_seg_s[1:] = np.cumsum(pre_seg_len[:-1], dtype=np.float64)
            pre_start_proj = _build_projection(
                centerline=centerline,
                seg_vec=pre_seg_vec,
                seg_len=pre_seg_len,
                seg_s=pre_seg_s,
                length=pre_track_len,
                x=float(start_mid[0]),
                y=float(start_mid[1]),
            )
            pre_strip_a_proj = _build_projection(
                centerline=centerline,
                seg_vec=pre_seg_vec,
                seg_len=pre_seg_len,
                seg_s=pre_seg_s,
                length=pre_track_len,
                x=float(start_strip_a[0]),
                y=float(start_strip_a[1]),
            )
            pre_strip_b_proj = _build_projection(
                centerline=centerline,
                seg_vec=pre_seg_vec,
                seg_len=pre_seg_len,
                seg_s=pre_seg_s,
                length=pre_track_len,
                x=float(start_strip_b[0]),
                y=float(start_strip_b[1]),
            )
            pre_start_s = _wrap_s(float(pre_start_proj.s), float(pre_track_len))
            fixed_a = _wrap_delta(float(pre_strip_a_proj.s), float(pre_start_s), float(pre_track_len))
            fixed_b = _wrap_delta(float(pre_strip_b_proj.s), float(pre_start_s), float(pre_track_len))
            fixed_margin = max(float(sample_spacing) * 2.0, float(half_width) * 0.25)
            fixed_min = min(float(fixed_a), float(fixed_b)) - float(fixed_margin)
            fixed_max = max(float(fixed_a), float(fixed_b)) + float(fixed_margin)
            signed_delta = ((pre_seg_s - float(pre_start_s) + 0.5 * float(pre_track_len)) % float(pre_track_len)) - 0.5 * float(pre_track_len)
            fixed_mask = (signed_delta >= float(fixed_min)) & (signed_delta <= float(fixed_max))
        centerline = _smooth_closed_centerline(
            centerline,
            fixed_mask=fixed_mask,
            iterations=int(smooth_iterations),
            strength=(0.46 if has_fold_template else 0.36),
        )

    seg_vec, seg_len, track_len = _closed_segment_lengths(centerline)
    if track_len <= 1e-6:
        raise RuntimeError("Track resampling produced invalid geometry length.")
    seg_s = np.zeros((centerline.shape[0],), dtype=np.float64)
    if centerline.shape[0] > 1:
        seg_s[1:] = np.cumsum(seg_len[:-1], dtype=np.float64)
    arc_s = np.asarray(seg_s, dtype=np.float64)

    tangent_vec = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangent_norm = np.linalg.norm(tangent_vec, axis=1)
    tangent_norm = np.maximum(tangent_norm, 1e-9)
    tangents = tangent_vec / tangent_norm[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    left_boundary = centerline + normals * float(half_width)
    right_boundary = centerline - normals * float(half_width)

    start_proj = _build_projection(
        centerline=centerline,
        seg_vec=seg_vec,
        seg_len=seg_len,
        seg_s=seg_s,
        length=track_len,
        x=float(start_mid[0]),
        y=float(start_mid[1]),
    )
    strip_a_proj = _build_projection(
        centerline=centerline,
        seg_vec=seg_vec,
        seg_len=seg_len,
        seg_s=seg_s,
        length=track_len,
        x=float(start_strip_a[0]),
        y=float(start_strip_a[1]),
    )
    strip_b_proj = _build_projection(
        centerline=centerline,
        seg_vec=seg_vec,
        seg_len=seg_len,
        seg_s=seg_s,
        length=track_len,
        x=float(start_strip_b[0]),
        y=float(start_strip_b[1]),
    )

    start_s = _wrap_s(float(start_proj.s), float(track_len))
    d_a = _wrap_delta(float(strip_a_proj.s), float(start_s), float(track_len))
    d_b = _wrap_delta(float(strip_b_proj.s), float(start_s), float(track_len))
    strip_delta_min = float(min(d_a, d_b))
    strip_delta_max = float(max(d_a, d_b))
    if strip_delta_min > 0.0:
        strip_delta_min = 0.0
    if strip_delta_max < 0.0:
        strip_delta_max = 0.0

    start_tangent = (float(start_dir_x), float(start_dir_y))
    start_normal = (-float(start_dir_y), float(start_dir_x))
    start_pos = (float(start_proj.point[0]), float(start_proj.point[1]))
    start_line = (
        (
            float(start_pos[0]) + float(start_normal[0]) * float(half_width),
            float(start_pos[1]) + float(start_normal[1]) * float(half_width),
        ),
        (
            float(start_pos[0]) - float(start_normal[0]) * float(half_width),
            float(start_pos[1]) - float(start_normal[1]) * float(half_width),
        ),
    )
    sample_strip_count = 17
    max_strip_tangent_dev = 0.0
    for i in range(sample_strip_count):
        alpha = float(i) / float(max(1, sample_strip_count - 1))
        delta = float(strip_delta_min) + (float(strip_delta_max) - float(strip_delta_min)) * alpha
        _, tangent, _ = _sample_arrays(
            centerline=centerline,
            seg_vec=seg_vec,
            seg_len=seg_len,
            seg_s=seg_s,
            length=track_len,
            s=float(start_s) + float(delta),
        )
        dot = _clamp(float(start_tangent[0]) * float(tangent[0]) + float(start_tangent[1]) * float(tangent[1]), -1.0, 1.0)
        dev = float(math.acos(dot))
        max_strip_tangent_dev = max(max_strip_tangent_dev, dev)
    if max_strip_tangent_dev > 0.03:
        raise RuntimeError("Start strip is not flat enough after resampling.")

    signed_delta = ((arc_s - float(start_s) + 0.5 * float(track_len)) % float(track_len)) - 0.5 * float(track_len)
    start_index = int(np.argmin(np.abs(signed_delta)))

    corner_sample_angles = (-0.25 * math.pi, 0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi)
    corner_sample_centers = (
        (right - corner_r, top + corner_r),
        (right - corner_r, bottom - corner_r),
        (left + corner_r, bottom - corner_r),
        (left + corner_r, top + corner_r),
    )
    main_corner_s: list[float] = []
    for (center_x, center_y), angle in zip(corner_sample_centers, corner_sample_angles):
        sample_x = float(center_x) + math.cos(float(angle)) * float(corner_r)
        sample_y = float(center_y) + math.sin(float(angle)) * float(corner_r)
        corner_proj = _build_projection(
            centerline=centerline,
            seg_vec=seg_vec,
            seg_len=seg_len,
            seg_s=seg_s,
            length=track_len,
            x=float(sample_x),
            y=float(sample_y),
        )
        main_corner_s.append(float(_wrap_s(float(corner_proj.s), float(track_len))))

    template_family = "deformed_loop_templates"
    seam = int(start_index) % int(centerline.shape[0])
    if seam != 0:
        road_left = np.vstack((left_boundary[seam:], left_boundary[:seam]))
        road_right = np.vstack((right_boundary[seam:], right_boundary[:seam]))
    else:
        road_left = left_boundary
        road_right = right_boundary
    raw_road_polygon = np.vstack((road_left, road_right[::-1]))
    road_poly_points = clean_polygon_vertices(
        [(float(row[0]), float(row[1])) for row in raw_road_polygon],
        eps=1e-4,
    )
    if len(road_poly_points) < 6:
        raise RuntimeError("Road polygon collapsed during cleanup.")
    road_polygon_np = np.asarray(road_poly_points, dtype=np.float32)
    road_edge_vectors = np.roll(road_polygon_np, -1, axis=0) - road_polygon_np
    start_strip_s_min = _wrap_s(float(start_s) + float(strip_delta_min), float(track_len))
    start_strip_s_max = _wrap_s(float(start_s) + float(strip_delta_max), float(track_len))

    track = TrackGeometry(
        centerline=np.asarray(centerline, dtype=np.float32),
        arc_s=np.asarray(arc_s, dtype=np.float32),
        length=float(track_len),
        tangents=np.asarray(tangents, dtype=np.float32),
        normals=np.asarray(normals, dtype=np.float32),
        half_width=float(half_width),
        left_boundary=np.asarray(left_boundary, dtype=np.float32),
        right_boundary=np.asarray(right_boundary, dtype=np.float32),
        road_polygon=road_polygon_np,
        start_s=float(start_s),
        start_pos=(float(start_pos[0]), float(start_pos[1])),
        start_tangent=(float(start_tangent[0]), float(start_tangent[1])),
        start_normal=(float(start_normal[0]), float(start_normal[1])),
        start_side=str(start_side),
        start_index=int(start_index),
        start_line=start_line,
        start_side_delta_min=float(strip_delta_min),
        start_side_delta_max=float(strip_delta_max),
        start_strip_s_min=float(start_strip_s_min),
        start_strip_s_max=float(start_strip_s_max),
        start_straight_len_px=float(strip_len),
        main_corner_s=tuple(float(value) for value in main_corner_s),
        template_family=str(template_family),
        side_templates=tuple((str(side), str(template)) for side, template in effective_side_templates),
        curved_sides=tuple(str(side) for side in curved_sides),
        _segment_vectors=np.asarray(seg_vec, dtype=np.float32),
        _segment_lengths=np.asarray(seg_len, dtype=np.float32),
        _segment_s=np.asarray(seg_s, dtype=np.float32),
        _road_edge_starts=road_polygon_np,
        _road_edge_vectors=np.asarray(road_edge_vectors, dtype=np.float32),
    )
    validate_track_geometry(track, min_centerline_clearance=float(min_centerline_clearance))
    return track


def _retry_seed(seed: int, attempt: int) -> int:
    if int(attempt) <= 0:
        return int(seed)
    return int((int(seed) + int(attempt) * 0x9E3779B97F4A7C15) % (2**63 - 1))


def build_track_geometry(
    seed: int,
    width: int,
    height: int,
    *,
    track_width_px: float,
    padding_px: float,
    footprint_scale: float,
    corner_radius_px: float,
    sample_spacing_px: float,
    start_straight_len_px: float,
    long_side_template_choices: tuple[str, ...] | list[str],
    short_side_template_choices: tuple[str, ...] | list[str],
    bell_amplitude_min_px: float,
    bell_amplitude_max_px: float,
    s_amplitude_min_px: float,
    s_amplitude_max_px: float,
    inset_width_cap_ratio: float,
    inset_length_cap_ratio: float,
    fold_gap_px: float,
    generation_max_attempts: int,
    complexity_min: float,
    complexity_max: float,
    use_complexity_filter: bool,
) -> TrackGeometry:
    last_error: Exception | None = None
    attempts = max(0, int(generation_max_attempts))
    for attempt in range(attempts):
        try:
            return _build_track_geometry_once(
                seed=_retry_seed(int(seed), int(attempt)),
                width=int(width),
                height=int(height),
                track_width_px=float(track_width_px),
                padding_px=float(padding_px),
                footprint_scale=float(footprint_scale),
                corner_radius_px=float(corner_radius_px),
                sample_spacing_px=float(sample_spacing_px),
                start_straight_len_px=float(start_straight_len_px),
                long_side_template_choices=long_side_template_choices,
                short_side_template_choices=short_side_template_choices,
                bell_amplitude_min_px=float(bell_amplitude_min_px),
                bell_amplitude_max_px=float(bell_amplitude_max_px),
                s_amplitude_min_px=float(s_amplitude_min_px),
                s_amplitude_max_px=float(s_amplitude_max_px),
                inset_width_cap_ratio=float(inset_width_cap_ratio),
                inset_length_cap_ratio=float(inset_length_cap_ratio),
                fold_gap_px=float(fold_gap_px),
                complexity_min=float(complexity_min),
                complexity_max=float(complexity_max),
                use_complexity_filter=bool(use_complexity_filter),
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc

    try:
        return _build_track_geometry_once(
            seed=_retry_seed(int(seed), int(attempts) + 1),
            width=int(width),
            height=int(height),
            track_width_px=float(track_width_px),
            padding_px=float(padding_px),
            footprint_scale=float(footprint_scale),
            corner_radius_px=float(corner_radius_px),
            sample_spacing_px=float(sample_spacing_px),
            start_straight_len_px=float(start_straight_len_px),
            long_side_template_choices=("straight",),
            short_side_template_choices=("straight",),
            bell_amplitude_min_px=float(bell_amplitude_min_px),
            bell_amplitude_max_px=float(bell_amplitude_max_px),
            s_amplitude_min_px=float(s_amplitude_min_px),
            s_amplitude_max_px=float(s_amplitude_max_px),
            inset_width_cap_ratio=float(inset_width_cap_ratio),
            inset_length_cap_ratio=float(inset_length_cap_ratio),
            fold_gap_px=float(fold_gap_px),
            complexity_min=0.0,
            complexity_max=0.0,
            use_complexity_filter=False,
        )
    except (RuntimeError, ValueError) as exc:
        if last_error is not None:
            raise RuntimeError(f"Vroom track generation failed after fallback: {last_error}") from exc
        raise


def sample_track_at_s(
    track: TrackGeometry,
    s: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return _sample_arrays(
        centerline=track.centerline,
        seg_vec=track._segment_vectors,
        seg_len=track._segment_lengths,
        seg_s=track._segment_s,
        length=float(track.length),
        s=float(s),
    )


def start_strip_pose(track: TrackGeometry) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return sample_track_at_s(track, float(track.start_s))


def project_point_to_track(track: TrackGeometry, pos: tuple[float, float]) -> TrackProjection:
    return _build_projection(
        centerline=track.centerline,
        seg_vec=track._segment_vectors,
        seg_len=track._segment_lengths,
        seg_s=track._segment_s,
        length=float(track.length),
        x=float(pos[0]),
        y=float(pos[1]),
    )


def is_on_track(track: TrackGeometry, pos: tuple[float, float]) -> bool:
    proj = project_point_to_track(track, pos)
    return abs(float(proj.lateral_offset)) <= float(track.half_width)


def raycast_track_edge(
    track: TrackGeometry,
    origin: tuple[float, float],
    direction: tuple[float, float],
    max_dist: float,
) -> float | None:
    return raycast_polygon_edges(
        origin=(float(origin[0]), float(origin[1])),
        direction=(float(direction[0]), float(direction[1])),
        edge_starts=track._road_edge_starts,
        edge_vectors=track._road_edge_vectors,
        max_dist=float(max_dist),
        origin_epsilon=1e-4,
    )


def spawn_pose(
    track: TrackGeometry,
    slot_idx: int,
    *,
    lateral_offset: float = 0.0,
    longitudinal_spacing: float = 24.0,
    start_back_offset: float | None = None,
) -> tuple[tuple[float, float], float]:
    spacing = max(1.0, float(longitudinal_spacing))
    if start_back_offset is None:
        start_back_offset = max(float(track.half_width) * 0.85, spacing * 0.55)

    side_span = max(0.0, float(track.start_side_delta_max) - float(track.start_side_delta_min))
    corner_margin = min(max(float(track.half_width) * 0.75, spacing * 0.70), max(0.0, 0.5 * side_span - 1.0))
    min_delta = float(track.start_side_delta_min) + float(corner_margin)
    max_delta = float(track.start_side_delta_max) - float(corner_margin)
    if min_delta > max_delta:
        mid = 0.5 * (float(track.start_side_delta_min) + float(track.start_side_delta_max))
        min_delta = mid
        max_delta = mid

    target_delta = -float(start_back_offset) - float(max(0, int(slot_idx))) * spacing
    clamped_delta = _clamp(float(target_delta), float(min_delta), float(max_delta))
    s = _wrap_s(float(track.start_s) + float(clamped_delta), float(track.length))

    pos, tangent, normal = sample_track_at_s(track, s)
    lane = _clamp(float(lateral_offset), -0.92 * float(track.half_width), 0.92 * float(track.half_width))
    x = float(pos[0]) + float(normal[0]) * lane
    y = float(pos[1]) + float(normal[1]) * lane
    heading = float(math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))))
    return (float(x), float(y)), float(heading)
