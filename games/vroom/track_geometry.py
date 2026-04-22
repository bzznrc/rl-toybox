"""Canonical geometry-first track model for Vroom."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SIDE_ORDER = ("top", "right", "bottom", "left")
LONG_SIDE_ORDER = ("top", "bottom")
LONG_SIDE_TEMPLATE_CHOICES = ("straight", "bell", "s_curve")


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


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    mag = math.hypot(float(dx), float(dy))
    if mag <= 1e-12:
        return 1.0, 0.0
    return float(dx) / mag, float(dy) / mag


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
    for i in range(point_count):
        t = float(i) / float(max(1, point_count - 1))
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
                # Two opposing smooth bends joined into a mild S profile.
                disp = float(template_amp) * math.sin(2.0 * math.pi * t) * fade
            elif kind != "straight":
                raise ValueError(f"Unsupported long-side template '{template_kind}'.")
        if abs(float(disp)) > 1e-9:
            x += nx * disp
            y += ny * disp
        out.append((float(x), float(y)))
    return out


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


def _has_self_intersection(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    if count < 4:
        return False
    for i in range(count):
        a0 = points[i]
        a1 = points[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i:
                continue
            if j == (i + 1) % count:
                continue
            if i == (j + 1) % count:
                continue
            if i == 0 and j == count - 1:
                continue
            b0 = points[j]
            b1 = points[(j + 1) % count]
            if _segments_intersect(a0, a1, b0, b1):
                return True
    return False


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

    point = np.asarray([float(x), float(y)], dtype=np.float64)
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


def raycast_polygon(
    origin: tuple[float, float],
    direction: tuple[float, float],
    polygon: np.ndarray,
    max_dist: float,
    *,
    origin_epsilon: float = 1e-4,
) -> float | None:
    ux, uy = _normalize(float(direction[0]), float(direction[1]))
    max_distance = max(0.0, float(max_dist))
    if max_distance <= 1e-9:
        return None

    count = int(polygon.shape[0])
    if count < 3:
        return None

    ox = float(origin[0])
    oy = float(origin[1])
    hit_min: float | None = None
    for i in range(count):
        ax = float(polygon[i, 0])
        ay = float(polygon[i, 1])
        bx = float(polygon[(i + 1) % count, 0])
        by = float(polygon[(i + 1) % count, 1])
        sx = bx - ax
        sy = by - ay
        den = ux * sy - uy * sx
        if abs(float(den)) <= 1e-10:
            continue

        qpx = ax - ox
        qpy = ay - oy
        ray_t = (qpx * sy - qpy * sx) / den
        seg_u = (qpx * uy - qpy * ux) / den
        if ray_t <= float(origin_epsilon):
            continue
        if seg_u < 0.0 or seg_u > 1.0:
            continue
        if ray_t > float(max_distance):
            continue
        if hit_min is None or float(ray_t) < float(hit_min):
            hit_min = float(ray_t)
    return hit_min


def validate_track_geometry(track: TrackGeometry) -> None:
    polygon = np.asarray(track.road_polygon, dtype=np.float64)
    if polygon.shape[0] < 6:
        raise RuntimeError("Road polygon is invalid: not enough vertices.")

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
    bell_amplitude_min_px: float,
    bell_amplitude_max_px: float,
    s_amplitude_min_px: float,
    s_amplitude_max_px: float,
    inset_width_cap_ratio: float,
    inset_length_cap_ratio: float,
) -> TrackGeometry:
    rng = np.random.default_rng(int(seed))
    half_width = max(8.0, 0.5 * float(track_width_px))
    sample_spacing = max(3.0, float(sample_spacing_px))
    template_choices = [
        str(value).strip().lower()
        for value in long_side_template_choices
        if str(value).strip().lower() in LONG_SIDE_TEMPLATE_CHOICES
    ]
    if not template_choices:
        template_choices = list(LONG_SIDE_TEMPLATE_CHOICES)
    start_side = str(SIDE_ORDER[int(rng.integers(0, len(SIDE_ORDER)))])
    long_side_templates = {
        "top": str(template_choices[int(rng.integers(0, len(template_choices)))]),
        "bottom": str(template_choices[int(rng.integers(0, len(template_choices)))]),
    }
    effective_long_side_templates = {
        "top": "straight" if start_side == "top" else str(long_side_templates.get("top", "straight")),
        "bottom": "straight" if start_side == "bottom" else str(long_side_templates.get("bottom", "straight")),
    }

    # Only the long top/bottom sides can bow outward, so reserve bend margin on
    # the specific long side that actually selected an outward-capable template.
    margin_x = float(padding_px) + float(half_width) + 2.0
    top_guard = float(s_amplitude_max_px) if str(effective_long_side_templates.get("top", "straight")) == "s_curve" else 0.0
    bottom_guard = (
        float(s_amplitude_max_px) if str(effective_long_side_templates.get("bottom", "straight")) == "s_curve" else 0.0
    )
    left = float(margin_x)
    right = float(width) - float(margin_x)
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

    side_params: dict[str, tuple[str, float]] = {}
    for side in SIDE_ORDER:
        side_info = side_defs[side]
        p0 = side_info["p0"]  # type: ignore[assignment]
        p1 = side_info["p1"]  # type: ignore[assignment]
        side_len = float(math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))  # type: ignore[index]
        if side in {"left", "right"}:
            side_params[side] = ("straight", 0.0)
            continue
        template_kind = str(long_side_templates.get(side, "straight"))
        if side == start_side:
            template_kind = "straight"
        amp_cap = min(
            max(0.0, float(inset_width_cap_ratio) * float(half_width)),
            max(0.0, float(inset_length_cap_ratio) * float(side_len)),
        )
        if template_kind == "bell":
            amp_low = min(max(0.0, float(bell_amplitude_min_px)), amp_cap)
            amp_high = min(max(float(amp_low), float(bell_amplitude_max_px)), amp_cap)
        elif template_kind == "s_curve":
            amp_low = min(max(0.0, float(s_amplitude_min_px)), amp_cap)
            amp_high = min(max(float(amp_low), float(s_amplitude_max_px)), amp_cap)
        else:
            amp_low = 0.0
            amp_high = 0.0
        if amp_high <= 1.0:
            side_params[side] = ("straight", 0.0)
            continue
        amp = float(rng.uniform(amp_low, amp_high)) if amp_high > amp_low + 1e-6 else float(amp_high)
        side_params[side] = (str(template_kind), float(amp))

    def _compose_loop(local_side_params: dict[str, tuple[str, float]]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        arc_count = max(8, int((0.5 * math.pi * corner_r) / max(1.0, sample_spacing * 0.6)) + 1)
        for side in SIDE_ORDER:
            side_info = side_defs[side]
            p0 = side_info["p0"]  # type: ignore[assignment]
            p1 = side_info["p1"]  # type: ignore[assignment]
            outward = side_info["outward"]  # type: ignore[assignment]
            side_len = float(math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))  # type: ignore[index]
            if side == start_side:
                lead_len = float(math.hypot(float(start_strip_a[0]) - float(p0[0]), float(start_strip_a[1]) - float(p0[1])))
                tail_len = float(math.hypot(float(p1[0]) - float(start_strip_b[0]), float(p1[1]) - float(start_strip_b[1])))
                lead_count = max(3, int(lead_len / max(1.0, sample_spacing * 0.65)) + 1)
                strip_count = max(8, int(float(strip_len) / max(1.0, sample_spacing * 0.45)) + 1)
                tail_count = max(3, int(tail_len / max(1.0, sample_spacing * 0.65)) + 1)
                side_pts: list[tuple[float, float]] = []
                _append_no_dup(side_pts, _line_points(p0, start_strip_a, lead_count, template_kind="straight", template_amp=0.0, template_normal=outward))  # type: ignore[arg-type]
                _append_no_dup(side_pts, _line_points(start_strip_a, start_strip_b, strip_count, template_kind="straight", template_amp=0.0, template_normal=outward))
                _append_no_dup(side_pts, _line_points(start_strip_b, p1, tail_count, template_kind="straight", template_amp=0.0, template_normal=outward))  # type: ignore[arg-type]
            else:
                side_count = max(5, int(side_len / max(1.0, sample_spacing * 0.65)) + 1)
                template_kind, amp = local_side_params[side]
                side_pts = _line_points(
                    p0,  # type: ignore[arg-type]
                    p1,  # type: ignore[arg-type]
                    side_count,
                    template_kind=str(template_kind),
                    template_amp=float(amp),
                    template_normal=outward,  # type: ignore[arg-type]
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
        for side in LONG_SIDE_ORDER
        if str(side_params[side][0]) != "straight"
    )

    dense_np = np.asarray(dense_points, dtype=np.float64)
    if dense_np.shape[0] < 24:
        raise RuntimeError("Track generation failed to build a valid closed loop.")

    _, _, dense_len = _closed_segment_lengths(dense_np)
    sample_count = max(96, int(round(dense_len / max(1.0, sample_spacing))))
    sample_count = min(2400, sample_count)
    centerline = _resample_closed_polyline(dense_np, sample_count)

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

    start_tangent = (float(start_proj.tangent[0]), float(start_proj.tangent[1]))
    start_normal = (float(start_proj.normal[0]), float(start_proj.normal[1]))
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

    template_family = "rounded_long_side_templates"
    raw_road_polygon = np.vstack((left_boundary, right_boundary[::-1]))
    road_poly_points = clean_polygon_vertices(
        [(float(row[0]), float(row[1])) for row in raw_road_polygon],
        eps=1e-4,
    )
    if len(road_poly_points) < 6:
        raise RuntimeError("Road polygon collapsed during cleanup.")
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
        road_polygon=np.asarray(road_poly_points, dtype=np.float32),
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
    )
    validate_track_geometry(track)
    return track


def sample_track_at_s(
    track: TrackGeometry,
    s: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return _sample_arrays(
        centerline=np.asarray(track.centerline, dtype=np.float64),
        seg_vec=np.asarray(track._segment_vectors, dtype=np.float64),
        seg_len=np.asarray(track._segment_lengths, dtype=np.float64),
        seg_s=np.asarray(track._segment_s, dtype=np.float64),
        length=float(track.length),
        s=float(s),
    )


def start_strip_pose(track: TrackGeometry) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return sample_track_at_s(track, float(track.start_s))


def project_point_to_track(track: TrackGeometry, pos: tuple[float, float]) -> TrackProjection:
    return _build_projection(
        centerline=np.asarray(track.centerline, dtype=np.float64),
        seg_vec=np.asarray(track._segment_vectors, dtype=np.float64),
        seg_len=np.asarray(track._segment_lengths, dtype=np.float64),
        seg_s=np.asarray(track._segment_s, dtype=np.float64),
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
    polygon = np.asarray(track.road_polygon, dtype=np.float64)
    return raycast_polygon(
        origin=(float(origin[0]), float(origin[1])),
        direction=(float(direction[0]), float(direction[1])),
        polygon=polygon,
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
