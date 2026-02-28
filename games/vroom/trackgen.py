"""Simple rounded-rectangle perimeter track generation for Vroom."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import arcade
import numpy as np
from PIL import Image

from core.arcade_style import COLOR_CHARCOAL, COLOR_FOG_GRAY


@dataclass(frozen=True)
class TrackGenConfig:
    track_width_px: float = 88.0
    wall_thickness_px: float = 4.0
    padding_px: float = 40.0
    corner_radius_px: float = 130.0
    line_safe_margin_px: float = 96.0
    base_step_px: float = 6.0
    arc_points: int = 18
    obstacle_tile_px: int = 20
    obstacle_clusters: int = 4
    obstacle_min_road_coverage: float = 1.0
    obstacle_edge_margin_px: int = 2


def _append_no_dup(target: list[tuple[float, float]], source: list[tuple[float, float]]) -> None:
    if not source:
        return
    if not target:
        target.extend(source)
        return
    sx, sy = source[0]
    tx, ty = target[-1]
    if abs(sx - tx) < 1e-6 and abs(sy - ty) < 1e-6:
        target.extend(source[1:])
    else:
        target.extend(source)


def _line_points(p0: tuple[float, float], p1: tuple[float, float], step: float) -> list[tuple[float, float]]:
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    count = max(2, int(dist / max(1.0, float(step))) + 1)
    return [(x0 + (x1 - x0) * (i / float(count - 1)), y0 + (y1 - y0) * (i / float(count - 1))) for i in range(count)]


def _arc_points(
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    count: int,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    point_count = max(4, int(count))
    for i in range(point_count):
        t = i / float(point_count - 1)
        ang = start_angle + (end_angle - start_angle) * t
        out.append((cx + math.cos(ang) * radius, cy + math.sin(ang) * radius))
    return out


def _disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return np.where(xx * xx + yy * yy <= r * r, 255, 0).astype(np.uint8)


def _stamp_point(mask: np.ndarray, kernel: np.ndarray, cx: int, cy: int) -> None:
    radius = kernel.shape[0] // 2
    x0 = cx - radius
    y0 = cy - radius
    x1 = x0 + kernel.shape[1]
    y1 = y0 + kernel.shape[0]

    mx0 = max(0, x0)
    my0 = max(0, y0)
    mx1 = min(mask.shape[1], x1)
    my1 = min(mask.shape[0], y1)
    if mx0 >= mx1 or my0 >= my1:
        return

    kx0 = mx0 - x0
    ky0 = my0 - y0
    kx1 = kx0 + (mx1 - mx0)
    ky1 = ky0 + (my1 - my0)
    np.maximum(mask[my0:my1, mx0:mx1], kernel[ky0:ky1, kx0:kx1], out=mask[my0:my1, mx0:mx1])


def _stamp_polyline(mask: np.ndarray, points: list[tuple[float, float]], kernel: np.ndarray) -> None:
    for px, py in points:
        _stamp_point(mask, kernel, int(round(px)), int(round(py)))


def _build_perimeter_centerline(
    width: int,
    height: int,
    cfg: TrackGenConfig,
) -> tuple[list[tuple[float, float]], dict[str, tuple[int, int]], float]:
    margin = float(cfg.padding_px) + 0.5 * float(cfg.track_width_px) + float(cfg.wall_thickness_px)
    left = margin
    right = float(width) - margin
    top = margin
    bottom = float(height) - margin
    corner_r = min(
        float(cfg.corner_radius_px),
        max(16.0, 0.5 * (right - left) - 10.0),
        max(16.0, 0.5 * (bottom - top) - 10.0),
    )

    step = max(2.0, float(cfg.base_step_px))
    arc_pts = max(10, int(cfg.arc_points))
    points: list[tuple[float, float]] = []
    side_ranges: dict[str, tuple[int, int]] = {}

    def _add_side(name: str, side_points: list[tuple[float, float]]) -> None:
        start = len(points)
        _append_no_dup(points, side_points)
        side_ranges[name] = (start, len(points) - 1)

    _add_side("top", _line_points((left + corner_r, top), (right - corner_r, top), step))
    _append_no_dup(points, _arc_points(right - corner_r, top + corner_r, corner_r, -math.pi / 2.0, 0.0, arc_pts))

    _add_side("right", _line_points((right, top + corner_r), (right, bottom - corner_r), step))
    _append_no_dup(points, _arc_points(right - corner_r, bottom - corner_r, corner_r, 0.0, math.pi / 2.0, arc_pts))

    _add_side("bottom", _line_points((right - corner_r, bottom), (left + corner_r, bottom), step))
    _append_no_dup(points, _arc_points(left + corner_r, bottom - corner_r, corner_r, math.pi / 2.0, math.pi, arc_pts))

    _add_side("left", _line_points((left, bottom - corner_r), (left, top + corner_r), step))
    _append_no_dup(points, _arc_points(left + corner_r, top + corner_r, corner_r, math.pi, 3.0 * math.pi / 2.0, arc_pts))

    return points, side_ranges, step


def build_track_mask(
    centerline: list[tuple[float, float]],
    width: int,
    height: int,
    track_width_px: float,
) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    kernel = _disk_kernel(max(2, int(round(float(track_width_px) * 0.5))))
    _stamp_polyline(mask, centerline, kernel)
    return mask


def mask_to_texture(
    mask: np.ndarray,
    *,
    texture_name: str,
    track_color: tuple[int, int, int],
) -> arcade.Texture:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = int(track_color[0])
    rgba[..., 1] = int(track_color[1])
    rgba[..., 2] = int(track_color[2])
    rgba[..., 3] = mask
    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    return arcade.Texture(image=image, hash=str(texture_name))


def _pick_start_line(
    rng: np.random.Generator,
    centerline: list[tuple[float, float]],
    side_ranges: dict[str, tuple[int, int]],
    step_px: float,
    track_width_px: float,
    safe_margin_px: float,
) -> tuple[int, str, tuple[tuple[float, float], tuple[float, float]]]:
    sides = ("top", "right", "bottom", "left")
    chosen_side = str(rng.choice(sides))
    side_start, side_end = side_ranges[chosen_side]

    safe_pts = max(4, int(float(safe_margin_px) / max(1.0, float(step_px))))
    low = side_start + safe_pts
    high = side_end - safe_pts
    if high <= low:
        low, high = side_start, side_end
    start_idx = int(rng.integers(low, high + 1))

    prev_idx = (start_idx - 1) % len(centerline)
    next_idx = (start_idx + 1) % len(centerline)
    px, py = centerline[start_idx]
    tx = centerline[next_idx][0] - centerline[prev_idx][0]
    ty = centerline[next_idx][1] - centerline[prev_idx][1]
    t_len = math.hypot(tx, ty)
    if t_len <= 1e-9:
        tx, ty = 1.0, 0.0
    else:
        tx, ty = tx / t_len, ty / t_len
    nx, ny = -ty, tx
    half_w = 0.5 * float(track_width_px)

    p1 = (float(px + nx * half_w), float(py + ny * half_w))
    p2 = (float(px - nx * half_w), float(py - ny * half_w))
    return int(start_idx), chosen_side, (p1, p2)


def _generate_obstacle_tiles(
    *,
    seed: int,
    width: int,
    height: int,
    cfg: TrackGenConfig,
    road_mask: np.ndarray,
    centerline: list[tuple[float, float]],
    side_ranges: dict[str, tuple[int, int]],
    start_index: int,
    start_point: tuple[float, float],
    start_clearance_px: float,
) -> list[tuple[float, float]]:
    tile_px = max(8, int(cfg.obstacle_tile_px))
    grid_w = int(width) // tile_px
    grid_h = int(height) // tile_px
    point_count = int(len(centerline))
    if grid_w <= 1 or grid_h <= 1 or point_count <= 8:
        return []

    rng = random.Random(int(seed) + 19117)
    occupied: set[tuple[int, int]] = set()
    used_indices: list[int] = []
    tiles: list[tuple[float, float]] = []

    index_spacing = max(5, int(float(cfg.track_width_px) / max(1.0, float(cfg.base_step_px))))
    start_guard = max(8, int(float(cfg.line_safe_margin_px) / max(1.0, float(cfg.base_step_px))))
    max_groups = max(0, int(cfg.obstacle_clusters))

    def _index_delta(a: int, b: int) -> int:
        diff = abs(int(a) - int(b))
        return min(diff, point_count - diff)

    def _dominant_grid_step(dx: float, dy: float) -> tuple[int, int]:
        if abs(float(dx)) >= abs(float(dy)):
            return (1 if float(dx) >= 0.0 else -1, 0)
        return (0, 1 if float(dy) >= 0.0 else -1)

    def _pixel_tile_valid(x0: int, y0: int) -> bool:
        x1 = int(x0) + tile_px
        y1 = int(y0) + tile_px
        if int(x0) < 0 or int(y0) < 0 or x1 > int(width) or y1 > int(height):
            return False
        area = road_mask[int(y0):y1, int(x0):x1]
        if area.shape[0] != tile_px or area.shape[1] != tile_px:
            return False
        coverage = float(np.count_nonzero(area)) / float(area.size)
        if coverage < float(cfg.obstacle_min_road_coverage):
            return False
        margin = max(0, int(cfg.obstacle_edge_margin_px))
        if margin <= 0:
            return True
        ex0 = int(x0) - margin
        ey0 = int(y0) - margin
        ex1 = x1 + margin
        ey1 = y1 + margin
        if ex0 < 0 or ey0 < 0 or ex1 > int(width) or ey1 > int(height):
            return False
        expanded = road_mask[ey0:ey1, ex0:ex1]
        return bool(expanded.size > 0 and np.all(expanded > 0))

    candidate_indices: list[int] = []
    corner_guard = max(
        4,
        int(
            (
                0.5 * float(cfg.track_width_px)
                + float(cfg.obstacle_tile_px)
                + float(cfg.obstacle_edge_margin_px)
            )
            / max(1.0, float(cfg.base_step_px))
        ),
    )
    for side_name in ("top", "right", "bottom", "left"):
        start_end = side_ranges.get(side_name)
        if not start_end:
            continue
        side_start, side_end = int(start_end[0]), int(start_end[1])
        if side_end < side_start:
            continue
        low = side_start + corner_guard
        high = side_end - corner_guard
        if high < low:
            continue
        candidate_indices.extend(range(low, high + 1))
    if not candidate_indices:
        candidate_indices = list(range(point_count))
    rng.shuffle(candidate_indices)

    for center_idx in candidate_indices:
        if len(used_indices) >= max_groups:
            break
        if _index_delta(int(center_idx), int(start_index)) < start_guard:
            continue
        if any(_index_delta(int(center_idx), int(existing)) < index_spacing for existing in used_indices):
            continue

        prev_idx = (int(center_idx) - 1) % point_count
        next_idx = (int(center_idx) + 1) % point_count
        tx = float(centerline[next_idx][0] - centerline[prev_idx][0])
        ty = float(centerline[next_idx][1] - centerline[prev_idx][1])
        nx, ny = -ty, tx

        n_step_x, n_step_y = _dominant_grid_step(nx, ny)
        t_step_x, t_step_y = _dominant_grid_step(tx, ty)
        if t_step_x == n_step_x and t_step_y == n_step_y:
            t_step_x, t_step_y = n_step_y, -n_step_x
        if rng.random() < 0.5:
            t_step_x, t_step_y = -t_step_x, -t_step_y

        px, py = centerline[int(center_idx)]
        base_gx = int(round(float(px) / float(tile_px)))
        base_gy = int(round(float(py) / float(tile_px)))

        # Keep obstacles as 1x2 only, oriented across the lane (perpendicular to tangent),
        # and try extreme lateral offsets first so some obstacles force one-side routing.
        offset_candidates = [-3, 2, -2, 1, -1, 0]
        rng.shuffle(offset_candidates)

        chosen_shape: list[tuple[int, int]] | None = None
        chosen_offset = 0
        for n_offset in offset_candidates:
            local_offsets = [(int(n_offset), 0), (int(n_offset) + 1, 0)]
            shape_tiles: list[tuple[int, int]] = []
            valid_shape = True
            for local_n, local_t in local_offsets:
                gx = int(base_gx + local_n * n_step_x + local_t * t_step_x)
                gy = int(base_gy + local_n * n_step_y + local_t * t_step_y)
                if gx < 0 or gy < 0 or gx >= grid_w or gy >= grid_h:
                    valid_shape = False
                    break
                if (gx, gy) in occupied or any((gx, gy) == existing for existing in shape_tiles):
                    valid_shape = False
                    break
                if not _pixel_tile_valid(int(gx) * tile_px, int(gy) * tile_px):
                    valid_shape = False
                    break
                tile_cx = float(gx * tile_px + tile_px * 0.5)
                tile_cy = float(gy * tile_px + tile_px * 0.5)
                if math.hypot(tile_cx - float(start_point[0]), tile_cy - float(start_point[1])) < float(start_clearance_px):
                    valid_shape = False
                    break
                shape_tiles.append((gx, gy))
            if valid_shape:
                chosen_shape = shape_tiles
                chosen_offset = int(n_offset)
                break

        if chosen_shape is None:
            continue

        side_sign = -1 if int(chosen_offset) < 0 else 1
        max_shift_px = max(0, int(float(tile_px) * 0.45))
        shift_x = 0
        shift_y = 0
        for shift_px in range(max_shift_px, -1, -1):
            cand_x = int(n_step_x * side_sign * shift_px)
            cand_y = int(n_step_y * side_sign * shift_px)
            valid_shift = True
            for gx, gy in chosen_shape:
                px0 = int(gx) * tile_px + cand_x
                py0 = int(gy) * tile_px + cand_y
                if not _pixel_tile_valid(px0, py0):
                    valid_shift = False
                    break
                tile_cx = float(px0 + tile_px * 0.5)
                tile_cy = float(py0 + tile_px * 0.5)
                if math.hypot(tile_cx - float(start_point[0]), tile_cy - float(start_point[1])) < float(start_clearance_px):
                    valid_shift = False
                    break
            if valid_shift:
                shift_x = cand_x
                shift_y = cand_y
                break

        for gx, gy in chosen_shape:
            occupied.add((gx, gy))
            tiles.append((float(int(gx) * tile_px + shift_x), float(int(gy) * tile_px + shift_y)))
        used_indices.append(int(center_idx))

    return tiles


def generate_track(
    seed: int,
    width: int,
    height: int,
    config: TrackGenConfig | None = None,
    *,
    build_texture: bool = True,
) -> dict[str, object]:
    cfg = config or TrackGenConfig()
    rng = np.random.default_rng(int(seed))

    centerline, side_ranges, step_px = _build_perimeter_centerline(int(width), int(height), cfg)
    if len(centerline) < 24:
        raise RuntimeError("Perimeter track generation failed.")

    start_index, start_side, start_line = _pick_start_line(
        rng=rng,
        centerline=centerline,
        side_ranges=side_ranges,
        step_px=step_px,
        track_width_px=float(cfg.track_width_px),
        safe_margin_px=float(cfg.line_safe_margin_px),
    )

    road_mask = build_track_mask(
        centerline,
        width=int(width),
        height=int(height),
        track_width_px=float(cfg.track_width_px),
    )
    wall_outer_mask = build_track_mask(
        centerline,
        width=int(width),
        height=int(height),
        track_width_px=float(cfg.track_width_px) + 2.0 * float(cfg.wall_thickness_px),
    )
    wall_mask = np.where((wall_outer_mask > 0) & (road_mask == 0), 255, 0).astype(np.uint8)

    obstacles = _generate_obstacle_tiles(
        seed=int(seed),
        width=int(width),
        height=int(height),
        cfg=cfg,
        road_mask=road_mask,
        centerline=centerline,
        side_ranges=side_ranges,
        start_index=int(start_index),
        start_point=tuple(centerline[int(start_index)]),
        start_clearance_px=max(48.0, float(cfg.track_width_px) * 1.6),
    )
    obstacle_mask = np.zeros_like(road_mask)
    tile_px = max(8, int(cfg.obstacle_tile_px))
    max_w = int(width)
    max_h = int(height)
    for ox, oy in obstacles:
        x0 = int(round(float(ox)))
        y0 = int(round(float(oy)))
        x1 = min(max_w, x0 + tile_px)
        y1 = min(max_h, y0 + tile_px)
        if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
            continue
        obstacle_mask[y0:y1, x0:x1] = 255
    wall_mask = np.maximum(wall_mask, obstacle_mask).astype(np.uint8)
    collision_mask = np.where((road_mask > 0) & (obstacle_mask == 0), 255, 0).astype(np.uint8)

    if bool(build_texture):
        wall_texture = mask_to_texture(
            wall_mask,
            texture_name=f"vroom_wall_{seed}",
            track_color=COLOR_FOG_GRAY,
        )
        road_texture = mask_to_texture(
            road_mask,
            texture_name=f"vroom_track_{seed}",
            track_color=COLOR_CHARCOAL,
        )
    else:
        wall_texture = None
        road_texture = None

    return {
        "centerline": centerline,
        "road_mask": road_mask,
        "collision_mask": collision_mask,
        "wall_mask": wall_mask,
        "road_texture": road_texture,
        "wall_texture": wall_texture,
        "start_index": int(start_index),
        "start_side": str(start_side),
        "start_line": start_line,
        "obstacles": obstacles,
        "obstacle_mask": obstacle_mask,
    }
