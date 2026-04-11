"""Geometry-first track generation wrapper for Vroom."""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np
from PIL import Image, ImageDraw

from core.arcade_style import COLOR_DARK_NEUTRAL, COLOR_LIGHT_NEUTRAL, DEFAULT_TILE_SIZE
from core.primitives import (
    ROAD_DASH_COLOR,
    build_dashed_path_top_left_polygons,
    road_dash_gap,
    road_dash_length,
    road_dash_thickness,
)
from games.vroom.track_geometry import (
    TrackGeometry,
    build_boundary_loops,
    build_track_geometry,
    clean_polygon_vertices,
    sample_track_at_s,
)


@dataclass(frozen=True)
class TrackGenConfig:
    track_width_px: float = 85.0
    padding_px: float = 8.0
    footprint_scale: float = 1.0
    corner_radius_px: float = 130.0
    sample_spacing_px: float = 6.0
    start_straight_len_px: float = 180.0
    long_side_template_choices: tuple[str, ...] = ("straight", "bell", "s_curve")
    bell_amplitude_min_px: float = 14.0
    bell_amplitude_max_px: float = 40.0
    s_amplitude_min_px: float = 10.0
    s_amplitude_max_px: float = 28.0
    inset_width_cap_ratio: float = 0.62
    inset_length_cap_ratio: float = 0.16


def _road_polygon(track: TrackGeometry) -> list[tuple[float, float]]:
    left_pts, right_pts = build_boundary_loops(track, seam_index=int(track.start_index))
    if len(left_pts) <= 2 or len(right_pts) <= 2:
        return []
    return clean_polygon_vertices(left_pts + list(reversed(right_pts)), eps=1e-4)


def build_track_mask(track: TrackGeometry, width: int, height: int) -> np.ndarray:
    mask_img = Image.new("L", (int(width), int(height)), 0)
    polygon = _road_polygon(track)
    if polygon:
        drawer = ImageDraw.Draw(mask_img)
        drawer.polygon(polygon, fill=255)
        # Seal raster edge seams so tiny one-pixel cracks cannot appear in the fill.
        ring = list(polygon)
        ring.append(polygon[0])
        drawer.line(ring, fill=255, width=3)
    return np.asarray(mask_img, dtype=np.uint8)


def build_start_checker_polygons(
    track: TrackGeometry,
) -> list[tuple[list[tuple[float, float]], tuple[int, int, int] | tuple[int, int, int, int]]]:
    left_pt = (float(track.start_line[0][0]), float(track.start_line[0][1]))
    right_pt = (float(track.start_line[1][0]), float(track.start_line[1][1]))
    dx = float(right_pt[0]) - float(left_pt[0])
    dy = float(right_pt[1]) - float(left_pt[1])
    line_len = max(1e-6, float(np.hypot(dx, dy)))
    tx = float(track.start_tangent[0])
    ty = float(track.start_tangent[1])

    row_count = 3
    cell_size = max(1.0, 0.5 * float(DEFAULT_TILE_SIZE))
    col_count = max(1, int(np.ceil(float(line_len) / float(cell_size))))
    start_offset = -0.5 * float(row_count) * float(cell_size)

    polygons: list[tuple[list[tuple[float, float]], tuple[int, int, int] | tuple[int, int, int, int]]] = []
    for row in range(row_count):
        row_offset_0 = float(start_offset + row * cell_size)
        row_offset_1 = float(row_offset_0 + cell_size)
        for col in range(col_count):
            cell_color = COLOR_DARK_NEUTRAL if ((row + col) % 2) == 0 else COLOR_LIGHT_NEUTRAL
            alpha0 = float(col) / float(col_count)
            alpha1 = float(col + 1) / float(col_count)
            x00 = float(left_pt[0]) + dx * alpha0 + tx * row_offset_0
            y00 = float(left_pt[1]) + dy * alpha0 + ty * row_offset_0
            x10 = float(left_pt[0]) + dx * alpha1 + tx * row_offset_0
            y10 = float(left_pt[1]) + dy * alpha1 + ty * row_offset_0
            x11 = float(left_pt[0]) + dx * alpha1 + tx * row_offset_1
            y11 = float(left_pt[1]) + dy * alpha1 + ty * row_offset_1
            x01 = float(left_pt[0]) + dx * alpha0 + tx * row_offset_1
            y01 = float(left_pt[1]) + dy * alpha0 + ty * row_offset_1
            polygons.append(
                (
                    [
                        (float(x00), float(y00)),
                        (float(x10), float(y10)),
                        (float(x11), float(y11)),
                        (float(x01), float(y01)),
                    ],
                    cell_color,
                )
            )
    return polygons


def mask_to_texture(
    mask: np.ndarray,
    *,
    texture_name: str,
    track_color: tuple[int, int, int],
    dash_polygons: list[list[tuple[float, float]]] | None = None,
    start_checker_polygons: list[
        tuple[list[tuple[float, float]], tuple[int, int, int] | tuple[int, int, int, int]]
    ] | None = None,
) -> arcade.Texture:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = int(track_color[0])
    rgba[..., 1] = int(track_color[1])
    rgba[..., 2] = int(track_color[2])
    rgba[..., 3] = mask
    image = Image.fromarray(rgba, mode="RGBA")
    if dash_polygons or start_checker_polygons:
        overlay = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
        drawer = ImageDraw.Draw(overlay, "RGBA")
        if dash_polygons:
            for polygon in dash_polygons:
                if len(polygon) >= 3:
                    drawer.polygon(polygon, fill=ROAD_DASH_COLOR)
        if start_checker_polygons:
            for polygon, color in start_checker_polygons:
                if len(polygon) >= 3:
                    drawer.polygon(polygon, fill=color)
        image = Image.alpha_composite(image, overlay)
    return arcade.Texture(image=image, hash=str(texture_name))


def generate_track(
    seed: int,
    width: int,
    height: int,
    config: TrackGenConfig | None = None,
    *,
    build_texture: bool = True,
    track_color: tuple[int, int, int] = COLOR_DARK_NEUTRAL,
) -> dict[str, object]:
    cfg = config or TrackGenConfig()
    geometry = build_track_geometry(
        seed=int(seed),
        width=int(width),
        height=int(height),
        track_width_px=float(cfg.track_width_px),
        padding_px=float(cfg.padding_px),
        footprint_scale=float(cfg.footprint_scale),
        corner_radius_px=float(cfg.corner_radius_px),
        sample_spacing_px=float(cfg.sample_spacing_px),
        start_straight_len_px=float(cfg.start_straight_len_px),
        long_side_template_choices=tuple(str(value) for value in cfg.long_side_template_choices),
        bell_amplitude_min_px=float(cfg.bell_amplitude_min_px),
        bell_amplitude_max_px=float(cfg.bell_amplitude_max_px),
        s_amplitude_min_px=float(cfg.s_amplitude_min_px),
        s_amplitude_max_px=float(cfg.s_amplitude_max_px),
        inset_width_cap_ratio=float(cfg.inset_width_cap_ratio),
        inset_length_cap_ratio=float(cfg.inset_length_cap_ratio),
    )
    road_mask = build_track_mask(track=geometry, width=int(width), height=int(height))
    collision_mask = np.asarray(road_mask, dtype=np.uint8)

    if bool(build_texture):
        mask_signature = hash(road_mask.tobytes())
        dash_base = float(geometry.half_width) * 2.0
        dash_length = road_dash_length(float(dash_base)) * 0.84
        dash_width = road_dash_thickness(float(dash_base))
        dash_gap = road_dash_gap(float(dash_base))
        dash_polygons = build_dashed_path_top_left_polygons(
            path_length=float(geometry.length),
            sample_fn=lambda s: sample_track_at_s(geometry, float(geometry.start_s) + float(s))[:2],
            dash_length=float(dash_length),
            dash_width=float(dash_width),
            gap_length=float(dash_gap),
            curve_step=max(4.0, min(10.0, 0.24 * float(dash_length))),
        )
        start_checker_polygons = build_start_checker_polygons(geometry)
        road_texture = mask_to_texture(
            road_mask,
            texture_name=f"vroom_track_{seed}_{width}x{height}_{mask_signature}",
            track_color=track_color,
            dash_polygons=dash_polygons,
            start_checker_polygons=start_checker_polygons,
        )
    else:
        road_texture = None

    centerline = [
        (float(x), float(y))
        for x, y in np.asarray(geometry.centerline, dtype=np.float32).tolist()
    ]

    return {
        "geometry": geometry,
        "centerline": centerline,
        "road_mask": road_mask,
        "collision_mask": collision_mask,
        "road_texture": road_texture,
        "wall_texture": None,
        "start_index": int(geometry.start_index),
        "start_side": str(geometry.start_side),
        "start_line": geometry.start_line,
    }
