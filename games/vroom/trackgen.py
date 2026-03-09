"""Geometry-first track generation wrapper for Vroom."""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np
from PIL import Image, ImageDraw

from core.arcade_style import COLOR_DARK_NEUTRAL
from games.vroom.track_geometry import (
    TrackGeometry,
    build_boundary_loops,
    build_track_geometry,
    clean_polygon_vertices,
)


@dataclass(frozen=True)
class TrackGenConfig:
    track_width_px: float = 85.0
    padding_px: float = 40.0
    footprint_scale: float = 0.975
    corner_radius_px: float = 130.0
    sample_spacing_px: float = 6.0
    start_straight_len_px: float = 180.0
    template_min_bulged_sides: int = 0
    template_max_bulged_sides: int = 3
    bulge_amplitude_min_px: float = 14.0
    bulge_amplitude_max_px: float = 40.0
    bulge_width_cap_ratio: float = 0.62
    bulge_length_cap_ratio: float = 0.16
    bulge_short_side_threshold_px: float = 260.0
    bulge_short_side_length_cap_ratio: float = 0.12


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
    image = Image.fromarray(rgba, mode="RGBA")
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
        template_min_bulged_sides=int(cfg.template_min_bulged_sides),
        template_max_bulged_sides=int(cfg.template_max_bulged_sides),
        bulge_amplitude_min_px=float(cfg.bulge_amplitude_min_px),
        bulge_amplitude_max_px=float(cfg.bulge_amplitude_max_px),
        bulge_width_cap_ratio=float(cfg.bulge_width_cap_ratio),
        bulge_length_cap_ratio=float(cfg.bulge_length_cap_ratio),
        bulge_short_side_threshold_px=float(cfg.bulge_short_side_threshold_px),
        bulge_short_side_length_cap_ratio=float(cfg.bulge_short_side_length_cap_ratio),
    )
    road_mask = build_track_mask(track=geometry, width=int(width), height=int(height))
    collision_mask = np.asarray(road_mask, dtype=np.uint8)

    if bool(build_texture):
        mask_signature = hash(road_mask.tobytes())
        road_texture = mask_to_texture(
            road_mask,
            texture_name=f"vroom_track_{seed}_{width}x{height}_{mask_signature}",
            track_color=track_color,
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
