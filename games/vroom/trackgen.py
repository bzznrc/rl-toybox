"""Geometry-first track generation wrapper for Vroom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from core.arcade_style import COLOR_DARK_NEUTRAL, COLOR_FOG_GRAY, COLOR_LIGHT_NEUTRAL, DEFAULT_TILE_SIZE
from games.vroom.track_geometry import (
    TrackGeometry,
    build_boundary_loops,
    build_track_geometry,
)

if TYPE_CHECKING:
    import arcade


@dataclass(frozen=True)
class TrackGenConfig:
    track_width_px: float = 80.0
    padding_px: float = 8.0
    footprint_scale: float = 1.0
    corner_radius_px: float = 130.0
    sample_spacing_px: float = 6.0
    start_straight_len_px: float = 180.0
    long_side_template_choices: tuple[str, ...] = ("straight", "bell", "s_curve", "fold")
    short_side_template_choices: tuple[str, ...] = ("straight", "bell")
    bell_amplitude_min_px: float = 14.0
    bell_amplitude_max_px: float = 40.0
    s_amplitude_min_px: float = 10.0
    s_amplitude_max_px: float = 28.0
    inset_width_cap_ratio: float = 0.62
    inset_length_cap_ratio: float = 0.16
    fold_gap_px: float = 16.0
    generation_max_attempts: int = 50
    complexity_min: float = 0.0
    complexity_max: float = 0.0
    use_complexity_filter: bool = True


ROAD_EDGE_WIDTH_PX = max(2, int(round(float(DEFAULT_TILE_SIZE) * 0.18)))
TRACK_TEXTURE_SUPERSAMPLE = 4


def _road_polygon(track: TrackGeometry) -> list[tuple[float, float]]:
    polygon = np.asarray(track.road_polygon, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] <= 2 or polygon.shape[1] != 2:
        return []
    return [(float(x), float(y)) for x, y in polygon.tolist()]


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


def _scale_polygon(
    polygon: list[tuple[float, float]],
    scale: int,
) -> list[tuple[float, float]]:
    factor = max(1, int(scale))
    return [(float(x) * factor, float(y) * factor) for x, y in polygon]


def _downsample_rgba(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgba = np.asarray(image, dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    premultiplied = rgba[..., :3] * alpha

    premul_img = Image.fromarray(np.clip(premultiplied, 0.0, 255.0).astype(np.uint8), mode="RGB")
    alpha_img = Image.fromarray(np.clip(rgba[..., 3], 0.0, 255.0).astype(np.uint8), mode="L")
    premul_small = np.asarray(premul_img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    alpha_small = np.asarray(alpha_img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)

    alpha_ratio = alpha_small[..., None] / 255.0
    rgb = np.zeros_like(premul_small)
    np.divide(premul_small, np.maximum(alpha_ratio, 1e-6), out=rgb, where=alpha_ratio > 1e-6)
    out = np.dstack((np.clip(rgb, 0.0, 255.0), np.clip(alpha_small, 0.0, 255.0)))
    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


def build_track_texture(
    track: TrackGeometry,
    width: int,
    height: int,
    *,
    texture_name: str,
    track_color: tuple[int, int, int],
) -> arcade.Texture:
    import arcade

    scale = max(1, int(TRACK_TEXTURE_SUPERSAMPLE))
    scaled_size = (int(width) * scale, int(height) * scale)
    image = Image.new("RGBA", scaled_size, (0, 0, 0, 0))
    alpha_mask = Image.new("L", scaled_size, 0)
    drawer = ImageDraw.Draw(image, "RGBA")
    mask_drawer = ImageDraw.Draw(alpha_mask)

    road_polygon = _road_polygon(track)
    if not road_polygon:
        empty = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
        return arcade.Texture(image=empty, hash=str(texture_name))

    scaled_road = _scale_polygon(road_polygon, scale)
    fill_color = (int(track_color[0]), int(track_color[1]), int(track_color[2]), 255)
    outline_color = (int(COLOR_FOG_GRAY[0]), int(COLOR_FOG_GRAY[1]), int(COLOR_FOG_GRAY[2]), 255)
    edge_width = max(1, int(ROAD_EDGE_WIDTH_PX) * scale * 2)
    start_width = max(1, int(edge_width) * 2)

    drawer.polygon(scaled_road, fill=fill_color)
    mask_drawer.polygon(scaled_road, fill=255)

    left_boundary, right_boundary = build_boundary_loops(track, seam_index=int(track.start_index))
    for boundary in (left_boundary, right_boundary):
        if len(boundary) <= 1:
            continue
        drawer.line(_scale_polygon(boundary, scale), fill=outline_color, width=edge_width, joint="curve")

    start_line = _scale_polygon(
        [
            (float(track.start_line[0][0]), float(track.start_line[0][1])),
            (float(track.start_line[1][0]), float(track.start_line[1][1])),
        ],
        scale,
    )
    start_color = (
        int(COLOR_LIGHT_NEUTRAL[0]),
        int(COLOR_LIGHT_NEUTRAL[1]),
        int(COLOR_LIGHT_NEUTRAL[2]),
        255,
    )
    drawer.line(start_line, fill=fill_color, width=max(start_width + edge_width, start_width), joint="curve")
    drawer.line(start_line, fill=start_color, width=start_width)

    image_rgba = np.asarray(image, dtype=np.uint8).copy()
    image_rgba[..., 3] = np.minimum(image_rgba[..., 3], np.asarray(alpha_mask, dtype=np.uint8))
    image = Image.fromarray(image_rgba, mode="RGBA")
    image = _downsample_rgba(image, (int(width), int(height)))
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
        short_side_template_choices=tuple(str(value) for value in cfg.short_side_template_choices),
        bell_amplitude_min_px=float(cfg.bell_amplitude_min_px),
        bell_amplitude_max_px=float(cfg.bell_amplitude_max_px),
        s_amplitude_min_px=float(cfg.s_amplitude_min_px),
        s_amplitude_max_px=float(cfg.s_amplitude_max_px),
        inset_width_cap_ratio=float(cfg.inset_width_cap_ratio),
        inset_length_cap_ratio=float(cfg.inset_length_cap_ratio),
        fold_gap_px=float(cfg.fold_gap_px),
        generation_max_attempts=int(cfg.generation_max_attempts),
        complexity_min=float(cfg.complexity_min),
        complexity_max=float(cfg.complexity_max),
        use_complexity_filter=bool(cfg.use_complexity_filter),
    )
    road_mask = build_track_mask(track=geometry, width=int(width), height=int(height))
    collision_mask = np.asarray(road_mask, dtype=np.uint8)

    if bool(build_texture):
        geometry_signature = hash(np.asarray(geometry.road_polygon, dtype=np.float32).tobytes())
        road_texture = build_track_texture(
            geometry,
            width=int(width),
            height=int(height),
            texture_name=f"vroom_track_smooth_v3_{seed}_{width}x{height}_{geometry_signature}",
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
