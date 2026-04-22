"""Geometry-first track generation wrapper for Vroom."""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np
from PIL import Image, ImageDraw

from core.arcade_style import COLOR_DARK_NEUTRAL, COLOR_FOG_GRAY, COLOR_LIGHT_NEUTRAL, DEFAULT_TILE_SIZE
from core.primitives import with_alpha
from games.vroom.track_geometry import (
    TrackGeometry,
    build_boundary_loops,
    build_track_geometry,
    clean_polygon_vertices,
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


ROAD_BLOCK_SIZE_PX = max(1, int(DEFAULT_TILE_SIZE) // 2)
ROAD_BLOCK_COVERAGE_THRESHOLD = 0.34
ROAD_BLOCK_OUTLINE_WIDTH_PX = max(2, int(round(float(DEFAULT_TILE_SIZE) * 0.18)))
VROOM_SURFACE_PATTERN_ALPHA = 84
VROOM_SURFACE_PATTERN_DARK_COLOR = with_alpha(COLOR_DARK_NEUTRAL, VROOM_SURFACE_PATTERN_ALPHA)
VROOM_SURFACE_PATTERN_LIGHT_COLOR = with_alpha(COLOR_FOG_GRAY, VROOM_SURFACE_PATTERN_ALPHA)
VROOM_SURFACE_PATTERN_DENSITY = 0.28
VROOM_SURFACE_PATTERN_SQUARE_SIZE_PX = max(2, int(round(float(ROAD_BLOCK_SIZE_PX) * 0.25)))


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


def build_track_block_mask(
    mask: np.ndarray,
    *,
    block_size_px: int = ROAD_BLOCK_SIZE_PX,
    coverage_threshold: float = ROAD_BLOCK_COVERAGE_THRESHOLD,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Track block mask expects a 2D road mask.")

    block_size = max(1, int(block_size_px))
    height, width = int(mask.shape[0]), int(mask.shape[1])
    pad_height = (-height) % block_size
    pad_width = (-width) % block_size
    padded = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant")

    grid_h = int(padded.shape[0] // block_size)
    grid_w = int(padded.shape[1] // block_size)
    coverage = padded.reshape(grid_h, block_size, grid_w, block_size).mean(axis=(1, 3)) / 255.0

    sample_y = np.minimum((np.arange(grid_h) * block_size) + (block_size // 2), padded.shape[0] - 1)
    sample_x = np.minimum((np.arange(grid_w) * block_size) + (block_size // 2), padded.shape[1] - 1)
    center_hits = padded[np.ix_(sample_y, sample_x)] > 0

    return np.logical_or(coverage >= float(coverage_threshold), center_hits)


def build_start_checker_polygons(
    track: TrackGeometry,
) -> list[tuple[list[tuple[float, float]], tuple[int, int, int] | tuple[int, int, int, int]]]:
    left_pt = (float(track.start_line[0][0]), float(track.start_line[0][1]))
    right_pt = (float(track.start_line[1][0]), float(track.start_line[1][1]))
    dx = float(right_pt[0]) - float(left_pt[0])
    dy = float(right_pt[1]) - float(left_pt[1])
    line_len = max(1e-6, float(np.hypot(dx, dy)))
    ux = dx / line_len
    uy = dy / line_len
    tx = float(track.start_tangent[0])
    ty = float(track.start_tangent[1])

    row_count = 3
    cell_size = float(ROAD_BLOCK_SIZE_PX)
    col_count = max(1, int(round(float(line_len) / float(cell_size))))
    covered_len = float(col_count) * float(cell_size)
    line_offset = 0.5 * float(max(0.0, line_len - covered_len))
    start_offset = -0.5 * float(row_count) * float(cell_size)

    polygons: list[tuple[list[tuple[float, float]], tuple[int, int, int] | tuple[int, int, int, int]]] = []
    for row in range(row_count):
        row_offset_0 = float(start_offset + row * cell_size)
        row_offset_1 = float(row_offset_0 + cell_size)
        for col in range(col_count):
            cell_color = COLOR_DARK_NEUTRAL if ((row + col) % 2) == 0 else COLOR_LIGHT_NEUTRAL
            line_pos_0 = float(line_offset + col * cell_size)
            line_pos_1 = float(line_pos_0 + cell_size)
            x00 = float(left_pt[0]) + ux * line_pos_0 + tx * row_offset_0
            y00 = float(left_pt[1]) + uy * line_pos_0 + ty * row_offset_0
            x10 = float(left_pt[0]) + ux * line_pos_1 + tx * row_offset_0
            y10 = float(left_pt[1]) + uy * line_pos_1 + ty * row_offset_0
            x11 = float(left_pt[0]) + ux * line_pos_1 + tx * row_offset_1
            y11 = float(left_pt[1]) + uy * line_pos_1 + ty * row_offset_1
            x01 = float(left_pt[0]) + ux * line_pos_0 + tx * row_offset_1
            y01 = float(left_pt[1]) + uy * line_pos_0 + ty * row_offset_1
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


def build_track_surface_pattern_rects(
    road_blocks: np.ndarray,
    *,
    pattern_seed: int,
    block_size_px: int = ROAD_BLOCK_SIZE_PX,
    square_size_px: int = VROOM_SURFACE_PATTERN_SQUARE_SIZE_PX,
    density: float = VROOM_SURFACE_PATTERN_DENSITY,
) -> list[tuple[int, int, int, int]]:
    if road_blocks.ndim != 2:
        raise ValueError("Track surface pattern expects a 2D road block mask.")

    block_size = max(1, int(block_size_px))
    square_size = max(1, min(int(square_size_px), block_size))
    density_ratio = max(0.0, min(1.0, float(density)))
    max_offset = max(0, block_size - square_size)
    rng = np.random.default_rng(int(pattern_seed))

    rects: list[tuple[int, int, int, int]] = []
    for row in range(int(road_blocks.shape[0])):
        for col in range(int(road_blocks.shape[1])):
            if not bool(road_blocks[row, col]):
                continue
            if float(rng.random()) > density_ratio:
                continue

            offset_x = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
            offset_y = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
            left = int(col * block_size + offset_x)
            top = int(row * block_size + offset_y)
            rects.append((left, top, left + square_size, top + square_size))

    return rects


def build_track_surface_pattern_marks(
    road_blocks: np.ndarray,
    *,
    pattern_seed: int,
    block_size_px: int = ROAD_BLOCK_SIZE_PX,
    square_size_px: int = VROOM_SURFACE_PATTERN_SQUARE_SIZE_PX,
    density: float = VROOM_SURFACE_PATTERN_DENSITY,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    rects = build_track_surface_pattern_rects(
        road_blocks,
        pattern_seed=int(pattern_seed),
        block_size_px=int(block_size_px),
        square_size_px=int(square_size_px),
        density=float(density),
    )
    if not rects:
        return []

    light_count = len(rects) // 2
    color_rng = np.random.default_rng(int(pattern_seed) ^ 0x9E3779B9)
    light_indices = {int(idx) for idx in color_rng.permutation(len(rects))[:light_count]}

    marks: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for idx, rect in enumerate(rects):
        color = VROOM_SURFACE_PATTERN_LIGHT_COLOR if idx in light_indices else VROOM_SURFACE_PATTERN_DARK_COLOR
        marks.append((rect, color))
    return marks


def mask_to_texture(
    mask: np.ndarray,
    *,
    texture_name: str,
    track_color: tuple[int, int, int],
    pattern_seed: int,
    start_checker_polygons: list[
        tuple[list[tuple[float, float]], tuple[int, int, int] | tuple[int, int, int, int]]
    ] | None = None,
) -> arcade.Texture:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    road_blocks = build_track_block_mask(mask)
    image = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(image, "RGBA")
    fill_color = (int(track_color[0]), int(track_color[1]), int(track_color[2]), 255)
    outline_color = (int(COLOR_FOG_GRAY[0]), int(COLOR_FOG_GRAY[1]), int(COLOR_FOG_GRAY[2]), 255)
    block_size = int(ROAD_BLOCK_SIZE_PX)
    outline_width = int(ROAD_BLOCK_OUTLINE_WIDTH_PX)

    for row in range(int(road_blocks.shape[0])):
        top = int(row * block_size)
        bottom = min(int(height), top + block_size)
        if bottom <= top:
            continue
        for col in range(int(road_blocks.shape[1])):
            if not bool(road_blocks[row, col]):
                continue
            left = int(col * block_size)
            right = min(int(width), left + block_size)
            if right <= left:
                continue
            drawer.rectangle((left, top, right - 1, bottom - 1), fill=fill_color)

    # Bake an irregular surface pattern into the static road texture once per
    # generated track instead of reconstructing it during frame rendering.
    for (left, top, right, bottom), color in build_track_surface_pattern_marks(
        road_blocks,
        pattern_seed=int(pattern_seed),
    ):
        drawer.rectangle(
            (left, top, right - 1, bottom - 1),
            fill=color,
        )

    for row in range(int(road_blocks.shape[0])):
        top = int(row * block_size)
        bottom = min(int(height), top + block_size)
        if bottom <= top:
            continue
        for col in range(int(road_blocks.shape[1])):
            if not bool(road_blocks[row, col]):
                continue
            left = int(col * block_size)
            right = min(int(width), left + block_size)
            if right <= left:
                continue

            top_open = row == 0 or not bool(road_blocks[row - 1, col])
            bottom_open = row == int(road_blocks.shape[0]) - 1 or not bool(road_blocks[row + 1, col])
            left_open = col == 0 or not bool(road_blocks[row, col - 1])
            right_open = col == int(road_blocks.shape[1]) - 1 or not bool(road_blocks[row, col + 1])

            if top_open:
                edge_bottom = min(bottom, top + outline_width)
                drawer.rectangle((left, top, right - 1, edge_bottom - 1), fill=outline_color)
            if bottom_open:
                edge_top = max(top, bottom - outline_width)
                drawer.rectangle((left, edge_top, right - 1, bottom - 1), fill=outline_color)
            if left_open:
                edge_right = min(right, left + outline_width)
                drawer.rectangle((left, top, edge_right - 1, bottom - 1), fill=outline_color)
            if right_open:
                edge_left = max(left, right - outline_width)
                drawer.rectangle((edge_left, top, right - 1, bottom - 1), fill=outline_color)

    if start_checker_polygons:
        overlay = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
        drawer = ImageDraw.Draw(overlay, "RGBA")
        if start_checker_polygons:
            for polygon, color in start_checker_polygons:
                if len(polygon) >= 3:
                    drawer.polygon(polygon, fill=color)
        overlay_rgba = np.asarray(overlay, dtype=np.uint8).copy()
        base_alpha = np.asarray(image, dtype=np.uint8)[..., 3]
        overlay_rgba[..., 3] = np.minimum(overlay_rgba[..., 3], base_alpha)
        overlay = Image.fromarray(overlay_rgba, mode="RGBA")
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
        start_checker_polygons = build_start_checker_polygons(geometry)
        road_texture = mask_to_texture(
            road_mask,
            texture_name=f"vroom_track_blocky_v2_{seed}_{width}x{height}_{mask_signature}",
            track_color=track_color,
            pattern_seed=int(seed),
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
