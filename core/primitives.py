"""Shared gameplay and rendering primitives used by multiple games."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Sequence, TypeVar

import arcade

from core.arcade_style import COLOR_DARK_NEUTRAL, COLOR_FOG_GRAY, COLOR_SLATE_GRAY
from core.arcade_style import DEFAULT_STATUS_BAR_FONT_SIZE, INTER_FONT_NAME
from core.runtime import ArcadeWindowController, TextCache

T = TypeVar("T")
StatusTextEntry = tuple[str, object]
STATUS_ICON_GAP = 6.0
STATUS_CLOCK_BASE_COLOR = COLOR_SLATE_GRAY
STATUS_CLOCK_FILL_COLOR = COLOR_FOG_GRAY
STATUS_CLOCK_OUTLINE_COLOR = COLOR_FOG_GRAY
ROAD_DASH_ALPHA = 255
ROAD_DASH_COLOR = (*COLOR_FOG_GRAY, int(ROAD_DASH_ALPHA))
ROAD_DASH_LENGTH_RATIO = 0.40
ROAD_DASH_THICKNESS_RATIO = 0.05
ROAD_DASH_GAP_RATIO = 0.40


def resolve_circle_collisions(
    positions: list[tuple[float, float]],
    velocities: list[tuple[float, float]],
    radii: list[float],
    *,
    sep_strength: float,
    overlap_cap: float,
    contact_damp: float,
    eps: float = 1e-6,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[bool]]:
    """Resolve deterministic pairwise circle overlaps and apply mild contact damping."""

    count = len(positions)
    if len(velocities) != count or len(radii) != count:
        raise ValueError("positions, velocities, and radii must have the same length")

    pos = [(float(x), float(y)) for x, y in positions]
    vel = [(float(vx), float(vy)) for vx, vy in velocities]
    contact_flags = [False] * count

    strength = max(0.0, float(sep_strength))
    cap = max(0.0, float(overlap_cap))

    for i in range(count):
        for j in range(i + 1, count):
            pix, piy = pos[i]
            pjx, pjy = pos[j]
            dx = pix - pjx
            dy = piy - pjy
            dist = math.hypot(dx, dy)
            overlap = (float(radii[i]) + float(radii[j])) - dist
            if overlap <= 0.0:
                continue

            contact_flags[i] = True
            contact_flags[j] = True

            if dist > eps:
                nx = dx / dist
                ny = dy / dist
            else:
                nx = 1.0
                ny = 0.0

            correction = min(overlap, cap) * strength
            shift = 0.5 * correction
            pos[i] = (pix + shift * nx, piy + shift * ny)
            pos[j] = (pjx - shift * nx, pjy - shift * ny)

    damp_scale = 1.0 - max(0.0, min(1.0, float(contact_damp)))
    for idx in range(count):
        if not contact_flags[idx]:
            continue
        vx, vy = vel[idx]
        vel[idx] = (vx * damp_scale, vy * damp_scale)

    return pos, vel, contact_flags


def _grow_connected_random_walk_shape(
    start: T,
    min_sections: int,
    max_sections: int,
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
) -> list[T]:
    target_sections = random.randint(int(min_sections), int(max_sections))
    shape = [start]
    current = start

    for _ in range(target_sections - 1):
        candidates = list(neighbor_candidates_fn(current))
        random.shuffle(candidates)
        for candidate in candidates:
            if is_candidate_valid_fn(candidate, shape):
                shape.append(candidate)
                current = candidate
                break
        else:
            break
    return shape


def spawn_connected_random_walk_shapes(
    shape_count: int,
    min_sections: int,
    max_sections: int,
    sample_start_fn: Callable[[], T | None],
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
) -> list[list[T]]:
    shapes: list[list[T]] = []
    for _ in range(int(shape_count)):
        start = sample_start_fn()
        if start is None:
            continue
        shape = _grow_connected_random_walk_shape(
            start=start,
            min_sections=min_sections,
            max_sections=max_sections,
            neighbor_candidates_fn=neighbor_candidates_fn,
            is_candidate_valid_fn=is_candidate_valid_fn,
        )
        if shape:
            shapes.append(shape)
    return shapes


def draw_two_tone_tile(
    window_controller: ArcadeWindowController,
    *,
    top_left_x: float,
    top_left_y: float,
    size: float,
    outer_color: tuple[int, int, int] | tuple[int, int, int, int],
    inner_color: tuple[int, int, int] | tuple[int, int, int, int],
    inset: float,
    glow_color: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
    glow_radius: float = 0.0,
    glow_alpha: int = 92,
    glow_layers: int = 4,
    glow_inside_bounds: bool = False,
    glow_source_inset: float = 0.0,
    draw_outer_fill: bool = True,
    draw_inner_fill: bool = True,
) -> None:
    bottom = window_controller.top_left_to_bottom(top_left_y, size)
    if bool(draw_outer_fill):
        arcade.draw_lbwh_rectangle_filled(top_left_x, bottom, size, size, outer_color)
    if glow_color is not None and float(glow_radius) > 0.0:
        draw_soft_glow_rect(
            left=float(top_left_x),
            bottom=float(bottom),
            width=float(size),
            height=float(size),
            color=glow_color,
            glow_radius=float(glow_radius),
            max_alpha=int(glow_alpha),
            layers=int(glow_layers),
            inside_bounds=bool(glow_inside_bounds),
            source_inset=float(glow_source_inset),
        )
    if not bool(draw_inner_fill):
        return
    inner_size = size - 2.0 * float(inset)
    if inner_size <= 0:
        return
    arcade.draw_lbwh_rectangle_filled(
        top_left_x + inset,
        bottom + inset,
        inner_size,
        inner_size,
        inner_color,
    )


def draw_two_tone_square_block(
    window_controller: ArcadeWindowController,
    *,
    top_left_x: float,
    top_left_y: float,
    tile_size: float,
    tiles_per_side: int,
    outer_color: tuple[int, int, int] | tuple[int, int, int, int],
    inner_color: tuple[int, int, int] | tuple[int, int, int, int],
    inset: float,
) -> None:
    side_tiles = max(1, int(tiles_per_side))
    draw_two_tone_tile(
        window_controller,
        top_left_x=float(top_left_x),
        top_left_y=float(top_left_y),
        size=float(tile_size) * float(side_tiles),
        outer_color=outer_color,
        inner_color=inner_color,
        inset=float(inset),
    )


def _color_rgba(
    color: tuple[int, int, int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if len(color) == 4:
        return int(color[0]), int(color[1]), int(color[2]), int(color[3])
    return int(color[0]), int(color[1]), int(color[2]), 255


def with_alpha(
    color: tuple[int, int, int] | tuple[int, int, int, int],
    alpha: int | float,
) -> tuple[int, int, int, int]:
    red, green, blue, _ = _color_rgba(color)
    return red, green, blue, int(max(0, min(255, round(float(alpha)))))


def draw_soft_glow_rect(
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    glow_radius: float,
    max_alpha: int = 92,
    layers: int = 4,
    inside_bounds: bool = False,
    source_inset: float = 0.0,
) -> None:
    source_pad = max(0.0, float(source_inset))
    rect_left = float(left) + source_pad
    rect_bottom = float(bottom) + source_pad
    rect_width = max(0.0, float(width) - 2.0 * source_pad)
    rect_height = max(0.0, float(height) - 2.0 * source_pad)
    radius = max(0.0, float(glow_radius))
    layer_count = max(0, int(layers))
    if rect_width <= 0.0 or rect_height <= 0.0 or radius <= 0.0 or layer_count <= 0:
        return

    peak_alpha = max(0, min(255, int(max_alpha)))
    for layer in range(layer_count, 0, -1):
        strength = float(layer) / float(layer_count)
        layer_alpha = peak_alpha * (strength * strength)
        if bool(inside_bounds):
            inset = radius * (1.0 - strength)
            layer_width = rect_width - 2.0 * inset
            layer_height = rect_height - 2.0 * inset
            if layer_width <= 0.0 or layer_height <= 0.0:
                continue
            arcade.draw_lbwh_rectangle_filled(
                rect_left + inset,
                rect_bottom + inset,
                layer_width,
                layer_height,
                with_alpha(color, layer_alpha),
            )
            continue

        expand = radius * strength
        arcade.draw_lbwh_rectangle_filled(
            rect_left - expand,
            rect_bottom - expand,
            rect_width + 2.0 * expand,
            rect_height + 2.0 * expand,
            with_alpha(color, layer_alpha),
        )


def draw_cell_union_outline(
    window_controller: ArcadeWindowController,
    *,
    cells: Iterable[tuple[int, int]],
    top_left_x: float,
    top_left_y: float,
    cell_size: float,
    border_width: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
) -> None:
    cell_set = {(int(col), int(row)) for col, row in cells}
    if not cell_set:
        return

    size = float(cell_size)
    border = max(1.0, float(border_width))

    for col, row in cell_set:
        left = float(top_left_x) + float(col) * size
        top = float(top_left_y) + float(row) * size
        top_open = (int(col), int(row) - 1) not in cell_set
        bottom_open = (int(col), int(row) + 1) not in cell_set
        left_open = (int(col) - 1, int(row)) not in cell_set
        right_open = (int(col) + 1, int(row)) not in cell_set

        if top_open:
            arcade.draw_lbwh_rectangle_filled(
                left,
                float(window_controller.to_arcade_y(top + border)),
                size,
                border,
                color,
            )
        if bottom_open:
            arcade.draw_lbwh_rectangle_filled(
                left,
                float(window_controller.to_arcade_y(top + size)),
                size,
                border,
                color,
            )
        if left_open:
            arcade.draw_lbwh_rectangle_filled(
                left,
                float(window_controller.to_arcade_y(top + size)),
                border,
                size,
                color,
            )
        if right_open:
            arcade.draw_lbwh_rectangle_filled(
                left + size - border,
                float(window_controller.to_arcade_y(top + size)),
                border,
                size,
                color,
            )


def draw_staggered_square_pattern(
    window_controller: ArcadeWindowController,
    *,
    top_left_x: float,
    top_left_y: float,
    width: float,
    height: float,
    square_size: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    gap_x: float = 0.0,
    gap_y: float | None = None,
    row_offset_ratio: float = 0.5,
) -> None:
    """Draw a staggered tiny-square fill pattern in top-left coordinate space."""

    pattern_left = float(top_left_x)
    pattern_top = float(top_left_y)
    pattern_right = pattern_left + max(0.0, float(width))
    pattern_bottom = pattern_top + max(0.0, float(height))
    if pattern_right <= pattern_left or pattern_bottom <= pattern_top:
        return

    tile_size = max(1.0, float(square_size))
    spacing_x = max(0.0, float(gap_x))
    spacing_y = spacing_x if gap_y is None else max(0.0, float(gap_y))
    step_x = tile_size + spacing_x
    step_y = tile_size + spacing_y
    if step_x <= 0.0 or step_y <= 0.0:
        return

    row_offset = step_x * float(row_offset_ratio)
    row_idx = 0
    y = pattern_top
    while y < pattern_bottom:
        x = pattern_left - (row_offset if (row_idx % 2) else 0.0)
        while x < pattern_right:
            if (x + tile_size) > pattern_left:
                bottom = window_controller.top_left_to_bottom(y, tile_size)
                arcade.draw_lbwh_rectangle_filled(x, bottom, tile_size, tile_size, color)
            x += step_x
        y += step_y
        row_idx += 1


def build_staggered_square_pattern_texture(
    *,
    width: int,
    height: int,
    square_size: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    gap_x: float = 0.0,
    gap_y: float | None = None,
    row_offset_ratio: float = 0.5,
    texture_name: str | None = None,
) -> arcade.Texture:
    """Build a reusable transparent texture with a staggered tiny-square pattern."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Staggered square pattern textures require numpy and Pillow.") from exc

    tex_w = max(1, int(width))
    tex_h = max(1, int(height))
    tile_size = max(1, int(round(float(square_size))))
    spacing_x = max(0, int(round(float(gap_x))))
    spacing_y = spacing_x if gap_y is None else max(0, int(round(float(gap_y))))
    step_x = max(1, tile_size + spacing_x)
    step_y = max(1, tile_size + spacing_y)
    row_offset = int(round(float(step_x) * float(row_offset_ratio)))

    color_rgba = (
        int(color[0]),
        int(color[1]),
        int(color[2]),
        int(color[3]) if len(color) == 4 else 255,
    )

    rgba = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
    row_idx = 0
    y0 = 0
    while y0 < tex_h:
        y1 = min(tex_h, y0 + tile_size)
        x = -row_offset if (row_idx % 2) else 0
        while x < tex_w:
            x0 = max(0, x)
            x1 = min(tex_w, x + tile_size)
            if x1 > x0 and y1 > y0:
                rgba[y0:y1, x0:x1] = color_rgba
            x += step_x
        y0 += step_y
        row_idx += 1

    texture_hash = (
        str(texture_name)
        if texture_name
        else (
            f"staggered_squares_{tex_w}x{tex_h}_s{tile_size}_gx{spacing_x}_gy{spacing_y}_"
            f"o{row_offset}_{color_rgba[0]}_{color_rgba[1]}_{color_rgba[2]}_{color_rgba[3]}"
        )
    )
    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    return arcade.Texture(image=image, hash=texture_hash)


def draw_control_marker(
    window_controller: ArcadeWindowController,
    *,
    center_x: float,
    center_y_top_left: float,
    marker_size: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
) -> None:
    arcade.draw_lbwh_rectangle_filled(
        center_x - marker_size / 2.0,
        window_controller.to_arcade_y(center_y_top_left) - marker_size / 2.0,
        marker_size,
        marker_size,
        color,
    )


def draw_facing_indicator(
    window_controller: ArcadeWindowController,
    *,
    center_x: float,
    center_y_top_left: float,
    angle_degrees: float,
    length: float,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    line_width: float = 2.0,
) -> None:
    radians = math.radians(float(angle_degrees))
    end_x = center_x + math.cos(radians) * float(length)
    end_y = center_y_top_left + math.sin(radians) * float(length)
    arcade.draw_line(
        center_x,
        window_controller.to_arcade_y(center_y_top_left),
        end_x,
        window_controller.to_arcade_y(end_y),
        color,
        float(line_width),
    )


def road_dash_length(base_unit: float) -> float:
    return max(2.0, float(base_unit) * float(ROAD_DASH_LENGTH_RATIO))


def road_dash_thickness(cross_size: float) -> float:
    return max(2.0, float(cross_size) * float(ROAD_DASH_THICKNESS_RATIO))


def road_dash_gap(base_unit: float) -> float:
    return max(0.0, float(base_unit) * float(ROAD_DASH_GAP_RATIO))


def build_dashed_path_top_left_polygons(
    *,
    path_length: float,
    sample_fn: Callable[[float], tuple[tuple[float, float], tuple[float, float]]],
    dash_length: float,
    dash_width: float,
    gap_length: float,
    curve_step: float | None = None,
) -> list[list[tuple[float, float]]]:
    """Build top-left-space dash polygons for a path.

    Each dash is sampled across its length so curved paths produce curved ribbon
    polygons instead of a single straight rectangle.
    """

    total_length = max(0.0, float(path_length))
    segment_length = max(1.0, float(dash_length))
    segment_width = max(1.0, float(dash_width))
    gap = max(0.0, float(gap_length))
    if total_length <= 1e-6:
        return []

    step = segment_length + gap
    if step <= 1e-6:
        return []

    sample_spacing = float(curve_step) if curve_step is not None else max(4.0, min(10.0, 0.24 * segment_length))
    sample_spacing = max(1.0, sample_spacing)
    half_width = 0.5 * float(segment_width)
    polygons: list[list[tuple[float, float]]] = []

    dash_start = 0.0
    while dash_start < total_length - 1e-6:
        current_length = min(segment_length, total_length - dash_start)
        sample_count = max(2, int(math.ceil(float(current_length) / float(sample_spacing))) + 1)
        left_pts: list[tuple[float, float]] = []
        right_pts: list[tuple[float, float]] = []
        for sample_idx in range(sample_count):
            alpha = float(sample_idx) / float(max(1, sample_count - 1))
            sample_s = float(dash_start) + float(current_length) * alpha
            (center_x, center_y), (tan_x, tan_y) = sample_fn(float(sample_s))
            tan_mag = math.hypot(float(tan_x), float(tan_y))
            if tan_mag <= 1e-9:
                continue

            tx = float(tan_x) / tan_mag
            ty = float(tan_y) / tan_mag
            nx = -ty
            ny = tx
            left_pts.append((float(center_x) - nx * half_width, float(center_y) - ny * half_width))
            right_pts.append((float(center_x) + nx * half_width, float(center_y) + ny * half_width))

        if len(left_pts) >= 2 and len(right_pts) >= 2:
            polygons.append([*left_pts, *reversed(right_pts)])
        dash_start += step

    return polygons


def draw_top_left_polygons(
    window_controller: ArcadeWindowController,
    *,
    polygons: Sequence[Sequence[tuple[float, float]]],
    color: tuple[int, int, int] | tuple[int, int, int, int] = ROAD_DASH_COLOR,
) -> None:
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        arcade.draw_polygon_filled(
            [(float(px), float(window_controller.to_arcade_y(float(py)))) for px, py in polygon],
            color,
        )


def draw_dashed_path_top_left(
    window_controller: ArcadeWindowController,
    *,
    path_length: float,
    sample_fn: Callable[[float], tuple[tuple[float, float], tuple[float, float]]],
    dash_length: float,
    dash_width: float,
    gap_length: float,
    curve_step: float | None = None,
    color: tuple[int, int, int] | tuple[int, int, int, int] = ROAD_DASH_COLOR,
) -> None:
    """Draw repeated dashes along a path in top-left coordinate space."""

    draw_top_left_polygons(
        window_controller,
        polygons=build_dashed_path_top_left_polygons(
            path_length=float(path_length),
            sample_fn=sample_fn,
            dash_length=float(dash_length),
            dash_width=float(dash_width),
            gap_length=float(gap_length),
            curve_step=curve_step,
        ),
        color=color,
    )


def status_icon_size(bottom_bar_height: float, tile_size: float) -> float:
    return max(12.0, min(float(bottom_bar_height - 8.0), float(tile_size)))


def status_icon_inset(cell_inset: float) -> float:
    return max(1.0, round(float(cell_inset)))


@dataclass(frozen=True)
class StatusBarLayout:
    center_y: float
    left_panel_left: float
    left_panel_right: float
    score_left: float
    score_right: float
    clock_center_x: float | None
    clock_radius: float
    clock_border_width: float


def status_bar_layout(
    *,
    width: float,
    bottom_bar_height: float,
    tile_size: float,
    cell_inset: float,
    left_padding: float = 8.0,
    right_padding: float = 10.0,
    center_gap: float = 14.0,
    left_panel_width: float = 0.0,
    include_clock: bool = True,
) -> StatusBarLayout:
    """Compute left/center/right regions for a bottom status bar.

    The layout reserves:
    - Left region (`left_panel_left`..`left_panel_right`) for optional text/logs.
    - Center region (`score_left`..`score_right`) for score/history icons.
    - Right region (clock center/radius) for a time indicator, if enabled.
    """

    bar_width = max(1.0, float(width))
    bar_height = max(1.0, float(bottom_bar_height))
    left_pad = max(0.0, float(left_padding))
    right_pad = max(0.0, float(right_padding))
    mid_gap = max(0.0, float(center_gap))
    panel_width = max(0.0, float(left_panel_width))

    icon_size = status_icon_size(bar_height, float(tile_size))
    indicator_diameter = icon_size * math.sqrt(2.0) * 0.8
    indicator_radius = indicator_diameter * 0.5
    indicator_border = max(1.0, round(float(cell_inset) * 0.5))

    left_panel_left = left_pad
    left_panel_right = left_panel_left + panel_width
    score_left = max(left_pad, left_panel_right + (mid_gap if panel_width > 0.0 else 0.0))

    if bool(include_clock):
        clock_center_x = bar_width - right_pad - indicator_radius
        score_right = max(score_left, clock_center_x - indicator_radius - mid_gap)
    else:
        clock_center_x = None
        score_right = max(score_left, bar_width - right_pad)

    return StatusBarLayout(
        center_y=bar_height * 0.5,
        left_panel_left=left_panel_left,
        left_panel_right=left_panel_right,
        score_left=score_left,
        score_right=score_right,
        clock_center_x=clock_center_x,
        clock_radius=indicator_radius if bool(include_clock) else 0.0,
        clock_border_width=indicator_border,
    )


def draw_status_bar(
    *,
    width: float,
    bottom_bar_height: float,
    tile_size: float,
    cell_inset: float,
    background_color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_DARK_NEUTRAL,
    left_padding: float = 8.0,
    right_padding: float = 10.0,
    center_gap: float = 14.0,
    left_panel_width: float = 0.0,
    include_clock: bool = True,
    text_cache: TextCache | None = None,
    left_text_entries: Sequence[StatusTextEntry] | None = None,
    text_color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_FOG_GRAY,
    text_inset_x: float = 0.0,
) -> StatusBarLayout:
    arcade.draw_lbwh_rectangle_filled(
        0,
        0,
        float(width),
        float(bottom_bar_height),
        background_color,
    )
    layout = status_bar_layout(
        width=float(width),
        bottom_bar_height=float(bottom_bar_height),
        tile_size=float(tile_size),
        cell_inset=float(cell_inset),
        left_padding=float(left_padding),
        right_padding=float(right_padding),
        center_gap=float(center_gap),
        left_panel_width=float(left_panel_width),
        include_clock=bool(include_clock),
    )
    if text_cache is not None and left_text_entries is not None:
        draw_status_left_text(
            text_cache,
            layout=layout,
            entries=left_text_entries,
            color=text_color,
            inset_x=float(text_inset_x),
        )
    return layout


def draw_status_icon_row(
    *,
    left: float,
    right: float,
    center_y: float,
    icon_size: float,
    items: Sequence[T],
    draw_item: Callable[[T, float, float, float], None],
    gap: float = STATUS_ICON_GAP,
) -> None:
    available_width = max(0.0, float(right) - float(left))
    size = max(0.0, float(icon_size))
    icon_gap = max(0.0, float(gap))
    if available_width <= 0.0 or size <= 0.0:
        return

    item_list = list(items)
    if not item_list:
        return

    max_icons = int((available_width + icon_gap) // (size + icon_gap))
    if max_icons <= 0:
        return

    visible_items = item_list[-max_icons:]
    total_width = len(visible_items) * size + max(0, len(visible_items) - 1) * icon_gap
    start_x = float(left) + (available_width - total_width) * 0.5
    for idx, item in enumerate(visible_items):
        center_x = start_x + size * 0.5 + idx * (size + icon_gap)
        draw_item(item, float(center_x), float(center_y), float(size))


def draw_status_clock(
    *,
    layout: StatusBarLayout,
    remaining_ratio: float,
    num_segments: int = 96,
) -> None:
    if layout.clock_center_x is None or float(layout.clock_radius) <= 0.0:
        return
    draw_time_pie_indicator(
        center_x=float(layout.clock_center_x),
        center_y=float(layout.center_y),
        radius=float(layout.clock_radius),
        border_width=float(layout.clock_border_width),
        remaining_ratio=float(remaining_ratio),
        base_color=STATUS_CLOCK_BASE_COLOR,
        fill_color=STATUS_CLOCK_FILL_COLOR,
        outline_color=STATUS_CLOCK_OUTLINE_COLOR,
        num_segments=int(num_segments),
    )


def format_status_text_entries(entries: Sequence[StatusTextEntry]) -> str:
    parts: list[str] = []
    for key, value in entries:
        parts.append(f"{str(key)}: {str(value)}")
    return "\t".join(parts)


def draw_status_left_text(
    text_cache: TextCache,
    *,
    layout: StatusBarLayout,
    entries: Sequence[StatusTextEntry],
    color: tuple[int, int, int] | tuple[int, int, int, int],
    font_size: int | float = DEFAULT_STATUS_BAR_FONT_SIZE,
    font_name: str | Iterable[str] = INTER_FONT_NAME,
    inset_x: float = 0.0,
    tab_gap_px: float | None = None,
) -> None:
    if float(layout.left_panel_right) <= float(layout.left_panel_left):
        return
    x = float(layout.left_panel_left) + float(inset_x)
    y = float(layout.center_y) - 1.0
    right = float(layout.left_panel_right)
    gap = max(12.0, float(tab_gap_px) if tab_gap_px is not None else float(font_size) * 2.5)

    for key, value in entries:
        segment = f"{str(key)}: {str(value)}"
        text_obj = text_cache.get_text(
            text=segment,
            color=color,
            font_size=float(font_size),
            font_name=font_name,
            anchor_x="left",
            anchor_y="center",
        )
        segment_width = float(getattr(text_obj, "content_width", 0.0))
        if x + segment_width > right and x > float(layout.left_panel_left) + float(inset_x):
            break
        text_obj.x = float(x)
        text_obj.y = float(y)
        text_obj.draw()
        x += segment_width + gap
        if x >= right:
            break


def draw_status_square_icon(
    *,
    center_x: float,
    center_y: float,
    size: float,
    outer_color: tuple[int, int, int] | tuple[int, int, int, int],
    inner_color: tuple[int, int, int] | tuple[int, int, int, int],
    inset: float,
    packed: bool = False,
    packed_marker_color: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
    packed_marker_size: float | None = None,
) -> None:
    bottom = center_y - size / 2.0
    left = center_x - size / 2.0
    arcade.draw_lbwh_rectangle_filled(left, bottom, size, size, outer_color)
    inner_size = max(1.0, size - 2.0 * float(inset))
    arcade.draw_lbwh_rectangle_filled(
        left + inset,
        bottom + inset,
        inner_size,
        inner_size,
        inner_color,
    )
    if packed:
        marker_color = outer_color if packed_marker_color is None else packed_marker_color
        marker_size = max(2.0, float(packed_marker_size) if packed_marker_size is not None else round(float(size) * 0.3))
        arcade.draw_lbwh_rectangle_filled(
            center_x - marker_size / 2.0,
            center_y - marker_size / 2.0,
            marker_size,
            marker_size,
            marker_color,
        )


def draw_time_pie_indicator(
    *,
    center_x: float,
    center_y: float,
    radius: float,
    border_width: float,
    remaining_ratio: float,
    base_color: tuple[int, int, int] | tuple[int, int, int, int],
    fill_color: tuple[int, int, int] | tuple[int, int, int, int],
    outline_color: tuple[int, int, int] | tuple[int, int, int, int],
    num_segments: int = 96,
) -> None:
    circle_segments = max(16, int(num_segments))
    ratio = max(0.0, min(1.0, float(remaining_ratio)))
    arcade.draw_circle_filled(center_x, center_y, radius, base_color, num_segments=circle_segments)
    inner_radius = max(1.0, radius - border_width)

    if ratio <= 0.0:
        arcade.draw_circle_outline(
            center_x,
            center_y,
            radius,
            outline_color,
            border_width,
            num_segments=circle_segments,
        )
        return

    if ratio >= 1.0:
        arcade.draw_circle_filled(
            center_x,
            center_y,
            inner_radius,
            fill_color,
            num_segments=circle_segments,
        )
        arcade.draw_circle_outline(
            center_x,
            center_y,
            radius,
            outline_color,
            border_width,
            num_segments=circle_segments,
        )
        return

    start_angle = 90.0
    end_angle = start_angle + 360.0 * ratio
    arcade.draw_arc_filled(
        center_x=center_x,
        center_y=center_y,
        width=inner_radius * 2.0,
        height=inner_radius * 2.0,
        color=fill_color,
        start_angle=start_angle,
        end_angle=end_angle,
        num_segments=circle_segments,
    )
    arcade.draw_circle_outline(
        center_x,
        center_y,
        radius,
        outline_color,
        border_width,
        num_segments=circle_segments,
    )
