"""Shared helpers for optional translucent observation overlays."""

from __future__ import annotations

import arcade

from core.arcade_style import COLOR_LIGHT_NEUTRAL
from core.runtime import ArcadeWindowController, Rect


DEFAULT_GHOST_ALPHA = 128


def ghost_color(alpha: int = DEFAULT_GHOST_ALPHA) -> tuple[int, int, int, int]:
    """Return the shared light neutral ghost color with bounded alpha."""

    return COLOR_LIGHT_NEUTRAL + (max(0, min(255, int(alpha))),)


def update_ghost_overlay_toggle(
    *,
    window_controller: ArcadeWindowController,
    visible: bool,
    previous_down: bool,
    enabled: bool,
    key: int = arcade.key.X,
) -> tuple[bool, bool]:
    """Edge-trigger an overlay visibility boolean from a held keyboard key."""

    if not bool(enabled):
        return bool(visible), False
    toggle_down = bool(window_controller.is_key_down(int(key)))
    if toggle_down and not bool(previous_down):
        visible = not bool(visible)
    return bool(visible), bool(toggle_down)


def draw_ghost_rect(
    window_controller: ArcadeWindowController,
    rect: Rect,
    *,
    camera_x: float = 0.0,
    color: tuple[int, int, int, int] | None = None,
    fill: bool = True,
    outline: bool = True,
    line_width: float = 1.5,
) -> None:
    """Draw a top-left-space rectangle as a ghost overlay."""

    rgba = ghost_color() if color is None else color
    left = float(rect.left) - float(camera_x)
    bottom = window_controller.top_left_to_bottom(float(rect.top), float(rect.height))
    width = float(rect.width)
    height = float(rect.height)
    if bool(fill):
        arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, rgba)
    if bool(outline):
        arcade.draw_lbwh_rectangle_outline(left, bottom, width, height, rgba, max(0.5, float(line_width)))


def draw_ghost_line(
    window_controller: ArcadeWindowController,
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    camera_x: float = 0.0,
    color: tuple[int, int, int, int] | None = None,
    line_width: float = 1.5,
) -> None:
    """Draw a top-left-space line as a ghost overlay."""

    rgba = ghost_color() if color is None else color
    arcade.draw_line(
        float(start_x) - float(camera_x),
        window_controller.to_arcade_y(float(start_y)),
        float(end_x) - float(camera_x),
        window_controller.to_arcade_y(float(end_y)),
        rgba,
        max(0.5, float(line_width)),
    )
