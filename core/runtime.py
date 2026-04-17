"""Arcade runtime and geometry helpers shared across games."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import arcade
from pyglet.math import Vec2


_LOADED_FONT_PATHS: set[str] = set()


def _env_visible_default(default: bool = True) -> bool:
    raw = os.getenv("RL_TOYBOX_RENDER_VISIBLE")
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_font_once(font_path: str | Path) -> None:
    resolved = str(Path(font_path).resolve())
    if resolved in _LOADED_FONT_PATHS:
        return
    if Path(resolved).exists():
        arcade.load_font(resolved)
        _LOADED_FONT_PATHS.add(resolved)


def _open_arcade_window(
    width: int,
    height: int,
    title: str,
    *,
    vsync: bool,
    visible: bool,
) -> arcade.Window:
    arcade.close_window()
    return arcade.open_window(
        int(width),
        int(height),
        title,
        vsync=bool(vsync),
        enable_polling=True,
        visible=_env_visible_default(bool(visible)),
    )


class ArcadeFrameClock:
    """Simple FPS limiter returning elapsed time in seconds."""

    def __init__(self) -> None:
        self._last = time.perf_counter()

    def tick(self, fps: int | float) -> float:
        now = time.perf_counter()
        elapsed = now - self._last
        fps_value = float(fps)

        if fps_value > 0:
            frame_time = 1.0 / fps_value
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
                now = time.perf_counter()
                elapsed = now - self._last

        self._last = now
        return elapsed


@dataclass(frozen=True)
class MousePress:
    x: float
    y: float
    button: int
    modifiers: int


class ArcadeWindowController:
    """Small wrapper for Arcade window and input polling."""

    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        enabled: bool = True,
        queue_input_events: bool = False,
        vsync: bool = False,
        visible: bool = True,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.enabled = bool(enabled)
        self.queue_input_events = bool(queue_input_events)

        self.window: arcade.Window | None = None
        self._key_presses: list[int] = []
        self._mouse_presses: list[MousePress] = []
        self._mouse_position: tuple[float, float] | None = None

        if not self.enabled:
            return

        self.window = _open_arcade_window(
            self.width,
            self.height,
            title,
            vsync=bool(vsync),
            visible=bool(visible),
        )
        if self.queue_input_events:
            self.window.push_handlers(self)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self.queue_input_events:
            self._key_presses.append(symbol)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        self._mouse_position = (float(x), float(y))
        if self.queue_input_events:
            self._mouse_presses.append(MousePress(x=x, y=y, button=button, modifiers=modifiers))

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int) -> None:
        del button, modifiers
        self._mouse_position = (float(x), float(y))

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float) -> None:
        del dx, dy
        self._mouse_position = (float(x), float(y))

    def on_mouse_drag(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        buttons: int,
        modifiers: int,
    ) -> None:
        del dx, dy, buttons, modifiers
        self._mouse_position = (float(x), float(y))

    def poll_events(self) -> bool:
        if self.window is None:
            return False
        self.window.dispatch_events()
        return bool(self.window.has_exit)

    def poll_events_or_raise(self) -> None:
        if self.poll_events():
            self.close()
            raise SystemExit

    def consume_key_presses(self) -> list[int]:
        key_presses = self._key_presses
        self._key_presses = []
        return key_presses

    def consume_mouse_presses(self) -> list[MousePress]:
        mouse_presses = self._mouse_presses
        self._mouse_presses = []
        return mouse_presses

    def is_key_down(self, symbol: int) -> bool:
        if self.window is None:
            return False
        return bool(self.window.keyboard[symbol])

    def mouse_position(self) -> tuple[float, float] | None:
        return self._mouse_position

    def clear(self, color: tuple[int, int, int] | tuple[int, int, int, int]) -> None:
        if self.window is None:
            return
        self.window.clear(color)

    def flip(self) -> None:
        if self.window is None:
            return
        self.window.flip()

    def close(self) -> None:
        if self.window is None:
            return
        try:
            active_window = arcade.get_window()
        except RuntimeError:
            active_window = None
        if active_window is self.window:
            arcade.close_window()
        else:
            self.window.close()
        self.window = None
        self._key_presses = []
        self._mouse_presses = []
        self._mouse_position = None

    def to_arcade_y(self, y_top: float) -> float:
        return float(self.height) - float(y_top)

    def to_top_left_y(self, y_arcade: float) -> float:
        return float(self.height) - float(y_arcade)

    def top_left_to_bottom(self, top_y: float, object_height: float) -> float:
        return self.to_arcade_y(float(top_y) + float(object_height))


class TextCache:
    """Reusable cache of `arcade.Text` objects."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max(1, int(max_entries))
        self._cached_text = lru_cache(maxsize=self.max_entries)(self._build_text)

    @staticmethod
    def _normalized_color(
        color: tuple[int, int, int] | tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if len(color) == 4:
            return int(color[0]), int(color[1]), int(color[2]), int(color[3])
        return int(color[0]), int(color[1]), int(color[2]), 255

    @staticmethod
    def _normalized_font_name(font_name: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(font_name, str):
            return (font_name,)
        return tuple(str(name) for name in font_name)

    @staticmethod
    def _build_text(
        text: str,
        color: tuple[int, int, int, int],
        font_size: int,
        font_name: tuple[str, ...],
        anchor_x: str,
        anchor_y: str,
    ) -> arcade.Text:
        return arcade.Text(
            text=text,
            x=0,
            y=0,
            color=color,
            font_size=font_size,
            font_name=font_name,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )

    def get_text(
        self,
        text: str,
        color: tuple[int, int, int] | tuple[int, int, int, int],
        font_size: int | float,
        font_name: str | Iterable[str],
        anchor_x: str = "left",
        anchor_y: str = "baseline",
    ) -> arcade.Text:
        return self._cached_text(
            str(text),
            self._normalized_color(color),
            int(font_size),
            self._normalized_font_name(font_name),
            str(anchor_x),
            str(anchor_y),
        )

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        color: tuple[int, int, int] | tuple[int, int, int, int],
        font_size: int | float,
        font_name: str | Iterable[str],
        anchor_x: str = "left",
        anchor_y: str = "baseline",
    ) -> None:
        text_obj = self.get_text(
            text=text,
            color=color,
            font_size=font_size,
            font_name=font_name,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        text_obj.x = float(x)
        text_obj.y = float(y)
        text_obj.draw()


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in top-left coordinate space."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def colliderect(self, other: "Rect") -> bool:
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.bottom <= other.top
            or self.top >= other.bottom
        )


def heading_to_vector(angle_degrees: float) -> Vec2:
    radians = math.radians(angle_degrees)
    return Vec2(math.cos(radians), math.sin(radians))


def rotate_degrees(vector: Vec2, angle_degrees: float) -> Vec2:
    return vector.rotate(math.radians(angle_degrees))


def length_squared(vector: Vec2) -> float:
    return vector.dot(vector)


def rect_from_center(position: Vec2, size: int | float) -> Rect:
    half = float(size) / 2.0
    return Rect(position.x - half, position.y - half, float(size), float(size))


def normalize_angle_degrees(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def _obstacle_xy(obstacle: Any) -> tuple[float, float]:
    if hasattr(obstacle, "x") and hasattr(obstacle, "y"):
        return float(obstacle.x), float(obstacle.y)
    if isinstance(obstacle, (tuple, list)) and len(obstacle) >= 2:
        return float(obstacle[0]), float(obstacle[1])
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")


class ArcadeSquareObstacleField:
    """Thin Arcade-backed adapter for square obstacle spatial queries."""

    def __init__(self, tile_size: int) -> None:
        self.tile_size = max(1, int(tile_size))
        self._sprites = arcade.SpriteList(
            use_spatial_hash=True,
            spatial_hash_cell_size=self.tile_size,
            visible=False,
        )

    @staticmethod
    def _sprite_rect(sprite: arcade.BasicSprite) -> Rect:
        return Rect(
            float(sprite.left),
            float(sprite.bottom),
            float(sprite.width),
            float(sprite.height),
        )

    @staticmethod
    def _overlaps_rect(rect: Rect, sprite: arcade.BasicSprite) -> bool:
        return rect.colliderect(ArcadeSquareObstacleField._sprite_rect(sprite))

    def _query_rect(self, rect: Rect) -> arcade.Rect:
        return arcade.LRBT(
            float(rect.left),
            float(rect.right),
            float(rect.top),
            float(rect.bottom),
        )

    def rebuild(self, obstacles: Iterable[Any]) -> None:
        self._sprites = arcade.SpriteList(
            use_spatial_hash=True,
            spatial_hash_cell_size=self.tile_size,
            visible=False,
        )
        half = float(self.tile_size) * 0.5
        for obstacle in obstacles:
            obstacle_x, obstacle_y = _obstacle_xy(obstacle)
            self._sprites.append(
                arcade.SpriteSolidColor(
                    self.tile_size,
                    self.tile_size,
                    center_x=float(obstacle_x) + half,
                    center_y=float(obstacle_y) + half,
                    color=(255, 255, 255, 0),
                )
            )

    def collides_with_rect(self, rect: Rect) -> bool:
        return any(
            self._overlaps_rect(rect, sprite)
            for sprite in arcade.get_sprites_in_rect(self._query_rect(rect), self._sprites)
        )

    def contains_point(self, x: float, y: float) -> bool:
        point_x = float(x)
        point_y = float(y)
        for sprite in arcade.get_sprites_at_point((point_x, point_y), self._sprites):
            if (
                float(sprite.left) <= point_x < float(sprite.right)
                and float(sprite.bottom) <= point_y < float(sprite.top)
            ):
                return True
        return False

    def segment_intersects(self, point_a: Vec2, point_b: Vec2) -> bool:
        bounds = Rect(
            left=min(float(point_a.x), float(point_b.x)),
            top=min(float(point_a.y), float(point_b.y)),
            width=max(1.0, abs(float(point_a.x) - float(point_b.x))),
            height=max(1.0, abs(float(point_a.y) - float(point_b.y))),
        )
        for sprite in arcade.get_sprites_in_rect(self._query_rect(bounds), self._sprites):
            if _line_intersects_rect(point_a, point_b, self._sprite_rect(sprite)):
                return True
        return False


def _line_intersects_rect(point_a: Vec2, point_b: Vec2, rect: Rect) -> bool:
    x0, y0 = point_a.x, point_a.y
    x1, y1 = point_b.x, point_b.y
    dx = x1 - x0
    dy = y1 - y0

    p = (-dx, dx, -dy, dy)
    q = (x0 - rect.left, rect.right - x0, y0 - rect.top, rect.bottom - y0)

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
            continue

        t = qi / pi
        if pi < 0:
            if t > u2:
                return False
            u1 = max(u1, t)
        else:
            if t < u1:
                return False
            u2 = min(u2, t)

    return True
