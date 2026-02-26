"""Arcade runtime helpers used by Snake modes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import arcade
from pyglet.window import key as pyglet_key


_LOADED_FONT_PATHS: set[str] = set()


def load_font_once(font_path: str | Path) -> None:
    resolved = str(Path(font_path).resolve())
    if resolved in _LOADED_FONT_PATHS:
        return
    if Path(resolved).exists():
        arcade.load_font(resolved)
        _LOADED_FONT_PATHS.add(resolved)


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
        self._key_state: pyglet_key.KeyStateHandler | None = None
        self._key_presses: list[int] = []
        self._mouse_presses: list[MousePress] = []

        if not self.enabled:
            return

        self.window = arcade.Window(
            self.width,
            self.height,
            title,
            vsync=bool(vsync),
            enable_polling=True,
            visible=bool(visible),
        )
        self._key_state = pyglet_key.KeyStateHandler()
        self.window.push_handlers(self._key_state)
        if self.queue_input_events:
            self.window.push_handlers(self)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self._key_presses.append(symbol)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        self._mouse_presses.append(MousePress(x=x, y=y, button=button, modifiers=modifiers))

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
        if self._key_state is None:
            return False
        return bool(self._key_state[symbol])

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
        self.window.close()
        self.window = None
        self._key_state = None
        self._key_presses = []
        self._mouse_presses = []

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
        self._cache: OrderedDict[tuple, arcade.Text] = OrderedDict()

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

    def get_text(
        self,
        text: str,
        color: tuple[int, int, int] | tuple[int, int, int, int],
        font_size: int | float,
        font_name: str | Iterable[str],
        anchor_x: str = "left",
        anchor_y: str = "baseline",
    ) -> arcade.Text:
        key = (
            str(text),
            self._normalized_color(color),
            int(font_size),
            self._normalized_font_name(font_name),
            str(anchor_x),
            str(anchor_y),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        cached = arcade.Text(
            text=str(text),
            x=0,
            y=0,
            color=self._normalized_color(color),
            font_size=int(font_size),
            font_name=key[3],
            anchor_x=str(anchor_x),
            anchor_y=str(anchor_y),
        )
        self._cache[key] = cached
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return cached

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
