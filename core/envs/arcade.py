"""Lightweight Arcade environment mixin shared by rendered games."""

from __future__ import annotations

import time

from core.runtime import ArcadeFrameClock, ArcadeWindowController


class ArcadeEnvMixin:
    """Small opt-in mixin for common Arcade window, close, and pacing behavior."""

    show_game: bool
    frame_clock: ArcadeFrameClock
    window_controller: ArcadeWindowController
    window: object | None

    def _init_arcade_runtime(
        self,
        *,
        width: int,
        height: int,
        title: str,
        render: bool,
        queue_input_events: bool = False,
        vsync: bool = False,
        render_fps: int | float = 60,
        training_fps: int | float = 0,
        eval_step_delay_seconds: float = 0.0,
    ) -> None:
        self.show_game = bool(render)
        self.render_fps = float(render_fps)
        self.training_fps = float(training_fps)
        self.eval_step_delay_seconds = max(0.0, float(eval_step_delay_seconds))
        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            int(width),
            int(height),
            str(title),
            enabled=self.show_game,
            queue_input_events=bool(queue_input_events),
            vsync=bool(vsync),
        )
        self.window = self.window_controller.window

    def _tick_arcade_frame(self, *, delay_seconds: float | None = None) -> float:
        fps = self.render_fps if bool(getattr(self, "show_game", False)) else self.training_fps
        elapsed = self.frame_clock.tick(fps)
        delay = self.eval_step_delay_seconds if delay_seconds is None else max(0.0, float(delay_seconds))
        if bool(getattr(self, "show_game", False)) and delay > 0.0:
            time.sleep(delay)
        return float(elapsed)

    def close(self) -> None:
        controller = getattr(self, "window_controller", None)
        if controller is not None:
            controller.close()
        self.window = None

