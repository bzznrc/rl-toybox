"""Minimal environment interface for toy RL environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Env(ABC):
    """Common environment interface for runners and scripts."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation vector."""

    @abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Step environment with an action and return transition tuple."""

    def render(self) -> None:
        """Render one frame. Envs that auto-render can keep this as a no-op."""

    def draw_frame(self) -> None:
        """Render one composed frame for tooling such as capture/export."""
        self.render()

    def capture_render_fps(self) -> float:
        """Return the rendered play-mode FPS used by capture/export tooling."""
        from core.shared_config import FPS

        value = getattr(self, "render_fps", None)
        if value is None:
            game = getattr(self, "game", None)
            value = getattr(game, "render_fps", None)
        if value is None:
            value = FPS
        return max(1.0, float(value))

    def capture_action_repeat_frames(self) -> int:
        """Return how many rendered frames one policy action spans in play mode."""
        value = getattr(self, "rl_action_repeat_frames", None)
        if value is None:
            game = getattr(self, "game", None)
            value = getattr(game, "rl_action_repeat_frames", None)
        return max(1, int(1 if value is None else value))

    def capture_step_seconds(self) -> float:
        """Return how much play-mode time one environment step represents."""
        delay_seconds = getattr(self, "eval_step_delay_seconds", None)
        if delay_seconds is None:
            game = getattr(self, "game", None)
            delay_seconds = getattr(game, "eval_step_delay_seconds", 0.0)
        return (
            float(self.capture_action_repeat_frames()) / float(self.capture_render_fps())
            + max(0.0, float(delay_seconds))
        )

    def get_window_controller(self):
        """Return the active window controller when the env is rendered."""
        controller = getattr(self, "window_controller", None)
        if controller is not None:
            return controller
        game = getattr(self, "game", None)
        return getattr(game, "window_controller", None)

    def get_render_window(self):
        """Return the active Arcade window when the env is rendered."""
        window = getattr(self, "window", None)
        if window is not None:
            return window
        controller = self.get_window_controller()
        if controller is not None:
            return getattr(controller, "window", None)
        game = getattr(self, "game", None)
        return getattr(game, "window", None)

    def close(self) -> None:
        """Release any external resources."""
