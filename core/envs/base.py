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
