"""Stomp environment stub for future continuous-control SAC."""

from __future__ import annotations

import numpy as np

from core.envs.base import Env


class StompEnv(Env):
    """Stub continuous-control environment.

    TODO: replace with articulated walker dynamics and contact simulation.
    """

    def __init__(self, mode: str = "train", render: bool = False) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.max_steps = 500
        self.reset()

    def _obs(self) -> np.ndarray:
        return np.asarray(
            [
                self.pos_x,
                self.pos_y,
                self.vel_x,
                self.vel_y,
                self.target_x - self.pos_x,
                self.target_y - self.pos_y,
            ],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.target_x = 1.0
        self.target_y = 0.0
        self.steps = 0
        self.done = False
        return self._obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._obs(), 0.0, True, {"win": False}

        control = np.asarray(action, dtype=np.float32)
        if control.shape != (2,):
            control = np.zeros((2,), dtype=np.float32)
        control = np.clip(control, -1.0, 1.0)

        self.vel_x = 0.9 * self.vel_x + 0.1 * float(control[0])
        self.vel_y = 0.9 * self.vel_y + 0.1 * float(control[1])
        self.pos_x += self.vel_x * 0.05
        self.pos_y += self.vel_y * 0.05

        distance = float(np.hypot(self.target_x - self.pos_x, self.target_y - self.pos_y))
        reward = -distance

        self.steps += 1
        done = self.steps >= self.max_steps or distance < 0.05
        self.done = done
        return self._obs(), float(reward), bool(done), {"win": bool(distance < 0.05)}

    def render(self) -> None:
        # TODO: add Arcade renderer for walker skeleton.
        return None

    def close(self) -> None:
        return None
