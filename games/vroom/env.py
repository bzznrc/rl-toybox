"""Minimal Vroom scaffold environment."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_FOG_GRAY,
    COLOR_NEAR_BLACK,
    COLOR_SLATE_GRAY,
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.envs.base import Env
from core.runtime import ArcadeFrameClock, ArcadeWindowController


# Keep Bang/Snake visual conventions.
TILE_SIZE = DEFAULT_TILE_SIZE
GRID_WIDTH = DEFAULT_GRID_COLUMNS
GRID_HEIGHT = DEFAULT_GRID_ROWS
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
SCREEN_WIDTH = screen_width(GRID_WIDTH, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT, TILE_SIZE, BB_HEIGHT)
FPS = 60
TRAINING_FPS = 0
WINDOW_TITLE = "Vroom"


class VroomEnv(Env):
    """Simple discrete driving sandbox.

    TODO: Replace with richer driving physics, checkpoints, and collision geometry.
    """

    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_ACCEL = 3
    ACTION_BRAKE = 4

    def __init__(self, mode: str = "train", render: bool = False) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=False,
            vsync=False,
        )

        self.track_left = SCREEN_WIDTH * 0.35
        self.track_right = SCREEN_WIDTH * 0.65
        self.track_top = 0.0
        self.track_bottom = SCREEN_HEIGHT - BB_HEIGHT

        self.max_steps = 1_000
        self.reset()

    def _get_obs(self) -> np.ndarray:
        center_x = (self.track_left + self.track_right) / 2.0
        norm_x = (self.car_x - center_x) / max(1.0, (self.track_right - self.track_left) / 2.0)
        norm_y = self.car_y / max(1.0, self.track_bottom)
        norm_vx = self.velocity_x / 8.0
        norm_vy = self.velocity_y / 8.0
        speed = np.sqrt(self.velocity_x**2 + self.velocity_y**2) / 10.0
        return np.asarray([norm_x, norm_y, norm_vx, norm_vy, speed], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.car_x = SCREEN_WIDTH / 2.0
        self.car_y = self.track_bottom - TILE_SIZE * 2
        self.velocity_x = 0.0
        self.velocity_y = -2.0
        self.steps = 0
        self.done = False
        return self._get_obs()

    def _resolve_human_action(self) -> int:
        if self.window_controller.is_key_down(arcade.key.LEFT) or self.window_controller.is_key_down(arcade.key.A):
            return self.ACTION_LEFT
        if self.window_controller.is_key_down(arcade.key.RIGHT) or self.window_controller.is_key_down(arcade.key.D):
            return self.ACTION_RIGHT
        if self.window_controller.is_key_down(arcade.key.UP) or self.window_controller.is_key_down(arcade.key.W):
            return self.ACTION_ACCEL
        if self.window_controller.is_key_down(arcade.key.DOWN) or self.window_controller.is_key_down(arcade.key.S):
            return self.ACTION_BRAKE
        return self.ACTION_NOOP

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._get_obs(), 0.0, True, {"win": False}

        self.window_controller.poll_events_or_raise()
        if self.mode == "human":
            action_idx = self._resolve_human_action()
        else:
            action_idx = int(action)

        if action_idx == self.ACTION_LEFT:
            self.velocity_x -= 0.4
        elif action_idx == self.ACTION_RIGHT:
            self.velocity_x += 0.4
        elif action_idx == self.ACTION_ACCEL:
            self.velocity_y -= 0.3
        elif action_idx == self.ACTION_BRAKE:
            self.velocity_y += 0.3

        self.velocity_x *= 0.96
        self.velocity_y *= 0.995

        self.car_x += self.velocity_x
        self.car_y += self.velocity_y
        self.steps += 1

        reward = 0.02
        done = False
        win = False

        # Reward forward progress toward top of track.
        reward += max(0.0, -self.velocity_y) * 0.01

        off_track = (
            self.car_x < self.track_left + TILE_SIZE / 2
            or self.car_x > self.track_right - TILE_SIZE / 2
            or self.car_y < self.track_top + TILE_SIZE / 2
            or self.car_y > self.track_bottom - TILE_SIZE / 2
        )
        if off_track:
            reward = -5.0
            done = True

        if self.car_y <= self.track_top + TILE_SIZE:
            reward = 10.0
            done = True
            win = True

        if self.steps >= self.max_steps:
            done = True

        self.done = done
        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        return self._get_obs(), float(reward), bool(done), {"win": bool(win)}

    def render(self) -> None:
        if self.window_controller.window is None:
            return

        self.window_controller.clear(COLOR_CHARCOAL)

        track_width = self.track_right - self.track_left
        track_bottom = self.window_controller.top_left_to_bottom(self.track_top, self.track_bottom - self.track_top)
        arcade.draw_lbwh_rectangle_filled(
            self.track_left,
            track_bottom,
            track_width,
            self.track_bottom - self.track_top,
            COLOR_SLATE_GRAY,
        )

        lane_x = (self.track_left + self.track_right) / 2.0 - TILE_SIZE * 0.15
        for y in range(0, int(self.track_bottom), TILE_SIZE * 2):
            arcade.draw_lbwh_rectangle_filled(
                lane_x,
                self.window_controller.top_left_to_bottom(y, TILE_SIZE),
                TILE_SIZE * 0.3,
                TILE_SIZE,
                COLOR_FOG_GRAY,
            )

        car_bottom = self.window_controller.top_left_to_bottom(self.car_y - TILE_SIZE / 2, TILE_SIZE)
        arcade.draw_lbwh_rectangle_filled(
            self.car_x - TILE_SIZE / 2,
            car_bottom,
            TILE_SIZE,
            TILE_SIZE,
            COLOR_CORAL,
        )
        arcade.draw_lbwh_rectangle_filled(
            self.car_x - TILE_SIZE / 2 + 3,
            car_bottom + 3,
            TILE_SIZE - 6,
            TILE_SIZE - 6,
            COLOR_BRICK_RED,
        )

        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, BB_HEIGHT, COLOR_NEAR_BLACK)
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
