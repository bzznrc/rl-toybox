"""Minimal football/pitch scaffold environment."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_NEAR_BLACK,
    COLOR_SLATE_GRAY,
    COLOR_SOFT_WHITE,
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.envs.base import Env
from core.runtime import ArcadeFrameClock, ArcadeWindowController


TILE_SIZE = DEFAULT_TILE_SIZE
GRID_WIDTH = DEFAULT_GRID_COLUMNS
GRID_HEIGHT = DEFAULT_GRID_ROWS
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
SCREEN_WIDTH = screen_width(GRID_WIDTH, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT, TILE_SIZE, BB_HEIGHT)
FPS = 60
TRAINING_FPS = 0
WINDOW_TITLE = "Kick"


class KickEnv(Env):
    """Top-down pitch scaffold for future multi-agent PPO.

    TODO: Parameter-sharing multi-agent (11 players): expose per-agent obs/action API
    or wrap this env with a small multi-agent adapter.
    """

    ACTION_NOOP = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_KICK = 5

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

        self.pitch_top = 0.0
        self.pitch_bottom = SCREEN_HEIGHT - BB_HEIGHT
        self.max_steps = 1_500
        self.reset()

    def _obs(self) -> np.ndarray:
        width = float(SCREEN_WIDTH)
        height = float(self.pitch_bottom)
        return np.asarray(
            [
                self.player_x / width,
                self.player_y / height,
                self.ball_x / width,
                self.ball_y / height,
                self.ball_vx / 10.0,
                self.ball_vy / 10.0,
                (self.opp_goal_x - self.ball_x) / width,
                (self.ball_y - height * 0.5) / (height * 0.5),
            ],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.player_x = SCREEN_WIDTH * 0.2
        self.player_y = (SCREEN_HEIGHT - BB_HEIGHT) * 0.5
        self.ball_x = SCREEN_WIDTH * 0.5
        self.ball_y = (SCREEN_HEIGHT - BB_HEIGHT) * 0.5
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.opp_goal_x = SCREEN_WIDTH - TILE_SIZE
        self.steps = 0
        self.done = False

        # Placeholder teammates/opponents for rendering only.
        self.teammates = [(SCREEN_WIDTH * 0.2, (SCREEN_HEIGHT - BB_HEIGHT) * 0.3)]
        self.opponents = [(SCREEN_WIDTH * 0.75, (SCREEN_HEIGHT - BB_HEIGHT) * 0.5)]
        return self._obs()

    def _human_action(self) -> int:
        if self.window_controller.is_key_down(arcade.key.W) or self.window_controller.is_key_down(arcade.key.UP):
            return self.ACTION_UP
        if self.window_controller.is_key_down(arcade.key.S) or self.window_controller.is_key_down(arcade.key.DOWN):
            return self.ACTION_DOWN
        if self.window_controller.is_key_down(arcade.key.A) or self.window_controller.is_key_down(arcade.key.LEFT):
            return self.ACTION_LEFT
        if self.window_controller.is_key_down(arcade.key.D) or self.window_controller.is_key_down(arcade.key.RIGHT):
            return self.ACTION_RIGHT
        if self.window_controller.is_key_down(arcade.key.SPACE):
            return self.ACTION_KICK
        return self.ACTION_NOOP

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._obs(), 0.0, True, {"win": False}

        self.window_controller.poll_events_or_raise()
        action_idx = self._human_action() if self.mode == "human" else int(action)

        move_speed = 4.0
        if action_idx == self.ACTION_UP:
            self.player_y -= move_speed
        elif action_idx == self.ACTION_DOWN:
            self.player_y += move_speed
        elif action_idx == self.ACTION_LEFT:
            self.player_x -= move_speed
        elif action_idx == self.ACTION_RIGHT:
            self.player_x += move_speed

        self.player_x = float(np.clip(self.player_x, TILE_SIZE, SCREEN_WIDTH - TILE_SIZE))
        self.player_y = float(np.clip(self.player_y, TILE_SIZE, self.pitch_bottom - TILE_SIZE))

        dx = self.ball_x - self.player_x
        dy = self.ball_y - self.player_y
        dist = np.hypot(dx, dy)
        if dist < TILE_SIZE * 1.5 and action_idx == self.ACTION_KICK:
            kick_scale = 7.0 / max(1e-6, dist)
            self.ball_vx = dx * kick_scale
            self.ball_vy = dy * kick_scale

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        self.ball_vx *= 0.96
        self.ball_vy *= 0.96

        if self.ball_y < TILE_SIZE or self.ball_y > self.pitch_bottom - TILE_SIZE:
            self.ball_vy *= -0.9
            self.ball_y = float(np.clip(self.ball_y, TILE_SIZE, self.pitch_bottom - TILE_SIZE))

        reward = -0.002
        done = False
        win = False

        if self.ball_x >= self.opp_goal_x:
            reward = 10.0
            done = True
            win = True
        elif self.ball_x <= TILE_SIZE:
            reward = -5.0
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        self.done = done
        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        return self._obs(), float(reward), bool(done), {"win": bool(win)}

    def render(self) -> None:
        if self.window_controller.window is None:
            return

        self.window_controller.clear(COLOR_CHARCOAL)

        pitch_h = self.pitch_bottom - self.pitch_top
        pitch_bottom = self.window_controller.top_left_to_bottom(self.pitch_top, pitch_h)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_DEEP_TEAL)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_AQUA + (30,))

        # Touch lines and center line.
        arcade.draw_lbwh_rectangle_outline(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_FOG_GRAY, 2)
        arcade.draw_line(
            SCREEN_WIDTH / 2,
            self.window_controller.to_arcade_y(self.pitch_top),
            SCREEN_WIDTH / 2,
            self.window_controller.to_arcade_y(self.pitch_bottom),
            COLOR_FOG_GRAY,
            2,
        )

        # Goal posts.
        goal_h = TILE_SIZE * 6
        goal_top = self.pitch_bottom / 2 - goal_h / 2
        left_goal_bottom = self.window_controller.top_left_to_bottom(goal_top, goal_h)
        arcade.draw_lbwh_rectangle_outline(0, left_goal_bottom, TILE_SIZE, goal_h, COLOR_SOFT_WHITE, 2)
        arcade.draw_lbwh_rectangle_outline(SCREEN_WIDTH - TILE_SIZE, left_goal_bottom, TILE_SIZE, goal_h, COLOR_SOFT_WHITE, 2)

        # Players (placeholder roster).
        player_bottom = self.window_controller.top_left_to_bottom(self.player_y - TILE_SIZE / 2, TILE_SIZE)
        arcade.draw_lbwh_rectangle_filled(self.player_x - TILE_SIZE / 2, player_bottom, TILE_SIZE, TILE_SIZE, COLOR_CORAL)
        arcade.draw_lbwh_rectangle_filled(
            self.player_x - TILE_SIZE / 2 + 3,
            player_bottom + 3,
            TILE_SIZE - 6,
            TILE_SIZE - 6,
            COLOR_BRICK_RED,
        )
        for teammate_x, teammate_y in self.teammates:
            teammate_bottom = self.window_controller.top_left_to_bottom(teammate_y - TILE_SIZE / 2, TILE_SIZE)
            arcade.draw_lbwh_rectangle_filled(
                teammate_x - TILE_SIZE / 2,
                teammate_bottom,
                TILE_SIZE,
                TILE_SIZE,
                COLOR_DEEP_TEAL,
            )
        for opp_x, opp_y in self.opponents:
            opp_bottom = self.window_controller.top_left_to_bottom(opp_y - TILE_SIZE / 2, TILE_SIZE)
            arcade.draw_lbwh_rectangle_filled(
                opp_x - TILE_SIZE / 2,
                opp_bottom,
                TILE_SIZE,
                TILE_SIZE,
                COLOR_SLATE_GRAY,
            )

        arcade.draw_circle_filled(
            self.ball_x,
            self.window_controller.to_arcade_y(self.ball_y),
            TILE_SIZE * 0.3,
            COLOR_FOG_GRAY,
        )

        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, BB_HEIGHT, COLOR_NEAR_BLACK)
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
