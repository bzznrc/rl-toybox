"""Snake game logic for human play and RL training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Callable, TypeVar

import arcade
import numpy as np

from snake_ai.config import (
    BB_HEIGHT,
    CELL_INSET,
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_NEAR_BLACK,
    COLOR_SLATE_GRAY,
    FPS,
    MAX_OBSTACLE_SECTIONS,
    MIN_OBSTACLE_SECTIONS,
    NN_CONTROL_MARKER_SIZE_PX,
    NUM_OBSTACLES,
    REWARD_DEATH_OR_TIMEOUT,
    REWARD_FOOD,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TRAINING_FPS,
    WINDOW_TITLE,
    WRAP_AROUND,
)
from snake_ai.runtime import ArcadeFrameClock, ArcadeWindowController

T = TypeVar("T")


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

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass(frozen=True)
class Point:
    x: float
    y: float


class BaseSnakeGame:
    """Shared world state and rendering for Snake."""

    def __init__(self, show_game: bool = True) -> None:
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.show_game = bool(show_game)
        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            self.width,
            self.height,
            WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=False,
            vsync=False,
        )
        self.window = self.window_controller.window

        self.direction = Direction.RIGHT
        self.head = Point(0, 0)
        self.snake: list[Point] = []
        self.score = 0
        self.food = Point(0, 0)
        self.obstacles: list[Point] = []
        self.frame_iteration = 0
        self.reset()

    def close(self) -> None:
        self.window_controller.close()
        self.window = None

    def poll_events(self) -> None:
        self.window_controller.poll_events_or_raise()

    def reset(self) -> None:
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2 - BB_HEIGHT // 2)
        self.snake = [
            self.head,
            Point(self.head.x - TILE_SIZE, self.head.y),
            Point(self.head.x - (2 * TILE_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = Point(0, 0)
        self.obstacles = []
        self._place_food()
        self._place_obstacles()
        self.frame_iteration = 0

    def _place_food(self) -> None:
        while True:
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            if y >= self.height - BB_HEIGHT:
                continue
            food = Point(x, y)
            if food not in self.snake and food not in self.obstacles:
                self.food = food
                return

    def _place_obstacles(self) -> None:
        self.obstacles = []
        shapes = spawn_connected_random_walk_shapes(
            shape_count=NUM_OBSTACLES,
            min_sections=MIN_OBSTACLE_SECTIONS,
            max_sections=MAX_OBSTACLE_SECTIONS,
            sample_start_fn=self._sample_valid_obstacle_start,
            neighbor_candidates_fn=self._neighbor_obstacle_candidates,
            is_candidate_valid_fn=self._is_valid_obstacle_tile,
        )
        for shape in shapes:
            self.obstacles.extend(shape)

    def _sample_valid_obstacle_start(self) -> Point | None:
        for _ in range(100):
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            point = Point(x, y)
            if self._is_valid_obstacle_tile(point, []):
                return point
        return None

    @staticmethod
    def _neighbor_obstacle_candidates(point: Point) -> list[Point]:
        return [
            Point(point.x - TILE_SIZE, point.y),
            Point(point.x + TILE_SIZE, point.y),
            Point(point.x, point.y - TILE_SIZE),
            Point(point.x, point.y + TILE_SIZE),
        ]

    def _is_valid_obstacle_tile(self, tile: Point, pending_tiles: list[Point]) -> bool:
        if not (0 <= tile.x < self.width and 0 <= tile.y < self.height - BB_HEIGHT):
            return False
        if tile in self.snake or tile == self.food:
            return False
        if tile in self.obstacles or tile in pending_tiles:
            return False
        return True

    def _to_window_y(self, top_y: float, height: float = 0) -> float:
        return self.window_controller.to_arcade_y(float(top_y) + float(height))

    def _draw_tile(self, top_left: Point, outer_color, inner_color) -> None:
        x = float(top_left.x)
        y = float(top_left.y)
        bottom = self._to_window_y(y, TILE_SIZE)

        arcade.draw_lbwh_rectangle_filled(x, bottom, TILE_SIZE, TILE_SIZE, outer_color)
        inner_size = TILE_SIZE - 2 * CELL_INSET
        if inner_size > 0:
            arcade.draw_lbwh_rectangle_filled(
                x + CELL_INSET,
                bottom + CELL_INSET,
                inner_size,
                inner_size,
                inner_color,
            )

    def _draw_tile_batch(self, tiles: list[Point], outer_color, inner_color) -> None:
        for tile in tiles:
            self._draw_tile(tile, outer_color, inner_color)

    def _draw_status_bar(self) -> None:
        arcade.draw_lbwh_rectangle_filled(0, 0, self.width, BB_HEIGHT, COLOR_NEAR_BLACK)
        self._draw_score_icons(left=8.0, right=self.width - 8.0, center_y=BB_HEIGHT / 2.0)

    @staticmethod
    def _status_icon_size() -> float:
        return max(12.0, min(float(BB_HEIGHT - 8), float(TILE_SIZE)))

    def _score_icons(self) -> list[bool]:
        compressed_icons = self.score // 5
        single_icons = self.score % 5
        return ([True] * compressed_icons) + ([False] * single_icons)

    def _draw_score_icons(self, left: float, right: float, center_y: float) -> None:
        available_width = max(0.0, float(right) - float(left))
        if available_width <= 0.0:
            return

        icon_size = self._status_icon_size()
        icon_gap = 6.0
        if icon_size <= 0.0:
            return

        max_icons = int((available_width + icon_gap) // (icon_size + icon_gap))
        if max_icons <= 0:
            return

        icons = self._score_icons()
        if not icons:
            return
        icons = icons[-max_icons:]

        total_width = len(icons) * icon_size + max(0, len(icons) - 1) * icon_gap
        start_x = float(left) + (available_width - total_width) / 2.0
        for idx, is_compressed in enumerate(icons):
            center_x = start_x + icon_size / 2.0 + idx * (icon_size + icon_gap)
            self._draw_fruit_icon(center_x=center_x, center_y=center_y, size=icon_size, compressed=is_compressed)

    def _draw_fruit_icon(self, center_x: float, center_y: float, size: float, compressed: bool = False) -> None:
        bottom = center_y - size / 2.0
        left = center_x - size / 2.0
        arcade.draw_lbwh_rectangle_filled(left, bottom, size, size, COLOR_CORAL)

        inset = max(1.0, round(CELL_INSET * (size / max(1.0, float(TILE_SIZE)))))
        inner_size = max(1.0, size - 2.0 * inset)
        arcade.draw_lbwh_rectangle_filled(
            left + inset,
            bottom + inset,
            inner_size,
            inner_size,
            COLOR_BRICK_RED,
        )

        if compressed:
            marker_size = max(2.0, round(NN_CONTROL_MARKER_SIZE_PX * (size / max(1.0, float(TILE_SIZE)))))
            arcade.draw_lbwh_rectangle_filled(
                center_x - marker_size / 2.0,
                center_y - marker_size / 2.0,
                marker_size,
                marker_size,
                COLOR_CORAL,
            )

    def draw_frame(self) -> None:
        if self.window is None:
            return

        self.window_controller.clear(COLOR_CHARCOAL)
        self._draw_tile_batch(self.snake, COLOR_AQUA, COLOR_DEEP_TEAL)
        self._draw_tile(self.food, COLOR_CORAL, COLOR_BRICK_RED)
        self._draw_tile_batch(self.obstacles, COLOR_FOG_GRAY, COLOR_SLATE_GRAY)
        self._draw_status_bar()
        self.window_controller.flip()

    def _is_out_of_bounds(self, point: Point) -> bool:
        return (
            point.x < 0
            or point.x >= self.width
            or point.y < 0
            or point.y >= self.height - BB_HEIGHT
        )

    def _move_one_tile(self, direction: Direction) -> None:
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += TILE_SIZE
        elif direction == Direction.LEFT:
            x -= TILE_SIZE
        elif direction == Direction.DOWN:
            y += TILE_SIZE
        elif direction == Direction.UP:
            y -= TILE_SIZE
        self.head = Point(x, y)

        if WRAP_AROUND:
            self._handle_wall_collision()

    def is_collision(self, point: Point | None = None) -> bool:
        point = self.head if point is None else point
        if point in self.snake[1:]:
            return True
        if point in self.obstacles:
            return True
        return False

    def _handle_wall_collision(self) -> None:
        x = self.head.x
        y = self.head.y

        if x >= self.width:
            x = 0
        elif x < 0:
            x = self.width - TILE_SIZE
        if y >= self.height - BB_HEIGHT:
            y = 0
        elif y < 0:
            y = self.height - BB_HEIGHT - TILE_SIZE

        self.head = Point(x, y)


class HumanSnakeGame(BaseSnakeGame):
    """User-controlled Snake game mode."""

    def play_step(self) -> tuple[bool, int]:
        self.frame_iteration += 1
        self.poll_events()

        if self.window_controller.is_key_down(arcade.key.A) and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif self.window_controller.is_key_down(arcade.key.D) and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT
        elif self.window_controller.is_key_down(arcade.key.W) and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif self.window_controller.is_key_down(arcade.key.S) and self.direction != Direction.UP:
            self.direction = Direction.DOWN

        self._move_one_tile(self.direction)
        self.snake.insert(0, self.head)

        if self._has_collision():
            return True, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self.draw_frame()
        self.frame_clock.tick(FPS)
        return False, self.score

    def _has_collision(self) -> bool:
        if not WRAP_AROUND and self._is_out_of_bounds(self.head):
            return True
        return self.is_collision()


class TrainingSnakeGame(BaseSnakeGame):
    """AI-controlled training environment."""

    def play_step(self, action: list[int]) -> tuple[int, bool, int]:
        self.frame_iteration += 1
        self.poll_events()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        if self._has_collision() or self.frame_iteration > 100 * len(self.snake):
            return int(REWARD_DEATH_OR_TIMEOUT), True, self.score

        if self.head == self.food:
            self.score += 1
            reward = int(REWARD_FOOD)
            self._place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()

        self.draw_frame()
        self.frame_clock.tick(TRAINING_FPS)

        return reward, False, self.score

    def get_state_vector(self) -> np.ndarray:
        head = self.snake[0]
        point_l = Point(head.x - TILE_SIZE, head.y)
        point_r = Point(head.x + TILE_SIZE, head.y)
        point_u = Point(head.x, head.y - TILE_SIZE)
        point_d = Point(head.x, head.y + TILE_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            (dir_r and self.is_collision(point_r))
            or (dir_l and self.is_collision(point_l))
            or (dir_u and self.is_collision(point_u))
            or (dir_d and self.is_collision(point_d)),
            (dir_u and self.is_collision(point_r))
            or (dir_d and self.is_collision(point_l))
            or (dir_l and self.is_collision(point_u))
            or (dir_r and self.is_collision(point_d)),
            (dir_d and self.is_collision(point_r))
            or (dir_u and self.is_collision(point_l))
            or (dir_r and self.is_collision(point_u))
            or (dir_l and self.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            self.food.x < self.head.x,
            self.food.x > self.head.x,
            self.food.y < self.head.y,
            self.food.y > self.head.y,
            len(self.snake),
        ]
        return np.array(state, dtype=int)

    def _move(self, action: list[int]) -> None:
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clockwise[(idx + 1) % 4]
        else:
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir
        self._move_one_tile(new_dir)

    def _has_collision(self) -> bool:
        if not WRAP_AROUND and self._is_out_of_bounds(self.head):
            return True
        return self.is_collision()
