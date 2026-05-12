"""Snake game logic for human play and RL training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_SLATE_GRAY,
)
from core.curriculum import (
    SharedCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.ghost_overlay import update_ghost_overlay_toggle
from core.io_schema import clip_signed, clip_unit, normalize_last_action, ordered_feature_vector, signed_potential_shaping
from core.match_tracker import compact_count_to_icons
from core.primitives import (
    draw_status_bar,
    draw_status_icon_row,
    status_icon_inset,
    draw_status_square_icon,
    draw_two_tone_cell,
    spawn_connected_random_walk_shapes,
    status_icon_size,
)
from core.ray_viz import draw_player_rays
from core.rewards import RewardBreakdown
from core.shared_config import (
    BB_HEIGHT,
    CELL_INSET,
    FPS,
    NN_CONTROL_MARKER_SIZE_PX,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TRAINING_FPS,
)
from games.snake.config import (
    ACTION_NAMES as SNAKE_ACTION_NAMES,
    ACT_DIM as SNAKE_ACT_DIM,
    CURRICULUM_PROMOTION,
    FOOD_TIMEOUT_STEPS,
    LEVEL_SETTINGS,
    MAX_OBSTACLE_SECTIONS,
    MAX_LEVEL,
    MIN_OBSTACLE_SECTIONS,
    MIN_LEVEL,
    OBS_DIM as SNAKE_OBS_DIM,
    INPUT_FEATURE_NAMES as SNAKE_INPUT_FEATURE_NAMES,
    PENALTY_LOSE,
    PROGRESS_CLIP,
    PROGRESS_SCALE,
    REWARD_FOOD,
    PENALTY_STEP,
    SHOW_GHOST_OVERLAY,
    SUCCESS_FOODS_REQUIRED,
    WINDOW_TITLE,
    WRAP_AROUND,
)
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level


validate_curriculum_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
)


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
        self.show_ghost_overlay = bool(SHOW_GHOST_OVERLAY)
        self.ghost_overlay_allowed = True
        self._prev_ghost_overlay_toggle_down = False
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
        self.num_obstacles = int(LEVEL_SETTINGS[int(MIN_LEVEL)]["num_obstacles"])
        self.frame_iteration = 0
        self.last_action_index = 0
        self.steps_since_food = 0
        self._prev_tgt_manhattan_norm: float | None = None
        self.reset()

    def close(self) -> None:
        self.window_controller.close()
        self.window = None

    def poll_events(self) -> None:
        self.window_controller.poll_events_or_raise()
        self.show_ghost_overlay, self._prev_ghost_overlay_toggle_down = update_ghost_overlay_toggle(
            window_controller=self.window_controller,
            visible=bool(self.show_ghost_overlay),
            previous_down=bool(self._prev_ghost_overlay_toggle_down),
            enabled=bool(self.show_game and self.ghost_overlay_allowed),
        )

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
        self.last_action_index = 0
        self.steps_since_food = 0
        self._prev_tgt_manhattan_norm = None

    @staticmethod
    def _clockwise_directions() -> list[Direction]:
        return [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

    def _action_index_for_direction_change(self, previous: Direction, current: Direction) -> int:
        clockwise = self._clockwise_directions()
        prev_idx = clockwise.index(previous)
        curr_idx = clockwise.index(current)
        if curr_idx == prev_idx:
            return 0
        if curr_idx == (prev_idx + 1) % len(clockwise):
            return 1
        return 2

    def _place_food(self) -> None:
        grid_w = max(1, int(self.width // TILE_SIZE))
        grid_h = max(1, int((self.height - BB_HEIGHT) // TILE_SIZE))
        preferred_tiles: list[Point] = []
        fallback_tiles: list[Point] = []
        for cell_x in range(grid_w):
            for cell_y in range(grid_h):
                tile = self._point_from_cells(cell_x, cell_y)
                if tile in self.snake or tile in self.obstacles:
                    continue
                fallback_tiles.append(tile)
                if self._count_free_neighbors(tile) >= 2:
                    preferred_tiles.append(tile)
        pool = preferred_tiles if preferred_tiles else fallback_tiles
        if not pool:
            self.food = self.head
            return
        self.food = random.choice(pool)

    @staticmethod
    def _point_from_cells(cell_x: int, cell_y: int) -> Point:
        return Point(float(cell_x * TILE_SIZE), float(cell_y * TILE_SIZE))

    def _normalize_neighbor_tile(self, tile: Point) -> Point | None:
        if WRAP_AROUND:
            wrapped_x = float(tile.x % self.width)
            wrapped_y = float(tile.y % (self.height - BB_HEIGHT))
            return Point(wrapped_x, wrapped_y)
        if 0 <= tile.x < self.width and 0 <= tile.y < self.height - BB_HEIGHT:
            return tile
        return None

    def _count_free_neighbors(self, tile: Point) -> int:
        neighbors = (
            Point(tile.x - TILE_SIZE, tile.y),
            Point(tile.x + TILE_SIZE, tile.y),
            Point(tile.x, tile.y - TILE_SIZE),
            Point(tile.x, tile.y + TILE_SIZE),
        )
        free_neighbors = 0
        for neighbor in neighbors:
            normalized = self._normalize_neighbor_tile(neighbor)
            if normalized is None:
                continue
            if normalized in self.snake or normalized in self.obstacles:
                continue
            free_neighbors += 1
        return free_neighbors

    def _place_obstacles(self) -> None:
        self.obstacles = []
        shape_count = max(0, int(self.num_obstacles))
        if shape_count <= 0:
            return
        shapes = spawn_connected_random_walk_shapes(
            shape_count=shape_count,
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

    def _draw_status_bar(self) -> None:
        layout = draw_status_bar(
            width=float(self.width),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=float(CELL_INSET),
            include_clock=False,
        )
        self._draw_score_icons(
            left=float(layout.score_left),
            right=float(layout.score_right),
            center_y=float(layout.center_y),
        )

    @staticmethod
    def _status_icon_size() -> float:
        return status_icon_size(float(BB_HEIGHT), float(TILE_SIZE))

    def _score_icons(self) -> list[bool]:
        return compact_count_to_icons(int(self.score), pack_size=5)

    def _draw_score_icons(self, left: float, right: float, center_y: float) -> None:
        icon_size = self._status_icon_size()
        icons = self._score_icons()
        draw_status_icon_row(
            left=float(left),
            right=float(right),
            center_y=float(center_y),
            icon_size=float(icon_size),
            items=icons,
            draw_item=lambda is_compressed, center_x, row_center_y, size: self._draw_fruit_icon(
                center_x=float(center_x),
                center_y=float(row_center_y),
                size=float(size),
                compressed=bool(is_compressed),
            ),
        )

    def _draw_fruit_icon(self, center_x: float, center_y: float, size: float, compressed: bool = False) -> None:
        inset = status_icon_inset(float(CELL_INSET))
        marker_size = max(2.0, round(NN_CONTROL_MARKER_SIZE_PX * (size / max(1.0, float(TILE_SIZE)))))
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=COLOR_CORAL,
            inner_color=COLOR_BRICK_RED,
            inset=float(inset),
            packed=bool(compressed),
            packed_marker_color=COLOR_CORAL,
            packed_marker_size=float(marker_size),
        )

    @staticmethod
    def _direction_vector(direction: Direction) -> tuple[int, int]:
        if direction == Direction.RIGHT:
            return 1, 0
        if direction == Direction.LEFT:
            return -1, 0
        if direction == Direction.UP:
            return 0, -1
        return 0, 1

    @staticmethod
    def _left_vector(dx: int, dy: int) -> tuple[int, int]:
        return dy, -dx

    @staticmethod
    def _right_vector(dx: int, dy: int) -> tuple[int, int]:
        return -dy, dx

    def _grid_dims(self) -> tuple[int, int]:
        return int(self.width // TILE_SIZE), int((self.height - BB_HEIGHT) // TILE_SIZE)

    def _cell_from_point(self, point: Point) -> tuple[int, int]:
        return int(point.x // TILE_SIZE), int(point.y // TILE_SIZE)

    def _point_from_cell(self, cell_x: int, cell_y: int) -> Point:
        return Point(float(cell_x * TILE_SIZE), float(cell_y * TILE_SIZE))

    def _is_collision_for_cell(self, cell_x: int, cell_y: int) -> bool:
        grid_w, grid_h = self._grid_dims()
        if WRAP_AROUND:
            cell_x = int(cell_x % grid_w)
            cell_y = int(cell_y % grid_h)
        else:
            if cell_x < 0 or cell_x >= grid_w or cell_y < 0 or cell_y >= grid_h:
                return True
        point = self._point_from_cell(cell_x, cell_y)
        return bool(point in self.snake[1:] or point in self.obstacles)

    def _ray_distance_to_collision(self, dir_x: int, dir_y: int) -> float:
        head_cell_x, head_cell_y = self._cell_from_point(self.head)
        grid_w, grid_h = self._grid_dims()
        max_range = max(1, int(max(grid_w, grid_h)))
        for step in range(1, max_range + 1):
            cell_x = head_cell_x + int(dir_x) * step
            cell_y = head_cell_y + int(dir_y) * step
            if self._is_collision_for_cell(cell_x, cell_y):
                return float(step - 1) / float(max_range)
        return 1.0

    def _draw_ghost_overlay(self) -> None:
        if not bool(self.show_ghost_overlay and self.show_game):
            return
        dir_x, dir_y = self._direction_vector(self.direction)
        left_x, left_y = self._left_vector(dir_x, dir_y)
        right_x, right_y = self._right_vector(dir_x, dir_y)
        ray_dirs = ((dir_x, dir_y), (left_x, left_y), (right_x, right_y))
        ray_values = tuple(float(self._ray_distance_to_collision(dx, dy)) for dx, dy in ray_dirs)
        grid_w, grid_h = self._grid_dims()
        max_distance = float(max(1, int(max(grid_w, grid_h))) * TILE_SIZE)
        draw_player_rays(
            origin_x=float(self.head.x + TILE_SIZE * 0.5),
            origin_y=float(self.head.y + TILE_SIZE * 0.5),
            ray_dirs=ray_dirs,
            ray_values=ray_values,
            ray_max_distances=(max_distance,) * len(ray_dirs),
            to_screen=lambda x, y: (float(x), float(self.window_controller.to_arcade_y(float(y)))),
            line_width=1.5,
        )

    def draw_frame(self) -> None:
        if self.window is None:
            return

        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        self._draw_ghost_overlay()
        for tile in self.snake:
            draw_two_tone_cell(
                self.window_controller,
                top_left_x=float(tile.x),
                top_left_y=float(tile.y),
                tile_size=float(TILE_SIZE),
                outer_color=COLOR_AQUA,
                inner_color=COLOR_DEEP_TEAL,
                cell_inset=float(CELL_INSET),
            )
        draw_two_tone_cell(
            self.window_controller,
            top_left_x=float(self.food.x),
            top_left_y=float(self.food.y),
            tile_size=float(TILE_SIZE),
            outer_color=COLOR_CORAL,
            inner_color=COLOR_BRICK_RED,
            cell_inset=float(CELL_INSET),
        )
        for tile in self.obstacles:
            draw_two_tone_cell(
                self.window_controller,
                top_left_x=float(tile.x),
                top_left_y=float(tile.y),
                tile_size=float(TILE_SIZE),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
                cell_inset=float(CELL_INSET),
            )
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
        previous_direction = self.direction

        if self.window_controller.is_key_down(arcade.key.A) and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif self.window_controller.is_key_down(arcade.key.D) and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT
        elif self.window_controller.is_key_down(arcade.key.W) and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif self.window_controller.is_key_down(arcade.key.S) and self.direction != Direction.UP:
            self.direction = Direction.DOWN
        self.last_action_index = self._action_index_for_direction_change(previous_direction, self.direction)

        self._move_one_tile(self.direction)
        self.snake.insert(0, self.head)

        if self._has_collision():
            return True, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
            self.steps_since_food = 0
            self._prev_tgt_manhattan_norm = None
        else:
            self.steps_since_food += 1
            self.snake.pop()

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)
        return False, self.score

    def _has_collision(self) -> bool:
        if not WRAP_AROUND and self._is_out_of_bounds(self.head):
            return True
        return self.is_collision()


class TrainingSnakeGame(BaseSnakeGame):
    """AI-controlled training environment."""

    def __init__(self, show_game: bool = True) -> None:
        super().__init__(show_game=show_game)
        self.foods_eaten = 0
        self.last_reward_breakdown: dict[str, float] = {}

    def reset(self) -> None:
        super().reset()
        self.foods_eaten = 0
        self.last_reward_breakdown = {
            "step.penalty_step": 0.0,
            "progress.shape": 0.0,
            "event.reward_food": 0.0,
            "outcome.penalty_lose": 0.0,
        }

    @staticmethod
    def _direction_vector(direction: Direction) -> tuple[int, int]:
        if direction == Direction.RIGHT:
            return 1, 0
        if direction == Direction.LEFT:
            return -1, 0
        if direction == Direction.UP:
            return 0, -1
        return 0, 1

    @staticmethod
    def _left_vector(dx: int, dy: int) -> tuple[int, int]:
        return dy, -dx

    @staticmethod
    def _right_vector(dx: int, dy: int) -> tuple[int, int]:
        return -dy, dx

    def _grid_dims(self) -> tuple[int, int]:
        return int(self.width // TILE_SIZE), int((self.height - BB_HEIGHT) // TILE_SIZE)

    def _cell_from_point(self, point: Point) -> tuple[int, int]:
        return int(point.x // TILE_SIZE), int(point.y // TILE_SIZE)

    def _point_from_cell(self, cell_x: int, cell_y: int) -> Point:
        return Point(float(cell_x * TILE_SIZE), float(cell_y * TILE_SIZE))

    @staticmethod
    def _wrap_delta_cells(delta: int, size: int) -> int:
        wrapped = int(delta)
        half = int(size) / 2.0
        if wrapped > half:
            wrapped -= int(size)
        elif wrapped < -half:
            wrapped += int(size)
        return wrapped

    def _action_index(self, action: list[int]) -> int:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size <= 0:
            return 0
        if values.size != SNAKE_ACT_DIM:
            return int(np.clip(int(values[0]), 0, SNAKE_ACT_DIM - 1))
        return int(np.argmax(values))

    def _is_collision_for_cell(self, cell_x: int, cell_y: int) -> bool:
        grid_w, grid_h = self._grid_dims()
        if WRAP_AROUND:
            cell_x = int(cell_x % grid_w)
            cell_y = int(cell_y % grid_h)
        else:
            if cell_x < 0 or cell_x >= grid_w or cell_y < 0 or cell_y >= grid_h:
                return True
        point = self._point_from_cell(cell_x, cell_y)
        return bool(point in self.snake[1:] or point in self.obstacles)

    def _ray_distance_to_collision(self, dir_x: int, dir_y: int) -> float:
        head_cell_x, head_cell_y = self._cell_from_point(self.head)
        grid_w, grid_h = self._grid_dims()
        max_range = max(1, int(max(grid_w, grid_h)))
        for step in range(1, max_range + 1):
            cell_x = head_cell_x + int(dir_x) * step
            cell_y = head_cell_y + int(dir_y) * step
            if self._is_collision_for_cell(cell_x, cell_y):
                return float(step - 1) / float(max_range)
        return 1.0

    def _food_manhattan_norm(self) -> float:
        grid_w, grid_h = self._grid_dims()
        max_manhattan = max(1, (grid_w // 2 + grid_h // 2) if WRAP_AROUND else (grid_w - 1 + grid_h - 1))
        head_cell_x, head_cell_y = self._cell_from_point(self.head)
        food_cell_x, food_cell_y = self._cell_from_point(self.food)
        raw_dx = int(food_cell_x - head_cell_x)
        raw_dy = int(food_cell_y - head_cell_y)
        dx_cells = self._wrap_delta_cells(raw_dx, grid_w) if WRAP_AROUND else raw_dx
        dy_cells = self._wrap_delta_cells(raw_dy, grid_h) if WRAP_AROUND else raw_dy
        manhattan_dist = abs(dx_cells) + abs(dy_cells)
        return float(clip_unit(float(manhattan_dist) / float(max_manhattan)))

    @staticmethod
    def _hunger_norm(steps_since_food: int) -> float:
        hunger_steps = max(0, int(steps_since_food))
        timeout_steps = max(1, int(FOOD_TIMEOUT_STEPS))
        return float(clip_unit(float(hunger_steps) / float(timeout_steps)))

    def _food_progress_potential(
        self,
        *,
        steps_since_food: int | None = None,
    ) -> float:
        dist_food_norm = float(self._food_manhattan_norm())
        hunger_steps = int(self.steps_since_food if steps_since_food is None else steps_since_food)
        hunger_norm = float(self._hunger_norm(hunger_steps))
        return float(-dist_food_norm - 0.5 * hunger_norm)

    def play_step(self, action: list[int]) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        self.poll_events()

        phi_prev = float(
            self._food_progress_potential(
                steps_since_food=int(self.steps_since_food),
            )
        )
        action_idx = self._action_index(action)
        self.last_action_index = int(action_idx)
        self._move(action_idx)
        self.snake.insert(0, self.head)

        reward = float(PENALTY_STEP)
        reward_breakdown = {
            "step.penalty_step": float(PENALTY_STEP),
            "progress.shape": 0.0,
            "event.reward_food": 0.0,
            "outcome.penalty_lose": 0.0,
        }
        if self._has_collision():
            reached_success_target = int(self.score) >= int(SUCCESS_FOODS_REQUIRED)
            if not reached_success_target:
                reward += float(PENALTY_LOSE)
                reward_breakdown["outcome.penalty_lose"] = float(PENALTY_LOSE)
            self.last_reward_breakdown = reward_breakdown
            return reward, True, self.score

        ate_food = bool(self.head == self.food)
        next_steps_since_food = 0 if ate_food else int(self.steps_since_food) + 1
        if not ate_food:
            self.snake.pop()
            self.steps_since_food = int(next_steps_since_food)
            if int(self.steps_since_food) >= int(FOOD_TIMEOUT_STEPS):
                reached_success_target = int(self.score) >= int(SUCCESS_FOODS_REQUIRED)
                if not reached_success_target:
                    reward += float(PENALTY_LOSE)
                    reward_breakdown["outcome.penalty_lose"] = float(PENALTY_LOSE)
                self.last_reward_breakdown = reward_breakdown
                return reward, True, self.score

        phi_next = float(
            self._food_progress_potential(
                steps_since_food=next_steps_since_food,
            )
        )
        progress_reward = float(
            signed_potential_shaping(
                phi_prev=phi_prev,
                phi_next=phi_next,
                scale=float(PROGRESS_SCALE),
                clip_abs=float(PROGRESS_CLIP),
            )
        )
        reward += progress_reward
        reward_breakdown["progress.shape"] = progress_reward

        if ate_food:
            self.score += 1
            self.foods_eaten += 1
            reward += float(REWARD_FOOD)
            reward_breakdown["event.reward_food"] = float(REWARD_FOOD)
            self._place_food()
            self.steps_since_food = 0
            self._prev_tgt_manhattan_norm = None

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)
        self.last_reward_breakdown = reward_breakdown

        return reward, False, self.score

    def get_state_vector(self) -> np.ndarray:
        grid_w, grid_h = self._grid_dims()
        grid_cells = max(1, grid_w * grid_h)
        max_manhattan = max(1, (grid_w // 2 + grid_h // 2) if WRAP_AROUND else (grid_w - 1 + grid_h - 1))

        dir_x, dir_y = self._direction_vector(self.direction)
        left_x, left_y = self._left_vector(dir_x, dir_y)
        right_x, right_y = self._right_vector(dir_x, dir_y)
        heading_sin = float(np.sin(np.arctan2(float(dir_y), float(dir_x))))
        heading_cos = float(np.cos(np.arctan2(float(dir_y), float(dir_x))))

        head_cell_x, head_cell_y = self._cell_from_point(self.head)
        food_cell_x, food_cell_y = self._cell_from_point(self.food)
        raw_dx = int(food_cell_x - head_cell_x)
        raw_dy = int(food_cell_y - head_cell_y)
        dx_cells = self._wrap_delta_cells(raw_dx, grid_w) if WRAP_AROUND else raw_dx
        dy_cells = self._wrap_delta_cells(raw_dy, grid_h) if WRAP_AROUND else raw_dy
        manhattan_dist = abs(dx_cells) + abs(dy_cells)
        manhattan_norm = clip_unit(float(manhattan_dist) / float(max_manhattan))
        target_norm = float(np.hypot(float(dx_cells), float(dy_cells)))
        if target_norm <= 1e-8:
            tgt_rel_angle_cos = 1.0
            tgt_rel_angle_sin = 0.0
        else:
            tgt_x = float(dx_cells) / target_norm
            tgt_y = float(dy_cells) / target_norm
            tgt_rel_angle_cos = float(clip_signed(float(heading_cos) * tgt_x + float(heading_sin) * tgt_y))
            tgt_rel_angle_sin = float(clip_signed(float(heading_cos) * tgt_y - float(heading_sin) * tgt_x))
        if self._prev_tgt_manhattan_norm is None:
            tgt_dist_delta = 0.0
        else:
            tgt_dist_delta = clip_signed(float(manhattan_norm) - float(self._prev_tgt_manhattan_norm))
        self._prev_tgt_manhattan_norm = float(manhattan_norm)

        feature_values = {
            "self_heading_sin": float(heading_sin),
            "self_heading_cos": float(heading_cos),
            "self_len_norm": float(clip_unit(float(len(self.snake)) / float(grid_cells))),
            "self_last_act_norm": float(normalize_last_action(self.last_action_index, SNAKE_ACT_DIM)),
            "self_hunger_norm": float(self._hunger_norm(int(self.steps_since_food))),
            "sens_fwd": float(self._ray_distance_to_collision(dir_x, dir_y)),
            "sens_left": float(self._ray_distance_to_collision(left_x, left_y)),
            "sens_right": float(self._ray_distance_to_collision(right_x, right_y)),
            "tgt_rel_angle_sin": float(tgt_rel_angle_sin),
            "tgt_rel_angle_cos": float(tgt_rel_angle_cos),
            "tgt_manhattan_norm": float(manhattan_norm),
            "tgt_dist_delta": float(tgt_dist_delta),
        }
        state = np.asarray(ordered_feature_vector(SNAKE_INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        assert len(state) == SNAKE_OBS_DIM
        if state.shape != (SNAKE_OBS_DIM,):
            raise RuntimeError(f"Snake observation expected {SNAKE_OBS_DIM} features, got {state.shape[0]}")
        return state

    def _move(self, action_idx: int) -> None:
        clockwise = self._clockwise_directions()
        idx = clockwise.index(self.direction)

        if int(action_idx) == 0:
            new_dir = clockwise[idx]
        elif int(action_idx) == 1:
            new_dir = clockwise[(idx + 1) % 4]
        else:
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir
        self._move_one_tile(new_dir)

    def _has_collision(self) -> bool:
        if not WRAP_AROUND and self._is_out_of_bounds(self.head):
            return True
        return self.is_collision()


class SnakeEnv(Env):
    """Env adapter for Snake game modes."""

    INPUT_FEATURE_NAMES = tuple(SNAKE_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(SNAKE_ACTION_NAMES)
    OBS_DIM = int(SNAKE_OBS_DIM)
    ACT_DIM = int(SNAKE_ACT_DIM)
    REWARD_COMPONENT_ORDER = ("F", "L", "P", "S")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "event.reward_food": "F",
        "outcome.penalty_lose": "L",
        "progress.shape": "P",
        "step.penalty_step": "S",
    }

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        curriculum_config = build_curriculum_config(
            min_level=int(MIN_LEVEL),
            max_level=int(MAX_LEVEL),
            promotion_settings=CURRICULUM_PROMOTION,
        )
        self._curriculum = (
            SharedCurriculum(config=curriculum_config, level_settings=LEVEL_SETTINGS)
            if self.mode == "train"
            else None
        )
        self._current_level = (
            int(self._curriculum.get_level())
            if self._curriculum is not None
            else resolve_play_level(level=level, min_level=MIN_LEVEL, max_level=MAX_LEVEL, default_level=MAX_LEVEL)
        )
        self._last_episode_level = int(self._current_level)
        self._last_episode_success = 0
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)
        if self.mode == "human":
            self.game = HumanSnakeGame(show_game=bool(render))
            self._apply_level_settings(int(self._current_level))
        else:
            self.game = TrainingSnakeGame(show_game=bool(render))
            self._apply_level_settings(int(self._current_level))
        self.game.ghost_overlay_allowed = bool(self.mode in {"human", "eval"})
        self.window_controller = self.game.window_controller
        self.window = self.game.window

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level), LEVEL_SETTINGS[int(MIN_LEVEL)])
        if not isinstance(self.game, BaseSnakeGame):
            return
        self.game.num_obstacles = max(0, int(settings["num_obstacles"]))

    @staticmethod
    def _action_to_one_hot(action_idx: int) -> list[int]:
        one_hot = [0] * int(SnakeEnv.ACT_DIM)
        action = max(0, min(int(action_idx), int(SnakeEnv.ACT_DIM) - 1))
        one_hot[action] = 1
        return one_hot

    def _state_vector(self) -> np.ndarray:
        if hasattr(self.game, "get_state_vector"):
            values = self.game.get_state_vector()
        elif self.mode == "human":
            values = np.zeros((self.OBS_DIM,), dtype=np.float32)
        else:
            values = TrainingSnakeGame.get_state_vector(self.game)  # type: ignore[misc]
        obs = np.asarray(values, dtype=np.float32)
        assert len(obs) == self.OBS_DIM
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Snake observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        if not np.isfinite(obs).all():
            raise RuntimeError("Snake observation contains non-finite values")
        return obs

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self.game.reset()
        self._episode_reward_components.reset()
        return self._state_vector()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.mode == "human":
            done, score = self.game.play_step()
            obs = self._state_vector()
            info: dict[str, object] = {
                "score": int(score),
                "win": False,
                "level": int(self._current_level),
                "success": 0,
            }
            if done:
                info["reward_components"] = self._episode_reward_components.totals()
            return obs, 0.0, bool(done), info

        reward, done, score = self.game.play_step(self._action_to_one_hot(int(action)))
        obs = self._state_vector()
        step_breakdown = dict(getattr(self.game, "last_reward_breakdown", {}))
        self._episode_reward_components.add_from_mapping(step_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)

        episode_level = int(self._current_level)
        info: dict[str, object] = {
            "score": int(score),
            "win": False,
            "level": int(episode_level),
            "success": 0,
            "level_changed": False,
            "reward_breakdown": step_breakdown,
        }
        if done:
            success = 1 if int(score) >= int(SUCCESS_FOODS_REQUIRED) else 0
            info["success"] = int(success)
            info["reward_components"] = self._episode_reward_components.totals()
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(success)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(success),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )
            info["level_changed"] = bool(level_changed)
        return obs, float(reward), bool(done), info

    def render(self) -> None:
        self.game.draw_frame()

    def close(self) -> None:
        self.game.close()
        self.window = None

