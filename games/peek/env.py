"""Peek stealth/navigation environment."""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_BARK,
    COLOR_LIGHT_NEUTRAL,
    COLOR_OCHRE,
    COLOR_SAND,
    COLOR_SLATE_GRAY,
    COLOR_WALNUT,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_signed, clip_unit, ordered_feature_vector
from core.primitives import draw_facing_indicator, draw_time_pie_indicator, draw_two_tone_tile, status_bar_layout
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level
from games.peek import config
from games.peek.layout import (
    Cell,
    GuardLane,
    PeekLayout,
    generate_layout,
    is_visible,
    is_walkable,
    manhattan,
)


validate_curriculum_level_settings(
    min_level=config.MIN_LEVEL,
    max_level=config.MAX_LEVEL,
    level_settings=config.LEVEL_SETTINGS,
)


def _level_progress(level: int) -> float:
    if int(config.MAX_LEVEL) <= int(config.MIN_LEVEL):
        return 0.0
    return float(int(level) - int(config.MIN_LEVEL)) / float(int(config.MAX_LEVEL) - int(config.MIN_LEVEL))


def _start_key_distance_ratio(level: int) -> float:
    progress = _level_progress(int(level))
    return float(
        config.START_KEY_DISTANCE_RATIO_MIN
        + (config.START_KEY_DISTANCE_RATIO_MAX - config.START_KEY_DISTANCE_RATIO_MIN) * progress
    )


def _min_start_key_dist(level: int) -> int:
    return max(8, int(round(float(config.MAP_PATH_SPAN_TILES) * _start_key_distance_ratio(int(level)))))


def _min_key_door_dist(level: int) -> int:
    return max(6, int(round(float(_min_start_key_dist(int(level))) * float(config.KEY_DOOR_DISTANCE_RATIO))))


def _max_steps(level: int) -> int:
    route_tiles = int(_min_start_key_dist(int(level)) + _min_key_door_dist(int(level)))
    return max(
        route_tiles + 1,
        int(round(route_tiles * float(config.STEP_BUDGET_PER_ROUTE_TILE))),
    )


def _resolve_level_settings(level: int) -> dict[str, object]:
    base = dict(config.LEVEL_SETTINGS[int(level)])
    guard_count = max(0, int(base["guard_count"]))
    return {
        **base,
        "room_count": max(int(base["room_count"]), 3 + guard_count),
        "room_size_range": config.ROOM_SIZE_RANGE,
        "max_steps": _max_steps(int(level)),
        "min_start_key_dist": _min_start_key_dist(int(level)),
        "min_key_door_dist": _min_key_door_dist(int(level)),
        "stationary_guards": bool(base.get("stationary_guards", False)),
    }


ACTION_TO_DELTA = {
    int(config.ACTION_MOVE_UP): (0, -1),
    int(config.ACTION_MOVE_DOWN): (0, 1),
    int(config.ACTION_MOVE_LEFT): (-1, 0),
    int(config.ACTION_MOVE_RIGHT): (1, 0),
    int(config.ACTION_WAIT): (0, 0),
}
ADJACENT_MEMORY_DIRECTIONS = (
    ("mem_visited_up", 0, -1),
    ("mem_visited_down", 0, 1),
    ("mem_visited_left", -1, 0),
    ("mem_visited_right", 1, 0),
)


@dataclass
class GuardActor:
    lane: GuardLane
    position: Cell
    step_dir: int
    facing_dx: int
    facing_dy: int

    @classmethod
    def from_lane(cls, lane: GuardLane) -> "GuardActor":
        return cls(
            lane=lane,
            position=lane.start,
            step_dir=0 if bool(lane.stationary) else 1,
            facing_dx=int(lane.facing_dx),
            facing_dy=int(lane.facing_dy),
        )

    def advance(self) -> None:
        if bool(self.lane.stationary):
            return
        target = self.lane.end if int(self.step_dir) > 0 else self.lane.start
        if self.position == target:
            self.step_dir *= -1
            target = self.lane.end if int(self.step_dir) > 0 else self.lane.start
        dx = int(np.sign(int(target.x) - int(self.position.x)))
        dy = int(np.sign(int(target.y) - int(self.position.y)))
        self.position = self.position.moved(dx, dy)
        self.facing_dx = int(dx if dx != 0 else self.facing_dx)
        self.facing_dy = int(dy if dy != 0 else self.facing_dy)


class PeekEnv(Env):
    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)
    REWARD_COMPONENT_ORDER = tuple(config.REWARD_COMPONENTS.keys())

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.log_ppo_metrics_line = bool(getattr(config, "PPO_METRICS_LOG_ENABLED", True))

        curriculum_config = build_curriculum_config(
            min_level=int(config.MIN_LEVEL),
            max_level=int(config.MAX_LEVEL),
            promotion_settings=config.CURRICULUM_PROMOTION,
        )
        self._curriculum = (
            ThreeLevelCurriculum(config=curriculum_config, level_settings=config.LEVEL_SETTINGS)
            if self.mode == "train"
            else None
        )
        self._current_level = (
            int(self._curriculum.get_level())
            if self._curriculum is not None
            else resolve_play_level(
                level=level,
                min_level=config.MIN_LEVEL,
                max_level=config.MAX_LEVEL,
                default_level=3,
            )
        )
        self._last_episode_level = int(self._current_level)
        self._last_episode_success = 0

        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            config.WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=False,
            vsync=False,
        )
        self._hud_text = arcade.Text(
            text="",
            x=8,
            y=max(2.0, float(config.BB_HEIGHT) * 0.5 - 6.0),
            color=COLOR_LIGHT_NEUTRAL,
            font_size=int(max(10.0, float(config.BB_HEIGHT) * 0.42)),
            font_name=("Roboto", "Arial", "sans-serif"),
            anchor_x="left",
            anchor_y="center",
        )

        self._layout: PeekLayout | None = None
        self._guards: list[GuardActor] = []
        self._episode_counter = 0
        self._settings = _resolve_level_settings(int(self._current_level))
        self.max_steps = int(self._settings.get("max_steps", 1))
        self.steps = 0
        self.done = False
        self.has_key = False
        self.player = Cell(0, 0)
        self._visit_counts = np.zeros((config.GRID_HEIGHT_TILES, config.GRID_WIDTH_TILES), dtype=np.int32)
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _apply_level_settings(self, level: int) -> None:
        self._current_level = int(level)
        self._settings = _resolve_level_settings(int(level))
        self.max_steps = int(self._settings["max_steps"])

    def get_entropy_coef_for_level(self, level: int | None = None) -> float | None:
        target_level = int(self._current_level if level is None else level)
        return float(_resolve_level_settings(int(target_level))["entropy_coef"])

    def _episode_seed(self) -> int:
        return int(config.BASE_SEED + self._episode_counter * 1_003 + self._current_level * 53)

    @staticmethod
    def _encode_current_tile_revisit(count: int) -> float:
        if int(count) <= 1:
            return 0.0
        if int(count) <= 3:
            return 0.5
        return 1.0

    @staticmethod
    def _encode_adjacent_tile_memory(count: int) -> float:
        if int(count) <= 0:
            return 0.0
        if int(count) <= 2:
            return 0.5
        return 1.0

    def _visit_count(self, cell: Cell) -> int:
        if self._layout is None or (not is_walkable(self._layout.walkable, cell)):
            return 0
        return int(self._visit_counts[int(cell.y), int(cell.x)])

    def _record_current_tile_visit(self) -> int:
        if self._layout is None or (not is_walkable(self._layout.walkable, self.player)):
            return 0
        self._visit_counts[int(self.player.y), int(self.player.x)] += 1
        return int(self._visit_counts[int(self.player.y), int(self.player.x)])

    def _door_distance(self, cell: Cell) -> int:
        if self._layout is None or (not is_walkable(self._layout.walkable, cell)):
            return -1
        return int(self._layout.door_distance_map[int(cell.y), int(cell.x)])

    def _progress_reward_before_key(self, *, first_visit_this_step: bool) -> float:
        if not bool(first_visit_this_step):
            return 0.0
        return float(config.REWARD_PROGRESS_BEFORE_KEY_FIRST_VISIT)

    def _progress_reward_after_key(self, previous_player: Cell) -> float:
        previous_distance = self._door_distance(previous_player)
        current_distance = self._door_distance(self.player)
        if previous_distance < 0 or current_distance < 0:
            return 0.0
        distance_delta = int(previous_distance - current_distance)
        clipped_delta = float(
            np.clip(
                float(distance_delta),
                -float(config.REWARD_PROGRESS_AFTER_KEY_DOOR_DELTA_CLIP),
                float(config.REWARD_PROGRESS_AFTER_KEY_DOOR_DELTA_CLIP),
            )
        )
        return float(config.REWARD_PROGRESS_AFTER_KEY_DOOR_SCALE) * clipped_delta

    def _progress_reward(
        self,
        *,
        had_key_before_step: bool,
        previous_player: Cell,
        first_visit_this_step: bool,
    ) -> float:
        if bool(had_key_before_step):
            return float(self._progress_reward_after_key(previous_player))
        return float(self._progress_reward_before_key(first_visit_this_step=first_visit_this_step))

    def _adjacent_visit_memory(self, dx: int, dy: int) -> float:
        if self._layout is None:
            return 0.0
        candidate = self.player.moved(dx, dy)
        if not is_walkable(self._layout.walkable, candidate):
            return 0.0
        return float(self._encode_adjacent_tile_memory(self._visit_count(candidate)))

    @staticmethod
    def _add_reward_component(reward_breakdown: dict[str, float], code: str, value: float) -> None:
        if abs(float(value)) <= 1e-12:
            return
        reward_breakdown[str(code)] = float(reward_breakdown.get(str(code), 0.0) + float(value))

    def _apply_reward(self, reward_breakdown: dict[str, float], code: str, value: float) -> float:
        if self.mode == "human":
            return 0.0
        reward_value = float(value)
        self._add_reward_component(reward_breakdown, str(code), reward_value)
        return reward_value

    def _move_player(self, action_idx: int) -> bool:
        dx, dy = ACTION_TO_DELTA.get(int(action_idx), (0, 0))
        if int(dx) == 0 and int(dy) == 0:
            return False
        candidate = self.player.moved(dx, dy)
        if is_walkable(self._layout.walkable, candidate):
            self.player = candidate
            return False
        return True

    def _human_action(self) -> int:
        if self.window_controller.is_key_down(arcade.key.W):
            return int(config.ACTION_MOVE_UP)
        if self.window_controller.is_key_down(arcade.key.S):
            return int(config.ACTION_MOVE_DOWN)
        if self.window_controller.is_key_down(arcade.key.A):
            return int(config.ACTION_MOVE_LEFT)
        if self.window_controller.is_key_down(arcade.key.D):
            return int(config.ACTION_MOVE_RIGHT)
        if self.window_controller.is_key_down(arcade.key.SPACE):
            return int(config.ACTION_WAIT)
        return int(config.ACTION_WAIT)

    def _guard_vision_cells(self, guard: GuardActor) -> list[Cell]:
        cells: list[Cell] = []
        for step in range(1, int(config.GUARD_VISION_RANGE) + 1):
            cell = guard.position.moved(int(guard.facing_dx) * step, int(guard.facing_dy) * step)
            if not is_walkable(self._layout.walkable, cell):
                break
            cells.append(cell)
        return cells

    def _is_caught(self) -> bool:
        for guard in self._guards:
            if guard.position == self.player:
                return True
            if self.player in self._guard_vision_cells(guard):
                return True
        return False

    def _visible_guards(self) -> list[GuardActor]:
        guards = [
            guard
            for guard in self._guards
            if is_visible(self._layout.walkable, self.player, guard.position, int(config.VISIBILITY_RANGE))
        ]
        guards.sort(key=lambda guard: (manhattan(self.player, guard.position), guard.lane.guard_id))
        return guards

    def _wall_ray(self, dx: int, dy: int) -> float:
        for step in range(1, int(config.RAY_RANGE) + 1):
            cell = self.player.moved(int(dx) * step, int(dy) * step)
            if not is_walkable(self._layout.walkable, cell):
                return float(clip_unit(float(step - 1) / float(max(1, int(config.RAY_RANGE)))))
        return 1.0

    def _rel_xy(self, cell: Cell) -> tuple[float, float]:
        scale = float(max(1, int(config.VISIBILITY_RANGE)))
        dx = clip_signed(float(int(cell.x) - int(self.player.x)) / scale)
        dy = clip_signed(float(int(cell.y) - int(self.player.y)) / scale)
        return float(dx), float(dy)

    def _build_observation(self) -> np.ndarray:
        obj_dx = 0.0
        obj_dy = 0.0
        obj_type = 0.0
        object_priority = (
            ((int(config.OBJECT_KEY), self._layout.key), (int(config.OBJECT_DOOR), self._layout.door))
            if not self.has_key
            else ((int(config.OBJECT_DOOR), self._layout.door), (int(config.OBJECT_KEY), self._layout.key))
        )
        for obj_code, obj_cell in object_priority:
            if int(obj_code) == int(config.OBJECT_KEY) and self.has_key:
                continue
            if is_visible(self._layout.walkable, self.player, obj_cell, int(config.VISIBILITY_RANGE)):
                obj_dx, obj_dy = self._rel_xy(obj_cell)
                obj_type = float(obj_code)
                break

        opp_dx = 0.0
        opp_dy = 0.0
        opp_facing_dx = 0.0
        opp_facing_dy = 0.0
        visible_guards = self._visible_guards()
        if visible_guards:
            visible_guards.sort(
                key=lambda guard: (
                    0 if self.player in self._guard_vision_cells(guard) else 1,
                    manhattan(self.player, guard.position),
                    guard.lane.guard_id,
                )
            )
            guard = visible_guards[0]
            opp_dx, opp_dy = self._rel_xy(guard.position)
            opp_facing_dx = float(int(guard.facing_dx))
            opp_facing_dy = float(int(guard.facing_dy))

        feature_values = {
            "self_has_key": 1.0 if self.has_key else 0.0,
            "self_time_left": float(clip_unit(float(self.max_steps - self.steps) / float(max(1, self.max_steps)))),
            "self_here_revisited": float(self._encode_current_tile_revisit(self._visit_count(self.player))),
            "ray_wall_up": float(self._wall_ray(0, -1)),
            "ray_wall_down": float(self._wall_ray(0, 1)),
            "ray_wall_left": float(self._wall_ray(-1, 0)),
            "ray_wall_right": float(self._wall_ray(1, 0)),
            "obj1_dx": float(obj_dx),
            "obj1_dy": float(obj_dy),
            "obj1_type": float(obj_type),
            "opp1_dx": float(opp_dx),
            "opp1_dy": float(opp_dy),
            "opp1_facing_dx": float(opp_facing_dx),
            "opp1_facing_dy": float(opp_facing_dy),
        }
        for feature_name, dx, dy in ADJACENT_MEMORY_DIRECTIONS:
            feature_values[str(feature_name)] = float(self._adjacent_visit_memory(int(dx), int(dy)))
        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Peek observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self._layout = generate_layout(
            seed=self._episode_seed(),
            layout_attempts=int(config.LAYOUT_ATTEMPTS),
            rows=int(config.GRID_HEIGHT_TILES),
            cols=int(config.GRID_WIDTH_TILES),
            room_count=int(self._settings["room_count"]),
            room_size_range=tuple(self._settings["room_size_range"]),
            room_place_attempts=int(config.ROOM_PLACE_ATTEMPTS),
            extra_connection=bool(self._settings["extra_connection"]),
            guard_count=int(self._settings["guard_count"]),
            guard_vision_range=int(config.GUARD_VISION_RANGE),
            guard_move_period=int(config.GUARD_MOVE_PERIOD),
            min_start_key_dist=int(self._settings["min_start_key_dist"]),
            min_key_door_dist=int(self._settings["min_key_door_dist"]),
            stationary_guards=bool(self._settings["stationary_guards"]),
        )
        self._episode_counter += 1
        self._guards = [GuardActor.from_lane(lane) for lane in self._layout.guards]
        self.player = self._layout.start
        self.has_key = False
        self.steps = 0
        self.done = False
        self._visit_counts = np.zeros(self._layout.walkable.shape, dtype=np.int32)
        self._episode_reward_components.reset()
        self._record_current_tile_visit()
        self._last_obs = self._build_observation()
        return np.asarray(self._last_obs, dtype=np.float32)

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return np.asarray(self._last_obs, dtype=np.float32), 0.0, True, {
                "win": bool(self._last_episode_success > 0),
                "success": int(self._last_episode_success),
                "level": int(self._last_episode_level),
                "reward_components": self._episode_reward_components.totals(),
            }

        self.window_controller.poll_events_or_raise()

        episode_level = int(self._current_level)
        action_idx = int(self._human_action() if self.mode == "human" else action)
        action_idx = int(np.clip(action_idx, 0, int(self.ACT_DIM) - 1))
        had_key_before_step = bool(self.has_key)
        previous_player = self.player
        blocked_move = self._move_player(action_idx)
        self.steps += 1
        current_tile_visit_count = self._record_current_tile_visit()
        first_visit_this_step = int(current_tile_visit_count) == 1

        reward = 0.0
        reward_breakdown: dict[str, float] = {}
        if blocked_move:
            reward += self._apply_reward(reward_breakdown, "B", float(config.PENALTY_BLOCKED_MOVE))
        if int(action_idx) == int(config.ACTION_WAIT):
            reward += self._apply_reward(reward_breakdown, "I", float(config.PENALTY_WAIT))

        if (not self.has_key) and self.player == self._layout.key:
            self.has_key = True
            reward += self._apply_reward(reward_breakdown, "K", float(config.REWARD_KEY))

        progress_reward = self._progress_reward(
            had_key_before_step=bool(had_key_before_step),
            previous_player=previous_player,
            first_visit_this_step=bool(first_visit_this_step),
        )
        reward += self._apply_reward(reward_breakdown, "P", float(progress_reward))

        caught = False
        timed_out = False
        success = 0
        if self.has_key and self.player == self._layout.door:
            self.done = True
            success = 1
            reward += self._apply_reward(reward_breakdown, "W", float(config.REWARD_WIN))
        else:
            if self._is_caught():
                caught = True
            if caught or self.steps >= int(self.max_steps):
                timed_out = bool((not caught) and self.steps >= int(self.max_steps))
                self.done = True
                reward += self._apply_reward(reward_breakdown, "L", float(config.PENALTY_LOSE))
            elif self.steps % int(config.GUARD_MOVE_PERIOD) == 0:
                for guard in self._guards:
                    guard.advance()

        if self.mode != "human":
            for code, value in reward_breakdown.items():
                self._episode_reward_components.add(str(code), float(value))

        level_changed = False
        if self.done:
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(success)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(success),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )

        self._last_obs = self._build_observation()
        self.render()
        self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)

        info: dict[str, object] = {
            "win": bool(success > 0) if self.done else False,
            "success": int(success) if self.done else 0,
            "level": int(episode_level),
            "level_changed": bool(level_changed),
            "caught": bool(caught),
            "timeout": bool(timed_out),
            "has_key": bool(self.has_key),
            "reward_breakdown": reward_breakdown if self.mode != "human" else {},
        }
        if self.done:
            info["reward_components"] = self._episode_reward_components.totals()
        return np.asarray(self._last_obs, dtype=np.float32), float(reward), bool(self.done), info

    def _draw_cell(self, cell: Cell, outer_color, inner_color) -> None:
        draw_two_tone_tile(
            self.window_controller,
            top_left_x=float(int(cell.x) * int(config.TILE_SIZE)),
            top_left_y=float(int(cell.y) * int(config.TILE_SIZE)),
            size=float(config.TILE_SIZE),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(config.CELL_INSET),
        )

    def _draw_world(self) -> None:
        for y in range(int(config.GRID_HEIGHT_TILES)):
            for x in range(int(config.GRID_WIDTH_TILES)):
                cell = Cell(x, y)
                if is_walkable(self._layout.walkable, cell):
                    continue
                self._draw_cell(cell, COLOR_DARK_NEUTRAL, COLOR_SLATE_GRAY)
        if not self.has_key:
            self._draw_cell(self._layout.key, COLOR_SAND, COLOR_OCHRE)
        self._draw_cell(self._layout.door, COLOR_WALNUT, COLOR_BARK)
        if bool(config.DRAW_GUARD_VISION):
            for guard in self._guards:
                for cell in self._guard_vision_cells(guard):
                    bottom = self.window_controller.top_left_to_bottom(float(cell.y * config.TILE_SIZE), float(config.TILE_SIZE))
                    arcade.draw_lbwh_rectangle_filled(
                        float(cell.x * config.TILE_SIZE),
                        bottom,
                        float(config.TILE_SIZE),
                        float(config.TILE_SIZE),
                        COLOR_DARK_NEUTRAL + (56,),
                    )
        for guard in self._guards:
            self._draw_cell(guard.position, COLOR_SLATE_GRAY, COLOR_DARK_NEUTRAL)
            draw_facing_indicator(
                self.window_controller,
                center_x=float(guard.position.x * config.TILE_SIZE + config.TILE_SIZE / 2),
                center_y_top_left=float(guard.position.y * config.TILE_SIZE + config.TILE_SIZE / 2),
                angle_degrees=float(np.degrees(np.arctan2(float(guard.facing_dy), float(guard.facing_dx)))),
                length=float(config.TILE_SIZE * 0.35),
                color=COLOR_LIGHT_NEUTRAL,
                line_width=2.0,
            )
        self._draw_cell(self.player, COLOR_AQUA, COLOR_DEEP_TEAL)

    def _draw_hud(self) -> None:
        arcade.draw_lbwh_rectangle_filled(0, 0, config.SCREEN_WIDTH, config.BB_HEIGHT, COLOR_DARK_NEUTRAL)
        layout = status_bar_layout(
            width=float(config.SCREEN_WIDTH),
            bottom_bar_height=float(config.BB_HEIGHT),
            tile_size=float(config.TILE_SIZE),
            cell_inset=float(config.CELL_INSET),
            include_clock=True,
        )
        if layout.clock_center_x is not None:
            draw_time_pie_indicator(
                center_x=float(layout.clock_center_x),
                center_y=float(layout.center_y),
                radius=float(layout.clock_radius),
                border_width=float(layout.clock_border_width),
                remaining_ratio=float(clip_unit(float(self.max_steps - self.steps) / float(max(1, self.max_steps)))),
                base_color=COLOR_SLATE_GRAY,
                fill_color=COLOR_FOG_GRAY,
                outline_color=COLOR_FOG_GRAY,
            )
        status = "RUN"
        if self.done:
            status = "WIN" if self._last_episode_success > 0 else "FAIL"
        self._hud_text.text = (
            f"Lv:{int(self._current_level)}  "
            f"Step:{int(self.steps):>3}/{int(self.max_steps):<3}  "
            f"Key:{1 if self.has_key else 0}  "
            f"Guards:{len(self._guards)}  "
            f"{status}"
        )
        self._hud_text.draw()

    def render(self) -> None:
        if self.window_controller.window is None or self._layout is None:
            return
        self.window_controller.clear(COLOR_LIGHT_NEUTRAL)
        arcade.draw_lbwh_rectangle_filled(
            0,
            float(config.BB_HEIGHT),
            float(config.SCREEN_WIDTH),
            float(config.SCREEN_HEIGHT - config.BB_HEIGHT),
            COLOR_FOG_GRAY,
        )
        self._draw_world()
        self._draw_hud()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
