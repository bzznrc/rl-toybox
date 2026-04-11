"""Frogger environment."""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BLUE,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_FOREST_GREEN,
    COLOR_LEAF_GREEN,
    COLOR_LIGHT_NEUTRAL,
    COLOR_NAVY,
    COLOR_OCHRE,
    COLOR_SAND,
    COLOR_SLATE_GRAY,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_unit
from core.match_tracker import compact_count_to_icons
from core.primitives import (
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    status_icon_inset,
    draw_status_square_icon,
    draw_two_tone_tile,
    status_icon_size,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level
from games.frogger import config


validate_curriculum_level_settings(
    min_level=config.MIN_LEVEL,
    max_level=config.MAX_LEVEL,
    level_settings=config.LEVEL_SETTINGS,
)


ACTION_TO_DELTA = {
    int(config.ACTION_UP): (0, -1),
    int(config.ACTION_DOWN): (0, 1),
    int(config.ACTION_LEFT): (-1, 0),
    int(config.ACTION_RIGHT): (1, 0),
    int(config.ACTION_WAIT): (0, 0),
}

SAFE_ROW_OUTER = COLOR_LEAF_GREEN
SAFE_ROW_INNER = COLOR_FOREST_GREEN
ROAD_ROW_OUTER = COLOR_SLATE_GRAY
ROAD_ROW_INNER = COLOR_DARK_NEUTRAL
GOAL_ROW_OUTER = COLOR_SAND
GOAL_ROW_INNER = COLOR_OCHRE
LANE_DASH_COLOR = COLOR_FOG_GRAY
BOARD_OUTLINE_COLOR = COLOR_FOG_GRAY
CAR_STYLE_DEFS = (
    {"name": "slow", "outer": COLOR_CORAL, "inner": COLOR_BRICK_RED, "speed": 0.36},
    {"name": "medium", "outer": COLOR_BLUE, "inner": COLOR_NAVY, "speed": 0.56},
    {"name": "fast", "outer": COLOR_SAND, "inner": COLOR_OCHRE, "speed": 0.78},
)


@dataclass
class LaneTraffic:
    row: int
    direction: int
    speed: float
    style_name: str
    outer_color: tuple[int, int, int]
    inner_color: tuple[int, int, int]
    car_spacing: float
    car_left_positions: list[float]

    @staticmethod
    def _fully_inside(left: float, *, width_tiles: int, car_length_tiles: float) -> bool:
        return 0.0 <= float(left) <= float(width_tiles) - float(car_length_tiles)

    def _project_positions(self, *, width_tiles: int, car_length_tiles: float) -> list[float]:
        moved = [float(position) + float(self.direction) * float(self.speed) for position in self.car_left_positions]
        if not moved:
            return moved

        if int(self.direction) > 0:
            current_min = min(moved)
            for idx in sorted(range(len(moved)), key=lambda index: moved[index]):
                if float(moved[idx]) > float(width_tiles):
                    current_min -= float(self.car_spacing)
                    moved[idx] = float(current_min)
                else:
                    current_min = min(float(current_min), float(moved[idx]))
            return moved

        current_max = max(moved)
        for idx in sorted(range(len(moved)), key=lambda index: moved[index], reverse=True):
            if float(moved[idx]) + float(car_length_tiles) < 0.0:
                current_max += float(self.car_spacing)
                moved[idx] = float(current_max)
            else:
                current_max = max(float(current_max), float(moved[idx]))
        return moved

    def advance(self, *, width_tiles: int, car_length_tiles: float) -> None:
        self.car_left_positions = self._project_positions(
            width_tiles=int(width_tiles),
            car_length_tiles=float(car_length_tiles),
        )

    def occupies_cell(self, x: int, *, width_tiles: int, car_length_tiles: float) -> bool:
        cell_left = float(int(x))
        cell_right = cell_left + 1.0
        for left in self.car_left_positions:
            if not self._fully_inside(float(left), width_tiles=int(width_tiles), car_length_tiles=float(car_length_tiles)):
                continue
            right = float(left) + float(car_length_tiles)
            if float(left) < cell_right and right > cell_left:
                return True
        return False

    def will_occupy_cell_next(self, x: int, *, width_tiles: int, car_length_tiles: float) -> bool:
        cell_left = float(int(x))
        cell_right = cell_left + 1.0
        for left in self._project_positions(width_tiles=int(width_tiles), car_length_tiles=float(car_length_tiles)):
            if not self._fully_inside(float(left), width_tiles=int(width_tiles), car_length_tiles=float(car_length_tiles)):
                continue
            right = float(left) + float(car_length_tiles)
            if float(left) < cell_right and right > cell_left:
                return True
        return False


class FroggerEnv(Env):
    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)
    REWARD_COMPONENT_ORDER = tuple(config.REWARD_COMPONENT_NAMES)

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

        self._lane_count = 1
        self._road_rows: tuple[int, ...] = ()
        self._start_row = 0
        self._goal_row = 0
        self.max_steps = 1
        self._board_rows = 0
        self._board_width_px = int(config.BOARD_WIDTH_TILES * config.COLUMN_WIDTH_PX)
        self._board_height_px = 0
        self._board_offset_x = int((config.WORLD_WIDTH - self._board_width_px) // 2)
        self._board_offset_y = 0
        self._row_band_height_px = max(1.0, float(config.ROW_PITCH_PX) * float(config.ROW_BAND_HEIGHT_RATIO))

        self._episode_counter = 0
        self._crossing_counter = 0
        self.steps = 0
        self.score = 0
        self.done = False
        self.frog_x = 0
        self.frog_y = 0
        self._best_row_reached = 0
        self._status_label = "Run"
        self._lanes: list[LaneTraffic] = []
        self._lane_by_row: dict[int, LaneTraffic] = {}
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _apply_level_settings(self, level: int) -> None:
        self._current_level = int(level)
        settings = dict(config.LEVEL_SETTINGS[int(level)])
        self._lane_count = int(settings["lane_count"])
        self._road_rows = tuple(range(1, self._lane_count + 1))
        self._goal_row = 0
        self._start_row = int(self._lane_count + 1)
        self._board_rows = int(self._lane_count + 2)
        self._board_height_px = int(self._board_rows * config.ROW_PITCH_PX)
        self._board_offset_y = int((config.WORLD_HEIGHT - self._board_height_px) // 2)
        self.max_steps = int(settings["max_steps"])

    def get_entropy_coef_for_level(self, level: int | None = None) -> float | None:
        target_level = int(self._current_level if level is None else level)
        return float(config.LEVEL_SETTINGS[int(target_level)]["entropy_coef"])

    def _crossing_seed(self) -> int:
        return int(
            config.BASE_SEED
            + self._episode_counter * 1009
            + self._current_level * 53
            + self._crossing_counter * 197
        )

    def _lane_for_row(self, row: int) -> LaneTraffic | None:
        return self._lane_by_row.get(int(row))

    def _generate_lanes(self) -> list[LaneTraffic]:
        settings = dict(config.LEVEL_SETTINGS[int(self._current_level)])
        style_indices = tuple(int(value) for value in settings["style_indices"])
        car_count_choices = tuple(int(value) for value in settings["car_count_choices"])
        rng = np.random.default_rng(self._crossing_seed())

        directions = rng.choice(np.asarray([-1, 1], dtype=np.int32), size=int(self._lane_count), replace=True)
        if int(self._lane_count) > 1 and int(np.unique(directions).size) == 1:
            flip_index = int(rng.integers(0, int(self._lane_count)))
            directions[flip_index] *= -1

        lanes: list[LaneTraffic] = []
        width_tiles = int(config.BOARD_WIDTH_TILES)
        car_length_tiles = float(config.CAR_LENGTH_TILES)
        for lane_index, row in enumerate(self._road_rows):
            style_index = int(rng.choice(np.asarray(style_indices, dtype=np.int32)))
            style = CAR_STYLE_DEFS[int(style_index) % len(CAR_STYLE_DEFS)]
            car_count = int(rng.choice(np.asarray(car_count_choices, dtype=np.int32)))
            car_spacing = max(
                float(car_length_tiles) + float(config.MIN_CAR_GAP_TILES),
                float(width_tiles) / float(max(1, car_count)),
            )
            start_span = max(0.0, float(width_tiles) - float(car_length_tiles) - float((car_count - 1) * car_spacing))
            phase = float(rng.uniform(0.5, max(0.75, start_span + 0.5)))
            left_positions = [float(phase + car_idx * car_spacing) for car_idx in range(int(car_count))]
            lanes.append(
                LaneTraffic(
                    row=int(row),
                    direction=int(directions[int(lane_index)]),
                    speed=float(style["speed"]),
                    style_name=str(style["name"]),
                    outer_color=style["outer"],
                    inner_color=style["inner"],
                    car_spacing=float(car_spacing),
                    car_left_positions=left_positions,
                )
            )
        return lanes

    def _start_crossing(self, *, preserve_score: bool) -> None:
        if not preserve_score:
            self.score = 0
        self._lanes = self._generate_lanes()
        self._lane_by_row = {int(lane.row): lane for lane in self._lanes}
        self.frog_x = int(config.BOARD_WIDTH_TILES // 2)
        self.frog_y = int(self._start_row)
        self._best_row_reached = int(self._start_row)
        self._crossing_counter += 1

    def _human_action(self) -> int:
        if self.window_controller.is_key_down(arcade.key.UP) or self.window_controller.is_key_down(arcade.key.W):
            return int(config.ACTION_UP)
        if self.window_controller.is_key_down(arcade.key.DOWN) or self.window_controller.is_key_down(arcade.key.S):
            return int(config.ACTION_DOWN)
        if self.window_controller.is_key_down(arcade.key.LEFT) or self.window_controller.is_key_down(arcade.key.A):
            return int(config.ACTION_LEFT)
        if self.window_controller.is_key_down(arcade.key.RIGHT) or self.window_controller.is_key_down(arcade.key.D):
            return int(config.ACTION_RIGHT)
        if self.window_controller.is_key_down(arcade.key.SPACE):
            return int(config.ACTION_WAIT)
        return int(config.ACTION_WAIT)

    def _is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= int(x) < int(config.BOARD_WIDTH_TILES) and 0 <= int(y) < int(self._board_rows)

    def _row_base_token(self, row: int) -> float:
        if int(row) == int(self._goal_row):
            return float(config.TOKEN_GOAL)
        if int(row) == int(self._start_row):
            return float(config.TOKEN_SAFE)
        if int(row) in self._road_rows:
            return float(config.TOKEN_ROAD)
        return float(config.TOKEN_BOUNDARY)

    def _cell_token(self, x: int, y: int) -> float:
        if not self._is_in_bounds(int(x), int(y)):
            return float(config.TOKEN_BOUNDARY)
        lane = self._lane_for_row(int(y))
        if lane is not None and lane.occupies_cell(
            int(x),
            width_tiles=int(config.BOARD_WIDTH_TILES),
            car_length_tiles=float(config.CAR_LENGTH_TILES),
        ):
            return float(config.TOKEN_CAR)
        return float(self._row_base_token(int(y)))

    def _local_patch_tokens(self) -> list[float]:
        tokens: list[float] = []
        for dy in range(-int(config.PATCH_RADIUS), int(config.PATCH_RADIUS) + 1):
            for dx in range(-int(config.PATCH_RADIUS), int(config.PATCH_RADIUS) + 1):
                tokens.append(float(self._cell_token(int(self.frog_x + dx), int(self.frog_y + dy))))
        return tokens

    def _frog_lane_id_norm(self) -> float:
        lane = self._lane_for_row(int(self.frog_y))
        if lane is None:
            return 0.0
        lane_id = max(1, int(lane.row))
        return float(clip_unit(float(lane_id) / float(max(1, self._lane_count))))

    def _goal_dy_norm(self) -> float:
        return float(clip_unit(float(self.frog_y - self._goal_row) / float(max(1, self._start_row - self._goal_row))))

    def _flag_danger_now(self) -> float:
        lane = self._lane_for_row(int(self.frog_y))
        if lane is None:
            return 0.0
        is_danger = lane.occupies_cell(
            int(self.frog_x),
            width_tiles=int(config.BOARD_WIDTH_TILES),
            car_length_tiles=float(config.CAR_LENGTH_TILES),
        ) or lane.will_occupy_cell_next(
            int(self.frog_x),
            width_tiles=int(config.BOARD_WIDTH_TILES),
            car_length_tiles=float(config.CAR_LENGTH_TILES),
        )
        return 1.0 if bool(is_danger) else 0.0

    def _lane_dir_here(self) -> float:
        lane = self._lane_for_row(int(self.frog_y))
        if lane is None:
            return 0.0
        return float(np.sign(int(lane.direction)))

    def _lane_speed_here_norm(self) -> float:
        lane = self._lane_for_row(int(self.frog_y))
        if lane is None:
            return 0.0
        return float(clip_unit(float(lane.speed) / float(max(1e-6, config.MAX_LANE_SPEED))))

    def _build_observation(self) -> np.ndarray:
        values = [
            *self._local_patch_tokens(),
            float(clip_unit(float(self.max_steps - self.steps) / float(max(1, self.max_steps)))),
            float(self._frog_lane_id_norm()),
            float(clip_unit(float(self.frog_x) / float(max(1, config.BOARD_WIDTH_TILES - 1)))),
            float(self._goal_dy_norm()),
            float(self._lane_dir_here()),
            float(self._lane_speed_here_norm()),
            float(self._flag_danger_now()),
        ]
        obs = np.asarray(values, dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Frogger observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def _car_hits_frog(self) -> bool:
        lane = self._lane_for_row(int(self.frog_y))
        if lane is None:
            return False
        return bool(
            lane.occupies_cell(
                int(self.frog_x),
                width_tiles=int(config.BOARD_WIDTH_TILES),
                car_length_tiles=float(config.CAR_LENGTH_TILES),
            )
        )

    def _advance_traffic(self) -> None:
        for lane in self._lanes:
            lane.advance(
                width_tiles=int(config.BOARD_WIDTH_TILES),
                car_length_tiles=float(config.CAR_LENGTH_TILES),
            )

    def _move_frog(self, action_idx: int) -> None:
        dx, dy = ACTION_TO_DELTA.get(int(action_idx), (0, 0))
        next_x = int(np.clip(int(self.frog_x) + int(dx), 0, int(config.BOARD_WIDTH_TILES) - 1))
        next_y = int(np.clip(int(self.frog_y) + int(dy), 0, int(self._board_rows) - 1))
        self.frog_x = int(next_x)
        self.frog_y = int(next_y)

    @staticmethod
    def _add_reward_component(reward_breakdown: dict[str, float], key: str, value: float) -> None:
        if abs(float(value)) <= 1e-12:
            return
        reward_breakdown[str(key)] = float(reward_breakdown.get(str(key), 0.0) + float(value))

    def _apply_reward(self, reward_breakdown: dict[str, float], key: str, value: float) -> float:
        if self.mode == "human":
            return 0.0
        reward_value = float(value)
        self._add_reward_component(reward_breakdown, str(key), reward_value)
        return reward_value

    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {str(name): 0.0 for name in self.REWARD_COMPONENT_ORDER}

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self._episode_counter += 1
        self._crossing_counter = 0
        self.steps = 0
        self.score = 0
        self.done = False
        self._status_label = "Run"
        self._episode_reward_components.reset()
        self._start_crossing(preserve_score=False)
        self._last_obs = self._build_observation()
        self.render()
        return np.asarray(self._last_obs, dtype=np.float32)

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return np.asarray(self._last_obs, dtype=np.float32), 0.0, True, {
                "win": bool(self._last_episode_success > 0),
                "success": int(self._last_episode_success),
                "level": int(self._last_episode_level),
                "lane_count": int(self._lane_count),
                "score": int(self.score),
                "reward_components": self._episode_reward_components.totals(),
            }

        self.window_controller.poll_events_or_raise()

        episode_level = int(self._current_level)
        episode_lane_count = int(self._lane_count)
        action_idx = int(self._human_action() if self.mode == "human" else action)
        action_idx = int(np.clip(action_idx, 0, int(self.ACT_DIM) - 1))

        self._status_label = "Run"
        self._move_frog(int(action_idx))
        self.steps += 1

        reward = 0.0
        reward_breakdown = self._empty_reward_breakdown()
        reward += self._apply_reward(reward_breakdown, "reward_cost_step", float(config.REWARD_COST_STEP))

        if int(self.frog_y) < int(self._best_row_reached):
            self._best_row_reached = int(self.frog_y)
            reward += self._apply_reward(
                reward_breakdown,
                "reward_progress_forward",
                float(config.REWARD_PROGRESS_FORWARD),
            )

        hit = bool(self._car_hits_frog())
        timed_out = False
        crossing_scored = False

        if not hit and int(self.frog_y) == int(self._goal_row):
            crossing_scored = True
            self.score += 1
            reward += self._apply_reward(reward_breakdown, "reward_event_goal", float(config.REWARD_EVENT_GOAL))
            reward += self._apply_reward(reward_breakdown, "reward_terminal_win", float(config.REWARD_TERMINAL_WIN))
            self._start_crossing(preserve_score=True)
            self._status_label = "Chain"
        elif hit:
            self.done = True
            reward += self._apply_reward(reward_breakdown, "reward_event_hit", float(config.REWARD_EVENT_HIT))
            reward += self._apply_reward(
                reward_breakdown,
                "reward_terminal_loss",
                float(config.REWARD_TERMINAL_LOSS),
            )
            self._status_label = "Hit"

        if (not self.done) and (not crossing_scored):
            self._advance_traffic()
            hit = bool(self._car_hits_frog())
            if hit:
                self.done = True
                reward += self._apply_reward(reward_breakdown, "reward_event_hit", float(config.REWARD_EVENT_HIT))
                reward += self._apply_reward(
                    reward_breakdown,
                    "reward_terminal_loss",
                    float(config.REWARD_TERMINAL_LOSS),
                )
                self._status_label = "Hit"

        if not self.done and self.steps >= int(self.max_steps):
            timed_out = True
            self.done = True
            reward += self._apply_reward(
                reward_breakdown,
                "reward_terminal_loss",
                float(config.REWARD_TERMINAL_LOSS),
            )
            self._status_label = "Timeout"

        if self.mode != "human":
            for key, value in reward_breakdown.items():
                self._episode_reward_components.add(str(key), float(value))

        run_success = 1 if int(self.score) > 0 else 0
        level_changed = False
        if self.done:
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(run_success)
            next_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(run_success),
                current_level=int(self._current_level),
                apply_level=None,
            )
            self._current_level = int(next_level)

        self._last_obs = self._build_observation()
        self.render()
        self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)

        info: dict[str, object] = {
            "win": bool(run_success > 0) if self.done else False,
            "success": int(run_success) if self.done else 0,
            "level": int(episode_level),
            "level_changed": bool(level_changed),
            "hit": bool(hit),
            "timeout": bool(timed_out),
            "lane_count": int(episode_lane_count),
            "score": int(self.score),
            "crossing_scored": bool(crossing_scored),
            "reward_breakdown": reward_breakdown if self.mode != "human" else {},
        }
        if self.done:
            info["reward_components"] = self._episode_reward_components.totals()

        reward_out = 0.0 if self.mode == "human" else float(reward)
        return np.asarray(self._last_obs, dtype=np.float32), float(reward_out), bool(self.done), info

    def _draw_rect_top_left(self, *, left: float, top: float, width: float, height: float, color) -> None:
        bottom = self.window_controller.top_left_to_bottom(float(top), float(height))
        arcade.draw_lbwh_rectangle_filled(float(left), float(bottom), float(width), float(height), color)

    def _column_left(self, column: int) -> float:
        return float(self._board_offset_x + int(column) * int(config.COLUMN_WIDTH_PX))

    def _row_top(self, row: int) -> float:
        return float(self._board_offset_y + int(row) * int(config.ROW_PITCH_PX))

    def _row_band_top(self, row: int) -> float:
        return float(self._row_top(int(row)) + (float(config.ROW_PITCH_PX) - self._row_band_height_px) * 0.5)

    def _draw_row_band(self, row: int, *, outer_color, inner_color) -> None:
        top = float(self._row_band_top(int(row)))
        left = float(self._board_offset_x)
        width = float(self._board_width_px)
        height = float(self._row_band_height_px)
        inset = float(max(2, int(config.CELL_INSET)))
        self._draw_rect_top_left(left=left, top=top, width=width, height=height, color=outer_color)
        self._draw_rect_top_left(
            left=left + inset,
            top=top + inset,
            width=max(1.0, width - 2.0 * inset),
            height=max(1.0, height - 2.0 * inset),
            color=inner_color,
        )

    def _draw_goal_markers(self, row: int) -> None:
        marker_height = max(3.0, float(self._row_band_height_px) * float(config.GOAL_MARKER_HEIGHT_RATIO))
        top = float(self._row_band_top(int(row)) + float(self._row_band_height_px) - marker_height - 4.0)
        left = float(self._board_offset_x) + 8.0
        width = float(self._board_width_px) - 16.0
        self._draw_rect_top_left(
            left=float(left),
            top=float(top),
            width=float(width),
            height=float(marker_height),
            color=COLOR_FOG_GRAY,
        )

    def _draw_lane_dashes(self, row: int) -> None:
        dash_width = float(config.COLUMN_WIDTH_PX) * float(config.LANE_DASH_WIDTH_RATIO)
        dash_height = max(2.0, float(self._row_band_height_px) * float(config.LANE_DASH_HEIGHT_RATIO))
        gap = float(config.COLUMN_WIDTH_PX) * float(config.LANE_DASH_GAP_RATIO)
        top = float(self._row_band_top(int(row)) + (float(self._row_band_height_px) - dash_height) * 0.5)
        left = float(self._board_offset_x) + gap
        right = float(self._board_offset_x + self._board_width_px)
        step = dash_width + gap
        while left < right - gap:
            self._draw_rect_top_left(
                left=float(left),
                top=float(top),
                width=min(float(dash_width), right - gap - left),
                height=float(dash_height),
                color=LANE_DASH_COLOR,
            )
            left += step

    def _draw_board(self) -> None:
        for row in range(int(self._board_rows)):
            if int(row) == int(self._goal_row):
                self._draw_row_band(int(row), outer_color=GOAL_ROW_OUTER, inner_color=GOAL_ROW_INNER)
                self._draw_goal_markers(int(row))
            elif int(row) == int(self._start_row):
                self._draw_row_band(int(row), outer_color=SAFE_ROW_OUTER, inner_color=SAFE_ROW_INNER)
            else:
                self._draw_row_band(int(row), outer_color=ROAD_ROW_OUTER, inner_color=ROAD_ROW_INNER)
                self._draw_lane_dashes(int(row))

        bottom = self.window_controller.top_left_to_bottom(float(self._board_offset_y), float(self._board_height_px))
        arcade.draw_lbwh_rectangle_outline(
            float(self._board_offset_x),
            float(bottom),
            float(self._board_width_px),
            float(self._board_height_px),
            BOARD_OUTLINE_COLOR,
            border_width=2.0,
        )

    def _draw_car_body(self, *, left: float, top: float, width: float, height: float, lane: LaneTraffic) -> None:
        self._draw_rect_top_left(left=left, top=top, width=width, height=height, color=lane.outer_color)
        inset = max(3.0, 0.14 * float(width))
        self._draw_rect_top_left(
            left=left + inset,
            top=top + inset,
            width=max(1.0, width - 2.0 * inset),
            height=max(1.0, height - 2.0 * inset),
            color=lane.inner_color,
        )
        strip_width = max(2.0, width * float(config.CAR_FRONT_STRIP_RATIO))
        strip_left = left + width - strip_width if int(lane.direction) > 0 else left
        self._draw_rect_top_left(
            left=float(strip_left),
            top=float(top + 0.15 * height),
            width=float(strip_width),
            height=max(1.0, 0.70 * height),
            color=COLOR_FOG_GRAY,
        )

    def _draw_cars(self) -> None:
        car_width_px = float(config.CAR_LENGTH_TILES) * float(config.COLUMN_WIDTH_PX)
        car_height_px = float(config.FROG_SIZE_PX) * float(config.CAR_HEIGHT_RATIO)

        for lane in self._lanes:
            row_top = float(self._row_band_top(int(lane.row)))
            car_top = row_top + (float(self._row_band_height_px) - car_height_px) * 0.5
            for left_tiles in lane.car_left_positions:
                if not LaneTraffic._fully_inside(
                    float(left_tiles),
                    width_tiles=int(config.BOARD_WIDTH_TILES),
                    car_length_tiles=float(config.CAR_LENGTH_TILES),
                ):
                    continue
                car_left = float(self._board_offset_x) + float(left_tiles) * float(config.COLUMN_WIDTH_PX)
                self._draw_car_body(
                    left=float(car_left),
                    top=float(car_top),
                    width=float(car_width_px),
                    height=float(car_height_px),
                    lane=lane,
                )

    def _draw_frog(self) -> None:
        frog_left = float(self._column_left(int(self.frog_x)) + (float(config.COLUMN_WIDTH_PX) - float(config.FROG_SIZE_PX)) * 0.5)
        frog_top = float(self._row_top(int(self.frog_y)) + (float(config.ROW_PITCH_PX) - float(config.FROG_SIZE_PX)) * 0.5)
        draw_two_tone_tile(
            self.window_controller,
            top_left_x=float(frog_left),
            top_left_y=float(frog_top),
            size=float(config.FROG_SIZE_PX),
            outer_color=COLOR_AQUA,
            inner_color=COLOR_DEEP_TEAL,
            inset=float(config.CELL_INSET),
        )
        eye_size = max(3.0, 0.14 * float(config.FROG_SIZE_PX))
        self._draw_rect_top_left(
            left=float(frog_left + 0.26 * config.FROG_SIZE_PX),
            top=float(frog_top + 0.20 * config.FROG_SIZE_PX),
            width=float(eye_size),
            height=float(eye_size),
            color=COLOR_LIGHT_NEUTRAL,
        )
        self._draw_rect_top_left(
            left=float(frog_left + 0.60 * config.FROG_SIZE_PX),
            top=float(frog_top + 0.20 * config.FROG_SIZE_PX),
            width=float(eye_size),
            height=float(eye_size),
            color=COLOR_LIGHT_NEUTRAL,
        )

    @staticmethod
    def _status_icon_size() -> float:
        return status_icon_size(float(config.BB_HEIGHT), float(config.FROG_SIZE_PX))

    def _score_icons(self) -> list[bool]:
        return compact_count_to_icons(int(self.score), pack_size=int(config.POINT_ICON_PACK_SIZE))

    def _draw_point_icon(self, center_x: float, center_y: float, size: float, compressed: bool = False) -> None:
        inset = status_icon_inset(float(config.CELL_INSET))
        marker_size = max(2.0, size * 0.26)
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=COLOR_AQUA,
            inner_color=COLOR_DEEP_TEAL,
            inset=float(inset),
            packed=bool(compressed),
            packed_marker_color=COLOR_LIGHT_NEUTRAL,
            packed_marker_size=float(marker_size),
        )

    def _draw_score_icons(self, left: float, right: float, center_y: float) -> None:
        icon_size = self._status_icon_size()
        icons = self._score_icons()
        draw_status_icon_row(
            left=float(left),
            right=float(right),
            center_y=float(center_y),
            icon_size=float(icon_size),
            items=icons,
            draw_item=lambda is_compressed, center_x, row_center_y, size: self._draw_point_icon(
                center_x=float(center_x),
                center_y=float(row_center_y),
                size=float(size),
                compressed=bool(is_compressed),
            ),
        )

    def _draw_hud(self) -> None:
        layout = draw_status_bar(
            width=float(config.SCREEN_WIDTH),
            bottom_bar_height=float(config.BB_HEIGHT),
            tile_size=float(config.FROG_SIZE_PX),
            cell_inset=float(config.CELL_INSET),
            include_clock=True,
            left_panel_width=0.0,
        )
        draw_status_clock(
            layout=layout,
            remaining_ratio=float(clip_unit(float(self.max_steps - self.steps) / float(max(1, self.max_steps)))),
        )
        self._draw_score_icons(
            left=float(layout.score_left),
            right=float(layout.score_right),
            center_y=float(layout.center_y),
        )

    def render(self) -> None:
        if self.window_controller.window is None:
            return
        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        arcade.draw_lbwh_rectangle_filled(
            0,
            float(config.BB_HEIGHT),
            float(config.WORLD_WIDTH),
            float(config.WORLD_HEIGHT),
            COLOR_DARK_NEUTRAL,
        )
        self._draw_board()
        self._draw_cars()
        self._draw_frog()
        self._draw_hud()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
