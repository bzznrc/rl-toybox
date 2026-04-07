"""Stealth environment."""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BARK,
    COLOR_BRICK_RED,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_FOREST_GREEN,
    COLOR_LEAF_GREEN,
    COLOR_LIGHT_NEUTRAL,
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
from core.io_schema import clip_signed, clip_unit
from core.primitives import draw_facing_indicator, draw_time_pie_indicator, draw_two_tone_tile, status_bar_layout
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level
from games.stealth import config
from games.stealth.layout import Cell, GuardLane, StealthLayout, generate_layout, guard_lane_cells, is_visible, is_walkable


validate_curriculum_level_settings(
    min_level=config.MIN_LEVEL,
    max_level=config.MAX_LEVEL,
    level_settings=config.LEVEL_SETTINGS,
)


ACTION_TO_DELTA = {
    int(config.ACTION_MOVE_UP): (0, -1),
    int(config.ACTION_MOVE_DOWN): (0, 1),
    int(config.ACTION_MOVE_LEFT): (-1, 0),
    int(config.ACTION_MOVE_RIGHT): (1, 0),
    int(config.ACTION_WAIT): (0, 0),
}


@dataclass
class GuardActor:
    lane: GuardLane
    path_cells: tuple[Cell, ...]
    path_index: int
    step_dir: int
    facing_dx: int
    facing_dy: int

    @classmethod
    def from_lane(cls, lane: GuardLane) -> "GuardActor":
        return cls(
            lane=lane,
            path_cells=tuple(guard_lane_cells(lane)),
            path_index=0,
            step_dir=1,
            facing_dx=int(lane.facing_dx),
            facing_dy=int(lane.facing_dy),
        )

    @property
    def position(self) -> Cell:
        return self.path_cells[int(self.path_index)]

    def advance(self) -> None:
        if len(self.path_cells) <= 1:
            return
        next_index = int(self.path_index + self.step_dir)
        if next_index >= len(self.path_cells) or next_index < 0:
            self.step_dir *= -1
            next_index = int(self.path_index + self.step_dir)
        current = self.position
        self.path_index = int(next_index)
        nxt = self.position
        self.facing_dx = int(np.sign(int(nxt.x) - int(current.x)))
        self.facing_dy = int(np.sign(int(nxt.y) - int(current.y)))

    def phase_norm(self) -> float:
        count = len(self.path_cells)
        if count <= 1:
            return 0.0
        cycle_len = 2 * (count - 1)
        cycle_index = int(self.path_index if self.step_dir > 0 else cycle_len - self.path_index)
        return float(cycle_index) / float(max(1, cycle_len))


def _resolve_level_settings(level: int) -> dict[str, object]:
    base = dict(config.LEVEL_SETTINGS[int(level)])
    return {
        **base,
        "max_steps": max(
            int(config.STEP_BUDGET_MIN),
            int(round(float(base["min_start_exit_dist"]) * float(config.STEP_BUDGET_PER_ROUTE_TILE))),
        ),
    }


class StealthEnv(Env):
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
        self._settings = _resolve_level_settings(int(self._current_level))
        self.max_steps = int(self._settings["max_steps"])

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

        self._layout: StealthLayout | None = None
        self._guards: list[GuardActor] = []
        self._episode_counter = 0
        self.steps = 0
        self.done = False
        self.player = Cell(0, 0)
        self._danger_tiles: set[Cell] = set()
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

    def _guard_vision_cells(self, guard: GuardActor) -> set[Cell]:
        assert self._layout is not None
        observed = {guard.position}
        for step in range(1, int(config.GUARD_VISION_RANGE) + 1):
            cell = guard.position.moved(int(guard.facing_dx) * step, int(guard.facing_dy) * step)
            if not is_walkable(self._layout.walkable, cell):
                break
            if cell in self._layout.covers:
                break
            observed.add(cell)
        return observed

    def _refresh_danger_tiles(self) -> None:
        danger_tiles: set[Cell] = set()
        for guard in self._guards:
            danger_tiles.update(self._guard_vision_cells(guard))
        self._danger_tiles = danger_tiles

    def _player_on_cover(self) -> bool:
        return self._layout is not None and self.player in self._layout.covers

    def _is_caught(self) -> bool:
        if self._player_on_cover():
            return any(guard.position == self.player for guard in self._guards)
        return any(self.player in self._guard_vision_cells(guard) for guard in self._guards)

    def _move_player(self, action_idx: int) -> bool:
        assert self._layout is not None
        dx, dy = ACTION_TO_DELTA.get(int(action_idx), (0, 0))
        if int(dx) == 0 and int(dy) == 0:
            return False
        candidate = self.player.moved(dx, dy)
        if is_walkable(self._layout.walkable, candidate):
            self.player = candidate
            return False
        return True

    def _exit_visible(self) -> bool:
        assert self._layout is not None
        return bool(
            is_visible(
                self._layout.walkable,
                self._layout.covers,
                self.player,
                self._layout.exit,
                int(config.EXIT_VIEW_RANGE),
            )
        )

    def _exit_rel_xy(self) -> tuple[float, float]:
        assert self._layout is not None
        scale = float(max(1, int(config.EXIT_VIEW_RANGE)))
        dx = clip_signed(float(int(self._layout.exit.x) - int(self.player.x)) / scale)
        dy = clip_signed(float(int(self._layout.exit.y) - int(self.player.y)) / scale)
        return float(dx), float(dy)

    def _exit_distance(self, cell: Cell) -> int:
        assert self._layout is not None
        if not is_walkable(self._layout.walkable, cell):
            return -1
        return int(self._layout.exit_distance_map[int(cell.y), int(cell.x)])

    def _progress_reward(self, previous_player: Cell) -> float:
        previous_distance = self._exit_distance(previous_player)
        current_distance = self._exit_distance(self.player)
        if previous_distance < 0 or current_distance < 0:
            return 0.0
        distance_delta = float(
            np.clip(
                float(previous_distance - current_distance),
                -float(config.REWARD_PROGRESS_CLIP),
                float(config.REWARD_PROGRESS_CLIP),
            )
        )
        return float(config.REWARD_PROGRESS_SCALE) * float(distance_delta)

    def _local_patch_tokens(self) -> list[float]:
        assert self._layout is not None
        tokens: list[float] = []
        for dy in range(-int(config.PATCH_RADIUS), int(config.PATCH_RADIUS) + 1):
            for dx in range(-int(config.PATCH_RADIUS), int(config.PATCH_RADIUS) + 1):
                cell = self.player.moved(dx, dy)
                if not is_walkable(self._layout.walkable, cell):
                    tokens.append(float(config.TOKEN_WALL))
                    continue
                if any(guard.position == cell for guard in self._guards):
                    tokens.append(float(config.TOKEN_GUARD))
                    continue
                if cell in self._danger_tiles and cell not in self._layout.covers:
                    tokens.append(float(config.TOKEN_DANGER))
                    continue
                if cell == self._layout.exit:
                    tokens.append(float(config.TOKEN_EXIT))
                    continue
                if cell in self._layout.covers:
                    tokens.append(float(config.TOKEN_COVER))
                    continue
                tokens.append(float(config.TOKEN_EMPTY))
        return tokens

    def _patrol_phase_norm(self) -> float:
        if not self._guards:
            return 0.0
        return float(self._guards[0].phase_norm())

    def _build_observation(self) -> np.ndarray:
        exit_visible = self._exit_visible()
        exit_dx, exit_dy = self._exit_rel_xy() if exit_visible else (0.0, 0.0)
        values = [
            *self._local_patch_tokens(),
            1.0 if exit_visible else 0.0,
            float(exit_dx),
            float(exit_dy),
            1.0 if self._player_on_cover() else 0.0,
            1.0 if self.player in self._danger_tiles and (not self._player_on_cover()) else 0.0,
            float(clip_unit(float(self.max_steps - self.steps) / float(max(1, self.max_steps)))),
            float(self._patrol_phase_norm()),
        ]
        obs = np.asarray(values, dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Stealth observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self._layout = generate_layout(
            seed=self._episode_seed(),
            layout_attempts=int(config.LAYOUT_ATTEMPTS),
            rows=int(config.GRID_HEIGHT_TILES),
            cols=int(config.GRID_WIDTH_TILES),
            room_count=int(self._settings["room_count"]),
            room_size_range=tuple(config.ROOM_SIZE_RANGE),
            room_place_attempts=int(config.ROOM_PLACE_ATTEMPTS),
            extra_connection=bool(self._settings["extra_connection"]),
            guard_count=int(self._settings["guard_count"]),
            cover_count=int(self._settings["cover_count"]),
            guard_vision_range=int(config.GUARD_VISION_RANGE),
            guard_move_period=int(config.GUARD_MOVE_PERIOD),
            min_start_exit_dist=int(self._settings["min_start_exit_dist"]),
        )
        self._episode_counter += 1
        self._guards = [GuardActor.from_lane(lane) for lane in self._layout.guards]
        self.player = self._layout.start
        self.steps = 0
        self.done = False
        self._episode_reward_components.reset()
        self._refresh_danger_tiles()
        self._last_obs = self._build_observation()
        return np.asarray(self._last_obs, dtype=np.float32)

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

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return np.asarray(self._last_obs, dtype=np.float32), 0.0, True, {
                "win": bool(self._last_episode_success > 0),
                "success": int(self._last_episode_success),
                "level": int(self._last_episode_level),
                "reward_components": self._episode_reward_components.totals(),
            }

        self.window_controller.poll_events_or_raise()

        assert self._layout is not None
        episode_level = int(self._current_level)
        action_idx = int(self._human_action() if self.mode == "human" else action)
        action_idx = int(np.clip(action_idx, 0, int(self.ACT_DIM) - 1))
        previous_player = self.player

        blocked_move = self._move_player(action_idx)
        self.steps += 1

        reward = 0.0
        reward_breakdown: dict[str, float] = {}
        reward += self._apply_reward(reward_breakdown, "S", float(config.PENALTY_STEP))
        if blocked_move:
            reward += self._apply_reward(reward_breakdown, "B", float(config.PENALTY_BLOCKED_MOVE))
        reward += self._apply_reward(reward_breakdown, "P", float(self._progress_reward(previous_player)))

        caught = self._is_caught()
        timed_out = False
        success = 0

        if (not caught) and self.player == self._layout.exit:
            self.done = True
            success = 1
            reward += self._apply_reward(reward_breakdown, "W", float(config.REWARD_WIN))
        elif caught:
            self.done = True
            reward += self._apply_reward(reward_breakdown, "L", float(config.PENALTY_LOSE))
        else:
            if self.steps % int(config.GUARD_MOVE_PERIOD) == 0:
                for guard in self._guards:
                    guard.advance()
                caught = self._is_caught()
                if caught:
                    self.done = True
                    reward += self._apply_reward(reward_breakdown, "L", float(config.PENALTY_LOSE))
            if (not self.done) and self.steps >= int(self.max_steps):
                timed_out = True
                self.done = True
                reward += self._apply_reward(reward_breakdown, "L", float(config.PENALTY_LOSE))

        if self.mode != "human":
            for code, value in reward_breakdown.items():
                self._episode_reward_components.add(str(code), float(value))

        self._refresh_danger_tiles()

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
        assert self._layout is not None
        for y in range(int(config.GRID_HEIGHT_TILES)):
            for x in range(int(config.GRID_WIDTH_TILES)):
                cell = Cell(x, y)
                if not is_walkable(self._layout.walkable, cell):
                    self._draw_cell(cell, COLOR_DARK_NEUTRAL, COLOR_SLATE_GRAY)
                elif cell in self._layout.covers:
                    self._draw_cell(cell, COLOR_WALNUT, COLOR_BARK)
        self._draw_cell(self._layout.exit, COLOR_LEAF_GREEN, COLOR_FOREST_GREEN)
        if bool(config.DRAW_GUARD_VISION):
            for cell in self._danger_tiles:
                if cell in self._layout.covers:
                    continue
                bottom = self.window_controller.top_left_to_bottom(float(cell.y * config.TILE_SIZE), float(config.TILE_SIZE))
                arcade.draw_lbwh_rectangle_filled(
                    float(cell.x * config.TILE_SIZE),
                    bottom,
                    float(config.TILE_SIZE),
                    float(config.TILE_SIZE),
                    COLOR_BRICK_RED + (52,),
                )
        for guard in self._guards:
            self._draw_cell(guard.position, COLOR_SLATE_GRAY, COLOR_DARK_NEUTRAL)
            draw_facing_indicator(
                self.window_controller,
                center_x=float(guard.position.x * config.TILE_SIZE + config.TILE_SIZE / 2),
                center_y_top_left=float(guard.position.y * config.TILE_SIZE + config.TILE_SIZE / 2),
                angle_degrees=float(np.degrees(np.arctan2(float(guard.facing_dy), float(guard.facing_dx)))),
                length=float(config.TILE_SIZE * 0.32),
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
            f"Guard:{len(self._guards)}  "
            f"Cover:{1 if self._player_on_cover() else 0}  "
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
