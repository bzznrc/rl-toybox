"""Trail environment: compact top-down light-cycles duel."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_signed, clip_unit, ordered_feature_vector
from core.primitives import (
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    draw_status_square_icon,
    draw_two_tone_tile,
    status_icon_inset,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.match_tracker import MatchTracker
from core.utils import resolve_play_level
from games.trail.config import (
    ACTION_NAMES as TRAIL_ACTION_NAMES,
    ACT_DIM as TRAIL_ACT_DIM,
    ARENA_OUTLINE_ALPHA,
    BB_HEIGHT,
    CELL_INSET,
    CURRICULUM_PROMOTION,
    FPS,
    GRID_HEIGHT_TILES,
    GRID_LINE_ALPHA,
    GRID_WIDTH_TILES,
    INPUT_FEATURE_NAMES as TRAIL_INPUT_FEATURE_NAMES,
    LEVEL_SETTINGS,
    MAX_EPISODE_STEPS,
    MAX_LEVEL,
    MIN_LEVEL,
    OBS_DIM as TRAIL_OBS_DIM,
    OPPONENT_COMMIT_MAX_TICKS,
    OPPONENT_COMMIT_MIN_TICKS,
    OPPONENT_NEAR_TIE_EPSILON,
    OPPONENT_OPENING_SHIFT_CHOICES,
    OPPONENT_OPENING_TOTAL_TICKS,
    PENALTY_LOSE,
    PLAY_TOTAL_GAMES,
    REWARD_DRAW,
    REWARD_WIN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    START_HORIZONTAL_SEPARATION_TILES,
    START_MARGIN_TILES,
    START_OFFSET_CHOICES,
    TILE_SIZE,
    TRAINING_TOTAL_GAMES,
    TRAINING_FPS,
    WINDOW_TITLE,
)


validate_curriculum_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
)


@dataclass(frozen=True)
class Cell:
    x: int
    y: int


@dataclass
class Rider:
    name: str
    cell: Cell
    dir_x: int
    dir_y: int
    trail: list[Cell]
    outer_color: tuple[int, int, int]
    inner_color: tuple[int, int, int]


@dataclass(frozen=True)
class CandidateMetrics:
    action_idx: int
    next_dir_x: int
    next_dir_y: int
    next_cell: Cell
    collision: bool
    self_area: float
    opp_best_area: float
    area_advantage: float
    opp_dist_norm: float
    forward_align: float
    center_clearance: float
    pressure_score: float


class TrailEnv(Env):
    """Compact Tron-style duel with one learned rider against a deterministic opponent."""

    TRAINING_TOTAL_GAMES = int(TRAINING_TOTAL_GAMES)
    PLAY_TOTAL_GAMES = int(PLAY_TOTAL_GAMES)
    INPUT_FEATURE_NAMES = tuple(TRAIL_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(TRAIL_ACTION_NAMES)
    OBS_DIM = int(TRAIL_OBS_DIM)
    ACT_DIM = int(TRAIL_ACT_DIM)
    ACTION_TURN_LEFT = 0
    ACTION_GO_STRAIGHT = 1
    ACTION_TURN_RIGHT = 2
    ACTION_PREFERENCE = (ACTION_GO_STRAIGHT, ACTION_TURN_LEFT, ACTION_TURN_RIGHT)
    PLAYER_ID = "player"
    OPPONENT_ID = "opponent"
    REWARD_COMPONENT_ORDER = ("W", "L")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_win": "W",
        "outcome.penalty_lose": "L",
    }

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode).strip().lower()
        self.show_game = bool(render)
        curriculum_config = build_curriculum_config(
            min_level=int(MIN_LEVEL),
            max_level=int(MAX_LEVEL),
            promotion_settings=CURRICULUM_PROMOTION,
        )
        self._curriculum = (
            ThreeLevelCurriculum(config=curriculum_config, level_settings=LEVEL_SETTINGS)
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

        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=False,
            vsync=False,
        )
        self.window = self.window_controller.window

        self.grid_width = int(GRID_WIDTH_TILES)
        self.grid_height = int(GRID_HEIGHT_TILES)
        self.tile_size = float(TILE_SIZE)
        self.arena_width_px = float(self.grid_width * int(TILE_SIZE))
        self.arena_height_px = float(self.grid_height * int(TILE_SIZE))
        self._max_probe_steps = max(1, max(int(self.grid_width), int(self.grid_height)) - 1)
        self._max_diag = max(1e-6, math.hypot(float(self.grid_width - 1), float(self.grid_height - 1)))
        self._total_cells = max(1, int(self.grid_width * self.grid_height))

        self.max_episode_steps = int(MAX_EPISODE_STEPS)
        self.total_games = int(self._resolve_total_games())
        self.current_game = 1
        self._opponent_area_weight = 1.0
        self._opponent_advantage_weight = 0.0
        self._opponent_pressure_weight = 0.0
        self._opponent_center_weight = 0.0
        self._opponent_straight_bias = 0.0
        self._opponent_commit_ticks_remaining = 0
        self._opponent_opening_release_step = 0
        self._opponent_opening_target_x = 0

        self.player: Rider | None = None
        self.opponent: Rider | None = None
        self.occupied_cells: set[Cell] = set()
        self.crash_cells: list[Cell] = []
        self.steps = 0
        self.done = False
        self.last_action_index = int(self.ACTION_GO_STRAIGHT)
        self._last_terminal_reward = 0.0
        self.match_tracker = MatchTracker[str](
            history_limit=int(self.total_games),
            match_limit=int(self.total_games),
            clock_duration_steps=int(self.max_episode_steps),
        )
        self.match_tracker.set_competitors((self.PLAYER_ID, self.OPPONENT_ID), preserve_existing=False)
        self.win_history: list[str | None] = self.match_tracker.history
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)
        self.last_reward_breakdown = self._zero_reward_breakdown()

        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _zero_reward_breakdown(self) -> dict[str, float]:
        return {
            "outcome.reward_win": 0.0,
            "outcome.penalty_lose": 0.0,
        }

    def get_entropy_coef_for_level(self, level: int | None = None) -> float | None:
        target_level = int(self._current_level if level is None else level)
        settings = LEVEL_SETTINGS.get(int(target_level))
        if settings is None:
            raise ValueError(f"Unsupported level '{target_level}' for Trail.")
        raw_entropy = settings.get("entropy_coef")
        if raw_entropy is None:
            return None
        return float(raw_entropy)

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Trail.")
        self._current_level = int(level)
        self.max_episode_steps = max(1, int(settings.get("max_episode_steps", MAX_EPISODE_STEPS)))
        self._opponent_area_weight = float(settings.get("opponent_area_weight", 1.0))
        self._opponent_advantage_weight = float(settings.get("opponent_advantage_weight", 0.0))
        self._opponent_pressure_weight = float(settings.get("opponent_pressure_weight", 0.0))
        self._opponent_center_weight = float(settings.get("opponent_center_weight", 0.0))
        self._opponent_straight_bias = float(settings.get("opponent_straight_bias", 0.0))

    def _resolve_total_games(self) -> int:
        if self.mode == "train":
            return int(self.TRAINING_TOTAL_GAMES)
        return int(self.PLAY_TOTAL_GAMES)

    @staticmethod
    def _turn_left(dir_x: int, dir_y: int) -> tuple[int, int]:
        return int(dir_y), int(-dir_x)

    @staticmethod
    def _turn_right(dir_x: int, dir_y: int) -> tuple[int, int]:
        return int(-dir_y), int(dir_x)

    @classmethod
    def _dir_after_action(cls, dir_x: int, dir_y: int, action_idx: int) -> tuple[int, int]:
        action = int(action_idx)
        if action == cls.ACTION_TURN_LEFT:
            return cls._turn_left(int(dir_x), int(dir_y))
        if action == cls.ACTION_TURN_RIGHT:
            return cls._turn_right(int(dir_x), int(dir_y))
        return int(dir_x), int(dir_y)

    @staticmethod
    def _clip_cell(value: int, low: int, high: int) -> int:
        return max(int(low), min(int(high), int(value)))

    def _sample_start_layout(self) -> tuple[Cell, tuple[int, int], Cell, tuple[int, int]]:
        margin_x = min(max(2, int(START_MARGIN_TILES)), max(2, self.grid_width // 3))
        margin_y = min(max(2, int(START_MARGIN_TILES)), max(2, self.grid_height // 3))
        center_x = self.grid_width // 2
        center_shift = int(random.choice(tuple(START_OFFSET_CHOICES)))
        half_separation = max(4, int(START_HORIZONTAL_SEPARATION_TILES) // 2)
        spawn_y = self._clip_cell(max(2, min(margin_y, 4)) + int(random.choice((0, 1))), 2, self.grid_height - 3)
        player_x = self._clip_cell(center_x + center_shift - half_separation, margin_x, self.grid_width - 1 - margin_x)
        opponent_x = self._clip_cell(
            center_x + center_shift + half_separation,
            margin_x,
            self.grid_width - 1 - margin_x,
        )
        if int(opponent_x) <= int(player_x):
            opponent_x = self._clip_cell(
                int(player_x) + max(3, half_separation),
                margin_x,
                self.grid_width - 1 - margin_x,
            )
            player_x = self._clip_cell(
                int(opponent_x) - max(3, half_separation),
                margin_x,
                self.grid_width - 1 - margin_x,
            )
        player_cell = Cell(player_x, spawn_y)
        opponent_cell = Cell(opponent_x, spawn_y)
        return player_cell, (0, 1), opponent_cell, (0, 1)

    @staticmethod
    def _build_rider(
        name: str,
        *,
        cell: Cell,
        direction: tuple[int, int],
        outer_color: tuple[int, int, int],
        inner_color: tuple[int, int, int],
    ) -> Rider:
        return Rider(
            name=str(name),
            cell=cell,
            dir_x=int(direction[0]),
            dir_y=int(direction[1]),
            trail=[cell],
            outer_color=outer_color,
            inner_color=inner_color,
        )

    def _setup_round(self) -> None:
        player_cell, player_dir, opponent_cell, opponent_dir = self._sample_start_layout()
        self.player = self._build_rider(
            self.PLAYER_ID,
            cell=player_cell,
            direction=player_dir,
            outer_color=COLOR_AQUA,
            inner_color=COLOR_DEEP_TEAL,
        )
        self.opponent = self._build_rider(
            self.OPPONENT_ID,
            cell=opponent_cell,
            direction=opponent_dir,
            outer_color=COLOR_CORAL,
            inner_color=COLOR_BRICK_RED,
        )
        self.occupied_cells = {player_cell, opponent_cell}
        self.crash_cells = []
        self.steps = 0
        self.done = False
        self.last_action_index = int(self.ACTION_GO_STRAIGHT)
        self._last_terminal_reward = 0.0
        self.last_reward_breakdown = self._zero_reward_breakdown()
        self._reset_opponent_policy(player_cell=player_cell, opponent_cell=opponent_cell)

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self.total_games = int(self._resolve_total_games())
        self.match_tracker.set_match_limit(int(self.total_games))
        self.match_tracker.set_history_limit(int(self.total_games))
        self.match_tracker.set_clock_duration(int(self.max_episode_steps))
        self.match_tracker.clear_history()
        self.current_game = 1
        self._episode_reward_components.reset()
        self._setup_round()
        return self._obs()

    def _decode_action(self, action: object) -> int:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if int(values.size) <= 0:
            return int(self.ACTION_GO_STRAIGHT)
        if int(values.size) == int(self.ACT_DIM):
            return int(np.argmax(values))
        return int(np.clip(int(values[0]), 0, int(self.ACT_DIM) - 1))

    def _requested_absolute_direction(self) -> tuple[int, int] | None:
        if self.window_controller.is_key_down(arcade.key.W) or self.window_controller.is_key_down(arcade.key.UP):
            return 0, -1
        if self.window_controller.is_key_down(arcade.key.D) or self.window_controller.is_key_down(arcade.key.RIGHT):
            return 1, 0
        if self.window_controller.is_key_down(arcade.key.S) or self.window_controller.is_key_down(arcade.key.DOWN):
            return 0, 1
        if self.window_controller.is_key_down(arcade.key.A) or self.window_controller.is_key_down(arcade.key.LEFT):
            return -1, 0
        return None

    def _human_action(self) -> int:
        assert self.player is not None
        desired = self._requested_absolute_direction()
        if desired is None:
            return int(self.ACTION_GO_STRAIGHT)
        if desired == (int(self.player.dir_x), int(self.player.dir_y)):
            return int(self.ACTION_GO_STRAIGHT)
        if desired == self._turn_left(int(self.player.dir_x), int(self.player.dir_y)):
            return int(self.ACTION_TURN_LEFT)
        if desired == self._turn_right(int(self.player.dir_x), int(self.player.dir_y)):
            return int(self.ACTION_TURN_RIGHT)
        return int(self.ACTION_GO_STRAIGHT)

    def _in_bounds(self, cell: Cell) -> bool:
        return 0 <= int(cell.x) < int(self.grid_width) and 0 <= int(cell.y) < int(self.grid_height)

    @staticmethod
    def _next_cell_from(cell: Cell, dir_x: int, dir_y: int) -> Cell:
        return Cell(int(cell.x) + int(dir_x), int(cell.y) + int(dir_y))

    def _free_space_ratio(self, origin: Cell, dir_x: int, dir_y: int) -> float:
        steps_open = 0
        cell = origin
        for _ in range(int(self._max_probe_steps)):
            cell = self._next_cell_from(cell, int(dir_x), int(dir_y))
            if (not self._in_bounds(cell)) or (cell in self.occupied_cells):
                break
            steps_open += 1
        return float(clip_unit(float(steps_open) / float(self._max_probe_steps)))

    def _normalized_dx(self, delta_x: int) -> float:
        return float(clip_signed(float(delta_x) / float(max(1, self.grid_width - 1))))

    def _normalized_dy(self, delta_y: int) -> float:
        return float(clip_signed(float(delta_y) / float(max(1, self.grid_height - 1))))

    def _distance_norm(self, cell_a: Cell, cell_b: Cell) -> float:
        dx = float(int(cell_b.x) - int(cell_a.x))
        dy = float(int(cell_b.y) - int(cell_a.y))
        return float(clip_unit(math.hypot(dx, dy) / float(self._max_diag)))

    @staticmethod
    def _direction_align(dir_x: int, dir_y: int, from_cell: Cell, to_cell: Cell) -> float:
        dx = float(int(to_cell.x) - int(from_cell.x))
        dy = float(int(to_cell.y) - int(from_cell.y))
        norm = math.hypot(dx, dy)
        if norm <= 1e-8:
            return 0.0
        return float(clip_signed(((float(dir_x) * dx) + (float(dir_y) * dy)) / norm))

    def _center_clearance(self, cell: Cell) -> float:
        wall_clearance = min(
            int(cell.x),
            int(self.grid_width - 1 - cell.x),
            int(cell.y),
            int(self.grid_height - 1 - cell.y),
        )
        max_clearance = max(1.0, 0.5 * float(min(self.grid_width - 1, self.grid_height - 1)))
        return float(clip_unit(float(wall_clearance) / max_clearance))

    def _reachable_area_ratio(self, start_cell: Cell, blocked_cells: set[Cell]) -> float:
        if (not self._in_bounds(start_cell)) or (start_cell in blocked_cells):
            return 0.0
        total_free = max(1, int(self._total_cells - len(blocked_cells)))
        stack = [start_cell]
        visited = {start_cell}
        while stack:
            cell = stack.pop()
            neighbors = (
                Cell(int(cell.x) + 1, int(cell.y)),
                Cell(int(cell.x) - 1, int(cell.y)),
                Cell(int(cell.x), int(cell.y) + 1),
                Cell(int(cell.x), int(cell.y) - 1),
            )
            for neighbor in neighbors:
                if neighbor in visited or neighbor in blocked_cells or not self._in_bounds(neighbor):
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        return float(clip_unit(float(len(visited)) / float(total_free)))

    def _reachable_ratio_after_action(self, rider: Rider, action_idx: int, blocked_cells: set[Cell]) -> float:
        next_dir_x, next_dir_y = self._dir_after_action(int(rider.dir_x), int(rider.dir_y), int(action_idx))
        next_cell = self._next_cell_from(rider.cell, next_dir_x, next_dir_y)
        if (not self._in_bounds(next_cell)) or (next_cell in blocked_cells):
            return 0.0
        return float(self._reachable_area_ratio(next_cell, blocked_cells))

    def _candidate_metrics(self, rider: Rider, opponent: Rider, action_idx: int) -> CandidateMetrics:
        blocked_cells = set(self.occupied_cells)
        next_dir_x, next_dir_y = self._dir_after_action(int(rider.dir_x), int(rider.dir_y), int(action_idx))
        next_cell = self._next_cell_from(rider.cell, next_dir_x, next_dir_y)
        collision = (not self._in_bounds(next_cell)) or (next_cell in blocked_cells)
        reference_cell = next_cell if self._in_bounds(next_cell) else rider.cell
        opp_dist_norm = float(self._distance_norm(reference_cell, opponent.cell))
        forward_align = float(self._direction_align(next_dir_x, next_dir_y, reference_cell, opponent.cell))
        center_clearance = 0.0 if collision else float(self._center_clearance(next_cell))

        if collision:
            opp_best_area = max(
                float(self._reachable_ratio_after_action(opponent, candidate_action, blocked_cells))
                for candidate_action in range(int(self.ACT_DIM))
            )
            area_advantage = float(clip_signed(-opp_best_area))
            pressure_score = -1.0
            self_area = 0.0
        else:
            self_area = float(self._reachable_area_ratio(next_cell, blocked_cells))
            blocked_for_opponent = set(blocked_cells)
            blocked_for_opponent.add(next_cell)
            opp_best_area = max(
                float(self._reachable_ratio_after_action(opponent, candidate_action, blocked_for_opponent))
                for candidate_action in range(int(self.ACT_DIM))
            )
            area_advantage = float(clip_signed(self_area - opp_best_area))
            pressure_score = float(
                clip_signed(0.65 * float(forward_align) + 0.35 * (1.0 - 2.0 * float(opp_dist_norm)))
            )

        return CandidateMetrics(
            action_idx=int(action_idx),
            next_dir_x=int(next_dir_x),
            next_dir_y=int(next_dir_y),
            next_cell=next_cell,
            collision=bool(collision),
            self_area=float(self_area),
            opp_best_area=float(opp_best_area),
            area_advantage=float(area_advantage),
            opp_dist_norm=float(opp_dist_norm),
            forward_align=float(forward_align),
            center_clearance=float(center_clearance),
            pressure_score=float(pressure_score),
        )

    def _build_observation(self, rider: Rider, opponent: Rider) -> np.ndarray:
        left_x, left_y = self._turn_left(int(rider.dir_x), int(rider.dir_y))
        right_x, right_y = self._turn_right(int(rider.dir_x), int(rider.dir_y))
        back_x, back_y = -int(rider.dir_x), -int(rider.dir_y)
        fwd_left_x = int(rider.dir_x) + int(left_x)
        fwd_left_y = int(rider.dir_y) + int(left_y)
        fwd_right_x = int(rider.dir_x) + int(right_x)
        fwd_right_y = int(rider.dir_y) + int(right_y)

        metrics_by_action = {
            int(action_idx): self._candidate_metrics(rider, opponent, int(action_idx))
            for action_idx in range(int(self.ACT_DIM))
        }
        best_advantage = max(float(metrics.area_advantage) for metrics in metrics_by_action.values())

        raw_dx = int(opponent.cell.x) - int(rider.cell.x)
        raw_dy = int(opponent.cell.y) - int(rider.cell.y)
        all_feature_values = {
            "self_dir_x": float(rider.dir_x),
            "self_dir_y": float(rider.dir_y),
            "sens_fwd": float(self._free_space_ratio(rider.cell, int(rider.dir_x), int(rider.dir_y))),
            "sens_left": float(self._free_space_ratio(rider.cell, int(left_x), int(left_y))),
            "sens_right": float(self._free_space_ratio(rider.cell, int(right_x), int(right_y))),
            "sens_back": float(self._free_space_ratio(rider.cell, int(back_x), int(back_y))),
            "sens_fwd_left": float(self._free_space_ratio(rider.cell, int(fwd_left_x), int(fwd_left_y))),
            "sens_fwd_right": float(self._free_space_ratio(rider.cell, int(fwd_right_x), int(fwd_right_y))),
            "opp_dx": float(self._normalized_dx(raw_dx)),
            "opp_dy": float(self._normalized_dy(raw_dy)),
            "opp_dir_x": float(opponent.dir_x),
            "opp_dir_y": float(opponent.dir_y),
            "opp_dist_norm": float(self._distance_norm(rider.cell, opponent.cell)),
            "opp_fwd_align": float(
                self._direction_align(int(opponent.dir_x), int(opponent.dir_y), opponent.cell, rider.cell)
            ),
            "map_area_left_norm": float(metrics_by_action[int(self.ACTION_TURN_LEFT)].self_area),
            "map_area_straight_norm": float(metrics_by_action[int(self.ACTION_GO_STRAIGHT)].self_area),
            "map_area_right_norm": float(metrics_by_action[int(self.ACTION_TURN_RIGHT)].self_area),
            "map_area_adv_norm": float(best_advantage),
            "map_fill_ratio_norm": float(self.fill_ratio()),
            "flag_time_norm": float(self.time_norm()),
        }
        feature_values = {
            name: float(all_feature_values[name])
            for name in self.INPUT_FEATURE_NAMES
        }
        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (int(self.OBS_DIM),):
            raise RuntimeError(f"Trail observation expected {int(self.OBS_DIM)} features, got {obs.shape[0]}")
        if not np.isfinite(obs).all():
            raise RuntimeError("Trail observation contains non-finite values.")
        return obs

    def _obs(self) -> np.ndarray:
        if self.player is None or self.opponent is None:
            return np.zeros((int(self.OBS_DIM),), dtype=np.float32)
        return self._build_observation(self.player, self.opponent)

    def fill_ratio(self) -> float:
        return float(clip_unit(float(len(self.occupied_cells)) / float(self._total_cells)))

    def time_norm(self) -> float:
        return float(clip_unit(float(self.steps) / float(max(1, self.max_episode_steps))))

    def _sample_opponent_commit_ticks(self) -> int:
        return int(random.randint(int(OPPONENT_COMMIT_MIN_TICKS), int(OPPONENT_COMMIT_MAX_TICKS)))

    def _reset_opponent_policy(self, *, player_cell: Cell, opponent_cell: Cell) -> None:
        self._opponent_commit_ticks_remaining = 0
        self._opponent_opening_release_step = int(
            random.randint(int(OPPONENT_COMMIT_MIN_TICKS), int(OPPONENT_COMMIT_MAX_TICKS))
        )
        opening_shift = int(random.choice(tuple(OPPONENT_OPENING_SHIFT_CHOICES)))
        self._opponent_opening_target_x = int(
            self._clip_cell(int(opponent_cell.x) + opening_shift, 1, self.grid_width - 2)
        )
        if abs(int(self._opponent_opening_target_x) - int(opponent_cell.x)) < 2:
            fallback_shift = 4 if int(opponent_cell.x) <= int(player_cell.x) else -4
            self._opponent_opening_target_x = int(
                self._clip_cell(int(opponent_cell.x) + fallback_shift, 1, self.grid_width - 2)
            )

    def _opening_opponent_action(self) -> int | None:
        assert self.opponent is not None
        if int(self.steps) >= int(OPPONENT_OPENING_TOTAL_TICKS):
            return None
        if int(self.steps) < int(self._opponent_opening_release_step):
            return int(self.ACTION_GO_STRAIGHT)

        delta_x = int(self._opponent_opening_target_x) - int(self.opponent.cell.x)
        if abs(delta_x) <= 1:
            if (int(self.opponent.dir_x), int(self.opponent.dir_y)) == (0, 1):
                return int(self.ACTION_GO_STRAIGHT)
            if int(self.opponent.dir_x) > 0:
                return int(self.ACTION_TURN_RIGHT)
            if int(self.opponent.dir_x) < 0:
                return int(self.ACTION_TURN_LEFT)
            return int(self.ACTION_GO_STRAIGHT)

        if (int(self.opponent.dir_x), int(self.opponent.dir_y)) == (0, 1):
            return int(self.ACTION_TURN_LEFT if delta_x > 0 else self.ACTION_TURN_RIGHT)
        if int(self.opponent.dir_x) != 0:
            return int(self.ACTION_GO_STRAIGHT)
        return int(self.ACTION_GO_STRAIGHT)

    def _score_opponent_action(self) -> int:
        assert self.player is not None and self.opponent is not None
        action_scores: list[tuple[int, float]] = []
        for action_idx in self.ACTION_PREFERENCE:
            metrics = self._candidate_metrics(self.opponent, self.player, int(action_idx))
            if metrics.collision:
                continue
            score = (
                float(self._opponent_area_weight) * float(metrics.self_area)
                + float(self._opponent_advantage_weight) * float(metrics.area_advantage)
                + float(self._opponent_pressure_weight) * float(metrics.pressure_score)
                + float(self._opponent_center_weight) * float(metrics.center_clearance)
                + (float(self._opponent_straight_bias) if int(action_idx) == int(self.ACTION_GO_STRAIGHT) else 0.0)
            )
            action_scores.append((int(action_idx), float(score)))
        if not action_scores:
            return int(self.ACTION_GO_STRAIGHT)

        best_score = max(float(score) for _, score in action_scores)
        near_best_actions = [
            int(action_idx)
            for action_idx, score in action_scores
            if float(best_score) - float(score) <= float(OPPONENT_NEAR_TIE_EPSILON)
        ]
        if len(near_best_actions) > 1:
            return int(random.choice(near_best_actions))
        for preferred_action in self.ACTION_PREFERENCE:
            for action_idx, score in action_scores:
                if int(action_idx) == int(preferred_action) and abs(float(score) - float(best_score)) <= 1e-9:
                    return int(action_idx)
        return int(action_scores[0][0])

    def _select_opponent_action(self) -> int:
        assert self.player is not None and self.opponent is not None
        if int(self._opponent_commit_ticks_remaining) > 0:
            self._opponent_commit_ticks_remaining -= 1
            return int(self.ACTION_GO_STRAIGHT)

        opening_action = self._opening_opponent_action()
        if opening_action is not None:
            opening_metrics = self._candidate_metrics(self.opponent, self.player, int(opening_action))
            selected_action = int(opening_action) if not opening_metrics.collision else int(self._score_opponent_action())
        else:
            selected_action = int(self._score_opponent_action())
        self._opponent_commit_ticks_remaining = max(0, int(self._sample_opponent_commit_ticks()) - 1)
        return int(selected_action)

    @staticmethod
    def _update_rider(rider: Rider, *, cell: Cell, direction: tuple[int, int]) -> None:
        rider.cell = cell
        rider.dir_x = int(direction[0])
        rider.dir_y = int(direction[1])
        rider.trail.append(cell)

    def _step_terminal_info(self, *, reward: float) -> dict[str, object]:
        return {
            "score": int(1 if reward > 0.0 else (-1 if reward < 0.0 else 0)),
            "win": bool(reward > 0.0),
            "draw": bool(abs(float(reward)) <= 1e-8),
            "level": int(self._last_episode_level),
            "success": int(self._last_episode_success),
            "game": int(min(self.current_game, self.total_games)),
            "races_finished": int(len(self.win_history)),
            "races_total": int(self.total_games),
            "fill_ratio": float(self.fill_ratio()),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "reward_components": self._episode_reward_components.totals(),
        }

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._obs(), 0.0, True, self._step_terminal_info(reward=self._last_terminal_reward)

        assert self.player is not None and self.opponent is not None
        self.window_controller.poll_events_or_raise()

        episode_level = int(self._current_level)
        player_action = int(self._human_action() if self.mode == "human" else self._decode_action(action))
        opponent_action = int(self._select_opponent_action())
        self.last_action_index = int(player_action)

        player_next_dir = self._dir_after_action(int(self.player.dir_x), int(self.player.dir_y), int(player_action))
        opponent_next_dir = self._dir_after_action(
            int(self.opponent.dir_x),
            int(self.opponent.dir_y),
            int(opponent_action),
        )
        player_next_cell = self._next_cell_from(self.player.cell, int(player_next_dir[0]), int(player_next_dir[1]))
        opponent_next_cell = self._next_cell_from(
            self.opponent.cell,
            int(opponent_next_dir[0]),
            int(opponent_next_dir[1]),
        )

        blocked_cells = set(self.occupied_cells)
        player_collision = (not self._in_bounds(player_next_cell)) or (player_next_cell in blocked_cells)
        opponent_collision = (not self._in_bounds(opponent_next_cell)) or (opponent_next_cell in blocked_cells)
        simultaneous_head_on = (
            (not player_collision)
            and (not opponent_collision)
            and player_next_cell == opponent_next_cell
        )
        if simultaneous_head_on:
            player_collision = True
            opponent_collision = True

        self.steps += 1
        self.crash_cells = []
        if bool(simultaneous_head_on) and self._in_bounds(player_next_cell):
            self.crash_cells.append(player_next_cell)
        else:
            if bool(player_collision) and self._in_bounds(player_next_cell):
                self.crash_cells.append(player_next_cell)
            if bool(opponent_collision) and self._in_bounds(opponent_next_cell) and opponent_next_cell not in self.crash_cells:
                self.crash_cells.append(opponent_next_cell)

        if not bool(player_collision):
            self._update_rider(self.player, cell=player_next_cell, direction=player_next_dir)
            self.occupied_cells.add(player_next_cell)
        if not bool(opponent_collision):
            self._update_rider(self.opponent, cell=opponent_next_cell, direction=opponent_next_dir)
            self.occupied_cells.add(opponent_next_cell)

        reward = 0.0
        done = False
        player_win = False
        draw = False
        step_breakdown = self._zero_reward_breakdown()

        if bool(player_collision) or bool(opponent_collision):
            done = True
            if bool(player_collision) and bool(opponent_collision):
                draw = True
                reward = float(REWARD_DRAW)
            elif bool(opponent_collision):
                reward = float(REWARD_WIN)
                player_win = True
                step_breakdown["outcome.reward_win"] = float(REWARD_WIN)
            else:
                reward = float(PENALTY_LOSE)
                step_breakdown["outcome.penalty_lose"] = float(PENALTY_LOSE)
        elif int(self.steps) >= int(self.max_episode_steps):
            done = True
            draw = True
            reward = float(REWARD_DRAW)

        self.last_reward_breakdown = dict(step_breakdown)
        self._episode_reward_components.add_from_mapping(step_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)
        round_finished = bool(done)
        self.done = bool(done)
        if round_finished:
            self._last_terminal_reward = float(reward)
            if bool(player_win):
                self.match_tracker.record_result(self.PLAYER_ID)
            elif bool(draw):
                self.match_tracker.record_draw()
            else:
                self.match_tracker.record_result(self.OPPONENT_ID)
            if self.match_tracker.match_limit_reached():
                self.done = True
            else:
                self.current_game = int(self.match_tracker.matches_played()) + 1
                self._setup_round()
                done = False

        obs = self._obs()
        info: dict[str, object] = {
            "score": int(1 if player_win else (-1 if (round_finished and not draw) else 0)),
            "win": bool(player_win),
            "draw": bool(draw),
            "level": int(episode_level),
            "success": int(player_win) if self.done else 0,
            "level_changed": False,
            "game": int(min(self.current_game, self.total_games)),
            "races_finished": int(len(self.win_history)),
            "races_total": int(self.total_games),
            "fill_ratio": float(self.fill_ratio()),
            "reward_breakdown": dict(step_breakdown),
        }
        if self.done:
            info["reward_components"] = self._episode_reward_components.totals()
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(1 if player_win else 0)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(1 if player_win else 0),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )
            info["level_changed"] = bool(level_changed)

        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)
        return obs, float(reward), bool(self.done), info

    def _cell_top_left(self, cell: Cell) -> tuple[float, float]:
        return float(int(cell.x) * int(TILE_SIZE)), float(int(cell.y) * int(TILE_SIZE))

    def _draw_arena(self) -> None:
        arena_bottom = self.window_controller.top_left_to_bottom(0.0, self.arena_height_px)
        arcade.draw_lbwh_rectangle_filled(
            0.0,
            float(arena_bottom),
            float(self.arena_width_px),
            float(self.arena_height_px),
            COLOR_DARK_NEUTRAL,
        )
        grid_color = (*COLOR_FOG_GRAY, int(GRID_LINE_ALPHA))
        outline_color = (*COLOR_FOG_GRAY, int(ARENA_OUTLINE_ALPHA))
        top_arcade_y = self.window_controller.to_arcade_y(0.0)
        bottom_arcade_y = self.window_controller.to_arcade_y(self.arena_height_px)
        for col in range(int(self.grid_width) + 1):
            x = float(col * int(TILE_SIZE))
            arcade.draw_line(x, top_arcade_y, x, bottom_arcade_y, grid_color, 1.0)
        for row in range(int(self.grid_height) + 1):
            y = self.window_controller.to_arcade_y(float(row * int(TILE_SIZE)))
            arcade.draw_line(0.0, y, float(self.arena_width_px), y, grid_color, 1.0)
        arcade.draw_lbwh_rectangle_outline(
            0.0,
            float(arena_bottom),
            float(self.arena_width_px),
            float(self.arena_height_px),
            outline_color,
            1.0,
        )

    def _draw_rider_trail(self, rider: Rider) -> None:
        for cell in rider.trail[:-1]:
            top_left_x, top_left_y = self._cell_top_left(cell)
            draw_two_tone_tile(
                self.window_controller,
                top_left_x=float(top_left_x),
                top_left_y=float(top_left_y),
                size=float(TILE_SIZE),
                outer_color=rider.outer_color,
                inner_color=rider.inner_color,
                inset=float(CELL_INSET),
            )

    def _draw_rider_head(self, rider: Rider) -> None:
        top_left_x, top_left_y = self._cell_top_left(rider.cell)
        draw_two_tone_tile(
            self.window_controller,
            top_left_x=float(top_left_x),
            top_left_y=float(top_left_y),
            size=float(TILE_SIZE),
            outer_color=rider.outer_color,
            inner_color=rider.inner_color,
            inset=float(CELL_INSET),
        )

    def _draw_crash_cells(self) -> None:
        for cell in self.crash_cells:
            top_left_x, top_left_y = self._cell_top_left(cell)
            draw_two_tone_tile(
                self.window_controller,
                top_left_x=float(top_left_x),
                top_left_y=float(top_left_y),
                size=float(TILE_SIZE),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_LIGHT_NEUTRAL,
                inset=float(CELL_INSET),
            )

    def _draw_winner_icon(self, winner: str | None, center_x: float, center_y: float, size: float) -> None:
        inset = status_icon_inset(float(CELL_INSET))
        if winner is None:
            draw_status_square_icon(
                center_x=float(center_x),
                center_y=float(center_y),
                size=float(size),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
                inset=float(inset),
            )
            return
        if str(winner) == self.PLAYER_ID:
            outer_color = COLOR_AQUA
            inner_color = COLOR_DEEP_TEAL
        else:
            outer_color = COLOR_CORAL
            inner_color = COLOR_BRICK_RED
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(inset),
        )

    def _draw_winner_history(self, left: float, right: float, center_y: float) -> None:
        icon_size = max(12.0, min(float(TILE_SIZE), float(BB_HEIGHT) - 8.0))
        draw_status_icon_row(
            left=float(left),
            right=float(right),
            center_y=float(center_y),
            icon_size=float(icon_size),
            items=list(self.win_history),
            draw_item=lambda winner, icon_center_x, row_center_y, size: self._draw_winner_icon(
                winner,
                float(icon_center_x),
                float(row_center_y),
                float(size),
            ),
        )

    def _remaining_time_ratio(self) -> float:
        return float(self.match_tracker.remaining_time_ratio(int(self.steps)))

    def render(self) -> None:
        if self.window_controller.window is None:
            return

        self.window_controller.clear(COLOR_SLATE_GRAY)
        self._draw_arena()
        if self.player is not None:
            self._draw_rider_trail(self.player)
        if self.opponent is not None:
            self._draw_rider_trail(self.opponent)
        if self.player is not None:
            self._draw_rider_head(self.player)
        if self.opponent is not None:
            self._draw_rider_head(self.opponent)
        self._draw_crash_cells()

        layout = draw_status_bar(
            width=float(SCREEN_WIDTH),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=float(CELL_INSET),
            include_clock=True,
        )
        draw_status_clock(
            layout=layout,
            remaining_ratio=float(self._remaining_time_ratio()),
        )
        self._draw_winner_history(
            left=float(layout.score_left),
            right=float(layout.score_right),
            center_y=float(layout.center_y),
        )
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
        self.window = None
