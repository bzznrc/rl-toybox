"""Procedural micro-platformer environment for Jump."""

from __future__ import annotations

from dataclasses import dataclass
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
    SharedCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.ghost_overlay import draw_ghost_line, draw_ghost_rect, ghost_color, update_ghost_overlay_toggle
from core.io_schema import clip_signed, clip_unit, ordered_feature_vector
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    draw_status_square_icon,
    draw_two_tone_cell,
    draw_two_tone_square_block,
    square_block_inset,
    status_icon_inset,
    status_icon_size,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController, Rect, TextCache
from core.utils import resolve_play_level
from games.jump import config


validate_curriculum_level_settings(
    min_level=config.MIN_LEVEL,
    max_level=config.MAX_LEVEL,
    level_settings=config.LEVEL_SETTINGS,
)


@dataclass(frozen=True)
class TerrainSegment:
    index: int
    left_tile: int
    width_tiles: int
    lane_index: int

    @property
    def left(self) -> float:
        return float(self.left_tile * config.TILE_SIZE)

    @property
    def width(self) -> float:
        return float(self.width_tiles * config.TILE_SIZE)

    @property
    def right(self) -> float:
        return float(self.left + self.width)

    @property
    def lane_surface_row(self) -> int:
        return int(config.LANE_SURFACE_ROWS[int(self.lane_index)])

    @property
    def surface_y(self) -> float:
        return float(self.lane_surface_row * config.TILE_SIZE)

    @property
    def rect(self) -> Rect:
        return Rect(
            left=float(self.left),
            top=float(self.surface_y),
            width=float(self.width),
            height=float(config.PLATFORM_THICKNESS_PX),
        )


@dataclass
class JumpEnemy:
    spawn_index: int
    platform_index: int
    x: float
    vx: float


@dataclass
class MovingPlatform:
    index: int
    source_segment_index: int
    target_segment_index: int
    lane_index: int
    width_tiles: int
    min_left_tile: int
    max_left_tile: int
    x: float
    vx: float

    @property
    def left(self) -> float:
        return float(self.x)

    @property
    def width(self) -> float:
        return float(self.width_tiles * config.TILE_SIZE)

    @property
    def right(self) -> float:
        return float(self.left + self.width)

    @property
    def min_left(self) -> float:
        return float(self.min_left_tile * config.TILE_SIZE)

    @property
    def max_left(self) -> float:
        return float(self.max_left_tile * config.TILE_SIZE)

    @property
    def lane_surface_row(self) -> int:
        return int(config.LANE_SURFACE_ROWS[int(self.lane_index)])

    @property
    def surface_y(self) -> float:
        return float(self.lane_surface_row * config.TILE_SIZE)

    @property
    def rect(self) -> Rect:
        return Rect(
            left=float(self.left),
            top=float(self.surface_y),
            width=float(self.width),
            height=float(config.PLATFORM_THICKNESS_PX),
        )


class JumpEnv(Env):
    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)
    REWARD_COMPONENT_ORDER = ("F", "X", "T", "P", "I")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_finish": "F",
        "outcome.penalty_fail": "X",
        "combat.reward_stomp": "T",
        "progress.shape": "P",
        "progress.penalty_stall": "I",
    }

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            config.WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=False,
            vsync=False,
        )
        self.window = self.window_controller.window
        self._text_cache = TextCache(max_entries=256)

        curriculum_config = build_curriculum_config(
            min_level=int(config.MIN_LEVEL),
            max_level=int(config.MAX_LEVEL),
            promotion_settings=config.CURRICULUM_PROMOTION,
        )
        self._curriculum = (
            SharedCurriculum(config=curriculum_config, level_settings=config.LEVEL_SETTINGS)
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
                default_level=config.MAX_LEVEL,
            )
        )
        self._last_episode_level = int(self._current_level)
        self._last_episode_success = 0
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        self._base_seed = int(random.randint(0, 2_000_000_000))
        self._reset_index = 0
        self.level_seed = 0
        self._history = MatchTracker[bool](
            history_limit=int(config.STATUS_HISTORY_LIMIT),
            clock_duration_steps=None,
        )

        self.world_width_px = float(config.SCREEN_WIDTH)
        self.level_length_tiles = int(config.LEVEL_SETTINGS[int(config.MIN_LEVEL)]["length_tiles"])
        self.max_episode_steps = int(round(float(self.level_length_tiles) * float(config.EPISODE_STEPS_PER_TILE)))
        self.segment_target = max(
            3,
            int(round(float(self.level_length_tiles) / float(config.SEGMENT_TARGET_SPACING_TILES))),
        )
        self.platform_size_tiles = tuple(int(value) for value in config.STANDARD_PLATFORM_SIZE_TILES)
        self.platform_min_width_tiles = min(self.platform_size_tiles)
        self.start_platform_tiles = int(config.STANDARD_START_PLATFORM_TILES)
        self.goal_stretch_tiles = int(config.STANDARD_GOAL_STRETCH_TILES)
        self.gap_min = int(config.BASE_GAP_MIN_TILES)
        self.gap_max = int(config.BASE_GAP_MIN_TILES + config.BASE_GAP_EXTRA_TILES)
        self.max_lane_index = 0
        self.lane_delta_choices = tuple(int(value) for value in config.DEFAULT_LANE_DELTA_CHOICES)
        self.min_upper_segments = 0
        self.min_top_segments = 0
        self.enemy_count_min = 0
        self.enemy_count_max = 0
        self.enemy_spawn_chance = 0.0
        self.moving_platform_frequency = 0.0

        self.segments: list[TerrainSegment] = []
        self.enemies: list[JumpEnemy] = []
        self.moving_platforms: list[MovingPlatform] = []
        self.flag_rect = Rect(0.0, 0.0, 1.0, 1.0)
        self.flag_center_x = 0.0
        self.goal_segment_index = -1
        self.player_spawn_center_x = 0.0
        self.steps = 0
        self.done = False
        self.success = 0
        self.failure_reason = ""
        self.player_x = 0.0
        self.player_y = 0.0
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.player_grounded = True
        self.player_support_index: int | None = None
        self.player_moving_support_index: int | None = None
        self.player_last_support_index: int | None = None
        self._coyote_steps_left = 0
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._best_progress_potential = 0.0
        self._last_step_breakdown = self._empty_reward_breakdown()
        self._prev_jump_down = False
        self.show_ghost_overlay = bool(config.SHOW_GHOST_OVERLAY)
        self._prev_ghost_overlay_toggle_down = False

        self._apply_level_settings(int(self._current_level))
        self.reset()

    @staticmethod
    def _empty_reward_breakdown() -> dict[str, float]:
        return {
            "outcome.reward_finish": 0.0,
            "outcome.penalty_fail": 0.0,
            "combat.reward_stomp": 0.0,
            "progress.shape": 0.0,
            "progress.penalty_stall": 0.0,
        }

    def _segment_target_for_length(self, length_tiles: int) -> int:
        return max(
            3,
            int(round(float(length_tiles) / float(config.SEGMENT_TARGET_SPACING_TILES))),
        )

    def _max_episode_steps_for_length(self, length_tiles: int) -> int:
        return max(1, int(round(float(length_tiles) * float(config.EPISODE_STEPS_PER_TILE))))

    def _derived_level_profile(self, *, level: int, settings: dict[str, object]) -> dict[str, object]:
        length_tiles = max(int(config.MIN_LEVEL_LENGTH_TILES), int(settings["length_tiles"]))
        lane_count = max(1, min(int(config.LANE_COUNT), int(settings.get("lane_count", 2))))
        enemy_frequency = float(max(0.0, min(1.0, float(settings["enemy_frequency"]))))
        moving_platform_frequency = float(max(0.0, min(1.0, float(settings.get("moving_platform_frequency", 0.0)))))
        segment_target = int(self._segment_target_for_length(int(length_tiles)))
        internal_segments = max(1, int(segment_target - 2))
        tutorial_flat = (
            int(lane_count) == 1
            and float(enemy_frequency) <= 0.0
            and float(moving_platform_frequency) <= 0.0
        )
        if tutorial_flat:
            gap_min = 0
            gap_max = 0
        else:
            gap_min = max(
                int(config.PLAYER_TILES),
                int(config.BASE_GAP_MIN_TILES)
                + (int(config.TOP_LEVEL_EXTRA_GAP_MIN_TILES) if int(level) >= int(config.MAX_LEVEL) else 0),
            )
            gap_max = max(
                int(gap_min),
                int(gap_min + config.BASE_GAP_EXTRA_TILES)
                + (int(config.ADVANCED_GAP_MAX_BONUS_TILES) if int(level) >= 2 else 0),
            )
        lane_delta_choices = (
            tuple(int(value) for value in config.ADVANCED_LANE_DELTA_CHOICES)
            if int(lane_count) >= 3 and int(level) >= int(config.MAX_LEVEL)
            else tuple(int(value) for value in config.DEFAULT_LANE_DELTA_CHOICES)
        )
        enemy_budget = min(
            int(internal_segments),
            max(
                0,
                int(
                    round(
                        (float(length_tiles) / float(config.ENEMY_SPACING_TILES))
                        * float(enemy_frequency)
                    )
                ),
            ),
        )
        return {
            "length_tiles": int(length_tiles),
            "segment_target": int(segment_target),
            "platform_size_tiles": tuple(int(value) for value in config.STANDARD_PLATFORM_SIZE_TILES),
            "start_platform_tiles": int(config.STANDARD_START_PLATFORM_TILES),
            "goal_stretch_tiles": int(config.STANDARD_GOAL_STRETCH_TILES),
            "gap_min": int(gap_min),
            "gap_max": int(gap_max),
            "max_lane_index": int(lane_count - 1),
            "lane_delta_choices": tuple(int(value) for value in lane_delta_choices),
            "min_upper_segments": max(
                0,
                min(int(internal_segments), int(level) if int(lane_count) >= 2 else 0),
            ),
            "min_top_segments": 1 if int(lane_count) >= 3 else 0,
            "enemy_count_min": max(0, int(enemy_budget - 1)),
            "enemy_count_max": int(enemy_budget),
            "enemy_spawn_chance": float(enemy_frequency),
            "moving_platform_frequency": float(moving_platform_frequency),
            "max_episode_steps": int(self._max_episode_steps_for_length(int(length_tiles))),
        }

    def _apply_level_settings(self, level: int) -> None:
        settings = dict(config.LEVEL_SETTINGS.get(int(level), config.LEVEL_SETTINGS[int(config.MIN_LEVEL)]))
        profile = self._derived_level_profile(level=int(level), settings=settings)
        self.level_length_tiles = int(profile["length_tiles"])
        self.world_width_px = float(int(self.level_length_tiles) * config.TILE_SIZE)
        self.segment_target = int(profile["segment_target"])
        platform_sizes = tuple(sorted({max(4, int(value)) for value in profile["platform_size_tiles"]}))
        if len(platform_sizes) != 3:
            raise RuntimeError(f"Jump expected exactly 3 platform sizes, got {platform_sizes}.")
        self.platform_size_tiles = platform_sizes
        self.platform_min_width_tiles = int(self.platform_size_tiles[0])
        self.start_platform_tiles = int(profile["start_platform_tiles"])
        self.goal_stretch_tiles = int(profile["goal_stretch_tiles"])
        if self.start_platform_tiles not in self.platform_size_tiles:
            raise RuntimeError("Jump start platform width must be one of the configured platform sizes.")
        if self.goal_stretch_tiles not in self.platform_size_tiles:
            raise RuntimeError("Jump goal platform width must be one of the configured platform sizes.")
        self.gap_min = max(0, int(profile["gap_min"]))
        self.gap_max = max(self.gap_min, int(profile["gap_max"]))
        self.max_lane_index = max(0, min(int(config.LANE_COUNT - 1), int(profile["max_lane_index"])))
        self.lane_delta_choices = tuple(int(value) for value in profile["lane_delta_choices"])
        self.min_upper_segments = max(0, int(profile["min_upper_segments"]))
        self.min_top_segments = max(0, int(profile["min_top_segments"]))
        self.enemy_count_min = max(0, int(profile["enemy_count_min"]))
        self.enemy_count_max = max(self.enemy_count_min, int(profile["enemy_count_max"]))
        self.enemy_spawn_chance = float(max(0.0, min(1.0, float(profile["enemy_spawn_chance"]))))
        self.moving_platform_frequency = float(max(0.0, min(1.0, float(profile["moving_platform_frequency"]))))
        self.max_episode_steps = max(1, int(profile["max_episode_steps"]))
        self._history.set_clock_duration(int(self.max_episode_steps))

    def _platform_width_choices_up_to(self, max_width_tiles: int) -> list[int]:
        return [
            int(width_tiles)
            for width_tiles in self.platform_size_tiles
            if int(width_tiles) <= int(max_width_tiles)
        ]

    def _episode_seed(self) -> int:
        return int(
            self._base_seed
            + (int(self._current_level) * 100_003)
            + (int(self._reset_index) * 1_000_003)
        )

    def _sample_gap_tiles(self, rng: random.Random) -> int:
        return int(rng.randint(int(self.gap_min), int(self.gap_max)))

    def _transition_is_reachable(
        self,
        *,
        from_lane_index: int,
        to_lane_index: int,
        gap_tiles: int,
        landing_width_tiles: int,
    ) -> bool:
        from_surface_y = float(config.LANE_SURFACE_ROWS[int(from_lane_index)] * config.TILE_SIZE)
        to_surface_y = float(config.LANE_SURFACE_ROWS[int(to_lane_index)] * config.TILE_SIZE)
        gap_px = float(max(0, int(gap_tiles)) * config.TILE_SIZE)
        landing_width_px = float(max(1, int(landing_width_tiles)) * config.TILE_SIZE)

        x = -float(config.PLAYER_SIZE + (config.GENERATION_RUNWAY_TILES * config.TILE_SIZE))
        y = float(from_surface_y - config.PLAYER_SIZE)
        vy = -float(config.JUMP_VELOCITY_PX_PER_SEC)
        dt = float(config.PHYSICS_DT)
        steps = max(1, int(round(2.0 / max(1e-6, dt))))

        for _ in range(steps):
            prev_y = float(y)
            x += float(config.PLAYER_RUN_SPEED_PX_PER_SEC) * dt
            vy = min(float(config.MAX_FALL_SPEED_PX_PER_SEC), float(vy + config.GRAVITY_PX_PER_SEC2 * dt))
            y += vy * dt
            if float(vy) < 0.0:
                continue

            overlap_left = max(float(x), float(gap_px))
            overlap_right = min(float(x + config.PLAYER_SIZE), float(gap_px + landing_width_px))
            if overlap_right <= overlap_left:
                continue

            prev_bottom = float(prev_y + config.PLAYER_SIZE)
            curr_bottom = float(y + config.PLAYER_SIZE)
            if prev_bottom <= float(to_surface_y) <= curr_bottom:
                return True
            if y > float(config.PLAYFIELD_HEIGHT + config.PLAYER_SIZE):
                break
        return False

    def _build_lane_plan(self, rng: random.Random) -> list[int]:
        lane_plan = [0]
        target_count = max(3, int(self.segment_target))
        for step_idx in range(1, target_count - 1):
            current_lane = int(lane_plan[-1])
            remaining_transitions_to_goal = int(target_count - 1 - step_idx)
            candidates: list[int] = []
            for lane_delta in self.lane_delta_choices:
                raw_candidate = int(current_lane + int(lane_delta))
                if raw_candidate < 0 or raw_candidate > int(self.max_lane_index):
                    continue
                candidate_lane = int(raw_candidate)
                if abs(int(candidate_lane)) > int(remaining_transitions_to_goal):
                    continue
                candidates.append(int(candidate_lane))
            if not candidates:
                raise ValueError("Jump lane plan generation found no valid candidates.")
            lane_plan.append(int(rng.choice(candidates)))
        lane_plan.append(0)

        upper_segments = sum(1 for lane_index in lane_plan[1:-1] if int(lane_index) >= 1)
        top_segments = sum(1 for lane_index in lane_plan[1:-1] if int(lane_index) >= 2)
        if upper_segments < int(self.min_upper_segments):
            raise ValueError("Jump lane plan did not include enough elevated platforms.")
        if top_segments < int(self.min_top_segments):
            raise ValueError("Jump lane plan did not include enough top-lane platforms.")
        return lane_plan

    def _future_min_tiles(self, remaining_internal: int) -> int:
        return (
            int(remaining_internal) * int(self.platform_min_width_tiles + self.gap_min)
            + int(self.gap_min)
            + int(self.goal_stretch_tiles)
        )

    def _width_choices_after_gap(
        self,
        *,
        world_width_tiles: int,
        current_left_tile: int,
        gap_tiles: int,
        remaining_internal: int,
    ) -> list[int]:
        future_min_tiles = int(self._future_min_tiles(int(remaining_internal)))
        max_width_tiles = int(world_width_tiles - (current_left_tile + gap_tiles + future_min_tiles))
        return self._platform_width_choices_up_to(int(max_width_tiles))

    def _static_transition_candidate(
        self,
        *,
        rng: random.Random,
        previous_segment: TerrainSegment,
        next_lane: int,
        current_left_tile: int,
        world_width_tiles: int,
        remaining_internal: int,
        target_segment_index: int,
    ) -> TerrainSegment | None:
        gap_tiles = self._sample_gap_tiles(rng)
        width_choices = self._width_choices_after_gap(
            world_width_tiles=int(world_width_tiles),
            current_left_tile=int(current_left_tile),
            gap_tiles=int(gap_tiles),
            remaining_internal=int(remaining_internal),
        )
        if not width_choices:
            return None
        next_width = int(rng.choice(width_choices))
        if not self._transition_is_reachable(
            from_lane_index=int(previous_segment.lane_index),
            to_lane_index=int(next_lane),
            gap_tiles=int(gap_tiles),
            landing_width_tiles=int(next_width),
        ):
            return None
        return TerrainSegment(
            index=int(target_segment_index),
            left_tile=int(current_left_tile + gap_tiles),
            width_tiles=int(next_width),
            lane_index=int(next_lane),
        )

    def _moving_transition_candidate(
        self,
        *,
        rng: random.Random,
        previous_segment: TerrainSegment,
        next_lane: int,
        current_left_tile: int,
        world_width_tiles: int,
        remaining_internal: int,
        target_segment_index: int,
        moving_platform_index: int,
    ) -> tuple[TerrainSegment, MovingPlatform] | None:
        source_lane_index = int(previous_segment.lane_index)
        moving_platform_width_choices = [
            int(width_tiles)
            for width_tiles in config.MOVING_PLATFORM_SIZE_TILES
            if int(width_tiles) > 0
        ]
        if not moving_platform_width_choices:
            raise ValueError("Jump moving platforms require at least one configured width.")
        platform_width_tiles = int(rng.choice(moving_platform_width_choices))
        entry_gap_tiles = int(
            rng.randint(
                int(config.MOVING_PLATFORM_ENTRY_GAP_MIN_TILES),
                int(config.MOVING_PLATFORM_ENTRY_GAP_MAX_TILES),
            )
        )
        exit_gap_tiles = int(
            rng.randint(
                int(config.MOVING_PLATFORM_EXIT_GAP_MIN_TILES),
                int(config.MOVING_PLATFORM_EXIT_GAP_MAX_TILES),
            )
        )
        travel_tiles = int(
            rng.randint(
                int(config.MOVING_PLATFORM_TRAVEL_MIN_TILES),
                int(config.MOVING_PLATFORM_TRAVEL_MAX_TILES),
            )
        )
        total_gap_tiles = int(entry_gap_tiles + platform_width_tiles + travel_tiles + exit_gap_tiles)
        width_choices = self._width_choices_after_gap(
            world_width_tiles=int(world_width_tiles),
            current_left_tile=int(current_left_tile),
            gap_tiles=int(total_gap_tiles),
            remaining_internal=int(remaining_internal),
        )
        if not width_choices:
            return None

        next_width = int(rng.choice(width_choices))
        if self._transition_is_reachable(
            from_lane_index=int(source_lane_index),
            to_lane_index=int(next_lane),
            gap_tiles=int(total_gap_tiles),
            landing_width_tiles=int(next_width),
        ):
            return None
        if not self._transition_is_reachable(
            from_lane_index=int(source_lane_index),
            to_lane_index=int(source_lane_index),
            gap_tiles=int(entry_gap_tiles),
            landing_width_tiles=int(platform_width_tiles),
        ):
            return None
        if not self._transition_is_reachable(
            from_lane_index=int(source_lane_index),
            to_lane_index=int(next_lane),
            gap_tiles=int(exit_gap_tiles),
            landing_width_tiles=int(next_width),
        ):
            return None

        left_tile = int(current_left_tile + total_gap_tiles)
        min_left_tile = int(current_left_tile + entry_gap_tiles)
        max_left_tile = int(left_tile - exit_gap_tiles - platform_width_tiles)
        if int(max_left_tile) <= int(min_left_tile):
            return None

        next_segment = TerrainSegment(
            index=int(target_segment_index),
            left_tile=int(left_tile),
            width_tiles=int(next_width),
            lane_index=int(next_lane),
        )
        start_left_tile = int(rng.randint(int(min_left_tile), int(max_left_tile)))
        start_direction = 1.0 if bool(rng.randint(0, 1)) else -1.0
        moving_platform = MovingPlatform(
            index=int(moving_platform_index),
            source_segment_index=int(previous_segment.index),
            target_segment_index=int(target_segment_index),
            lane_index=int(source_lane_index),
            width_tiles=int(platform_width_tiles),
            min_left_tile=int(min_left_tile),
            max_left_tile=int(max_left_tile),
            x=float(start_left_tile * config.TILE_SIZE),
            vx=float(start_direction * config.MOVING_PLATFORM_SPEED_PX_PER_SEC),
        )
        return next_segment, moving_platform

    def _build_segments(self, rng: random.Random) -> tuple[list[TerrainSegment], list[MovingPlatform]]:
        lane_plan = self._build_lane_plan(rng)
        world_width_tiles = int(round(self.world_width_px / config.TILE_SIZE))
        segments: list[TerrainSegment] = []
        moving_platforms: list[MovingPlatform] = []
        start_width_tiles = int(self.start_platform_tiles)
        current_left_tile = 0
        segments.append(
            TerrainSegment(
                index=0,
                left_tile=int(current_left_tile),
                width_tiles=int(start_width_tiles),
                lane_index=0,
            )
        )
        current_left_tile += int(start_width_tiles)

        internal_lanes = list(lane_plan[1:-1])
        for internal_idx, next_lane in enumerate(internal_lanes):
            accepted = False
            previous_segment = segments[-1]
            remaining_internal = int(len(internal_lanes) - internal_idx - 1)
            candidate_modes = (
                (True, False)
                if float(rng.random()) < float(self.moving_platform_frequency)
                else (False, True)
            )
            for use_moving_platform in candidate_modes:
                for _ in range(40):
                    target_segment_index = int(len(segments))
                    if bool(use_moving_platform):
                        transition = self._moving_transition_candidate(
                            rng=rng,
                            previous_segment=previous_segment,
                            next_lane=int(next_lane),
                            current_left_tile=int(current_left_tile),
                            world_width_tiles=int(world_width_tiles),
                            remaining_internal=int(remaining_internal),
                            target_segment_index=int(target_segment_index),
                            moving_platform_index=int(len(moving_platforms)),
                        )
                        if transition is None:
                            continue
                        next_segment, moving_platform = transition
                        moving_platforms.append(moving_platform)
                    else:
                        next_segment = self._static_transition_candidate(
                            rng=rng,
                            previous_segment=previous_segment,
                            next_lane=int(next_lane),
                            current_left_tile=int(current_left_tile),
                            world_width_tiles=int(world_width_tiles),
                            remaining_internal=int(remaining_internal),
                            target_segment_index=int(target_segment_index),
                        )
                        if next_segment is None:
                            continue
                    segments.append(next_segment)
                    current_left_tile = int(next_segment.left_tile + next_segment.width_tiles)
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                raise ValueError("Jump generation failed to place an internal platform.")

        goal_gap_choices = list(range(int(self.gap_min), int(self.gap_max) + 1))
        rng.shuffle(goal_gap_choices)
        goal_added = False
        for goal_gap in goal_gap_choices:
            goal_left_tile = int(current_left_tile + goal_gap)
            remaining_tiles = int(world_width_tiles - goal_left_tile)
            if remaining_tiles < int(self.goal_stretch_tiles):
                continue
            goal_width = int(self.goal_stretch_tiles)
            if self._transition_is_reachable(
                from_lane_index=int(segments[-1].lane_index),
                to_lane_index=0,
                gap_tiles=int(goal_gap),
                landing_width_tiles=int(goal_width),
            ):
                segments.append(
                    TerrainSegment(
                        index=int(len(segments)),
                        left_tile=int(goal_left_tile),
                        width_tiles=int(goal_width),
                        lane_index=0,
                    )
                )
                goal_added = True
                break
        if len(segments) < 2 or not bool(goal_added):
            raise ValueError("Jump generation failed to create enough terrain segments.")
        return segments, moving_platforms

    def _validate_route(
        self,
        segments: list[TerrainSegment],
        moving_platforms: list[MovingPlatform],
    ) -> None:
        if len(segments) < 2:
            raise ValueError("Jump requires at least start and goal segments.")
        moving_by_transition = {
            (int(platform.source_segment_index), int(platform.target_segment_index)): platform
            for platform in moving_platforms
        }
        if len(moving_by_transition) != len(moving_platforms):
            raise ValueError("Jump moving-platform transitions must be unique.")
        for idx in range(len(segments) - 1):
            current = segments[idx]
            nxt = segments[idx + 1]
            moving_platform = moving_by_transition.get((int(current.index), int(nxt.index)))
            if moving_platform is not None:
                total_gap_tiles = max(0, int(round((nxt.left - current.right) / config.TILE_SIZE)))
                if self._transition_is_reachable(
                    from_lane_index=int(current.lane_index),
                    to_lane_index=int(nxt.lane_index),
                    gap_tiles=int(total_gap_tiles),
                    landing_width_tiles=int(nxt.width_tiles),
                ):
                    raise ValueError(
                        f"Jump moving-platform transition {idx}->{idx + 1} should not be directly reachable."
                    )

                entry_gap_tiles = max(
                    0,
                    int(round((moving_platform.min_left - current.right) / config.TILE_SIZE)),
                )
                exit_gap_tiles = max(
                    0,
                    int(round((nxt.left - (moving_platform.max_left + moving_platform.width)) / config.TILE_SIZE)),
                )
                if not self._transition_is_reachable(
                    from_lane_index=int(current.lane_index),
                    to_lane_index=int(moving_platform.lane_index),
                    gap_tiles=int(entry_gap_tiles),
                    landing_width_tiles=int(moving_platform.width_tiles),
                ):
                    raise ValueError(f"Jump moving-platform entry {idx}->{idx + 1} is not reachable.")
                if not self._transition_is_reachable(
                    from_lane_index=int(moving_platform.lane_index),
                    to_lane_index=int(nxt.lane_index),
                    gap_tiles=int(exit_gap_tiles),
                    landing_width_tiles=int(nxt.width_tiles),
                ):
                    raise ValueError(f"Jump moving-platform exit {idx}->{idx + 1} is not reachable.")
                continue

            gap_tiles = max(0, int(round((nxt.left - current.right) / config.TILE_SIZE)))
            if not self._transition_is_reachable(
                from_lane_index=int(current.lane_index),
                to_lane_index=int(nxt.lane_index),
                gap_tiles=int(gap_tiles),
                landing_width_tiles=int(nxt.width_tiles),
            ):
                raise ValueError(f"Jump segment transition {idx}->{idx + 1} is not reachable.")

    def _spawn_enemies(self, rng: random.Random, segments: list[TerrainSegment]) -> list[JumpEnemy]:
        target_count = int(rng.randint(int(self.enemy_count_min), int(self.enemy_count_max)))
        if target_count <= 0:
            return []

        candidates = [segment for segment in segments[1:-1] if segment.width_tiles >= int(config.ENEMY_TILES)]
        rng.shuffle(candidates)
        enemies: list[JumpEnemy] = []
        for segment in candidates:
            if len(enemies) >= int(target_count):
                break
            if rng.random() > float(self.enemy_spawn_chance):
                continue
            min_x = float(segment.left)
            max_x = float(segment.right - config.ENEMY_SIZE)
            if max_x <= min_x:
                continue
            enemies.append(
                JumpEnemy(
                    spawn_index=int(len(enemies)),
                    platform_index=int(segment.index),
                    x=float(rng.uniform(min_x, max_x)),
                    vx=float(rng.choice((-1.0, 1.0))) * float(config.ENEMY_RUN_SPEED_PX_PER_SEC),
                )
            )
        return enemies

    def _generate_level(self, seed: int) -> None:
        attempt_seed = int(seed)
        for _ in range(int(config.LEVEL_GENERATION_ATTEMPTS)):
            rng = random.Random(int(attempt_seed))
            try:
                segments, moving_platforms = self._build_segments(rng)
                self._validate_route(segments, moving_platforms)
                enemies = self._spawn_enemies(rng, segments)
                break
            except (RuntimeError, ValueError):
                attempt_seed += 1
                continue
        else:
            raise RuntimeError("Jump failed to generate a reachable procedural level.")

        self.segments = list(segments)
        self.enemies = list(enemies)
        self.moving_platforms = list(moving_platforms)
        self.goal_segment_index = int(self.segments[-1].index)
        goal_segment = self.segments[self.goal_segment_index]
        self.world_width_px = float(max(config.SCREEN_WIDTH, goal_segment.right + (2.0 * config.TILE_SIZE)))

        start_segment = self.segments[0]
        self.player_x = float(start_segment.left + 2.0 * config.TILE_SIZE)
        self.player_y = float(start_segment.surface_y - config.PLAYER_SIZE)
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.player_grounded = True
        self.player_support_index = int(start_segment.index)
        self.player_moving_support_index = None
        self.player_last_support_index = int(start_segment.index)
        self._coyote_steps_left = int(config.COYOTE_TIME_STEPS)

        pole_height = float(config.GOAL_FLAG_HEIGHT_TILES * config.TILE_SIZE)
        flag_width = float(config.GOAL_FLAG_WIDTH_TILES * config.TILE_SIZE)
        flag_left = float(goal_segment.right - 2.0 * config.TILE_SIZE)
        flag_top = float(goal_segment.surface_y - pole_height)
        self.flag_rect = Rect(
            left=float(flag_left),
            top=float(flag_top),
            width=float(flag_width),
            height=float(flag_width),
        )
        self.flag_center_x = float(self.flag_rect.left + self.flag_rect.width * 0.5)
        self.player_spawn_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)

    def _player_rect(self) -> Rect:
        return Rect(
            left=float(self.player_x),
            top=float(self.player_y),
            width=float(config.PLAYER_SIZE),
            height=float(config.PLAYER_SIZE),
        )

    def _enemy_rect(self, enemy: JumpEnemy) -> Rect:
        segment = self.segments[int(enemy.platform_index)]
        return Rect(
            left=float(enemy.x),
            top=float(segment.surface_y - config.ENEMY_SIZE),
            width=float(config.ENEMY_SIZE),
            height=float(config.ENEMY_SIZE),
        )

    def _support_surfaces(self) -> list[TerrainSegment | MovingPlatform]:
        return [*self.segments, *self.moving_platforms]

    def _rect_collides_supports(self, rect: Rect) -> list[TerrainSegment | MovingPlatform]:
        return [surface for surface in self._support_surfaces() if rect.colliderect(surface.rect)]

    def _support_surface_below(
        self,
        *,
        tolerance_px: float = 0.0,
    ) -> TerrainSegment | MovingPlatform | None:
        player_rect = self._player_rect()
        best_surface: TerrainSegment | MovingPlatform | None = None
        best_score: tuple[float, float, int, int] | None = None
        feet_y = float(player_rect.bottom)
        for surface in self._support_surfaces():
            if player_rect.right <= float(surface.left + 2.0):
                continue
            if player_rect.left >= float(surface.right - 2.0):
                continue
            gap = float(surface.surface_y - feet_y)
            if gap < -1.0:
                continue
            if gap > float(tolerance_px):
                continue
            score = (
                float(gap),
                float(abs((surface.left + surface.right) * 0.5 - (player_rect.left + player_rect.right) * 0.5)),
                0 if isinstance(surface, MovingPlatform) else 1,
                int(surface.index),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_surface = surface
        return best_surface

    @staticmethod
    def _clip_action(action_idx: int) -> int:
        return max(0, min(int(action_idx), int(config.ACT_DIM) - 1))

    def _parse_action(self, action) -> int:
        if isinstance(action, np.ndarray):
            flat = np.asarray(action).reshape(-1)
            if flat.size <= 0:
                return int(config.ACTION_MOVE_STOP)
            return self._clip_action(int(flat[0]))
        return self._clip_action(int(action))

    def get_action_mask(self, _obs: object | None = None) -> np.ndarray:
        mask = np.ones((int(config.ACT_DIM),), dtype=np.bool_)
        if not self._can_jump():
            mask[int(config.ACTION_JUMP)] = False
        return mask

    def _resolve_human_action(self) -> int:
        left = self.window_controller.is_key_down(arcade.key.A) or self.window_controller.is_key_down(arcade.key.LEFT)
        right = self.window_controller.is_key_down(arcade.key.D) or self.window_controller.is_key_down(arcade.key.RIGHT)
        jump_down = (
            self.window_controller.is_key_down(arcade.key.W)
            or self.window_controller.is_key_down(arcade.key.UP)
            or self.window_controller.is_key_down(arcade.key.SPACE)
        )
        jump_pressed = bool(jump_down and not self._prev_jump_down)
        self._prev_jump_down = bool(jump_down)
        if jump_pressed:
            return int(config.ACTION_JUMP)
        if left and not right:
            return int(config.ACTION_MOVE_LEFT)
        if right and not left:
            return int(config.ACTION_MOVE_RIGHT)
        return int(config.ACTION_MOVE_STOP)

    def _can_toggle_visual_overlay(self) -> bool:
        return bool(self.show_game and self.mode in {"human", "eval"})

    def _update_visual_overlay_toggle(self) -> None:
        self.show_ghost_overlay, self._prev_ghost_overlay_toggle_down = update_ghost_overlay_toggle(
            window_controller=self.window_controller,
            visible=bool(self.show_ghost_overlay),
            previous_down=bool(self._prev_ghost_overlay_toggle_down),
            enabled=bool(self._can_toggle_visual_overlay()),
        )

    def _should_draw_ghost_overlay(self) -> bool:
        return bool(self.show_ghost_overlay and self.show_game and self.mode != "train")

    def _apply_action(self, action_idx: int) -> None:
        if int(action_idx) == int(config.ACTION_MOVE_LEFT):
            self.player_vx = -float(config.PLAYER_RUN_SPEED_PX_PER_SEC)
        elif int(action_idx) == int(config.ACTION_MOVE_RIGHT):
            self.player_vx = float(config.PLAYER_RUN_SPEED_PX_PER_SEC)
        elif int(action_idx) == int(config.ACTION_MOVE_STOP):
            self.player_vx = 0.0
        elif int(action_idx) == int(config.ACTION_JUMP):
            if self._can_jump():
                self.player_vy = -float(config.JUMP_VELOCITY_PX_PER_SEC)
                self.player_grounded = False
                self.player_support_index = None
                self.player_moving_support_index = None
                self._coyote_steps_left = 0

    def _can_jump(self) -> bool:
        return bool(self.player_grounded or int(self._coyote_steps_left) > 0)

    def _step_moving_platforms(self) -> None:
        carry_dx = 0.0
        supported_platform_index = (
            None if self.player_moving_support_index is None else int(self.player_moving_support_index)
        )
        for platform in self.moving_platforms:
            prev_x = float(platform.x)
            next_x = float(platform.x + platform.vx * config.PHYSICS_DT)
            if next_x <= float(platform.min_left):
                next_x = float(platform.min_left)
                platform.vx = abs(float(config.MOVING_PLATFORM_SPEED_PX_PER_SEC))
            elif next_x >= float(platform.max_left):
                next_x = float(platform.max_left)
                platform.vx = -abs(float(config.MOVING_PLATFORM_SPEED_PX_PER_SEC))
            platform.x = float(next_x)
            if (
                supported_platform_index is not None
                and bool(self.player_grounded)
                and int(platform.index) == int(supported_platform_index)
            ):
                carry_dx = float(platform.x - prev_x)
        if float(carry_dx) != 0.0:
            self.player_x = float(max(0.0, min(self.player_x + carry_dx, self.world_width_px - config.PLAYER_SIZE)))

    def _step_player(self) -> None:
        dt = float(config.PHYSICS_DT)
        prev_rect = self._player_rect()

        next_x = float(self.player_x + self.player_vx * dt)
        next_x = float(max(0.0, min(next_x, self.world_width_px - config.PLAYER_SIZE)))
        test_rect = Rect(next_x, float(self.player_y), float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
        colliders = self._rect_collides_supports(test_rect)
        if float(self.player_vx) > 0.0:
            for surface in sorted(colliders, key=lambda item: float(item.left)):
                if prev_rect.right <= float(surface.left) and test_rect.right > float(surface.left):
                    next_x = float(surface.left - config.PLAYER_SIZE)
                    test_rect = Rect(next_x, float(self.player_y), float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
                    break
        elif float(self.player_vx) < 0.0:
            for surface in sorted(colliders, key=lambda item: float(item.right), reverse=True):
                if prev_rect.left >= float(surface.right) and test_rect.left < float(surface.right):
                    next_x = float(surface.right)
                    test_rect = Rect(next_x, float(self.player_y), float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
                    break
        self.player_x = float(next_x)

        self.player_vy = min(
            float(config.MAX_FALL_SPEED_PX_PER_SEC),
            float(self.player_vy + config.GRAVITY_PX_PER_SEC2 * dt),
        )
        next_y = float(self.player_y + self.player_vy * dt)
        test_rect = Rect(float(self.player_x), next_y, float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
        grounded = False
        landed_surface: TerrainSegment | MovingPlatform | None = None
        colliders = self._rect_collides_supports(test_rect)
        if float(self.player_vy) < 0.0:
            for surface in sorted(colliders, key=lambda item: float(item.rect.bottom), reverse=True):
                surface_bottom = float(surface.rect.bottom)
                if prev_rect.top >= surface_bottom and test_rect.top < surface_bottom:
                    next_y = float(surface_bottom)
                    self.player_vy = 0.0
                    test_rect = Rect(float(self.player_x), next_y, float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
                    break
        elif float(self.player_vy) >= 0.0:
            for surface in sorted(colliders, key=lambda item: float(item.surface_y)):
                if prev_rect.bottom <= float(surface.surface_y) and test_rect.bottom >= float(surface.surface_y):
                    next_y = float(surface.surface_y - config.PLAYER_SIZE)
                    self.player_vy = 0.0
                    grounded = True
                    landed_surface = surface
                    break
        self.player_y = float(next_y)

        support_surface = landed_surface
        if support_surface is None:
            support_surface = self._support_surface_below(tolerance_px=float(config.GROUND_SNAP_PX))
            if support_surface is not None and float(self.player_vy) >= 0.0:
                self.player_y = float(support_surface.surface_y - config.PLAYER_SIZE)
                self.player_vy = 0.0
                grounded = True

        self.player_grounded = bool(grounded)
        self.player_support_index = None
        self.player_moving_support_index = None
        if self.player_grounded:
            if isinstance(support_surface, TerrainSegment):
                self.player_support_index = int(support_surface.index)
                self.player_last_support_index = int(support_surface.index)
            elif isinstance(support_surface, MovingPlatform):
                self.player_moving_support_index = int(support_surface.index)
            self._coyote_steps_left = int(config.COYOTE_TIME_STEPS)
        else:
            self._coyote_steps_left = max(0, int(self._coyote_steps_left) - 1)

    def _step_enemies(self) -> None:
        for enemy in self.enemies:
            segment = self.segments[int(enemy.platform_index)]
            min_x = float(segment.left)
            max_x = float(segment.right - config.ENEMY_SIZE)
            if max_x <= min_x:
                enemy.x = float(min_x)
                enemy.vx = 0.0
                continue
            next_x = float(enemy.x + enemy.vx * config.PHYSICS_DT)
            if next_x <= min_x:
                next_x = float(min_x)
                enemy.vx = abs(float(config.ENEMY_RUN_SPEED_PX_PER_SEC))
            elif next_x >= max_x:
                next_x = float(max_x)
                enemy.vx = -abs(float(config.ENEMY_RUN_SPEED_PX_PER_SEC))
            enemy.x = float(next_x)

    def _player_flag_progress(self) -> float:
        denom = max(1.0, float(self.flag_center_x - self.player_spawn_center_x))
        center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        return float(clip_unit((center_x - self.player_spawn_center_x) / denom))

    def _time_left_ratio(self) -> float:
        return float(self._history.remaining_time_ratio(int(self.steps)))

    def _route_transition_label(self, prev_support: int | None, next_support: int | None) -> str:
        if prev_support is None and next_support is None:
            return "air"
        if prev_support is None:
            return "land"
        if next_support is None:
            return "jump"
        if int(next_support) > int(prev_support):
            return "advance"
        if int(next_support) < int(prev_support):
            return "back"
        return "same"

    def _sens_ghost_probe_states(self) -> list[tuple[str, Rect, bool]]:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        tile_size = float(config.TILE_SIZE)
        current_lane = self._lane_index_for_y(float(self.player_y))
        fallback_surface_y = float(config.LANE_SURFACE_ROWS[int(current_lane)] * config.TILE_SIZE)
        probe_states: list[tuple[str, Rect, bool]] = []

        def probe_rect(sample_x: float, surface: TerrainSegment | MovingPlatform | None) -> Rect:
            surface_y = fallback_surface_y if surface is None else float(surface.surface_y)
            return Rect(
                left=float(sample_x - tile_size * 0.5),
                top=float(surface_y - tile_size),
                width=float(tile_size),
                height=float(tile_size),
            )

        for label, offset_tiles in (("ground_l2", -4), ("ground_l1", -2), ("ground_c0", 0), ("ground_r1", 2), ("ground_r2", 4)):
            sample_x = float(player_center_x + tile_size * float(offset_tiles))
            support = self._support_at_x(float(sample_x))
            probe_states.append((f"sens_{label}", probe_rect(float(sample_x), support), support is not None))

        for idx, offset_tiles in enumerate((4, 8, 12), start=1):
            sample_x = float(player_center_x + tile_size * float(offset_tiles))
            support = self._support_at_x(float(sample_x))
            probe_states.append((f"sens_gap_f{idx}", probe_rect(float(sample_x), support), support is None))

        return probe_states

    @staticmethod
    def _moving_platform_anchor(platform: MovingPlatform) -> tuple[float, float]:
        return (
            float(platform.left + platform.width * 0.5),
            float(platform.surface_y - config.PLAYER_SIZE * 0.5),
        )

    def _closest_relevant_moving_platform(self) -> MovingPlatform | None:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        best_match: tuple[tuple[float, float, float, int], MovingPlatform] | None = None
        for platform in self.moving_platforms:
            anchor_x, anchor_y = self._moving_platform_anchor(platform)
            dx = float(anchor_x - player_center_x)
            dy = float(anchor_y - player_center_y)
            if abs(float(dx)) > float(config.LOCAL_DX_NORM_PX):
                continue
            if abs(float(dy)) > float(config.LOCAL_DY_NORM_PX):
                continue
            score = (
                float((dx * dx) + (dy * dy)),
                float(abs(dx)),
                float(abs(dy)),
                int(platform.index),
            )
            if best_match is None or score < best_match[0]:
                best_match = (score, platform)
        return None if best_match is None else best_match[1]

    def _moving_platform_feature_values(self) -> dict[str, float]:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        platform = self._closest_relevant_moving_platform()
        if platform is None:
            return {
                "land_move_dx": 0.0,
                "land_move_dy": 0.0,
                "land_move_vx_norm": 0.0,
                "land_move_phase": 0.0,
            }

        anchor_x, anchor_y = self._moving_platform_anchor(platform)
        platform_speed_norm = float(max(1.0, config.MOVING_PLATFORM_SPEED_PX_PER_SEC))
        return {
            "land_move_dx": float(
                clip_signed((float(anchor_x) - player_center_x) / float(config.LOCAL_DX_NORM_PX))
            ),
            "land_move_dy": float(
                clip_signed((float(anchor_y) - player_center_y) / float(config.LOCAL_DY_NORM_PX))
            ),
            "land_move_vx_norm": float(clip_signed(float(platform.vx) / platform_speed_norm)),
            "land_move_phase": float(
                np.clip(
                    (float(platform.left) - float(platform.min_left))
                    / max(1.0, float(platform.max_left) - float(platform.min_left)),
                    0.0,
                    1.0,
                )
            ),
        }

    def _closest_relevant_enemy(self) -> JumpEnemy | None:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        best_match: tuple[tuple[float, float, float, int], JumpEnemy] | None = None
        for enemy in self.enemies:
            rect = self._enemy_rect(enemy)
            enemy_center_x = float(rect.left + rect.width * 0.5)
            enemy_center_y = float(rect.top + rect.height * 0.5)
            dx = float(enemy_center_x - player_center_x)
            dy = float(enemy_center_y - player_center_y)
            if abs(float(dx)) > float(config.LOCAL_DX_NORM_PX):
                continue
            if abs(float(dy)) > float(config.LOCAL_DY_NORM_PX):
                continue
            score = (
                float((dx * dx) + (dy * dy)),
                float(abs(dx)),
                float(abs(dy)),
                int(enemy.spawn_index),
            )
            if best_match is None or score < best_match[0]:
                best_match = (score, enemy)
        return None if best_match is None else best_match[1]

    def _closest_relevant_enemies(self, *, k: int) -> list[JumpEnemy]:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        scored: list[tuple[tuple[float, float, float, int], JumpEnemy]] = []
        for enemy in self.enemies:
            rect = self._enemy_rect(enemy)
            enemy_center_x = float(rect.left + rect.width * 0.5)
            enemy_center_y = float(rect.top + rect.height * 0.5)
            dx = float(enemy_center_x - player_center_x)
            dy = float(enemy_center_y - player_center_y)
            if abs(dx) > float(config.LOCAL_DX_NORM_PX) or abs(dy) > float(config.LOCAL_DY_NORM_PX):
                continue
            scored.append(
                (
                    (float((dx * dx) + (dy * dy)), float(abs(dx)), float(abs(dy)), int(enemy.spawn_index)),
                    enemy,
                )
            )
        scored.sort(key=lambda item: item[0])
        return [enemy for _, enemy in scored[: max(0, int(k))]]

    def _enemy_feature_values(self) -> dict[str, float]:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        enemy_speed_norm = float(max(1.0, config.ENEMY_RUN_SPEED_PX_PER_SEC))
        enemies = self._closest_relevant_enemies(k=2)
        values = {
            "opp1_dx": 0.0,
            "opp1_dy": 0.0,
            "opp1_vx_norm": 0.0,
            "opp1_tti": 0.0,
            "opp2_dx": 0.0,
            "opp2_dy": 0.0,
        }
        for idx, enemy in enumerate(enemies, start=1):
            rect = self._enemy_rect(enemy)
            enemy_center_x = float(rect.left + rect.width * 0.5)
            enemy_center_y = float(rect.top + rect.height * 0.5)
            dx = float(enemy_center_x - player_center_x)
            dy = float(enemy_center_y - player_center_y)
            values[f"opp{idx}_dx"] = float(clip_signed(dx / float(config.LOCAL_DX_NORM_PX)))
            values[f"opp{idx}_dy"] = float(clip_signed(dy / float(config.LOCAL_DY_NORM_PX)))
            if idx == 1:
                rel_speed = max(0.0, float(self.player_vx) - float(enemy.vx))
                values["opp1_vx_norm"] = float(clip_signed(float(enemy.vx) / enemy_speed_norm))
                values["opp1_tti"] = (
                    0.0
                    if dx <= 0.0 or rel_speed <= 1e-6
                    else float(np.clip(1.0 - ((dx / rel_speed) / 2.0), 0.0, 1.0))
                )
        return values

    def _lane_index_for_y(self, y: float) -> int:
        surface_rows = tuple(int(row) for row in config.LANE_SURFACE_ROWS)
        surface_y = float(y + config.PLAYER_SIZE)
        return int(
            min(
                range(len(surface_rows)),
                key=lambda idx: abs(float(surface_y) - float(surface_rows[idx] * config.TILE_SIZE)),
            )
        )

    def _support_at_x(self, x: float) -> TerrainSegment | MovingPlatform | None:
        best: tuple[float, TerrainSegment | MovingPlatform] | None = None
        foot_y = float(self.player_y + config.PLAYER_SIZE)
        for surface in self._support_surfaces():
            if not (float(surface.left) <= float(x) <= float(surface.right)):
                continue
            vertical_gap = abs(float(surface.surface_y) - foot_y)
            if best is None or vertical_gap < best[0]:
                best = (float(vertical_gap), surface)
        return None if best is None else best[1]

    def _sens_route_feature_values(self) -> dict[str, float]:
        center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        tile = float(config.TILE_SIZE)
        values: dict[str, float] = {}
        for label, offset_tiles in (("l2", -4), ("l1", -2), ("c0", 0), ("r1", 2), ("r2", 4)):
            values[f"sens_ground_{label}"] = 1.0 if self._support_at_x(center_x + tile * offset_tiles) else 0.0
        for idx, offset_tiles in enumerate((4, 8, 12), start=1):
            values[f"sens_gap_f{idx}"] = 0.0 if self._support_at_x(center_x + tile * offset_tiles) else 1.0
        return values

    def _next_route_segment(self) -> TerrainSegment | None:
        center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        current_idx = self.player_support_index
        candidates = [
            segment
            for segment in self.segments
            if float(segment.right) > center_x + float(config.TILE_SIZE)
            and (current_idx is None or int(segment.index) > int(current_idx))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda segment: (float(segment.left), int(segment.index)))

    def _route_feature_values(self) -> dict[str, float]:
        center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        current_lane = self._lane_index_for_y(float(self.player_y))
        lane_norm = float(clip_signed((current_lane / max(1, config.LANE_COUNT - 1)) * 2.0 - 1.0))
        next_segment = self._next_route_segment()
        if next_segment is None:
            return {
                "self_lane_norm": lane_norm,
                "land_next_dx": 0.0,
                "land_next_dy": 0.0,
                "land_next_width": 0.0,
                "land_next_lane_delta": 0.0,
                "land_gap_dx": 0.0,
                "land_gap_width": 0.0,
            }

        next_center_x = float(next_segment.left + next_segment.width * 0.5)
        next_center_y = float(next_segment.surface_y - config.PLAYER_SIZE * 0.5)
        gap_left = center_x
        if self.player_support_index is not None and 0 <= int(self.player_support_index) < len(self.segments):
            gap_left = float(self.segments[int(self.player_support_index)].right)
        gap_width = max(0.0, float(next_segment.left) - float(gap_left))
        lane_delta = int(next_segment.lane_index) - int(current_lane)
        return {
            "self_lane_norm": lane_norm,
            "land_next_dx": float(clip_signed((next_center_x - center_x) / float(config.LOCAL_DX_NORM_PX))),
            "land_next_dy": float(clip_signed((next_center_y - center_y) / float(config.LOCAL_DY_NORM_PX))),
            "land_next_width": float(np.clip(float(next_segment.width) / float(config.LOCAL_DX_NORM_PX), 0.0, 1.0)),
            "land_next_lane_delta": float(clip_signed(float(lane_delta) / max(1.0, float(config.LANE_COUNT - 1)))),
            "land_gap_dx": float(clip_signed((float(gap_left) - center_x) / float(config.LOCAL_DX_NORM_PX))),
            "land_gap_width": float(np.clip(gap_width / float(config.LOCAL_DX_NORM_PX), 0.0, 1.0)),
        }

    def _hazard_feature_values(self) -> dict[str, float]:
        center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        current_lane = self._lane_index_for_y(float(self.player_y))
        route_enemy: JumpEnemy | None = None
        lane_enemy: JumpEnemy | None = None
        for enemy in self.enemies:
            segment = self.segments[int(enemy.platform_index)]
            rect = self._enemy_rect(enemy)
            enemy_center_x = float(rect.left + rect.width * 0.5)
            if enemy_center_x < center_x:
                continue
            if route_enemy is None or enemy_center_x < float(self._enemy_rect(route_enemy).left):
                route_enemy = enemy
            if int(segment.lane_index) == int(current_lane):
                if lane_enemy is None or enemy_center_x < float(self._enemy_rect(lane_enemy).left):
                    lane_enemy = enemy

        def _encode(enemy: JumpEnemy | None) -> tuple[float, float]:
            if enemy is None:
                return 0.0, 0.0
            rect = self._enemy_rect(enemy)
            enemy_center_x = float(rect.left + rect.width * 0.5)
            dx = max(0.0, enemy_center_x - center_x)
            rel_speed = max(0.0, float(self.player_vx) - float(enemy.vx))
            tti = 0.0 if rel_speed <= 1e-6 else float(np.clip(1.0 - ((dx / rel_speed) / 2.0), 0.0, 1.0))
            return float(clip_signed(dx / float(config.LOCAL_DX_NORM_PX))), float(tti)

        route_dx, route_tti = _encode(route_enemy)
        lane_dx, lane_tti = _encode(lane_enemy)
        return {
            "haz_route_dx": route_dx,
            "haz_route_tti": route_tti,
            "haz_lane_dx": lane_dx,
            "haz_lane_tti": lane_tti,
        }

    def _compute_obs(self) -> np.ndarray:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        sens_features = self._sens_route_feature_values()
        route_features = self._route_feature_values()
        landing_features = self._moving_platform_feature_values()
        enemy_features = self._enemy_feature_values()
        hazard_features = self._hazard_feature_values()
        feature_values = {
            "self_vx_norm": float(
                clip_signed(float(self.player_vx) / float(max(1.0, config.PLAYER_RUN_SPEED_PX_PER_SEC)))
            ),
            "self_vy_norm": float(
                clip_signed(float(self.player_vy) / float(max(config.JUMP_VELOCITY_PX_PER_SEC, config.MAX_FALL_SPEED_PX_PER_SEC)))
            ),
            "self_grounded": 1.0 if bool(self.player_grounded) else 0.0,
            "flag_goal_dx": float(
                clip_signed((self.flag_center_x - player_center_x) / float(config.LOCAL_DX_NORM_PX))
            ),
            "flag_goal_dy": float(
                clip_signed((float(self.flag_rect.top + self.flag_rect.height * 0.5) - player_center_y) / float(config.LOCAL_DY_NORM_PX))
            ),
            "flag_progress_norm": float(self._player_flag_progress()),
            "flag_time_left": float(self._time_left_ratio()),
            **sens_features,
            **route_features,
            **landing_features,
            **enemy_features,
            **hazard_features,
        }
        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Jump observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        if not np.isfinite(obs).all():
            raise RuntimeError("Jump observation contains non-finite values.")
        self._last_obs = obs
        return obs

    def _check_enemy_collision(self) -> bool:
        player_rect = self._player_rect()
        for enemy in self.enemies:
            if player_rect.colliderect(self._enemy_rect(enemy)):
                self.failure_reason = "enemy"
                return True
        return False

    def _is_top_enemy_contact(self, prev_player_rect: Rect, enemy_rect: Rect) -> bool:
        player_rect = self._player_rect()
        overlap_width = min(float(player_rect.right), float(enemy_rect.right)) - max(
            float(player_rect.left),
            float(enemy_rect.left),
        )
        if float(overlap_width) < float(config.ENEMY_STOMP_MIN_OVERLAP_PX):
            return False

        overlap_height = min(float(player_rect.bottom), float(enemy_rect.bottom)) - max(
            float(player_rect.top),
            float(enemy_rect.top),
        )
        if float(overlap_height) <= 0.0:
            return False

        return bool(
            float(prev_player_rect.bottom) <= float(enemy_rect.top + config.ENEMY_STOMP_TOP_WINDOW_PX)
            and float(player_rect.bottom) >= float(enemy_rect.top)
            and float(player_rect.top) < float(enemy_rect.top)
            and float(overlap_height) <= float(overlap_width + config.ENEMY_STOMP_TOP_WINDOW_PX)
        )

    def _resolve_enemy_stomps(self, prev_player_rect: Rect) -> int:
        stomped_spawn_indices: set[int] = set()
        stomp_surface_top: float | None = None
        for enemy in self.enemies:
            enemy_rect = self._enemy_rect(enemy)
            if not self._player_rect().colliderect(enemy_rect):
                continue

            if not self._is_top_enemy_contact(prev_player_rect, enemy_rect):
                continue

            stomped_spawn_indices.add(int(enemy.spawn_index))
            if stomp_surface_top is None or float(enemy_rect.top) < float(stomp_surface_top):
                stomp_surface_top = float(enemy_rect.top)

        if not stomped_spawn_indices:
            return 0

        self.enemies = [
            enemy for enemy in self.enemies if int(enemy.spawn_index) not in stomped_spawn_indices
        ]
        if stomp_surface_top is not None:
            self.player_y = float(min(self.player_y, float(stomp_surface_top - config.PLAYER_SIZE)))
        self.player_vy = -float(config.ENEMY_STOMP_BOUNCE_VELOCITY_PX_PER_SEC)
        self.player_grounded = False
        self.player_support_index = None
        self.player_moving_support_index = None
        self._coyote_steps_left = 0
        return int(len(stomped_spawn_indices))

    def _check_flag_reached(self) -> bool:
        player_rect = self._player_rect()
        return bool(player_rect.colliderect(self.flag_rect) or player_rect.right >= float(self.flag_rect.left))

    def _camera_x(self) -> float:
        center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        raw = float(center_x - config.CAMERA_LOOKAHEAD_PX)
        max_camera = max(0.0, float(self.world_width_px - config.SCREEN_WIDTH))
        return float(max(0.0, min(raw, max_camera)))

    def _draw_platform_obstacle_cells(
        self,
        *,
        left: float,
        top: float,
        width_tiles: int,
        camera_x: float,
    ) -> None:
        cell_size = float(config.TILE_SIZE)
        for tile_offset in range(int(width_tiles)):
            draw_two_tone_cell(
                self.window_controller,
                top_left_x=float(left - camera_x + tile_offset * cell_size),
                top_left_y=float(top),
                tile_size=cell_size,
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
                cell_inset=float(config.CELL_INSET),
            )

    def _draw_world(self) -> None:
        arcade.draw_lbwh_rectangle_filled(
            0.0,
            float(config.BB_HEIGHT),
            float(config.SCREEN_WIDTH),
            float(config.PLAYFIELD_HEIGHT),
            COLOR_DARK_NEUTRAL,
        )
        camera_x = self._camera_x()
        for segment in self.segments:
            if float(segment.right) < float(camera_x) or float(segment.left) > float(camera_x + config.SCREEN_WIDTH):
                continue
            self._draw_platform_obstacle_cells(
                left=float(segment.left),
                top=float(segment.surface_y),
                width_tiles=int(segment.width_tiles),
                camera_x=float(camera_x),
            )
        for platform in self.moving_platforms:
            if float(platform.right) < float(camera_x) or float(platform.left) > float(camera_x + config.SCREEN_WIDTH):
                continue
            self._draw_platform_obstacle_cells(
                left=float(platform.left),
                top=float(platform.surface_y),
                width_tiles=int(platform.width_tiles),
                camera_x=float(camera_x),
            )

        goal_segment = self.segments[int(self.goal_segment_index)]
        pole_width = float(max(3.0, round(config.TILE_SIZE * 0.25)))
        flag_pole_left = float(self.flag_rect.left - camera_x - pole_width)
        pole_height = float(max(1.0, goal_segment.surface_y - self.flag_rect.top))
        pole_bottom = self.window_controller.top_left_to_bottom(float(self.flag_rect.top), float(pole_height))
        arcade.draw_lbwh_rectangle_filled(
            flag_pole_left,
            float(pole_bottom),
            pole_width,
            pole_height,
            COLOR_LIGHT_NEUTRAL,
        )
        draw_two_tone_square_block(
            self.window_controller,
            top_left_x=float(self.flag_rect.left - camera_x),
            top_left_y=float(self.flag_rect.top),
            tile_size=float(config.TILE_SIZE),
            tiles_per_side=int(config.GOAL_FLAG_WIDTH_TILES),
            outer_color=COLOR_AQUA,
            inner_color=COLOR_DEEP_TEAL,
            inset=square_block_inset(float(config.CELL_INSET), int(config.GOAL_FLAG_WIDTH_TILES)),
        )

        for enemy in self.enemies:
            rect = self._enemy_rect(enemy)
            draw_two_tone_square_block(
                self.window_controller,
                top_left_x=float(rect.left - camera_x),
                top_left_y=float(rect.top),
                tile_size=float(config.TILE_SIZE),
                tiles_per_side=int(config.ENEMY_TILES),
                outer_color=COLOR_CORAL,
                inner_color=COLOR_BRICK_RED,
                inset=square_block_inset(float(config.CELL_INSET), int(config.ENEMY_TILES)),
            )

        draw_two_tone_square_block(
            self.window_controller,
            top_left_x=float(self.player_x - camera_x),
            top_left_y=float(self.player_y),
            tile_size=float(config.TILE_SIZE),
            tiles_per_side=int(config.PLAYER_TILES),
            outer_color=COLOR_AQUA,
            inner_color=COLOR_DEEP_TEAL,
            inset=square_block_inset(float(config.CELL_INSET), int(config.PLAYER_TILES)),
        )

    def _draw_ghost_overlay(self) -> None:
        if not self._should_draw_ghost_overlay():
            return
        camera_x = self._camera_x()
        overlay_color = ghost_color(int(config.GHOST_OVERLAY_ALPHA))
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)

        for _, cell_rect, is_active in self._sens_ghost_probe_states():
            if float(cell_rect.right) < float(camera_x) or float(cell_rect.left) > float(camera_x + config.SCREEN_WIDTH):
                continue
            draw_ghost_line(
                self.window_controller,
                start_x=player_center_x,
                start_y=player_center_y,
                end_x=float(cell_rect.left + cell_rect.width * 0.5),
                end_y=float(cell_rect.top + cell_rect.height * 0.5),
                camera_x=float(camera_x),
                color=overlay_color,
                line_width=1.0,
            )
            draw_ghost_rect(
                self.window_controller,
                cell_rect,
                camera_x=float(camera_x),
                color=overlay_color,
                fill=bool(is_active),
                outline=True,
                line_width=1.5,
            )

        next_segment = self._next_route_segment()
        if next_segment is not None:
            segment_rect = Rect(
                left=float(next_segment.left),
                top=float(next_segment.surface_y - config.TILE_SIZE),
                width=float(next_segment.width),
                height=float(config.TILE_SIZE),
            )
            if float(segment_rect.right) >= float(camera_x) and float(segment_rect.left) <= float(camera_x + config.SCREEN_WIDTH):
                draw_ghost_rect(
                    self.window_controller,
                    segment_rect,
                    camera_x=float(camera_x),
                    color=overlay_color,
                    fill=False,
                    outline=True,
                    line_width=2.0,
                )

    def _draw_history_icon(self, success: bool, center_x: float, center_y: float, size: float) -> None:
        outer_color = COLOR_AQUA if bool(success) else COLOR_CORAL
        inner_color = COLOR_DEEP_TEAL if bool(success) else COLOR_BRICK_RED
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(status_icon_inset(float(config.CELL_INSET))),
        )

    def _draw_bottom_bar(self) -> None:
        layout = draw_status_bar(
            width=float(config.SCREEN_WIDTH),
            bottom_bar_height=float(config.BB_HEIGHT),
            tile_size=float(config.TILE_SIZE),
            cell_inset=float(config.CELL_INSET),
            include_clock=True,
        )
        draw_status_clock(layout=layout, remaining_ratio=float(self._time_left_ratio()))
        draw_status_icon_row(
            left=float(layout.score_left),
            right=float(layout.score_right),
            center_y=float(layout.center_y),
            icon_size=float(status_icon_size(float(config.BB_HEIGHT), float(config.TILE_SIZE))),
            items=list(self._history.history),
            draw_item=lambda success, icon_center_x, row_center_y, size: self._draw_history_icon(
                bool(success),
                float(icon_center_x),
                float(row_center_y),
                float(size),
            ),
        )

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self._reset_index += 1
        self.level_seed = int(self._episode_seed())
        self._generate_level(int(self.level_seed))
        self.done = False
        self.success = 0
        self.failure_reason = ""
        self.steps = 0
        self._episode_reward_components.reset()
        self._last_step_breakdown = self._empty_reward_breakdown()
        self._best_progress_potential = float(self._player_flag_progress())
        self._prev_jump_down = False
        return self._compute_obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._last_obs, 0.0, True, {
                "win": bool(self.success),
                "success": int(self._last_episode_success),
                "level": int(self._last_episode_level),
                "level_seed": int(self.level_seed),
                "failure_reason": str(self.failure_reason),
                "reward_components": self._episode_reward_components.totals(),
            }

        self.window_controller.poll_events_or_raise()
        self._update_visual_overlay_toggle()
        action_idx = int(self._resolve_human_action() if self.mode == "human" else self._parse_action(action))
        progress_prev_step = float(self._player_flag_progress())
        prev_support_index = self.player_support_index
        self._apply_action(int(action_idx))
        prev_player_rect = self._player_rect()

        self._step_moving_platforms()
        self._step_player()
        self._step_enemies()
        stomps_this_step = int(self._resolve_enemy_stomps(prev_player_rect))
        self.steps += 1

        reward = 0.0
        reward_breakdown = self._empty_reward_breakdown()
        if self.mode != "human":
            phi_best = float(self._best_progress_potential)
            phi_next = float(self._player_flag_progress())
            progress_delta = float(phi_next - progress_prev_step)
            forward_reward = min(
                float(config.PROGRESS_CLIP),
                float(config.PROGRESS_SCALE) * max(0.0, float(phi_next - phi_best)),
            )
            backtrack_penalty = min(
                float(config.PROGRESS_CLIP),
                float(config.PROGRESS_SCALE) * max(0.0, float(progress_prev_step - phi_next)),
            )
            progress_reward = float(forward_reward - backtrack_penalty)
            reward += float(progress_reward)
            reward_breakdown["progress.shape"] = float(progress_reward)
            if abs(float(progress_delta)) <= float(config.STALL_PROGRESS_EPS):
                reward += float(config.PENALTY_STALL)
                reward_breakdown["progress.penalty_stall"] = float(config.PENALTY_STALL)
            self._best_progress_potential = float(max(phi_best, phi_next))

        episode_level = int(self._current_level)
        episode_success = 0

        if self._check_enemy_collision():
            self.done = True
        elif self.player_y > float(config.PLAYFIELD_HEIGHT + config.PLAYER_SIZE):
            self.done = True
            self.failure_reason = "gap"
        elif self._check_flag_reached():
            self.done = True
            self.success = 1
            episode_success = 1
            self.failure_reason = "flag"
        elif int(self.steps) >= int(self.max_episode_steps):
            self.done = True
            self.failure_reason = "timeout"

        if self.mode != "human" and int(stomps_this_step) > 0 and not (self.done and not self.success):
            stomp_reward = min(float(config.REWARD_STOMP_MAX), float(stomps_this_step) * float(config.REWARD_STOMP))
            reward += float(stomp_reward)
            reward_breakdown["combat.reward_stomp"] = float(stomp_reward)

        if self.done:
            if self.success:
                if self.mode != "human":
                    reward += float(config.REWARD_FINISH)
                    reward_breakdown["outcome.reward_finish"] = float(config.REWARD_FINISH)
            else:
                if self.mode != "human":
                    reward += float(config.PENALTY_FAIL)
                    reward_breakdown["outcome.penalty_fail"] = float(config.PENALTY_FAIL)
            self._history.record_result(bool(self.success))

        self._last_step_breakdown = dict(reward_breakdown)
        if self.mode != "human":
            self._episode_reward_components.add_from_mapping(reward_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)

        self.render()
        self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)

        obs = self._compute_obs()
        info: dict[str, object] = {
            "win": bool(self.success),
            "success": int(episode_success),
            "level": int(episode_level),
            "level_seed": int(self.level_seed),
            "progress_norm": float(self._player_flag_progress()),
            "time_left_ratio": float(self._time_left_ratio()),
            "stomps": int(stomps_this_step),
            "failure_reason": str(self.failure_reason),
            "route_transition": self._route_transition_label(prev_support_index, self.player_support_index),
            "enemy_count": int(len(self.enemies)),
            "moving_platform_count": int(len(self.moving_platforms)),
            "level_changed": False,
            "reward_breakdown": reward_breakdown if self.mode != "human" else {},
        }
        if self.done:
            info["reward_components"] = self._episode_reward_components.totals()
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(episode_success)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(episode_success),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )
            info["level_changed"] = bool(level_changed)
        return obs, float(reward), bool(self.done), info

    def render(self) -> None:
        if self.window_controller.window is None:
            return
        self.window_controller.clear(COLOR_SLATE_GRAY)
        self._draw_world()
        self._draw_ghost_overlay()
        self._draw_bottom_bar()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
        self.window = None
