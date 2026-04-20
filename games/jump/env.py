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
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_signed, clip_unit, ordered_feature_vector
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    draw_status_square_icon,
    draw_two_tone_square_block,
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


class JumpEnv(Env):
    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)
    REWARD_COMPONENT_ORDER = ("F", "X", "T", "P", "S")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_finish": "F",
        "outcome.penalty_fail": "X",
        "combat.reward_stomp": "T",
        "progress.forward_scale": "P",
        "step.penalty_step": "S",
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

        self.segments: list[TerrainSegment] = []
        self.enemies: list[JumpEnemy] = []
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
        self.player_last_support_index: int | None = None
        self._coyote_steps_left = 0
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._best_progress_potential = 0.0
        self._last_step_breakdown = self._empty_reward_breakdown()
        self._prev_jump_down = False

        self._apply_level_settings(int(self._current_level))
        self.reset()

    @staticmethod
    def _empty_reward_breakdown() -> dict[str, float]:
        return {
            "outcome.reward_finish": 0.0,
            "outcome.penalty_fail": 0.0,
            "combat.reward_stomp": 0.0,
            "progress.forward_scale": 0.0,
            "step.penalty_step": 0.0,
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
        segment_target = int(self._segment_target_for_length(int(length_tiles)))
        internal_segments = max(1, int(segment_target - 2))
        gap_min = max(
            int(config.PLAYER_TILES),
            int(config.BASE_GAP_MIN_TILES)
            + (int(config.LEVEL3_EXTRA_GAP_MIN_TILES) if int(level) >= int(config.MAX_LEVEL) else 0),
        )
        gap_max = max(
            int(gap_min),
            int(gap_min + config.BASE_GAP_EXTRA_TILES)
            + (int(config.LEVEL2_EXTRA_GAP_MAX_TILES) if int(level) >= 2 else 0),
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
        self.gap_min = max(int(config.PLAYER_TILES), int(profile["gap_min"]))
        self.gap_max = max(self.gap_min, int(profile["gap_max"]))
        self.max_lane_index = max(0, min(int(config.LANE_COUNT - 1), int(profile["max_lane_index"])))
        self.lane_delta_choices = tuple(int(value) for value in profile["lane_delta_choices"])
        self.min_upper_segments = max(0, int(profile["min_upper_segments"]))
        self.min_top_segments = max(0, int(profile["min_top_segments"]))
        self.enemy_count_min = max(0, int(profile["enemy_count_min"]))
        self.enemy_count_max = max(self.enemy_count_min, int(profile["enemy_count_max"]))
        self.enemy_spawn_chance = float(max(0.0, min(1.0, float(profile["enemy_spawn_chance"]))))
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

    def _build_segments(self, rng: random.Random) -> list[TerrainSegment]:
        lane_plan = self._build_lane_plan(rng)
        world_width_tiles = int(round(self.world_width_px / config.TILE_SIZE))
        segments: list[TerrainSegment] = []
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
            for _ in range(40):
                gap_tiles = self._sample_gap_tiles(rng)
                remaining_internal = int(len(internal_lanes) - internal_idx - 1)
                future_min_tiles = (
                    int(remaining_internal) * int(self.platform_min_width_tiles + self.gap_min)
                    + int(self.gap_min)
                    + int(self.goal_stretch_tiles)
                )
                max_width_tiles = int(world_width_tiles - (current_left_tile + gap_tiles + future_min_tiles))
                width_choices = self._platform_width_choices_up_to(int(max_width_tiles))
                if not width_choices:
                    continue
                next_width = int(rng.choice(width_choices))
                if not self._transition_is_reachable(
                    from_lane_index=int(segments[-1].lane_index),
                    to_lane_index=int(next_lane),
                    gap_tiles=int(gap_tiles),
                    landing_width_tiles=int(next_width),
                ):
                    continue

                left_tile = int(current_left_tile + gap_tiles)
                segments.append(
                    TerrainSegment(
                        index=int(len(segments)),
                        left_tile=int(left_tile),
                        width_tiles=int(next_width),
                        lane_index=int(next_lane),
                    )
                )
                current_left_tile = int(left_tile + next_width)
                accepted = True
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
        return segments

    def _validate_segments(self, segments: list[TerrainSegment]) -> None:
        if len(segments) < 2:
            raise ValueError("Jump requires at least start and goal segments.")
        for idx in range(len(segments) - 1):
            current = segments[idx]
            nxt = segments[idx + 1]
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
                segments = self._build_segments(rng)
                self._validate_segments(segments)
                enemies = self._spawn_enemies(rng, segments)
                break
            except (RuntimeError, ValueError):
                attempt_seed += 1
                continue
        else:
            raise RuntimeError("Jump failed to generate a reachable procedural level.")

        self.segments = list(segments)
        self.enemies = list(enemies)
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

    def _segment_for_index(self, index: int | None) -> TerrainSegment | None:
        if index is None:
            return None
        if int(index) < 0 or int(index) >= len(self.segments):
            return None
        return self.segments[int(index)]

    def _support_segment_below(self, *, tolerance_px: float = 0.0) -> int | None:
        player_rect = self._player_rect()
        best_index: int | None = None
        best_gap: float | None = None
        feet_y = float(player_rect.bottom)
        for segment in self.segments:
            if player_rect.right <= float(segment.left + 2.0):
                continue
            if player_rect.left >= float(segment.right - 2.0):
                continue
            gap = float(segment.surface_y - feet_y)
            if gap < -1.0:
                continue
            if gap > float(tolerance_px):
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_index = int(segment.index)
        return best_index

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

    def _rect_collides_terrain(self, rect: Rect) -> list[TerrainSegment]:
        return [segment for segment in self.segments if rect.colliderect(segment.rect)]

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
                self._coyote_steps_left = 0

    def _can_jump(self) -> bool:
        return bool(self.player_grounded or int(self._coyote_steps_left) > 0)

    def _step_player(self) -> None:
        dt = float(config.PHYSICS_DT)
        prev_rect = self._player_rect()

        next_x = float(self.player_x + self.player_vx * dt)
        next_x = float(max(0.0, min(next_x, self.world_width_px - config.PLAYER_SIZE)))
        test_rect = Rect(next_x, float(self.player_y), float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
        colliders = self._rect_collides_terrain(test_rect)
        if float(self.player_vx) > 0.0:
            for segment in sorted(colliders, key=lambda item: float(item.left)):
                if prev_rect.right <= float(segment.left) and test_rect.right > float(segment.left):
                    next_x = float(segment.left - config.PLAYER_SIZE)
                    test_rect = Rect(next_x, float(self.player_y), float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
                    break
        elif float(self.player_vx) < 0.0:
            for segment in sorted(colliders, key=lambda item: float(item.right), reverse=True):
                if prev_rect.left >= float(segment.right) and test_rect.left < float(segment.right):
                    next_x = float(segment.right)
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
        landed_support: int | None = None
        colliders = self._rect_collides_terrain(test_rect)
        if float(self.player_vy) < 0.0:
            for segment in sorted(colliders, key=lambda item: float(item.rect.bottom), reverse=True):
                segment_bottom = float(segment.rect.bottom)
                if prev_rect.top >= segment_bottom and test_rect.top < segment_bottom:
                    next_y = float(segment_bottom)
                    self.player_vy = 0.0
                    test_rect = Rect(float(self.player_x), next_y, float(config.PLAYER_SIZE), float(config.PLAYER_SIZE))
                    break
        elif float(self.player_vy) >= 0.0:
            for segment in sorted(colliders, key=lambda item: float(item.surface_y)):
                if prev_rect.bottom <= float(segment.surface_y) and test_rect.bottom >= float(segment.surface_y):
                    next_y = float(segment.surface_y - config.PLAYER_SIZE)
                    self.player_vy = 0.0
                    grounded = True
                    landed_support = int(segment.index)
                    break
        self.player_y = float(next_y)

        support_index = landed_support
        if support_index is None:
            support_index = self._support_segment_below(tolerance_px=float(config.GROUND_SNAP_PX))
            if support_index is not None and float(self.player_vy) >= 0.0:
                support = self.segments[int(support_index)]
                self.player_y = float(support.surface_y - config.PLAYER_SIZE)
                self.player_vy = 0.0
                grounded = True

        self.player_grounded = bool(grounded)
        self.player_support_index = None if support_index is None else int(support_index)
        if self.player_grounded:
            self.player_last_support_index = int(self.player_support_index) if self.player_support_index is not None else None
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

    def _down_ground_distance(self) -> float:
        player_rect = self._player_rect()
        feet_y = float(player_rect.bottom)
        best_distance: float | None = None
        for segment in self.segments:
            if player_rect.right <= float(segment.left + 2.0):
                continue
            if player_rect.left >= float(segment.right - 2.0):
                continue
            if float(segment.surface_y) < feet_y:
                continue
            distance = float(segment.surface_y - feet_y)
            if best_distance is None or distance < best_distance:
                best_distance = distance
        return float(best_distance if best_distance is not None else config.SENS_DOWN_RANGE_PX)

    def _standable_target_at_x(
        self,
        sample_center_x: float,
        *,
        reference_feet_y: float,
        rise_px: float,
        drop_px: float,
    ) -> tuple[TerrainSegment, float] | None:
        sample_left = float(sample_center_x - config.PLAYER_SIZE * 0.5)
        sample_right = float(sample_left + config.PLAYER_SIZE)
        if sample_left < 0.0 or sample_right > float(self.world_width_px):
            return None

        best_match: tuple[tuple[float, float, int], TerrainSegment, float] | None = None
        for segment in self.segments:
            if sample_left < float(segment.left) - 1e-6:
                continue
            if sample_right > float(segment.right) + 1e-6:
                continue

            gap = float(segment.surface_y - reference_feet_y)
            if gap < -float(rise_px) or gap > float(drop_px):
                continue

            stand_rect = Rect(
                left=float(sample_left),
                top=float(segment.surface_y - config.PLAYER_SIZE),
                width=float(config.PLAYER_SIZE),
                height=float(config.PLAYER_SIZE),
            )
            blocked = any(int(other.index) != int(segment.index) for other in self._rect_collides_terrain(stand_rect))
            if blocked:
                continue

            segment_center_x = float(segment.left + segment.width * 0.5)
            score = (
                float(abs(gap)),
                float(abs(segment_center_x - sample_center_x)),
                int(segment.index),
            )
            if best_match is None or score < best_match[0]:
                best_match = (score, segment, float(gap))

        if best_match is None:
            return None
        return best_match[1], float(best_match[2])

    def _floor_probe(self, offset_px: float) -> float:
        player_rect = self._player_rect()
        player_center_x = float(player_rect.left + player_rect.width * 0.5)
        target = self._standable_target_at_x(
            float(player_center_x + offset_px),
            reference_feet_y=float(player_rect.bottom),
            rise_px=float(config.FLOOR_PROBE_STEP_UP_PX),
            drop_px=float(config.FLOOR_PROBE_DROP_PX),
        )
        if target is None:
            return 0.0
        _, gap = target
        gap_scale = max(1.0, float(max(config.FLOOR_PROBE_STEP_UP_PX, config.FLOOR_PROBE_DROP_PX)))
        return float(1.0 - clip_unit(abs(float(gap)) / float(gap_scale)))

    def _arc_probe(self, offset_px: float) -> float:
        player_rect = self._player_rect()
        player_center_x = float(player_rect.left + player_rect.width * 0.5)
        player_center_y = float(player_rect.top + player_rect.height * 0.5)
        target_center_x = float(player_center_x + offset_px)
        target = self._standable_target_at_x(
            float(target_center_x),
            reference_feet_y=float(player_rect.bottom),
            rise_px=float(config.ARC_PROBE_RISE_PX),
            drop_px=float(config.ARC_PROBE_DROP_PX),
        )
        if target is None:
            return 0.0

        target_segment, _ = target
        target_center_y = float(target_segment.surface_y - config.PLAYER_SIZE * 0.5)
        rise_required = max(0.0, float(player_center_y - target_center_y))
        drop_required = max(0.0, float(target_center_y - player_center_y))
        if rise_required > float(config.ARC_PROBE_RISE_PX):
            return 0.0
        if drop_required > float(config.ARC_PROBE_DROP_PX):
            return 0.0

        peak_lift = max(
            float(config.ARC_PROBE_PEAK_EXTRA_PX),
            0.5 * float(abs(offset_px)),
            float(rise_required + config.ARC_PROBE_PEAK_EXTRA_PX),
        )
        if peak_lift > float(config.ARC_PROBE_RISE_PX):
            return 0.0
        peak_y = float(player_center_y - peak_lift)

        sample_count = max(1, int(config.ARC_PROBE_SAMPLES))
        for sample_idx in range(1, sample_count + 1):
            t = float(sample_idx) / float(sample_count + 1)
            sample_center_x = float(player_center_x + (target_center_x - player_center_x) * t)
            sample_center_y = float(
                ((1.0 - t) * (1.0 - t) * player_center_y)
                + (2.0 * (1.0 - t) * t * peak_y)
                + (t * t * target_center_y)
            )
            sample_rect = Rect(
                left=float(sample_center_x - config.PLAYER_SIZE * 0.5),
                top=float(sample_center_y - config.PLAYER_SIZE * 0.5),
                width=float(config.PLAYER_SIZE),
                height=float(config.PLAYER_SIZE),
            )
            if self._rect_collides_terrain(sample_rect):
                return 0.0
        return 1.0

    def _up_clear_norm(self) -> float:
        player_rect = self._player_rect()
        best_clearance = float(player_rect.top)
        for segment in self.segments:
            overlap_width = min(float(player_rect.right), float(segment.right)) - max(
                float(player_rect.left),
                float(segment.left),
            )
            if overlap_width <= 0.0:
                continue
            ceiling_y = float(segment.rect.bottom)
            if ceiling_y > float(player_rect.top):
                continue
            best_clearance = min(best_clearance, float(player_rect.top - ceiling_y))
        return float(clip_unit(best_clearance / float(config.SENS_UP_CLEAR_RANGE_PX)))

    def _landing_reference_index(self) -> int:
        if self.player_support_index is not None:
            return int(self.player_support_index)
        if self.player_last_support_index is not None:
            return int(self.player_last_support_index)

        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        best_index = -1
        for segment in self.segments:
            if float(segment.right) <= float(player_center_x):
                best_index = max(int(best_index), int(segment.index))
        return int(best_index)

    @staticmethod
    def _segment_landing_anchor(segment: TerrainSegment) -> tuple[float, float]:
        return (
            float(segment.left + config.PLAYER_SIZE * 0.5),
            float(segment.surface_y - config.PLAYER_SIZE * 0.5),
        )

    def _future_landing_segments(self) -> list[TerrainSegment]:
        reference_index = int(self._landing_reference_index())
        return [segment for segment in self.segments if int(segment.index) > int(reference_index)][:2]

    def _landing_anchor_feature_values(self) -> dict[str, float]:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        future_segments = self._future_landing_segments()
        feature_values: dict[str, float] = {}
        for feature_idx in range(2):
            key_prefix = "land_next" if feature_idx == 0 else "land_next2"
            if feature_idx < len(future_segments):
                anchor_x, anchor_y = self._segment_landing_anchor(future_segments[feature_idx])
                feature_values[f"{key_prefix}_dx"] = float(
                    clip_signed((float(anchor_x) - player_center_x) / float(config.LOCAL_DX_NORM_PX))
                )
                feature_values[f"{key_prefix}_dy"] = float(
                    clip_signed((float(anchor_y) - player_center_y) / float(config.LOCAL_DY_NORM_PX))
                )
            else:
                feature_values[f"{key_prefix}_dx"] = 0.0
                feature_values[f"{key_prefix}_dy"] = 0.0
        return feature_values

    def _enemy_relevance_key(self, enemy: JumpEnemy) -> tuple[int, float, float, int, int]:
        rect = self._enemy_rect(enemy)
        enemy_center_x = float(rect.left + rect.width * 0.5)
        enemy_center_y = float(rect.top + rect.height * 0.5)
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        dx = float(enemy_center_x - player_center_x)
        dy = float(enemy_center_y - player_center_y)
        player_platform_index = -1 if self.player_last_support_index is None else int(self.player_last_support_index)
        same_platform = 0 if int(enemy.platform_index) == int(player_platform_index) else 1
        ahead_bias = 0 if dx >= 0.0 else 1
        return (
            int(same_platform),
            float(abs(dx)),
            float(abs(dy)),
            int(ahead_bias),
            int(enemy.spawn_index),
        )

    def _enemy_slots(self) -> list[JumpEnemy]:
        ordered = sorted(self.enemies, key=self._enemy_relevance_key)
        return ordered[:2]

    def _compute_obs(self) -> np.ndarray:
        player_center_x = float(self.player_x + config.PLAYER_SIZE * 0.5)
        player_center_y = float(self.player_y + config.PLAYER_SIZE * 0.5)
        enemy_slots = self._enemy_slots()
        landing_features = self._landing_anchor_feature_values()
        enemy_features: dict[str, float] = {}
        for slot_index in range(2):
            if slot_index < len(enemy_slots):
                enemy = enemy_slots[slot_index]
                rect = self._enemy_rect(enemy)
                enemy_center_x = float(rect.left + rect.width * 0.5)
                enemy_center_y = float(rect.top + rect.height * 0.5)
                key_prefix = f"opp{slot_index + 1}_"
                enemy_features[f"{key_prefix}dx"] = float(
                    clip_signed((enemy_center_x - player_center_x) / float(config.LOCAL_DX_NORM_PX))
                )
                enemy_features[f"{key_prefix}dy"] = float(
                    clip_signed((enemy_center_y - player_center_y) / float(config.LOCAL_DY_NORM_PX))
                )
                enemy_features[f"{key_prefix}vx_norm"] = float(
                    clip_signed(float(enemy.vx) / float(max(1.0, config.ENEMY_RUN_SPEED_PX_PER_SEC)))
                )
            else:
                key_prefix = f"opp{slot_index + 1}_"
                enemy_features[f"{key_prefix}dx"] = 0.0
                enemy_features[f"{key_prefix}dy"] = 0.0
                enemy_features[f"{key_prefix}vx_norm"] = 0.0

        feature_values = {
            "self_vx_norm": float(
                clip_signed(float(self.player_vx) / float(max(1.0, config.PLAYER_RUN_SPEED_PX_PER_SEC)))
            ),
            "self_vy_norm": float(
                clip_signed(float(self.player_vy) / float(max(config.JUMP_VELOCITY_PX_PER_SEC, config.MAX_FALL_SPEED_PX_PER_SEC)))
            ),
            "self_grounded": 1.0 if bool(self.player_grounded) else 0.0,
            "sens_floor_f1_norm": float(self._floor_probe(float(config.FLOOR_PROBE_F1_OFFSET_PX))),
            "sens_floor_f2_norm": float(self._floor_probe(float(config.FLOOR_PROBE_F2_OFFSET_PX))),
            "sens_floor_b1_norm": float(self._floor_probe(-float(config.FLOOR_PROBE_F1_OFFSET_PX))),
            "sens_floor_b2_norm": float(self._floor_probe(-float(config.FLOOR_PROBE_F2_OFFSET_PX))),
            "sens_arc_f1_norm": float(self._arc_probe(float(config.ARC_PROBE_F1_OFFSET_PX))),
            "sens_arc_f2_norm": float(self._arc_probe(float(config.ARC_PROBE_F2_OFFSET_PX))),
            "sens_up_clear_norm": float(self._up_clear_norm()),
            "sens_down_ground_norm": float(
                clip_unit(self._down_ground_distance() / float(config.SENS_DOWN_RANGE_PX))
            ),
            "flag_goal_dx": float(
                clip_signed((self.flag_center_x - player_center_x) / float(config.LOCAL_DX_NORM_PX))
            ),
            "flag_goal_dy": float(
                clip_signed((float(self.flag_rect.top + self.flag_rect.height * 0.5) - player_center_y) / float(config.LOCAL_DY_NORM_PX))
            ),
            "flag_progress_norm": float(self._player_flag_progress()),
            **landing_features,
            **enemy_features,
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

    def _draw_two_tone_rect(
        self,
        *,
        left: float,
        top: float,
        width: float,
        height: float,
        outer_color: tuple[int, int, int],
        inner_color: tuple[int, int, int],
        inset: float,
    ) -> None:
        bottom = self.window_controller.top_left_to_bottom(float(top), float(height))
        arcade.draw_lbwh_rectangle_filled(float(left), float(bottom), float(width), float(height), outer_color)
        inner_width = max(1.0, float(width) - 2.0 * float(inset))
        inner_height = max(1.0, float(height) - 2.0 * float(inset))
        inner_top = float(top + inset)
        inner_bottom = self.window_controller.top_left_to_bottom(float(inner_top), float(inner_height))
        arcade.draw_lbwh_rectangle_filled(
            float(left + inset),
            float(inner_bottom),
            float(inner_width),
            float(inner_height),
            inner_color,
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
            self._draw_two_tone_rect(
                left=float(segment.left - camera_x),
                top=float(segment.surface_y),
                width=float(segment.width),
                height=float(config.PLATFORM_THICKNESS_PX),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
                inset=float(max(2.0, config.CELL_INSET)),
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
            inset=float(config.CELL_INSET),
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
                inset=float(config.CELL_INSET),
            )

        draw_two_tone_square_block(
            self.window_controller,
            top_left_x=float(self.player_x - camera_x),
            top_left_y=float(self.player_y),
            tile_size=float(config.TILE_SIZE),
            tiles_per_side=int(config.PLAYER_TILES),
            outer_color=COLOR_AQUA,
            inner_color=COLOR_DEEP_TEAL,
            inset=float(config.CELL_INSET),
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
        action_idx = int(self._resolve_human_action() if self.mode == "human" else self._parse_action(action))
        progress_prev_step = float(self._player_flag_progress())
        self._apply_action(int(action_idx))
        prev_player_rect = self._player_rect()

        self._step_player()
        self._step_enemies()
        stomps_this_step = int(self._resolve_enemy_stomps(prev_player_rect))
        self.steps += 1

        reward = 0.0
        reward_breakdown = self._empty_reward_breakdown()
        if self.mode != "human":
            reward += float(config.PENALTY_STEP)
            reward_breakdown["step.penalty_step"] = float(config.PENALTY_STEP)

            phi_prev = float(self._best_progress_potential)
            phi_next = float(self._player_flag_progress())
            progress_delta = max(0.0, float(phi_next - phi_prev))
            forward_reward = min(
                float(config.FORWARD_PROGRESS_CLIP),
                float(config.FORWARD_PROGRESS_SCALE) * float(progress_delta),
            )
            backtrack_delta = max(0.0, float(progress_prev_step - phi_next))
            backtrack_penalty = min(
                float(config.BACKTRACK_PENALTY_CLIP),
                float(config.BACKTRACK_PENALTY_SCALE) * float(backtrack_delta),
            )
            progress_reward = float(forward_reward - backtrack_penalty)
            reward += float(progress_reward)
            reward_breakdown["progress.forward_scale"] = float(progress_reward)
            self._best_progress_potential = float(max(phi_prev, phi_next))

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
        self._draw_bottom_bar()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
        self.window = None
