"""Top-down one-lap racing environment with geometry-first procedural tracks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
import time

import arcade
import numpy as np

from core.arcade_style import (
    ACCENT_PAIRS,
    COLOR_AQUA,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
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
from core.envs.arcade import ArcadeEnvMixin
from core.envs.base import Env
from core.ghost_overlay import ghost_color, update_ghost_overlay_toggle
from core.io_schema import (
    clip_signed,
    clip_unit,
    ordered_feature_vector,
)
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_facing_indicator,
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    draw_status_square_icon,
    draw_two_tone_cell,
    resolve_circle_collisions,
    status_icon_inset,
    status_icon_size,
)
from core.ray_viz import draw_player_rays
from core.rewards import RewardBreakdown
from core.shared_config import (
    BB_HEIGHT,
    CELL_INSET,
    FPS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
)
from core.utils import resolve_play_level
from games.vroom.config import (
    ACTION_NAMES as VROOM_ACTION_NAMES,
    ACT_DIM as VROOM_ACT_DIM,
    CURRICULUM_PROMOTION,
    DRAW_RAYS,
    FORWARD_RAY_MAX_DISTANCE_PX,
    LEVEL_SETTINGS,
    INPUT_FEATURE_NAMES as VROOM_INPUT_FEATURE_NAMES,
    LATERAL_DAMPING_OFF_TRACK,
    LATERAL_DAMPING_ON_TRACK,
    MAX_LEVEL,
    MAX_EPISODE_STEPS,
    MIN_LEVEL,
    OBS_DIM as VROOM_OBS_DIM,
    OFF_TRACK_SURFACE_GRIP,
    OFF_TRACK_MAX_SPEED_FACTOR,
    OFF_TRACK_PENALTY_MARGIN_PX,
    OFF_TRACK_SPEED_TRANSITION_SECONDS,
    OFF_TRACK_TERMINATE_SEVERITY,
    OFF_TRACK_TERMINATE_STEPS,
    NO_PROGRESS_EPS_NORM,
    NO_PROGRESS_TIMEOUT_STEPS,
    OPPONENT_BEND_CAUTION_MULT_RANGE,
    OPPONENT_BRAKE_RESPONSE,
    OPPONENT_MIN_BEND_SPEED_FACTOR,
    OPPONENT_SPEED_MULT_RANGE,
    EDGE_PROBE_MAX_DISTANCE_PX,
    PENALTY_CONTACT,
    PENALTY_LOSE,
    PENALTY_OFF_TRACK,
    PENALTY_STEP,
    PROGRESS_CLIP,
    PROGRESS_SCALE,
    RANDOM_START_MIN_REMAINING_PROGRESS_NORM,
    REWARD_WIN,
    ROUTE_LOOKAHEAD_RANGES_PX,
    SENS_CAR_RANGE_PX,
    SENS_CAR_SIDE_RANGE_PX,
    STEER_FULL_SPEED_NORM,
    STEER_MIN_SPEED_FACTOR,
    STEER_SPEED_DECAY,
    TRACK_CORNER_RADIUS_PX,
    TRACK_BEND_SMOOTHING_PASSES,
    TRACK_COMPLEXITY_HARD_SAMPLE_RATE,
    TRACK_FOLD_GAP_PX,
    TRACK_FOOTPRINT_SCALE,
    TRACK_GENERATION_MAX_ATTEMPTS,
    TRACK_LONG_SIDE_BELL_AMPLITUDE_MAX_PX,
    TRACK_LONG_SIDE_BELL_AMPLITUDE_MIN_PX,
    TRACK_LONG_SIDE_INSET_LENGTH_CAP_RATIO,
    TRACK_LONG_SIDE_INSET_WIDTH_CAP_RATIO,
    TRACK_LONG_SIDE_S_AMPLITUDE_MAX_PX,
    TRACK_LONG_SIDE_S_AMPLITUDE_MIN_PX,
    TRACK_LONG_SIDE_TEMPLATE_CHOICES,
    TRACK_PADDING_PX,
    TRACK_SAMPLE_SPACING_PX,
    TRACK_SHORT_SIDE_TEMPLATE_CHOICES,
    TRACK_START_STRAIGHT_LEN_PX,
    TRACK_VALID_HYSTERESIS_PX,
    TRACK_WIDTH_PX,
    TURN_THROTTLE_LOSS,
    WINDOW_TITLE,
)
from games.vroom.track_geometry import (
    TrackGeometry,
    TrackProjection,
    project_point_to_track,
    raycast_track_edge,
    sample_track_at_s,
    spawn_pose,
)
from games.vroom.trackgen import TrackGenConfig, generate_track


validate_curriculum_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
)


@dataclass
class RaceCar:
    x: float
    y: float
    vx: float
    vy: float
    heading_degrees: float
    outer_color: tuple[int, int, int]
    inner_color: tuple[int, int, int]
    yaw_rate: float = 0.0
    in_contact: bool = False
    ai_lane_home: float = 0.0
    ai_lane_offset: float = 0.0
    ai_rejoin_steps: int = 0
    speed_mult: float = 1.0
    bend_caution_mult: float = 1.0
    off_track: bool = False
    off_track_blend: float = 0.0
    track_progress: float = 0.0
    prev_track_progress: float = 0.0
    lap_armed: bool = False
    track_index: int = 0
    lap_progress: float = 0.0
    finished: bool = False


class VroomEnv(ArcadeEnvMixin, Env):
    INPUT_FEATURE_NAMES = tuple(VROOM_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(VROOM_ACTION_NAMES)
    OBS_DIM = int(VROOM_OBS_DIM)
    ACT_DIM = int(VROOM_ACT_DIM)

    NUM_CARS = 4
    TRAINING_TOTAL_RACES = 1
    PLAY_TOTAL_RACES = 10
    REWARD_COMPONENT_ORDER = ("W", "L", "P", "T", "C", "S")
    # Keep raw reward keys stable; compact T/C now mean off-track severity and contact duration.
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_win": "W",
        "outcome.penalty_lose": "L",
        "progress.shape": "P",
        "track.penalty_coverage": "T",
        "event.penalty_collision": "C",
        "step.penalty_step": "S",
    }

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self._init_arcade_runtime(
            width=SCREEN_WIDTH,
            height=SCREEN_HEIGHT,
            title=WINDOW_TITLE,
            render=bool(render),
            queue_input_events=False,
            vsync=False,
            render_fps=FPS,
            training_fps=0,
        )

        self.track_bottom = float(SCREEN_HEIGHT - BB_HEIGHT)
        self.track_half_width = float(TILE_SIZE * 1.25)

        self.car_size = float(TILE_SIZE)
        self.car_half = self.car_size * 0.5
        self.car_radius = self.car_half * 0.95
        # Heavier handling: full-throttle cornering should understeer unless the car slows.
        self.max_speed = 7.0
        self.max_reverse_speed = 2.4
        self.accel_force = 0.36
        self.brake_force = 0.23
        self.turn_rate = 4.1
        self.drag = 0.985
        self.max_steps = int(MAX_EPISODE_STEPS)
        self.yaw_rate_norm = max(1e-6, self.turn_rate)

        self.car_contact_radius = self.car_radius
        self.track_probe_radius = self.car_radius * 0.46
        diag = self.track_probe_radius * 0.70710678
        self.track_probe_offsets = (
            (0.0, 0.0),
            (self.track_probe_radius, 0.0),
            (-self.track_probe_radius, 0.0),
            (0.0, self.track_probe_radius),
            (0.0, -self.track_probe_radius),
            (diag, diag),
            (-diag, diag),
            (diag, -diag),
            (-diag, -diag),
        )
        inner_probe_r = self.car_half * 0.55
        outer_probe_r = self.car_half * 0.92
        inner_diag = inner_probe_r * 0.70710678
        outer_diag = outer_probe_r * 0.70710678
        self.off_track_probe_offsets = (
            (0.0, 0.0),
            (inner_probe_r, 0.0),
            (-inner_probe_r, 0.0),
            (0.0, inner_probe_r),
            (0.0, -inner_probe_r),
            (inner_diag, inner_diag),
            (-inner_diag, inner_diag),
            (inner_diag, -inner_diag),
            (-inner_diag, -inner_diag),
            (outer_probe_r, 0.0),
            (-outer_probe_r, 0.0),
            (0.0, outer_probe_r),
            (0.0, -outer_probe_r),
            (outer_diag, outer_diag),
            (-outer_diag, outer_diag),
            (outer_diag, -outer_diag),
            (-outer_diag, -outer_diag),
        )
        self.off_track_enter_ratio = 0.58
        self.off_track_exit_ratio = 0.74
        self.off_track_max_speed_factor = self._clamp(float(OFF_TRACK_MAX_SPEED_FACTOR), 0.05, 1.0)
        self.off_track_surface_grip = self._clamp(float(OFF_TRACK_SURFACE_GRIP), 0.05, 1.0)
        self.off_track_transition_seconds = max(0.0, float(OFF_TRACK_SPEED_TRANSITION_SECONDS))
        self.off_track_penalty_margin_px = max(1.0, float(OFF_TRACK_PENALTY_MARGIN_PX))
        self.off_track_terminate_steps = max(1, int(OFF_TRACK_TERMINATE_STEPS))
        self.off_track_terminate_severity = self._clamp(float(OFF_TRACK_TERMINATE_SEVERITY), 0.0, 1.0)
        self.track_valid_hysteresis_px = max(0.0, float(TRACK_VALID_HYSTERESIS_PX))
        self.random_start_min_remaining_progress_norm = self._clamp(
            float(RANDOM_START_MIN_REMAINING_PROGRESS_NORM),
            0.0,
            1.0,
        )
        if self.off_track_transition_seconds <= 1e-6:
            self.off_track_blend_step = 1.0
        else:
            self.off_track_blend_step = 1.0 / (float(max(1, FPS)) * self.off_track_transition_seconds)
        self.steer_speed_decay = max(0.0, float(STEER_SPEED_DECAY))
        self.turn_throttle_loss = self._clamp(float(TURN_THROTTLE_LOSS), 0.0, 0.95)
        self.lateral_damping_on_track = self._clamp(float(LATERAL_DAMPING_ON_TRACK), 0.0, 1.0)
        self.lateral_damping_off_track = self._clamp(float(LATERAL_DAMPING_OFF_TRACK), 0.0, 1.0)
        self.edge_probe_max_distance = max(8.0, float(EDGE_PROBE_MAX_DISTANCE_PX))
        self.forward_ray_max_distance = max(self.edge_probe_max_distance, float(FORWARD_RAY_MAX_DISTANCE_PX))
        self.route_lookahead_ranges_px = tuple(
            (max(1.0, float(low)), max(1.0, float(high)))
            for low, high in ROUTE_LOOKAHEAD_RANGES_PX
        )
        if len(self.route_lookahead_ranges_px) != 3:
            raise RuntimeError("Vroom ROUTE_LOOKAHEAD_RANGES_PX must define exactly three route probes.")
        self.steer_full_speed_norm = max(1e-6, float(STEER_FULL_SPEED_NORM))
        self.steer_min_speed_factor = self._clamp(float(STEER_MIN_SPEED_FACTOR), 0.0, 1.0)
        self.sens_car_range_px = max(self.car_contact_radius * 2.0 + 1.0, float(SENS_CAR_RANGE_PX))
        self.sens_car_side_range_px = max(self.car_contact_radius * 2.0 + 1.0, float(SENS_CAR_SIDE_RANGE_PX))
        self.contact_sep_strength = 1.0
        self.contact_overlap_cap = self.car_radius * 0.12
        self.contact_damp = 0.12
        self.contact_accel_scale = 0.85
        self.no_progress_timeout_steps = max(1, int(NO_PROGRESS_TIMEOUT_STEPS))
        self.no_progress_eps_norm = max(0.0, float(NO_PROGRESS_EPS_NORM))

        self.track_config = TrackGenConfig(
            track_width_px=float(TRACK_WIDTH_PX),
            padding_px=float(TRACK_PADDING_PX),
            footprint_scale=float(TRACK_FOOTPRINT_SCALE),
            corner_radius_px=float(TRACK_CORNER_RADIUS_PX),
            sample_spacing_px=float(TRACK_SAMPLE_SPACING_PX),
            start_straight_len_px=float(TRACK_START_STRAIGHT_LEN_PX),
            long_side_template_choices=tuple(str(value) for value in TRACK_LONG_SIDE_TEMPLATE_CHOICES),
            short_side_template_choices=tuple(str(value) for value in TRACK_SHORT_SIDE_TEMPLATE_CHOICES),
            bell_amplitude_min_px=float(TRACK_LONG_SIDE_BELL_AMPLITUDE_MIN_PX),
            bell_amplitude_max_px=float(TRACK_LONG_SIDE_BELL_AMPLITUDE_MAX_PX),
            s_amplitude_min_px=float(TRACK_LONG_SIDE_S_AMPLITUDE_MIN_PX),
            s_amplitude_max_px=float(TRACK_LONG_SIDE_S_AMPLITUDE_MAX_PX),
            inset_width_cap_ratio=float(TRACK_LONG_SIDE_INSET_WIDTH_CAP_RATIO),
            inset_length_cap_ratio=float(TRACK_LONG_SIDE_INSET_LENGTH_CAP_RATIO),
            fold_gap_px=float(TRACK_FOLD_GAP_PX),
            bend_smoothing_passes=int(TRACK_BEND_SMOOTHING_PASSES),
            generation_max_attempts=int(TRACK_GENERATION_MAX_ATTEMPTS),
        )
        self.track_width_px = float(self.track_config.track_width_px)
        self.track_half_width = self.track_width_px * 0.5
        initial_complexity_range = LEVEL_SETTINGS[int(MIN_LEVEL)]["track_complexity_range"]
        self.track_complexity_range = tuple(float(value) for value in initial_complexity_range)
        self.track_complexity_hard_sample_rate = self._clamp(float(TRACK_COMPLEXITY_HARD_SAMPLE_RATE), 0.0, 1.0)
        self.current_track_complexity = 0.0

        self.track_geometry: TrackGeometry | None = None
        self.track_mask = np.zeros((int(self.track_bottom), SCREEN_WIDTH), dtype=np.uint8)
        self.collision_mask = np.zeros((int(self.track_bottom), SCREEN_WIDTH), dtype=np.uint8)
        self.track_texture: arcade.Texture | None = None
        self.wall_texture: arcade.Texture | None = None
        self.track_rect = arcade.LRBT(0.0, float(SCREEN_WIDTH), float(BB_HEIGHT), float(SCREEN_HEIGHT))
        self.background_rect = arcade.LRBT(0.0, float(SCREEN_WIDTH), 0.0, float(SCREEN_HEIGHT))

        self.track_seed = 0
        self.track_count = 0
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_index = 0
        self.start_line: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))
        self.start_side = "top"
        self.start_tangent = (1.0, 0.0)
        self.start_normal = (0.0, 1.0)
        self.cars: list[RaceCar] = []
        self.player_color_pairs = list(ACCENT_PAIRS)
        self.player_index = 0
        self.winner_index: int | None = None
        self.last_race_winner: int | None = None
        self.total_races = int(self._resolve_total_races())
        self.match_tracker = MatchTracker[int](
            match_limit=int(self.total_races),
            clock_duration_steps=(int(self.max_steps) if int(self.max_steps) > 0 else None),
        )
        self.current_race = 1
        self.win_history: list[int | None] = self.match_tracker.history
        self.num_cars = int(self.NUM_CARS)
        self.opponent_speed_cap = 1.0
        self.random_start_prob = 0.0
        curriculum_config = build_curriculum_config(
            min_level=int(MIN_LEVEL),
            max_level=int(MAX_LEVEL),
            promotion_settings=self._curriculum_promotion_settings(),
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

        self.steps = 0
        self.done = False
        self.show_rays = bool(DRAW_RAYS)
        self._prev_ghost_overlay_toggle_down = False
        self.last_action = np.zeros((self.ACT_DIM,), dtype=np.float32)
        self._last_sensor_origin = (0.0, 0.0)
        self._last_edge_ray_values = np.ones((5,), dtype=np.float32)
        self._last_edge_ray_dirs: list[tuple[float, float]] = [(1.0, 0.0)] * 5
        self._last_edge_ray_max_distances = tuple(
            [float(self.forward_ray_max_distance)] + [float(self.edge_probe_max_distance)] * 4
        )
        self._last_car_ray_values = np.ones((3,), dtype=np.float32)
        self._last_car_ray_dirs: list[tuple[float, float]] = [(1.0, 0.0)] * 3
        self._last_car_ray_max_distances = (
            float(self.sens_car_side_range_px),
            float(self.sens_car_range_px),
            float(self.sens_car_side_range_px),
        )
        self._last_route_markers: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        self._player_contact_memory_steps = 0
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self.race_random_start_active = False
        self._player_soft_off_track = False
        self._prev_s_norm = 0.0
        self._target_progress_norm = 1.0
        self._unwrapped_progress_norm = 0.0
        self._best_unwrapped_progress_norm = 0.0
        self._no_progress_anchor_norm = 0.0
        self._no_progress_steps = 0
        self._hard_offtrack_steps = 0
        self._race_progress_reward_total = 0.0
        self._prev_player_forward_speed = 0.0
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)
        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _resolve_total_races(self) -> int:
        if self.mode == "train":
            return int(self.TRAINING_TOTAL_RACES)
        return int(self.PLAY_TOTAL_RACES)

    @staticmethod
    def _curriculum_promotion_settings() -> dict[str, object]:
        return dict(CURRICULUM_PROMOTION)

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level), LEVEL_SETTINGS[int(MIN_LEVEL)])
        self.num_cars = max(1, min(int(settings["num_cars"]), len(self.player_color_pairs)))
        self.opponent_speed_cap = self._clamp(float(settings.get("opponent_speed_cap", 1.0)), 0.0, 1.0)
        self.random_start_prob = self._clamp(float(settings.get("random_start_prob", 0.0)), 0.0, 1.0)
        complexity_range = settings.get("track_complexity_range")
        if not isinstance(complexity_range, (list, tuple)) or len(complexity_range) != 2:
            raise ValueError("Vroom LEVEL_SETTINGS entries must define track_complexity_range as a pair.")
        self.track_complexity_range = (
            self._clamp(float(complexity_range[0]), 0.0, 1.0),
            self._clamp(float(complexity_range[1]), 0.0, 1.0),
        )
        if self.track_complexity_range[1] < self.track_complexity_range[0]:
            self.track_complexity_range = (self.track_complexity_range[1], self.track_complexity_range[0])

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    @staticmethod
    def _lerp(low: float, high: float, t: float) -> float:
        return float(low) + (float(high) - float(low)) * float(t)

    def _track_progress_norm_from_s(self, track: TrackGeometry, s_value: float) -> float:
        if float(track.length) <= 1e-9:
            return 0.0
        progress_px = (float(s_value) - float(track.start_s)) % float(track.length)
        return float(progress_px) / float(track.length)

    def _remaining_progress_to_finish_norm(self, progress_norm: float) -> float:
        progress = float(progress_norm) % 1.0
        remaining = 1.0 - float(progress)
        if float(remaining) <= 1e-9:
            return 1.0
        return self._clamp(float(remaining), 0.0, 1.0)

    @staticmethod
    def _normalize(dx: float, dy: float) -> tuple[float, float]:
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return 1.0, 0.0
        return dx / length, dy / length

    @staticmethod
    def _normalize_degrees(degrees: float) -> float:
        return (float(degrees) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _sample_range(value_range: tuple[float, float] | list[float]) -> float:
        low = float(value_range[0])
        high = float(value_range[1])
        if high < low:
            low, high = high, low
        return float(random.uniform(low, high))

    def _route_lookaheads_for_speed(self, speed_norm: float) -> tuple[float, float, float]:
        speed = self._clamp(float(speed_norm), 0.0, 1.0)
        return (
            self._lerp(self.route_lookahead_ranges_px[0][0], self.route_lookahead_ranges_px[0][1], speed),
            self._lerp(self.route_lookahead_ranges_px[1][0], self.route_lookahead_ranges_px[1][1], speed),
            self._lerp(self.route_lookahead_ranges_px[2][0], self.route_lookahead_ranges_px[2][1], speed),
        )

    def _hard_complexity_threshold(self, complexity_range: tuple[float, float] | None = None) -> float:
        low, high = self.track_complexity_range if complexity_range is None else complexity_range
        low = self._clamp(float(low), 0.0, 1.0)
        high = self._clamp(float(high), 0.0, 1.0)
        if high < low:
            low, high = high, low
        return float(low + (high - low) * (2.0 / 3.0))

    def _sample_track_complexity(self) -> float:
        low, high = self.track_complexity_range
        low = self._clamp(float(low), 0.0, 1.0)
        high = self._clamp(float(high), 0.0, 1.0)
        if high < low:
            low, high = high, low
        if high <= low + 1e-9:
            return float(high)
        if random.random() < float(self.track_complexity_hard_sample_rate):
            return self._sample_range((self._hard_complexity_threshold((low, high)), high))
        return self._sample_range((low, high))

    def _require_track_geometry(self) -> TrackGeometry:
        if self.track_geometry is None:
            raise RuntimeError("Track geometry is not initialized.")
        return self.track_geometry

    def _generate_track(self, seed: int) -> None:
        complexity = float(self._sample_track_complexity())
        self.current_track_complexity = float(complexity)
        track_config = replace(
            self.track_config,
            complexity_min=float(complexity),
            complexity_max=float(complexity),
        )
        track = generate_track(
            seed=int(seed),
            width=int(SCREEN_WIDTH),
            height=int(self.track_bottom),
            config=track_config,
            build_texture=bool(self.show_game),
            track_color=COLOR_SLATE_GRAY,
        )
        geometry = track.get("geometry")
        if not isinstance(geometry, TrackGeometry):
            raise RuntimeError("Vroom track generation did not return canonical geometry.")
        self.track_geometry = geometry
        self.track_mask = np.asarray(track["road_mask"], dtype=np.uint8)  # type: ignore[arg-type]
        self.collision_mask = np.asarray(track.get("collision_mask", track["road_mask"]), dtype=np.uint8)  # type: ignore[arg-type]
        self.track_texture = track["road_texture"] if self.show_game else None  # type: ignore[assignment]
        self.wall_texture = track["wall_texture"] if self.show_game else None  # type: ignore[assignment]

        self.track_count = int(len(geometry.centerline))

        self.track_width_px = 2.0 * float(geometry.half_width)
        self.track_half_width = float(geometry.half_width)
        self.start_index = int(geometry.start_index % max(1, self.track_count))
        self.start_x = float(geometry.start_pos[0])
        self.start_y = float(geometry.start_pos[1])
        self.start_side = str(geometry.start_side)
        self.start_tangent = (float(geometry.start_tangent[0]), float(geometry.start_tangent[1]))
        self.start_normal = (float(geometry.start_normal[0]), float(geometry.start_normal[1]))
        self.start_line = (
            (float(geometry.start_line[0][0]), float(geometry.start_line[0][1])),
            (float(geometry.start_line[1][0]), float(geometry.start_line[1][1])),
        )
    def _is_on_track(self, x: float, y: float) -> bool:
        if self.collision_mask.size == 0:
            return False
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        if iy < 0 or ix < 0 or iy >= int(self.collision_mask.shape[0]) or ix >= int(self.collision_mask.shape[1]):
            return False
        return bool(self.collision_mask[iy, ix] > 0)

    def _is_on_track_footprint(self, x: float, y: float) -> bool:
        for ox, oy in self.track_probe_offsets:
            px = float(x) + ox
            py = float(y) + oy
            if not self._is_on_track(px, py):
                return False
        return True

    def _track_coverage_ratio(self, x: float, y: float) -> float:
        samples = self.off_track_probe_offsets
        if not samples:
            return 1.0 if self._is_on_track(float(x), float(y)) else 0.0
        on_count = 0
        total = len(samples)
        for ox, oy in samples:
            if self._is_on_track(float(x) + float(ox), float(y) + float(oy)):
                on_count += 1
        return float(on_count) / float(max(1, total))

    def _car_offtrack_severity(self, car: RaceCar) -> float:
        track = self._require_track_geometry()
        coverage_severity = 1.0 - self._track_coverage_ratio(float(car.x), float(car.y))
        proj = project_point_to_track(track, (float(car.x), float(car.y)))
        safe_allowed = max(0.0, float(track.half_width) - float(self.car_radius) * 0.8)
        offtrack_excess_px = max(0.0, abs(float(proj.lateral_offset)) - float(safe_allowed))
        centerline_severity = self._clamp(
            float(offtrack_excess_px) / float(self.off_track_penalty_margin_px),
            0.0,
            1.0,
        )
        return self._clamp(max(float(coverage_severity), float(centerline_severity)), 0.0, 1.0)

    def _player_offtrack_severity(self) -> float:
        if not self.cars:
            return 1.0
        return self._car_offtrack_severity(self.cars[self.player_index])

    def _car_finish_valid(self, car: RaceCar) -> bool:
        if not self._is_on_track(float(car.x), float(car.y)):
            return False
        track = self._require_track_geometry()
        proj = project_point_to_track(track, (float(car.x), float(car.y)))
        safe_allowed = max(0.0, float(track.half_width) - float(self.car_radius) * 0.8)
        return bool(abs(float(proj.lateral_offset)) <= float(safe_allowed))

    def _player_soft_offtrack_flag(self, severity: float | None = None) -> bool:
        offtrack_severity = self._player_offtrack_severity() if severity is None else float(severity)
        lateral_flag = self._player_soft_lateral_flag()
        if bool(lateral_flag):
            self._player_soft_off_track = True
        elif bool(self._player_soft_off_track):
            self._player_soft_off_track = bool(float(offtrack_severity) > 0.0)
        else:
            self._player_soft_off_track = bool(float(offtrack_severity) > 1e-6)
        return bool(self._player_soft_off_track)

    def _player_valid_progress_step(self, offtrack_severity: float) -> tuple[float, float, bool]:
        curr_s_norm = float(self._player_raw_progress_norm())
        progress_delta = float(curr_s_norm) - float(self._prev_s_norm)
        if float(progress_delta) < -0.5:
            progress_delta += 1.0
        elif float(progress_delta) > 0.5:
            progress_delta -= 1.0
        self._prev_s_norm = float(curr_s_norm)

        progress_valid = bool(float(offtrack_severity) <= 1e-6)
        clipped_delta = 0.0
        if progress_valid:
            clipped_delta = self._clamp(float(progress_delta), -float(PROGRESS_CLIP), float(PROGRESS_CLIP))
            self._unwrapped_progress_norm += float(clipped_delta)

        if self.cars:
            track = self._require_track_geometry()
            player = self.cars[self.player_index]
            player.lap_progress = self._clamp(
                float(self._unwrapped_progress_norm) * float(track.length),
                0.0,
                float(track.length),
            )

        return float(progress_delta), float(clipped_delta), bool(progress_valid)

    def _player_soft_lateral_flag(self) -> bool:
        if not self.cars:
            return False
        player = self.cars[self.player_index]
        track = self._require_track_geometry()
        proj = project_point_to_track(track, (float(player.x), float(player.y)))
        allowed = float(track.half_width) - float(self.car_radius) * 0.8
        on_threshold = max(0.0, float(allowed))
        off_threshold = max(0.0, float(allowed) - float(self.track_valid_hysteresis_px))
        abs_offset = abs(float(proj.lateral_offset))
        if bool(self._player_soft_off_track):
            return bool(abs_offset > off_threshold)
        return bool(abs_offset > on_threshold)

    def _update_off_track_state(self, car: RaceCar) -> None:
        # Physics still uses the real road mask; learning validity is handled separately.
        coverage = self._track_coverage_ratio(float(car.x), float(car.y))
        if bool(car.off_track):
            if coverage >= float(self.off_track_exit_ratio):
                car.off_track = False
        elif coverage <= float(self.off_track_enter_ratio):
            car.off_track = True

        target = 1.0 if bool(car.off_track) else 0.0
        step = float(max(1e-6, self.off_track_blend_step))
        if float(car.off_track_blend) < target:
            car.off_track_blend = min(target, float(car.off_track_blend) + step)
        else:
            car.off_track_blend = max(target, float(car.off_track_blend) - step)

    def _off_track_speed_multiplier(self, car: RaceCar) -> float:
        blend = self._clamp(float(car.off_track_blend), 0.0, 1.0)
        return self._clamp(
            1.0 - blend * (1.0 - float(self.off_track_max_speed_factor)),
            float(self.off_track_max_speed_factor),
            1.0,
        )

    def _surface_grip(self, car: RaceCar) -> float:
        blend = self._clamp(float(car.off_track_blend), 0.0, 1.0)
        return self._clamp(
            1.0 - blend * (1.0 - float(self.off_track_surface_grip)),
            float(self.off_track_surface_grip),
            1.0,
        )

    def _lateral_damping(self, surface_grip: float) -> float:
        off_track_ratio = 1.0 - self._clamp(float(surface_grip), 0.0, 1.0)
        return self._clamp(
            float(self.lateral_damping_on_track)
            + off_track_ratio * (float(self.lateral_damping_off_track) - float(self.lateral_damping_on_track)),
            0.0,
            1.0,
        )

    def _signed_angle_norm(self, from_x: float, from_y: float, to_x: float, to_y: float) -> float:
        dot = self._clamp(float(from_x) * float(to_x) + float(from_y) * float(to_y), -1.0, 1.0)
        cross = float(from_x) * float(to_y) - float(from_y) * float(to_x)
        angle = math.atan2(cross, dot)
        return float(clip_signed(angle / math.pi))

    def _probe_distance(
        self,
        origin_x: float,
        origin_y: float,
        dir_x: float,
        dir_y: float,
        max_distance: float,
    ) -> float:
        ux, uy = self._normalize(float(dir_x), float(dir_y))
        max_dist = max(1.0, float(max_distance))
        if not self._is_on_track(float(origin_x), float(origin_y)):
            return 0.0

        hit = raycast_track_edge(
            self._require_track_geometry(),
            origin=(float(origin_x), float(origin_y)),
            direction=(float(ux), float(uy)),
            max_dist=float(max_dist),
        )
        if hit is None:
            return 1.0
        return float(clip_unit(float(max(0.0, hit)) / float(max_dist)))

    def _create_car_grid(self) -> list[RaceCar]:
        track = self._require_track_geometry()
        cars: list[RaceCar] = []
        car_count = max(1, int(self.num_cars))
        if car_count == 1:
            lane_offsets = [0.0]
        else:
            max_lane_span = max(0.0, float(track.half_width) - self.car_half * 0.68)
            lane_span = min(max_lane_span, max(self.car_size * 0.82, max_lane_span * 0.82))
            lane_offsets = np.linspace(-lane_span, lane_span, num=car_count, dtype=np.float32).astype(float).tolist()
        random.shuffle(lane_offsets)
        spawn_slots = list(range(int(self.num_cars)))
        random.shuffle(spawn_slots)

        longitudinal_spacing = max(self.car_size * 0.95, float(track.half_width) * 0.58)
        start_back_offset = max(self.car_size * 1.35, float(track.half_width) * 0.95)

        for idx in range(int(self.num_cars)):
            desired_lateral = float(lane_offsets[idx % len(lane_offsets)])
            spawn_slot = int(spawn_slots[idx % len(spawn_slots)])
            (x, y), heading = spawn_pose(
                track,
                slot_idx=spawn_slot,
                lateral_offset=float(desired_lateral),
                longitudinal_spacing=float(longitudinal_spacing),
                start_back_offset=float(start_back_offset),
            )
            outer, inner = self.player_color_pairs[idx % len(self.player_color_pairs)]
            is_player = int(idx) == int(self.player_index)
            speed_mult = 1.0 if bool(is_player) else self._sample_range(OPPONENT_SPEED_MULT_RANGE)
            bend_caution_mult = 1.0 if bool(is_player) else self._sample_range(OPPONENT_BEND_CAUTION_MULT_RANGE)
            car = RaceCar(
                x=float(x),
                y=float(y),
                vx=0.0,
                vy=0.0,
                heading_degrees=float(heading),
                outer_color=outer,
                inner_color=inner,
                ai_lane_home=float(desired_lateral),
                ai_lane_offset=float(desired_lateral),
                speed_mult=float(speed_mult),
                bend_caution_mult=float(bend_caution_mult),
            )
            if not self._is_on_track_footprint(car.x, car.y):
                (fallback_x, fallback_y), fallback_heading = spawn_pose(
                    track,
                    slot_idx=0,
                    lateral_offset=0.0,
                    longitudinal_spacing=float(longitudinal_spacing),
                    start_back_offset=float(start_back_offset),
                )
                car.x = float(fallback_x)
                car.y = float(fallback_y)
                car.heading_degrees = float(fallback_heading)
                car.ai_lane_home = 0.0
                car.ai_lane_offset = 0.0
            if not self._is_on_track_footprint(car.x, car.y):
                raise RuntimeError("Spawn placement failed: generated spawn is off track.")
            proj = project_point_to_track(track, (float(car.x), float(car.y)))
            car.track_index = int(proj.seg_index % max(1, self.track_count))
            progress = float((float(proj.s) - float(track.start_s)) % float(track.length))
            car.track_progress = float(progress)
            car.prev_track_progress = float(progress)
            car.lap_progress = 0.0
            car.lap_armed = False
            car.off_track = False
            car.off_track_blend = 0.0
            cars.append(car)
        return cars

    def _should_use_random_start(self) -> bool:
        return bool(self.mode == "train" and random.random() < float(self.random_start_prob))

    def _is_clear_of_random_start_placements(
        self,
        x: float,
        y: float,
        placements: list[tuple[RaceCar, float, float, float, float, float, float]],
    ) -> bool:
        min_distance = max(1.0, 2.15 * float(self.car_contact_radius))
        for _other, other_x, other_y, _heading, _speed, _lateral, _progress in placements:
            if self._distance(float(x), float(y), float(other_x), float(other_y)) < float(min_distance):
                return False
        return True

    def _apply_race_random_start(self) -> bool:
        if not self.cars:
            return False
        track = self._require_track_geometry()
        if float(track.length) <= 1e-6:
            return False

        safe_lateral = max(0.0, float(track.half_width) - 1.15 * float(self.car_radius))
        car_count = len(self.cars)
        if car_count <= 1:
            lane_offsets = [random.uniform(-safe_lateral, safe_lateral) if safe_lateral > 1e-6 else 0.0]
        else:
            lane_offsets = np.linspace(-safe_lateral, safe_lateral, num=car_count, dtype=np.float32).astype(float).tolist()
            random.shuffle(lane_offsets)
        slots = list(range(car_count))
        random.shuffle(slots)
        longitudinal_spacing = max(self.car_size * 0.95, float(track.half_width) * 0.58)

        for _ in range(24):
            base_s = random.uniform(0.0, float(track.length))
            placements: list[tuple[RaceCar, float, float, float, float, float, float]] = []
            for idx, car in enumerate(self.cars):
                slot = int(slots[idx % len(slots)])
                sample_s = float(base_s) - float(slot) * float(longitudinal_spacing)
                progress_norm = self._track_progress_norm_from_s(track, float(sample_s))
                remaining_norm = self._remaining_progress_to_finish_norm(float(progress_norm))
                if float(remaining_norm) < float(self.random_start_min_remaining_progress_norm):
                    break
                lateral = float(lane_offsets[idx % len(lane_offsets)])
                (center_x, center_y), (tan_x, tan_y), (norm_x, norm_y) = sample_track_at_s(track, sample_s)
                x = float(center_x) + float(norm_x) * float(lateral)
                y = float(center_y) + float(norm_y) * float(lateral)
                if not self._is_on_track_footprint(float(x), float(y)):
                    break
                if not self._is_clear_of_random_start_placements(float(x), float(y), placements):
                    break

                tangent_heading = math.degrees(math.atan2(float(tan_y), float(tan_x)))
                heading = self._normalize_degrees(float(tangent_heading) + random.uniform(-45.0, 45.0))
                speed = random.uniform(0.0, 0.25 * float(self.max_speed))
                placements.append(
                    (car, float(x), float(y), float(heading), float(speed), float(lateral), float(progress_norm))
                )
            else:
                for car, x, y, heading, speed, lateral, _progress_norm in placements:
                    heading_rad = math.radians(float(heading))
                    car.x = float(x)
                    car.y = float(y)
                    car.heading_degrees = float(heading)
                    car.vx = math.cos(heading_rad) * float(speed)
                    car.vy = math.sin(heading_rad) * float(speed)
                    car.yaw_rate = 0.0
                    car.in_contact = False
                    car.off_track = False
                    car.off_track_blend = 0.0
                    car.lap_armed = False
                    car.lap_progress = 0.0
                    car.finished = False
                    proj = project_point_to_track(track, (float(car.x), float(car.y)))
                    progress = float((float(proj.s) - float(track.start_s)) % float(track.length))
                    car.track_index = int(proj.seg_index % max(1, self.track_count))
                    car.track_progress = float(progress)
                    car.prev_track_progress = float(progress)
                    car.ai_lane_home = float(lateral)
                    car.ai_lane_offset = float(lateral)
                    car.ai_rejoin_steps = 12
                return True
        return False

    def _car_axes(self, car: RaceCar) -> tuple[tuple[float, float], tuple[float, float]]:
        heading_rad = math.radians(car.heading_degrees)
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        right_x = -forward_y
        right_y = forward_x
        return (float(forward_x), float(forward_y)), (float(right_x), float(right_y))

    def _project_to_car_frame(self, car: RaceCar) -> tuple[float, float, float, float]:
        (forward_x, forward_y), (right_x, right_y) = self._car_axes(car)
        forward_speed = car.vx * forward_x + car.vy * forward_y
        lateral_speed = car.vx * right_x + car.vy * right_y
        return forward_x, forward_y, forward_speed, lateral_speed

    def _apply_car_controls(
        self,
        car: RaceCar,
        steer: float,
        throttle: float,
        brake: float,
        *,
        max_forward_speed: float | None = None,
    ) -> None:
        _, _, forward_speed, lateral_speed = self._project_to_car_frame(car)
        speed_ratio = self._clamp(abs(float(forward_speed)) / max(1e-6, float(self.max_speed)), 0.0, 1.0)
        surface_grip = self._surface_grip(car)
        low_speed_steer_scale = self._clamp(
            float(speed_ratio) / float(self.steer_full_speed_norm),
            float(self.steer_min_speed_factor),
            1.0,
        )
        effective_steer = float(steer) * float(low_speed_steer_scale)
        steer_speed_scale = 1.0 / (1.0 + float(self.steer_speed_decay) * speed_ratio * speed_ratio)
        steer_surface_scale = 0.45 + 0.55 * float(surface_grip)
        steer_authority = self._clamp(float(steer_speed_scale) * float(steer_surface_scale), 0.12, 1.0)
        heading_delta = float(effective_steer) * float(self.turn_rate) * float(steer_authority)
        car.heading_degrees = self._normalize_degrees(float(car.heading_degrees) + heading_delta)
        car.yaw_rate = float(heading_delta)

        heading_rad = math.radians(car.heading_degrees)
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        side_x = -forward_y
        side_y = forward_x

        accel_scale = self.contact_accel_scale if car.in_contact else 1.0
        steer_load = self._clamp(abs(float(effective_steer)), 0.0, 1.0)
        throttle_turn_scale = 1.0 - float(self.turn_throttle_loss) * (steer_load**1.25)
        throttle_turn_scale = self._clamp(throttle_turn_scale, 0.0, 1.0)
        effective_throttle = self._clamp(float(throttle), 0.0, 1.0)
        effective_brake = self._clamp(float(brake), 0.0, 1.0)
        if effective_throttle > 0.0:
            effective_throttle *= throttle_turn_scale
        forward_speed += effective_throttle * self.accel_force * accel_scale * float(surface_grip)
        if effective_brake > 0.0:
            forward_speed -= effective_brake * self.brake_force * accel_scale * float(surface_grip)
        allowed_forward_speed = float(self.max_speed if max_forward_speed is None else max_forward_speed)
        forward_speed = self._clamp(forward_speed, -self.max_reverse_speed, allowed_forward_speed)
        lateral_speed *= self._lateral_damping(float(surface_grip))

        car.vx = forward_x * forward_speed + side_x * lateral_speed
        car.vy = forward_y * forward_speed + side_y * lateral_speed
        car.vx *= self.drag
        car.vy *= self.drag

    def _resolve_car_contacts(self) -> None:
        positions = [(car.x, car.y) for car in self.cars]
        velocities = [(car.vx, car.vy) for car in self.cars]
        radii = [float(self.car_contact_radius)] * len(self.cars)
        new_positions, new_velocities, contact_flags = resolve_circle_collisions(
            positions,
            velocities,
            radii,
            sep_strength=self.contact_sep_strength,
            overlap_cap=self.contact_overlap_cap,
            contact_damp=self.contact_damp,
        )

        for idx, car in enumerate(self.cars):
            car.x, car.y = new_positions[idx]
            car.vx, car.vy = new_velocities[idx]
            car.in_contact = bool(contact_flags[idx])

    def _resolve_screen_bounds(self, car: RaceCar) -> None:
        min_x = self.car_half
        max_x = SCREEN_WIDTH - self.car_half
        min_y = self.car_half
        max_y = self.track_bottom - self.car_half
        clamped_x = self._clamp(car.x, min_x, max_x)
        clamped_y = self._clamp(car.y, min_y, max_y)
        if clamped_x != car.x:
            car.vx = 0.0
        if clamped_y != car.y:
            car.vy = 0.0
        car.x = clamped_x
        car.y = clamped_y

    def _update_lap_progress_and_finish(self) -> None:
        track = self._require_track_geometry()
        if self.track_count <= 0 or float(track.length) <= 1e-9:
            return
        lap_length = float(track.length)
        arm_threshold = 0.70 * lap_length
        for idx, car in enumerate(self.cars):
            if car.finished:
                continue

            prev_s = float(car.track_progress)
            proj = project_point_to_track(track, (float(car.x), float(car.y)))
            curr_s = float((float(proj.s) - float(track.start_s)) % lap_length)
            car.prev_track_progress = float(prev_s)
            car.track_progress = float(curr_s)
            car.track_index = int(proj.seg_index % max(1, self.track_count))

            wrapped_delta = curr_s - prev_s
            if wrapped_delta > 0.5 * lap_length:
                wrapped_delta -= lap_length
            elif wrapped_delta < -0.5 * lap_length:
                wrapped_delta += lap_length

            is_forward = wrapped_delta > 0.0
            if int(idx) == int(self.player_index):
                on_track_now = self._car_finish_valid(car)
            else:
                on_track_now = self._track_coverage_ratio(car.x, car.y) >= float(self.off_track_exit_ratio)
            crossed_arm_threshold_forward = prev_s < arm_threshold <= curr_s and is_forward
            if (not car.lap_armed) and on_track_now and crossed_arm_threshold_forward:
                car.lap_armed = True

            crossed_start_forward = (
                curr_s < prev_s
                and is_forward
            )
            car.lap_progress = self._clamp(float(car.lap_progress) + float(wrapped_delta), 0.0, float(lap_length))

            canonical_finish = bool(
                on_track_now
                and crossed_start_forward
                and (bool(car.lap_armed) or bool(self.race_random_start_active))
            )
            if canonical_finish:
                car.finished = True
                car.lap_armed = False
                car.lap_progress = lap_length
                if self.winner_index is None:
                    self.winner_index = idx

    def _player_raw_progress_norm(self) -> float:
        track = self._require_track_geometry()
        if not self.cars or float(track.length) <= 1e-9:
            return 0.0
        player = self.cars[self.player_index]
        raw_progress = float(player.track_progress) / float(max(1e-6, track.length))
        return float(raw_progress % 1.0)

    def _update_no_progress_guard(self) -> bool:
        self._best_unwrapped_progress_norm = max(
            float(self._best_unwrapped_progress_norm),
            float(self._unwrapped_progress_norm),
        )
        if (
            float(self._best_unwrapped_progress_norm)
            >= float(self._no_progress_anchor_norm) + float(self.no_progress_eps_norm)
        ):
            self._no_progress_anchor_norm = float(self._best_unwrapped_progress_norm)
            self._no_progress_steps = 0
            return False

        self._no_progress_steps += 1
        return bool(int(self._no_progress_steps) >= int(self.no_progress_timeout_steps))

    def _setup_race(self) -> None:
        self.track_seed = random.randint(0, 2_000_000_000)
        self._generate_track(self.track_seed)
        self.cars = self._create_car_grid()
        self.race_random_start_active = bool(self._should_use_random_start() and self._apply_race_random_start())
        for car in self.cars:
            car.ai_lane_offset = float(car.ai_lane_home)
        self.winner_index = None
        self.steps = 0
        self._player_soft_off_track = False
        self._prev_s_norm = float(self._player_raw_progress_norm())
        self._target_progress_norm = (
            self._remaining_progress_to_finish_norm(float(self._prev_s_norm))
            if bool(self.race_random_start_active)
            else 1.0
        )
        self._unwrapped_progress_norm = 0.0
        self._best_unwrapped_progress_norm = 0.0
        self._no_progress_anchor_norm = 0.0
        self._no_progress_steps = 0
        self._hard_offtrack_steps = 0
        self._race_progress_reward_total = 0.0
        self._prev_player_forward_speed = 0.0
        self._player_contact_memory_steps = 0

    def _finalize_race(self, winner_idx: int | None) -> None:
        self.last_race_winner = None if winner_idx is None else int(winner_idx)
        self.match_tracker.record_result(self.last_race_winner)
        if self.match_tracker.match_limit_reached():
            self.done = True
            self.winner_index = self.last_race_winner
            return
        self.current_race = int(self.match_tracker.matches_played()) + 1
        self._setup_race()

    def _resolve_human_action(self) -> np.ndarray:
        left = self.window_controller.is_key_down(arcade.key.LEFT) or self.window_controller.is_key_down(arcade.key.A)
        right = self.window_controller.is_key_down(arcade.key.RIGHT) or self.window_controller.is_key_down(arcade.key.D)
        throttle = self.window_controller.is_key_down(arcade.key.UP) or self.window_controller.is_key_down(arcade.key.W)
        brake = self.window_controller.is_key_down(arcade.key.DOWN) or self.window_controller.is_key_down(arcade.key.S)
        steer = 0.0
        if left and (not right):
            steer = -1.0
        elif right and (not left):
            steer = 1.0
        return self._normalized_action_vector(
            steer=float(steer),
            throttle=(1.0 if bool(throttle) else 0.0),
            brake=(1.0 if bool(brake) else 0.0),
        )

    def _can_toggle_visual_overlay(self) -> bool:
        return bool(self.show_game and self.mode in {"human", "eval"})

    def _update_visual_overlay_toggle(self) -> None:
        self.show_rays, self._prev_ghost_overlay_toggle_down = update_ghost_overlay_toggle(
            window_controller=self.window_controller,
            visible=bool(self.show_rays),
            previous_down=bool(self._prev_ghost_overlay_toggle_down),
            enabled=bool(self._can_toggle_visual_overlay()),
        )

    def _ai_lane_limit(self) -> float:
        return max(4.0, float(self.track_half_width) - float(self.track_probe_radius) - 1.0)

    def _opponent_lane_target(self, car: RaceCar, signed_lateral: float) -> float:
        lane_limit = self._ai_lane_limit()
        if bool(car.in_contact) or bool(car.off_track):
            car.ai_rejoin_steps = max(int(car.ai_rejoin_steps), 18)
            car.ai_lane_offset = self._clamp(float(signed_lateral), -lane_limit, lane_limit)

        blend = 0.08
        if int(car.ai_rejoin_steps) > 0:
            blend = 0.18
            car.ai_rejoin_steps = max(0, int(car.ai_rejoin_steps) - 1)
        car.ai_lane_offset += (float(car.ai_lane_home) - float(car.ai_lane_offset)) * float(blend)
        if abs(float(car.ai_lane_offset) - float(car.ai_lane_home)) <= 0.25:
            car.ai_lane_offset = float(car.ai_lane_home)
        car.ai_lane_offset = self._clamp(float(car.ai_lane_offset), -lane_limit, lane_limit)
        return float(car.ai_lane_offset)

    def _opponent_bend_severity(
        self,
        track: TrackGeometry,
        proj: TrackProjection,
        speed_ratio: float,
    ) -> float:
        lookaheads_px = self._route_lookaheads_for_speed(float(speed_ratio))
        bends: list[float] = []
        for lookahead in lookaheads_px:
            probe_s = float(proj.s) + float(lookahead)
            bends.append(float(self._route_bend_for_lookahead(track, float(probe_s), float(lookahead))))
        if len(bends) < 3:
            return 0.0
        bend = max(
            float(bends[0]),
            0.90 * float(bends[1]),
            0.80 * float(bends[2]),
            0.75 * float(sum(bends)) / float(len(bends)),
        )
        return self._clamp(float(bend), 0.0, 1.0)

    def _ai_control_for_car(self, car: RaceCar) -> tuple[float, float, float]:
        track = self._require_track_geometry()
        proj = project_point_to_track(track, (float(car.x), float(car.y)))
        (forward_x, forward_y), _ = self._car_axes(car)
        car_speed = math.hypot(float(car.vx), float(car.vy))
        max_forward_speed = max(
            0.0,
            float(self.max_speed) * float(self.opponent_speed_cap) * float(car.speed_mult),
        )
        if float(max_forward_speed) <= 1e-6:
            return 0.0, 0.0, 1.0

        signed_lateral = float(proj.lateral_offset)
        speed_ratio = self._clamp(float(car_speed) / max(1.0, float(self.max_speed)), 0.0, 1.0)
        bend = self._opponent_bend_severity(track, proj, float(speed_ratio))
        bend = self._clamp(float(bend) * float(car.bend_caution_mult), 0.0, 1.0)
        lane_limit = max(1e-6, float(self._ai_lane_limit()))
        edge_risk = self._clamp(abs(float(signed_lateral)) / float(lane_limit), 0.0, 1.0)
        edge_recovery = self._clamp((float(edge_risk) - 0.55) / 0.35, 0.0, 1.0)
        normal_x, normal_y = float(proj.normal[0]), float(proj.normal[1])
        outward_heading = 0.0
        if abs(float(signed_lateral)) > 1e-6:
            side = 1.0 if float(signed_lateral) > 0.0 else -1.0
            outward_heading = self._clamp(
                side * (float(forward_x) * normal_x + float(forward_y) * normal_y),
                0.0,
                1.0,
            )
        edge_pressure = max(float(edge_recovery), float(edge_risk) * float(outward_heading))

        lane_target = self._opponent_lane_target(car, signed_lateral)
        path_safety = max(float(edge_pressure), self._clamp(float(bend) * 1.15, 0.0, 1.0))
        lane_target *= 1.0 - 0.55 * float(edge_pressure)
        if bool(car.off_track):
            lane_target *= 0.50

        look_ahead_s = self._clamp(
            self._lerp(48.0, 84.0, float(speed_ratio)) * self._lerp(1.0, 0.72, float(edge_recovery)),
            32.0,
            88.0,
        )
        (target_x, target_y), _, (target_nx, target_ny) = sample_track_at_s(
            track,
            float(proj.s) + float(look_ahead_s),
        )
        aim_x = float(target_x) + target_nx * float(lane_target)
        aim_y = float(target_y) + target_ny * float(lane_target)
        desired_heading = math.degrees(math.atan2(aim_y - car.y, aim_x - car.x))
        delta = self._normalize_degrees(desired_heading - car.heading_degrees)
        turn_demand = self._clamp(abs(float(delta)) / 58.0, 0.0, 1.0)
        safety = max(float(path_safety), float(turn_demand))

        if abs(delta) <= 1.5:
            steer = 0.0
        else:
            steer_gain = self._lerp(16.0, 10.0, float(safety))
            steer = self._clamp(float(delta) / float(steer_gain), -1.0, 1.0)

        bend_speed_factor = self._lerp(1.0, float(OPPONENT_MIN_BEND_SPEED_FACTOR), float(bend))
        edge_speed_factor = self._lerp(1.0, 0.45, float(edge_pressure))
        turn_speed_factor = self._lerp(1.0, 0.50, float(turn_demand))
        target_speed = float(max_forward_speed) * min(
            float(bend_speed_factor),
            float(edge_speed_factor),
            float(turn_speed_factor),
        )
        if bool(car.in_contact) or bool(car.off_track):
            target_speed *= 0.65

        current_speed = max(0.0, float(car_speed))
        if float(current_speed) > float(target_speed):
            throttle = 0.0
            brake = self._clamp(
                (float(current_speed) - float(target_speed))
                / max(1e-6, float(OPPONENT_BRAKE_RESPONSE) * float(self.max_speed)),
                0.0,
                1.0,
            )
        else:
            throttle = self._clamp(1.0 - 0.85 * float(safety), 0.10, 1.0)
            brake = 0.0

        return float(steer), float(throttle), float(brake)

    def _normalized_action_vector(self, *, steer: float, throttle: float, brake: float) -> np.ndarray:
        return np.asarray(
            [
                self._clamp(float(steer), -1.0, 1.0),
                self._clamp(float(throttle), 0.0, 1.0),
                self._clamp(float(brake), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _player_controls_from_action(self, action: object) -> tuple[float, float, float]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != int(self.ACT_DIM):
            raise ValueError(f"Vroom expected continuous action with {self.ACT_DIM} values, got {action_array.size}.")
        normalized = self._normalized_action_vector(
            steer=float(action_array[0]),
            throttle=float(action_array[1]),
            brake=float(action_array[2]),
        )
        return float(normalized[0]), float(normalized[1]), float(normalized[2])

    def _step_simulation(self, action: object) -> None:
        for car in self.cars:
            self._update_off_track_state(car)

        # Opponent top speed is curriculum-capped, then reduced by the shared off-road speed limit.
        for idx, car in enumerate(self.cars):
            speed_multiplier = self._off_track_speed_multiplier(car)
            if idx == self.player_index:
                steer, throttle, brake = self._player_controls_from_action(action)
                allowed_speed = float(self.max_speed) * float(speed_multiplier)
            else:
                steer, throttle, brake = self._ai_control_for_car(car)
                effective_speed_cap = float(self.opponent_speed_cap) * float(car.speed_mult)
                allowed_speed = float(self.max_speed) * float(effective_speed_cap) * float(speed_multiplier)
            self._apply_car_controls(car, steer, throttle, brake, max_forward_speed=allowed_speed)

        for car in self.cars:
            car.x += car.vx
            car.y += car.vy

        self._resolve_car_contacts()
        if self.cars:
            player = self.cars[self.player_index]
            if bool(player.in_contact):
                self._player_contact_memory_steps = 3
            else:
                self._player_contact_memory_steps = max(0, int(self._player_contact_memory_steps) - 1)
        for car in self.cars:
            self._resolve_screen_bounds(car)
        self._update_lap_progress_and_finish()

    def _edge_probe_values(
        self,
        car: RaceCar,
    ) -> tuple[float, float, float, float, float, list[tuple[float, float]], tuple[float, ...]]:
        (forward_x, forward_y), (right_x, right_y) = self._car_axes(car)
        left_x = -float(right_x)
        left_y = -float(right_y)

        def _norm(dx: float, dy: float) -> tuple[float, float]:
            return self._normalize(float(dx), float(dy))

        f_dir = (float(forward_x), float(forward_y))
        fl_dir = _norm(forward_x + 0.70 * left_x, forward_y + 0.70 * left_y)
        fr_dir = _norm(forward_x + 0.70 * right_x, forward_y + 0.70 * right_y)
        l_dir = (float(left_x), float(left_y))
        r_dir = (float(right_x), float(right_y))
        probe_dirs = [f_dir, fl_dir, fr_dir, l_dir, r_dir]
        probe_max_distances = (
            float(self.forward_ray_max_distance),
            float(self.edge_probe_max_distance),
            float(self.edge_probe_max_distance),
            float(self.edge_probe_max_distance),
            float(self.edge_probe_max_distance),
        )
        probe_vals = [
            self._probe_distance(
                float(car.x),
                float(car.y),
                float(dir_x),
                float(dir_y),
                float(max_distance),
            )
            for (dir_x, dir_y), max_distance in zip(probe_dirs, probe_max_distances)
        ]
        return (
            float(probe_vals[0]),
            float(probe_vals[1]),
            float(probe_vals[2]),
            float(probe_vals[3]),
            float(probe_vals[4]),
            probe_dirs,
            probe_max_distances,
        )

    def _route_sensor_values(
        self,
        track: TrackGeometry,
        proj: TrackProjection,
        car: RaceCar,
        heading_x: float,
        heading_y: float,
        right_x: float,
        right_y: float,
        lookaheads_px: tuple[float, float, float],
    ) -> tuple[dict[str, float], list[tuple[tuple[float, float], tuple[float, float], float]]]:
        feature_values: dict[str, float] = {}
        markers: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        for idx, lookahead in enumerate(lookaheads_px, start=1):
            probe_s = float(proj.s) + float(lookahead)
            (point_x, point_y), (tan_x, tan_y), _ = sample_track_at_s(
                track,
                float(probe_s),
            )
            dx = float(point_x) - float(car.x)
            dy = float(point_y) - float(car.y)
            fwd = dx * float(heading_x) + dy * float(heading_y)
            lat = dx * float(right_x) + dy * float(right_y)
            tan_err = float(self._signed_angle_norm(heading_x, heading_y, tan_x, tan_y))
            tan_err_rad = float(tan_err) * math.pi
            fwd_scale = max(1.0, float(lookahead))
            lat_scale = max(float(track.half_width), 0.45 * float(lookahead), 1.0)
            bend = self._route_bend_for_lookahead(track, float(probe_s), float(lookahead))

            prefix = f"sens_route{idx}"
            feature_values[f"{prefix}_fwd"] = float(clip_signed(float(fwd) / float(fwd_scale)))
            feature_values[f"{prefix}_lat"] = float(clip_signed(float(lat) / float(lat_scale)))
            feature_values[f"{prefix}_tan_sin"] = float(clip_signed(math.sin(tan_err_rad)))
            feature_values[f"{prefix}_tan_cos"] = float(clip_signed(math.cos(tan_err_rad)))
            feature_values[f"{prefix}_bend"] = float(bend)
            markers.append(((float(point_x), float(point_y)), (float(tan_x), float(tan_y)), float(bend)))
        return feature_values, markers

    def _route_bend_for_lookahead(self, track: TrackGeometry, probe_s: float, lookahead: float) -> float:
        bend_window = min(120.0, max(42.0, float(track.half_width) + 0.35 * float(lookahead)))
        return float(self._bend_sensor_value(track, float(probe_s), float(bend_window)))

    def _bend_sensor_value(self, track: TrackGeometry, center_s: float, window_s: float) -> float:
        if float(track.length) <= 1e-6:
            return 0.0
        sample_count = 8
        span = min(max(12.0, float(window_s)), 0.25 * float(track.length))
        sample_start = float(center_s) - 0.5 * float(span)
        _, prev_tangent, _ = sample_track_at_s(track, float(sample_start))
        total_abs_turn = 0.0
        for sample_idx in range(1, sample_count + 1):
            sample_s = float(sample_start) + float(span) * float(sample_idx) / float(sample_count)
            _, tangent, _ = sample_track_at_s(track, sample_s)
            turn = self._signed_angle_norm(
                float(prev_tangent[0]),
                float(prev_tangent[1]),
                float(tangent[0]),
                float(tangent[1]),
            )
            total_abs_turn += abs(float(turn))
            prev_tangent = tangent
        return float(clip_unit(float(total_abs_turn) / 0.50))

    def _car_clearance_values(
        self,
        car: RaceCar,
    ) -> tuple[tuple[float, float, float], list[tuple[float, float]], tuple[float, ...]]:
        (forward_x, forward_y), (right_x, right_y) = self._car_axes(car)
        path_len = float(self.sens_car_range_px)
        side_lat = 0.58 * float(self.sens_car_side_range_px)
        side_start_lat = 0.70 * float(self.car_contact_radius)
        side_end_fwd = 0.82 * float(path_len)
        path_specs = (
            (-1, (0.45 * float(self.car_contact_radius), -side_start_lat), (side_end_fwd, -side_lat)),
            (0, (0.75 * float(self.car_contact_radius), 0.0), (path_len, 0.0)),
            (1, (0.45 * float(self.car_contact_radius), side_start_lat), (side_end_fwd, side_lat)),
        )
        sensor_dirs: list[tuple[float, float]] = []
        sensor_max_distances: list[float] = []
        for _sign, start, end in path_specs:
            delta_fwd = float(end[0]) - float(start[0])
            delta_right = float(end[1]) - float(start[1])
            dir_x = float(forward_x) * delta_fwd + float(right_x) * delta_right
            dir_y = float(forward_y) * delta_fwd + float(right_y) * delta_right
            sensor_dirs.append(self._normalize(float(dir_x), float(dir_y)))
            sensor_max_distances.append(max(1.0, math.hypot(delta_fwd, delta_right)))

        values = [1.0, 1.0, 1.0]
        touch_distance = max(1.0, 2.0 * float(self.car_contact_radius))
        block_distance = max(1.0, float(self.car_contact_radius) * 1.72)
        soft_distance = max(16.0, float(self.car_contact_radius) * 0.9)
        nearby_limit = max(float(path_len), float(side_end_fwd)) + abs(float(side_lat)) + float(touch_distance)

        def _point_segment_distance(
            px: float,
            py: float,
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> tuple[float, float, float]:
            sx, sy = float(start[0]), float(start[1])
            ex, ey = float(end[0]), float(end[1])
            vx = ex - sx
            vy = ey - sy
            denom = max(1e-9, vx * vx + vy * vy)
            t = self._clamp(((float(px) - sx) * vx + (float(py) - sy) * vy) / denom, 0.0, 1.0)
            nearest_x = sx + vx * float(t)
            nearest_y = sy + vy * float(t)
            return math.hypot(float(px) - nearest_x, float(py) - nearest_y), float(t), float(nearest_x)

        for other in self.cars:
            if other is car or bool(other.finished):
                continue
            dx = float(other.x) - float(car.x)
            dy = float(other.y) - float(car.y)
            center_distance = math.hypot(dx, dy)
            if float(center_distance) > float(nearby_limit):
                continue

            local_fwd = dx * float(forward_x) + dy * float(forward_y)
            local_right = dx * float(right_x) + dy * float(right_y)
            if float(center_distance) <= float(touch_distance):
                if abs(float(local_right)) > 0.75 * abs(float(local_fwd)):
                    contact_idx = 0 if float(local_right) < 0.0 else 2
                elif float(local_fwd) >= -0.35 * float(touch_distance):
                    contact_idx = 1
                else:
                    contact_idx = -1
                if int(contact_idx) >= 0:
                    values[int(contact_idx)] = 0.0

            for sensor_idx, (side_sign, start, end) in enumerate(path_specs):
                if int(side_sign) < 0 and float(local_right) > 0.55 * float(self.car_contact_radius):
                    continue
                if int(side_sign) > 0 and float(local_right) < -0.55 * float(self.car_contact_radius):
                    continue
                if (
                    int(side_sign) == 0
                    and abs(float(local_right)) > 0.85 * float(touch_distance)
                    and float(local_fwd) < 1.10 * float(touch_distance)
                ):
                    continue

                distance_to_path, _segment_t, nearest_fwd = _point_segment_distance(
                    float(local_fwd),
                    float(local_right),
                    start,
                    end,
                )
                if float(nearest_fwd) < -0.25 * float(touch_distance):
                    continue
                lateral_clearance = clip_unit(
                    (float(distance_to_path) - float(block_distance)) / float(soft_distance)
                )
                along_clearance = clip_unit(
                    (float(nearest_fwd) - float(touch_distance))
                    / max(1.0, float(end[0]) - float(touch_distance))
                )
                value = max(float(lateral_clearance), float(along_clearance))
                values[int(sensor_idx)] = min(float(values[int(sensor_idx)]), float(value))

        return (
            (float(values[0]), float(values[1]), float(values[2])),
            sensor_dirs,
            tuple(float(value) for value in sensor_max_distances),
        )

    def _compute_obs(self) -> np.ndarray:
        track = self._require_track_geometry()
        player = self.cars[self.player_index]
        proj = project_point_to_track(track, (float(player.x), float(player.y)))
        tangent_x, tangent_y = float(proj.tangent[0]), float(proj.tangent[1])
        lat_off = clip_signed(float(proj.lateral_offset) / max(1e-6, float(track.half_width)))

        heading_rad = math.radians(player.heading_degrees)
        heading_x = math.cos(heading_rad)
        heading_y = math.sin(heading_rad)
        _, (right_x, right_y) = self._car_axes(player)
        _, _, fwd_speed_raw, lat_speed_raw = self._project_to_car_frame(player)
        spd_fwd = clip_signed(float(fwd_speed_raw) / max(1.0, float(self.max_speed)))
        route_speed = self._clamp(float(spd_fwd), 0.0, 1.0)
        route_lookaheads_px = self._route_lookaheads_for_speed(float(route_speed))
        lat_vel = clip_signed(float(lat_speed_raw) / max(1.0, float(self.max_speed)))
        spd_delta = clip_signed(float(fwd_speed_raw - self._prev_player_forward_speed) / max(1.0, float(self.accel_force * 2.0)))
        yaw_rt = clip_signed(float(player.yaw_rate) / float(max(1e-6, self.yaw_rate_norm)))
        heading_err = float(self._signed_angle_norm(heading_x, heading_y, tangent_x, tangent_y))
        heading_err_rad = float(heading_err) * math.pi
        heading_err_sin = clip_signed(math.sin(heading_err_rad))
        heading_err_cos = clip_signed(math.cos(heading_err_rad))

        route_values, route_markers = self._route_sensor_values(
            track,
            proj,
            player,
            float(heading_x),
            float(heading_y),
            float(right_x),
            float(right_y),
            route_lookaheads_px,
        )

        ray_f, edg_fl, edg_fr, edg_l, edg_r, probe_dirs, probe_max_distances = self._edge_probe_values(player)
        car_values, car_dirs, car_max_distances = self._car_clearance_values(player)
        self._last_sensor_origin = (float(player.x), float(player.y))
        self._last_edge_ray_values = np.asarray(
            [ray_f, edg_fl, edg_fr, edg_l, edg_r],
            dtype=np.float32,
        )
        self._last_edge_ray_dirs = [(float(dx), float(dy)) for dx, dy in probe_dirs]
        self._last_edge_ray_max_distances = tuple(float(value) for value in probe_max_distances)
        self._last_car_ray_values = np.asarray(car_values, dtype=np.float32)
        self._last_car_ray_dirs = [(float(dx), float(dy)) for dx, dy in car_dirs]
        self._last_car_ray_max_distances = tuple(float(value) for value in car_max_distances)
        self._last_route_markers = route_markers
        self._prev_player_forward_speed = float(fwd_speed_raw)

        feature_values = {
            "self_lat_off": float(lat_off),
            "self_spd_lat": float(lat_vel),
            "self_spd_fwd": float(spd_fwd),
            "self_spd_delta": float(spd_delta),
            "self_yaw_rate": float(yaw_rt),
            "self_head_err_sin": float(heading_err_sin),
            "self_head_err_cos": float(heading_err_cos),
            "sens_edge_fwd": float(ray_f),
            "sens_edge_left_front": float(edg_fl),
            "sens_edge_right_front": float(edg_fr),
            "sens_edge_left": float(edg_l),
            "sens_edge_right": float(edg_r),
            "sens_car_left": float(car_values[0]),
            "sens_car_fwd": float(car_values[1]),
            "sens_car_right": float(car_values[2]),
            "flag_contact": 1.0 if bool(player.in_contact) or int(self._player_contact_memory_steps) > 0 else 0.0,
            "flag_off_track": 1.0 if self._player_soft_offtrack_flag() else 0.0,
        }
        feature_values.update(route_values)
        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Vroom observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        self._last_obs = obs
        return obs

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self.match_tracker.set_match_limit(int(self.total_races))
        self.match_tracker.set_clock_duration(int(self.max_steps) if int(self.max_steps) > 0 else None)
        self.current_race = 1
        self.match_tracker.clear_history()
        self.last_race_winner = None
        self.player_index = 0
        self.done = False
        self._episode_reward_components.reset()
        self.last_action = np.zeros((self.ACT_DIM,), dtype=np.float32)
        self._setup_race()
        return self._compute_obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            done_info: dict[str, object] = {
                "win": self.last_race_winner == self.player_index,
                "success": int(self._last_episode_success),
                "level": int(self._last_episode_level),
                "reward_components": self._episode_reward_components.totals(),
            }
            return self._last_obs, 0.0, True, done_info

        self.window_controller.poll_events_or_raise()
        if self.show_game:
            self._update_visual_overlay_toggle()
        if self.mode == "human":
            action_array = self._resolve_human_action()
        else:
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != int(self.ACT_DIM):
            raise ValueError(f"Vroom expected action with {self.ACT_DIM} values, got {action_array.size}.")
        action_array = np.asarray(
            [
                self._clamp(float(action_array[0]), -1.0, 1.0),
                self._clamp(float(action_array[1]), 0.0, 1.0),
                self._clamp(float(action_array[2]), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        self.last_action = action_array.copy()

        self._step_simulation(action_array)
        self.steps += 1

        reward = 0.0
        reward_breakdown = {
            "step.penalty_step": 0.0,
            "progress.shape": 0.0,
            "track.penalty_coverage": 0.0,
            "event.penalty_collision": 0.0,
            "outcome.reward_win": 0.0,
            "outcome.penalty_lose": 0.0,
        }
        episode_level = int(self._current_level)
        episode_success = 0
        stuck_no_progress = False
        hard_offtrack_timeout = False
        player = self.cars[self.player_index]
        offtrack_severity = float(self._player_offtrack_severity())
        _progress_delta, clipped_progress_delta, progress_valid = self._player_valid_progress_step(
            float(offtrack_severity)
        )
        if self.mode != "human":
            reward += float(PENALTY_STEP)
            reward_breakdown["step.penalty_step"] = float(PENALTY_STEP)

            progress_reward = 0.0
            if bool(progress_valid):
                progress_reward = float(PROGRESS_SCALE) * float(clipped_progress_delta)
            if float(progress_reward) > 0.0:
                progress_reward = min(
                    float(progress_reward),
                    max(0.0, float(PROGRESS_SCALE) - float(self._race_progress_reward_total)),
                )
            reward += progress_reward
            reward_breakdown["progress.shape"] = progress_reward
            self._race_progress_reward_total += float(progress_reward)

            offtrack_penalty = float(PENALTY_OFF_TRACK) * float(offtrack_severity)
            reward += offtrack_penalty
            reward_breakdown["track.penalty_coverage"] = offtrack_penalty

            if bool(player.in_contact):
                reward += float(PENALTY_CONTACT)
                reward_breakdown["event.penalty_collision"] = float(PENALTY_CONTACT)

            if float(offtrack_severity) >= float(self.off_track_terminate_severity):
                self._hard_offtrack_steps += 1
            else:
                self._hard_offtrack_steps = 0
            hard_offtrack_timeout = bool(
                int(self._hard_offtrack_steps) >= int(self.off_track_terminate_steps)
            )

            if self.winner_index is None and not bool(hard_offtrack_timeout):
                stuck_no_progress = bool(self._update_no_progress_guard())

        timed_out = bool((self.winner_index is None) and (self.steps >= self.max_steps))
        race_finished = bool(
            (self.winner_index is not None)
            or timed_out
            or stuck_no_progress
            or hard_offtrack_timeout
        )
        if race_finished:
            race_winner = int(self.winner_index) if self.winner_index is not None else None
            if self.mode != "human":
                player_won = race_winner == self.player_index
                if player_won:
                    progress_target_reward = float(PROGRESS_SCALE) * float(self._target_progress_norm)
                    progress_top_up = max(
                        0.0,
                        float(progress_target_reward) - float(self._race_progress_reward_total),
                    )
                    reward += float(progress_top_up)
                    reward_breakdown["progress.shape"] += float(progress_top_up)
                    self._race_progress_reward_total += float(progress_top_up)
                    reward += float(REWARD_WIN)
                    reward_breakdown["outcome.reward_win"] = float(REWARD_WIN)
                else:
                    reward += float(PENALTY_LOSE)
                    reward_breakdown["outcome.penalty_lose"] = float(PENALTY_LOSE)
                episode_success = 1 if player_won else 0
            self._finalize_race(race_winner)
        if self.mode != "human":
            self._episode_reward_components.add_from_mapping(reward_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)

        if self.show_game:
            self.render()
            self._tick_arcade_frame()

        info = {
            "win": bool(self.last_race_winner == self.player_index) if race_finished else False,
            "success": int(episode_success) if race_finished else 0,
            "winner_index": -1 if self.last_race_winner is None else int(self.last_race_winner),
            "race": int(min(self.current_race, self.total_races)),
            "races_finished": int(len(self.win_history)),
            "races_total": int(self.total_races),
            "level": int(episode_level),
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
        return self._compute_obs(), float(reward), bool(self.done), info

    def _draw_track(self) -> None:
        if self.track_texture is not None:
            arcade.draw_texture_rect(self.track_texture, self.track_rect)

    def _draw_route_markers(self) -> None:
        point_color = ghost_color(170)
        tangent_color = COLOR_AQUA + (150,)
        tangent_len = 18.0
        for (point_x, point_y), (tan_x, tan_y), bend in self._last_route_markers:
            screen_x = float(point_x)
            screen_y = float(self.window_controller.to_arcade_y(float(point_y)))
            end_x = float(point_x) + float(tan_x) * float(tangent_len)
            end_y = float(point_y) + float(tan_y) * float(tangent_len)
            screen_end_x = float(end_x)
            screen_end_y = float(self.window_controller.to_arcade_y(float(end_y)))
            bend_alpha = int(90 + 150 * self._clamp(float(bend), 0.0, 1.0))
            bend_radius = 3.0 + 3.5 * self._clamp(float(bend), 0.0, 1.0)
            arcade.draw_line(screen_x, screen_y, screen_end_x, screen_end_y, tangent_color, 1.5)
            arcade.draw_circle_filled(screen_x, screen_y, bend_radius, COLOR_AQUA + (bend_alpha,))
            arcade.draw_circle_outline(screen_x, screen_y, bend_radius, point_color, 1.0)

    def _draw_rays(self) -> None:
        if not bool(self.show_rays and self.show_game):
            return
        origin_x, origin_y = self._last_sensor_origin
        self._draw_route_markers()
        draw_player_rays(
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            ray_dirs=self._last_edge_ray_dirs,
            ray_values=self._last_edge_ray_values.tolist(),
            ray_max_distances=self._last_edge_ray_max_distances,
            to_screen=lambda x, y: (float(x), float(self.window_controller.to_arcade_y(float(y)))),
            line_width=1.0,
        )
        draw_player_rays(
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            ray_dirs=self._last_car_ray_dirs,
            ray_values=self._last_car_ray_values.tolist(),
            ray_max_distances=self._last_car_ray_max_distances,
            to_screen=lambda x, y: (float(x), float(self.window_controller.to_arcade_y(float(y)))),
            color=COLOR_CORAL + (135,),
            line_width=1.25,
        )

    def _draw_cars(self) -> None:
        for idx, car in enumerate(self.cars):
            draw_two_tone_cell(
                self.window_controller,
                top_left_x=car.x - self.car_half,
                top_left_y=car.y - self.car_half,
                tile_size=self.car_size,
                outer_color=car.outer_color,
                inner_color=car.inner_color,
                cell_inset=float(CELL_INSET),
            )
            draw_facing_indicator(
                self.window_controller,
                center_x=car.x,
                center_y_top_left=car.y,
                angle_degrees=car.heading_degrees,
                length=self.car_half * 1.35,
                color=COLOR_LIGHT_NEUTRAL if idx == self.player_index else COLOR_FOG_GRAY,
                line_width=2.0,
            )

    @staticmethod
    def _status_icon_size() -> float:
        return status_icon_size(float(BB_HEIGHT), float(TILE_SIZE))

    def _draw_player_icon(self, winner_idx: int, center_x: float, center_y: float, size: float) -> None:
        pair = self.player_color_pairs[int(winner_idx) % len(self.player_color_pairs)]
        outline_color, fill_color = pair[0], pair[1]
        inset = status_icon_inset(4.0)
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outline_color,
            inner_color=fill_color,
            inset=float(inset),
        )

    def _draw_winner_history(self, left: float, right: float, center_y: float) -> None:
        icon_size = self._status_icon_size()
        draw_status_icon_row(
            left=float(left),
            right=float(right),
            center_y=float(center_y),
            icon_size=float(icon_size),
            items=list(self.win_history),
            draw_item=lambda winner, icon_center_x, row_center_y, size: self._draw_player_icon(
                int(winner),
                float(icon_center_x),
                float(row_center_y),
                float(size),
            )
            if winner is not None
            else None,
        )

    def _remaining_time_ratio(self) -> float:
        return float(self.match_tracker.remaining_time_ratio(int(self.steps)))

    def _draw_status_bar(self) -> None:
        include_clock = self.match_tracker.clock_duration_steps is not None
        bar_layout = draw_status_bar(
            width=float(SCREEN_WIDTH),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=4.0,
            include_clock=bool(include_clock),
        )
        draw_status_clock(
            layout=bar_layout,
            remaining_ratio=float(self._remaining_time_ratio()),
        )
        self._draw_winner_history(
            float(bar_layout.score_left),
            float(bar_layout.score_right),
            float(bar_layout.center_y),
        )

    def render(self) -> float:
        if self.window_controller.window is None:
            return 0.0

        draw_t0 = time.perf_counter()
        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        self._draw_track()
        self._draw_cars()
        self._draw_rays()
        self._draw_status_bar()
        self.window_controller.flip()
        return time.perf_counter() - draw_t0

    def close(self) -> None:
        super().close()
