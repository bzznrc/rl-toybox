"""Top-down one-lap racing environment with geometry-first procedural tracks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BLUE,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_PURPLE,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_NAVY,
    COLOR_PURPLE,
    COLOR_SLATE_GRAY,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import (
    clip_signed,
    clip_unit,
    ordered_feature_vector,
    signed_potential_shaping,
)
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_facing_indicator,
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    draw_status_square_icon,
    draw_two_tone_tile,
    resolve_circle_collisions,
    status_icon_inset,
    status_icon_size,
)
from core.ray_viz import draw_player_rays
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level
from games.vroom.config import (
    ACTION_NAMES as VROOM_ACTION_NAMES,
    ACT_DIM as VROOM_ACT_DIM,
    BB_HEIGHT,
    CURRICULUM_PROMOTION,
    DRAW_RAYS,
    FPS,
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
    OFF_TRACK_SPEED_TRANSITION_SECONDS,
    EDGE_PROBE_MAX_DISTANCE_PX,
    PENALTY_COLLISION,
    PENALTY_LOSE,
    PENALTY_STEP,
    PROGRESS_CLIP,
    PROGRESS_SCALE,
    REWARD_WIN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STEER_SPEED_DECAY,
    TILE_SIZE,
    TRACK_CORNER_RADIUS_PX,
    TRACK_FOOTPRINT_SCALE,
    TRACK_LONG_SIDE_BELL_AMPLITUDE_MAX_PX,
    TRACK_LONG_SIDE_BELL_AMPLITUDE_MIN_PX,
    TRACK_LONG_SIDE_INSET_LENGTH_CAP_RATIO,
    TRACK_LONG_SIDE_INSET_WIDTH_CAP_RATIO,
    TRACK_LONG_SIDE_S_AMPLITUDE_MAX_PX,
    TRACK_LONG_SIDE_S_AMPLITUDE_MIN_PX,
    TRACK_LONG_SIDE_TEMPLATE_CHOICES,
    TRACK_PADDING_PX,
    TRACK_SAMPLE_SPACING_PX,
    TRACK_START_STRAIGHT_LEN_PX,
    TRACK_WIDTH_PX,
    TURN_THROTTLE_LOSS,
    TRAINING_FPS,
    WINDOW_TITLE,
)
from games.vroom.track_geometry import (
    TrackGeometry,
    TrackProjection,
    build_boundary_loops,
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
    ai_curve_error_percent: float = 0.0
    ai_curve_key: int = -1
    ai_rejoin_steps: int = 0
    off_track: bool = False
    off_track_blend: float = 0.0
    track_progress: float = 0.0
    prev_track_progress: float = 0.0
    lap_armed: bool = False
    track_index: int = 0
    lap_progress: float = 0.0
    finished: bool = False


class VroomEnv(Env):
    INPUT_FEATURE_NAMES = tuple(VROOM_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(VROOM_ACTION_NAMES)
    OBS_DIM = int(VROOM_OBS_DIM)
    ACT_DIM = int(VROOM_ACT_DIM)

    NUM_CARS = 4
    TRAINING_TOTAL_RACES = 1
    PLAY_TOTAL_RACES = 10
    REWARD_COMPONENT_ORDER = ("W", "L", "P", "C", "S")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_win": "W",
        "outcome.penalty_lose": "L",
        "progress.shape": "P",
        "event.penalty_collision": "C",
        "step.penalty_step": "S",
    }

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
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

        self.track_bottom = float(SCREEN_HEIGHT - BB_HEIGHT)
        self.track_half_width = float(TILE_SIZE * 1.25)

        self.car_size = float(TILE_SIZE * 0.86)
        self.car_half = self.car_size * 0.5
        self.car_radius = self.car_half * 0.95
        # Heavier handling: full-throttle cornering should understeer unless the car coasts.
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
        self.contact_sep_strength = 1.0
        self.contact_overlap_cap = self.car_radius * 0.12
        self.contact_damp = 0.12
        self.contact_accel_scale = 0.85

        self.track_config = TrackGenConfig(
            track_width_px=float(TRACK_WIDTH_PX),
            padding_px=float(TRACK_PADDING_PX),
            footprint_scale=float(TRACK_FOOTPRINT_SCALE),
            corner_radius_px=float(TRACK_CORNER_RADIUS_PX),
            sample_spacing_px=float(TRACK_SAMPLE_SPACING_PX),
            start_straight_len_px=float(TRACK_START_STRAIGHT_LEN_PX),
            long_side_template_choices=tuple(str(value) for value in TRACK_LONG_SIDE_TEMPLATE_CHOICES),
            bell_amplitude_min_px=float(TRACK_LONG_SIDE_BELL_AMPLITUDE_MIN_PX),
            bell_amplitude_max_px=float(TRACK_LONG_SIDE_BELL_AMPLITUDE_MAX_PX),
            s_amplitude_min_px=float(TRACK_LONG_SIDE_S_AMPLITUDE_MIN_PX),
            s_amplitude_max_px=float(TRACK_LONG_SIDE_S_AMPLITUDE_MAX_PX),
            inset_width_cap_ratio=float(TRACK_LONG_SIDE_INSET_WIDTH_CAP_RATIO),
            inset_length_cap_ratio=float(TRACK_LONG_SIDE_INSET_LENGTH_CAP_RATIO),
        )
        self.track_width_px = float(self.track_config.track_width_px)
        self.track_half_width = self.track_width_px * 0.5

        self.track_geometry: TrackGeometry | None = None
        self.track_centerline: list[tuple[float, float]] = []
        self.track_tangents: list[tuple[float, float]] = []
        self.track_mask = np.zeros((int(self.track_bottom), SCREEN_WIDTH), dtype=np.uint8)
        self.collision_mask = np.zeros((int(self.track_bottom), SCREEN_WIDTH), dtype=np.uint8)
        self.track_points_np = np.zeros((0, 2), dtype=np.float32)
        self.track_x_np = np.zeros((0,), dtype=np.float32)
        self.track_y_np = np.zeros((0,), dtype=np.float32)
        self.track_texture: arcade.Texture | None = None
        self.wall_texture: arcade.Texture | None = None
        self._track_left_outline_screen: list[tuple[float, float]] = []
        self._track_right_outline_screen: list[tuple[float, float]] = []
        self.track_rect = arcade.LRBT(0.0, float(SCREEN_WIDTH), float(BB_HEIGHT), float(SCREEN_HEIGHT))
        self.background_rect = arcade.LRBT(0.0, float(SCREEN_WIDTH), 0.0, float(SCREEN_HEIGHT))

        self.track_seed = 0
        self.track_count = 0
        self.max_track_index_step = 1
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_index = 0
        self.start_line: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))
        self.start_side = "top"
        self.start_tangent = (1.0, 0.0)
        self.start_normal = (0.0, 1.0)

        self.cars: list[RaceCar] = []
        self.player_color_pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
            (COLOR_AQUA, COLOR_DEEP_TEAL),
            (COLOR_CORAL, COLOR_BRICK_RED),
            (COLOR_BLUE, COLOR_NAVY),
            (COLOR_PURPLE, COLOR_DEEP_PURPLE),
        ]
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
        self.opponent_coast_error_choices: tuple[float, ...] = (0.0,)
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
            else resolve_play_level(level=level, min_level=MIN_LEVEL, max_level=MAX_LEVEL, default_level=3)
        )
        self._last_episode_level = int(self._current_level)
        self._last_episode_success = 0

        self.steps = 0
        self.done = False
        self.show_rays = bool(DRAW_RAYS)
        self._prev_overlay_toggle_down = False
        self.last_action = np.zeros((self.ACT_DIM,), dtype=np.float32)
        self._last_ray_values = np.ones((5,), dtype=np.float32)
        self._last_ray_origin = (0.0, 0.0)
        self._last_ray_dirs: list[tuple[float, float]] = [(1.0, 0.0)] * 5
        self._last_ray_max_distances = tuple(
            [float(self.forward_ray_max_distance)] + [float(self.edge_probe_max_distance)] * 4
        )
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._prev_progress_potential = 0.0
        self._prev_player_in_contact = False
        self._prev_player_forward_speed = 0.0
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)
        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _resolve_total_races(self) -> int:
        if self.mode == "train":
            return int(self.TRAINING_TOTAL_RACES)
        return int(self.PLAY_TOTAL_RACES)

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level), LEVEL_SETTINGS[int(MIN_LEVEL)])
        self.num_cars = max(1, min(int(settings["num_cars"]), len(self.player_color_pairs)))
        self.opponent_speed_cap = self._clamp(float(settings["opponent_speed_cap"]), 0.0, 1.0)
        coast_error_choices = settings.get("opponent_coast_error_choices", [0.0])
        if not isinstance(coast_error_choices, (list, tuple)) or len(coast_error_choices) == 0:
            raise ValueError("Vroom LEVEL_SETTINGS entries must define non-empty opponent_coast_error_choices.")
        try:
            self.opponent_coast_error_choices = tuple(float(value) for value in coast_error_choices)
        except (TypeError, ValueError) as exc:
            raise ValueError("Vroom opponent_coast_error_choices entries must be numeric.") from exc

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

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

    def _require_track_geometry(self) -> TrackGeometry:
        if self.track_geometry is None:
            raise RuntimeError("Track geometry is not initialized.")
        return self.track_geometry

    def _cache_track_draw_geometry(self) -> None:
        self._track_left_outline_screen = []
        self._track_right_outline_screen = []
        track = self.track_geometry
        if track is None:
            return

        left_pts, right_pts = build_boundary_loops(track, seam_index=int(track.start_index))
        if len(left_pts) > 2:
            left_loop = [
                (float(px), float(self.window_controller.to_arcade_y(float(py))))
                for px, py in left_pts
            ]
            left_loop.append(left_loop[0])
            self._track_left_outline_screen = left_loop
        if len(right_pts) > 2:
            right_loop = [
                (float(px), float(self.window_controller.to_arcade_y(float(py))))
                for px, py in right_pts
            ]
            right_loop.append(right_loop[0])
            self._track_right_outline_screen = right_loop

    def _generate_track(self, seed: int) -> None:
        track = generate_track(
            seed=int(seed),
            width=int(SCREEN_WIDTH),
            height=int(self.track_bottom),
            config=self.track_config,
            build_texture=bool(self.show_game),
            track_color=COLOR_SLATE_GRAY,
        )
        geometry = track.get("geometry")
        if not isinstance(geometry, TrackGeometry):
            raise RuntimeError("Vroom track generation did not return canonical geometry.")
        self.track_geometry = geometry
        self.track_centerline = [
            (float(x), float(y))
            for x, y in np.asarray(geometry.centerline, dtype=np.float32).tolist()
        ]
        self.track_tangents = [
            (float(tx), float(ty))
            for tx, ty in np.asarray(geometry.tangents, dtype=np.float32).tolist()
        ]
        self.track_mask = np.asarray(track["road_mask"], dtype=np.uint8)  # type: ignore[arg-type]
        self.collision_mask = np.asarray(track.get("collision_mask", track["road_mask"]), dtype=np.uint8)  # type: ignore[arg-type]
        self.track_texture = track["road_texture"] if self.show_game else None  # type: ignore[assignment]
        self.wall_texture = track["wall_texture"] if self.show_game else None  # type: ignore[assignment]

        self.track_points_np = np.asarray(self.track_centerline, dtype=np.float32)
        self.track_x_np = self.track_points_np[:, 0] if self.track_points_np.size else np.zeros((0,), dtype=np.float32)
        self.track_y_np = self.track_points_np[:, 1] if self.track_points_np.size else np.zeros((0,), dtype=np.float32)
        self.track_count = int(len(self.track_centerline))

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
        self._cache_track_draw_geometry()

        self.max_track_index_step = max(4, int(self.max_speed / 6.0 * 2.2))

    def _is_on_track(self, x: float, y: float) -> bool:
        track = self._require_track_geometry()
        proj = project_point_to_track(track, (float(x), float(y)))
        return abs(float(proj.lateral_offset)) <= float(track.half_width)

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

    def _update_off_track_state(self, car: RaceCar) -> None:
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
        track = self._require_track_geometry()
        ux, uy = self._normalize(float(dir_x), float(dir_y))
        max_dist = max(1.0, float(max_distance))
        # Match the raycast epsilon so edge contact reads as zero free space.
        origin_epsilon = 1e-4
        hit = raycast_track_edge(
            track,
            origin=(float(origin_x), float(origin_y)),
            direction=(float(ux), float(uy)),
            max_dist=float(max_dist),
        )
        if hit is None:
            return 1.0
        return float(clip_unit(float(max(0.0, hit - origin_epsilon)) / float(max_dist)))

    def _nearest_track_sample(self, x: float, y: float) -> tuple[int, float, float, float]:
        if self.track_count <= 0:
            return 0, 0.0, 0.0, 0.0
        track = self._require_track_geometry()
        proj = project_point_to_track(track, (float(x), float(y)))
        px, py = proj.point
        return (
            int(proj.seg_index % max(1, self.track_count)),
            float(proj.distance),
            float(x) - float(px),
            float(y) - float(py),
        )

    def _relative_track_progress(self, x: float, y: float) -> float:
        track = self._require_track_geometry()
        if self.track_count <= 0 or float(track.length) <= 1e-9:
            return 0.0
        proj = project_point_to_track(track, (float(x), float(y)))
        return float((float(proj.s) - float(track.start_s)) % float(track.length))

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
            car.lap_progress = float(progress)
            car.lap_armed = False
            car.off_track = False
            car.off_track_blend = 0.0
            cars.append(car)
        return cars

    def _project_to_car_frame(self, car: RaceCar) -> tuple[float, float, float, float]:
        heading_rad = math.radians(car.heading_degrees)
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        side_x = -forward_y
        side_y = forward_x
        forward_speed = car.vx * forward_x + car.vy * forward_y
        lateral_speed = car.vx * side_x + car.vy * side_y
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
        steer_speed_scale = 1.0 / (1.0 + float(self.steer_speed_decay) * speed_ratio * speed_ratio)
        steer_surface_scale = 0.45 + 0.55 * float(surface_grip)
        steer_authority = self._clamp(float(steer_speed_scale) * float(steer_surface_scale), 0.12, 1.0)
        heading_delta = float(steer) * float(self.turn_rate) * float(steer_authority)
        car.heading_degrees = self._normalize_degrees(float(car.heading_degrees) + heading_delta)
        car.yaw_rate = float(heading_delta)

        heading_rad = math.radians(car.heading_degrees)
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        side_x = -forward_y
        side_y = forward_x

        accel_scale = self.contact_accel_scale if car.in_contact else 1.0
        steer_load = self._clamp(abs(float(steer)), 0.0, 1.0)
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
        seam_window = max(float(track.half_width) * 0.9, 0.05 * lap_length)
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
            on_track_now = self._track_coverage_ratio(car.x, car.y) >= float(self.off_track_exit_ratio)
            crossed_arm_threshold_forward = prev_s < arm_threshold <= curr_s and is_forward
            if (not car.lap_armed) and on_track_now and crossed_arm_threshold_forward:
                car.lap_armed = True

            crossed_start_forward = (
                prev_s >= (lap_length - seam_window)
                and curr_s <= seam_window
                and is_forward
            )
            if crossed_start_forward and bool(car.lap_armed) and on_track_now:
                car.finished = True
                car.lap_armed = False
                car.lap_progress = lap_length
                if self.winner_index is None:
                    self.winner_index = idx
            else:
                car.lap_progress = float(curr_s)

    def _player_progress_potential(self) -> float:
        track = self._require_track_geometry()
        if not self.cars or float(track.length) <= 1e-9:
            return 0.0
        player = self.cars[self.player_index]
        return float(clip_unit(float(player.lap_progress) / float(max(1e-6, track.length))))

    def _setup_race(self) -> None:
        self.track_seed = random.randint(0, 2_000_000_000)
        self._generate_track(self.track_seed)
        self.cars = self._create_car_grid()
        for car in self.cars:
            car.ai_lane_offset = float(car.ai_lane_home)
        self.winner_index = None
        self.steps = 0
        self._prev_progress_potential = float(self._player_progress_potential())
        self._prev_player_in_contact = bool(self.cars[self.player_index].in_contact) if self.cars else False
        self._prev_player_forward_speed = 0.0

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
        if not self._can_toggle_visual_overlay():
            self._prev_overlay_toggle_down = False
            return
        toggle_down = bool(self.window_controller.is_key_down(arcade.key.X))
        if toggle_down and not self._prev_overlay_toggle_down:
            self.show_rays = not bool(self.show_rays)
        self._prev_overlay_toggle_down = bool(toggle_down)

    def _ai_lane_limit(self) -> float:
        return max(4.0, float(self.track_half_width) - float(self.track_probe_radius) - 1.0)

    def _next_main_corner(
        self,
        track: TrackGeometry,
        proj: TrackProjection,
    ) -> tuple[float, int | None]:
        if float(track.length) <= 1e-6 or not track.main_corner_s:
            return 0.0, None

        next_distance: float | None = None
        next_key: int | None = None
        for corner_key, corner_s in enumerate(track.main_corner_s):
            distance = (float(corner_s) - float(proj.s)) % float(track.length)
            if distance <= 1e-6:
                continue
            if next_distance is None or float(distance) < float(next_distance):
                next_distance = float(distance)
                next_key = int(corner_key)
        if next_distance is None:
            return 0.0, None
        return float(next_distance), next_key

    def _opponent_coast_error_percent(self, car: RaceCar, corner_key: int | None) -> float:
        if corner_key is None:
            car.ai_curve_key = -1
            car.ai_curve_error_percent = 0.0
            return 0.0
        if int(car.ai_curve_key) != int(corner_key):
            car.ai_curve_key = int(corner_key)
            car.ai_curve_error_percent = float(random.choice(self.opponent_coast_error_choices))
        return float(car.ai_curve_error_percent)

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

    def _side_template_factor(self, track: TrackGeometry, side_name: str) -> float:
        side_templates = dict(track.side_templates)
        template_name = str(side_templates.get(str(side_name), "straight"))
        if template_name == "bell":
            return 0.80
        if template_name == "s_curve":
            return 0.72
        return 1.0

    def _opponent_drive_plan(
        self,
        track: TrackGeometry,
        proj: TrackProjection,
        car: RaceCar,
        speed_ratio: float,
    ) -> tuple[float, float, bool]:
        del speed_ratio
        side_before_corner = ("top", "right", "bottom", "left")
        next_corner_distance, next_corner_key = self._next_main_corner(track, proj)
        if next_corner_key is None:
            return 1.0, 28.0, False

        current_side = str(side_before_corner[int(next_corner_key) % len(side_before_corner)])
        side_speed_factor = self._side_template_factor(track, current_side)

        ideal_corner_distance = max(72.0, 2.75 * float(self.track_half_width) + 18.0)
        coast_error_percent = self._opponent_coast_error_percent(car, next_corner_key)
        corner_entry_distance = self._clamp(
            float(ideal_corner_distance) * (1.0 - float(coast_error_percent) / 100.0),
            36.0,
            220.0,
        )

        previous_corner_key = (int(next_corner_key) - 1) % len(track.main_corner_s)
        previous_corner_s = float(track.main_corner_s[int(previous_corner_key)])
        distance_since_previous_corner = (float(proj.s) - float(previous_corner_s)) % float(track.length)
        corner_release_distance = max(20.0, 0.90 * float(self.track_half_width))

        in_corner_zone = bool(
            float(next_corner_distance) <= float(corner_entry_distance)
            or float(distance_since_previous_corner) <= float(corner_release_distance)
        )
        if in_corner_zone:
            return 0.5, 15.0, True
        if side_speed_factor < 1.0:
            return float(side_speed_factor), 22.0, False
        return 1.0, 30.0, False

    def _ai_control_for_car(self, car_index: int, car: RaceCar) -> tuple[float, float, float]:
        del car_index
        track = self._require_track_geometry()
        proj = project_point_to_track(track, (float(car.x), float(car.y)))
        _, _, forward_speed, _ = self._project_to_car_frame(car)
        max_forward_speed = max(1.0, float(self.max_speed) * float(self.opponent_speed_cap))
        signed_lateral = float(proj.lateral_offset)
        speed_ratio = self._clamp(abs(float(forward_speed)) / max(1.0, max_forward_speed), 0.0, 1.0)
        lane_target = self._opponent_lane_target(car, signed_lateral)
        target_speed_factor, base_look_ahead_s, in_corner_zone = self._opponent_drive_plan(
            track,
            proj,
            car,
            speed_ratio,
        )
        look_ahead_s = float(base_look_ahead_s) + 12.0 * float(speed_ratio)
        (target_x, target_y), _, (target_nx, target_ny) = sample_track_at_s(
            track,
            float(proj.s) + float(look_ahead_s),
        )
        aim_x = float(target_x) + target_nx * float(lane_target)
        aim_y = float(target_y) + target_ny * float(lane_target)
        desired_heading = math.degrees(math.atan2(aim_y - car.y, aim_x - car.x))
        delta = self._normalize_degrees(desired_heading - car.heading_degrees)

        if abs(delta) <= 1.5:
            steer = 0.0
        else:
            steer_gain = 12.0 if in_corner_zone else 14.0 if target_speed_factor < 1.0 else 16.0
            steer = self._clamp(float(delta) / float(steer_gain), -1.0, 1.0)

        target_speed = float(max_forward_speed) * float(target_speed_factor)
        if bool(car.in_contact) or bool(car.off_track):
            target_speed *= 0.85

        cruise_throttle = 0.18 + 0.30 * float(target_speed_factor)
        speed_gain = max(1.0, 0.32 * float(max_forward_speed))
        speed_error = float(target_speed) - float(forward_speed)
        throttle = self._clamp(float(cruise_throttle) + float(speed_error) / float(speed_gain), 0.0, 1.0)
        brake = 0.0

        overspeed_margin = max(0.15 * float(max_forward_speed), 0.25)
        if float(forward_speed) > float(target_speed) + float(overspeed_margin):
            throttle = 0.0
            brake = self._clamp(
                (float(forward_speed) - float(target_speed)) / max(0.5, 0.35 * float(max_forward_speed)),
                0.0,
                1.0,
            )

        if abs(float(delta)) >= 42.0:
            throttle = min(float(throttle), 0.55)
        if bool(car.in_contact) or bool(car.off_track):
            throttle = max(float(throttle), 0.30)
            brake = 0.0

        min_forward_speed = max(0.75, 0.28 * float(max_forward_speed))
        if float(forward_speed) < float(min_forward_speed):
            throttle = max(float(throttle), 0.55)
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

        # Player and opponents share the same off-road/physics path; only base speed cap differs.
        for idx, car in enumerate(self.cars):
            speed_multiplier = self._off_track_speed_multiplier(car)
            if idx == self.player_index:
                steer, throttle, brake = self._player_controls_from_action(action)
                allowed_speed = float(self.max_speed) * float(speed_multiplier)
            else:
                steer, throttle, brake = self._ai_control_for_car(idx, car)
                allowed_speed = float(self.max_speed) * float(self.opponent_speed_cap) * float(speed_multiplier)
            self._apply_car_controls(car, steer, throttle, brake, max_forward_speed=allowed_speed)

        for car in self.cars:
            car.x += car.vx
            car.y += car.vy

        self._resolve_car_contacts()
        for car in self.cars:
            self._resolve_screen_bounds(car)
        self._update_lap_progress_and_finish()

    def _edge_probe_values(
        self,
        car: RaceCar,
    ) -> tuple[float, float, float, float, float, list[tuple[float, float]], tuple[float, ...]]:
        heading_rad = math.radians(float(car.heading_degrees))
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        left_x = -forward_y
        left_y = forward_x

        def _norm(dx: float, dy: float) -> tuple[float, float]:
            return self._normalize(float(dx), float(dy))

        f_dir = (float(forward_x), float(forward_y))
        fl_dir = _norm(forward_x + 0.70 * left_x, forward_y + 0.70 * left_y)
        fr_dir = _norm(forward_x - 0.70 * left_x, forward_y - 0.70 * left_y)
        l_dir = (float(left_x), float(left_y))
        r_dir = (float(-left_x), float(-left_y))
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

    def _compute_obs(self) -> np.ndarray:
        track = self._require_track_geometry()
        player = self.cars[self.player_index]
        proj = project_point_to_track(track, (float(player.x), float(player.y)))
        tangent_x, tangent_y = float(proj.tangent[0]), float(proj.tangent[1])
        lat_off = clip_signed(float(proj.lateral_offset) / max(1e-6, float(track.half_width)))

        heading_rad = math.radians(player.heading_degrees)
        heading_x = math.cos(heading_rad)
        heading_y = math.sin(heading_rad)
        forward_x, forward_y, fwd_speed_raw, lat_speed_raw = self._project_to_car_frame(player)
        spd_fwd = clip_signed(float(fwd_speed_raw) / max(1.0, float(self.max_speed)))
        lat_vel = clip_signed(float(lat_speed_raw) / max(1.0, float(self.max_speed)))
        spd_delta = clip_signed(float(fwd_speed_raw - self._prev_player_forward_speed) / max(1.0, float(self.accel_force * 2.0)))
        yaw_rt = clip_signed(float(player.yaw_rate) / float(max(1e-6, self.yaw_rate_norm)))
        heading_err = float(self._signed_angle_norm(heading_x, heading_y, tangent_x, tangent_y))
        heading_err_rad = float(heading_err) * math.pi
        heading_err_sin = clip_signed(math.sin(heading_err_rad))
        heading_err_cos = clip_signed(math.cos(heading_err_rad))

        near_lookahead_s = max(18.0, min(70.0, 0.07 * float(track.length)))
        far_lookahead_s = max(near_lookahead_s + 24.0, min(190.0, 0.22 * float(track.length)))
        _, (near_tx, near_ty), _ = sample_track_at_s(track, float(proj.s) + float(near_lookahead_s))
        _, (far_tx, far_ty), _ = sample_track_at_s(track, float(proj.s) + float(far_lookahead_s))
        look_near = float(self._signed_angle_norm(heading_x, heading_y, near_tx, near_ty))
        look_far = float(self._signed_angle_norm(heading_x, heading_y, far_tx, far_ty))
        curve_near = float(self._signed_angle_norm(tangent_x, tangent_y, near_tx, near_ty))
        curve_far = float(self._signed_angle_norm(tangent_x, tangent_y, far_tx, far_ty))
        look_near_rad = float(look_near) * math.pi
        look_far_rad = float(look_far) * math.pi

        ray_f, edg_fl, edg_fr, edg_l, edg_r, probe_dirs, probe_max_distances = self._edge_probe_values(player)
        self._last_ray_origin = (float(player.x), float(player.y))
        self._last_ray_values = np.asarray(
            [ray_f, edg_fl, edg_fr, edg_l, edg_r],
            dtype=np.float32,
        )
        self._last_ray_dirs = [(float(dx), float(dy)) for dx, dy in probe_dirs]
        self._last_ray_max_distances = tuple(float(value) for value in probe_max_distances)
        self._prev_player_forward_speed = float(fwd_speed_raw)

        feature_values = {
            "self_lat_off": float(lat_off),
            "self_spd_lat": float(lat_vel),
            "self_spd_fwd": float(spd_fwd),
            "self_spd_delta": float(spd_delta),
            "self_yaw_rate": float(yaw_rt),
            "self_head_err_sin": float(heading_err_sin),
            "self_head_err_cos": float(heading_err_cos),
            "sens_look_near_sin": float(clip_signed(math.sin(look_near_rad))),
            "sens_look_near_cos": float(clip_signed(math.cos(look_near_rad))),
            "sens_look_far_sin": float(clip_signed(math.sin(look_far_rad))),
            "sens_look_far_cos": float(clip_signed(math.cos(look_far_rad))),
            "sens_curve_near": float(curve_near),
            "sens_curve_far": float(curve_far),
            "sens_fwd": float(ray_f),
            "sens_left_front": float(edg_fl),
            "sens_right_front": float(edg_fr),
            "sens_left": float(edg_l),
            "sens_right": float(edg_r),
            "flag_contact": 1.0 if bool(player.in_contact) else 0.0,
            "flag_off_track": 1.0 if bool(player.off_track) else 0.0,
        }
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

        phi_prev = float(self._prev_progress_potential)
        was_in_contact = bool(self._prev_player_in_contact)
        self._step_simulation(action_array)
        self.steps += 1

        reward = 0.0
        reward_breakdown = {
            "step.penalty_step": 0.0,
            "progress.shape": 0.0,
            "event.penalty_collision": 0.0,
            "outcome.reward_win": 0.0,
            "outcome.penalty_lose": 0.0,
        }
        episode_level = int(self._current_level)
        episode_success = 0
        if self.mode != "human":
            reward += float(PENALTY_STEP)
            reward_breakdown["step.penalty_step"] = float(PENALTY_STEP)

            player = self.cars[self.player_index]
            phi_next = float(self._player_progress_potential())
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
            self._prev_progress_potential = float(phi_next)

            collision_started = (not was_in_contact) and bool(player.in_contact)
            if collision_started:
                reward += float(PENALTY_COLLISION)
                reward_breakdown["event.penalty_collision"] = float(PENALTY_COLLISION)
            self._prev_player_in_contact = bool(player.in_contact)

        timed_out = bool((self.winner_index is None) and (self.steps >= self.max_steps))
        race_finished = bool((self.winner_index is not None) or timed_out)
        if race_finished:
            race_winner = int(self.winner_index) if self.winner_index is not None else None
            if self.mode != "human":
                player_won = race_winner == self.player_index
                if player_won:
                    reward += float(REWARD_WIN)
                    reward_breakdown["outcome.reward_win"] = float(REWARD_WIN)
                else:
                    reward += float(PENALTY_LOSE)
                    reward_breakdown["outcome.penalty_lose"] = float(PENALTY_LOSE)
                episode_success = 1 if player_won else 0
            self._finalize_race(race_winner)
        if self.mode != "human":
            self._episode_reward_components.add_from_mapping(reward_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)

        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

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

        edge_color = COLOR_FOG_GRAY
        edge_width = 3.0
        if len(self._track_left_outline_screen) > 2:
            arcade.draw_line_strip(self._track_left_outline_screen, edge_color, edge_width)
        if len(self._track_right_outline_screen) > 2:
            arcade.draw_line_strip(self._track_right_outline_screen, edge_color, edge_width)

    def _draw_rays(self) -> None:
        if not bool(self.show_rays and self.show_game):
            return
        origin_x, origin_y = self._last_ray_origin
        draw_player_rays(
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            ray_dirs=self._last_ray_dirs,
            ray_values=self._last_ray_values.tolist(),
            ray_max_distances=self._last_ray_max_distances,
            to_screen=lambda x, y: (float(x), float(self.window_controller.to_arcade_y(float(y)))),
            line_width=1.0,
        )

    def _draw_cars(self) -> None:
        inset = max(2.0, self.car_size * 0.22)
        for idx, car in enumerate(self.cars):
            draw_two_tone_tile(
                self.window_controller,
                top_left_x=car.x - self.car_half,
                top_left_y=car.y - self.car_half,
                size=self.car_size,
                outer_color=car.outer_color,
                inner_color=car.inner_color,
                inset=inset,
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
        self.window_controller.close()
