"""Top-down one-lap racing environment with mask-based procedural tracks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
import time

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_NEAR_BLACK,
    COLOR_P3_BLUE,
    COLOR_P3_NAVY,
    COLOR_P4_DEEP_PURPLE,
    COLOR_P4_PURPLE,
    COLOR_SLATE_GRAY,
    COLOR_SOFT_WHITE,
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
    normalize_last_action,
    normalized_ray_first_hit,
    ordered_feature_vector,
    signed_potential_shaping,
)
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_facing_indicator,
    draw_status_square_icon,
    draw_two_tone_tile,
    resolve_circle_collisions,
    status_icon_size,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level
from games.vroom.config import (
    ACTION_NAMES as VROOM_ACTION_NAMES,
    ACT_DIM as VROOM_ACT_DIM,
    BB_HEIGHT,
    CURRICULUM_PROMOTION,
    FPS,
    LEVEL_SETTINGS,
    INPUT_FEATURE_NAMES as VROOM_INPUT_FEATURE_NAMES,
    MAX_LEVEL,
    MIN_LEVEL,
    OBS_DIM as VROOM_OBS_DIM,
    PENALTY_COLLISION,
    PENALTY_LOSE,
    PENALTY_STEP,
    PROGRESS_CLIP,
    PROGRESS_SCALE,
    REWARD_WIN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TRAINING_FPS,
    WINDOW_TITLE,
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
    in_contact: bool = False
    obstacle_avoid_frames: int = 0
    obstacle_avoid_steer: float = 0.0
    obstacle_avoid_obs_idx: int = -1
    obstacle_avoid_coast_until_past: bool = False
    ai_lane_offset: float = 0.0
    track_index: int = 0
    lap_progress: float = 0.0
    finished: bool = False


class VroomEnv(Env):
    INPUT_FEATURE_NAMES = tuple(VROOM_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(VROOM_ACTION_NAMES)
    OBS_DIM = int(VROOM_OBS_DIM)
    ACT_DIM = int(VROOM_ACT_DIM)

    ACTION_COAST = 0
    ACTION_THROTTLE = 1
    ACTION_LEFT_COAST = 2
    ACTION_RIGHT_COAST = 3
    ACTION_LEFT_THROTTLE = 4
    ACTION_RIGHT_THROTTLE = 5

    NUM_CARS = 4
    TRAINING_TOTAL_RACES = 1
    HUMAN_TOTAL_RACES = 10
    WINNER_BAR_HISTORY_SIZE = 10
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
        self.lateral_grip = 0.76
        self.max_steps = 2_000

        self.car_contact_radius = self.car_radius
        self.wall_probe_radius = self.car_radius * 0.46
        self.wall_pushback = max(0.8, self.wall_probe_radius * 0.9)
        self.wall_bounce = 0.40
        self.wall_tangent_keep = 0.96
        diag = self.wall_probe_radius * 0.70710678
        self.wall_probe_offsets = (
            (0.0, 0.0),
            (self.wall_probe_radius, 0.0),
            (-self.wall_probe_radius, 0.0),
            (0.0, self.wall_probe_radius),
            (0.0, -self.wall_probe_radius),
            (diag, diag),
            (-diag, diag),
            (diag, -diag),
            (-diag, -diag),
        )
        self.contact_sep_strength = 1.0
        self.contact_overlap_cap = self.car_radius * 0.12
        self.contact_damp = 0.12
        self.contact_accel_scale = 0.85
        self.obstacle_contact_radius = self.car_radius * 0.85
        self.obstacle_bounce = 0.14
        self.obstacle_tangent_keep = 0.86
        self.obstacle_push_extra = 0.8
        self.obstacle_resolve_iters = 2

        self.track_config = TrackGenConfig(
            track_width_px=88.0,
            wall_thickness_px=4.0,
            padding_px=40.0,
            corner_radius_px=130.0,
            line_safe_margin_px=96.0,
        )
        self.track_width_px = float(self.track_config.track_width_px)
        self.track_half_width = self.track_width_px * 0.5

        self.track_centerline: list[tuple[float, float]] = []
        self.track_tangents: list[tuple[float, float]] = []
        self.track_mask = np.zeros((int(self.track_bottom), SCREEN_WIDTH), dtype=np.uint8)
        self.collision_mask = np.zeros((int(self.track_bottom), SCREEN_WIDTH), dtype=np.uint8)
        self.wall_mask = np.zeros((int(self.track_bottom), SCREEN_WIDTH), dtype=np.uint8)
        self.track_points_np = np.zeros((0, 2), dtype=np.float32)
        self.track_x_np = np.zeros((0,), dtype=np.float32)
        self.track_y_np = np.zeros((0,), dtype=np.float32)
        self.track_texture: arcade.Texture | None = None
        self.wall_texture: arcade.Texture | None = None
        self.track_rect = arcade.LRBT(0.0, float(SCREEN_WIDTH), float(BB_HEIGHT), float(SCREEN_HEIGHT))
        self.obstacles: list[tuple[float, float]] = []
        self.obstacle_size = float(TILE_SIZE)
        self.obstacle_inset = 4.0
        self.ai_obstacle_avoid_chance = 0.75
        self.ai_avoid_lookahead = self.obstacle_size * 4.6
        self.ai_avoid_lateral = max(self.obstacle_size * 1.8, self.track_half_width * 0.95)
        self.ai_avoid_close_distance = self.obstacle_size * 1.75
        self.ai_avoid_frames = 14
        self.ai_obstacle_rolls: dict[tuple[int, int], bool] = {}

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
            (COLOR_P3_BLUE, COLOR_P3_NAVY),
            (COLOR_P4_PURPLE, COLOR_P4_DEEP_PURPLE),
        ]
        self.player_index = 0
        self.winner_index: int | None = None
        self.last_race_winner: int | None = None
        self.total_races = int(self._resolve_total_races())
        self.match_tracker = MatchTracker[int](match_limit=int(self.total_races))
        self.winner_bar_tracker = MatchTracker[int](history_limit=int(self.WINNER_BAR_HISTORY_SIZE))
        self.current_race = 1
        self.win_history: list[int | None] = self.match_tracker.history
        self.winner_bar_history: list[int | None] = self.winner_bar_tracker.history
        self.num_cars = int(self.NUM_CARS)
        self.opponent_speed_cap = 1.0
        self.track_obstacle_clusters = int(self.track_config.obstacle_clusters)
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
        self.last_action_index = self.ACTION_COAST
        self._prev_self_lat_offset = 0.0
        self._prev_self_fwd_speed = 0.0
        self._ray_near_range = max(1.0, self.obstacle_size * 3.2)
        self._ray_far_range = max(self._ray_near_range + 1.0, self.obstacle_size * 6.0)
        self._ray_step_size = max(0.75, self.obstacle_size * 0.2)
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._force_zero_deltas = False
        self._prev_progress_potential = 0.0
        self._prev_player_in_contact = False
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)
        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _resolve_total_races(self) -> int:
        if self.mode == "human":
            return int(self.HUMAN_TOTAL_RACES)
        return int(self.TRAINING_TOTAL_RACES)

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level), LEVEL_SETTINGS[int(MIN_LEVEL)])
        self.num_cars = max(1, min(int(settings["num_cars"]), len(self.player_color_pairs)))
        self.opponent_speed_cap = self._clamp(float(settings["opponent_speed_cap"]), 0.0, 1.0)
        self.track_obstacle_clusters = max(0, int(settings["obstacle_clusters"]))
        self.ai_obstacle_avoid_chance = self._clamp(float(settings.get("opponent_obstacle_avoid_chance", 0.75)), 0.0, 1.0)

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

    def _generate_track(self, seed: int) -> None:
        track_config = replace(self.track_config, obstacle_clusters=int(self.track_obstacle_clusters))
        track = generate_track(
            seed=int(seed),
            width=int(SCREEN_WIDTH),
            height=int(self.track_bottom),
            config=track_config,
            build_texture=self.show_game,
        )
        self.track_centerline = list(track["centerline"])  # type: ignore[arg-type]
        self.track_mask = np.asarray(track["road_mask"], dtype=np.uint8)  # type: ignore[arg-type]
        self.collision_mask = np.asarray(track.get("collision_mask", track["road_mask"]), dtype=np.uint8)  # type: ignore[arg-type]
        self.wall_mask = np.asarray(track["wall_mask"], dtype=np.uint8)  # type: ignore[arg-type]
        self.obstacles = [(float(item[0]), float(item[1])) for item in track.get("obstacles", [])]  # type: ignore[arg-type]
        self.track_texture = track["road_texture"] if self.show_game else None  # type: ignore[assignment]
        self.wall_texture = track["wall_texture"] if self.show_game else None  # type: ignore[assignment]

        self.track_points_np = np.asarray(self.track_centerline, dtype=np.float32)
        self.track_x_np = self.track_points_np[:, 0] if self.track_points_np.size else np.zeros((0,), dtype=np.float32)
        self.track_y_np = self.track_points_np[:, 1] if self.track_points_np.size else np.zeros((0,), dtype=np.float32)
        self.track_count = int(len(self.track_centerline))

        default_start_idx = int(np.argmax(self.track_y_np)) if self.track_y_np.size else 0
        start_idx = int(track.get("start_index", default_start_idx))
        start_idx = int(start_idx % max(1, self.track_count))
        self.start_index = int(start_idx)
        self.start_x = float(self.track_centerline[start_idx][0])
        self.start_y = float(self.track_centerline[start_idx][1])
        self.start_side = str(track.get("start_side", "top"))
        prev_idx = (start_idx - 1) % self.track_count
        next_idx = (start_idx + 1) % self.track_count
        tx, ty = self._normalize(
            float(self.track_centerline[next_idx][0] - self.track_centerline[prev_idx][0]),
            float(self.track_centerline[next_idx][1] - self.track_centerline[prev_idx][1]),
        )
        self.start_tangent = (tx, ty)
        self.start_normal = (-ty, tx)
        line_obj = track.get("start_line")
        if isinstance(line_obj, (list, tuple)) and len(line_obj) == 2:
            p1 = line_obj[0]
            p2 = line_obj[1]
            self.start_line = ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        else:
            nx, ny = self.start_normal
            half_w = 0.5 * self.track_width_px
            self.start_line = (
                (self.start_x + nx * half_w, self.start_y + ny * half_w),
                (self.start_x - nx * half_w, self.start_y - ny * half_w),
            )

        self.track_tangents = []
        for i in range(self.track_count):
            x_prev, y_prev = self.track_centerline[(i - 1) % self.track_count]
            x_next, y_next = self.track_centerline[(i + 1) % self.track_count]
            self.track_tangents.append(self._normalize(x_next - x_prev, y_next - y_prev))

        self.max_track_index_step = max(4, int(self.max_speed / 6.0 * 2.2))

    def _is_on_track(self, x: float, y: float) -> bool:
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        return (
            0 <= iy < int(self.track_mask.shape[0])
            and 0 <= ix < int(self.track_mask.shape[1])
            and self.track_mask[iy, ix] != 0
        )

    def _is_wall(self, x: float, y: float) -> bool:
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        return (
            0 <= iy < int(self.wall_mask.shape[0])
            and 0 <= ix < int(self.wall_mask.shape[1])
            and self.wall_mask[iy, ix] != 0
        )

    def _is_driveable_footprint(self, x: float, y: float) -> bool:
        return self._is_on_track_footprint(x, y) and (not self._hits_wall_footprint(x, y))

    def _is_on_track_footprint(self, x: float, y: float) -> bool:
        for ox, oy in self.wall_probe_offsets:
            px = float(x) + ox
            py = float(y) + oy
            if not self._is_on_track(px, py):
                return False
        return True

    def _hits_wall_footprint(self, x: float, y: float) -> bool:
        for ox, oy in self.wall_probe_offsets:
            if self._is_wall(float(x) + ox, float(y) + oy):
                return True
        return False

    def _enforce_track_containment(self, car: RaceCar) -> None:
        if self._is_driveable_footprint(car.x, car.y):
            return

        idx, _, dx, dy = self._nearest_track_sample(car.x, car.y)
        cx, cy = self.track_centerline[idx]
        tx, ty = self.track_tangents[idx]
        nx, ny = -ty, tx

        signed_lateral = dx * nx + dy * ny
        allowed = max(4.0, self.track_half_width - self.wall_probe_radius - 1.5)
        clamped_lateral = self._clamp(signed_lateral, -allowed, allowed)
        candidate_x = cx + nx * clamped_lateral
        candidate_y = cy + ny * clamped_lateral

        if not self._is_driveable_footprint(candidate_x, candidate_y):
            candidate_x, candidate_y = cx, cy
        if not self._is_driveable_footprint(candidate_x, candidate_y):
            for jump in (1, 2, 3, 5, 8, 12):
                for direction in (-1, 1):
                    probe_idx = (idx + direction * jump) % max(1, self.track_count)
                    px, py = self.track_centerline[probe_idx]
                    if self._is_driveable_footprint(px, py):
                        candidate_x, candidate_y = px, py
                        break
                if self._is_driveable_footprint(candidate_x, candidate_y):
                    break

        vn = car.vx * nx + car.vy * ny
        vt = car.vx * tx + car.vy * ty
        if abs(signed_lateral) >= allowed and vn * math.copysign(1.0, signed_lateral) > 0.0:
            vn = -abs(vn) * self.wall_bounce
        else:
            vn *= 0.2
        vt *= self.wall_tangent_keep

        car.x = float(candidate_x)
        car.y = float(candidate_y)
        car.vx = tx * vt + nx * vn
        car.vy = ty * vt + ny * vn
        car.in_contact = True
        if not self._is_driveable_footprint(car.x, car.y):
            car.x, car.y = self.track_centerline[idx]
            car.vx = 0.0
            car.vy = 0.0
            car.in_contact = True

    def _track_normal_from_point(self, x: float, y: float) -> tuple[float, float]:
        idx, _, dx, dy = self._nearest_track_sample(x, y)
        if abs(dx) + abs(dy) > 1e-6:
            return self._normalize(dx, dy)
        tx, ty = self.track_tangents[idx]
        return -ty, tx

    def _push_inside_track(self, x: float, y: float, nx: float, ny: float) -> tuple[float, float, bool]:
        px = float(x)
        py = float(y)
        for i in range(8):
            if self._is_driveable_footprint(px, py):
                return px, py, True
            step = self.wall_pushback * (0.8 + 0.22 * float(i))
            px -= nx * step
            py -= ny * step
        return px, py, self._is_driveable_footprint(px, py)

    def _nearest_track_sample(self, x: float, y: float) -> tuple[int, float, float, float]:
        if self.track_x_np.size == 0:
            return 0, 0.0, 0.0, 0.0
        dxs = self.track_x_np - float(x)
        dys = self.track_y_np - float(y)
        sq_distances = dxs * dxs + dys * dys
        idx = int(np.argmin(sq_distances))
        px = float(self.track_x_np[idx])
        py = float(self.track_y_np[idx])
        return idx, float(math.sqrt(float(sq_distances[idx]))), float(x - px), float(y - py)

    def _create_car_grid(self) -> list[RaceCar]:
        index_gap = max(6, int(self.car_size * 0.30))

        cars: list[RaceCar] = []
        car_count = max(1, int(self.num_cars))
        if car_count == 1:
            lane_offsets = [0.0]
        else:
            max_lane_span = max(0.0, self.track_half_width - self.car_half * 0.65)
            lane_span = min(max_lane_span, max(self.car_size * 0.9, max_lane_span * 0.78))
            lane_offsets = np.linspace(-lane_span, lane_span, num=car_count, dtype=np.float32).astype(float).tolist()
            random.shuffle(lane_offsets)

        base_idx = (self.start_index - index_gap) % max(1, self.track_count)
        max_jump = max(1, min(self.track_count // 4, 70))
        spacing_scales = (1.0, 0.92, 0.84, 0.76, 0.68, 0.60, 0.52)

        spawn_idx = base_idx
        resolved_offsets = [float(offset) for offset in lane_offsets]
        found_layout = False
        for scale in spacing_scales:
            offsets = [float(offset) * float(scale) for offset in lane_offsets]
            for jump in range(max_jump + 1):
                directions = (1,) if jump == 0 else (1, -1)
                for direction in directions:
                    probe_idx = (base_idx + direction * jump) % max(1, self.track_count)
                    cx, cy = self.track_centerline[probe_idx]
                    tx, ty = self.track_tangents[probe_idx]
                    nx, ny = -ty, tx
                    valid = True
                    for lateral in offsets:
                        px = cx + nx * lateral
                        py = cy + ny * lateral
                        if not self._is_driveable_footprint(px, py):
                            valid = False
                            break
                    if valid:
                        spawn_idx = int(probe_idx)
                        resolved_offsets = offsets
                        found_layout = True
                        break
                if found_layout:
                    break
            if found_layout:
                break

        cx, cy = self.track_centerline[spawn_idx]
        tx, ty = self.track_tangents[spawn_idx]
        nx, ny = -ty, tx
        heading = math.degrees(math.atan2(ty, tx))
        for idx in range(int(self.num_cars)):
            desired_lateral = resolved_offsets[idx % len(resolved_offsets)]
            x = cx + nx * desired_lateral
            y = cy + ny * desired_lateral
            outer, inner = self.player_color_pairs[idx % len(self.player_color_pairs)]
            car = RaceCar(
                x=float(x),
                y=float(y),
                vx=0.0,
                vy=0.0,
                heading_degrees=float(heading),
                outer_color=outer,
                inner_color=inner,
                ai_lane_offset=float(desired_lateral),
            )
            if not self._is_driveable_footprint(car.x, car.y):
                car.x = float(cx)
                car.y = float(cy)
                car.ai_lane_offset = 0.0
            car.track_index = self._nearest_track_sample(car.x, car.y)[0]
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
        *,
        max_forward_speed: float | None = None,
    ) -> None:
        _, _, forward_speed, lateral_speed = self._project_to_car_frame(car)
        speed_ratio = min(1.0, abs(forward_speed) / max(1e-6, self.max_speed))
        throttle_load = max(0.0, float(throttle))
        # Steering authority drops as speed/load rise, making coasting important in bends.
        steer_authority = self._clamp(
            1.0 - 0.45 * speed_ratio - 0.20 * speed_ratio * throttle_load,
            0.35,
            1.0,
        )
        car.heading_degrees += float(steer) * self.turn_rate * steer_authority

        heading_rad = math.radians(car.heading_degrees)
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        side_x = -forward_y
        side_y = forward_x

        accel_scale = self.contact_accel_scale if car.in_contact else 1.0
        forward_speed += float(throttle) * self.accel_force * accel_scale
        if throttle < 0.0:
            forward_speed += float(throttle) * self.brake_force * accel_scale
        allowed_forward_speed = float(self.max_speed if max_forward_speed is None else max_forward_speed)
        forward_speed = self._clamp(forward_speed, -self.max_reverse_speed, allowed_forward_speed)
        lateral_speed *= self.lateral_grip

        car.vx = forward_x * forward_speed + side_x * lateral_speed
        car.vy = forward_y * forward_speed + side_y * lateral_speed
        car.vx *= self.drag
        car.vy *= self.drag

    def _resolve_obstacle_contacts(self, car: RaceCar, prev_x: float, prev_y: float, *, car_index: int) -> None:
        if not self.obstacles:
            return

        radius = max(1.0, float(self.obstacle_contact_radius))
        radius_sq = radius * radius
        push_margin = max(0.4, float(self.obstacle_push_extra))
        tile_size = float(self.obstacle_size)

        for _ in range(max(1, int(self.obstacle_resolve_iters))):
            hit_any = False
            for ox, oy in self.obstacles:
                left = float(ox)
                top = float(oy)
                right = left + tile_size
                bottom = top + tile_size

                nearest_x = self._clamp(car.x, left, right)
                nearest_y = self._clamp(car.y, top, bottom)
                dx = float(car.x) - nearest_x
                dy = float(car.y) - nearest_y
                dist_sq = dx * dx + dy * dy
                if dist_sq >= radius_sq:
                    continue

                hit_any = True
                if dist_sq > 1e-8:
                    dist = math.sqrt(dist_sq)
                    nx, ny = dx / dist, dy / dist
                    penetration = radius - dist
                else:
                    move_x = float(car.x) - float(prev_x)
                    move_y = float(car.y) - float(prev_y)
                    if abs(move_x) + abs(move_y) > 1e-6:
                        nx, ny = self._normalize(move_x, move_y)
                        nearest_side = 0.0
                    else:
                        d_left = abs(float(car.x) - left)
                        d_right = abs(right - float(car.x))
                        d_top = abs(float(car.y) - top)
                        d_bottom = abs(bottom - float(car.y))
                        nearest_side = min(d_left, d_right, d_top, d_bottom)
                        if nearest_side == d_left:
                            nx, ny = -1.0, 0.0
                        elif nearest_side == d_right:
                            nx, ny = 1.0, 0.0
                        elif nearest_side == d_top:
                            nx, ny = 0.0, -1.0
                        else:
                            nx, ny = 0.0, 1.0
                    penetration = radius + max(0.0, float(nearest_side))

                car.x += nx * (penetration + push_margin)
                car.y += ny * (penetration + push_margin)

                tx, ty = -ny, nx
                vn = car.vx * nx + car.vy * ny
                vt = car.vx * tx + car.vy * ty
                if vn < 0.0:
                    vn = -vn * self.obstacle_bounce
                else:
                    vn *= 0.25
                vt *= self.obstacle_tangent_keep
                car.vx = tx * vt + nx * vn
                car.vy = ty * vt + ny * vn
                car.in_contact = True

                if int(car_index) != self.player_index:
                    heading_rad = math.radians(float(car.heading_degrees))
                    left_x = -math.sin(heading_rad)
                    left_y = math.cos(heading_rad)
                    obstacle_center_x = left + tile_size * 0.5
                    obstacle_center_y = top + tile_size * 0.5
                    rel_x = obstacle_center_x - float(car.x)
                    rel_y = obstacle_center_y - float(car.y)
                    lateral = rel_x * left_x + rel_y * left_y
                    car.obstacle_avoid_steer = -1.0 if lateral >= 0.0 else 1.0
                    car.obstacle_avoid_frames = max(int(car.obstacle_avoid_frames), int(self.ai_avoid_frames))
            if not hit_any:
                break

    def _resolve_track_contacts(self, car: RaceCar, prev_x: float, prev_y: float) -> None:
        curr_x, curr_y = float(car.x), float(car.y)
        prev_ok = self._is_on_track_footprint(prev_x, prev_y)
        curr_ok = self._is_on_track_footprint(curr_x, curr_y)
        if prev_ok and curr_ok:
            return

        if not prev_ok:
            nearest_idx = self._nearest_track_sample(prev_x, prev_y)[0]
            prev_x, prev_y = self.track_centerline[nearest_idx]
            prev_ok = self._is_on_track_footprint(prev_x, prev_y)
            if not prev_ok:
                prev_x, prev_y = self.track_centerline[self._nearest_track_sample(curr_x, curr_y)[0]]

        if curr_ok:
            hit_x, hit_y = curr_x, curr_y
            nx, ny = self._track_normal_from_point(hit_x, hit_y)
        else:
            t_good = 0.0
            t_bad = 1.0
            for _ in range(10):
                t_mid = 0.5 * (t_good + t_bad)
                sample_x = prev_x + (curr_x - prev_x) * t_mid
                sample_y = prev_y + (curr_y - prev_y) * t_mid
                if self._is_on_track_footprint(sample_x, sample_y):
                    t_good = t_mid
                else:
                    t_bad = t_mid
            hit_x = prev_x + (curr_x - prev_x) * t_good
            hit_y = prev_y + (curr_y - prev_y) * t_good
            nx, ny = self._track_normal_from_point(curr_x, curr_y)

        tangent_x, tangent_y = -ny, nx
        vn = car.vx * nx + car.vy * ny
        vt = car.vx * tangent_x + car.vy * tangent_y
        delta_tangent = (curr_x - prev_x) * tangent_x + (curr_y - prev_y) * tangent_y

        if vn > 0.0:
            vn = -vn * self.wall_bounce
        else:
            vn *= 0.28
        if abs(vt) < 0.10 and abs(delta_tangent) > 0.05:
            vt = delta_tangent * 0.9
        vt *= self.wall_tangent_keep

        candidate_x = hit_x - nx * self.wall_pushback
        candidate_y = hit_y - ny * self.wall_pushback
        candidate_x, candidate_y, ok = self._push_inside_track(candidate_x, candidate_y, nx, ny)
        if not ok:
            nearest_idx = self._nearest_track_sample(candidate_x, candidate_y)[0]
            candidate_x, candidate_y = self.track_centerline[nearest_idx]

        car.x = float(candidate_x)
        car.y = float(candidate_y)
        car.vx = tangent_x * vt + nx * vn
        car.vy = tangent_y * vt + ny * vn
        car.in_contact = True

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
        if self.track_count <= 0:
            return
        half_track = self.track_count // 2
        for idx, car in enumerate(self.cars):
            if car.finished:
                continue

            new_index = self._nearest_track_sample(car.x, car.y)[0]
            delta = new_index - car.track_index
            if delta > half_track:
                delta -= self.track_count
            elif delta < -half_track:
                delta += self.track_count
            delta = max(-self.max_track_index_step, min(self.max_track_index_step, int(delta)))

            car.lap_progress = max(0.0, car.lap_progress + float(delta))
            car.track_index = int(new_index)
            if car.lap_progress >= float(self.track_count):
                car.finished = True
                if self.winner_index is None:
                    self.winner_index = idx

    def _player_progress_potential(self) -> float:
        if not self.cars or self.track_count <= 0:
            return 0.0
        player = self.cars[self.player_index]
        return float(clip_unit(float(player.lap_progress) / float(max(1, self.track_count))))

    def _setup_race(self) -> None:
        self.track_seed = random.randint(0, 2_000_000_000)
        self._generate_track(self.track_seed)
        self._reset_ai_obstacle_rolls()
        self.cars = self._create_car_grid()
        self.winner_index = None
        self.steps = 0
        self._force_zero_deltas = True
        self._prev_progress_potential = float(self._player_progress_potential())
        self._prev_player_in_contact = bool(self.cars[self.player_index].in_contact) if self.cars else False

    def _finalize_race(self, winner_idx: int | None) -> None:
        self.last_race_winner = None if winner_idx is None else int(winner_idx)
        self.match_tracker.record_result(self.last_race_winner)
        self.winner_bar_tracker.record_result(self.last_race_winner)
        if self.match_tracker.match_limit_reached():
            self.done = True
            self.winner_index = self.last_race_winner
            return
        self.current_race = int(self.match_tracker.matches_played()) + 1
        self._setup_race()

    def _resolve_human_action(self) -> int:
        left = self.window_controller.is_key_down(arcade.key.LEFT) or self.window_controller.is_key_down(arcade.key.A)
        right = self.window_controller.is_key_down(arcade.key.RIGHT) or self.window_controller.is_key_down(arcade.key.D)
        throttle = self.window_controller.is_key_down(arcade.key.UP) or self.window_controller.is_key_down(arcade.key.W)
        if left and (not right):
            return self.ACTION_LEFT_THROTTLE if throttle else self.ACTION_LEFT_COAST
        if right and (not left):
            return self.ACTION_RIGHT_THROTTLE if throttle else self.ACTION_RIGHT_COAST
        if throttle:
            return self.ACTION_THROTTLE
        return self.ACTION_COAST

    def _reset_ai_obstacle_rolls(self) -> None:
        self.ai_obstacle_rolls = {}

    def _closest_obstacle_ahead(self, car: RaceCar) -> tuple[int, float, float] | None:
        if not self.obstacles:
            return None

        idx, _, _, _ = self._nearest_track_sample(float(car.x), float(car.y))
        forward_x, forward_y = self.track_tangents[idx]
        left_x = -forward_y
        left_y = forward_x

        best: tuple[int, float, float] | None = None
        for obs_idx, (ox, oy) in enumerate(self.obstacles):
            cx = float(ox) + 0.5 * float(self.obstacle_size)
            cy = float(oy) + 0.5 * float(self.obstacle_size)
            rel_x = cx - float(car.x)
            rel_y = cy - float(car.y)
            forward = rel_x * forward_x + rel_y * forward_y
            if forward <= 0.0 or forward > float(self.ai_avoid_lookahead):
                continue
            lateral = rel_x * left_x + rel_y * left_y
            if abs(lateral) > float(self.ai_avoid_lateral):
                continue
            if best is None or forward < best[1]:
                best = (int(obs_idx), float(forward), float(lateral))
        return best

    def _obstacle_center(self, obs_idx: int) -> tuple[float, float]:
        ox, oy = self.obstacles[int(obs_idx)]
        return (
            float(ox) + 0.5 * float(self.obstacle_size),
            float(oy) + 0.5 * float(self.obstacle_size),
        )

    def _obstacle_clearance_distance(self) -> float:
        return 0.5 * float(self.obstacle_size) + 1.05 * float(self.car_radius)

    def _is_lane_danger_for_obstacle(self, obstacle_forward: float, obstacle_lateral: float) -> bool:
        if float(obstacle_forward) <= 0.0:
            return False
        return abs(float(obstacle_lateral)) <= float(self._obstacle_clearance_distance())

    def _is_obstacle_side_safe(self, obs_idx: int, side_sign: float) -> bool:
        obstacle_center_x, obstacle_center_y = self._obstacle_center(int(obs_idx))
        track_idx, _, _, _ = self._nearest_track_sample(float(obstacle_center_x), float(obstacle_center_y))
        forward_x, forward_y = self.track_tangents[track_idx]
        left_x = -forward_y
        left_y = forward_x
        pass_distance = float(self._obstacle_clearance_distance())
        side_x = left_x * float(side_sign)
        side_y = left_y * float(side_sign)
        pass_x = float(obstacle_center_x) + side_x * pass_distance
        pass_y = float(obstacle_center_y) + side_y * pass_distance
        forward_probe = 0.70 * float(self.car_radius)
        return (
            self._is_driveable_footprint(pass_x, pass_y)
            and self._is_driveable_footprint(pass_x + forward_x * forward_probe, pass_y + forward_y * forward_probe)
            and self._is_driveable_footprint(pass_x - forward_x * forward_probe, pass_y - forward_y * forward_probe)
        )

    def _closest_safe_obstacle_avoid_steer(self, obs_idx: int, obstacle_lateral: float) -> float:
        clearance = float(self._obstacle_clearance_distance())
        preferred = -1.0 if float(obstacle_lateral) >= 0.0 else 1.0
        candidates: list[tuple[float, float, int]] = []

        if self._is_obstacle_side_safe(int(obs_idx), 1.0):
            left_shift = abs(float(obstacle_lateral) + clearance)
            left_tiebreak = 0 if preferred == 1.0 else 1
            candidates.append((float(left_shift), 1.0, int(left_tiebreak)))
        if self._is_obstacle_side_safe(int(obs_idx), -1.0):
            right_shift = abs(float(obstacle_lateral) - clearance)
            right_tiebreak = 0 if preferred == -1.0 else 1
            candidates.append((float(right_shift), -1.0, int(right_tiebreak)))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[2]))
            return float(candidates[0][1])
        return float(preferred)

    @staticmethod
    def _should_keep_coasting_for_obstacle(
        car: RaceCar,
        ahead: tuple[int, float, float] | None,
    ) -> bool:
        if not bool(car.obstacle_avoid_coast_until_past):
            return False
        if int(car.obstacle_avoid_obs_idx) < 0:
            return False
        if ahead is None:
            return False
        obs_idx, obstacle_forward, _ = ahead
        if int(obs_idx) != int(car.obstacle_avoid_obs_idx):
            return False
        return float(obstacle_forward) > 0.0

    def _ai_control_for_car(self, car_index: int, car: RaceCar) -> tuple[float, float]:
        idx, distance, dx, dy = self._nearest_track_sample(car.x, car.y)
        _, _, forward_speed, _ = self._project_to_car_frame(car)
        max_forward_speed = float(self.max_speed) * float(self.opponent_speed_cap)
        min_forward_speed = 0.5 * max_forward_speed

        tx, ty = self.track_tangents[idx]
        nx, ny = -ty, tx
        signed_lateral = dx * nx + dy * ny
        lane_target = float(car.ai_lane_offset)
        lane_error = signed_lateral - lane_target
        correction_mag = self._clamp(lane_error * 0.65, -self.track_half_width * 0.65, self.track_half_width * 0.65)
        speed_ratio = self._clamp(abs(float(forward_speed)) / max(1.0, max_forward_speed), 0.0, 1.0)
        look_ahead = 8 + int(8.0 * speed_ratio + min(8.0, distance / max(1.0, self.track_half_width) * 6.0))
        target_idx = (idx + look_ahead) % self.track_count
        target_x, target_y = self.track_centerline[target_idx]
        target_tx, target_ty = self.track_tangents[target_idx]
        target_nx, target_ny = -target_ty, target_tx
        target_lateral = lane_target - correction_mag
        aim_x = float(target_x) + target_nx * float(target_lateral)
        aim_y = float(target_y) + target_ny * float(target_lateral)
        desired_heading = math.degrees(math.atan2(aim_y - car.y, aim_x - car.x))
        delta = self._normalize_degrees(desired_heading - car.heading_degrees)

        if abs(delta) <= 2.0:
            steer = 0.0
        else:
            steer = self._clamp(float(delta) / 18.0, -1.0, 1.0)

        throttle = 1.0
        abs_delta = abs(float(delta))
        if abs_delta > 55.0:
            throttle = 0.0
        elif abs_delta > 35.0:
            throttle = 0.20

        # Launch behavior: keep opponents from immediately dropping pace on opening straights.
        if self.steps < 90 and abs_delta < 35.0:
            launch_floor = float(max_forward_speed) * 0.78
            if forward_speed < launch_floor:
                throttle = 1.0

        ahead = self._closest_obstacle_ahead(car)
        keep_coasting = self._should_keep_coasting_for_obstacle(car, ahead)
        if keep_coasting:
            car.obstacle_avoid_frames = max(1, int(car.obstacle_avoid_frames) - 1)
            return float(car.obstacle_avoid_steer), 0.0
        if bool(car.obstacle_avoid_coast_until_past):
            car.obstacle_avoid_coast_until_past = False
            car.obstacle_avoid_obs_idx = -1

        if int(car.obstacle_avoid_frames) > 0:
            car.obstacle_avoid_frames = int(car.obstacle_avoid_frames) - 1
            steer = float(car.obstacle_avoid_steer)
            throttle = min(throttle, 0.20)
            if forward_speed < float(min_forward_speed):
                throttle = 1.0
            return steer, throttle

        if ahead is not None:
            obs_idx, forward_to_obstacle, lateral = ahead
            if self._is_lane_danger_for_obstacle(float(forward_to_obstacle), float(lateral)):
                roll_key = (int(car_index), int(obs_idx))
                should_avoid = self.ai_obstacle_rolls.get(roll_key)
                if should_avoid is None:
                    should_avoid = bool(random.random() < float(self.ai_obstacle_avoid_chance))
                    self.ai_obstacle_rolls[roll_key] = bool(should_avoid)
                if should_avoid:
                    steer = self._closest_safe_obstacle_avoid_steer(int(obs_idx), float(lateral))
                    close_now = float(forward_to_obstacle) <= float(self.ai_avoid_close_distance)
                    car.obstacle_avoid_coast_until_past = bool(close_now)
                    car.obstacle_avoid_obs_idx = int(obs_idx)
                    throttle = 0.0 if close_now else min(throttle, 0.20)
                    if (not close_now) and forward_speed < float(min_forward_speed):
                        throttle = 1.0
                    car.obstacle_avoid_steer = float(steer)
                    car.obstacle_avoid_frames = int(self.ai_avoid_frames)
                    return steer, throttle
        if forward_speed < float(min_forward_speed):
            throttle = 1.0
        return steer, throttle

    def _player_controls_from_action(self, action_idx: int) -> tuple[float, float]:
        if action_idx == self.ACTION_THROTTLE:
            return 0.0, 1.0
        if action_idx == self.ACTION_LEFT_COAST:
            return -1.0, 0.0
        if action_idx == self.ACTION_RIGHT_COAST:
            return 1.0, 0.0
        if action_idx == self.ACTION_LEFT_THROTTLE:
            return -1.0, 1.0
        if action_idx == self.ACTION_RIGHT_THROTTLE:
            return 1.0, 1.0
        return 0.0, 0.0

    def _step_simulation(self, action_idx: int) -> None:
        previous_positions = [(car.x, car.y) for car in self.cars]
        for idx, car in enumerate(self.cars):
            if idx == self.player_index:
                steer, throttle = self._player_controls_from_action(action_idx)
                allowed_speed = float(self.max_speed)
            else:
                steer, throttle = self._ai_control_for_car(idx, car)
                allowed_speed = float(self.max_speed) * float(self.opponent_speed_cap)
            self._apply_car_controls(car, steer, throttle, max_forward_speed=allowed_speed)

        for car in self.cars:
            car.x += car.vx
            car.y += car.vy

        self._resolve_car_contacts()
        for idx, car in enumerate(self.cars):
            prev_x, prev_y = previous_positions[idx]
            self._resolve_obstacle_contacts(car, prev_x=float(prev_x), prev_y=float(prev_y), car_index=idx)
            self._resolve_track_contacts(car, prev_x=float(prev_x), prev_y=float(prev_y))
            self._resolve_screen_bounds(car)
            self._enforce_track_containment(car)
        self._update_lap_progress_and_finish()

    def _ray_distance(self, car: RaceCar, relative_angle_degrees: float, max_distance: float) -> float:
        heading = math.radians(float(car.heading_degrees) + float(relative_angle_degrees))
        return normalized_ray_first_hit(
            origin_x=float(car.x),
            origin_y=float(car.y),
            dir_x=math.cos(heading),
            dir_y=math.sin(heading),
            max_distance=float(max_distance),
            is_blocked=self._is_wall,
            step_size=self._ray_step_size,
            start_offset=self.car_radius * 0.35,
        )

    def _nearest_opponent(self, player: RaceCar) -> RaceCar:
        opponents = [car for idx, car in enumerate(self.cars) if idx != self.player_index]
        if not opponents:
            return player
        return min(opponents, key=lambda other: self._distance(player.x, player.y, other.x, other.y))

    def _compute_obs(self, *, zero_deltas: bool = False) -> np.ndarray:
        if self._force_zero_deltas:
            zero_deltas = True
        player = self.cars[self.player_index]
        nearest_idx, _, dx, dy = self._nearest_track_sample(player.x, player.y)
        tangent_x, tangent_y = self.track_tangents[nearest_idx]
        normal_x, normal_y = -tangent_y, tangent_x
        self_lat_offset = clip_signed((dx * normal_x + dy * normal_y) / max(1e-6, self.track_half_width))

        heading_rad = math.radians(player.heading_degrees)
        heading_x = math.cos(heading_rad)
        heading_y = math.sin(heading_rad)
        _, _, fwd_speed_raw, _ = self._project_to_car_frame(player)
        self_fwd_speed = clip_signed(fwd_speed_raw / max(1.0, self.max_speed))
        if zero_deltas:
            self_lat_offset_delta = 0.0
            self_fwd_speed_delta = 0.0
        else:
            self_lat_offset_delta = clip_signed(self_lat_offset - self._prev_self_lat_offset)
            self_fwd_speed_delta = clip_signed(self_fwd_speed - self._prev_self_fwd_speed)
        self._prev_self_lat_offset = float(self_lat_offset)
        self._prev_self_fwd_speed = float(self_fwd_speed)

        ray_fwd_near = self._ray_distance(player, 0.0, self._ray_near_range)
        ray_fwd_far = self._ray_distance(player, 0.0, self._ray_far_range)
        ray_fwd_left = self._ray_distance(player, -15.0, self._ray_far_range)
        ray_fwd_right = self._ray_distance(player, 15.0, self._ray_far_range)

        target = self._nearest_opponent(player)
        rel_scale_x = max(1.0, float(SCREEN_WIDTH))
        rel_scale_y = max(1.0, float(self.track_bottom))
        vel_scale = max(1.0, self.max_speed + self.max_reverse_speed)
        tgt_dx = clip_signed((target.x - player.x) / rel_scale_x)
        tgt_dy = clip_signed((target.y - player.y) / rel_scale_y)
        tgt_dvx = clip_signed((target.vx - player.vx) / vel_scale)
        tgt_dvy = clip_signed((target.vy - player.vy) / vel_scale)

        lookahead_samples = max(4, min(20, self.track_count // 10 if self.track_count > 0 else 4))
        lookahead_idx = (nearest_idx + lookahead_samples) % max(1, self.track_count)
        lookahead_x, lookahead_y = self.track_centerline[lookahead_idx]
        to_look_x, to_look_y = self._normalize(lookahead_x - player.x, lookahead_y - player.y)
        lookahead_dist = self._distance(player.x, player.y, lookahead_x, lookahead_y)
        lookahead_tangent_x, lookahead_tangent_y = self.track_tangents[lookahead_idx]

        feature_values = {
            "self_lat_offset": float(self_lat_offset),
            "self_lat_offset_delta": float(self_lat_offset_delta),
            "self_fwd_speed": float(self_fwd_speed),
            "self_fwd_speed_delta": float(self_fwd_speed_delta),
            "self_heading_sin": float(math.sin(heading_rad)),
            "self_heading_cos": float(math.cos(heading_rad)),
            "self_in_contact": 1.0 if player.in_contact else 0.0,
            "self_last_action": float(normalize_last_action(self.last_action_index, self.ACT_DIM)),
            "ray_fwd_near": float(ray_fwd_near),
            "ray_fwd_far": float(ray_fwd_far),
            "ray_fwd_left": float(ray_fwd_left),
            "ray_fwd_right": float(ray_fwd_right),
            "tgt_dx": float(tgt_dx),
            "tgt_dy": float(tgt_dy),
            "tgt_dvx": float(tgt_dvx),
            "tgt_dvy": float(tgt_dvy),
            "trk_lookahead_sin": float(clip_signed(heading_x * to_look_y - heading_y * to_look_x)),
            "trk_lookahead_cos": float(clip_signed(heading_x * to_look_x + heading_y * to_look_y)),
            "trk_lookahead_dist": float(
                clip_unit(lookahead_dist / max(1.0, self._ray_far_range * 2.5))
            ),
            "trk_curvature_ahead": float(
                clip_signed(tangent_x * lookahead_tangent_y - tangent_y * lookahead_tangent_x)
            ),
        }
        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Vroom observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        self._last_obs = obs
        self._force_zero_deltas = False
        return obs

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self.match_tracker.set_match_limit(int(self.total_races))
        self.current_race = 1
        self.match_tracker.clear_history()
        self.last_race_winner = None
        self.player_index = 0
        self.done = False
        self._episode_reward_components.reset()
        self.last_action_index = self.ACTION_COAST
        self._prev_self_lat_offset = 0.0
        self._prev_self_fwd_speed = 0.0
        self._setup_race()
        return self._compute_obs(zero_deltas=True)

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._last_obs, 0.0, True, {
                "win": self.last_race_winner == self.player_index,
                "success": int(self._last_episode_success),
                "level": int(self._last_episode_level),
                "reward_components": self._episode_reward_components.totals(),
            }

        self.window_controller.poll_events_or_raise()
        if self.mode == "human":
            action_idx = self._resolve_human_action()
        else:
            action_idx = int(action)
        action_idx = int(np.clip(action_idx, 0, self.ACT_DIM - 1))
        self.last_action_index = int(action_idx)

        phi_prev = float(self._prev_progress_potential)
        was_in_contact = bool(self._prev_player_in_contact)
        self._step_simulation(action_idx)
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
        if self.wall_texture is not None:
            arcade.draw_texture_rect(self.wall_texture, self.track_rect, pixelated=True)
        if self.track_texture is not None:
            arcade.draw_texture_rect(self.track_texture, self.track_rect, pixelated=True)
        (x1, y1), (x2, y2) = self.start_line
        ay1 = self.window_controller.to_arcade_y(y1)
        ay2 = self.window_controller.to_arcade_y(y2)
        arcade.draw_line(x1, ay1, x2, ay2, COLOR_CHARCOAL, 6.0)
        arcade.draw_line(x1, ay1, x2, ay2, COLOR_SOFT_WHITE, 3.0)

    def _draw_obstacles(self) -> None:
        if not self.obstacles:
            return
        for ox, oy in self.obstacles:
            draw_two_tone_tile(
                self.window_controller,
                top_left_x=float(ox),
                top_left_y=float(oy),
                size=float(self.obstacle_size),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
                inset=float(self.obstacle_inset),
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
                color=COLOR_SOFT_WHITE if idx == self.player_index else COLOR_FOG_GRAY,
                line_width=2.0,
            )

    @staticmethod
    def _status_icon_size() -> float:
        return status_icon_size(float(BB_HEIGHT), float(TILE_SIZE))

    def _draw_player_icon(self, winner_idx: int, center_x: float, center_y: float, size: float) -> None:
        pair = self.player_color_pairs[int(winner_idx) % len(self.player_color_pairs)]
        outline_color, fill_color = pair[0], pair[1]
        inset = max(1.0, round(4.0 * (size / max(1.0, float(TILE_SIZE)))))
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outline_color,
            inner_color=fill_color,
            inset=float(inset),
        )

    def _draw_winner_history(self, left: float, right: float, center_y: float) -> None:
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

        max_display = min(max_icons, int(self.WINNER_BAR_HISTORY_SIZE))
        winners = self.winner_bar_history[-max_display:]
        if not winners:
            return

        total_width = len(winners) * icon_size + max(0, len(winners) - 1) * icon_gap
        start_x = float(left) + (available_width - total_width) / 2.0
        for idx, winner in enumerate(winners):
            if winner is None:
                continue
            center_x = start_x + icon_size / 2.0 + idx * (icon_size + icon_gap)
            self._draw_player_icon(int(winner), center_x, center_y, icon_size)

    def render(self) -> float:
        if self.window_controller.window is None:
            return 0.0

        draw_t0 = time.perf_counter()
        self.window_controller.clear(COLOR_SLATE_GRAY)
        self._draw_track()
        self._draw_obstacles()
        self._draw_cars()

        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, BB_HEIGHT, COLOR_NEAR_BLACK)
        center_y = BB_HEIGHT / 2.0
        winners_left = 8.0
        winners_right = float(SCREEN_WIDTH) - 8.0
        self._draw_winner_history(winners_left, winners_right, center_y)
        self.window_controller.flip()
        return time.perf_counter() - draw_t0

    def close(self) -> None:
        self.window_controller.close()
