"""Top-down one-lap racing environment with mask-based procedural tracks."""

from __future__ import annotations

from dataclasses import dataclass
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
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_TILE_SIZE,
    screen_height,
    screen_width,
)
from core.envs.base import Env
from core.primitives import (
    draw_facing_indicator,
    draw_status_square_icon,
    draw_two_tone_tile,
    resolve_circle_collisions,
    status_icon_size,
)
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from games.vroom.trackgen import TrackGenConfig, generate_track


TILE_SIZE = DEFAULT_TILE_SIZE
GRID_WIDTH = DEFAULT_GRID_COLUMNS
GRID_HEIGHT = DEFAULT_GRID_ROWS
BB_HEIGHT = DEFAULT_BOTTOM_BAR_HEIGHT
SCREEN_WIDTH = screen_width(GRID_WIDTH, TILE_SIZE)
SCREEN_HEIGHT = screen_height(GRID_HEIGHT, TILE_SIZE, BB_HEIGHT)
FPS = 60
TRAINING_FPS = 0
WINDOW_TITLE = "Vroom"


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
    track_index: int = 0
    lap_progress: float = 0.0
    finished: bool = False


class VroomEnv(Env):
    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_ACCEL = 3
    ACTION_BRAKE = 4

    NUM_CARS = 4
    TOTAL_RACES = 10

    def __init__(self, mode: str = "train", render: bool = False) -> None:
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
        self.max_speed = 7.0
        self.max_reverse_speed = 2.4
        self.accel_force = 0.33
        self.brake_force = 0.23
        self.turn_rate = 4.4
        self.drag = 0.985
        self.lateral_grip = 0.82
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
        self.ai_avoid_lookahead = self.obstacle_size * 2.7
        self.ai_avoid_lateral = self.obstacle_size * 1.15
        self.ai_avoid_frames = 12
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
        self.total_races = int(self.TOTAL_RACES)
        self.current_race = 1
        self.win_history: list[int | None] = []

        self.steps = 0
        self.done = False
        self.reset()

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
        track = generate_track(
            seed=int(seed),
            width=int(SCREEN_WIDTH),
            height=int(self.track_bottom),
            config=self.track_config,
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
        lateral_spacing = min(self.track_half_width * 0.42, self.car_size * 1.15)
        index_gap = max(6, int(self.car_size * 0.30))

        cars: list[RaceCar] = []
        lane_offsets = (-1.5, -0.5, 0.5, 1.5)
        base_idx = (self.start_index - index_gap) % max(1, self.track_count)
        max_jump = max(1, min(self.track_count // 4, 70))
        spacing_scales = (1.0, 0.9, 0.8, 0.7, 0.6)

        spawn_idx = base_idx
        resolved_offsets = [float(offset) * lateral_spacing for offset in lane_offsets]
        found_layout = False
        for scale in spacing_scales:
            offsets = [float(offset) * lateral_spacing * float(scale) for offset in lane_offsets]
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
        for idx in range(self.NUM_CARS):
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
            )
            if not self._is_driveable_footprint(car.x, car.y):
                car.x = float(cx)
                car.y = float(cy)
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

    def _apply_car_controls(self, car: RaceCar, steer: float, throttle: float) -> None:
        _, _, forward_speed, lateral_speed = self._project_to_car_frame(car)
        speed_ratio = min(1.0, abs(forward_speed) / max(1e-6, self.max_speed))
        car.heading_degrees += float(steer) * self.turn_rate * (0.5 + 0.5 * speed_ratio)

        heading_rad = math.radians(car.heading_degrees)
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        side_x = -forward_y
        side_y = forward_x

        accel_scale = self.contact_accel_scale if car.in_contact else 1.0
        forward_speed += float(throttle) * self.accel_force * accel_scale
        if throttle < 0.0:
            forward_speed += float(throttle) * self.brake_force * accel_scale
        forward_speed = self._clamp(forward_speed, -self.max_reverse_speed, self.max_speed)
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

    def _leader_index(self) -> int | None:
        if not self.cars:
            return None
        best_idx = 0
        best_key = (self.cars[0].lap_progress, float(self.cars[0].track_index))
        for idx, car in enumerate(self.cars[1:], start=1):
            key = (car.lap_progress, float(car.track_index))
            if key > best_key:
                best_key = key
                best_idx = idx
        return int(best_idx)

    def _setup_race(self) -> None:
        self.track_seed = random.randint(0, 2_000_000_000)
        self._generate_track(self.track_seed)
        self._reset_ai_obstacle_rolls()
        self.cars = self._create_car_grid()
        self.winner_index = None
        self.steps = 0

    def _finalize_race(self, winner_idx: int | None) -> None:
        self.last_race_winner = None if winner_idx is None else int(winner_idx)
        self.win_history.append(self.last_race_winner)
        if len(self.win_history) >= self.total_races:
            self.done = True
            self.winner_index = self.last_race_winner
            return
        self.current_race = len(self.win_history) + 1
        self._setup_race()

    def _resolve_human_action(self) -> int:
        if self.window_controller.is_key_down(arcade.key.LEFT) or self.window_controller.is_key_down(arcade.key.A):
            return self.ACTION_LEFT
        if self.window_controller.is_key_down(arcade.key.RIGHT) or self.window_controller.is_key_down(arcade.key.D):
            return self.ACTION_RIGHT
        if self.window_controller.is_key_down(arcade.key.UP) or self.window_controller.is_key_down(arcade.key.W):
            return self.ACTION_ACCEL
        if self.window_controller.is_key_down(arcade.key.DOWN) or self.window_controller.is_key_down(arcade.key.S):
            return self.ACTION_BRAKE
        return self.ACTION_NOOP

    def _reset_ai_obstacle_rolls(self) -> None:
        self.ai_obstacle_rolls = {}
        obstacle_count = int(len(self.obstacles))
        if obstacle_count <= 0:
            return
        for car_idx in range(self.NUM_CARS):
            if car_idx == self.player_index:
                continue
            for obs_idx in range(obstacle_count):
                self.ai_obstacle_rolls[(car_idx, obs_idx)] = bool(random.random() < self.ai_obstacle_avoid_chance)

    def _closest_obstacle_ahead(self, car: RaceCar) -> tuple[int, float, float] | None:
        if not self.obstacles:
            return None

        heading_rad = math.radians(float(car.heading_degrees))
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
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

    def _ai_control_for_car(self, car_index: int, car: RaceCar) -> tuple[float, float]:
        idx, distance, _, _ = self._nearest_track_sample(car.x, car.y)
        look_ahead = 10 + int(distance / max(1.0, self.track_half_width) * 8.0)
        target_idx = (idx + look_ahead) % self.track_count
        target_x, target_y = self.track_centerline[target_idx]
        desired_heading = math.degrees(math.atan2(target_y - car.y, target_x - car.x))
        delta = self._normalize_degrees(desired_heading - car.heading_degrees)

        steer = 0.0
        if delta > 5.0:
            steer = 1.0
        elif delta < -5.0:
            steer = -1.0

        throttle = 1.0
        if abs(delta) > 42.0:
            throttle = 0.42
        if distance > self.track_half_width * 0.9:
            throttle = max(throttle, 0.55)
        if car.in_contact:
            throttle = max(throttle, 0.65)

        if int(car.obstacle_avoid_frames) > 0:
            car.obstacle_avoid_frames = int(car.obstacle_avoid_frames) - 1
            steer = float(car.obstacle_avoid_steer)
            throttle = max(throttle, 0.72)
            return steer, throttle

        ahead = self._closest_obstacle_ahead(car)
        if ahead is not None:
            obs_idx, _, lateral = ahead
            should_avoid = bool(self.ai_obstacle_rolls.get((int(car_index), int(obs_idx)), True))
            if should_avoid:
                steer = -1.0 if lateral >= 0.0 else 1.0
                car.obstacle_avoid_steer = float(steer)
                car.obstacle_avoid_frames = int(self.ai_avoid_frames)
                throttle = max(throttle, 0.68)
                return steer, throttle
        return steer, throttle

    def _player_controls_from_action(self, action_idx: int) -> tuple[float, float]:
        steer = 0.0
        throttle = 0.0
        if action_idx == self.ACTION_LEFT:
            steer = -1.0
        elif action_idx == self.ACTION_RIGHT:
            steer = 1.0
        elif action_idx == self.ACTION_ACCEL:
            throttle = 1.0
        elif action_idx == self.ACTION_BRAKE:
            throttle = -1.0
        return steer, throttle

    def _step_simulation(self, action_idx: int) -> None:
        previous_positions = [(car.x, car.y) for car in self.cars]
        for idx, car in enumerate(self.cars):
            if idx == self.player_index:
                steer, throttle = self._player_controls_from_action(action_idx)
            else:
                steer, throttle = self._ai_control_for_car(idx, car)
            self._apply_car_controls(car, steer, throttle)

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

    def _get_obs(self) -> np.ndarray:
        player = self.cars[self.player_index]
        nearest_idx, _, dx, dy = self._nearest_track_sample(player.x, player.y)
        tangent_x, tangent_y = self.track_tangents[nearest_idx]
        normal_x, normal_y = -tangent_y, tangent_x
        signed_lateral = (dx * normal_x + dy * normal_y) / max(1e-6, self.track_half_width)

        heading_rad = math.radians(player.heading_degrees)
        heading_x = math.cos(heading_rad)
        heading_y = math.sin(heading_rad)
        heading_alignment_sin = heading_x * tangent_y - heading_y * tangent_x
        heading_alignment_cos = heading_x * tangent_x + heading_y * tangent_y
        forward_speed = player.vx * tangent_x + player.vy * tangent_y

        nearest_opponent_dist = min(
            self._distance(player.x, player.y, car.x, car.y)
            for idx, car in enumerate(self.cars)
            if idx != self.player_index
        )
        nearest_opponent_dist = nearest_opponent_dist / max(1.0, SCREEN_WIDTH * 0.6)

        return np.asarray(
            [
                float(self._clamp(signed_lateral, -1.0, 1.0)),
                float(self._clamp(forward_speed / max(1.0, self.max_speed), -1.0, 1.0)),
                float(self._clamp(heading_alignment_sin, -1.0, 1.0)),
                float(self._clamp(heading_alignment_cos, -1.0, 1.0)),
                float(self._clamp(nearest_opponent_dist, 0.0, 1.0)),
                1.0 if player.in_contact else 0.0,
            ],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.current_race = 1
        self.win_history = []
        self.last_race_winner = None
        self.player_index = 0
        self.done = False
        self._setup_race()
        return self._get_obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._get_obs(), 0.0, True, {"win": self.last_race_winner == self.player_index}

        self.window_controller.poll_events_or_raise()
        if self.mode == "human":
            action_idx = self._resolve_human_action()
        else:
            action_idx = int(action)
        action_idx = int(np.clip(action_idx, 0, 4))

        self._step_simulation(action_idx)
        self.steps += 1

        reward = 0.0
        if self.mode != "human":
            reward -= 0.002
            player = self.cars[self.player_index]
            nearest_idx = self._nearest_track_sample(player.x, player.y)[0]
            tangent_x, tangent_y = self.track_tangents[nearest_idx]
            progress_speed = max(0.0, player.vx * tangent_x + player.vy * tangent_y)
            reward += progress_speed * 0.01
            if not self._is_on_track(player.x, player.y):
                reward -= 0.01

        race_finished = (self.winner_index is not None) or (self.steps >= self.max_steps)
        if race_finished:
            race_winner = self.winner_index if self.winner_index is not None else self._leader_index()
            if self.mode != "human":
                reward += 10.0 if race_winner == self.player_index else -5.0
            self._finalize_race(race_winner)

        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        info = {
            "win": bool(self.last_race_winner == self.player_index) if race_finished else False,
            "winner_index": -1 if self.last_race_winner is None else int(self.last_race_winner),
            "race": int(min(self.current_race, self.total_races)),
            "races_finished": int(len(self.win_history)),
            "races_total": int(self.total_races),
        }
        return self._get_obs(), float(reward), bool(self.done), info

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

        winners = self.win_history[-max_icons:]
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
        self.window_controller.clear(COLOR_CHARCOAL)
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
