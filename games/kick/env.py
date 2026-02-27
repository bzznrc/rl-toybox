"""Simple 11v11 football environment with human and RL control modes."""

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
    COLOR_SLATE_GRAY,
    COLOR_SOFT_WHITE,
)
from core.envs.base import Env
from core.io_schema import clip_signed, normalize_last_action, ordered_feature_vector
from core.primitives import (
    draw_control_marker,
    draw_facing_indicator,
    draw_status_square_icon,
    draw_time_pie_indicator,
    draw_two_tone_tile,
    resolve_circle_collisions,
    status_icon_size,
)
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from games.kick.config import (
    ACTION_NAMES as KICK_ACTION_NAMES,
    ACT_DIM as KICK_ACT_DIM,
    BALL_RADIUS_SCALE,
    BB_HEIGHT,
    CELL_INSET,
    FPS,
    GAME_SPEED_SCALE,
    INPUT_FEATURE_NAMES as KICK_INPUT_FEATURE_NAMES,
    OBS_DIM as KICK_OBS_DIM,
    PHYSICS_DT,
    PENALTY_AREA_DEPTH_RATIO,
    PENALTY_AREA_WIDTH_RATIO,
    PLAYER_A_MAX_PX_PER_SEC2,
    PLAYER_V_MAX_PX_PER_SEC,
    PITCH_LINE_WIDTH,
    PITCH_BACKGROUND_ACCENT_COLOR,
    PITCH_BACKGROUND_COLOR,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STAMINA_DRAIN_SECONDS,
    STAMINA_MAX,
    STAMINA_MIN,
    STAMINA_RECOVER_SECONDS,
    TILE_SIZE,
    TRAINING_FPS,
    WINDOW_TITLE,
)


@dataclass
class KickPlayer:
    team: str
    role: str
    x: float
    y: float
    home_x: float
    home_y: float
    angle: float
    has_ball: bool = False
    contest_cooldown: int = 0
    vx: float = 0.0
    vy: float = 0.0
    stamina: float = STAMINA_MAX
    stamina_delta: float = 0.0
    in_contact: bool = False


class KickEnv(Env):
    """11v11 top-down football environment.

    Human controls:
    - Move: WASD
    - Aim: Mouse
    - Shoot: Left click hold/release
    - Change controlled player: TAB (closest to ball)

    RL controls (Discrete 12):
    - 0: STAY
    - 1..8: MOVE_N, MOVE_NE, MOVE_E, MOVE_SE, MOVE_S, MOVE_SW, MOVE_W, MOVE_NW
    - 9..11: KICK_LOW, KICK_MID, KICK_HIGH
    """

    ACTION_STAY = 0
    ACTION_MOVE_N = 1
    ACTION_MOVE_NE = 2
    ACTION_MOVE_E = 3
    ACTION_MOVE_SE = 4
    ACTION_MOVE_S = 5
    ACTION_MOVE_SW = 6
    ACTION_MOVE_W = 7
    ACTION_MOVE_NW = 8
    ACTION_KICK_LOW = 9
    ACTION_KICK_MID = 10
    ACTION_KICK_HIGH = 11
    INPUT_FEATURE_NAMES = tuple(KICK_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(KICK_ACTION_NAMES)
    OBS_DIM = int(KICK_OBS_DIM)
    ACT_DIM = int(KICK_ACT_DIM)
    NUM_ACTIONS = ACT_DIM

    TEAM_LEFT = "left"
    TEAM_RIGHT = "right"
    ROLE_ORDER = ("GK", "LB", "LCB", "RCB", "RB", "LM", "LCM", "RCM", "RM", "ST1", "ST2")
    ACTION_TO_DIRECTION = {
        ACTION_STAY: (0.0, 0.0),
        ACTION_MOVE_N: (0.0, -1.0),
        ACTION_MOVE_NE: (1.0, -1.0),
        ACTION_MOVE_E: (1.0, 0.0),
        ACTION_MOVE_SE: (1.0, 1.0),
        ACTION_MOVE_S: (0.0, 1.0),
        ACTION_MOVE_SW: (-1.0, 1.0),
        ACTION_MOVE_W: (-1.0, 0.0),
        ACTION_MOVE_NW: (-1.0, -1.0),
    }
    KICK_ACTION_TO_KIND = {
        ACTION_KICK_LOW: 1,
        ACTION_KICK_MID: 2,
        ACTION_KICK_HIGH: 3,
    }
    OBS_NEAREST_PLAYERS = 2
    MATCH_DURATION_SECONDS = 60.0

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

        self.pitch_top = 0.0
        self.pitch_bottom = float(SCREEN_HEIGHT - BB_HEIGHT)
        self.pitch_height = self.pitch_bottom - self.pitch_top
        self.pitch_center_y = self.pitch_height * 0.5
        self.goal_half_height = TILE_SIZE * 3.0
        self.goal_top = self.pitch_center_y - self.goal_half_height
        self.goal_bottom = self.pitch_center_y + self.goal_half_height
        self.goal_box_depth = TILE_SIZE * 6.0
        self.player_size = float(TILE_SIZE)
        self.player_half = self.player_size * 0.5
        self.speed_scale = float(GAME_SPEED_SCALE)
        self.ball_radius = max(3.0, TILE_SIZE * 0.2 * float(BALL_RADIUS_SCALE))
        self.ball_drag_offset = TILE_SIZE * 0.58
        self.physics_dt = float(PHYSICS_DT)
        self.player_vmax_base = float(PLAYER_V_MAX_PX_PER_SEC)
        self.player_amax_base = float(PLAYER_A_MAX_PX_PER_SEC2)
        self.ball_max_speed = max(1.0, 14.5 * self.speed_scale)
        self.ball_friction = 0.985
        self.pickup_range = TILE_SIZE * 0.7 * max(0.75, self.speed_scale)
        self.contest_range = TILE_SIZE * 1.45
        self.contest_cooldown_frames = max(1, int(FPS))
        self.freeze_after_restart = 20
        self.max_steps = int(FPS * float(self.MATCH_DURATION_SECONDS))
        self.max_player_speed = max(1.0, self.player_vmax_base * STAMINA_MAX)
        self.stamina_drain_per_step = (STAMINA_MAX - STAMINA_MIN) / max(1.0, STAMINA_DRAIN_SECONDS * FPS)
        self.stamina_recover_per_step = (STAMINA_MAX - STAMINA_MIN) / max(1.0, STAMINA_RECOVER_SECONDS * FPS)
        self.player_contact_radius = self.player_size * 0.44
        self.contact_sep_strength = 0.5
        self.contact_overlap_cap = self.player_size * 0.02
        self.contact_damp = 0.08
        self.contact_accel_scale = 0.7

        self.left_players: list[KickPlayer] = []
        self.right_players: list[KickPlayer] = []
        self.all_players: list[KickPlayer] = []
        self.left_goalkeeper: KickPlayer | None = None
        self.right_goalkeeper: KickPlayer | None = None
        self.controlled_index = 9

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_owner: KickPlayer | None = None

        self.last_touch_team = self.TEAM_LEFT
        self.left_score = 0
        self.right_score = 0
        self.steps = 0
        self.done = False
        self.freeze_frames = 0
        self.last_action_index = self.ACTION_STAY

        self._goal_scored_team: str | None = None
        self._kick_outcome_reward_event = 0.0
        self._pending_controlled_kick = False
        self._left_possession_before_step = False

        self._prev_tab_down = False
        self._prev_left_mouse_down = False
        self._human_shot_hold_start: float | None = None

        self._build_teams()
        self.reset()

    def _build_teams(self) -> None:
        self.left_players = self._team_from_442(self.TEAM_LEFT)
        self.right_players = self._team_from_442(self.TEAM_RIGHT)
        self.all_players = [*self.left_players, *self.right_players]
        self.left_goalkeeper = self.left_players[0]
        self.right_goalkeeper = self.right_players[0]

    def _team_from_442(self, team: str) -> list[KickPlayer]:
        y4 = (
            self.pitch_top + self.pitch_height * 0.16,
            self.pitch_top + self.pitch_height * 0.38,
            self.pitch_top + self.pitch_height * 0.62,
            self.pitch_top + self.pitch_height * 0.84,
        )
        y2 = (
            self.pitch_top + self.pitch_height * 0.40,
            self.pitch_top + self.pitch_height * 0.60,
        )
        line_x_left = {
            "GK": SCREEN_WIDTH * 0.06,
            "D": SCREEN_WIDTH * 0.20,
            "M": SCREEN_WIDTH * 0.38,
            "S": SCREEN_WIDTH * 0.56,
        }
        if team == self.TEAM_RIGHT:
            line_x = {key: SCREEN_WIDTH - value for key, value in line_x_left.items()}
            default_angle = 180.0
        else:
            line_x = line_x_left
            default_angle = 0.0

        placement = {
            "GK": (line_x["GK"], self.pitch_center_y),
            "LB": (line_x["D"], y4[0]),
            "LCB": (line_x["D"], y4[1]),
            "RCB": (line_x["D"], y4[2]),
            "RB": (line_x["D"], y4[3]),
            "LM": (line_x["M"], y4[0]),
            "LCM": (line_x["M"], y4[1]),
            "RCM": (line_x["M"], y4[2]),
            "RM": (line_x["M"], y4[3]),
            "ST1": (line_x["S"], y2[0]),
            "ST2": (line_x["S"], y2[1]),
        }

        players: list[KickPlayer] = []
        for role in self.ROLE_ORDER:
            px, py = placement[role]
            players.append(
                KickPlayer(
                    team=team,
                    role=role,
                    x=float(px),
                    y=float(py),
                    home_x=float(px),
                    home_y=float(py),
                    angle=float(default_angle),
                )
            )
        return players

    def _controlled_player(self) -> KickPlayer:
        idx = max(0, min(int(self.controlled_index), len(self.left_players) - 1))
        self.controlled_index = idx
        return self.left_players[idx]

    def _left_has_possession(self) -> bool:
        return self.ball_owner is not None and self.ball_owner.team == self.TEAM_LEFT

    def _resolve_pending_kick_outcome(self, success: bool) -> None:
        if not self._pending_controlled_kick:
            return
        self._pending_controlled_kick = False
        self._kick_outcome_reward_event += 0.2 if bool(success) else -0.2

    def _set_ball_owner(self, owner: KickPlayer | None) -> None:
        if owner is not None and self._pending_controlled_kick:
            controlled = self._controlled_player()
            if owner.team == self.TEAM_LEFT and owner is not controlled:
                self._resolve_pending_kick_outcome(True)
            else:
                self._resolve_pending_kick_outcome(False)

        for player in self.all_players:
            player.has_ball = False
        self.ball_owner = owner
        if owner is None:
            return
        owner.has_ball = True
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.last_touch_team = owner.team

        if self.mode == "human" and owner.team == self.TEAM_LEFT and owner in self.left_players:
            new_index = self.left_players.index(owner)
            if new_index != self.controlled_index:
                self.controlled_index = new_index
                self._human_shot_hold_start = None

        self._attach_ball_to_owner()
        if owner is not self._controlled_player():
            self._human_shot_hold_start = None

    def _attach_ball_to_owner(self) -> None:
        owner = self.ball_owner
        if owner is None:
            return
        radians = math.radians(owner.angle)
        self.ball_x = owner.x + math.cos(radians) * self.ball_drag_offset
        self.ball_y = owner.y + math.sin(radians) * self.ball_drag_offset
        self.ball_x = float(np.clip(self.ball_x, self.ball_radius, SCREEN_WIDTH - self.ball_radius))
        self.ball_y = float(np.clip(self.ball_y, self.ball_radius, self.pitch_bottom - self.ball_radius))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _angle_degrees(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
        return (math.degrees(math.atan2(to_y - from_y, to_x - from_x)) + 360.0) % 360.0

    def _nearest_player(self, team: str, x: float, y: float, exclude: KickPlayer | None = None) -> KickPlayer:
        pool = self.left_players if team == self.TEAM_LEFT else self.right_players
        candidates = [player for player in pool if player is not exclude]
        return min(candidates, key=lambda player: self._distance(player.x, player.y, x, y))

    def _nearest_players(
        self,
        team: str,
        x: float,
        y: float,
        *,
        k: int,
        exclude: KickPlayer | None = None,
    ) -> list[KickPlayer]:
        pool = self.left_players if team == self.TEAM_LEFT else self.right_players
        candidates = [player for player in pool if player is not exclude]
        candidates.sort(key=lambda player: self._distance(player.x, player.y, x, y))
        return candidates[: max(0, int(k))]

    def _role_scalar(self, role: str) -> float:
        try:
            idx = self.ROLE_ORDER.index(str(role))
        except ValueError:
            idx = 0
        return float(idx) / max(1.0, float(len(self.ROLE_ORDER) - 1))

    def _update_stamina(self, player: KickPlayer, moved: bool) -> None:
        previous = float(player.stamina)
        if moved:
            player.stamina = self._clamp(previous - self.stamina_drain_per_step, STAMINA_MIN, STAMINA_MAX)
        else:
            player.stamina = self._clamp(previous + self.stamina_recover_per_step, STAMINA_MIN, STAMINA_MAX)
        delta = float(player.stamina) - previous
        norm = max(self.stamina_drain_per_step, self.stamina_recover_per_step, 1e-6)
        player.stamina_delta = self._clamp(delta / norm, -1.0, 1.0)

    def _decode_action(self, action) -> int:
        action_idx = self.ACTION_STAY
        try:
            if isinstance(action, np.ndarray):
                flat = np.asarray(action).reshape(-1)
                if flat.size > 0:
                    action_idx = int(flat[0])
            elif isinstance(action, (list, tuple)) and len(action) > 0:
                action_idx = int(action[0])
            else:
                action_idx = int(action)
        except (TypeError, ValueError):
            action_idx = self.ACTION_STAY
        return int(np.clip(action_idx, 0, self.NUM_ACTIONS - 1))

    @staticmethod
    def _move_action_from_vector(move_x: float, move_y: float) -> int:
        sx = int(np.sign(float(move_x)))
        sy = int(np.sign(float(move_y)))
        direction_to_action = {
            (0, 0): KickEnv.ACTION_STAY,
            (0, -1): KickEnv.ACTION_MOVE_N,
            (1, -1): KickEnv.ACTION_MOVE_NE,
            (1, 0): KickEnv.ACTION_MOVE_E,
            (1, 1): KickEnv.ACTION_MOVE_SE,
            (0, 1): KickEnv.ACTION_MOVE_S,
            (-1, 1): KickEnv.ACTION_MOVE_SW,
            (-1, 0): KickEnv.ACTION_MOVE_W,
            (-1, -1): KickEnv.ACTION_MOVE_NW,
        }
        return int(direction_to_action.get((sx, sy), KickEnv.ACTION_STAY))

    @staticmethod
    def _kick_action_from_kind(kick_type: int) -> int:
        if int(kick_type) == 1:
            return KickEnv.ACTION_KICK_LOW
        if int(kick_type) == 2:
            return KickEnv.ACTION_KICK_MID
        if int(kick_type) == 3:
            return KickEnv.ACTION_KICK_HIGH
        return KickEnv.ACTION_STAY

    @staticmethod
    def _clamp_vector_magnitude(x: float, y: float, max_magnitude: float) -> tuple[float, float]:
        magnitude = math.hypot(x, y)
        if magnitude <= max_magnitude or magnitude <= 1e-9:
            return float(x), float(y)
        scale = float(max_magnitude) / magnitude
        return float(x * scale), float(y * scale)

    def _max_speed_for(self, player: KickPlayer) -> float:
        return max(0.0, self.player_vmax_base * float(player.stamina))

    def _max_accel_for(self, player: KickPlayer) -> float:
        accel = self.player_amax_base * float(player.stamina)
        if player.in_contact:
            accel *= self.contact_accel_scale
        return max(0.0, accel)

    def _set_player_stationary(self, player: KickPlayer) -> None:
        player.vx = 0.0
        player.vy = 0.0
        player.in_contact = False
        self._update_stamina(player, moved=False)

    def _player_bounds(self, player: KickPlayer) -> tuple[float, float, float, float]:
        min_x = self.player_half
        max_x = SCREEN_WIDTH - self.player_half
        min_y = self.player_half
        max_y = self.pitch_bottom - self.player_half
        if player.role == "GK":
            keeper_margin = SCREEN_WIDTH * 0.12
            if player.team == self.TEAM_LEFT:
                max_x = min(max_x, keeper_margin)
            else:
                min_x = max(min_x, SCREEN_WIDTH - keeper_margin)
        return min_x, max_x, min_y, max_y

    def _clamp_player_position(self, player: KickPlayer) -> None:
        min_x, max_x, min_y, max_y = self._player_bounds(player)
        player.x = self._clamp(player.x, min_x, max_x)
        player.y = self._clamp(player.y, min_y, max_y)

    def _resolve_player_contacts(self) -> None:
        positions = [(player.x, player.y) for player in self.all_players]
        velocities = [(player.vx, player.vy) for player in self.all_players]
        radii = [float(self.player_contact_radius)] * len(self.all_players)
        new_positions, new_velocities, contact_flags = resolve_circle_collisions(
            positions,
            velocities,
            radii,
            sep_strength=self.contact_sep_strength,
            overlap_cap=self.contact_overlap_cap,
            contact_damp=self.contact_damp,
        )

        for idx, player in enumerate(self.all_players):
            player.x, player.y = new_positions[idx]
            player.vx, player.vy = new_velocities[idx]
            player.in_contact = bool(contact_flags[idx])
            self._clamp_player_position(player)
            player.vx, player.vy = self._clamp_vector_magnitude(
                player.vx,
                player.vy,
                self._max_speed_for(player),
            )

    def _move_player(self, player: KickPlayer, direction_x: float, direction_y: float) -> None:
        length = math.hypot(direction_x, direction_y)
        if length > 1e-9:
            dir_x = direction_x / length
            dir_y = direction_y / length
        else:
            dir_x = 0.0
            dir_y = 0.0

        max_speed = self._max_speed_for(player)
        max_accel = self._max_accel_for(player)
        desired_vx = dir_x * max_speed
        desired_vy = dir_y * max_speed

        dvx = desired_vx - float(player.vx)
        dvy = desired_vy - float(player.vy)
        max_delta_v = max_accel * self.physics_dt
        dvx, dvy = self._clamp_vector_magnitude(dvx, dvy, max_delta_v)

        next_vx = float(player.vx) + dvx
        next_vy = float(player.vy) + dvy
        next_vx, next_vy = self._clamp_vector_magnitude(next_vx, next_vy, max_speed)

        next_x = float(player.x) + next_vx * self.physics_dt
        next_y = float(player.y) + next_vy * self.physics_dt

        min_x, max_x, min_y, max_y = self._player_bounds(player)

        clamped_x = self._clamp(next_x, min_x, max_x)
        clamped_y = self._clamp(next_y, min_y, max_y)
        if clamped_x != next_x:
            next_vx = 0.0
        if clamped_y != next_y:
            next_vy = 0.0

        player.x = clamped_x
        player.y = clamped_y
        player.vx = next_vx
        player.vy = next_vy
        moved = math.hypot(player.vx, player.vy) > 1e-3
        self._update_stamina(player, moved=moved)

    def _shoot_speed(self, hold_seconds: float) -> float:
        if hold_seconds < 0.1:
            return 5.5 * self.speed_scale
        if hold_seconds < 0.3:
            return 8.5 * self.speed_scale
        if hold_seconds < 0.5:
            return 11.5 * self.speed_scale
        return 14.5 * self.speed_scale

    def _kick_type_from_hold_seconds(self, hold_seconds: float) -> int:
        if hold_seconds < 0.3:
            return 1
        if hold_seconds < 0.5:
            return 2
        return 3

    def _apply_kick_or_contest(self, player: KickPlayer, kick_type: int) -> None:
        kind = int(np.clip(int(kick_type), 0, 3))
        if kind <= 0:
            return
        if player.has_ball and self.ball_owner is player:
            if kind == 1:
                speed = self._shoot_speed(0.15)
            elif kind == 2:
                speed = self._shoot_speed(0.35)
            else:
                speed = self._shoot_speed(0.60)
            self._kick_ball(player, speed=speed)
        return

    @staticmethod
    def _point_segment_distance(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-9:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        return math.hypot(px - closest_x, py - closest_y)

    def _contest_chance(self, owner: KickPlayer, challenger: KickPlayer) -> float:
        to_challenger_x = challenger.x - owner.x
        to_challenger_y = challenger.y - owner.y
        mag = math.hypot(to_challenger_x, to_challenger_y)
        if mag <= 1e-6:
            return 0.5
        to_challenger_x /= mag
        to_challenger_y /= mag
        facing_x = math.cos(math.radians(owner.angle))
        facing_y = math.sin(math.radians(owner.angle))
        dot = facing_x * to_challenger_x + facing_y * to_challenger_y
        if dot >= 0.5:
            return 0.75
        if dot <= -0.5:
            return 0.25
        return 0.5

    def _kick_ball(self, player: KickPlayer, speed: float, angle_degrees: float | None = None) -> None:
        if not player.has_ball or self.ball_owner is not player:
            return
        if player.team == self.TEAM_LEFT and player is self._controlled_player():
            self._pending_controlled_kick = True
        angle = float(player.angle if angle_degrees is None else angle_degrees) % 360.0
        radians = math.radians(angle)
        self._set_ball_owner(None)
        self.ball_x = player.x + math.cos(radians) * self.ball_drag_offset
        self.ball_y = player.y + math.sin(radians) * self.ball_drag_offset
        self.ball_vx = math.cos(radians) * float(speed)
        self.ball_vy = math.sin(radians) * float(speed)
        self.last_touch_team = player.team

    def _attempt_contest(self, player: KickPlayer) -> bool:
        owner = self.ball_owner
        if owner is None or owner.team == player.team:
            return False
        if player.contest_cooldown > 0:
            return False

        distance_to_ball = self._distance(player.x, player.y, self.ball_x, self.ball_y)
        if distance_to_ball > self.contest_range:
            return False

        chance = self._contest_chance(owner, player)
        success = random.random() < chance

        player.contest_cooldown = self.contest_cooldown_frames

        if success:
            loser = owner
            self._set_ball_owner(player)
            loser.contest_cooldown = max(loser.contest_cooldown, self.contest_cooldown_frames)
            return True
        return False

    def _switch_controlled_to_closest_left_player(self) -> None:
        closest = self._nearest_player(self.TEAM_LEFT, self.ball_x, self.ball_y)
        self.controlled_index = self.left_players.index(closest)
        self._human_shot_hold_start = None

    def _decay_timers(self) -> None:
        for player in self.all_players:
            if player.contest_cooldown > 0:
                player.contest_cooldown -= 1

    def _run_auto_contests(self) -> None:
        owner = self.ball_owner
        if owner is None:
            return
        challengers = self.left_players if owner.team == self.TEAM_RIGHT else self.right_players
        in_range = [
            player
            for player in challengers
            if player.contest_cooldown <= 0 and self._distance(player.x, player.y, self.ball_x, self.ball_y) <= self.contest_range
        ]
        in_range.sort(key=lambda player: self._distance(player.x, player.y, self.ball_x, self.ball_y))
        for challenger in in_range:
            current_owner = self.ball_owner
            if current_owner is None or current_owner.team == challenger.team:
                return
            if self._attempt_contest(challenger):
                return

    def _human_controlled_step(self) -> None:
        controlled = self._controlled_player()

        tab_down = self.window_controller.is_key_down(arcade.key.TAB)
        if tab_down and not self._prev_tab_down:
            self._switch_controlled_to_closest_left_player()
            controlled = self._controlled_player()
        self._prev_tab_down = tab_down

        up = self.window_controller.is_key_down(arcade.key.W) or self.window_controller.is_key_down(arcade.key.UP)
        down = self.window_controller.is_key_down(arcade.key.S) or self.window_controller.is_key_down(arcade.key.DOWN)
        left = self.window_controller.is_key_down(arcade.key.A) or self.window_controller.is_key_down(arcade.key.LEFT)
        right = self.window_controller.is_key_down(arcade.key.D) or self.window_controller.is_key_down(arcade.key.RIGHT)

        move_x = float(right) - float(left)
        move_y = float(down) - float(up)
        self.last_action_index = self._move_action_from_vector(move_x, move_y)
        self._move_player(controlled, move_x, move_y)

        mouse_pos = self.window_controller.mouse_position()
        if mouse_pos is not None:
            mouse_x, mouse_y_arcade = mouse_pos
            mouse_y = self.window_controller.to_top_left_y(mouse_y_arcade)
            if self._distance(controlled.x, controlled.y, mouse_x, mouse_y) > 0.0:
                controlled.angle = self._angle_degrees(controlled.x, controlled.y, mouse_x, mouse_y)

        left_mouse_down = self.window_controller.is_mouse_button_down(arcade.MOUSE_BUTTON_LEFT)
        if left_mouse_down and not self._prev_left_mouse_down:
            if controlled.has_ball:
                self._human_shot_hold_start = time.perf_counter()

        if (not left_mouse_down) and self._prev_left_mouse_down:
            if controlled.has_ball and self._human_shot_hold_start is not None:
                hold = max(0.0, time.perf_counter() - self._human_shot_hold_start)
                kick_type = self._kick_type_from_hold_seconds(hold)
                self._apply_kick_or_contest(controlled, kick_type=kick_type)
                self.last_action_index = self._kick_action_from_kind(kick_type)
            self._human_shot_hold_start = None

        self._prev_left_mouse_down = left_mouse_down

    def _rl_controlled_step(self, action) -> None:
        controlled = self._controlled_player()
        action_idx = self._decode_action(action)
        self.last_action_index = int(action_idx)

        if action_idx <= self.ACTION_MOVE_NW:
            move_x, move_y = self.ACTION_TO_DIRECTION.get(action_idx, (0.0, 0.0))
            self._move_player(controlled, move_x, move_y)
            if move_x != 0.0 or move_y != 0.0:
                controlled.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)
            return

        # Kicks imply no movement; without possession they are ignored as STAY.
        self._move_player(controlled, 0.0, 0.0)
        if not controlled.has_ball or self.ball_owner is not controlled:
            return

        kick_type = self.KICK_ACTION_TO_KIND.get(action_idx, 0)
        if kick_type <= 0:
            return
        self._apply_kick_or_contest(controlled, kick_type=kick_type)

    def _is_lane_blocked(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        defenders: list[KickPlayer],
        *,
        margin: float = 0.9,
    ) -> bool:
        lane_margin = TILE_SIZE * float(margin)
        for defender in defenders:
            if self._point_segment_distance(defender.x, defender.y, from_x, from_y, to_x, to_y) <= lane_margin:
                return True
        return False

    def _has_defender_in_front(self, carrier: KickPlayer, defenders: list[KickPlayer]) -> bool:
        facing_x = math.cos(math.radians(carrier.angle))
        facing_y = math.sin(math.radians(carrier.angle))
        for defender in defenders:
            rel_x = defender.x - carrier.x
            rel_y = defender.y - carrier.y
            forward = rel_x * facing_x + rel_y * facing_y
            if forward <= TILE_SIZE * 0.6 or forward > TILE_SIZE * 9.5:
                continue
            lateral = abs(rel_x * facing_y - rel_y * facing_x)
            if lateral <= TILE_SIZE * 1.5:
                return True
        return False

    def _select_progressive_pass_target(self, carrier: KickPlayer) -> KickPlayer | None:
        attack_sign = 1.0 if carrier.team == self.TEAM_LEFT else -1.0
        teammates = [p for p in self.all_players if p.team == carrier.team and p is not carrier]
        defenders = [p for p in self.all_players if p.team != carrier.team]
        best_target: KickPlayer | None = None
        best_score = -1e9
        for teammate in teammates:
            progress = (teammate.x - carrier.x) * attack_sign
            if progress <= TILE_SIZE * 0.6:
                continue
            if not defenders:
                min_clearance = TILE_SIZE * 2.0
            else:
                min_clearance = min(
                    self._point_segment_distance(
                        defender.x,
                        defender.y,
                        carrier.x,
                        carrier.y,
                        teammate.x,
                        teammate.y,
                    )
                    for defender in defenders
                )
            if min_clearance < TILE_SIZE * 0.55:
                continue
            distance = self._distance(carrier.x, carrier.y, teammate.x, teammate.y)
            score = progress - 0.18 * distance + 0.35 * min_clearance
            if score > best_score:
                best_score = score
                best_target = teammate
        return best_target

    def _ai_player_step(self, player: KickPlayer) -> None:
        controlled = self._controlled_player()
        if player is controlled:
            return

        if player.role == "GK":
            self._ai_goalkeeper_step(player)
            return

        target_x = player.home_x
        target_y = player.home_y

        if player.has_ball:
            goal_x = SCREEN_WIDTH - self.ball_radius if player.team == self.TEAM_LEFT else self.ball_radius
            defenders = [candidate for candidate in self.all_players if candidate.team != player.team]
            blocker_in_front = self._has_defender_in_front(player, defenders)
            pass_target = self._select_progressive_pass_target(player) if blocker_in_front else None

            if pass_target is not None:
                pass_angle = self._angle_degrees(player.x, player.y, pass_target.x, pass_target.y)
                player.angle = pass_angle
                self._kick_ball(player, speed=self._shoot_speed(0.35), angle_degrees=pass_angle)
                return

            if blocker_in_front:
                lane_sign = -1.0 if player.y > self.pitch_center_y else 1.0
                target_x = player.x + (1.0 if player.team == self.TEAM_LEFT else -1.0) * TILE_SIZE * 4.0
                target_y = self._clamp(player.y + lane_sign * TILE_SIZE * 3.2, TILE_SIZE, self.pitch_bottom - TILE_SIZE)
            else:
                target_x = goal_x
                home_bias = (player.home_y - self.pitch_center_y) * 0.25
                target_y = self._clamp(self.pitch_center_y + home_bias, TILE_SIZE, self.pitch_bottom - TILE_SIZE)

            player.angle = self._angle_degrees(player.x, player.y, target_x, target_y)
            self._move_player(player, target_x - player.x, target_y - player.y)

            shoot_lane_blocked = self._is_lane_blocked(player.x, player.y, goal_x, self.pitch_center_y, defenders, margin=1.0)
            shoot_threshold = SCREEN_WIDTH * 0.27
            distance_to_goal = abs(goal_x - player.x)
            if distance_to_goal < shoot_threshold and (not shoot_lane_blocked) and random.random() < 0.12:
                variance = random.uniform(-self.goal_half_height * 0.7, self.goal_half_height * 0.7)
                player.angle = self._angle_degrees(player.x, player.y, goal_x, self.pitch_center_y + variance)
                self._kick_ball(player, speed=11.0 * self.speed_scale)
            return

        owner = self.ball_owner
        if owner is None:
            nearest_teammate = self._nearest_player(player.team, self.ball_x, self.ball_y)
            if nearest_teammate is player:
                target_x = self.ball_x
                target_y = self.ball_y
            else:
                pull = 0.28
                target_x = (1.0 - pull) * player.home_x + pull * self.ball_x
                target_y = (1.0 - pull) * player.home_y + pull * self.ball_y
        elif owner.team == player.team:
            if owner is player:
                return
            support = 0.20 if player.role.startswith("S") else 0.12
            target_x = (1.0 - support) * player.home_x + support * owner.x
            target_y = (1.0 - support) * player.home_y + support * owner.y
        else:
            nearest_teammate = self._nearest_player(player.team, owner.x, owner.y)
            if nearest_teammate is player:
                target_x = owner.x
                target_y = owner.y
            else:
                press = 0.18
                target_x = (1.0 - press) * player.home_x + press * owner.x
                target_y = (1.0 - press) * player.home_y + press * owner.y
            if self._distance(player.x, player.y, self.ball_x, self.ball_y) < self.contest_range * 0.9:
                self._attempt_contest(player)

        distance = self._distance(player.x, player.y, target_x, target_y)
        if distance > 1.0:
            player.angle = self._angle_degrees(player.x, player.y, target_x, target_y)
            self._move_player(player, target_x - player.x, target_y - player.y)
        else:
            self._move_player(player, 0.0, 0.0)

    def _ai_goalkeeper_step(self, keeper: KickPlayer) -> None:
        defend_left = keeper.team == self.TEAM_LEFT
        home_x = keeper.home_x

        target_x = home_x
        target_y = self._clamp(self.ball_y, self.goal_top + self.player_half, self.goal_bottom - self.player_half)
        if self.ball_owner is not None and self.ball_owner.team != keeper.team:
            step_out = TILE_SIZE * 0.6
            target_x = home_x + step_out if defend_left else home_x - step_out

        keeper.angle = self._angle_degrees(keeper.x, keeper.y, self.ball_x, self.ball_y)
        self._move_player(keeper, target_x - keeper.x, target_y - keeper.y)

        if keeper.has_ball and random.random() < 0.08:
            clear_x = SCREEN_WIDTH * 0.50
            clear_y = random.uniform(self.pitch_top + TILE_SIZE, self.pitch_bottom - TILE_SIZE)
            keeper.angle = self._angle_degrees(keeper.x, keeper.y, clear_x, clear_y)
            self._kick_ball(keeper, speed=10.0 * self.speed_scale)

    def _step_ball(self) -> None:
        if self.ball_owner is not None:
            self._attach_ball_to_owner()
            return

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        self.ball_vx *= self.ball_friction
        self.ball_vy *= self.ball_friction

        if abs(self.ball_vx) < 0.02:
            self.ball_vx = 0.0
        if abs(self.ball_vy) < 0.02:
            self.ball_vy = 0.0

    def _try_pickup_free_ball(self) -> None:
        if self.ball_owner is not None:
            return
        nearest = min(
            self.all_players,
            key=lambda player: self._distance(player.x, player.y, self.ball_x, self.ball_y),
        )
        distance = self._distance(nearest.x, nearest.y, self.ball_x, self.ball_y)
        ball_speed = math.hypot(self.ball_vx, self.ball_vy)
        if distance <= self.pickup_range and ball_speed <= max(3.5, 9.0 * self.speed_scale):
            self._set_ball_owner(nearest)

    def _try_goalkeeper_catch(self) -> None:
        if self.ball_owner is not None:
            return

        keepers = [keeper for keeper in (self.left_goalkeeper, self.right_goalkeeper) if keeper is not None]
        for keeper in keepers:
            if keeper is None:
                continue
            defend_left = keeper.team == self.TEAM_LEFT
            toward_goal = self.ball_vx < -0.1 if defend_left else self.ball_vx > 0.1
            if not toward_goal:
                continue

            in_box = self.ball_x < self.goal_box_depth if defend_left else self.ball_x > SCREEN_WIDTH - self.goal_box_depth
            if not in_box:
                continue
            if self.ball_y < self.goal_top - TILE_SIZE or self.ball_y > self.goal_bottom + TILE_SIZE:
                continue

            if self._distance(self.ball_x, self.ball_y, keeper.x, keeper.y) > TILE_SIZE * 1.8:
                continue

            offset_from_keeper = abs(self.ball_y - keeper.y)
            catch_prob = self._clamp(1.0 - offset_from_keeper / max(1.0, self.goal_half_height), 0.0, 1.0)
            if random.random() < catch_prob:
                self._set_ball_owner(keeper)
                break

    def _restart_kickoff(self, kickoff_team: str) -> None:
        for player in self.all_players:
            player.x = player.home_x
            player.y = player.home_y
            player.contest_cooldown = 0
            player.has_ball = False
            player.angle = 0.0 if player.team == self.TEAM_LEFT else 180.0
            player.vx = 0.0
            player.vy = 0.0
            player.stamina = STAMINA_MAX
            player.stamina_delta = 0.0
            player.in_contact = False

        self.ball_x = SCREEN_WIDTH * 0.5
        self.ball_y = self.pitch_center_y
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        starter = self.left_players[9] if kickoff_team == self.TEAM_LEFT else self.right_players[9]
        starter.x = self.ball_x - (TILE_SIZE * 0.6 if kickoff_team == self.TEAM_LEFT else -TILE_SIZE * 0.6)
        starter.y = self.ball_y
        starter.angle = 0.0 if kickoff_team == self.TEAM_LEFT else 180.0
        self._set_ball_owner(starter)

        self._pending_controlled_kick = False
        self.freeze_frames = self.freeze_after_restart
        self._human_shot_hold_start = None

    def _restart_throw_in(self, team: str, x: float, y_top: bool) -> None:
        throw_y = TILE_SIZE * 1.2 if y_top else self.pitch_bottom - TILE_SIZE * 1.2
        throw_x = self._clamp(x, TILE_SIZE * 1.5, SCREEN_WIDTH - TILE_SIZE * 1.5)
        player = self._nearest_player(team, throw_x, throw_y)
        player.x = throw_x
        player.y = throw_y
        player.vx = 0.0
        player.vy = 0.0
        target_x = SCREEN_WIDTH * 0.55 if team == self.TEAM_LEFT else SCREEN_WIDTH * 0.45
        target_y = self.pitch_center_y
        player.angle = self._angle_degrees(player.x, player.y, target_x, target_y)
        self._set_ball_owner(player)
        self.freeze_frames = self.freeze_after_restart

    def _restart_corner(self, team: str, left_side: bool, top_corner: bool) -> None:
        corner_x = TILE_SIZE * 0.8 if left_side else SCREEN_WIDTH - TILE_SIZE * 0.8
        corner_y = TILE_SIZE * 0.8 if top_corner else self.pitch_bottom - TILE_SIZE * 0.8
        player = self._nearest_player(team, corner_x, corner_y)
        player.x = corner_x
        player.y = corner_y
        player.vx = 0.0
        player.vy = 0.0
        goal_x = SCREEN_WIDTH - self.ball_radius if team == self.TEAM_LEFT else self.ball_radius
        goal_y = self.pitch_center_y
        player.angle = self._angle_degrees(player.x, player.y, goal_x, goal_y)
        self._set_ball_owner(player)
        self.freeze_frames = self.freeze_after_restart

    def _restart_goal_kick(self, defending_team: str) -> None:
        keeper = self.left_goalkeeper if defending_team == self.TEAM_LEFT else self.right_goalkeeper
        if keeper is None:
            return
        keeper.x = keeper.home_x + TILE_SIZE * 1.8 if defending_team == self.TEAM_LEFT else keeper.home_x - TILE_SIZE * 1.8
        keeper.y = self.pitch_center_y
        keeper.vx = 0.0
        keeper.vy = 0.0
        target_x = SCREEN_WIDTH * 0.45 if defending_team == self.TEAM_LEFT else SCREEN_WIDTH * 0.55
        target_y = self.pitch_center_y
        keeper.angle = self._angle_degrees(keeper.x, keeper.y, target_x, target_y)
        self._set_ball_owner(keeper)
        self.freeze_frames = self.freeze_after_restart

    def _handle_ball_boundaries(self) -> None:
        if self.ball_owner is not None:
            return

        if self.ball_x <= 0.0 and self.goal_top <= self.ball_y <= self.goal_bottom:
            self.right_score += 1
            self._goal_scored_team = self.TEAM_RIGHT
            self._resolve_pending_kick_outcome(False)
            self._restart_kickoff(self.TEAM_LEFT)
            return
        if self.ball_x >= SCREEN_WIDTH and self.goal_top <= self.ball_y <= self.goal_bottom:
            self.left_score += 1
            self._goal_scored_team = self.TEAM_LEFT
            self._resolve_pending_kick_outcome(True)
            self._restart_kickoff(self.TEAM_RIGHT)
            return

        out_top = self.ball_y < 0.0
        out_bottom = self.ball_y > self.pitch_bottom
        if out_top or out_bottom:
            self._resolve_pending_kick_outcome(False)
            receiving_team = self.TEAM_RIGHT if self.last_touch_team == self.TEAM_LEFT else self.TEAM_LEFT
            self._restart_throw_in(receiving_team, x=self.ball_x, y_top=out_top)
            return

        out_left = self.ball_x < 0.0
        out_right = self.ball_x > SCREEN_WIDTH
        if out_left or out_right:
            self._resolve_pending_kick_outcome(False)
            top_corner = self.ball_y < self.pitch_center_y
            if out_left:
                if self.last_touch_team == self.TEAM_LEFT:
                    self._restart_corner(self.TEAM_RIGHT, left_side=True, top_corner=top_corner)
                else:
                    self._restart_goal_kick(self.TEAM_LEFT)
            else:
                if self.last_touch_team == self.TEAM_RIGHT:
                    self._restart_corner(self.TEAM_LEFT, left_side=False, top_corner=top_corner)
                else:
                    self._restart_goal_kick(self.TEAM_RIGHT)

    def _step_players(self, action) -> None:
        if self.mode == "human":
            self._human_controlled_step()
        else:
            self._rl_controlled_step(action)

        for player in self.all_players:
            self._ai_player_step(player)

    def _reset_step_events(self) -> None:
        self._goal_scored_team = None
        self._kick_outcome_reward_event = 0.0

    def _tick(self, action) -> None:
        self.window_controller.poll_events_or_raise()
        self._reset_step_events()
        self._left_possession_before_step = self._left_has_possession()

        if self.freeze_frames > 0:
            self.freeze_frames -= 1
            for player in self.all_players:
                self._set_player_stationary(player)
        else:
            self._step_players(action)
            self._resolve_player_contacts()
            self._run_auto_contests()
            self._decay_timers()
            self._step_ball()
            self._try_pickup_free_ball()
            self._try_goalkeeper_catch()
            self._handle_ball_boundaries()

        self.steps += 1

    def _score_reward(self) -> float:
        reward = -0.005
        if self._goal_scored_team == self.TEAM_LEFT:
            reward += 20.0
        elif self._goal_scored_team == self.TEAM_RIGHT:
            reward -= 10.0

        left_has_possession = self._left_has_possession()
        if left_has_possession:
            reward += 0.01

        # Suppress possession transition penalties/rewards on goal steps to avoid kickoff artifacts.
        if self._goal_scored_team is None:
            if self._left_possession_before_step and not left_has_possession:
                reward -= 0.2
            elif (not self._left_possession_before_step) and left_has_possession:
                reward += 0.1

        reward += float(self._kick_outcome_reward_event)
        return reward

    def _obs(self) -> np.ndarray:
        controlled = self._controlled_player()
        width = float(SCREEN_WIDTH)
        height = float(self.pitch_bottom)
        player_vel_norm = max(1.0, self.max_player_speed)
        ball_vel_norm = max(1.0, self.ball_max_speed)
        nearest_count = int(self.OBS_NEAREST_PLAYERS)
        teammates = self._nearest_players(
            self.TEAM_LEFT,
            controlled.x,
            controlled.y,
            k=nearest_count,
            exclude=controlled,
        )
        opponents = self._nearest_players(self.TEAM_RIGHT, controlled.x, controlled.y, k=nearest_count)

        angle_rad = math.radians(controlled.angle)
        ball_is_free = 1.0 if self.ball_owner is None else 0.0
        if self.ball_owner is None:
            ball_owner_team = 0.0
        elif self.ball_owner.team == self.TEAM_LEFT:
            ball_owner_team = 1.0
        else:
            ball_owner_team = -1.0

        feature_values: dict[str, float] = {
            "self_vx": float(clip_signed(controlled.vx / player_vel_norm)),
            "self_vy": float(clip_signed(controlled.vy / player_vel_norm)),
            "self_theta_cos": float(math.cos(angle_rad)),
            "self_theta_sin": float(math.sin(angle_rad)),
            "self_has_ball": 1.0 if controlled.has_ball else 0.0,
            "self_role": float(self._role_scalar(controlled.role)),
            "self_stamina": float(controlled.stamina),
            "self_stamina_delta": float(clip_signed(controlled.stamina_delta)),
            "self_in_contact": 1.0 if controlled.in_contact else 0.0,
            "self_last_action": float(normalize_last_action(self.last_action_index, self.ACT_DIM)),
            "tgt_dx": float(clip_signed((self.ball_x - controlled.x) / width)),
            "tgt_dy": float(clip_signed((self.ball_y - controlled.y) / height)),
            "tgt_dvx": float(clip_signed((self.ball_vx - controlled.vx) / ball_vel_norm)),
            "tgt_dvy": float(clip_signed((self.ball_vy - controlled.vy) / ball_vel_norm)),
            "tgt_is_free": float(ball_is_free),
            "tgt_owner_team": float(ball_owner_team),
            "goal_opp_dx": float(clip_signed((float(SCREEN_WIDTH) - controlled.x) / width)),
            "goal_opp_dy": float(clip_signed((self.pitch_center_y - controlled.y) / height)),
            "goal_own_dx": float(clip_signed((0.0 - controlled.x) / width)),
            "goal_own_dy": float(clip_signed((self.pitch_center_y - controlled.y) / height)),
        }

        while len(teammates) < nearest_count:
            teammates.append(controlled)
        while len(opponents) < nearest_count:
            opponents.append(self.right_players[0])

        for idx, teammate in enumerate(teammates[:nearest_count], start=1):
            feature_values[f"ally{idx}_dx"] = float(clip_signed((teammate.x - controlled.x) / width))
            feature_values[f"ally{idx}_dy"] = float(clip_signed((teammate.y - controlled.y) / height))
            feature_values[f"ally{idx}_dvx"] = float(clip_signed((teammate.vx - controlled.vx) / player_vel_norm))
            feature_values[f"ally{idx}_dvy"] = float(clip_signed((teammate.vy - controlled.vy) / player_vel_norm))

        for idx, opponent in enumerate(opponents[:nearest_count], start=1):
            feature_values[f"foe{idx}_dx"] = float(clip_signed((opponent.x - controlled.x) / width))
            feature_values[f"foe{idx}_dy"] = float(clip_signed((opponent.y - controlled.y) / height))
            feature_values[f"foe{idx}_dvx"] = float(clip_signed((opponent.vx - controlled.vx) / player_vel_norm))
            feature_values[f"foe{idx}_dvy"] = float(clip_signed((opponent.vy - controlled.vy) / player_vel_norm))

        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Kick observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def reset(self) -> np.ndarray:
        self.left_score = 0
        self.right_score = 0
        self.steps = 0
        self.done = False
        self.freeze_frames = 0
        self.controlled_index = 9
        self._pending_controlled_kick = False
        self._kick_outcome_reward_event = 0.0
        self._prev_tab_down = False
        self._prev_left_mouse_down = False
        self._human_shot_hold_start = None
        self.last_action_index = self.ACTION_STAY
        self._restart_kickoff(self.TEAM_LEFT)
        self._left_possession_before_step = self._left_has_possession()
        return self._obs()

    def _draw_pitch(self) -> None:
        pitch_h = self.pitch_height
        pitch_bottom = self.window_controller.top_left_to_bottom(self.pitch_top, pitch_h)
        line_width = float(PITCH_LINE_WIDTH)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, PITCH_BACKGROUND_COLOR)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, PITCH_BACKGROUND_ACCENT_COLOR + (24,))

        arcade.draw_lbwh_rectangle_outline(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_FOG_GRAY, line_width)
        arcade.draw_line(
            SCREEN_WIDTH * 0.5,
            self.window_controller.to_arcade_y(self.pitch_top),
            SCREEN_WIDTH * 0.5,
            self.window_controller.to_arcade_y(self.pitch_bottom),
            COLOR_FOG_GRAY,
            line_width,
        )

        center_radius = TILE_SIZE * 2.4
        arcade.draw_circle_outline(
            SCREEN_WIDTH * 0.5,
            self.window_controller.to_arcade_y(self.pitch_center_y),
            center_radius,
            COLOR_FOG_GRAY,
            line_width,
        )

        penalty_depth = SCREEN_WIDTH * float(PENALTY_AREA_DEPTH_RATIO)
        penalty_height = self.pitch_height * float(PENALTY_AREA_WIDTH_RATIO)
        penalty_top = self.pitch_center_y - penalty_height * 0.5
        penalty_bottom = self.window_controller.top_left_to_bottom(penalty_top, penalty_height)
        arcade.draw_lbwh_rectangle_outline(
            0,
            penalty_bottom,
            penalty_depth,
            penalty_height,
            COLOR_FOG_GRAY,
            line_width,
        )
        arcade.draw_lbwh_rectangle_outline(
            SCREEN_WIDTH - penalty_depth,
            penalty_bottom,
            penalty_depth,
            penalty_height,
            COLOR_FOG_GRAY,
            line_width,
        )

        goal_h = self.goal_half_height * 2.0
        goal_top = self.pitch_center_y - self.goal_half_height
        left_goal_bottom = self.window_controller.top_left_to_bottom(goal_top, goal_h)
        arcade.draw_lbwh_rectangle_outline(0, left_goal_bottom, TILE_SIZE, goal_h, COLOR_SOFT_WHITE, line_width)
        arcade.draw_lbwh_rectangle_outline(
            SCREEN_WIDTH - TILE_SIZE,
            left_goal_bottom,
            TILE_SIZE,
            goal_h,
            COLOR_SOFT_WHITE,
            line_width,
        )

    def _draw_player(self, player: KickPlayer, *, controlled_marker: bool) -> None:
        if player.team == self.TEAM_LEFT:
            outer = COLOR_AQUA
            inner = COLOR_DEEP_TEAL
        else:
            outer = COLOR_CORAL
            inner = COLOR_BRICK_RED

        draw_two_tone_tile(
            self.window_controller,
            top_left_x=player.x - self.player_half,
            top_left_y=player.y - self.player_half,
            size=self.player_size,
            outer_color=outer,
            inner_color=inner,
            inset=float(CELL_INSET),
        )

        if controlled_marker:
            marker_size = max(3.0, self.player_size * 0.28)
            draw_control_marker(
                self.window_controller,
                center_x=player.x,
                center_y_top_left=player.y,
                marker_size=marker_size,
                color=outer,
            )

        draw_facing_indicator(
            self.window_controller,
            center_x=player.x,
            center_y_top_left=player.y,
            angle_degrees=player.angle,
            length=self.player_size * 0.48,
            color=COLOR_SOFT_WHITE,
            line_width=2.0,
        )

    @staticmethod
    def _status_icon_size() -> float:
        return status_icon_size(float(BB_HEIGHT), float(TILE_SIZE))

    def _team_color_pair(self, team: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if str(team) == self.TEAM_LEFT:
            return COLOR_AQUA, COLOR_DEEP_TEAL
        return COLOR_CORAL, COLOR_BRICK_RED

    def _draw_team_icon(self, team: str, center_x: float, center_y: float, size: float) -> None:
        outline_color, fill_color = self._team_color_pair(team)
        inset = max(1.0, round(CELL_INSET * (size / max(1.0, float(TILE_SIZE)))))
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outline_color,
            inner_color=fill_color,
            inset=float(inset),
        )

    def _remaining_time_ratio(self) -> float:
        frames_left = max(0, int(self.max_steps) - int(self.steps))
        return frames_left / max(1, int(self.max_steps))

    def _draw_time_indicator(self, center_x: float, center_y: float, radius: float, border_width: float) -> None:
        draw_time_pie_indicator(
            center_x=float(center_x),
            center_y=float(center_y),
            radius=float(radius),
            border_width=float(border_width),
            remaining_ratio=float(self._remaining_time_ratio()),
            base_color=COLOR_SLATE_GRAY,
            fill_color=COLOR_FOG_GRAY,
            outline_color=COLOR_FOG_GRAY,
            num_segments=96,
        )

    def _draw_goal_icons(self, left: float, right: float, center_y: float) -> None:
        available_width = max(0.0, float(right) - float(left))
        if available_width <= 0.0:
            return

        icon_size = self._status_icon_size()
        icon_gap = 6.0
        center_x = float(left) + available_width * 0.5
        max_per_team = int(((available_width * 0.5) - icon_gap) // (icon_size + icon_gap))
        if max_per_team <= 0:
            return

        left_count = min(int(self.left_score), max_per_team)
        right_count = min(int(self.right_score), max_per_team)
        left_total_width = left_count * icon_size + max(0, left_count - 1) * icon_gap
        left_start_x = center_x - (icon_gap * 0.5) - left_total_width
        for idx in range(left_count):
            icon_center_x = left_start_x + icon_size * 0.5 + idx * (icon_size + icon_gap)
            self._draw_team_icon(self.TEAM_LEFT, icon_center_x, center_y, icon_size)

        right_start_x = center_x + icon_gap * 0.5
        for idx in range(right_count):
            icon_center_x = right_start_x + icon_size * 0.5 + idx * (icon_size + icon_gap)
            self._draw_team_icon(self.TEAM_RIGHT, icon_center_x, center_y, icon_size)

    def render(self) -> None:
        if self.window_controller.window is None:
            return

        self.window_controller.clear(COLOR_CHARCOAL)
        self._draw_pitch()

        controlled = self._controlled_player()
        for player in self.all_players:
            self._draw_player(player, controlled_marker=(self.mode == "human" and player is controlled))

        arcade.draw_circle_filled(
            self.ball_x,
            self.window_controller.to_arcade_y(self.ball_y),
            self.ball_radius,
            COLOR_FOG_GRAY,
        )
        arcade.draw_circle_filled(
            self.ball_x,
            self.window_controller.to_arcade_y(self.ball_y),
            self.ball_radius * 0.62,
            COLOR_SLATE_GRAY,
        )

        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, BB_HEIGHT, COLOR_NEAR_BLACK)
        center_y = BB_HEIGHT * 0.5
        icon_size = self._status_icon_size()
        indicator_diameter = icon_size * math.sqrt(2.0) * 0.8
        indicator_radius = indicator_diameter * 0.5
        indicator_border = max(1.0, round(CELL_INSET * 0.5))
        indicator_center_x = SCREEN_WIDTH - 10.0 - indicator_radius
        self._draw_time_indicator(
            center_x=indicator_center_x,
            center_y=center_y,
            radius=indicator_radius,
            border_width=indicator_border,
        )
        goals_left = 8.0
        goals_right = max(goals_left, indicator_center_x - indicator_radius - 14.0)
        self._draw_goal_icons(goals_left, goals_right, center_y)
        self.window_controller.flip()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._obs(), 0.0, True, {
                "win": bool(self.left_score > self.right_score),
                "score_left": int(self.left_score),
                "score_right": int(self.right_score),
                "time_left_ratio": float(self._remaining_time_ratio()),
                "controlled_role": self._controlled_player().role,
            }

        parsed_action = action if self.mode != "human" else self.ACTION_STAY
        self._tick(parsed_action)
        if self.steps >= self.max_steps:
            self.done = True

        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        reward = self._score_reward() if self.mode != "human" else 0.0

        done = bool(self.done)
        win = bool(done and self.left_score > self.right_score)
        info = {
            "win": win,
            "score_left": int(self.left_score),
            "score_right": int(self.right_score),
            "time_left_ratio": float(self._remaining_time_ratio()),
            "controlled_role": self._controlled_player().role,
        }
        return self._obs(), float(reward), bool(done), info

    def close(self) -> None:
        self.window_controller.close()
