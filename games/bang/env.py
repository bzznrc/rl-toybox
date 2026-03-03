"""Bang core gameplay, rendering, and game modes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import arcade
import numpy as np
from core.arcade_style import (
    COLOR_AMBER,
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
from core.curriculum import ThreeLevelCurriculum, advance_curriculum, build_curriculum_config
from core.envs.base import Env
from core.io_schema import (
    clip_signed,
    clip_unit,
    normalized_ray_first_hit,
    ordered_feature_vector,
    signed_potential_shaping,
)
from core.primitives import (
    draw_control_marker,
    draw_facing_indicator,
    draw_two_tone_tile,
    spawn_connected_random_walk_shapes,
)
from core.rewards import RewardBreakdown

from games.bang.config import (
    ACTION_NAMES as BANG_ACTION_NAMES,
    ACTION_AIM_LEFT,
    ACTION_AIM_RIGHT,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP,
    ACTION_SHOOT,
    ACTION_STOP_MOVE,
    AIM_RATE_PER_STEP,
    AIM_TOLERANCE_DEGREES,
    BB_HEIGHT,
    CELL_INSET,
    CURRICULUM_PROMOTION,
    ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES,
    ENEMY_ESCAPE_FOLLOW_FRAMES,
    ENEMY_SHOT_ERROR_CHOICES,
    ENEMY_SPAWN_X_RATIO,
    EVENT_TIMER_NORMALIZATION_FRAMES,
    FPS,
    INPUT_FEATURE_NAMES as BANG_INPUT_FEATURE_NAMES,
    OBS_DIM as BANG_OBS_DIM,
    ACT_DIM as BANG_ACT_DIM,
    LEVEL_SETTINGS,
    MAX_EPISODE_STEPS,
    MAX_LEVEL,
    MAX_OBSTACLE_SECTIONS,
    MIN_LEVEL,
    MIN_OBSTACLE_SECTIONS,
    NN_CONTROL_MARKER_SIZE_PX,
    OBSTACLE_START_ATTEMPTS,
    ENGAGEMENT_CLIP,
    ENGAGEMENT_SCALE,
    HAZARD_CLIP,
    HAZARD_SCALE,
    PENALTY_STEP,
    PENALTY_LOSE,
    PLAYER_MOVE_SPEED,
    PLAYER_SPAWN_X_RATIO,
    PROJECTILE_HITBOX_SIZE,
    PROJECTILE_SPEED,
    PROJECTILE_TRAJECTORY_DOT_THRESHOLD,
    REWARD_KILL,
    REWARD_WIN,
    SAFE_RADIUS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SHOOT_COOLDOWN_FRAMES,
    SPAWN_Y_OFFSET,
    TILE_SIZE,
    TRAINING_FPS,
    WINDOW_TITLE,
)
from core.runtime import (
    ArcadeFrameClock,
    ArcadeWindowController,
    Vec2,
    collides_with_square_arena,
    heading_to_vector,
    length_squared,
    normalize_angle_degrees,
    rect_from_center,
    rotate_degrees,
    square_obstacle_between_points,
)
from core.utils import resolve_play_level, validate_level_settings


ALL_PLAYER_ORDER = ("P1", "P2", "P3", "P4")
SUPPORTED_PLAYER_COUNTS = (2, 3, 4)
validate_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
    valid_player_counts=SUPPORTED_PLAYER_COUNTS,
)


def _resolve_player_order(num_players: int) -> tuple[str, ...]:
    count = int(num_players)
    if count not in SUPPORTED_PLAYER_COUNTS:
        raise ValueError(f"num_players must be one of {SUPPORTED_PLAYER_COUNTS}, got {count}")
    return ALL_PLAYER_ORDER[:count]


def _num_players_for_level(level: int) -> int:
    return int(LEVEL_SETTINGS[int(level)]["num_players"])


PLAYER_STYLES = {
    "P1": {
        "render_fill": COLOR_DEEP_TEAL,
        "render_outline": COLOR_AQUA,
        "status_color": COLOR_DEEP_TEAL,
        "scripted": False,
    },
    "P2": {
        "render_fill": COLOR_BRICK_RED,
        "render_outline": COLOR_CORAL,
        "status_color": COLOR_BRICK_RED,
        "scripted": True,
    },
    "P3": {
        "render_fill": COLOR_P3_NAVY,
        "render_outline": COLOR_P3_BLUE,
        "status_color": COLOR_P3_BLUE,
        "scripted": True,
    },
    "P4": {
        "render_fill": COLOR_P4_DEEP_PURPLE,
        "render_outline": COLOR_P4_PURPLE,
        "status_color": COLOR_P4_PURPLE,
        "scripted": True,
    },
}
SPAWN_AREA_LEFT = "left_column"
SPAWN_AREA_RIGHT = "right_column"
SPAWN_AREA_BOTTOM = "bottom_strip"
SPAWN_AREA_TOP = "top_strip"
SPAWN_AREA_ORDER = (
    SPAWN_AREA_LEFT,
    SPAWN_AREA_RIGHT,
    SPAWN_AREA_BOTTOM,
    SPAWN_AREA_TOP,
)

PLAYER_TARGET_POLICY = {
    "max_lost_frames": 40,
    "switch_distance_ratio": 0.8,
    "random_switch_prob": 0.005,
    "hold_min_frames": 30,
    "hold_max_frames": 75,
}
SCRIPTED_TARGET_POLICY = {
    "max_lost_frames": 45,
    "switch_distance_ratio": 0.85,
    "random_switch_prob": 0.02,
    "hold_min_frames": 18,
    "hold_max_frames": 60,
}
CONTROL_MODE_HUMAN = "human"
CONTROL_MODE_SCRIPTED = "scripted"
CONTROL_MODE_NN = "nn"
HUMAN_ACTION_KEY_BINDINGS = {
    ACTION_MOVE_UP: (arcade.key.W,),
    ACTION_MOVE_DOWN: (arcade.key.S,),
    ACTION_MOVE_LEFT: (arcade.key.A,),
    ACTION_MOVE_RIGHT: (arcade.key.D,),
    ACTION_AIM_LEFT: (arcade.key.LEFT,),
    ACTION_AIM_RIGHT: (arcade.key.RIGHT,),
    ACTION_SHOOT: (arcade.key.SPACE,),
}


@dataclass
class TargetState:
    target_id: str | None = None
    target_lost_frames: int = 0
    target_switch_cooldown: int = 0
    last_update_frame: int = -1


@dataclass
class ScriptedMoveState:
    escape_frames_remaining: int = 0
    escape_offset_degrees: float = float(ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES[0])


class Actor:
    """A movable actor that can rotate and shoot projectiles."""

    def __init__(self, position: Vec2, angle: float, team: str = "P1") -> None:
        self.position = position
        self.angle = angle
        self.cooldown_frames = 0
        self.max_health = 1
        self.health = self.max_health
        self.is_alive = True
        self.team = team
        self.vx = 0.0
        self.vy = 0.0

        # Sticky controller state: persists across environment steps.
        self.move_intent_x = 0
        self.move_intent_y = 0
        self.aim_intent = 0

    def step_sticky_intents(self) -> Vec2:
        self.angle = (self.angle + self.aim_intent * AIM_RATE_PER_STEP) % 360
        # Game coordinates are top-left origin, so world +Y maps to screen-up (negative local Y).
        movement = Vec2(float(self.move_intent_x), float(-self.move_intent_y))
        return movement * PLAYER_MOVE_SPEED

    def shoot(self):
        if self.cooldown_frames > 0 or not self.is_alive:
            return None

        direction = heading_to_vector(self.angle)
        self.cooldown_frames = SHOOT_COOLDOWN_FRAMES
        return {
            "pos": self.position + direction * 20,
            "velocity": direction * PROJECTILE_SPEED,
            "owner": self.team,
        }

    def take_hit(self, damage: int = 1) -> bool:
        if not self.is_alive:
            return False
        self.health = max(0, self.health - int(damage))
        if self.health <= 0:
            self.is_alive = False
            return True
        return False

    def tick(self) -> None:
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1


class Renderer:
    """Arcade renderer for the Bang arena."""

    def __init__(self, game, width: int, height: int, title: str, enabled: bool) -> None:
        self.game = game
        self.enabled = bool(enabled)
        self.width = int(width)
        self.height = int(height)

        self.window_controller = ArcadeWindowController(
            self.width,
            self.height,
            title,
            enabled=self.enabled,
            queue_input_events=False,
            vsync=False,
        )
        self.window = self.window_controller.window

    def close(self) -> None:
        self.window_controller.close()
        self.window = None

    def poll_events(self) -> None:
        self.window_controller.poll_events_or_raise()

    def draw_frame(self) -> None:
        if self.window is None:
            return

        self.window_controller.clear(COLOR_CHARCOAL)

        for obstacle in self.game.obstacles:
            self._draw_two_tone_tile(
                top_left_x=float(obstacle.x),
                top_left_y=float(obstacle.y),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
            )

        for player_id in self.game.player_order:
            actor = self.game.players_by_id[player_id]
            if not actor.is_alive:
                continue
            fill_color, outline_color = self.game.player_render_colors.get(
                player_id,
                (COLOR_DEEP_TEAL, COLOR_AQUA),
            )
            self._draw_actor(
                actor,
                fill_color,
                outline_color,
                draw_nn_marker=self.game.is_nn_controlled_player(player_id),
            )

        for projectile in self.game.projectiles:
            owner_id = str(projectile.get("owner", ""))
            projectile_color = self.game.player_projectile_colors.get(owner_id, COLOR_AMBER)
            arcade.draw_circle_filled(
                projectile["pos"].x,
                self.window_controller.to_arcade_y(projectile["pos"].y),
                5,
                projectile_color,
            )

        self._draw_status_bar()
        self.window_controller.flip()

    def _draw_status_bar(self) -> None:
        arcade.draw_lbwh_rectangle_filled(0, 0, self.width, BB_HEIGHT, COLOR_NEAR_BLACK)
        center_y = BB_HEIGHT / 2.0
        icon_size = self._status_icon_size()
        indicator_diameter = icon_size * math.sqrt(2.0) * 0.8
        indicator_radius = indicator_diameter / 2.0
        indicator_border = max(1.0, round(CELL_INSET * 0.5))
        indicator_center_x = self.width - 10.0 - indicator_radius
        self._draw_time_indicator(
            center_x=indicator_center_x,
            center_y=center_y,
            radius=indicator_radius,
            border_width=indicator_border,
        )

        right_reserved = indicator_diameter + 24.0
        winners_left = 8.0
        winners_right = max(winners_left, self.width - right_reserved)
        self._draw_winner_history(winners_left, winners_right, center_y)

    def _remaining_time_ratio(self) -> float:
        frames_left = max(0, MAX_EPISODE_STEPS - int(self.game.frame_count))
        return frames_left / max(1, MAX_EPISODE_STEPS)

    def _draw_time_indicator(self, center_x: float, center_y: float, radius: float, border_width: float) -> None:
        circle_segments = 96
        arcade.draw_circle_filled(center_x, center_y, radius, COLOR_SLATE_GRAY, num_segments=circle_segments)
        inner_radius = max(1.0, radius - border_width)

        remaining_ratio = self._remaining_time_ratio()
        if remaining_ratio <= 0.0:
            arcade.draw_circle_outline(
                center_x,
                center_y,
                radius,
                COLOR_FOG_GRAY,
                border_width,
                num_segments=circle_segments,
            )
            return
        if remaining_ratio >= 1.0:
            arcade.draw_circle_filled(
                center_x,
                center_y,
                inner_radius,
                COLOR_FOG_GRAY,
                num_segments=circle_segments,
            )
            arcade.draw_circle_outline(
                center_x,
                center_y,
                radius,
                COLOR_FOG_GRAY,
                border_width,
                num_segments=circle_segments,
            )
            return

        start_angle = 90.0
        end_angle = start_angle + 360.0 * remaining_ratio
        arcade.draw_arc_filled(
            center_x=center_x,
            center_y=center_y,
            width=inner_radius * 2.0,
            height=inner_radius * 2.0,
            color=COLOR_FOG_GRAY,
            start_angle=start_angle,
            end_angle=end_angle,
            num_segments=circle_segments,
        )
        arcade.draw_circle_outline(
            center_x,
            center_y,
            radius,
            COLOR_FOG_GRAY,
            border_width,
            num_segments=circle_segments,
        )

    @staticmethod
    def _status_icon_size() -> float:
        return max(12.0, min(float(BB_HEIGHT - 8), float(TILE_SIZE)))

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

        winners = self.game.win_history[-max_icons:]
        if not winners:
            return

        total_width = len(winners) * icon_size + max(0, len(winners) - 1) * icon_gap
        start_x = float(left) + (available_width - total_width) / 2.0
        for idx, player_id in enumerate(winners):
            center_x = start_x + icon_size / 2.0 + idx * (icon_size + icon_gap)
            if player_id is None:
                continue
            self._draw_player_icon(player_id, center_x, center_y, icon_size)

    def _draw_player_icon(self, player_id: str, center_x: float, center_y: float, size: float) -> None:
        style = PLAYER_STYLES.get(player_id, {})
        fill_color = style.get("render_fill", COLOR_DEEP_TEAL)
        outline_color = style.get("render_outline", COLOR_AQUA)
        bottom = center_y - size / 2.0
        left = center_x - size / 2.0
        arcade.draw_lbwh_rectangle_filled(left, bottom, size, size, outline_color)

        inset = max(1.0, round(CELL_INSET * (size / max(1.0, float(TILE_SIZE)))))
        inner_size = max(1.0, size - 2.0 * inset)
        arcade.draw_lbwh_rectangle_filled(
            left + inset,
            bottom + inset,
            inner_size,
            inner_size,
            fill_color,
        )
        if self.game.is_nn_controlled_player(player_id):
            marker_size = max(2.0, round(NN_CONTROL_MARKER_SIZE_PX * (size / max(1.0, float(TILE_SIZE)))))
            arcade.draw_lbwh_rectangle_filled(
                center_x - marker_size / 2.0,
                center_y - marker_size / 2.0,
                marker_size,
                marker_size,
                outline_color,
            )

    def _draw_two_tone_tile(self, top_left_x: float, top_left_y: float, outer_color, inner_color) -> None:
        draw_two_tone_tile(
            self.window_controller,
            top_left_x=float(top_left_x),
            top_left_y=float(top_left_y),
            size=float(TILE_SIZE),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(CELL_INSET),
        )

    def _draw_actor(self, actor: Actor, fill_color, outline_color, draw_nn_marker: bool = False) -> None:
        self._draw_two_tone_tile(
            top_left_x=actor.position.x - TILE_SIZE / 2,
            top_left_y=actor.position.y - TILE_SIZE / 2,
            outer_color=outline_color,
            inner_color=fill_color,
        )
        if draw_nn_marker:
            self._draw_nn_control_marker(actor, outline_color)

        draw_facing_indicator(
            self.window_controller,
            center_x=float(actor.position.x),
            center_y_top_left=float(actor.position.y),
            angle_degrees=float(actor.angle),
            length=float(TILE_SIZE // 2),
            color=COLOR_SOFT_WHITE,
            line_width=2.0,
        )

    def _draw_nn_control_marker(self, actor: Actor, color) -> None:
        draw_control_marker(
            self.window_controller,
            center_x=float(actor.position.x),
            center_y_top_left=float(actor.position.y),
            marker_size=float(NN_CONTROL_MARKER_SIZE_PX),
            color=color,
        )


class BaseGame:
    """Top-down free-for-all arena game logic."""

    def __init__(self, level: int = 1, show_game: bool = True):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.playable_height = float(self.height - BB_HEIGHT)
        self._ray_max_range = max(float(TILE_SIZE) * 10.0, min(float(self.width), self.playable_height) * 0.55)
        self._ray_step_size = max(0.75, float(TILE_SIZE) * 0.35)
        self.show_game = bool(show_game)
        self.frame_clock = ArcadeFrameClock()

        initial_level = max(MIN_LEVEL, min(int(level), MAX_LEVEL))
        initial_player_count = _num_players_for_level(initial_level)
        self.player_order = _resolve_player_order(initial_player_count)
        self.player_render_colors: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
        # Shots use the player's lighter accent color. This already includes reserved styles like P4.
        self.player_projectile_colors = {
            player_id: PLAYER_STYLES[player_id]["render_outline"]
            for player_id in PLAYER_STYLES
        }
        self.scores: dict[str, int] = {}
        self.win_history: list[str | None] = []
        self.player_control_modes: dict[str, str] = {}
        self._set_player_count(initial_player_count)

        self.players: list[Actor] = []
        self.players_by_id: dict[str, Actor] = {}
        self.scripted_players: list[Actor] = []
        self.target_states: dict[str, TargetState] = {}
        self.scripted_move_states: dict[str, ScriptedMoveState] = {}

        self.renderer = Renderer(
            game=self,
            width=self.width,
            height=self.height,
            title=WINDOW_TITLE,
            enabled=self.show_game,
        )
        self.window_controller = self.renderer.window_controller
        self.window = self.renderer.window

        self.level = level
        self.configure_level()
        self.reset()

    def close(self) -> None:
        self.renderer.close()

    def poll_events(self) -> None:
        self.renderer.poll_events()

    def draw_frame(self) -> None:
        self.renderer.draw_frame()

    def _non_scripted_control_mode(self) -> str:
        return CONTROL_MODE_HUMAN

    def is_nn_controlled_player(self, player_id: str) -> bool:
        return self.player_control_modes.get(player_id) == CONTROL_MODE_NN

    def configure_level(self) -> None:
        level = max(MIN_LEVEL, min(self.level, MAX_LEVEL))
        self.level = level
        settings = LEVEL_SETTINGS[level]
        self._set_player_count(_num_players_for_level(level))

        self.num_obstacles = settings["num_obstacles"]
        self.enemy_move_probability = settings["enemy_move_probability"]
        self.enemy_shot_error_choices = list(ENEMY_SHOT_ERROR_CHOICES)
        self.enemy_shoot_probability = settings["enemy_shoot_probability"]

    def _set_player_count(self, num_players: int) -> None:
        player_order = _resolve_player_order(num_players)
        if player_order == self.player_order and self.scores:
            return

        old_scores = dict(self.scores)
        self.player_order = player_order
        self.player_render_colors = {
            player_id: (
                PLAYER_STYLES[player_id]["render_fill"],
                PLAYER_STYLES[player_id]["render_outline"],
            )
            for player_id in self.player_order
        }
        non_scripted_mode = self._non_scripted_control_mode()
        self.player_control_modes = {
            player_id: (
                CONTROL_MODE_SCRIPTED
                if PLAYER_STYLES[player_id]["scripted"]
                else non_scripted_mode
            )
            for player_id in self.player_order
        }
        self.scores = {
            player_id: int(old_scores.get(player_id, 0))
            for player_id in self.player_order
        }
        for player_id in self.player_order:
            setattr(self, f"{player_id}_score", self.scores[player_id])

    def reset(self) -> None:
        spawn_positions = self._spawn_positions_by_player()

        self.players_by_id = {}
        for player_id in self.player_order:
            spawn_pos = spawn_positions[player_id]
            self.players_by_id[player_id] = Actor(
                spawn_pos,
                angle=self._sample_inner_facing_angle(spawn_pos),
                team=player_id,
            )

        self.players = [self.players_by_id[player_id] for player_id in self.player_order]
        self.player = self.players_by_id["P1"]
        # Backward-compatible aliases for older callers.
        self.enemy = self.players_by_id["P2"]
        self.enemy2 = self.players_by_id.get("P3")
        self.enemy3 = self.players_by_id.get("P4")
        self.scripted_players = [
            self.players_by_id[player_id]
            for player_id in self.player_order
            if PLAYER_STYLES[player_id]["scripted"]
        ]

        self.obstacles: list[Vec2] = []
        self.projectiles: list[dict[str, object]] = []
        self.frame_count = 0
        self.last_action_index = ACTION_STOP_MOVE
        self.frames_since_last_shot = SHOOT_COOLDOWN_FRAMES
        self.last_seen_enemy_frame = -EVENT_TIMER_NORMALIZATION_FRAMES
        self.target_states = {
            actor.team: TargetState()
            for actor in self.players
        }
        self.scripted_move_states = {
            actor.team: ScriptedMoveState(
                escape_frames_remaining=0,
                escape_offset_degrees=float(ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES[0]),
            )
            for actor in self.scripted_players
        }
        self._place_obstacles()

    def _spawn_y_bounds(self) -> tuple[float, float]:
        center_y = self.height / 2 - BB_HEIGHT // 2
        min_y = center_y - SPAWN_Y_OFFSET
        max_y = center_y + SPAWN_Y_OFFSET

        min_actor_y = TILE_SIZE / 2
        max_actor_y = self.height - BB_HEIGHT - TILE_SIZE / 2
        min_y = max(min_y, min_actor_y)
        max_y = min(max_y, max_actor_y)
        return min_y, max_y

    def _spawn_x_bounds(self) -> tuple[float, float]:
        center_x = self.width / 2
        min_x = center_x - SPAWN_Y_OFFSET
        max_x = center_x + SPAWN_Y_OFFSET

        min_actor_x = TILE_SIZE / 2
        max_actor_x = self.width - TILE_SIZE / 2
        min_x = max(min_x, min_actor_x)
        max_x = min(max_x, max_actor_x)
        return min_x, max_x

    def _spawn_bottom_strip_y(self) -> float:
        playable_height = self.height - BB_HEIGHT
        bottom_edge_y = playable_height - TILE_SIZE / 2
        bottom_padding = playable_height * PLAYER_SPAWN_X_RATIO
        return max(TILE_SIZE / 2, bottom_edge_y - bottom_padding)

    def _spawn_top_strip_y(self) -> float:
        top_edge_y = TILE_SIZE / 2
        top_padding = self.height * PLAYER_SPAWN_X_RATIO
        return min(self.height - BB_HEIGHT - TILE_SIZE / 2, top_edge_y + top_padding)

    def _active_spawn_areas(self) -> tuple[str, ...]:
        player_count = len(self.player_order)
        if player_count <= 0:
            return tuple()
        return SPAWN_AREA_ORDER[: min(player_count, len(SPAWN_AREA_ORDER))]

    def _sample_spawn_position(self, area: str) -> Vec2:
        min_y, max_y = self._spawn_y_bounds()
        min_x, max_x = self._spawn_x_bounds()

        if area == SPAWN_AREA_LEFT:
            return Vec2(self.width * PLAYER_SPAWN_X_RATIO, random.uniform(min_y, max_y))
        if area == SPAWN_AREA_RIGHT:
            return Vec2(self.width * ENEMY_SPAWN_X_RATIO, random.uniform(min_y, max_y))
        if area == SPAWN_AREA_BOTTOM:
            return Vec2(random.uniform(min_x, max_x), self._spawn_bottom_strip_y())
        if area == SPAWN_AREA_TOP:
            return Vec2(random.uniform(min_x, max_x), self._spawn_top_strip_y())
        raise ValueError(f"Unknown spawn area: {area}")

    def _spawn_positions_by_player(self) -> dict[str, Vec2]:
        area_order = list(self._active_spawn_areas())
        random.shuffle(area_order)
        positions: dict[str, Vec2] = {}
        for idx, player_id in enumerate(self.player_order):
            area = area_order[idx % len(area_order)]
            positions[player_id] = self._sample_spawn_position(area)
        return positions

    def _sample_inner_facing_angle(self, position: Vec2) -> float:
        arena_center = Vec2(self.width / 2.0, (self.height - BB_HEIGHT) / 2.0)
        to_center = arena_center - position
        if length_squared(to_center) == 0:
            base_angle = random.uniform(0.0, 360.0)
        else:
            base_angle = math.degrees(math.atan2(to_center.y, to_center.x))
        return (base_angle + random.uniform(-90.0, 90.0)) % 360.0

    def _player_attempts_translation(self) -> bool:
        if not self.player.is_alive:
            return False
        return self.player.move_intent_x != 0 or self.player.move_intent_y != 0

    def _reset_actor_velocities(self) -> None:
        for actor in self.players:
            actor.vx = 0.0
            actor.vy = 0.0

    @staticmethod
    def _set_actor_move_intent(actor: Actor, move_x: int, move_y: int) -> None:
        actor.move_intent_x = max(-1, min(1, int(move_x)))
        actor.move_intent_y = max(-1, min(1, int(move_y)))

    @staticmethod
    def _set_actor_aim_intent(actor: Actor, aim_intent: int) -> None:
        actor.aim_intent = max(-1, min(1, int(aim_intent)))

    def _apply_action_to_player_intents(self, action_index: int) -> bool:
        if action_index == ACTION_MOVE_UP:
            self._set_actor_move_intent(self.player, 0, 1)
            return False
        if action_index == ACTION_MOVE_DOWN:
            self._set_actor_move_intent(self.player, 0, -1)
            return False
        if action_index == ACTION_MOVE_LEFT:
            self._set_actor_move_intent(self.player, -1, 0)
            return False
        if action_index == ACTION_MOVE_RIGHT:
            self._set_actor_move_intent(self.player, 1, 0)
            return False
        if action_index == ACTION_STOP_MOVE:
            self._set_actor_move_intent(self.player, 0, 0)
            return False
        if action_index == ACTION_AIM_LEFT:
            self._set_actor_aim_intent(self.player, -1)
            return False
        if action_index == ACTION_AIM_RIGHT:
            self._set_actor_aim_intent(self.player, 1)
            return False
        if action_index == ACTION_SHOOT:
            projectile = self.player.shoot()
            if projectile:
                self.projectiles.append(projectile)
                return True
        return False

    def _human_action_pressed(self, action_index: int) -> bool:
        keys = HUMAN_ACTION_KEY_BINDINGS.get(int(action_index), ())
        return any(self.window_controller.is_key_down(key) for key in keys)

    def _resolve_human_action(self) -> int:
        move_up = self._human_action_pressed(ACTION_MOVE_UP)
        move_down = self._human_action_pressed(ACTION_MOVE_DOWN)
        move_left = self._human_action_pressed(ACTION_MOVE_LEFT)
        move_right = self._human_action_pressed(ACTION_MOVE_RIGHT)
        aim_left = self._human_action_pressed(ACTION_AIM_LEFT)
        aim_right = self._human_action_pressed(ACTION_AIM_RIGHT)
        shoot = self._human_action_pressed(ACTION_SHOOT)

        # Single discrete action per step: shoot > aim > movement > stop.
        if shoot:
            return ACTION_SHOOT
        if aim_left and not aim_right:
            return ACTION_AIM_LEFT
        if aim_right and not aim_left:
            return ACTION_AIM_RIGHT
        if move_up and not move_down:
            return ACTION_MOVE_UP
        if move_down and not move_up:
            return ACTION_MOVE_DOWN
        if move_left and not move_right:
            return ACTION_MOVE_LEFT
        if move_right and not move_left:
            return ACTION_MOVE_RIGHT
        return ACTION_STOP_MOVE

    def apply_player_action(self, action_index: int | None) -> None:
        if not self.player.is_alive:
            self.frames_since_last_shot += 1
            return

        control_mode = self.player_control_modes.get(self.player.team)
        # Human and NN-controlled player aim are per-step (non-sticky).
        if control_mode in (CONTROL_MODE_HUMAN, CONTROL_MODE_NN):
            self._set_actor_aim_intent(self.player, 0)
        if control_mode == CONTROL_MODE_HUMAN:
            # Match move_stop behavior when no WASD movement action is selected this frame.
            self._set_actor_move_intent(self.player, 0, 0)

        shot_fired = False
        if action_index is not None:
            self.last_action_index = int(action_index)
            shot_fired = self._apply_action_to_player_intents(self.last_action_index)

        movement = self.player.step_sticky_intents()
        self._update_actor_position(self.player, movement)

        if shot_fired:
            self.frames_since_last_shot = 0
        else:
            self.frames_since_last_shot += 1

    def _update_actor_position(self, actor: Actor, movement: Vec2) -> None:
        if not actor.is_alive:
            actor.vx = 0.0
            actor.vy = 0.0
            return

        previous_position = actor.position
        new_position = actor.position + movement
        actor_rect = rect_from_center(new_position, TILE_SIZE)
        if collides_with_square_arena(
            rect=actor_rect,
            obstacles=self.obstacles,
            tile_size=TILE_SIZE,
            arena_width=self.width,
            arena_height=self.height,
            bottom_bar_height=BB_HEIGHT,
        ):
            actor.vx = 0.0
            actor.vy = 0.0
            return

        for other in self.players:
            if other is actor or not other.is_alive:
                continue
            other_rect = rect_from_center(other.position, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                actor.vx = 0.0
                actor.vy = 0.0
                return

        actor.position = new_position
        actor.vx = float(actor.position.x - previous_position.x)
        actor.vy = float(actor.position.y - previous_position.y)

    def _would_collide(self, actor: Actor, movement: Vec2) -> bool:
        if not actor.is_alive:
            return True

        new_position = actor.position + movement
        actor_rect = rect_from_center(new_position, TILE_SIZE)
        if collides_with_square_arena(
            rect=actor_rect,
            obstacles=self.obstacles,
            tile_size=TILE_SIZE,
            arena_width=self.width,
            arena_height=self.height,
            bottom_bar_height=BB_HEIGHT,
        ):
            return True

        for other in self.players:
            if other is actor or not other.is_alive:
                continue
            other_rect = rect_from_center(other.position, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                return True
        return False

    def _point_blocked_for_ray(self, x: float, y: float) -> bool:
        if x < 0.0 or x >= float(self.width) or y < 0.0 or y >= float(self.height - BB_HEIGHT):
            return True
        for obstacle in self.obstacles:
            if (
                obstacle.x <= float(x) < obstacle.x + float(TILE_SIZE)
                and obstacle.y <= float(y) < obstacle.y + float(TILE_SIZE)
            ):
                return True
        return False

    def _ray_distance(self, angle_degrees: float) -> float:
        radians = math.radians(float(angle_degrees))
        return normalized_ray_first_hit(
            origin_x=float(self.player.position.x),
            origin_y=float(self.player.position.y),
            dir_x=math.cos(radians),
            dir_y=math.sin(radians),
            max_distance=self._ray_max_range,
            is_blocked=self._point_blocked_for_ray,
            step_size=self._ray_step_size,
            start_offset=float(TILE_SIZE) * 0.25,
        )

    @staticmethod
    def _normalize_elapsed_frames(
        frames: int,
        normalization_frames: int = EVENT_TIMER_NORMALIZATION_FRAMES,
    ) -> float:
        return min(1.0, max(0, frames) / max(1, normalization_frames))

    def _update_enemy_seen_timer(self, enemy_in_los: bool) -> float:
        if enemy_in_los:
            self.last_seen_enemy_frame = self.frame_count
        return self._normalize_elapsed_frames(self.frame_count - self.last_seen_enemy_frame)

    @staticmethod
    def _build_state_vector_from_features(feature_values: dict[str, float]) -> list[float]:
        return ordered_feature_vector(BANG_INPUT_FEATURE_NAMES, feature_values)

    def _place_obstacles(self) -> None:
        self.obstacles = []
        shapes = spawn_connected_random_walk_shapes(
            shape_count=self.num_obstacles,
            min_sections=MIN_OBSTACLE_SECTIONS,
            max_sections=MAX_OBSTACLE_SECTIONS,
            sample_start_fn=self._sample_valid_obstacle_start,
            neighbor_candidates_fn=self._neighbor_obstacle_candidates,
            is_candidate_valid_fn=self._is_valid_obstacle_tile,
        )
        for shape in shapes:
            self.obstacles.extend(shape)

    def _sample_valid_obstacle_start(self):
        for _ in range(OBSTACLE_START_ATTEMPTS):
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            point = Vec2(x, y)
            if self._is_valid_obstacle_tile(point, []):
                return point
        return None

    def _is_valid_obstacle_tile(self, tile: Vec2, pending_tiles) -> bool:
        if not (0 <= tile.x < self.width and 0 <= tile.y < self.height - BB_HEIGHT):
            return False
        if any(tile == existing for existing in self.obstacles) or any(tile == existing for existing in pending_tiles):
            return False
        if any(tile.distance(actor.position) < SAFE_RADIUS for actor in self.players if actor.is_alive):
            return False
        return True

    @staticmethod
    def _neighbor_obstacle_candidates(tile: Vec2) -> list[Vec2]:
        return [
            Vec2(tile.x - TILE_SIZE, tile.y),
            Vec2(tile.x + TILE_SIZE, tile.y),
            Vec2(tile.x, tile.y - TILE_SIZE),
            Vec2(tile.x, tile.y + TILE_SIZE),
        ]

    @staticmethod
    def _move_vector_for_angle(angle_degrees: float) -> Vec2:
        return rotate_degrees(Vec2(1, 0), angle_degrees) * PLAYER_MOVE_SPEED

    def _move_actor_in_direction(self, actor: Actor, angle_degrees: float) -> bool:
        previous_position = actor.position
        movement = self._move_vector_for_angle(angle_degrees)
        self._update_actor_position(actor, movement)
        return length_squared(actor.position - previous_position) > 0

    def _alive_opponents(self, actor: Actor) -> list[Actor]:
        return [other for other in self.players if other is not actor and other.is_alive]

    def _resolve_alive_target(self, actor: Actor, target_id: str | None) -> Actor | None:
        if target_id is None:
            return None
        target = self.players_by_id.get(target_id)
        if target is None or target is actor or not target.is_alive:
            return None
        return target

    def _has_clear_path_between(self, actor: Actor, target: Actor) -> bool:
        return not square_obstacle_between_points(
            point_a=actor.position,
            point_b=target.position,
            obstacles=self.obstacles,
            tile_size=TILE_SIZE,
        )

    @staticmethod
    def _is_actor_aimed_at_target(actor: Actor, target: Actor) -> bool:
        to_target = target.position - actor.position
        if length_squared(to_target) == 0:
            return True
        target_angle = math.degrees(math.atan2(to_target.y, to_target.x))
        relative = normalize_angle_degrees(target_angle - actor.angle)
        return abs(relative) <= AIM_TOLERANCE_DEGREES

    def _nearest_target(self, actor: Actor, candidates: list[Actor], require_clear_path: bool) -> Actor | None:
        filtered = candidates
        if require_clear_path:
            filtered = [candidate for candidate in candidates if self._has_clear_path_between(actor, candidate)]
        if not filtered:
            return None
        return min(filtered, key=lambda candidate: actor.position.distance(candidate.position))

    def _reset_player_target_tracking(self, target: Actor | None) -> None:
        self.last_seen_enemy_frame = (
            self.frame_count if target is not None and self.has_line_of_sight(target) else -EVENT_TIMER_NORMALIZATION_FRAMES
        )

    def _set_target_state(
        self,
        actor: Actor,
        state: TargetState,
        target: Actor | None,
        policy: dict[str, float | int],
    ) -> None:
        previous_target_id = state.target_id
        state.target_id = target.team if target is not None else None
        state.target_lost_frames = 0
        state.target_switch_cooldown = (
            random.randint(
                int(policy["hold_min_frames"]),
                int(policy["hold_max_frames"]),
            )
            if target is not None
            else 0
        )
        if actor is self.player and state.target_id != previous_target_id:
            self._reset_player_target_tracking(target)

    def _select_target(
        self,
        actor: Actor,
        policy: dict[str, float | int],
        cache_by_frame: bool,
    ) -> Actor | None:
        state = self.target_states.setdefault(actor.team, TargetState())
        if cache_by_frame and state.last_update_frame == self.frame_count:
            return self._resolve_alive_target(actor, state.target_id)
        state.last_update_frame = self.frame_count

        if not actor.is_alive:
            self._set_target_state(actor, state, None, policy)
            return None

        candidates = self._alive_opponents(actor)
        if not candidates:
            self._set_target_state(actor, state, None, policy)
            return None

        current = self._resolve_alive_target(actor, state.target_id)
        if current is not None and self._has_clear_path_between(actor, current):
            state.target_lost_frames = 0
        elif current is not None:
            state.target_lost_frames += 1
        else:
            state.target_lost_frames = 0

        nearest_visible = self._nearest_target(actor, candidates, require_clear_path=True)
        nearest_any = self._nearest_target(actor, candidates, require_clear_path=False)
        preferred = nearest_visible if nearest_visible is not None else nearest_any

        if state.target_switch_cooldown > 0:
            state.target_switch_cooldown -= 1

        should_switch = current is None
        if current is not None and state.target_lost_frames > int(policy["max_lost_frames"]):
            should_switch = True

        if (
            not should_switch
            and current is not None
            and preferred is not None
            and preferred is not current
            and state.target_switch_cooldown <= 0
        ):
            current_visible = self._has_clear_path_between(actor, current)
            preferred_visible = self._has_clear_path_between(actor, preferred)
            current_distance = actor.position.distance(current.position)
            preferred_distance = actor.position.distance(preferred.position)
            if preferred_visible and not current_visible:
                should_switch = True
            elif preferred_distance < current_distance * float(policy["switch_distance_ratio"]):
                should_switch = True

        if (
            not should_switch
            and current is not None
            and state.target_switch_cooldown <= 0
            and random.random() < float(policy["random_switch_prob"])
        ):
            alternatives = [candidate for candidate in candidates if candidate is not current]
            if alternatives:
                preferred = random.choice(alternatives)
                should_switch = True

        if should_switch:
            next_target = preferred if preferred is not None else random.choice(candidates)
            self._set_target_state(actor, state, next_target, policy)
            current = next_target

        return current

    def _get_player_target(self) -> Actor | None:
        return self._select_target(
            actor=self.player,
            policy=PLAYER_TARGET_POLICY,
            cache_by_frame=True,
        )

    def _scripted_desired_move_angle(self, actor: Actor, target: Actor, angle_to_target: float) -> float:
        distance = actor.position.distance(target.position)
        if distance < SAFE_RADIUS * 0.9:
            return (angle_to_target + 180.0) % 360.0
        if distance > SAFE_RADIUS * 1.8:
            return angle_to_target
        if random.random() < 0.35:
            return (angle_to_target + random.choice((90.0, -90.0))) % 360.0
        return angle_to_target

    def _scripted_move_state(self, actor: Actor) -> ScriptedMoveState:
        return self.scripted_move_states.setdefault(
            actor.team,
            ScriptedMoveState(
                escape_frames_remaining=0,
                escape_offset_degrees=float(ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES[0]),
            ),
        )

    @staticmethod
    def _turn_toward_angle(current_angle: float, target_angle: float, max_step_degrees: float) -> float:
        delta = normalize_angle_degrees(float(target_angle) - float(current_angle))
        max_step = max(0.0, float(max_step_degrees))
        step = max(-max_step, min(max_step, float(delta)))
        return (float(current_angle) + float(step)) % 360.0

    def _available_escape_offsets(self, actor: Actor, angle_to_target: float) -> list[float]:
        free_offsets: list[float] = []
        for offset in ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES:
            escape_angle = (angle_to_target + float(offset)) % 360.0
            candidate_move = self._move_vector_for_angle(escape_angle)
            if not self._would_collide(actor, candidate_move):
                free_offsets.append(float(offset))
        return free_offsets

    def _pick_random_escape_offset(self, actor: Actor, angle_to_target: float) -> float | None:
        free_offsets = self._available_escape_offsets(actor, angle_to_target)
        if not free_offsets:
            return None
        return random.choice(free_offsets)

    def _attempt_scripted_escape_move(
        self,
        actor: Actor,
        angle_to_target: float,
        move_state: ScriptedMoveState,
    ) -> bool:
        escape_angle = (angle_to_target + move_state.escape_offset_degrees) % 360.0
        if self._move_actor_in_direction(actor, escape_angle):
            return True

        new_offset = self._pick_random_escape_offset(actor, angle_to_target)
        if new_offset is None:
            return False
        move_state.escape_offset_degrees = new_offset
        alternate_angle = (angle_to_target + move_state.escape_offset_degrees) % 360.0
        return self._move_actor_in_direction(actor, alternate_angle)

    def _step_scripted_movement(self, actor: Actor, target: Actor, angle_to_target: float) -> None:
        move_state = self._scripted_move_state(actor)
        if move_state.escape_frames_remaining <= 0 and random.random() >= self.enemy_move_probability:
            return
        moved = False

        if move_state.escape_frames_remaining > 0:
            moved = self._attempt_scripted_escape_move(actor, angle_to_target, move_state)
            move_state.escape_frames_remaining -= 1
        else:
            move_angle = self._scripted_desired_move_angle(actor, target, angle_to_target)
            moved = self._move_actor_in_direction(actor, move_angle)
            if not moved:
                escape_offset = self._pick_random_escape_offset(actor, angle_to_target)
                if escape_offset is not None:
                    move_state.escape_offset_degrees = escape_offset
                    move_state.escape_frames_remaining = ENEMY_ESCAPE_FOLLOW_FRAMES
                    moved = self._attempt_scripted_escape_move(actor, angle_to_target, move_state)
                    if moved:
                        move_state.escape_frames_remaining -= 1

        if not moved:
            move_state.escape_frames_remaining = 0

    def _step_scripted_actor(self, actor: Actor) -> None:
        if not actor.is_alive:
            return

        target = self._select_target(
            actor=actor,
            policy=SCRIPTED_TARGET_POLICY,
            cache_by_frame=False,
        )
        if target is None:
            return

        to_target = target.position - actor.position
        if length_squared(to_target) == 0:
            angle_to_target = actor.angle
        else:
            angle_to_target = math.degrees(math.atan2(to_target.y, to_target.x)) % 360

        actor.angle = self._turn_toward_angle(
            current_angle=float(actor.angle),
            target_angle=float(angle_to_target),
            max_step_degrees=float(AIM_RATE_PER_STEP),
        )

        self._step_scripted_movement(actor, target, angle_to_target)

        shoot_probability = self.enemy_shoot_probability
        if self._has_clear_path_between(actor, target):
            shoot_probability = min(1.0, shoot_probability * 1.25)
        if random.random() < shoot_probability:
            original_angle = float(actor.angle)
            aim_error = random.choice(self.enemy_shot_error_choices)
            actor.angle = (float(angle_to_target) + float(aim_error)) % 360.0
            projectile = actor.shoot()
            actor.angle = original_angle
            if projectile:
                self.projectiles.append(projectile)

    def _step_scripted_players(self) -> None:
        for actor in self.scripted_players:
            self._step_scripted_actor(actor)

    def _step_projectiles(self):
        events = {"player_kills": 0, "player_killed_by": None}
        next_projectiles = []

        for projectile in self.projectiles:
            projectile["pos"] += projectile["velocity"]
            projectile_rect = rect_from_center(projectile["pos"], PROJECTILE_HITBOX_SIZE)
            if collides_with_square_arena(
                rect=projectile_rect,
                obstacles=self.obstacles,
                tile_size=TILE_SIZE,
                arena_width=self.width,
                arena_height=self.height,
                bottom_bar_height=BB_HEIGHT,
            ):
                continue

            owner_id = str(projectile["owner"])
            colliding_targets = []
            for target in self.players:
                if not target.is_alive or target.team == owner_id:
                    continue
                target_rect = rect_from_center(target.position, TILE_SIZE)
                if projectile_rect.colliderect(target_rect):
                    colliding_targets.append(target)

            if colliding_targets:
                target = min(colliding_targets, key=lambda candidate: candidate.position.distance(projectile["pos"]))
                eliminated = bool(target.take_hit(1))
                if owner_id == self.player.team and eliminated:
                    events["player_kills"] += 1
                if target is self.player and not self.player.is_alive:
                    events["player_killed_by"] = owner_id
                continue

            next_projectiles.append(projectile)

        self.projectiles = next_projectiles
        return events

    def _nearest_hostile_projectile(self) -> dict[str, object] | None:
        hostile_projectiles = [p for p in self.projectiles if p["owner"] != self.player.team]
        if not hostile_projectiles:
            return None
        return min(
            hostile_projectiles,
            key=lambda projectile: self.player.position.distance(projectile["pos"]),
        )

    def _projectile_in_trajectory(self, projectile: dict[str, object]) -> bool:
        to_player = self.player.position - projectile["pos"]
        if length_squared(to_player) == 0:
            return True
        projectile_dir = projectile["velocity"].normalize()
        return projectile_dir.dot(to_player.normalize()) > PROJECTILE_TRAJECTORY_DOT_THRESHOLD

    def is_player_in_projectile_trajectory(self) -> bool:
        for projectile in self.projectiles:
            if projectile["owner"] == self.player.team:
                continue
            if self._projectile_in_trajectory(projectile):
                return True
        return False

    def has_line_of_sight(self, target: Actor | None = None) -> bool:
        if target is None:
            target = self._get_player_target()
        if target is None:
            return False
        return self._is_actor_aimed_at_target(self.player, target) and self._has_clear_path_between(self.player, target)

    def _nearest_alive_opponent_to_player(self) -> Actor | None:
        opponents = [actor for actor in self.players if actor is not self.player and actor.is_alive]
        if not opponents:
            return None
        return min(opponents, key=lambda actor: self.player.position.distance(actor.position))

    def _engagement_potential(self) -> float:
        target = self._nearest_alive_opponent_to_player()
        if target is None:
            return 0.0
        dist_scale = max(1.0, max(float(self.width), float(self.height - BB_HEIGHT)))
        tgt_dist_norm = clip_unit(self.player.position.distance(target.position) / dist_scale)
        tgt_in_los = 1.0 if self.has_line_of_sight(target) else 0.0
        return float(tgt_in_los - tgt_dist_norm)

    def _hazard_potential(self) -> float:
        nearest_projectile = self._nearest_hostile_projectile()
        if nearest_projectile is None:
            haz_dist_norm = 1.0
            haz_in_trajectory = 0.0
        else:
            dist_scale = max(1.0, max(float(self.width), float(self.height - BB_HEIGHT)))
            haz_dist_norm = clip_unit(self.player.position.distance(nearest_projectile["pos"]) / dist_scale)
            haz_in_trajectory = 1.0 if self._projectile_in_trajectory(nearest_projectile) else 0.0
        return float(haz_dist_norm - 1.5 * haz_in_trajectory)

    def get_state_vector(self) -> list[float]:
        target = self._get_player_target()
        pos_scale_x = max(1.0, float(self.width))
        pos_scale_y = max(1.0, float(self.height - BB_HEIGHT))
        dist_scale = max(1.0, max(float(self.width), float(self.height)))
        actor_vel_scale = max(1.0, float(PLAYER_MOVE_SPEED))

        if target is None:
            tgt_dx = 0.0
            tgt_dy = 0.0
            tgt_dvx = 0.0
            tgt_dvy = 0.0
            tgt_dist = 1.0
            tgt_in_los = 0.0
        else:
            to_target = target.position - self.player.position
            tgt_dx = clip_signed(to_target.x / pos_scale_x)
            tgt_dy = clip_signed(to_target.y / pos_scale_y)
            tgt_dvx = clip_signed((float(target.vx) - float(self.player.vx)) / actor_vel_scale)
            tgt_dvy = clip_signed((float(target.vy) - float(self.player.vy)) / actor_vel_scale)
            tgt_dist = clip_unit(self.player.position.distance(target.position) / dist_scale)
            tgt_in_los = 1.0 if self.has_line_of_sight(target) else 0.0

        time_since_last_seen_enemy = self._update_enemy_seen_timer(bool(tgt_in_los))
        nearest_projectile = self._nearest_hostile_projectile()
        hazard_vel_scale = max(1.0, float(PROJECTILE_SPEED) + actor_vel_scale)
        if nearest_projectile is None:
            haz_dx = 0.0
            haz_dy = 0.0
            haz_dvx = 0.0
            haz_dvy = 0.0
            haz_dist = 1.0
            haz_in_trajectory = 0.0
        else:
            projectile_pos = nearest_projectile["pos"]
            projectile_vel = nearest_projectile["velocity"]
            rel_pos = projectile_pos - self.player.position
            rel_vel = projectile_vel - Vec2(float(self.player.vx), float(self.player.vy))
            haz_dx = clip_signed(rel_pos.x / pos_scale_x)
            haz_dy = clip_signed(rel_pos.y / pos_scale_y)
            haz_dvx = clip_signed(rel_vel.x / hazard_vel_scale)
            haz_dvy = clip_signed(rel_vel.y / hazard_vel_scale)
            haz_dist = clip_unit(self.player.position.distance(projectile_pos) / dist_scale)
            haz_in_trajectory = 1.0 if self._projectile_in_trajectory(nearest_projectile) else 0.0

        ray_fwd = self._ray_distance(self.player.angle)
        ray_left = self._ray_distance(self.player.angle - 90.0)
        ray_right = self._ray_distance(self.player.angle + 90.0)
        ray_back = self._ray_distance(self.player.angle + 180.0)
        player_angle_radians = math.radians(self.player.angle)
        player_angle_sin = float(math.sin(player_angle_radians))
        player_angle_cos = float(math.cos(player_angle_radians))

        tgt_vec_x = float(tgt_dx)
        tgt_vec_y = float(tgt_dy)
        tgt_norm = math.sqrt((tgt_vec_x * tgt_vec_x) + (tgt_vec_y * tgt_vec_y))
        if tgt_norm <= 1e-8:
            tgt_rel_angle_sin = 0.0
            tgt_rel_angle_cos = 1.0
        else:
            tgt_rel_angle_cos = clip_signed(
                ((player_angle_cos * tgt_vec_x) + (player_angle_sin * tgt_vec_y)) / tgt_norm
            )
            tgt_rel_angle_sin = clip_signed(
                ((player_angle_cos * tgt_vec_y) - (player_angle_sin * tgt_vec_x)) / tgt_norm
            )

        self_shot_cd_norm = clip_unit(float(self.player.cooldown_frames) / max(1, SHOOT_COOLDOWN_FRAMES))
        self_tgt_seen_norm = clip_unit(float(time_since_last_seen_enemy))

        feature_values = {
            "self_angle_sin": player_angle_sin,
            "self_angle_cos": player_angle_cos,
            "self_move_intent_x": float(self.player.move_intent_x),
            "self_move_intent_y": float(self.player.move_intent_y),
            "self_shot_cd_norm": float(self_shot_cd_norm),
            "self_tgt_seen_norm": float(self_tgt_seen_norm),
            "ray_fwd": float(ray_fwd),
            "ray_left": float(ray_left),
            "ray_right": float(ray_right),
            "ray_back": float(ray_back),
            "tgt_dx": float(tgt_dx),
            "tgt_dy": float(tgt_dy),
            "tgt_dvx": float(tgt_dvx),
            "tgt_dvy": float(tgt_dvy),
            "tgt_dist": float(tgt_dist),
            "tgt_in_los": float(tgt_in_los),
            "tgt_rel_angle_sin": float(tgt_rel_angle_sin),
            "tgt_rel_angle_cos": float(tgt_rel_angle_cos),
            "haz_dx": float(haz_dx),
            "haz_dy": float(haz_dy),
            "haz_dvx": float(haz_dvx),
            "haz_dvy": float(haz_dvy),
            "haz_dist": float(haz_dist),
            "haz_in_trajectory": float(haz_in_trajectory),
        }
        return self._build_state_vector_from_features(feature_values)

    def _tick_players(self) -> None:
        for actor in self.players:
            actor.tick()

    def _last_alive_player(self) -> Actor | None:
        alive_players = [actor for actor in self.players if actor.is_alive]
        if len(alive_players) == 1:
            return alive_players[0]
        return None

    def is_player_last_alive(self) -> bool:
        return self.player.is_alive and self._last_alive_player() is self.player

    def _increment_score(self, player_id: str) -> None:
        if player_id not in self.scores:
            return
        self.scores[player_id] += 1
        self.win_history.append(player_id)
        setattr(self, f"{player_id}_score", self.scores[player_id])

    def _record_round_draw(self) -> None:
        self.win_history.append(None)


class HumanGame(BaseGame):
    """Human-play mode."""

    def __init__(self, show_game: bool = True, level: int = 1):
        super().__init__(level=int(level), show_game=show_game)

    def play_step(self) -> None:
        self.frame_count += 1
        self.poll_events()
        self._reset_actor_velocities()

        action = None
        if self.player.is_alive:
            action = self._resolve_human_action()
        else:
            self._set_actor_move_intent(self.player, 0, 0)
            self._set_actor_aim_intent(self.player, 0)

        self.apply_player_action(action)
        self._step_scripted_players()
        self._step_projectiles()
        self._tick_players()

        winner = self._last_alive_player()
        if winner is not None:
            self._increment_score(winner.team)
            self.reset()
        elif self.frame_count >= MAX_EPISODE_STEPS:
            self._record_round_draw()
            self.reset()

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else 0)


class TrainingGame(BaseGame):
    """Environment used by DQN training."""

    def _non_scripted_control_mode(self) -> str:
        return CONTROL_MODE_NN

    def __init__(self, level: int = 1, show_game: bool = True, end_on_player_death: bool = True):
        self.end_on_player_death = bool(end_on_player_death)
        self.player_loss_recorded = False
        super().__init__(level=level, show_game=show_game)

    def reset(self) -> None:
        super().reset()
        self.player_loss_recorded = False

    def play_step(self, action: list[int]):
        self.frame_count += 1
        action_index = action.index(1) if 1 in action else 0
        self.poll_events()
        self._reset_actor_velocities()
        phi_eng_prev = float(self._engagement_potential())
        phi_haz_prev = float(self._hazard_potential())

        self.apply_player_action(action_index)
        self._step_scripted_players()
        projectile_events = self._step_projectiles()

        self._tick_players()

        phi_eng_next = float(self._engagement_potential())
        phi_haz_next = float(self._hazard_potential())

        reward = float(PENALTY_STEP)
        reward_breakdown = {
            "step.penalty_step": float(PENALTY_STEP),
            "event.reward_kill": 0.0,
            "progress.engagement_shape": 0.0,
            "progress.hazard_shape": 0.0,
            "outcome.reward_win": 0.0,
            "outcome.penalty_lose": 0.0,
        }

        engagement_shape = float(
            signed_potential_shaping(
                phi_prev=phi_eng_prev,
                phi_next=phi_eng_next,
                scale=float(ENGAGEMENT_SCALE),
                clip_abs=float(ENGAGEMENT_CLIP),
            )
        )
        hazard_shape = float(
            signed_potential_shaping(
                phi_prev=phi_haz_prev,
                phi_next=phi_haz_next,
                scale=float(HAZARD_SCALE),
                clip_abs=float(HAZARD_CLIP),
            )
        )
        reward += engagement_shape
        reward += hazard_shape
        reward_breakdown["progress.engagement_shape"] = engagement_shape
        reward_breakdown["progress.hazard_shape"] = hazard_shape

        player_kills = int(projectile_events["player_kills"])
        if player_kills > 0:
            kill_reward = float(REWARD_KILL) * float(player_kills)
            reward += kill_reward
            reward_breakdown["event.reward_kill"] = kill_reward

        player_just_died = (not self.player.is_alive) and (not self.player_loss_recorded)
        if player_just_died:
            self.player_loss_recorded = True

        done = False
        timed_out = False
        winner = self._last_alive_player()
        player_won = self.player.is_alive and winner is self.player
        player_lost_match = (not self.player.is_alive) or (winner is not None and winner is not self.player)
        if self.end_on_player_death and not self.player.is_alive:
            done = True
        elif player_won:
            done = True
        elif (not self.end_on_player_death) and (winner is not None):
            done = True
        elif self.frame_count >= MAX_EPISODE_STEPS:
            done = True
            timed_out = True

        if done:
            if player_won:
                reward += float(REWARD_WIN)
                reward_breakdown["outcome.reward_win"] = float(REWARD_WIN)
            elif player_lost_match:
                reward += float(PENALTY_LOSE)
                reward_breakdown["outcome.penalty_lose"] = float(PENALTY_LOSE)

        if done:
            if self.end_on_player_death and not self.player.is_alive:
                killer_id = projectile_events.get("player_killed_by")
                if isinstance(killer_id, str) and killer_id in self.scores:
                    self._increment_score(killer_id)
                else:
                    if winner is not None:
                        self._increment_score(winner.team)
            else:
                if winner is not None:
                    self._increment_score(winner.team)
            if timed_out and winner is None:
                self._record_round_draw()

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        return reward, done, reward_breakdown


class BangEnv(Env):
    """Env adapter exposing Bang through the shared interface."""

    INPUT_FEATURE_NAMES = tuple(BANG_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(BANG_ACTION_NAMES)
    OBS_DIM = int(BANG_OBS_DIM)
    ACT_DIM = int(BANG_ACT_DIM)
    REWARD_COMPONENT_ORDER = ("W", "L", "K", "E", "D", "S")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_win": "W",
        "outcome.penalty_lose": "L",
        "event.reward_kill": "K",
        "progress.engagement_shape": "E",
        "progress.hazard_shape": "D",
        "step.penalty_step": "S",
    }

    def __init__(
        self,
        mode: str = "train",
        render: bool = False,
        *,
        level: int | None = None,
        end_on_player_death: bool | None = None,
    ) -> None:
        self.mode = str(mode)
        show_game = bool(render)
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
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        if self.mode == "human":
            self.game = HumanGame(show_game=show_game, level=int(self._current_level))
            return

        level = int(self._current_level)
        if end_on_player_death is None:
            end_on_player_death = self.mode == "train"

        self._current_level = int(level)
        self.game = TrainingGame(
            level=int(level),
            show_game=show_game,
            end_on_player_death=bool(end_on_player_death),
        )
        self._apply_level_settings(int(self._current_level))

    def _apply_level_settings(self, level: int) -> None:
        if not hasattr(self, "game"):
            return
        game_level = int(max(MIN_LEVEL, min(int(level), MAX_LEVEL)))
        self.game.level = int(game_level)
        self.game.configure_level()

    @staticmethod
    def _action_to_one_hot(action_idx: int) -> list[int]:
        one_hot = [0] * int(BangEnv.ACT_DIM)
        action = max(0, min(int(action_idx), len(one_hot) - 1))
        one_hot[action] = 1
        return one_hot

    @staticmethod
    def _obs_from_state_vector(state_vector: list[float]) -> np.ndarray:
        obs = np.asarray(state_vector, dtype=np.float32)
        assert len(obs) == int(BangEnv.OBS_DIM)
        if obs.shape != (int(BangEnv.OBS_DIM),):
            raise RuntimeError(f"Bang observation expected {BangEnv.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def reset(self) -> np.ndarray:
        if self.mode == "train":
            self._apply_level_settings(int(self._current_level))
        self.game.reset()
        self._episode_reward_components.reset()
        return self._obs_from_state_vector(self.game.get_state_vector())

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.mode == "human":
            self.game.play_step()
            obs = self._obs_from_state_vector(self.game.get_state_vector())
            return obs, 0.0, False, {"level": int(getattr(self.game, "level", 1)), "success": 0}

        action_idx = int(action)
        reward, done, reward_breakdown = self.game.play_step(self._action_to_one_hot(action_idx))
        self._episode_reward_components.add_from_mapping(reward_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)
        obs = self._obs_from_state_vector(self.game.get_state_vector())
        episode_level = int(self._current_level) if self.mode == "train" else int(getattr(self.game, "level", 1))
        win = bool(done and self.game.is_player_last_alive())
        success = 1 if win else 0
        info: dict[str, object] = {
            "reward_breakdown": reward_breakdown,
            "win": bool(win),
            "success": int(success) if done else 0,
            "level": int(episode_level),
            "level_changed": False,
        }
        if done:
            info["reward_components"] = self._episode_reward_components.totals()
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(success)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(success),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )
            info["level_changed"] = bool(level_changed)
        return obs, float(reward), bool(done), info

    def render(self) -> None:
        # Bang self-renders inside play_step when show_game is enabled.
        return None

    def close(self) -> None:
        self.game.close()

