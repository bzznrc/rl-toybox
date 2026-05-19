"""Bang core gameplay, rendering, and game modes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

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
    COLOR_SAND,
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
from core.ghost_overlay import update_ghost_overlay_toggle
from core.io_schema import (
    clip_signed,
    clip_unit,
    normalized_ray_first_hit,
    ordered_feature_vector,
    signed_potential_shaping,
)
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_control_marker,
    draw_facing_indicator,
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    status_icon_inset,
    draw_status_square_icon,
    draw_two_tone_cell,
    spawn_connected_random_walk_shapes,
    status_icon_size,
)
from core.ray_viz import draw_player_rays
from core.rewards import RewardBreakdown
from core.shared_config import (
    BB_HEIGHT,
    CELL_INSET,
    FPS,
    NN_CONTROL_MARKER_SIZE_PX,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TRAINING_FPS,
)

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
    BANG_MODE_CHOICES,
    BANG_MODE_LABELS,
    BANG_MODE_SETTINGS,
    CURRICULUM_PROMOTION,
    DEFAULT_BANG_MODE,
    ENEMY_HIDDEN_URGENCY_FRAMES,
    ENEMY_MOVE_COMMIT_FRAMES,
    ENEMY_RECENT_POSITION_MEMORY,
    ENEMY_RECENT_POSITION_PENALTY,
    ENEMY_SHOT_ERROR_CHOICES,
    ENEMY_SPAWN_X_RATIO,
    EVENT_TIMER_NORMALIZATION_FRAMES,
    INPUT_FEATURE_NAMES as BANG_INPUT_FEATURE_NAMES,
    OBS_DIM as BANG_OBS_DIM,
    ACT_DIM as BANG_ACT_DIM,
    LEVEL_SETTINGS,
    MAX_EPISODE_STEPS,
    MAX_LEVEL,
    MAX_OBSTACLE_SECTIONS,
    MIN_LEVEL,
    MIN_OBSTACLE_SECTIONS,
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
    SHOOT_COOLDOWN_FRAMES,
    SHOW_GHOST_OVERLAY,
    SPAWN_Y_OFFSET,
    WINDOW_TITLE,
)
from core.runtime import (
    ArcadeSquareObstacleField,
    Vec2,
    heading_to_vector,
    length_squared,
    normalize_angle_degrees,
    rect_from_center,
    rotate_degrees,
)
from core.utils import resolve_play_level


CONTROLLED_PLAYER_ID = "P1"
ALL_PLAYER_ORDER = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
validate_curriculum_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
)


@dataclass(frozen=True)
class BangModeSpec:
    key: str
    label: str
    player_order: tuple[str, ...]
    team_by_player: dict[str, int]
    controlled_team_size: int

    @property
    def team_order(self) -> tuple[int, ...]:
        ordered: list[int] = []
        for player_id in self.player_order:
            team_id = int(self.team_by_player[str(player_id)])
            if team_id not in ordered:
                ordered.append(team_id)
        return tuple(ordered)

    @property
    def score_id_by_team(self) -> dict[int, str]:
        score_ids: dict[int, str] = {}
        for player_id in self.player_order:
            team_id = int(self.team_by_player[str(player_id)])
            score_ids.setdefault(team_id, str(player_id))
        return score_ids

    @property
    def controlled_player_order(self) -> tuple[str, ...]:
        controlled: list[str] = []
        for player_id in self.player_order:
            if int(self.team_by_player[str(player_id)]) == 0:
                controlled.append(str(player_id))
        return tuple(controlled[: int(self.controlled_team_size)])


def _build_bang_mode_spec(mode: str) -> BangModeSpec:
    settings = dict(BANG_MODE_SETTINGS[str(mode)])
    max_players = int(settings["max_players"])
    team_sizes = [int(size) for size in list(settings["team_sizes"])]
    if sum(team_sizes) != max_players:
        raise ValueError(f"Bang mode '{mode}' team_sizes must sum to max_players.")
    player_order = tuple(ALL_PLAYER_ORDER[:max_players])
    team_by_player: dict[str, int] = {}
    cursor = 0
    for team_id, team_size in enumerate(team_sizes):
        for _ in range(int(team_size)):
            team_by_player[player_order[cursor]] = int(team_id)
            cursor += 1
    return BangModeSpec(
        key=str(mode),
        label=str(settings.get("display_name", BANG_MODE_LABELS[str(mode)])),
        player_order=player_order,
        team_by_player=team_by_player,
        controlled_team_size=int(settings.get("controlled_team_size", 1)),
    )


BANG_MODE_SPECS = {
    mode: _build_bang_mode_spec(mode)
    for mode in BANG_MODE_CHOICES
}

for _level in range(int(MIN_LEVEL), int(MAX_LEVEL) + 1):
    _level_settings = dict(LEVEL_SETTINGS[int(_level)])
    _active_enemies = dict(_level_settings.get("active_enemies", {}))
    _configured_modes = {str(mode) for mode in _active_enemies}
    _expected_modes = set(BANG_MODE_CHOICES)
    if _configured_modes != _expected_modes:
        raise ValueError(
            f"Bang LEVEL_SETTINGS[{int(_level)}]['active_enemies'] must define every Bang mode."
        )
    for _mode, _count in _active_enemies.items():
        _max_enemies = sum(
            1
            for _player_id, _team_id in BANG_MODE_SPECS[str(_mode)].team_by_player.items()
            if int(_team_id) != 0
        )
        if not (0 <= int(_count) <= int(_max_enemies)):
            raise ValueError(
                f"Bang LEVEL_SETTINGS[{int(_level)}]['active_enemies'][{str(_mode)!r}] must be 0..{int(_max_enemies)}."
            )


def _resolve_bang_mode(value: str | None) -> str:
    mode_key = str(DEFAULT_BANG_MODE if value is None else value).strip().lower()
    mode_key = mode_key.replace("-", "_").replace(" ", "_")
    if mode_key not in BANG_MODE_CHOICES:
        valid = ", ".join(BANG_MODE_CHOICES)
        raise ValueError(f"Unsupported Bang mode '{value}'. Expected one of: {valid}.")
    return mode_key


TEAM_RENDER_STYLES = {
    0: (COLOR_DEEP_TEAL, COLOR_AQUA),
    1: (COLOR_BRICK_RED, COLOR_CORAL),
    2: (COLOR_NAVY, COLOR_BLUE),
    3: (COLOR_DEEP_PURPLE, COLOR_PURPLE),
}
SPAWN_AREA_LEFT = "left_column"
SPAWN_AREA_RIGHT = "right_column"
SPAWN_AREA_BOTTOM = "bottom_strip"
SPAWN_AREA_TOP = "top_strip"
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
    planned_move_angle: float | None = None
    planned_target_id: str | None = None
    commit_frames_remaining: int = 0
    blocked_frames: int = 0
    hidden_frames: int = 0
    recent_position_keys: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class ScriptedMoveOption:
    name: str
    angle_offset_degrees: float | None


SCRIPTED_MOVE_OPTIONS = (
    ScriptedMoveOption("hold", None),
    ScriptedMoveOption("advance", 0.0),
    ScriptedMoveOption("advance_left", 45.0),
    ScriptedMoveOption("advance_right", -45.0),
    ScriptedMoveOption("strafe_left", 90.0),
    ScriptedMoveOption("strafe_right", -90.0),
    ScriptedMoveOption("retreat_left", 135.0),
    ScriptedMoveOption("retreat_right", -135.0),
    ScriptedMoveOption("retreat", 180.0),
)


class Actor:
    """A movable actor that can rotate and shoot projectiles."""

    def __init__(
        self,
        position: Vec2,
        angle: float,
        player_id: str = "P1",
        team_id: int = 0,
        active: bool = True,
    ) -> None:
        self.position = position
        self.angle = angle
        self.cooldown_frames = 0
        self.max_health = 1
        self.health = self.max_health
        self.is_active = bool(active)
        self.is_alive = bool(active)
        self.player_id = str(player_id)
        self.team_id = int(team_id)
        self.team = self.player_id
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
        if self.cooldown_frames > 0 or not self.is_active or not self.is_alive:
            return None

        direction = heading_to_vector(self.angle)
        self.cooldown_frames = SHOOT_COOLDOWN_FRAMES
        return {
            "pos": self.position + direction * 20,
            "velocity": direction * PROJECTILE_SPEED,
            "owner": self.player_id,
            "owner_team": self.team_id,
        }

    def take_hit(self, damage: int = 1) -> bool:
        if not self.is_active or not self.is_alive:
            return False
        self.health = max(0, self.health - int(damage))
        if self.health <= 0:
            self.is_alive = False
            return True
        return False

    def tick(self) -> None:
        if not self.is_active:
            return
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1


class Renderer(ArcadeEnvMixin):
    """Arcade renderer for the Bang arena."""

    def __init__(self, game, width: int, height: int, title: str, enabled: bool) -> None:
        self.game = game
        self.enabled = bool(enabled)
        self.width = int(width)
        self.height = int(height)

        self._init_arcade_runtime(
            width=self.width,
            height=self.height,
            title=title,
            render=self.enabled,
            queue_input_events=False,
            vsync=False,
            render_fps=FPS,
            training_fps=TRAINING_FPS,
        )

    def close(self) -> None:
        super().close()

    def poll_events(self) -> None:
        self.window_controller.poll_events_or_raise()

    def draw_frame(self) -> None:
        if self.window is None:
            return

        self.window_controller.clear(COLOR_DARK_NEUTRAL)

        for obstacle in self.game.obstacles:
            draw_two_tone_cell(
                self.window_controller,
                top_left_x=float(obstacle.x),
                top_left_y=float(obstacle.y),
                tile_size=float(TILE_SIZE),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
                cell_inset=float(CELL_INSET),
            )

        self._draw_ghost_overlay()

        for player_id in self.game.player_order:
            actor = self.game.players_by_id[player_id]
            if not actor.is_active or not actor.is_alive:
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
            projectile_color = self.game.player_projectile_colors.get(owner_id, COLOR_SAND)
            arcade.draw_circle_filled(
                projectile["pos"].x,
                self.window_controller.to_arcade_y(projectile["pos"].y),
                5,
                projectile_color,
            )

        self._draw_status_bar()
        self.window_controller.flip()

    def _draw_ghost_overlay(self) -> None:
        if not bool(self.game.show_ghost_overlay and self.game.show_game):
            return
        player = self.game.player
        if player is None or not bool(player.is_active and player.is_alive):
            return
        angles = (
            float(player.angle),
            float(player.angle - 90.0),
            float(player.angle + 90.0),
            float(player.angle + 180.0),
        )
        ray_dirs = tuple((math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle in angles)
        ray_values = tuple(float(self.game._ray_distance(angle)) for angle in angles)
        draw_player_rays(
            origin_x=float(player.position.x),
            origin_y=float(player.position.y),
            ray_dirs=ray_dirs,
            ray_values=ray_values,
            ray_max_distances=(float(self.game._ray_max_range),) * len(ray_dirs),
            to_screen=lambda x, y: (float(x), float(self.window_controller.to_arcade_y(float(y)))),
            line_width=1.5,
        )

    def _draw_status_bar(self) -> None:
        bar_layout = draw_status_bar(
            width=float(self.width),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=float(CELL_INSET),
            include_clock=True,
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

    def _remaining_time_ratio(self) -> float:
        return float(self.game.match_tracker.remaining_time_ratio(int(self.game.frame_count)))

    @staticmethod
    def _status_icon_size() -> float:
        return status_icon_size(float(BB_HEIGHT), float(TILE_SIZE))

    def _draw_winner_history(self, left: float, right: float, center_y: float) -> None:
        icon_size = self._status_icon_size()
        winners = list(self.game.win_history)

        def _draw_history_item(player_id: str | None, center_x: float, row_center_y: float, size: float) -> None:
            if player_id is None:
                return
            self._draw_player_icon(str(player_id), float(center_x), float(row_center_y), float(size))

        draw_status_icon_row(
            left=float(left),
            right=float(right),
            center_y=float(center_y),
            icon_size=float(icon_size),
            items=winners,
            draw_item=_draw_history_item,
        )

    def _draw_player_icon(self, player_id: str, center_x: float, center_y: float, size: float) -> None:
        fill_color, outline_color = self.game.player_render_colors.get(
            player_id,
            (COLOR_DEEP_TEAL, COLOR_AQUA),
        )
        inset = status_icon_inset(float(CELL_INSET))
        marker_size = max(2.0, round(NN_CONTROL_MARKER_SIZE_PX * (size / max(1.0, float(TILE_SIZE)))))
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outline_color,
            inner_color=fill_color,
            inset=float(inset),
            packed=self.game.is_nn_controlled_player(player_id),
            packed_marker_color=outline_color,
            packed_marker_size=float(marker_size),
        )

    def _draw_actor(self, actor: Actor, fill_color, outline_color, draw_nn_marker: bool = False) -> None:
        draw_two_tone_cell(
            self.window_controller,
            top_left_x=actor.position.x - TILE_SIZE / 2,
            top_left_y=actor.position.y - TILE_SIZE / 2,
            tile_size=float(TILE_SIZE),
            outer_color=outline_color,
            inner_color=fill_color,
            cell_inset=float(CELL_INSET),
        )
        if draw_nn_marker:
            self._draw_nn_control_marker(actor, outline_color)

        draw_facing_indicator(
            self.window_controller,
            center_x=float(actor.position.x),
            center_y_top_left=float(actor.position.y),
            angle_degrees=float(actor.angle),
            length=float(TILE_SIZE // 2),
            color=COLOR_LIGHT_NEUTRAL,
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


class BaseGame(ArcadeEnvMixin):
    """Top-down arena shooter with selectable solo and team modes."""

    def __init__(self, level: int = 1, show_game: bool = True, bang_mode: str | None = None):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.playable_height = float(self.height - BB_HEIGHT)
        self._ray_max_range = max(float(TILE_SIZE) * 10.0, min(float(self.width), self.playable_height) * 0.55)
        self._ray_step_size = max(0.75, float(TILE_SIZE) * 0.35)
        self.show_game = bool(show_game)
        self.show_ghost_overlay = bool(SHOW_GHOST_OVERLAY)
        self.ghost_overlay_allowed = True
        self._prev_ghost_overlay_toggle_down = False
        self._init_arcade_timing(render=bool(show_game), render_fps=FPS, training_fps=TRAINING_FPS)

        self.bang_mode = _resolve_bang_mode(bang_mode)
        self.mode_spec = BANG_MODE_SPECS[self.bang_mode]
        self.player_order = tuple(self.mode_spec.player_order)
        self.team_by_player = dict(self.mode_spec.team_by_player)
        self.team_order = tuple(self.mode_spec.team_order)
        self.score_id_by_team = dict(self.mode_spec.score_id_by_team)
        self.controlled_player_ids = tuple(self.mode_spec.controlled_player_order)
        self.player_render_colors: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
        self.player_projectile_colors: dict[str, tuple[int, int, int]] = {}
        self.active_player_ids: tuple[str, ...] = tuple()
        self.match_tracker = MatchTracker[str](clock_duration_steps=int(MAX_EPISODE_STEPS))
        self.scores: dict[str, int] = self.match_tracker.scores
        self.win_history: list[str | None] = self.match_tracker.history
        self.player_control_modes: dict[str, str] = {}
        self._set_bang_mode(self.bang_mode)

        self.players: list[Actor] = []
        self.players_by_id: dict[str, Actor] = {}
        self.scripted_players: list[Actor] = []
        self.target_states: dict[str, TargetState] = {}
        self.scripted_move_states: dict[str, ScriptedMoveState] = {}
        self.obstacle_field = ArcadeSquareObstacleField(TILE_SIZE)

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
        self.show_ghost_overlay, self._prev_ghost_overlay_toggle_down = update_ghost_overlay_toggle(
            window_controller=self.window_controller,
            visible=bool(self.show_ghost_overlay),
            previous_down=bool(self._prev_ghost_overlay_toggle_down),
            enabled=bool(self.show_game and self.ghost_overlay_allowed),
        )

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
        self.active_player_ids = self._active_player_ids_for_level(level)

        self.num_obstacles = settings["num_obstacles"]
        self.enemy_movement = float(clip_unit(float(settings["enemy_movement"])))
        self.enemy_repositioning = float(clip_unit(float(settings["enemy_repositioning"])))
        self.enemy_shot_error_choices = list(ENEMY_SHOT_ERROR_CHOICES)
        self.enemy_shoot_probability = settings["enemy_shoot_probability"]

    def _enemy_player_order(self) -> tuple[str, ...]:
        return tuple(
            player_id
            for player_id in self.player_order
            if int(self.team_by_player[player_id]) != 0
        )

    def _active_enemy_count_for_level(self, level: int) -> int:
        level_counts = dict(LEVEL_SETTINGS[int(level)]["active_enemies"])
        max_count = len(self._enemy_player_order())
        return max(0, min(int(max_count), int(level_counts[self.bang_mode])))

    def _active_player_ids_for_level(self, level: int) -> tuple[str, ...]:
        active_ids: list[str] = list(self.mode_spec.controlled_player_order)
        active_ids.extend(self._enemy_player_order()[: self._active_enemy_count_for_level(level)])
        active_set = set(active_ids)
        return tuple(player_id for player_id in self.player_order if player_id in active_set)

    def _set_bang_mode(self, bang_mode: str) -> None:
        mode_key = _resolve_bang_mode(bang_mode)
        mode_spec = BANG_MODE_SPECS[mode_key]
        player_order = tuple(mode_spec.player_order)
        score_id_by_team = dict(mode_spec.score_id_by_team)
        if mode_key == self.bang_mode and player_order == self.player_order and self.scores:
            return

        self.bang_mode = mode_key
        self.mode_spec = mode_spec
        self.player_order = player_order
        self.team_by_player = dict(mode_spec.team_by_player)
        self.team_order = tuple(mode_spec.team_order)
        self.score_id_by_team = score_id_by_team
        self.controlled_player_ids = tuple(mode_spec.controlled_player_order)
        self.player_render_colors = {
            player_id: TEAM_RENDER_STYLES[int(self.team_by_player[player_id])]
            for player_id in self.player_order
        }
        self.player_projectile_colors = {
            player_id: self.player_render_colors[player_id][1]
            for player_id in self.player_order
        }
        non_scripted_mode = self._non_scripted_control_mode()
        self.player_control_modes = {
            player_id: (
                non_scripted_mode
                if player_id in self.controlled_player_ids
                else CONTROL_MODE_SCRIPTED
            )
            for player_id in self.player_order
        }
        self.match_tracker.set_competitors(
            [score_id_by_team[team_id] for team_id in self.team_order],
            preserve_existing=True,
        )
        self.scores = self.match_tracker.scores
        self.win_history = self.match_tracker.history
        for team_id in self.team_order:
            score_id = score_id_by_team[int(team_id)]
            setattr(self, f"{score_id}_score", self.scores[score_id])

    def reset(self) -> None:
        spawn_positions = self._spawn_positions_by_player()

        self.players_by_id = {}
        inactive_position = Vec2(-float(TILE_SIZE) * 10.0, -float(TILE_SIZE) * 10.0)
        for player_id in self.player_order:
            is_active = player_id in self.active_player_ids
            spawn_pos = spawn_positions.get(player_id, inactive_position)
            self.players_by_id[player_id] = Actor(
                spawn_pos,
                angle=self._sample_inner_facing_angle(spawn_pos) if is_active else 0.0,
                player_id=player_id,
                team_id=int(self.team_by_player[player_id]),
                active=bool(is_active),
            )

        self.players = [self.players_by_id[player_id] for player_id in self.player_order]
        self.player = self.players_by_id[CONTROLLED_PLAYER_ID]
        self.controlled_players = [
            self.players_by_id[player_id]
            for player_id in self.controlled_player_ids
            if self.players_by_id[player_id].is_active
        ]
        # Backward-compatible aliases for older callers.
        self.enemy = next(iter(self._alive_enemies(self.player)), self.players_by_id.get("P2"))
        self.enemy2 = self.players_by_id.get("P3")
        self.enemy3 = self.players_by_id.get("P4")
        self.scripted_players = [
            self.players_by_id[player_id]
            for player_id in self.player_order
            if self.player_control_modes.get(player_id) == CONTROL_MODE_SCRIPTED
            and player_id in self.active_player_ids
        ]

        self.obstacles: list[Vec2] = []
        self.projectiles: list[dict[str, object]] = []
        self.frame_count = 0
        self.last_action_index = ACTION_STOP_MOVE
        self.frames_since_last_shot = SHOOT_COOLDOWN_FRAMES
        self.last_seen_enemy_frame = -EVENT_TIMER_NORMALIZATION_FRAMES
        self.target_states = {
            actor.player_id: TargetState()
            for actor in self.players
        }
        self.scripted_move_states = {
            actor.player_id: ScriptedMoveState()
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

    def _spawn_anchor_for_area(self, area: str) -> Vec2:
        min_y, max_y = self._spawn_y_bounds()
        min_x, max_x = self._spawn_x_bounds()
        center_y = (float(min_y) + float(max_y)) * 0.5
        center_x = (float(min_x) + float(max_x)) * 0.5
        if area == SPAWN_AREA_LEFT:
            return Vec2(self.width * PLAYER_SPAWN_X_RATIO, center_y)
        if area == SPAWN_AREA_RIGHT:
            return Vec2(self.width * ENEMY_SPAWN_X_RATIO, center_y)
        if area == SPAWN_AREA_BOTTOM:
            return Vec2(center_x, self._spawn_bottom_strip_y())
        if area == SPAWN_AREA_TOP:
            return Vec2(center_x, self._spawn_top_strip_y())
        raise ValueError(f"Unknown spawn area: {area}")

    def _clamp_spawn_position(self, position: Vec2) -> Vec2:
        half = float(TILE_SIZE) * 0.5
        return Vec2(
            max(half, min(float(self.width) - half, float(position.x))),
            max(half, min(float(self.height - BB_HEIGHT) - half, float(position.y))),
        )

    def _paired_spawn_position(self, area: str, slot_index: int) -> Vec2:
        anchor = self._spawn_anchor_for_area(area)
        offset = float(TILE_SIZE) * 1.35
        jitter = random.uniform(-float(TILE_SIZE) * 0.35, float(TILE_SIZE) * 0.35)
        side = -1.0 if int(slot_index) <= 0 else 1.0
        if area in {SPAWN_AREA_LEFT, SPAWN_AREA_RIGHT}:
            position = Vec2(anchor.x + jitter, anchor.y + side * offset)
        else:
            position = Vec2(anchor.x + side * offset, anchor.y + jitter)
        return self._clamp_spawn_position(position)

    def _spawn_positions_by_player(self) -> dict[str, Vec2]:
        area_by_team = {
            0: SPAWN_AREA_LEFT,
            1: SPAWN_AREA_RIGHT,
            2: SPAWN_AREA_BOTTOM,
            3: SPAWN_AREA_TOP,
        }
        positions: dict[str, Vec2] = {}
        if self.bang_mode == "team_arena":
            team_slot_counts: dict[int, int] = {}
            for player_id in self.active_player_ids:
                team_id = int(self.team_by_player[player_id])
                slot_index = int(team_slot_counts.get(team_id, 0))
                team_slot_counts[team_id] = slot_index + 1
                positions[player_id] = self._paired_spawn_position(area_by_team[team_id], slot_index)
            return positions

        for player_id in self.active_player_ids:
            team_id = int(self.team_by_player[player_id])
            positions[player_id] = self._sample_spawn_position(area_by_team[team_id])
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
        if not self.player.is_active or not self.player.is_alive:
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

    def _apply_action_to_actor_intents(self, actor: Actor, action_index: int) -> bool:
        if action_index == ACTION_MOVE_UP:
            self._set_actor_move_intent(actor, 0, 1)
            return False
        if action_index == ACTION_MOVE_DOWN:
            self._set_actor_move_intent(actor, 0, -1)
            return False
        if action_index == ACTION_MOVE_LEFT:
            self._set_actor_move_intent(actor, -1, 0)
            return False
        if action_index == ACTION_MOVE_RIGHT:
            self._set_actor_move_intent(actor, 1, 0)
            return False
        if action_index == ACTION_STOP_MOVE:
            self._set_actor_move_intent(actor, 0, 0)
            return False
        if action_index == ACTION_AIM_LEFT:
            self._set_actor_aim_intent(actor, -1)
            return False
        if action_index == ACTION_AIM_RIGHT:
            self._set_actor_aim_intent(actor, 1)
            return False
        if action_index == ACTION_SHOOT:
            projectile = actor.shoot()
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

    def apply_player_action(self, action_index: int | None, actor: Actor | None = None) -> None:
        actor = self.player if actor is None else actor
        if not actor.is_active or not actor.is_alive:
            self.frames_since_last_shot += 1
            return

        control_mode = self.player_control_modes.get(actor.player_id)
        # Human and NN-controlled actor aim are per-step (non-sticky).
        if control_mode in (CONTROL_MODE_HUMAN, CONTROL_MODE_NN):
            self._set_actor_aim_intent(actor, 0)
        if control_mode == CONTROL_MODE_HUMAN:
            # Match move_stop behavior when no WASD movement action is selected this frame.
            self._set_actor_move_intent(actor, 0, 0)

        shot_fired = False
        if action_index is not None:
            self.last_action_index = int(action_index)
            shot_fired = self._apply_action_to_actor_intents(actor, self.last_action_index)

        movement = actor.step_sticky_intents()
        self._update_actor_position(actor, movement)

        if shot_fired:
            self.frames_since_last_shot = 0
        else:
            self.frames_since_last_shot += 1

    def _update_actor_position(self, actor: Actor, movement: Vec2) -> None:
        if not actor.is_active or not actor.is_alive:
            actor.vx = 0.0
            actor.vy = 0.0
            return

        previous_position = actor.position
        new_position = actor.position + movement
        actor_rect = rect_from_center(new_position, TILE_SIZE)
        if self._collides_with_playfield_or_obstacle(actor_rect):
            actor.vx = 0.0
            actor.vy = 0.0
            return

        for other in self.players:
            if other is actor or not other.is_active or not other.is_alive:
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
        if not actor.is_active or not actor.is_alive:
            return True

        new_position = actor.position + movement
        actor_rect = rect_from_center(new_position, TILE_SIZE)
        if self._collides_with_playfield_or_obstacle(actor_rect):
            return True

        for other in self.players:
            if other is actor or not other.is_active or not other.is_alive:
                continue
            other_rect = rect_from_center(other.position, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                return True
        return False

    def _collides_with_playfield_or_obstacle(self, rect) -> bool:
        if (
            rect.left < 0
            or rect.right > self.width
            or rect.top < 0
            or rect.bottom > self.height - BB_HEIGHT
        ):
            return True
        return self.obstacle_field.collides_with_rect(rect)

    def _point_blocked_for_ray(self, x: float, y: float) -> bool:
        if x < 0.0 or x >= float(self.width) or y < 0.0 or y >= float(self.height - BB_HEIGHT):
            return True
        return self.obstacle_field.contains_point(float(x), float(y))

    def _ray_distance(self, angle_degrees: float, actor: Actor | None = None) -> float:
        actor = self.player if actor is None else actor
        radians = math.radians(float(angle_degrees))
        return normalized_ray_first_hit(
            origin_x=float(actor.position.x),
            origin_y=float(actor.position.y),
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
        self.obstacle_field.rebuild(self.obstacles)

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
        if any(tile.distance(actor.position) < SAFE_RADIUS for actor in self.players if actor.is_active and actor.is_alive):
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
    def _move_vector_for_angle(angle_degrees: float, speed_scale: float = 1.0) -> Vec2:
        return rotate_degrees(Vec2(1, 0), angle_degrees) * PLAYER_MOVE_SPEED * max(0.0, float(speed_scale))

    def _move_actor_in_direction(self, actor: Actor, angle_degrees: float) -> bool:
        previous_position = actor.position
        movement = self._move_vector_for_angle(angle_degrees, self.enemy_movement)
        self._update_actor_position(actor, movement)
        return length_squared(actor.position - previous_position) > 0

    def _alive_enemies(self, actor: Actor) -> list[Actor]:
        return [
            other
            for other in self.players
            if other is not actor and other.is_active and other.is_alive and int(other.team_id) != int(actor.team_id)
        ]

    def _alive_opponents(self, actor: Actor) -> list[Actor]:
        return self._alive_enemies(actor)

    def _alive_allies(self, actor: Actor) -> list[Actor]:
        return [
            other
            for other in self.players
            if other is not actor and other.is_active and other.is_alive and int(other.team_id) == int(actor.team_id)
        ]

    def _resolve_alive_target(self, actor: Actor, target_id: str | None) -> Actor | None:
        if target_id is None:
            return None
        target = self.players_by_id.get(target_id)
        if target is None or target is actor or not target.is_active or not target.is_alive:
            return None
        if int(target.team_id) == int(actor.team_id):
            return None
        return target

    def _has_clear_path_between(self, actor: Actor, target: Actor) -> bool:
        return self._has_clear_path_between_points(actor.position, target.position)

    def _has_clear_path_between_points(self, point_a: Vec2, point_b: Vec2) -> bool:
        return not self.obstacle_field.segment_intersects(point_a, point_b)

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
        state.target_id = target.player_id if target is not None else None
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
        state = self.target_states.setdefault(actor.player_id, TargetState())
        if cache_by_frame and state.last_update_frame == self.frame_count:
            return self._resolve_alive_target(actor, state.target_id)
        state.last_update_frame = self.frame_count

        if not actor.is_alive:
            self._set_target_state(actor, state, None, policy)
            return None

        candidates = self._alive_enemies(actor)
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

    def _scripted_move_state(self, actor: Actor) -> ScriptedMoveState:
        return self.scripted_move_states.setdefault(actor.player_id, ScriptedMoveState())

    @staticmethod
    def _turn_toward_angle(current_angle: float, target_angle: float, max_step_degrees: float) -> float:
        delta = normalize_angle_degrees(float(target_angle) - float(current_angle))
        max_step = max(0.0, float(max_step_degrees))
        step = max(-max_step, min(max_step, float(delta)))
        return (float(current_angle) + float(step)) % 360.0

    @staticmethod
    def _scripted_position_key(position: Vec2) -> tuple[int, int]:
        cell_size = max(1.0, float(PLAYER_MOVE_SPEED) * 2.0)
        return (
            int(round(float(position.x) / cell_size)),
            int(round(float(position.y) / cell_size)),
        )

    def _remember_scripted_position(self, actor: Actor, move_state: ScriptedMoveState) -> None:
        key = self._scripted_position_key(actor.position)
        if move_state.recent_position_keys and move_state.recent_position_keys[-1] == key:
            return
        move_state.recent_position_keys.append(key)
        max_memory = max(1, int(ENEMY_RECENT_POSITION_MEMORY))
        if len(move_state.recent_position_keys) > max_memory:
            del move_state.recent_position_keys[:-max_memory]

    @staticmethod
    def _normalized_move_offset(angle_degrees: float, angle_to_target: float) -> float:
        return abs(normalize_angle_degrees(float(angle_degrees) - float(angle_to_target)))

    def _score_scripted_move_option(
        self,
        actor: Actor,
        target: Actor,
        angle_to_target: float,
        current_has_los: bool,
        current_distance: float,
        move_state: ScriptedMoveState,
        option: ScriptedMoveOption,
    ) -> tuple[float, float | None] | None:
        candidate_angle: float | None = None
        candidate_position = actor.position
        if option.angle_offset_degrees is not None:
            candidate_angle = (float(angle_to_target) + float(option.angle_offset_degrees)) % 360.0
            movement = self._move_vector_for_angle(candidate_angle, self.enemy_movement)
            if self._would_collide(actor, movement):
                return None
            candidate_position = actor.position + movement

        candidate_distance = float(candidate_position.distance(target.position))
        candidate_has_los = self._has_clear_path_between_points(candidate_position, target.position)
        candidate_key = self._scripted_position_key(candidate_position)

        desired_distance = float(SAFE_RADIUS)
        distance_error = abs(float(candidate_distance) - desired_distance) / max(1.0, desired_distance)
        score = 0.7 - min(1.25, float(distance_error))

        hidden_ratio = min(1.0, float(move_state.hidden_frames) / max(1.0, float(ENEMY_HIDDEN_URGENCY_FRAMES)))
        reposition_urgency = 1.0 + float(self.enemy_repositioning) * float(hidden_ratio)

        if candidate_has_los:
            score += 0.9
            if not current_has_los:
                score += 1.0 * float(reposition_urgency)
        elif not current_has_los:
            score -= 0.15 * float(reposition_urgency)
        else:
            score -= 0.6

        angle_offset = (
            0.0
            if candidate_angle is None
            else self._normalized_move_offset(candidate_angle, angle_to_target)
        )
        lateral_bias = max(0.0, 1.0 - abs(float(angle_offset) - 90.0) / 90.0)
        advance_bias = max(0.0, 1.0 - float(angle_offset) / 90.0) if angle_offset <= 90.0 else 0.0
        retreat_bias = max(0.0, 1.0 - abs(float(angle_offset) - 180.0) / 90.0) if angle_offset >= 90.0 else 0.0

        if current_has_los:
            if desired_distance * 0.85 <= float(candidate_distance) <= desired_distance * 1.35:
                score += 0.30 * float(lateral_bias)
            elif float(candidate_distance) < desired_distance * 0.85:
                score += 0.45 * float(retreat_bias)
            else:
                score += 0.35 * float(advance_bias)
        else:
            distance_delta = (float(current_distance) - float(candidate_distance)) / max(1.0, float(SAFE_RADIUS))
            score += 0.55 * float(reposition_urgency) * float(lateral_bias)
            score += 0.20 * float(distance_delta)

        recent_revisits = move_state.recent_position_keys[:-1]
        if candidate_key in recent_revisits:
            score -= float(ENEMY_RECENT_POSITION_PENALTY)

        if candidate_angle is None:
            score -= 0.45 * float(reposition_urgency) if not current_has_los else 0.08

        score += random.uniform(0.0, 0.01)
        return float(score), candidate_angle

    def _plan_scripted_move(
        self,
        actor: Actor,
        target: Actor,
        angle_to_target: float,
        move_state: ScriptedMoveState,
    ) -> None:
        current_has_los = self._has_clear_path_between(actor, target)
        current_distance = float(actor.position.distance(target.position))
        best_score: float | None = None
        best_angle: float | None = None

        for option in SCRIPTED_MOVE_OPTIONS:
            scored = self._score_scripted_move_option(
                actor=actor,
                target=target,
                angle_to_target=float(angle_to_target),
                current_has_los=bool(current_has_los),
                current_distance=float(current_distance),
                move_state=move_state,
                option=option,
            )
            if scored is None:
                continue
            score, candidate_angle = scored
            if best_score is None or float(score) > float(best_score):
                best_score = float(score)
                best_angle = candidate_angle

        move_state.planned_move_angle = best_angle
        move_state.planned_target_id = target.player_id
        move_state.commit_frames_remaining = (
            max(4, int(ENEMY_MOVE_COMMIT_FRAMES) - min(4, int(move_state.hidden_frames) // 6))
            if best_angle is not None
            else 0
        )
        move_state.blocked_frames = 0

    def _execute_scripted_move(self, actor: Actor, move_state: ScriptedMoveState) -> bool:
        if move_state.planned_move_angle is None:
            move_state.blocked_frames = 0
            return False

        moved = self._move_actor_in_direction(actor, move_state.planned_move_angle)
        if moved:
            move_state.commit_frames_remaining = max(0, int(move_state.commit_frames_remaining) - 1)
            move_state.blocked_frames = 0
            return True

        move_state.blocked_frames += 1
        move_state.commit_frames_remaining = 0
        return False

    def _step_scripted_movement(self, actor: Actor, target: Actor, angle_to_target: float) -> None:
        if float(self.enemy_movement) <= 0.0:
            return

        move_state = self._scripted_move_state(actor)
        has_los = self._has_clear_path_between(actor, target)
        move_state.hidden_frames = 0 if has_los else int(move_state.hidden_frames) + 1
        target_changed = move_state.planned_target_id != target.player_id

        if target_changed or int(move_state.commit_frames_remaining) <= 0 or int(move_state.blocked_frames) > 0:
            self._plan_scripted_move(actor, target, angle_to_target, move_state)

        moved = self._execute_scripted_move(actor, move_state)
        if not moved and move_state.planned_move_angle is not None:
            self._plan_scripted_move(actor, target, angle_to_target, move_state)
            self._execute_scripted_move(actor, move_state)

        self._remember_scripted_position(actor, move_state)

    def _step_scripted_actor(self, actor: Actor) -> None:
        if not actor.is_active or not actor.is_alive:
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

    def _projectile_owner_team_id(self, projectile: dict[str, object]) -> int | None:
        owner_team = projectile.get("owner_team")
        if owner_team is not None:
            try:
                return int(owner_team)
            except (TypeError, ValueError):
                return None
        owner_id = str(projectile.get("owner", ""))
        team_id = self.team_by_player.get(owner_id)
        return None if team_id is None else int(team_id)

    def _step_projectiles(self):
        events = {"player_kills": 0, "player_killed_by": None, "player_killed_by_team": None}
        next_projectiles = []

        for projectile in self.projectiles:
            projectile["pos"] += projectile["velocity"]
            projectile_rect = rect_from_center(projectile["pos"], PROJECTILE_HITBOX_SIZE)
            if self._collides_with_playfield_or_obstacle(projectile_rect):
                continue

            owner_id = str(projectile["owner"])
            owner_team_id = self._projectile_owner_team_id(projectile)
            colliding_targets = []
            for target in self.players:
                if not target.is_active or not target.is_alive:
                    continue
                if owner_team_id is not None and int(target.team_id) == int(owner_team_id):
                    continue
                target_rect = rect_from_center(target.position, TILE_SIZE)
                if projectile_rect.colliderect(target_rect):
                    colliding_targets.append(target)

            if colliding_targets:
                target = min(colliding_targets, key=lambda candidate: candidate.position.distance(projectile["pos"]))
                eliminated = bool(target.take_hit(1))
                if (
                    owner_id in self.controlled_player_ids
                    and owner_team_id == int(self.player.team_id)
                    and int(target.team_id) != int(self.player.team_id)
                    and eliminated
                ):
                    events["player_kills"] += 1
                if (
                    target.player_id in self.controlled_player_ids
                    and not target.is_alive
                    and events["player_killed_by"] is None
                ):
                    events["player_killed_by"] = owner_id
                    events["player_killed_by_team"] = owner_team_id
                continue

            next_projectiles.append(projectile)

        self.projectiles = next_projectiles
        return events

    def _nearest_hostile_projectile(self, actor: Actor | None = None) -> dict[str, object] | None:
        actor = self.player if actor is None else actor
        hostile_projectiles = [
            p
            for p in self.projectiles
            if self._projectile_owner_team_id(p) != int(actor.team_id)
        ]
        if not hostile_projectiles:
            return None
        return min(
            hostile_projectiles,
            key=lambda projectile: actor.position.distance(projectile["pos"]),
        )

    def _projectile_in_trajectory(self, projectile: dict[str, object], actor: Actor | None = None) -> bool:
        actor = self.player if actor is None else actor
        to_player = actor.position - projectile["pos"]
        if length_squared(to_player) == 0:
            return True
        projectile_dir = projectile["velocity"].normalize()
        return projectile_dir.dot(to_player.normalize()) > PROJECTILE_TRAJECTORY_DOT_THRESHOLD

    def is_player_in_projectile_trajectory(self) -> bool:
        for projectile in self.projectiles:
            if self._projectile_owner_team_id(projectile) == int(self.player.team_id):
                continue
            if self._projectile_in_trajectory(projectile, self.player):
                return True
        return False

    def has_line_of_sight(self, target: Actor | None = None, actor: Actor | None = None) -> bool:
        actor = self.player if actor is None else actor
        if target is None:
            target = self._get_player_target()
        if target is None:
            return False
        return self._is_actor_aimed_at_target(actor, target) and self._has_clear_path_between(actor, target)

    def _nearest_alive_opponent_to_player(self, actor: Actor | None = None) -> Actor | None:
        actor = self.player if actor is None else actor
        opponents = self._alive_enemies(actor)
        if not opponents:
            return None
        return min(opponents, key=lambda opponent: actor.position.distance(opponent.position))

    def _ordered_player_observation_opponents(self, actor: Actor | None = None) -> list[tuple[str, Actor, float]]:
        actor = self.player if actor is None else actor
        order_index = {player_id: idx for idx, player_id in enumerate(ALL_PLAYER_ORDER)}
        ordered_opponents: list[tuple[str, Actor, float]] = []
        for player_id in ALL_PLAYER_ORDER:
            if player_id == actor.player_id:
                continue
            opponent = self.players_by_id.get(player_id)
            if opponent is None or not opponent.is_active or not opponent.is_alive:
                continue
            if int(opponent.team_id) == int(actor.team_id):
                continue
            distance = float(actor.position.distance(opponent.position))
            ordered_opponents.append((player_id, opponent, distance))
        ordered_opponents.sort(key=lambda item: (float(item[2]), int(order_index[str(item[0])])))
        return ordered_opponents

    def _nearest_alive_ally_to_player(self, actor: Actor | None = None) -> Actor | None:
        actor = self.player if actor is None else actor
        allies = self._alive_allies(actor)
        if not allies:
            return None
        order_index = {player_id: idx for idx, player_id in enumerate(ALL_PLAYER_ORDER)}
        return min(
            allies,
            key=lambda ally: (
                float(actor.position.distance(ally.position)),
                int(order_index.get(str(ally.player_id), 999)),
            ),
        )

    def _primary_controlled_player(self) -> Actor:
        for actor in self.controlled_players:
            if actor.is_active:
                return actor
        return self.player

    def _active_controlled_players(self) -> list[Actor]:
        return [actor for actor in self.controlled_players if actor.is_active]

    def _controlled_players_alive(self) -> bool:
        return all(actor.is_alive for actor in self._active_controlled_players())

    def _engagement_potential(self, actor: Actor | None = None) -> float:
        actor = self._primary_controlled_player() if actor is None else actor
        target = self._nearest_alive_opponent_to_player(actor)
        if target is None:
            return 0.0
        dist_scale = max(1.0, max(float(self.width), float(self.height - BB_HEIGHT)))
        tgt_dist_norm = clip_unit(actor.position.distance(target.position) / dist_scale)
        tgt_in_los = 1.0 if self.has_line_of_sight(target, actor=actor) else 0.0
        return float(tgt_in_los - tgt_dist_norm)

    def _hazard_potential(self, actor: Actor | None = None) -> float:
        actor = self._primary_controlled_player() if actor is None else actor
        nearest_projectile = self._nearest_hostile_projectile(actor)
        if nearest_projectile is None:
            haz_dist_norm = 1.0
            haz_in_traj = 0.0
        else:
            dist_scale = max(1.0, max(float(self.width), float(self.height - BB_HEIGHT)))
            haz_dist_norm = clip_unit(actor.position.distance(nearest_projectile["pos"]) / dist_scale)
            haz_in_traj = 1.0 if self._projectile_in_trajectory(nearest_projectile, actor) else 0.0
        return float(haz_dist_norm - 1.5 * haz_in_traj)

    def get_state_vector(self, actor: Actor | None = None) -> list[float]:
        actor = self._primary_controlled_player() if actor is None else actor
        dist_scale = max(1.0, max(float(self.width), float(self.height)))
        nearest_projectile = self._nearest_hostile_projectile(actor)
        if nearest_projectile is None:
            haz_tti_norm = 0.0
            haz_miss_norm = 0.0
            haz_in_traj = 0.0
        else:
            projectile_pos = nearest_projectile["pos"]
            projectile_vel = nearest_projectile["velocity"]
            player_vel = Vec2(float(actor.vx), float(actor.vy))
            rel_pos = projectile_pos - actor.position
            rel_vel = projectile_vel - player_vel
            rel_speed_sq = length_squared(rel_vel)
            rel_speed = math.sqrt(rel_speed_sq) if rel_speed_sq > 1e-8 else 0.0
            tti_horizon = dist_scale / max(1.0, rel_speed)

            approaching = False
            t_closest = 0.0
            if rel_speed_sq > 1e-8:
                closing = -float(rel_pos.dot(rel_vel))
                if closing > 0.0:
                    approaching = True
                    t_closest = closing / rel_speed_sq

            if approaching:
                haz_tti_norm = clip_unit(1.0 - clip_unit(t_closest / max(1e-6, tti_horizon)))
            else:
                haz_tti_norm = 0.0

            closest_rel = rel_pos + (rel_vel * t_closest)
            miss_distance = math.sqrt(length_squared(closest_rel))
            lateral_cross = (float(rel_vel.x) * float(closest_rel.y)) - (float(rel_vel.y) * float(closest_rel.x))
            if lateral_cross > 1e-8:
                miss_sign = 1.0
            elif lateral_cross < -1e-8:
                miss_sign = -1.0
            else:
                miss_sign = 0.0
            haz_miss_norm = clip_signed((miss_sign * miss_distance) / dist_scale)
            haz_in_traj = 1.0 if self._projectile_in_trajectory(nearest_projectile, actor) else 0.0

        sens_fwd = self._ray_distance(actor.angle, actor)
        sens_left = self._ray_distance(actor.angle - 90.0, actor)
        sens_right = self._ray_distance(actor.angle + 90.0, actor)
        sens_back = self._ray_distance(actor.angle + 180.0, actor)
        player_angle_radians = math.radians(actor.angle)
        player_angle_sin = float(math.sin(player_angle_radians))
        player_angle_cos = float(math.cos(player_angle_radians))

        self_shot_cd_norm = clip_unit(float(actor.cooldown_frames) / max(1, SHOOT_COOLDOWN_FRAMES))
        ally = self._nearest_alive_ally_to_player(actor)
        ordered_opponents = self._ordered_player_observation_opponents(actor)
        opp_near_dist_norm = (
            clip_unit(float(ordered_opponents[0][2]) / dist_scale)
            if ordered_opponents
            else 1.0
        )

        feature_values = {
            "self_ang_sin": player_angle_sin,
            "self_ang_cos": player_angle_cos,
            "self_move_x": float(actor.move_intent_x),
            "self_move_y": float(actor.move_intent_y),
            "self_shot_cd_norm": float(self_shot_cd_norm),
            "sens_fwd": float(sens_fwd),
            "sens_left": float(sens_left),
            "sens_right": float(sens_right),
            "sens_back": float(sens_back),
            "ally_dx": 0.0,
            "ally_dy": 0.0,
            "ally_dist_norm": 0.0,
            "ally_los": 0.0,
            "ally_ang_sin": 0.0,
            "ally_ang_cos": 0.0,
            "ally_shot_cd_norm": 0.0,
            "ally_active": 0.0,
            "opp1_dx": 0.0,
            "opp1_dy": 0.0,
            "opp1_los": 0.0,
            "opp1_ang_sin": 0.0,
            "opp1_ang_cos": 0.0,
            "opp2_dx": 0.0,
            "opp2_dy": 0.0,
            "opp2_los": 0.0,
            "opp2_ang_sin": 0.0,
            "opp2_ang_cos": 0.0,
            "opp3_dx": 0.0,
            "opp3_dy": 0.0,
            "opp3_los": 0.0,
            "opp3_ang_sin": 0.0,
            "opp3_ang_cos": 0.0,
            "opp_near_dist_norm": float(opp_near_dist_norm),
            "haz_tti_norm": float(haz_tti_norm),
            "haz_miss_norm": float(haz_miss_norm),
            "haz_in_traj": float(haz_in_traj),
        }

        if ally is not None:
            to_ally = ally.position - actor.position
            ego_dx = (player_angle_cos * float(to_ally.x)) + (player_angle_sin * float(to_ally.y))
            ego_dy = (-player_angle_sin * float(to_ally.x)) + (player_angle_cos * float(to_ally.y))
            ally_angle = math.degrees(math.atan2(to_ally.y, to_ally.x))
            relative_angle = normalize_angle_degrees(ally_angle - float(actor.angle))
            relative_angle_radians = math.radians(relative_angle)
            feature_values["ally_dx"] = float(clip_signed(ego_dx / dist_scale))
            feature_values["ally_dy"] = float(clip_signed(ego_dy / dist_scale))
            feature_values["ally_dist_norm"] = float(clip_unit(actor.position.distance(ally.position) / dist_scale))
            feature_values["ally_los"] = 1.0 if self._has_clear_path_between(actor, ally) else 0.0
            feature_values["ally_ang_sin"] = float(math.sin(relative_angle_radians))
            feature_values["ally_ang_cos"] = float(math.cos(relative_angle_radians))
            feature_values["ally_shot_cd_norm"] = float(
                clip_unit(float(ally.cooldown_frames) / max(1, SHOOT_COOLDOWN_FRAMES))
            )
            feature_values["ally_active"] = 1.0

        for slot_index, (_, opponent, _) in enumerate(ordered_opponents[:3], start=1):
            to_opponent = opponent.position - actor.position
            ego_dx = (player_angle_cos * float(to_opponent.x)) + (player_angle_sin * float(to_opponent.y))
            ego_dy = (-player_angle_sin * float(to_opponent.x)) + (player_angle_cos * float(to_opponent.y))
            opponent_angle = math.degrees(math.atan2(to_opponent.y, to_opponent.x))
            relative_angle = normalize_angle_degrees(opponent_angle - float(actor.angle))
            relative_angle_radians = math.radians(relative_angle)

            feature_values[f"opp{slot_index}_dx"] = float(clip_signed(ego_dx / dist_scale))
            feature_values[f"opp{slot_index}_dy"] = float(clip_signed(ego_dy / dist_scale))
            feature_values[f"opp{slot_index}_los"] = 1.0 if self._has_clear_path_between(actor, opponent) else 0.0
            feature_values[f"opp{slot_index}_ang_sin"] = float(math.sin(relative_angle_radians))
            feature_values[f"opp{slot_index}_ang_cos"] = float(math.cos(relative_angle_radians))

        return self._build_state_vector_from_features(feature_values)

    def get_controlled_state_vectors(self) -> list[list[float]]:
        return [self.get_state_vector(actor) for actor in self._active_controlled_players()]

    def _tick_players(self) -> None:
        for actor in self.players:
            actor.tick()

    def _last_alive_player(self) -> Actor | None:
        alive_players = [actor for actor in self.players if actor.is_active and actor.is_alive]
        if len(alive_players) == 1:
            return alive_players[0]
        return None

    def _alive_team_ids(self) -> set[int]:
        return {int(actor.team_id) for actor in self.players if actor.is_active and actor.is_alive}

    def _active_enemy_players(self) -> list[Actor]:
        return [
            actor
            for actor in self.players
            if actor.is_active and int(actor.team_id) != int(self.player.team_id)
        ]

    def _active_enemies_defeated(self) -> bool:
        return not any(actor.is_alive for actor in self._active_enemy_players())

    def _winner_team_id(self) -> int | None:
        alive_team_ids = self._alive_team_ids()
        if len(alive_team_ids) == 1:
            return next(iter(alive_team_ids))
        return None

    def _controlled_team_won(self) -> bool:
        if not self._controlled_players_alive():
            return False
        return self._active_enemies_defeated()

    def is_player_last_alive(self) -> bool:
        return self._controlled_team_won()

    def _team_id_for_player_id(self, player_id: str | None) -> int | None:
        if player_id is None:
            return None
        team_id = self.team_by_player.get(str(player_id))
        return None if team_id is None else int(team_id)

    def _score_id_for_team(self, team_id: int | None) -> str | None:
        if team_id is None:
            return None
        return self.score_id_by_team.get(int(team_id))

    def _increment_team_score(self, team_id: int | None) -> None:
        score_id = self._score_id_for_team(team_id)
        if score_id is not None:
            self._increment_score(score_id)

    def _increment_score(self, player_id: str) -> None:
        if player_id not in self.scores:
            return
        self.match_tracker.increment_score(player_id)
        self.match_tracker.record_result(player_id)
        setattr(self, f"{player_id}_score", self.scores[player_id])

    def _record_round_draw(self) -> None:
        self.match_tracker.record_draw()


class HumanGame(BaseGame):
    """Human-play mode."""

    def __init__(self, show_game: bool = True, level: int = 1, bang_mode: str | None = None):
        super().__init__(level=int(level), show_game=show_game, bang_mode=bang_mode)

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
        projectile_events = self._step_projectiles()
        self._tick_players()

        player_defeated = not self.player.is_alive
        enemies_defeated = self._active_enemies_defeated()
        if player_defeated:
            killer_team_id = projectile_events.get("player_killed_by_team")
            self._increment_team_score(killer_team_id)
            self.reset()
        elif enemies_defeated:
            self._increment_team_score(self.player.team_id)
            self.reset()
        elif self.frame_count >= MAX_EPISODE_STEPS:
            self._record_round_draw()
            self.reset()

        self.draw_frame()
        self._tick_arcade_frame()


class TrainingGame(BaseGame):
    """Environment used by DQN training."""

    def _non_scripted_control_mode(self) -> str:
        return CONTROL_MODE_NN

    def __init__(
        self,
        level: int = 1,
        show_game: bool = True,
        end_on_player_death: bool = True,
        bang_mode: str | None = None,
    ):
        self.end_on_player_death = bool(end_on_player_death)
        self.player_loss_recorded = False
        super().__init__(level=level, show_game=show_game, bang_mode=bang_mode)

    def reset(self) -> None:
        super().reset()
        self.player_loss_recorded = False

    @staticmethod
    def _action_index_from_value(action_value: object) -> int:
        action_array = np.asarray(action_value)
        if action_array.ndim == 0:
            return int(action_array.item())
        flat = action_array.reshape(-1)
        if int(flat.size) == int(BANG_ACT_DIM):
            return int(np.argmax(flat))
        if int(flat.size) > 0:
            return int(flat[0])
        return ACTION_STOP_MOVE

    def _controlled_action_indices(self, actions: object) -> list[int]:
        action_array = np.asarray(actions)
        controlled_count = len(self._active_controlled_players())
        if controlled_count <= 1:
            return [self._action_index_from_value(actions)]
        if action_array.ndim >= 2:
            values = [action_array[index] for index in range(min(controlled_count, int(action_array.shape[0])))]
        else:
            flat = action_array.reshape(-1)
            if int(flat.size) == int(controlled_count):
                values = [flat[index] for index in range(controlled_count)]
            elif int(flat.size) == int(BANG_ACT_DIM):
                values = [flat]
            else:
                values = [flat[index] for index in range(min(controlled_count, int(flat.size)))]
        indices = [self._action_index_from_value(value) for value in values]
        while len(indices) < controlled_count:
            indices.append(ACTION_STOP_MOVE)
        return indices[:controlled_count]

    def play_step(self, action: object):
        self.frame_count += 1
        self.poll_events()
        self._reset_actor_velocities()
        phi_eng_prev = float(self._engagement_potential())
        phi_haz_prev = float(self._hazard_potential())

        for actor, action_index in zip(self._active_controlled_players(), self._controlled_action_indices(action)):
            self.apply_player_action(action_index, actor=actor)
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

        player_just_died = (not self._controlled_players_alive()) and (not self.player_loss_recorded)
        if player_just_died:
            self.player_loss_recorded = True

        done = False
        timed_out = False
        player_defeated = not self._controlled_players_alive()
        enemies_defeated = self._active_enemies_defeated()
        player_won = bool((not player_defeated) and enemies_defeated)
        player_lost_match = bool(player_defeated)
        if player_defeated:
            done = True
        elif player_won:
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
            if player_defeated:
                killer_team_id = projectile_events.get("player_killed_by_team")
                self._increment_team_score(killer_team_id)
            else:
                self._increment_team_score(self.player.team_id if player_won else None)
            if timed_out:
                self._record_round_draw()

        self.draw_frame()
        self._tick_arcade_frame()

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
        bang_mode: str | None = None,
    ) -> None:
        self.mode = str(mode)
        self.bang_mode = _resolve_bang_mode(bang_mode)
        show_game = bool(render)
        curriculum_config = build_curriculum_config(
            min_level=int(MIN_LEVEL),
            max_level=int(MAX_LEVEL),
            promotion_settings=CURRICULUM_PROMOTION,
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
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        if self.mode == "human":
            self.game = HumanGame(
                show_game=show_game,
                level=int(self._current_level),
                bang_mode=self.bang_mode,
            )
        else:
            level = int(self._current_level)
            if end_on_player_death is None:
                end_on_player_death = self.mode == "train"

            self._current_level = int(level)
            self.game = TrainingGame(
                level=int(level),
                show_game=show_game,
                end_on_player_death=bool(end_on_player_death),
                bang_mode=self.bang_mode,
            )
            self._apply_level_settings(int(self._current_level))
        self.game.ghost_overlay_allowed = bool(self.mode in {"human", "eval"})
        self.window_controller = self.game.window_controller
        self.window = self.game.window

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

    @staticmethod
    def _obs_from_state_vectors(state_vectors: list[list[float]]) -> np.ndarray:
        observations = [BangEnv._obs_from_state_vector(state_vector) for state_vector in state_vectors]
        if not observations:
            return np.zeros((0, int(BangEnv.OBS_DIM)), dtype=np.float32)
        return np.stack(observations, axis=0).astype(np.float32, copy=False)

    def _controlled_obs(self) -> np.ndarray:
        return self._obs_from_state_vectors(self.game.get_controlled_state_vectors())

    def _controlled_agent_ids(self) -> list[str]:
        return [actor.player_id for actor in self.game._active_controlled_players()]

    def reset(self) -> np.ndarray:
        if self.mode == "train":
            self._apply_level_settings(int(self._current_level))
        self.game.reset()
        self._episode_reward_components.reset()
        if self.mode == "human":
            return self._obs_from_state_vector(self.game.get_state_vector())
        return self._controlled_obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.mode == "human":
            self.game.play_step()
            obs = self._obs_from_state_vector(self.game.get_state_vector())
            return obs, 0.0, False, {
                "level": int(getattr(self.game, "level", 1)),
                "bang_mode": str(self.bang_mode),
                "success": 0,
            }

        reward, done, reward_breakdown = self.game.play_step(action)
        self._episode_reward_components.add_from_mapping(reward_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)
        obs = self._controlled_obs()
        episode_level = int(self._current_level) if self.mode == "train" else int(getattr(self.game, "level", 1))
        win = bool(done and self.game.is_player_last_alive())
        success = 1 if win else 0
        controlled_agent_ids = self._controlled_agent_ids()
        info: dict[str, object] = {
            "reward_breakdown": reward_breakdown,
            "win": bool(win),
            "success": int(success) if done else 0,
            "level": int(episode_level),
            "bang_mode": str(self.bang_mode),
            "level_changed": False,
            "controlled_agent_ids": controlled_agent_ids,
            "reward_vec": np.full((len(controlled_agent_ids),), float(reward), dtype=np.float32),
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
        self.game.draw_frame()

    def close(self) -> None:
        self.game.close()
        self.window = None

