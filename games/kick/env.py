"""Simplified scalable football environment with human and RL control modes."""

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
    SharedCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.arcade import ArcadeEnvMixin
from core.envs.base import Env
from core.io_schema import clip_signed, ordered_feature_vector
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_control_marker,
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
from core.rewards import RewardBreakdown
from core.shared_config import (
    BB_HEIGHT,
    CELL_INSET,
    FPS,
    PHYSICS_DT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TRAINING_FPS,
)
from core.utils import resolve_play_level
from games.kick.config import (
    ACTION_NAMES as KICK_ACTION_NAMES,
    ACT_DIM as KICK_ACT_DIM,
    BALL_RADIUS_SCALE,
    BALL_SUPPORT_CLIP,
    BALL_SUPPORT_SCALE,
    BALL_SUPPORT_TARGET_DIST_TILES,
    CENTRAL_OBS_DIM,
    CONTROLLED_PROGRESS_STABLE_STEPS,
    CONTROLLED_PROGRESS_STEP_CLIP,
    CURRICULUM_PROMOTION,
    DEBUG_SANITY_CHECKS,
    DEFAULT_TEAM_SIZE,
    GAME_SPEED_SCALE,
    INPUT_FEATURE_NAMES as KICK_INPUT_FEATURE_NAMES,
    LEVEL_SCRIPTED_SETTINGS,
    MAX_TEAM_PLAYERS,
    MAX_LEVEL,
    MIN_LEVEL,
    OBS_DIM as KICK_OBS_DIM,
    PENALTY_AREA_DEPTH_RATIO,
    PENALTY_AREA_WIDTH_RATIO,
    PENALTY_CONCEDE,
    PITCH_LINE_WIDTH,
    PLAYER_A_MAX_PX_PER_SEC2,
    PLAYER_V_MAX_PX_PER_SEC,
    PPO_METRICS_LOG_ENABLED,
    REWARD_PROGRESS_CONTROLLED,
    REWARD_SCORE,
    RL_ACTION_REPEAT_FRAMES,
    TEAM_SHAPE_CLIP,
    TEAM_SHAPE_LINEAR_COEF,
    TEAM_SHAPE_MIN_DIST_NORM,
    TEAM_SHAPE_QUADRATIC_COEF,
    TEAM_SIZE_CHOICES,
    TEAM_SIZE_LABELS,
    WINDOW_TITLE,
)


def _build_central_feature_names() -> tuple[str, ...]:
    player_features = (
        "x_norm",
        "y_norm",
        "vx",
        "vy",
        "theta_cos",
        "theta_sin",
        "has_ball",
        "active",
    )
    feature_names = [
        f"{team}{slot_index}_{feature_name}"
        for team in ("left", "right")
        for slot_index in range(1, int(MAX_TEAM_PLAYERS) + 1)
        for feature_name in player_features
    ]
    feature_names.extend(
        [
            "tgt_x_norm",
            "tgt_y_norm",
            "tgt_vx",
            "tgt_vy",
            "tgt_owner_left",
            "tgt_owner_right",
            "tgt_owner_free",
            "land_ball_to_opp_goal_dx",
            "land_ball_to_opp_goal_dy",
            "land_ball_to_own_goal_dx",
            "land_ball_to_own_goal_dy",
            "state_time_norm",
            "state_left_score_norm",
            "state_right_score_norm",
            "state_level_norm",
            "state_team_size_norm",
        ]
    )
    if len(feature_names) != int(CENTRAL_OBS_DIM):
        raise RuntimeError(f"Kick centralized feature names expected {int(CENTRAL_OBS_DIM)} entries, got {len(feature_names)}.")
    return tuple(feature_names)


def _build_level_settings_by_team_size() -> dict[int, dict[int, dict[str, float | int]]]:
    level_order = tuple(range(int(MIN_LEVEL), int(MAX_LEVEL) + 1))
    if set(LEVEL_SCRIPTED_SETTINGS.keys()) != set(level_order):
        raise RuntimeError("Kick LEVEL_SCRIPTED_SETTINGS must define every curriculum level.")
    settings_by_team_size: dict[int, dict[int, dict[str, float | int]]] = {
        int(team_size): {} for team_size in TEAM_SIZE_CHOICES
    }
    for level in level_order:
        level_settings = dict(LEVEL_SCRIPTED_SETTINGS[int(level)])
        right_players_by_team_size = dict(level_settings.pop("right_players", {}))
        if set(right_players_by_team_size.keys()) != set(TEAM_SIZE_CHOICES):
            raise RuntimeError(
                f"Kick LEVEL_SCRIPTED_SETTINGS[{int(level)}]['right_players'] must define every team-size choice."
            )
        for team_size in TEAM_SIZE_CHOICES:
            settings_by_team_size[int(team_size)][int(level)] = {
                "right_players": int(right_players_by_team_size[int(team_size)]),
                **level_settings,
            }
    return settings_by_team_size


KICK_CENTRAL_FEATURE_NAMES = _build_central_feature_names()
LEVEL_SETTINGS_BY_TEAM_SIZE = _build_level_settings_by_team_size()

if len(KICK_INPUT_FEATURE_NAMES) != int(KICK_OBS_DIM):
    raise RuntimeError(
        f"Kick actor feature names expected {int(KICK_OBS_DIM)} entries, got {len(KICK_INPUT_FEATURE_NAMES)}."
    )
if len(KICK_ACTION_NAMES) != int(KICK_ACT_DIM):
    raise RuntimeError(f"Kick action names expected {int(KICK_ACT_DIM)} entries, got {len(KICK_ACTION_NAMES)}.")

for _team_size, _level_settings in LEVEL_SETTINGS_BY_TEAM_SIZE.items():
    validate_curriculum_level_settings(
        min_level=MIN_LEVEL,
        max_level=MAX_LEVEL,
        level_settings=_level_settings,
    )


@dataclass(eq=False)
class KickPlayer:
    team: str
    slot_index: int
    x: float
    y: float
    spawn_x: float
    spawn_y: float
    angle: float
    has_ball: bool = False
    contest_cooldown: int = 0
    vx: float = 0.0
    vy: float = 0.0
    in_contact: bool = False


@dataclass(frozen=True)
class ScriptedJob:
    name: str
    job_target: tuple[float, float]
    shape_anchor: tuple[float, float]
    avoid_opponents: bool = True
    preferred_player: KickPlayer | None = None


class KickEnv(ArcadeEnvMixin, Env):
    """Top-down football environment with a scalable centralized critic signal."""

    ACTION_STAY = 0
    ACTION_MOVE_N = 1
    ACTION_MOVE_NE = 2
    ACTION_MOVE_E = 3
    ACTION_MOVE_SE = 4
    ACTION_MOVE_S = 5
    ACTION_MOVE_SW = 6
    ACTION_MOVE_W = 7
    ACTION_MOVE_NW = 8
    ACTION_KICK = 9

    INPUT_FEATURE_NAMES = tuple(KICK_INPUT_FEATURE_NAMES)
    CENTRAL_FEATURE_NAMES = tuple(KICK_CENTRAL_FEATURE_NAMES)
    ACTION_NAMES = tuple(KICK_ACTION_NAMES)
    OBS_DIM = int(KICK_OBS_DIM)
    ACT_DIM = int(KICK_ACT_DIM)
    NUM_ACTIONS = ACT_DIM
    REWARD_COMPONENT_ORDER = ("G", "C", "P", "B", "TS")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_score": "G",
        "outcome.penalty_concede": "C",
        "progress.reward_controlled": "P",
        "support.reward_ball_support": "B",
        "shape.penalty_team_shape": "TS",
    }

    TEAM_LEFT = "left"
    TEAM_RIGHT = "right"
    TEAM_SIZE_CHOICES = tuple(int(size) for size in TEAM_SIZE_CHOICES)
    DEFAULT_TEAM_SIZE = int(DEFAULT_TEAM_SIZE)
    MAX_TEAM_PLAYERS = int(MAX_TEAM_PLAYERS)
    RIGHT_ACTIVE_SLOT_ORDER_BY_TEAM_SIZE = {
        3: (2, 0, 1),
        5: (4, 2, 3, 0, 1),
        7: (6, 4, 5, 1, 2, 3, 0),
    }
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

    MATCH_DURATION_SECONDS = 60.0
    RL_ACTION_REPEAT_FRAMES = int(RL_ACTION_REPEAT_FRAMES)
    KICK_SPEED_SCALE = 10.5
    KICK_TARGET_MIN_ALIGNMENT = 0.15
    KICK_TEAMMATE_BLEND = 0.70
    KICK_GOAL_BLEND = 0.82
    KICK_GOAL_MIN_ALIGNMENT = 0.20
    SCRIPTED_KICK_DISTANCE_TILES = 9.0
    SCRIPTED_PRESSURE_RADIUS_TILES = 3.2
    SCRIPTED_MOVE_DEADBAND_TILES = 0.32
    SCRIPTED_SUPPORT_FORWARD_TILES = 6
    SCRIPTED_SUPPORT_SAFE_TILES = 4.2
    SCRIPTED_SUPPORT_WIDE_TILES = 4.2
    SCRIPTED_SUPPORT_SAFE_WIDE_TILES = 3.2
    SCRIPTED_COVER_WIDE_TILES = 3.2
    SCRIPTED_SEPARATION_TILES = 3.6
    SCRIPTED_AVOID_RADIUS_TILES = 2.4
    SCRIPTED_JOB_TARGET_WEIGHT = 0.75
    SCRIPTED_SHAPE_ANCHOR_WEIGHT = 0.25
    SCRIPTED_PRESS_OFFSET_TILES = 1.45
    POSSESSION_STYLES = ("direct", "wide_upper", "wide_lower", "patient")

    def __init__(
        self,
        mode: str = "train",
        render: bool = False,
        level: int | None = None,
        team_size: int | None = None,
    ) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.log_ppo_metrics_line = bool(PPO_METRICS_LOG_ENABLED)
        self.team_size = self._resolve_team_size(team_size)
        self.level_settings = LEVEL_SETTINGS_BY_TEAM_SIZE[int(self.team_size)]
        curriculum_config = build_curriculum_config(
            min_level=int(MIN_LEVEL),
            max_level=int(MAX_LEVEL),
            promotion_settings=CURRICULUM_PROMOTION,
        )
        self._curriculum = (
            SharedCurriculum(config=curriculum_config, level_settings=self.level_settings)
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

        self._init_arcade_runtime(
            width=SCREEN_WIDTH,
            height=SCREEN_HEIGHT,
            title=WINDOW_TITLE,
            render=bool(render),
            queue_input_events=False,
            vsync=False,
            render_fps=FPS,
            training_fps=TRAINING_FPS,
        )

        self.pitch_top = 0.0
        self.pitch_bottom = float(SCREEN_HEIGHT - BB_HEIGHT)
        self.pitch_height = self.pitch_bottom - self.pitch_top
        self.pitch_center_y = self.pitch_height * 0.5
        self.base_goal_half_height = TILE_SIZE * 3.0
        self.left_goal_half_height = float(self.base_goal_half_height)
        self.right_goal_half_height = float(self.base_goal_half_height)
        self.left_goal_top = self.pitch_center_y - self.left_goal_half_height
        self.left_goal_bottom = self.pitch_center_y + self.left_goal_half_height
        self.right_goal_top = self.pitch_center_y - self.right_goal_half_height
        self.right_goal_bottom = self.pitch_center_y + self.right_goal_half_height
        self.player_size = float(TILE_SIZE)
        self.player_half = self.player_size * 0.5
        self.speed_scale = float(GAME_SPEED_SCALE)
        self.ball_radius = max(3.0, TILE_SIZE * 0.2 * float(BALL_RADIUS_SCALE))
        self.ball_drag_offset = TILE_SIZE * 0.58
        self.physics_dt = float(PHYSICS_DT)
        self.player_vmax_base = float(PLAYER_V_MAX_PX_PER_SEC)
        self.player_amax_base = float(PLAYER_A_MAX_PX_PER_SEC2)
        self.ball_max_speed = max(1.0, 14.0 * self.speed_scale)
        self.ball_friction = 0.985
        self.pickup_range = TILE_SIZE * 0.70 * max(0.75, self.speed_scale)
        self.contest_range = TILE_SIZE * 1.35
        self.contest_cooldown_frames = max(1, int(FPS * 0.65))
        self.freeze_after_restart = 14
        self.max_steps = int(FPS * float(self.MATCH_DURATION_SECONDS))
        self.rl_action_repeat_frames = max(1, int(self.RL_ACTION_REPEAT_FRAMES))
        self.max_player_speed = max(1.0, self.player_vmax_base)
        self.player_contact_radius = self.player_size * 0.44
        self.contact_sep_strength = 0.5
        self.contact_overlap_cap = self.player_size * 0.03
        self.contact_damp = 0.08
        self.contact_accel_scale = 0.75

        self.rng = random.Random(random.randint(0, 2_000_000_000))
        self._right_player_count = int(self.team_size)
        self.right_scripted_reaction_frames = 1
        self.right_scripted_pass_probability = 0.0
        self.right_scripted_mistake_probability = 0.0
        self.right_scripted_player_speed = 1.0
        self.left_scripted_reaction_frames = 1
        self.left_scripted_pass_probability = 0.0
        self.left_scripted_mistake_probability = 0.0
        self.left_scripted_player_speed = 1.0
        self.debug_sanity_checks = bool(DEBUG_SANITY_CHECKS)
        self.central_obs_dim = int(CENTRAL_OBS_DIM)

        self.match_tracker = MatchTracker[str](clock_duration_steps=int(self.max_steps))
        self.match_tracker.set_competitors((self.TEAM_LEFT, self.TEAM_RIGHT), preserve_existing=False)

        self.left_players: list[KickPlayer] = []
        self.right_players: list[KickPlayer] = []
        self.all_players: list[KickPlayer] = []
        self.controlled_index = 0

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_owner: KickPlayer | None = None

        self.last_touch_team: str | None = None
        self.last_touch_player_id: int | None = None
        self.left_score = int(self.match_tracker.score(self.TEAM_LEFT))
        self.right_score = int(self.match_tracker.score(self.TEAM_RIGHT))
        self.steps = 0
        self.done = False
        self.freeze_frames = 0
        self.last_action_index = self.ACTION_STAY
        self._goal_scored_team: str | None = None
        self._controlled_progress_frontier = 0.0
        self._controlled_progress_owner_team: str | None = None
        self._controlled_progress_owner_id: int | None = None
        self._controlled_progress_frames = 0
        self._last_action_by_player_id: dict[int, int] = {}
        self._prev_kick_down = False
        self._scripted_action_cache: dict[tuple[str, int], int] = {}
        self._scripted_target_cache: dict[tuple[str, int], tuple[float, float]] = {}
        self._scripted_force_recompute = True
        self._scripted_possession_team: str | None = None
        self._possession_style_by_team: dict[str, str] = {
            self.TEAM_LEFT: str(self.rng.choice(self.POSSESSION_STYLES)),
            self.TEAM_RIGHT: str(self.rng.choice(self.POSSESSION_STYLES)),
        }
        self._intended_kick_target: dict[tuple[str, int], tuple[float, float]] = {}
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        self._apply_level_change(int(self._current_level))
        self.reset()

    @classmethod
    def _resolve_team_size(cls, team_size: int | None) -> int:
        selected = int(cls.DEFAULT_TEAM_SIZE if team_size is None else team_size)
        if selected not in cls.TEAM_SIZE_CHOICES:
            valid = ", ".join(str(TEAM_SIZE_LABELS.get(size, size)) for size in cls.TEAM_SIZE_CHOICES)
            raise ValueError(f"Unsupported Kick team_size '{selected}'. Expected one of: {valid}.")
        return int(selected)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(float(low), min(float(high), float(value))))

    @staticmethod
    def _distance(ax: float, ay: float, bx: float, by: float) -> float:
        return float(math.hypot(float(ax) - float(bx), float(ay) - float(by)))

    @staticmethod
    def _angle_degrees(ax: float, ay: float, bx: float, by: float) -> float:
        return float(math.degrees(math.atan2(float(by) - float(ay), float(bx) - float(ax))) % 360.0)

    @staticmethod
    def _clamp_vector_magnitude(x: float, y: float, max_magnitude: float) -> tuple[float, float]:
        magnitude = math.hypot(float(x), float(y))
        if magnitude <= float(max_magnitude) or magnitude <= 1e-9:
            return float(x), float(y)
        scale = float(max_magnitude) / magnitude
        return float(x) * scale, float(y) * scale

    @staticmethod
    def _blend_angle(angle_a: float, angle_b: float, weight_b: float) -> float:
        a = math.radians(float(angle_a))
        b = math.radians(float(angle_b))
        weight = float(np.clip(float(weight_b), 0.0, 1.0))
        x = (1.0 - weight) * math.cos(a) + weight * math.cos(b)
        y = (1.0 - weight) * math.sin(a) + weight * math.sin(b)
        if math.hypot(x, y) <= 1e-9:
            return float(angle_b) % 360.0
        return float(math.degrees(math.atan2(y, x)) % 360.0)

    @staticmethod
    def _facing_unit(player: KickPlayer) -> tuple[float, float]:
        radians = math.radians(float(player.angle))
        return math.cos(radians), math.sin(radians)

    @classmethod
    def _attack_sign_for_team(cls, team: str) -> float:
        return 1.0 if str(team) == cls.TEAM_LEFT else -1.0

    def _reset_goal_bounds(self) -> None:
        self.left_goal_half_height = float(self.base_goal_half_height)
        self.right_goal_half_height = float(self.base_goal_half_height)
        self.left_goal_top = self.pitch_center_y - self.left_goal_half_height
        self.left_goal_bottom = self.pitch_center_y + self.left_goal_half_height
        self.right_goal_top = self.pitch_center_y - self.right_goal_half_height
        self.right_goal_bottom = self.pitch_center_y + self.right_goal_half_height

    def _goal_bounds_for_defending_team(self, team: str) -> tuple[float, float]:
        if str(team) == self.TEAM_LEFT:
            return self.left_goal_top, self.left_goal_bottom
        return self.right_goal_top, self.right_goal_bottom

    def _apply_level_settings(self, level: int) -> None:
        settings = self.level_settings.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Kick.")
        required_keys = {
            "right_players",
            "reaction_frames",
            "pass_probability",
            "mistake_probability",
            "scripted_player_speed",
        }
        missing_keys = sorted(required_keys - set(settings.keys()))
        if missing_keys:
            raise ValueError(f"Kick LEVEL_SETTINGS_BY_TEAM_SIZE[{int(self.team_size)}][{int(level)}] missing keys: {missing_keys}")
        try:
            right_players = int(settings["right_players"])
            reaction_frames = int(settings["reaction_frames"])
            pass_probability = float(settings["pass_probability"])
            mistake_probability = float(settings["mistake_probability"])
            scripted_player_speed = float(settings["scripted_player_speed"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Kick LEVEL_SETTINGS numeric fields must be numeric.") from exc

        self._right_player_count = int(np.clip(right_players, 1, int(self.team_size)))
        self.right_scripted_reaction_frames = max(1, int(reaction_frames))
        self.right_scripted_pass_probability = float(np.clip(pass_probability, 0.0, 1.0))
        self.right_scripted_mistake_probability = float(np.clip(mistake_probability, 0.0, 1.0))
        self.right_scripted_player_speed = float(np.clip(scripted_player_speed, 0.05, 1.5))
        left_settings = dict(self.level_settings[int(MAX_LEVEL)])
        self.left_scripted_reaction_frames = max(1, int(left_settings["reaction_frames"]))
        self.left_scripted_pass_probability = float(np.clip(float(left_settings["pass_probability"]), 0.0, 1.0))
        self.left_scripted_mistake_probability = float(np.clip(float(left_settings["mistake_probability"]), 0.0, 1.0))
        self.left_scripted_player_speed = float(np.clip(float(left_settings["scripted_player_speed"]), 0.05, 1.5))
        self._reset_goal_bounds()
        self._current_level = int(level)
        self._scripted_force_recompute = True

    def _apply_level_change(self, level: int) -> None:
        self._apply_level_settings(int(level))
        self._build_teams()
        if hasattr(self, "_scripted_action_cache"):
            self._scripted_action_cache.clear()
            self._scripted_target_cache.clear()
            self._intended_kick_target.clear()
            self._scripted_force_recompute = True

    def _spawn_slots_for_team(self, team: str) -> dict[int, tuple[float, float]]:
        templates = {
            3: (
                (0.22, 0.35),
                (0.22, 0.65),
                (0.34, 0.50),
            ),
            5: (
                (0.18, 0.32),
                (0.18, 0.68),
                (0.30, 0.24),
                (0.30, 0.76),
                (0.40, 0.50),
            ),
            7: (
                (0.16, 0.28),
                (0.16, 0.50),
                (0.16, 0.72),
                (0.29, 0.22),
                (0.29, 0.50),
                (0.29, 0.78),
                (0.42, 0.50),
            ),
        }
        ratios = templates[int(self.team_size)]
        slots: dict[int, tuple[float, float]] = {}
        for slot_index, (x_ratio, y_ratio) in enumerate(ratios):
            spawn_x_ratio = float(x_ratio) if str(team) == self.TEAM_LEFT else 1.0 - float(x_ratio)
            slots[int(slot_index)] = (
                float(SCREEN_WIDTH) * spawn_x_ratio,
                float(self.pitch_top) + float(self.pitch_height) * float(y_ratio),
            )
        return slots

    def _build_team(self, team: str, slot_indices: tuple[int, ...]) -> list[KickPlayer]:
        spawns = self._spawn_slots_for_team(team)
        angle = 0.0 if str(team) == self.TEAM_LEFT else 180.0
        players: list[KickPlayer] = []
        for slot_index in slot_indices:
            spawn_x, spawn_y = spawns[int(slot_index)]
            players.append(
                KickPlayer(
                    team=str(team),
                    slot_index=int(slot_index),
                    x=float(spawn_x),
                    y=float(spawn_y),
                    spawn_x=float(spawn_x),
                    spawn_y=float(spawn_y),
                    angle=float(angle),
                )
            )
        players.sort(key=lambda player: int(player.slot_index))
        return players

    def _build_teams(self) -> None:
        self.left_players = self._build_team(self.TEAM_LEFT, tuple(range(int(self.team_size))))
        right_slots = self._active_right_slots(int(self._right_player_count))
        self.right_players = self._build_team(self.TEAM_RIGHT, tuple(int(slot) for slot in right_slots))
        self.all_players = [*self.left_players, *self.right_players]
        self.controlled_index = 2 if len(self.left_players) >= 3 else 0

    def _active_right_slots(self, right_player_count: int) -> tuple[int, ...]:
        count = int(np.clip(int(right_player_count), 1, int(self.team_size)))
        order = self.RIGHT_ACTIVE_SLOT_ORDER_BY_TEAM_SIZE[int(self.team_size)]
        return tuple(sorted(int(slot) for slot in order[:count]))

    def _controlled_player(self) -> KickPlayer:
        if not self.left_players:
            raise RuntimeError("Kick has no LEFT players.")
        self.controlled_index = int(np.clip(int(self.controlled_index), 0, len(self.left_players) - 1))
        return self.left_players[int(self.controlled_index)]

    def _nearest_player(
        self,
        team: str,
        x: float,
        y: float,
        *,
        exclude: KickPlayer | None = None,
    ) -> KickPlayer:
        candidates = [player for player in self.all_players if player.team == team and player is not exclude]
        if not candidates:
            raise RuntimeError(f"Kick has no candidates for team '{team}'.")
        return min(candidates, key=lambda player: (self._distance(player.x, player.y, x, y), int(player.slot_index)))

    def _nearest_players(
        self,
        team: str,
        x: float,
        y: float,
        *,
        k: int,
        exclude: KickPlayer | None = None,
    ) -> list[KickPlayer]:
        candidates = [player for player in self.all_players if player.team == team and player is not exclude]
        candidates.sort(key=lambda player: (self._distance(player.x, player.y, x, y), int(player.slot_index)))
        return candidates[: max(0, int(k))]

    def _set_ball_owner(self, owner: KickPlayer | None) -> None:
        previous_team = self._scripted_possession_team
        for player in self.all_players:
            player.has_ball = False
        self.ball_owner = owner
        next_team = None if owner is None else str(owner.team)
        if next_team != previous_team:
            self._scripted_force_recompute = True
            self._scripted_possession_team = next_team
            if next_team is not None:
                self._possession_style_by_team[next_team] = str(self.rng.choice(self.POSSESSION_STYLES))
        if owner is None:
            return
        owner.has_ball = True
        self.last_touch_team = owner.team
        self.last_touch_player_id = int(owner.slot_index)
        self._attach_ball_to_owner()

    def _attach_ball_to_owner(self) -> None:
        owner = self.ball_owner
        if owner is None:
            return
        facing_x, facing_y = self._facing_unit(owner)
        self.ball_x = float(owner.x) + facing_x * float(self.ball_drag_offset)
        self.ball_y = float(owner.y) + facing_y * float(self.ball_drag_offset)
        self.ball_vx = 0.0
        self.ball_vy = 0.0

    def physical_owner_team(self) -> str | None:
        return None if self.ball_owner is None else str(self.ball_owner.team)

    def physical_owner_id(self) -> int | None:
        return None if self.ball_owner is None else int(self.ball_owner.slot_index)

    def effective_possession_team(self) -> str | None:
        if self.ball_owner is not None:
            return str(self.ball_owner.team)
        return self.last_touch_team

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

    def _decode_team_actions(self, actions) -> np.ndarray:
        team_size = len(self.left_players)
        if team_size <= 0:
            return np.zeros((0,), dtype=np.int64)
        if np.isscalar(actions):
            if team_size != 1:
                raise ValueError(f"Kick RL mode expects {team_size} actions, got scalar action.")
            return np.asarray([self._decode_action(actions)], dtype=np.int64)
        action_array = np.asarray(actions).reshape(-1)
        if int(action_array.size) != int(team_size):
            raise ValueError(f"Kick RL mode expects {team_size} actions, got {int(action_array.size)}.")
        return np.asarray(np.clip(action_array.astype(np.int64, copy=False), 0, self.NUM_ACTIONS - 1), dtype=np.int64)

    def get_action_mask(self, _obs: object | None = None) -> np.ndarray:
        if self.mode == "human":
            mask = np.ones((self.NUM_ACTIONS,), dtype=np.bool_)
            controlled = self._controlled_player()
            if not (controlled.has_ball and self.ball_owner is controlled):
                mask[self.ACTION_KICK] = False
            return mask

        mask = np.ones((len(self.left_players), self.NUM_ACTIONS), dtype=np.bool_)
        for idx, player in enumerate(self.left_players):
            if player.has_ball and self.ball_owner is player:
                continue
            mask[idx, self.ACTION_KICK] = False
        return mask

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

    def _max_speed_for(self, player: KickPlayer, *, speed_scale: float = 1.0) -> float:
        return max(0.0, self.player_vmax_base * float(speed_scale))

    def _max_accel_for(self, player: KickPlayer, *, speed_scale: float = 1.0) -> float:
        accel = self.player_amax_base * float(speed_scale)
        if player.in_contact:
            accel *= self.contact_accel_scale
        return max(0.0, float(accel))

    def _set_player_stationary(self, player: KickPlayer) -> None:
        player.vx = 0.0
        player.vy = 0.0
        player.in_contact = False

    def _player_bounds(self, player: KickPlayer) -> tuple[float, float, float, float]:
        min_x = self.player_half
        max_x = SCREEN_WIDTH - self.player_half
        min_y = self.player_half
        max_y = self.pitch_bottom - self.player_half
        if player.has_ball and self.ball_owner is player:
            min_y = -self.player_half
            max_y = self.pitch_bottom + self.player_half
        return min_x, max_x, min_y, max_y

    def _clamp_player_position(self, player: KickPlayer) -> None:
        min_x, max_x, min_y, max_y = self._player_bounds(player)
        player.x = self._clamp(player.x, min_x, max_x)
        player.y = self._clamp(player.y, min_y, max_y)

    def _move_player(
        self,
        player: KickPlayer,
        direction_x: float,
        direction_y: float,
        *,
        speed_scale: float = 1.0,
    ) -> None:
        length = math.hypot(float(direction_x), float(direction_y))
        if length > 1e-9:
            dir_x = float(direction_x) / length
            dir_y = float(direction_y) / length
        else:
            dir_x = 0.0
            dir_y = 0.0

        max_speed = self._max_speed_for(player, speed_scale=float(speed_scale))
        max_accel = self._max_accel_for(player, speed_scale=float(speed_scale))
        desired_vx = dir_x * max_speed
        desired_vy = dir_y * max_speed
        dvx = desired_vx - float(player.vx)
        dvy = desired_vy - float(player.vy)
        dvx, dvy = self._clamp_vector_magnitude(dvx, dvy, max_accel * self.physics_dt)

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

    def _resolve_player_contacts(self) -> None:
        if len(self.all_players) < 2:
            return
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
            player.vx, player.vy = self._clamp_vector_magnitude(player.vx, player.vy, self._max_speed_for(player))

    def _kick_speed(self) -> float:
        return float(self.KICK_SPEED_SCALE) * self.speed_scale

    def _select_aligned_teammate_target(self, carrier: KickPlayer) -> KickPlayer | None:
        facing_x, facing_y = self._facing_unit(carrier)
        teammates = [player for player in self.all_players if player.team == carrier.team and player is not carrier]
        best_target: KickPlayer | None = None
        best_score = -1e9
        for teammate in teammates:
            rel_x = float(teammate.x) - float(carrier.x)
            rel_y = float(teammate.y) - float(carrier.y)
            distance = math.hypot(rel_x, rel_y)
            if distance <= TILE_SIZE * 0.8:
                continue
            alignment = (rel_x * facing_x + rel_y * facing_y) / max(1e-6, distance)
            if alignment < float(self.KICK_TARGET_MIN_ALIGNMENT):
                continue
            forward = rel_x * self._attack_sign_for_team(carrier.team)
            score = 1.5 * alignment + 0.20 * (forward / max(1.0, SCREEN_WIDTH)) - 0.15 * (
                distance / max(1.0, SCREEN_WIDTH)
            )
            if score > best_score:
                best_score = float(score)
                best_target = teammate
        return best_target

    def _kick_ball(self, player: KickPlayer, *, speed: float, angle_degrees: float) -> None:
        if not player.has_ball or self.ball_owner is not player:
            return
        angle = float(angle_degrees) % 360.0
        radians = math.radians(angle)
        self._set_ball_owner(None)
        self.ball_x = float(player.x) + math.cos(radians) * float(self.ball_drag_offset)
        self.ball_y = float(player.y) + math.sin(radians) * float(self.ball_drag_offset)
        self.ball_vx = math.cos(radians) * float(speed)
        self.ball_vy = math.sin(radians) * float(speed)
        self.last_touch_team = str(player.team)
        self.last_touch_player_id = int(player.slot_index)

    def _goal_center_for_attack(self, team: str) -> tuple[float, float]:
        if str(team) == self.TEAM_LEFT:
            return float(SCREEN_WIDTH - self.ball_radius), float((self.right_goal_top + self.right_goal_bottom) * 0.5)
        return float(self.ball_radius), float((self.left_goal_top + self.left_goal_bottom) * 0.5)

    def _goal_alignment(self, player: KickPlayer) -> float:
        goal_x, goal_y = self._goal_center_for_attack(player.team)
        rel_x = float(goal_x) - float(player.x)
        rel_y = float(goal_y) - float(player.y)
        distance = math.hypot(rel_x, rel_y)
        if distance <= 1e-9:
            return 1.0
        facing_x, facing_y = self._facing_unit(player)
        return float(np.clip((rel_x * facing_x + rel_y * facing_y) / distance, -1.0, 1.0))

    def _resolve_kick_action(self, player: KickPlayer) -> None:
        if not player.has_ball or self.ball_owner is not player:
            return
        key = self._player_key(player)
        intended_target = self._intended_kick_target.pop(key, None)
        if intended_target is not None:
            target_angle = self._angle_degrees(player.x, player.y, intended_target[0], intended_target[1])
            kick_angle = self._blend_angle(player.angle, target_angle, 0.88)
        else:
            teammate_target = self._select_aligned_teammate_target(player)
            if teammate_target is not None:
                target_angle = self._angle_degrees(player.x, player.y, teammate_target.x, teammate_target.y)
                kick_angle = self._blend_angle(player.angle, target_angle, float(self.KICK_TEAMMATE_BLEND))
            elif self._goal_alignment(player) >= float(self.KICK_GOAL_MIN_ALIGNMENT):
                goal_x, goal_y = self._goal_center_for_attack(player.team)
                target_angle = self._angle_degrees(player.x, player.y, goal_x, goal_y)
                kick_angle = self._blend_angle(player.angle, target_angle, float(self.KICK_GOAL_BLEND))
            else:
                kick_angle = float(player.angle)
        player.angle = float(kick_angle)
        self._kick_ball(player, speed=self._kick_speed(), angle_degrees=kick_angle)

    def _contest_chance(self, owner: KickPlayer, challenger: KickPlayer) -> float:
        to_challenger_x = float(challenger.x) - float(owner.x)
        to_challenger_y = float(challenger.y) - float(owner.y)
        mag = math.hypot(to_challenger_x, to_challenger_y)
        if mag <= 1e-6:
            return 0.5
        to_challenger_x /= mag
        to_challenger_y /= mag
        facing_x, facing_y = self._facing_unit(owner)
        dot = facing_x * to_challenger_x + facing_y * to_challenger_y
        if dot >= 0.5:
            return 0.72
        if dot <= -0.5:
            return 0.28
        return 0.50

    def _attempt_contest(self, player: KickPlayer) -> bool:
        owner = self.ball_owner
        if owner is None or owner.team == player.team or player.contest_cooldown > 0:
            return False
        if self._distance(player.x, player.y, self.ball_x, self.ball_y) > self.contest_range:
            return False
        chance = self._contest_chance(owner, player)
        owner.contest_cooldown = self.contest_cooldown_frames
        player.contest_cooldown = self.contest_cooldown_frames
        if self.rng.random() > chance:
            return False
        player.angle = self._angle_degrees(player.x, player.y, self._attacking_goal_x_for_team(player.team), self.pitch_center_y)
        self._set_ball_owner(player)
        return True

    def _attacking_goal_x_for_team(self, team: str) -> float:
        return float(SCREEN_WIDTH - self.ball_radius) if str(team) == self.TEAM_LEFT else float(self.ball_radius)

    def _run_auto_contests(self) -> None:
        owner = self.ball_owner
        if owner is None:
            return
        challengers = self.left_players if owner.team == self.TEAM_RIGHT else self.right_players
        candidates = [
            player
            for player in challengers
            if player.contest_cooldown <= 0
            and self._distance(player.x, player.y, self.ball_x, self.ball_y) <= self.contest_range
        ]
        candidates.sort(key=lambda player: (self._distance(player.x, player.y, self.ball_x, self.ball_y), player.slot_index))
        for challenger in candidates:
            if self._attempt_contest(challenger):
                break

    def _decay_timers(self) -> None:
        for player in self.all_players:
            if player.contest_cooldown > 0:
                player.contest_cooldown -= 1

    def _human_controlled_step(self) -> KickPlayer:
        self._auto_select_human_controlled_player()
        controlled = self._controlled_player()
        up = self.window_controller.is_key_down(arcade.key.W)
        down = self.window_controller.is_key_down(arcade.key.S)
        left = self.window_controller.is_key_down(arcade.key.A)
        right = self.window_controller.is_key_down(arcade.key.D)
        move_x = float(right) - float(left)
        move_y = float(down) - float(up)
        self.last_action_index = self._move_action_from_vector(move_x, move_y)
        self._move_player(controlled, move_x, move_y)
        if math.hypot(move_x, move_y) > 1e-6:
            controlled.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)

        kick_down = self.window_controller.is_key_down(arcade.key.SPACE) or self.window_controller.is_key_down(arcade.key.ENTER)
        if kick_down and not self._prev_kick_down and controlled.has_ball:
            self._resolve_kick_action(controlled)
            self.last_action_index = self.ACTION_KICK
        self._prev_kick_down = bool(kick_down)
        return controlled

    def _apply_rl_action_to_player(self, player: KickPlayer, action_idx: int) -> None:
        action_idx = int(np.clip(int(action_idx), 0, self.NUM_ACTIONS - 1))
        self.last_action_index = int(action_idx)
        self._last_action_by_player_id[int(player.slot_index)] = int(action_idx)
        if action_idx <= self.ACTION_MOVE_NW:
            move_x, move_y = self.ACTION_TO_DIRECTION.get(action_idx, (0.0, 0.0))
            self._move_player(player, move_x, move_y)
            if math.hypot(move_x, move_y) > 1e-6:
                player.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)
            return
        self._move_player(player, 0.0, 0.0)
        if not player.has_ball or self.ball_owner is not player:
            return
        if action_idx == self.ACTION_KICK:
            self._resolve_kick_action(player)

    def _rl_team_step(self, actions) -> np.ndarray:
        action_indices = self._decode_team_actions(actions)
        if self.debug_sanity_checks and self.mode == "eval":
            action_mask = self.get_action_mask()
            for idx, action_idx in enumerate(action_indices):
                if int(action_idx) != self.ACTION_KICK:
                    continue
                if bool(action_mask[int(idx), int(action_idx)]):
                    continue
                raise RuntimeError(
                    f"Kick eval produced invalid kick for player index {int(idx)} with mask disabled action {int(action_idx)}."
                )
        for player, action_idx in zip(self.left_players, action_indices):
            self._apply_rl_action_to_player(player, int(action_idx))
        return action_indices

    def _player_key(self, player: KickPlayer) -> tuple[str, int]:
        return str(player.team), int(player.slot_index)

    def _team_players(self, team: str) -> list[KickPlayer]:
        return self.left_players if str(team) == self.TEAM_LEFT else self.right_players

    def _opponent_players(self, team: str) -> list[KickPlayer]:
        return self.right_players if str(team) == self.TEAM_LEFT else self.left_players

    def _own_goal_center_for_team(self, team: str) -> tuple[float, float]:
        if str(team) == self.TEAM_LEFT:
            return float(self.ball_radius), float((self.left_goal_top + self.left_goal_bottom) * 0.5)
        return float(SCREEN_WIDTH - self.ball_radius), float((self.right_goal_top + self.right_goal_bottom) * 0.5)

    @staticmethod
    def _point_between(a: tuple[float, float], b: tuple[float, float], ratio: float) -> tuple[float, float]:
        amount = float(np.clip(float(ratio), 0.0, 1.0))
        return (
            float(a[0]) + (float(b[0]) - float(a[0])) * amount,
            float(a[1]) + (float(b[1]) - float(a[1])) * amount,
        )

    def _clamp_to_field(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            self._clamp(float(point[0]), self.player_half, float(SCREEN_WIDTH) - self.player_half),
            self._clamp(float(point[1]), self.player_half, float(self.pitch_bottom) - self.player_half),
        )

    @staticmethod
    def _unit_toward(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return 0.0, 0.0
        return dx / length, dy / length

    @staticmethod
    def _perp(vec: tuple[float, float]) -> tuple[float, float]:
        return -float(vec[1]), float(vec[0])

    def _unit_or_fallback(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        unit_x, unit_y = self._unit_toward(start, end)
        if math.hypot(unit_x, unit_y) > 1e-9:
            return unit_x, unit_y
        fallback_x = float(fallback[0])
        fallback_y = float(fallback[1])
        fallback_length = math.hypot(fallback_x, fallback_y)
        if fallback_length <= 1e-9:
            return 1.0, 0.0
        return fallback_x / fallback_length, fallback_y / fallback_length

    def _offset_point(
        self,
        origin: tuple[float, float],
        forward_dir: tuple[float, float],
        forward_tiles: float,
        side_dir: tuple[float, float],
        side_tiles: float,
    ) -> tuple[float, float]:
        return (
            float(origin[0])
            + float(forward_dir[0]) * float(TILE_SIZE) * float(forward_tiles)
            + float(side_dir[0]) * float(TILE_SIZE) * float(side_tiles),
            float(origin[1])
            + float(forward_dir[1]) * float(TILE_SIZE) * float(forward_tiles)
            + float(side_dir[1]) * float(TILE_SIZE) * float(side_tiles),
        )

    def _style_side_bias_for_dir(self, team: str, side_dir: tuple[float, float]) -> float:
        bias = float(self._style_lateral_bias(team))
        magnitude = max(0.1, abs(bias))
        side_y = float(side_dir[1])
        if abs(side_y) <= 1e-6:
            return bias
        if bias < 0.0:
            return -magnitude if side_y > 0.0 else magnitude
        return magnitude if side_y > 0.0 else -magnitude

    def _side_sign_toward_y(
        self,
        side_dir: tuple[float, float],
        *,
        from_y: float,
        target_y: float,
        fallback: float,
    ) -> float:
        side_y = float(side_dir[1])
        if abs(side_y) <= 1e-6 or abs(float(target_y) - float(from_y)) <= float(TILE_SIZE) * 0.2:
            return 1.0 if float(fallback) >= 0.0 else -1.0
        wants_down = float(target_y) > float(from_y)
        return 1.0 if (side_y > 0.0) == wants_down else -1.0

    def _blend_scripted_job_target(self, job: ScriptedJob) -> tuple[float, float]:
        job_weight = float(self.SCRIPTED_JOB_TARGET_WEIGHT)
        anchor_weight = float(self.SCRIPTED_SHAPE_ANCHOR_WEIGHT)
        return self._clamp_to_field(
            (
                float(job.job_target[0]) * job_weight + float(job.shape_anchor[0]) * anchor_weight,
                float(job.job_target[1]) * job_weight + float(job.shape_anchor[1]) * anchor_weight,
            )
        )

    def _target_separation_push(
        self,
        player: KickPlayer,
        target: tuple[float, float],
        *,
        teammates: list[KickPlayer],
        other_targets: list[tuple[float, float]],
        min_dist: float,
    ) -> tuple[float, float]:
        push_x = 0.0
        push_y = 0.0
        sources = list(other_targets)
        sources.extend((float(teammate.x), float(teammate.y)) for teammate in teammates if teammate is not player)
        for source_x, source_y in sources:
            dx = float(target[0]) - float(source_x)
            dy = float(target[1]) - float(source_y)
            distance = math.hypot(dx, dy)
            if distance >= float(min_dist):
                continue
            if distance <= 1e-6:
                angle = math.radians((int(player.slot_index) + 1) * 137.5)
                dx = math.cos(angle)
                dy = math.sin(angle)
                distance = 1.0
            strength = (float(min_dist) - distance) / max(1.0, float(min_dist))
            push_x += (dx / distance) * strength
            push_y += (dy / distance) * strength
        push_x, push_y = self._clamp_vector_magnitude(
            push_x * float(TILE_SIZE) * 1.15,
            push_y * float(TILE_SIZE) * 1.15,
            float(TILE_SIZE) * 1.8,
        )
        return push_x, push_y

    def _opponent_avoidance_push(
        self,
        player: KickPlayer,
        opponents: list[KickPlayer],
        radius: float,
    ) -> tuple[float, float]:
        push_x = 0.0
        push_y = 0.0
        for opponent in opponents:
            dx = float(player.x) - float(opponent.x)
            dy = float(player.y) - float(opponent.y)
            distance = math.hypot(dx, dy)
            if distance <= 1e-6 or distance >= float(radius):
                continue
            strength = (float(radius) - distance) / max(1.0, float(radius))
            push_x += (dx / distance) * strength
            push_y += (dy / distance) * strength
        return push_x * TILE_SIZE * 1.0, push_y * TILE_SIZE * 1.0

    def _move_action_toward(
        self,
        player: KickPlayer,
        target: tuple[float, float],
        speed_scale: float,
    ) -> int:
        del speed_scale
        dx = float(target[0]) - float(player.x)
        dy = float(target[1]) - float(player.y)
        if math.hypot(dx, dy) <= float(TILE_SIZE) * float(self.SCRIPTED_MOVE_DEADBAND_TILES):
            return self.ACTION_STAY
        return self._move_action_from_vector(dx, dy)

    def _style_for_team(self, team: str) -> str:
        style = str(self._possession_style_by_team.get(str(team), "direct"))
        return style if style in self.POSSESSION_STYLES else "direct"

    def _style_lateral_bias(self, team: str) -> float:
        style = self._style_for_team(team)
        if style == "wide_upper":
            return -1.0
        if style == "wide_lower":
            return 1.0
        if style == "patient":
            return -0.85 if float(self.ball_y) > float(self.pitch_center_y) else 0.85
        return -0.65 if float(self.ball_y) > float(self.pitch_center_y) else 0.65

    def _style_pass_probability(self, team: str) -> float:
        bonus = 0.15 if self._style_for_team(team) == "patient" else 0.0
        return float(np.clip(float(self._scripted_pass_probability_for_team(team)) + bonus, 0.0, 1.0))

    def _scripted_reaction_frames_for_team(self, team: str) -> int:
        if str(team) == self.TEAM_LEFT:
            return max(1, int(self.left_scripted_reaction_frames))
        return max(1, int(self.right_scripted_reaction_frames))

    def _scripted_pass_probability_for_team(self, team: str) -> float:
        if str(team) == self.TEAM_LEFT:
            return float(self.left_scripted_pass_probability)
        return float(self.right_scripted_pass_probability)

    def _scripted_mistake_probability_for_team(self, team: str) -> float:
        if str(team) == self.TEAM_LEFT:
            return float(self.left_scripted_mistake_probability)
        return float(self.right_scripted_mistake_probability)

    def _scripted_player_speed_for_team(self, team: str) -> float:
        if str(team) == self.TEAM_LEFT:
            return float(self.left_scripted_player_speed)
        return float(self.right_scripted_player_speed)

    def _adjust_scripted_target(
        self,
        player: KickPlayer,
        target: tuple[float, float],
        *,
        teammates: list[KickPlayer],
        opponents: list[KickPlayer],
        other_targets: list[tuple[float, float]] | None = None,
        avoid_opponents: bool = True,
    ) -> tuple[float, float]:
        sep_x, sep_y = self._target_separation_push(
            player,
            target,
            teammates=teammates,
            other_targets=list(other_targets or []),
            min_dist=float(TILE_SIZE) * float(self.SCRIPTED_SEPARATION_TILES),
        )
        avoid_x = 0.0
        avoid_y = 0.0
        if avoid_opponents:
            avoid_x, avoid_y = self._opponent_avoidance_push(
                player,
                opponents,
                radius=float(TILE_SIZE) * float(self.SCRIPTED_AVOID_RADIUS_TILES),
            )
        return self._clamp_to_field((float(target[0]) + sep_x + avoid_x, float(target[1]) + sep_y + avoid_y))

    def _scripted_target_action(
        self,
        player: KickPlayer,
        target: tuple[float, float],
        *,
        teammates: list[KickPlayer],
        opponents: list[KickPlayer],
        avoid_opponents: bool = True,
        target_already_adjusted: bool = False,
    ) -> int:
        if target_already_adjusted:
            adjusted = self._clamp_to_field(target)
        else:
            adjusted = self._adjust_scripted_target(
                player,
                target,
                teammates=teammates,
                opponents=opponents,
                avoid_opponents=avoid_opponents,
            )
        self._scripted_target_cache[self._player_key(player)] = adjusted
        return self._move_action_toward(player, adjusted, self._scripted_player_speed_for_team(player.team))

    def _carrier_pressure_count(self, carrier: KickPlayer) -> int:
        defenders = self._opponent_players(carrier.team)
        radius = float(self.SCRIPTED_PRESSURE_RADIUS_TILES) * float(TILE_SIZE)
        return sum(1 for defender in defenders if self._distance(defender.x, defender.y, carrier.x, carrier.y) <= radius)

    def _scripted_carrier_action(
        self,
        player: KickPlayer,
        *,
        target: tuple[float, float],
        teammates: list[KickPlayer],
        opponents: list[KickPlayer],
    ) -> int:
        team = str(player.team)
        goal = self._goal_center_for_attack(team)
        distance_to_goal = self._distance(player.x, player.y, goal[0], goal[1])
        pressure = self._carrier_pressure_count(player)
        if (
            distance_to_goal <= float(self.SCRIPTED_KICK_DISTANCE_TILES) * float(TILE_SIZE)
            and self._goal_alignment(player) >= float(self.KICK_GOAL_MIN_ALIGNMENT)
        ):
            self._intended_kick_target[self._player_key(player)] = goal
            self._scripted_target_cache[self._player_key(player)] = goal
            return self.ACTION_KICK

        teammate_target = self._select_aligned_teammate_target(player) if pressure > 0 else None
        if teammate_target is not None and self.rng.random() < self._style_pass_probability(team):
            target = (float(teammate_target.x), float(teammate_target.y))
            self._intended_kick_target[self._player_key(player)] = target
            self._scripted_target_cache[self._player_key(player)] = target
            return self.ACTION_KICK

        return self._scripted_target_action(
            player,
            target,
            teammates=teammates,
            opponents=opponents,
            avoid_opponents=True,
            target_already_adjusted=True,
        )

    def _dangerous_offball_opponent(self, team: str, carrier: KickPlayer) -> KickPlayer | None:
        own_goal = self._own_goal_center_for_team(team)
        carrier_point = (float(carrier.x), float(carrier.y))
        candidates = [player for player in self._opponent_players(team) if player is not carrier]
        candidates.sort(
            key=lambda player: (
                self._distance(player.x, player.y, own_goal[0], own_goal[1]),
                self._distance(player.x, player.y, carrier_point[0], carrier_point[1]),
                int(player.slot_index),
            )
        )
        for candidate in candidates:
            if self._distance(candidate.x, candidate.y, carrier_point[0], carrier_point[1]) >= float(TILE_SIZE) * 2.4:
                return candidate
        return None

    def _scripted_attack_jobs(
        self,
        team: str,
        scripted_players: list[KickPlayer],
        owner: KickPlayer,
    ) -> list[ScriptedJob]:
        anchor = (float(owner.x), float(owner.y))
        goal = self._goal_center_for_attack(team)
        goal_dir = self._unit_or_fallback(anchor, goal, (self._attack_sign_for_team(team), 0.0))
        side_dir = self._perp(goal_dir)
        side_bias = self._style_side_bias_for_dir(team, side_dir)
        style = self._style_for_team(team)
        forward_tiles = float(self.SCRIPTED_SUPPORT_FORWARD_TILES) + (0.8 if style == "direct" else -0.3)
        safe_tiles = float(self.SCRIPTED_SUPPORT_SAFE_TILES) + (0.9 if style == "patient" else 0.0)
        spread_tiles = float(self.SCRIPTED_SUPPORT_WIDE_TILES) + (0.8 if style in {"wide_upper", "wide_lower"} else 0.0)
        safe_spread_tiles = float(self.SCRIPTED_SUPPORT_SAFE_WIDE_TILES) + (0.8 if style == "patient" else 0.0)
        jobs: list[ScriptedJob] = []

        owner_is_scripted = owner in scripted_players
        if owner_is_scripted:
            carrier_side = side_bias * (0.35 if style == "direct" else 1.15 if style == "patient" else 0.8)
            carrier_job = self._offset_point(anchor, goal_dir, 4.2, side_dir, carrier_side)
            carrier_anchor = self._offset_point(anchor, goal_dir, 3.0, side_dir, carrier_side * 0.45)
            jobs.append(
                ScriptedJob(
                    "carrier",
                    self._clamp_to_field(carrier_job),
                    self._clamp_to_field(carrier_anchor),
                    avoid_opponents=True,
                    preferred_player=owner,
                )
            )

        support_count = len(scripted_players) - (1 if owner_is_scripted else 0)
        for support_index in range(max(0, support_count)):
            if support_index == 0:
                job_target = self._offset_point(anchor, goal_dir, forward_tiles, side_dir, spread_tiles * side_bias)
                shape_anchor = self._offset_point(anchor, goal_dir, forward_tiles * 0.78, side_dir, spread_tiles * side_bias * 0.70)
                name = "support_a"
            elif support_index == 1:
                job_target = self._offset_point(anchor, goal_dir, -safe_tiles, side_dir, -safe_spread_tiles * side_bias)
                shape_anchor = self._offset_point(anchor, goal_dir, -safe_tiles * 1.10, side_dir, -safe_spread_tiles * side_bias * 0.60)
                name = "support_b"
            else:
                extra_index = support_index - 2
                side = side_bias if extra_index % 2 == 0 else -side_bias
                depth = forward_tiles + 1.2 + 0.8 * (extra_index // 2)
                if extra_index % 3 == 1:
                    depth = -safe_tiles - 0.8 * (extra_index // 2)
                width = spread_tiles + 1.3 + 0.7 * (extra_index // 2)
                job_target = self._offset_point(anchor, goal_dir, depth, side_dir, width * side)
                shape_anchor = self._offset_point(anchor, goal_dir, depth * 0.75, side_dir, width * side * 0.65)
                name = "extra_support"
            jobs.append(
                ScriptedJob(
                    name,
                    self._clamp_to_field(job_target),
                    self._clamp_to_field(shape_anchor),
                    avoid_opponents=True,
                )
            )
        return jobs

    def _scripted_loose_jobs(self, team: str, scripted_players: list[KickPlayer]) -> list[ScriptedJob]:
        ball_point = (float(self.ball_x), float(self.ball_y))
        goal = self._goal_center_for_attack(team)
        goal_dir = self._unit_or_fallback(ball_point, goal, (self._attack_sign_for_team(team), 0.0))
        side_dir = self._perp(goal_dir)
        side_bias = self._style_side_bias_for_dir(team, side_dir)
        jobs: list[ScriptedJob] = [
            ScriptedJob("chaser", self._clamp_to_field(ball_point), self._clamp_to_field(ball_point), avoid_opponents=False)
        ]
        for support_index in range(max(0, len(scripted_players) - 1)):
            if support_index == 0:
                job_target = self._offset_point(
                    ball_point,
                    goal_dir,
                    float(self.SCRIPTED_SUPPORT_FORWARD_TILES) * 0.75,
                    side_dir,
                    float(self.SCRIPTED_SUPPORT_WIDE_TILES) * side_bias,
                )
                shape_anchor = self._offset_point(
                    ball_point,
                    goal_dir,
                    float(self.SCRIPTED_SUPPORT_FORWARD_TILES) * 0.55,
                    side_dir,
                    float(self.SCRIPTED_SUPPORT_WIDE_TILES) * side_bias * 0.65,
                )
                name = "support_a"
            elif support_index == 1:
                job_target = self._offset_point(
                    ball_point,
                    goal_dir,
                    -float(self.SCRIPTED_SUPPORT_SAFE_TILES),
                    side_dir,
                    -float(self.SCRIPTED_SUPPORT_SAFE_WIDE_TILES) * side_bias,
                )
                shape_anchor = self._offset_point(
                    ball_point,
                    goal_dir,
                    -float(self.SCRIPTED_SUPPORT_SAFE_TILES) * 1.15,
                    side_dir,
                    -float(self.SCRIPTED_SUPPORT_SAFE_WIDE_TILES) * side_bias * 0.55,
                )
                name = "safety"
            else:
                extra_index = support_index - 2
                side = side_bias if extra_index % 2 == 0 else -side_bias
                depth = 2.4 + 1.1 * (extra_index // 2)
                if extra_index % 3 == 1:
                    depth = -3.0 - 0.8 * (extra_index // 2)
                width = float(self.SCRIPTED_SUPPORT_WIDE_TILES) + 1.0 + 0.5 * (extra_index // 2)
                job_target = self._offset_point(ball_point, goal_dir, depth, side_dir, width * side)
                shape_anchor = self._offset_point(ball_point, goal_dir, depth * 0.70, side_dir, width * side * 0.60)
                name = "extra_support"
            jobs.append(
                ScriptedJob(
                    name,
                    self._clamp_to_field(job_target),
                    self._clamp_to_field(shape_anchor),
                    avoid_opponents=True,
                )
            )
        return jobs

    def _scripted_defense_jobs(
        self,
        team: str,
        scripted_players: list[KickPlayer],
        carrier: KickPlayer,
    ) -> list[ScriptedJob]:
        carrier_point = (float(carrier.x), float(carrier.y))
        own_goal = self._own_goal_center_for_team(team)
        goal_dir = self._unit_or_fallback(carrier_point, own_goal, (-self._attack_sign_for_team(team), 0.0))
        side_dir = self._perp(goal_dir)
        cover_center = self._point_between(carrier_point, own_goal, 0.35)
        cover_center_anchor = self._point_between(carrier_point, own_goal, 0.46)
        fallback_side = self._style_side_bias_for_dir(team, side_dir)
        dangerous = self._dangerous_offball_opponent(team, carrier)
        if dangerous is not None:
            side_sign = self._side_sign_toward_y(
                side_dir,
                from_y=cover_center[1],
                target_y=float(dangerous.y),
                fallback=fallback_side,
            )
            dangerous_point = (float(dangerous.x), float(dangerous.y))
            cover_side = self._point_between(dangerous_point, own_goal, 0.30)
            cover_side_anchor = self._offset_point(
                cover_center_anchor,
                goal_dir,
                0.3,
                side_dir,
                float(self.SCRIPTED_COVER_WIDE_TILES) * 0.60 * side_sign,
            )
        else:
            side_sign = self._side_sign_toward_y(
                side_dir,
                from_y=cover_center[1],
                target_y=float(self.pitch_center_y),
                fallback=fallback_side,
            )
            cover_side = self._offset_point(
                cover_center,
                goal_dir,
                0.2,
                side_dir,
                float(self.SCRIPTED_COVER_WIDE_TILES) * side_sign,
            )
            cover_side_anchor = self._offset_point(
                cover_center_anchor,
                goal_dir,
                0.5,
                side_dir,
                float(self.SCRIPTED_COVER_WIDE_TILES) * 0.65 * side_sign,
            )

        jobs = [
            ScriptedJob(
                "stopper",
                self._clamp_to_field(
                    self._offset_point(
                        carrier_point,
                        goal_dir,
                        float(self.SCRIPTED_PRESS_OFFSET_TILES),
                        side_dir,
                        0.0,
                    )
                ),
                self._clamp_to_field(
                    self._offset_point(
                        carrier_point,
                        goal_dir,
                        float(self.SCRIPTED_PRESS_OFFSET_TILES) + 0.8,
                        side_dir,
                        0.0,
                    )
                ),
                avoid_opponents=False,
            ),
            ScriptedJob(
                "cover_center",
                self._clamp_to_field(cover_center),
                self._clamp_to_field(cover_center_anchor),
                avoid_opponents=False,
            ),
            ScriptedJob(
                "cover_side",
                self._clamp_to_field(cover_side),
                self._clamp_to_field(cover_side_anchor),
                avoid_opponents=True,
            ),
        ]

        extra_count = max(0, len(scripted_players) - 3)
        for extra_index in range(extra_count):
            side = side_sign if extra_index % 2 == 0 else -side_sign
            ratio = min(0.68, 0.48 + 0.07 * (extra_index // 2))
            width = float(self.SCRIPTED_COVER_WIDE_TILES) * (1.05 + 0.25 * (extra_index // 2))
            lane = self._point_between(carrier_point, own_goal, ratio)
            anchor = self._point_between(carrier_point, own_goal, min(0.74, ratio + 0.08))
            name = "extra_cover" if extra_index % 2 == 0 else "mark_space"
            jobs.append(
                ScriptedJob(
                    name,
                    self._clamp_to_field(self._offset_point(lane, goal_dir, 0.2, side_dir, width * side)),
                    self._clamp_to_field(self._offset_point(anchor, goal_dir, 0.0, side_dir, width * side * 0.70)),
                    avoid_opponents=True,
                )
            )
        return jobs

    def _assign_scripted_jobs(
        self,
        players: list[KickPlayer],
        jobs: list[ScriptedJob],
    ) -> list[tuple[KickPlayer, ScriptedJob]]:
        remaining = list(players)
        assignments: list[tuple[KickPlayer, ScriptedJob]] = []
        for job in jobs:
            if not remaining:
                break
            if job.preferred_player is not None and job.preferred_player in remaining:
                player = job.preferred_player
            else:
                player = min(
                    remaining,
                    key=lambda candidate: (
                        self._distance(candidate.x, candidate.y, job.job_target[0], job.job_target[1]),
                        int(candidate.slot_index),
                    ),
                )
            remaining.remove(player)
            assignments.append((player, job))
        return assignments

    def _final_scripted_targets(
        self,
        assignments: list[tuple[KickPlayer, ScriptedJob]],
        *,
        full_team: list[KickPlayer],
        opponents: list[KickPlayer],
    ) -> dict[KickPlayer, tuple[float, float]]:
        final_targets: dict[KickPlayer, tuple[float, float]] = {}
        for player, job in assignments:
            blended = self._blend_scripted_job_target(job)
            final_targets[player] = self._adjust_scripted_target(
                player,
                blended,
                teammates=[teammate for teammate in full_team if teammate is not player],
                opponents=opponents,
                other_targets=list(final_targets.values()),
                avoid_opponents=bool(job.avoid_opponents),
            )
        return final_targets

    def _maybe_scripted_mistake(self, team: str, action: int) -> int:
        action = int(np.clip(int(action), 0, self.NUM_ACTIONS - 1))
        if self.rng.random() >= self._scripted_mistake_probability_for_team(team):
            return action
        if action == self.ACTION_KICK:
            return self.ACTION_STAY
        if action == self.ACTION_STAY:
            return self.ACTION_STAY
        move_actions = [
            self.ACTION_MOVE_N,
            self.ACTION_MOVE_NE,
            self.ACTION_MOVE_E,
            self.ACTION_MOVE_SE,
            self.ACTION_MOVE_S,
            self.ACTION_MOVE_SW,
            self.ACTION_MOVE_W,
            self.ACTION_MOVE_NW,
        ]
        idx = move_actions.index(action) if action in move_actions else 0
        if self.rng.random() < 0.45:
            return self.ACTION_STAY
        offset = -1 if self.rng.random() < 0.5 else 1
        return int(move_actions[(idx + offset) % len(move_actions)])

    def _cached_scripted_action_valid(self, player: KickPlayer, action: int) -> bool:
        if int(action) != self.ACTION_KICK:
            return True
        return bool(player.has_ball and self.ball_owner is player)

    def _scripted_actions_for_team(
        self,
        team: str,
        scripted_players: list[KickPlayer],
        *,
        force_recompute: bool,
    ) -> dict[KickPlayer, int]:
        if not scripted_players:
            return {}
        cached_actions: dict[KickPlayer, int] = {}
        if not force_recompute:
            cache_valid = True
            for player in scripted_players:
                key = self._player_key(player)
                cached = self._scripted_action_cache.get(key)
                if cached is None or not self._cached_scripted_action_valid(player, int(cached)):
                    cache_valid = False
                    break
                cached_actions[player] = int(cached)
            if cache_valid:
                return cached_actions

        team = str(team)
        full_team = self._team_players(team)
        opponents = self._opponent_players(team)
        owner = self.ball_owner
        actions: dict[KickPlayer, int] = {}

        if owner is not None and owner.team == team:
            jobs = self._scripted_attack_jobs(team, scripted_players, owner)
        elif owner is None:
            jobs = self._scripted_loose_jobs(team, scripted_players)
        else:
            jobs = self._scripted_defense_jobs(team, scripted_players, owner)

        assignments = self._assign_scripted_jobs(scripted_players, jobs)
        final_targets = self._final_scripted_targets(assignments, full_team=full_team, opponents=opponents)
        for player, job in assignments:
            target = final_targets[player]
            if job.name == "carrier" and self.ball_owner is player:
                actions[player] = self._scripted_carrier_action(
                    player,
                    target=target,
                    teammates=[teammate for teammate in full_team if teammate is not player],
                    opponents=opponents,
                )
            else:
                actions[player] = self._scripted_target_action(
                    player,
                    target,
                    teammates=[teammate for teammate in full_team if teammate is not player],
                    opponents=opponents,
                    avoid_opponents=bool(job.avoid_opponents),
                    target_already_adjusted=True,
                )

        for player in scripted_players:
            action = self._maybe_scripted_mistake(team, int(actions.get(player, self.ACTION_STAY)))
            if action == self.ACTION_KICK and not (player.has_ball and self.ball_owner is player):
                action = self.ACTION_STAY
            if action != self.ACTION_KICK:
                self._intended_kick_target.pop(self._player_key(player), None)
            key = self._player_key(player)
            self._scripted_action_cache[key] = int(action)
            actions[player] = int(action)
        return actions

    def _apply_scripted_action_to_player(self, player: KickPlayer, action_idx: int) -> None:
        action_idx = int(np.clip(int(action_idx), 0, self.NUM_ACTIONS - 1))
        self._last_action_by_player_id[int(player.slot_index)] = int(action_idx)
        if action_idx <= self.ACTION_MOVE_NW:
            move_x, move_y = self.ACTION_TO_DIRECTION.get(action_idx, (0.0, 0.0))
            self._move_player(player, move_x, move_y, speed_scale=self._scripted_player_speed_for_team(player.team))
            if math.hypot(move_x, move_y) > 1e-6:
                player.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)
            return
        self._move_player(player, 0.0, 0.0, speed_scale=self._scripted_player_speed_for_team(player.team))
        if action_idx == self.ACTION_KICK and player.has_ball and self.ball_owner is player:
            self._resolve_kick_action(player)

    def _scripted_recompute_due(self, team: str) -> bool:
        due = bool(self._scripted_force_recompute)
        if int(self.steps) % self._scripted_reaction_frames_for_team(team) == 0:
            due = True
        return bool(due)

    def _step_players(self, action):
        left_recompute = self._scripted_recompute_due(self.TEAM_LEFT)
        right_recompute = self._scripted_recompute_due(self.TEAM_RIGHT)
        self._scripted_force_recompute = False
        if self.mode == "human":
            controlled = self._human_controlled_step()
            left_scripted = [player for player in self.left_players if player is not controlled]
            left_actions = self._scripted_actions_for_team(
                self.TEAM_LEFT,
                left_scripted,
                force_recompute=left_recompute,
            )
            right_actions = self._scripted_actions_for_team(
                self.TEAM_RIGHT,
                list(self.right_players),
                force_recompute=right_recompute,
            )
            for player, action_idx in left_actions.items():
                self._apply_scripted_action_to_player(player, int(action_idx))
            for player, action_idx in right_actions.items():
                self._apply_scripted_action_to_player(player, int(action_idx))
        else:
            action = self._rl_team_step(action)
            right_actions = self._scripted_actions_for_team(
                self.TEAM_RIGHT,
                list(self.right_players),
                force_recompute=right_recompute,
            )
            for player, action_idx in right_actions.items():
                self._apply_scripted_action_to_player(player, int(action_idx))
        return action

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
        if self.ball_owner is not None or not self.all_players:
            return
        nearest = min(self.all_players, key=lambda p: (self._distance(p.x, p.y, self.ball_x, self.ball_y), p.slot_index))
        distance = self._distance(nearest.x, nearest.y, self.ball_x, self.ball_y)
        ball_speed = math.hypot(self.ball_vx, self.ball_vy)
        if distance <= self.pickup_range and ball_speed <= max(3.5, 9.0 * self.speed_scale):
            self._set_ball_owner(nearest)

    def _restart_kickoff(self) -> None:
        self.last_touch_team = None
        self.last_touch_player_id = None
        for player in self.all_players:
            player.x = float(player.spawn_x)
            player.y = float(player.spawn_y)
            player.contest_cooldown = 0
            player.has_ball = False
            player.angle = 0.0 if player.team == self.TEAM_LEFT else 180.0
            player.vx = 0.0
            player.vy = 0.0
            player.in_contact = False
        self.ball_x = SCREEN_WIDTH * 0.5
        self.ball_y = self.pitch_center_y
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self._set_ball_owner(None)
        self._reset_controlled_progress_state()
        self.freeze_frames = 0
        self._scripted_action_cache.clear()
        self._scripted_target_cache.clear()
        self._intended_kick_target.clear()
        self._scripted_force_recompute = True
        self._possession_style_by_team[self.TEAM_LEFT] = str(self.rng.choice(self.POSSESSION_STYLES))
        self._possession_style_by_team[self.TEAM_RIGHT] = str(self.rng.choice(self.POSSESSION_STYLES))

    def _restart_throw_in(self, team: str, x: float, y_top: bool) -> None:
        throw_y = TILE_SIZE * 1.2 if y_top else self.pitch_bottom - TILE_SIZE * 1.2
        throw_x = self._clamp(x, TILE_SIZE * 1.5, SCREEN_WIDTH - TILE_SIZE * 1.5)
        player = self._nearest_player(team, throw_x, throw_y)
        player.x = throw_x
        player.y = throw_y
        player.vx = 0.0
        player.vy = 0.0
        target_x = self._attacking_goal_x_for_team(player.team)
        player.angle = self._angle_degrees(player.x, player.y, target_x, self.pitch_center_y)
        self._set_ball_owner(player)
        self._reset_controlled_progress_state()
        self.freeze_frames = self.freeze_after_restart

    def _restart_corner(self, team: str, left_side: bool, top_corner: bool) -> None:
        corner_x = TILE_SIZE * 0.8 if left_side else SCREEN_WIDTH - TILE_SIZE * 0.8
        corner_y = TILE_SIZE * 0.8 if top_corner else self.pitch_bottom - TILE_SIZE * 0.8
        player = self._nearest_player(team, corner_x, corner_y)
        player.x = corner_x
        player.y = corner_y
        player.vx = 0.0
        player.vy = 0.0
        goal_x = self._attacking_goal_x_for_team(team)
        player.angle = self._angle_degrees(player.x, player.y, goal_x, self.pitch_center_y)
        self._set_ball_owner(player)
        self._reset_controlled_progress_state()
        self.freeze_frames = self.freeze_after_restart

    def _restart_goal_kick(self, defending_team: str) -> None:
        goal_x = TILE_SIZE * 1.8 if defending_team == self.TEAM_LEFT else SCREEN_WIDTH - TILE_SIZE * 1.8
        player = self._nearest_player(defending_team, goal_x, self.pitch_center_y)
        player.x = goal_x
        player.y = self.pitch_center_y
        player.vx = 0.0
        player.vy = 0.0
        target_x = SCREEN_WIDTH * 0.45 if defending_team == self.TEAM_LEFT else SCREEN_WIDTH * 0.55
        player.angle = self._angle_degrees(player.x, player.y, target_x, self.pitch_center_y)
        self._set_ball_owner(player)
        self._reset_controlled_progress_state()
        self.freeze_frames = self.freeze_after_restart

    def _sync_team_scores_from_tracker(self) -> None:
        self.left_score = int(self.match_tracker.score(self.TEAM_LEFT))
        self.right_score = int(self.match_tracker.score(self.TEAM_RIGHT))

    def _reset_team_scores(self) -> None:
        self.match_tracker.reset_scores()
        self._sync_team_scores_from_tracker()

    def _increment_team_score(self, team: str) -> None:
        self.match_tracker.increment_score(str(team))
        self._sync_team_scores_from_tracker()

    def _release_owner_out_of_play(self) -> None:
        if self.ball_owner is None:
            return
        if 0.0 <= self.ball_x <= float(SCREEN_WIDTH) and 0.0 <= self.ball_y <= float(self.pitch_bottom):
            return
        exit_x = float(np.clip(float(self.ball_x), -float(self.ball_radius), float(SCREEN_WIDTH) + float(self.ball_radius)))
        exit_y = float(self.ball_y)
        self._set_ball_owner(None)
        self.ball_x = exit_x
        self.ball_y = exit_y
        self.ball_vx = 0.0
        self.ball_vy = 0.0

    def _handle_ball_boundaries(self) -> None:
        left_goal_top, left_goal_bottom = self._goal_bounds_for_defending_team(self.TEAM_LEFT)
        right_goal_top, right_goal_bottom = self._goal_bounds_for_defending_team(self.TEAM_RIGHT)
        if self.ball_x <= 0.0 and left_goal_top <= self.ball_y <= left_goal_bottom:
            self._increment_team_score(self.TEAM_RIGHT)
            self._goal_scored_team = self.TEAM_RIGHT
            self._restart_kickoff()
            return
        if self.ball_x >= SCREEN_WIDTH and right_goal_top <= self.ball_y <= right_goal_bottom:
            self._increment_team_score(self.TEAM_LEFT)
            self._goal_scored_team = self.TEAM_LEFT
            self._restart_kickoff()
            return

        self._release_owner_out_of_play()
        out_top = self.ball_y < 0.0
        out_bottom = self.ball_y > self.pitch_bottom
        if out_top or out_bottom:
            receiving_team = self.TEAM_RIGHT if self.last_touch_team == self.TEAM_LEFT else self.TEAM_LEFT
            self._restart_throw_in(receiving_team, x=self.ball_x, y_top=out_top)
            return

        out_left = self.ball_x < 0.0
        out_right = self.ball_x > SCREEN_WIDTH
        if out_left or out_right:
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

    def _reset_step_events(self) -> None:
        self._goal_scored_team = None

    def _tick(self, action):
        self.window_controller.poll_events_or_raise()
        self._reset_step_events()
        applied_action = action
        if self.freeze_frames > 0:
            self.freeze_frames -= 1
            for player in self.all_players:
                self._set_player_stationary(player)
            self._step_ball()
            self._handle_ball_boundaries()
        else:
            applied_action = self._step_players(action)
            self._resolve_player_contacts()
            self._run_auto_contests()
            self._decay_timers()
            self._step_ball()
            self._try_pickup_free_ball()
            self._handle_ball_boundaries()
        self.steps += 1
        return applied_action

    def _ball_depth_progress(self) -> float:
        width = max(1.0, float(SCREEN_WIDTH))
        return float(np.clip(float(self.ball_x) / width, 0.0, 1.0))

    def _reset_controlled_progress_state(self) -> None:
        self._controlled_progress_frontier = self._ball_depth_progress()
        self._controlled_progress_owner_team = self.physical_owner_team()
        self._controlled_progress_owner_id = self.physical_owner_id()
        self._controlled_progress_frames = 1 if self._controlled_progress_owner_id is not None else 0

    def _compute_controlled_progress_reward(self) -> tuple[float, int | None]:
        depth_now = self._ball_depth_progress()
        owner_team = self.physical_owner_team()
        owner_id = self.physical_owner_id()
        possession_changed = bool(
            owner_team != self._controlled_progress_owner_team
            or owner_id != self._controlled_progress_owner_id
        )
        if possession_changed:
            self._controlled_progress_owner_team = owner_team
            self._controlled_progress_owner_id = owner_id
            self._controlled_progress_frames = 1 if owner_id is not None else 0
            self._controlled_progress_frontier = depth_now
            return 0.0, None
        if owner_team != self.TEAM_LEFT or owner_id is None:
            self._controlled_progress_frames = 0
            self._controlled_progress_frontier = depth_now
            return 0.0, None
        self._controlled_progress_frames = max(0, int(self._controlled_progress_frames)) + 1
        if self._controlled_progress_frames < int(CONTROLLED_PROGRESS_STABLE_STEPS):
            self._controlled_progress_frontier = depth_now
            return 0.0, None
        forward_gain = max(0.0, depth_now - float(self._controlled_progress_frontier))
        clipped_gain = float(np.clip(forward_gain, 0.0, float(CONTROLLED_PROGRESS_STEP_CLIP)))
        self._controlled_progress_frontier = max(float(self._controlled_progress_frontier), depth_now)
        return float(REWARD_PROGRESS_CONTROLLED) * clipped_gain, int(owner_id)

    def _compute_ball_support_reward(
        self,
        *,
        prev_left_positions: dict[int, tuple[float, float]],
        prev_ball_position: tuple[float, float],
    ) -> tuple[float, int | None]:
        candidates = list(self.left_players)
        owner = self.ball_owner
        if owner is not None and owner.team == self.TEAM_LEFT:
            candidates = [player for player in candidates if player is not owner]
        if not candidates:
            return 0.0, None
        support_player = min(
            candidates,
            key=lambda player: (self._distance(player.x, player.y, self.ball_x, self.ball_y), int(player.slot_index)),
        )
        support_player_id = int(support_player.slot_index)
        prev_position = prev_left_positions.get(support_player_id)
        if prev_position is None:
            return 0.0, None
        prev_ball_x, prev_ball_y = prev_ball_position
        target_dist = float(TILE_SIZE) * float(BALL_SUPPORT_TARGET_DIST_TILES)
        prev_dist = self._distance(prev_position[0], prev_position[1], prev_ball_x, prev_ball_y)
        curr_dist = self._distance(support_player.x, support_player.y, self.ball_x, self.ball_y)
        prev_error = abs(prev_dist - target_dist)
        curr_error = abs(curr_dist - target_dist)
        support_improve = float(prev_error - curr_error)
        reward_value = float(BALL_SUPPORT_SCALE) * float(
            np.clip(support_improve, -float(BALL_SUPPORT_CLIP), float(BALL_SUPPORT_CLIP))
        )
        return float(reward_value), int(support_player_id)

    def _compute_team_shape_penalty(self) -> tuple[dict[int, float], float]:
        penalties: dict[int, float] = {}
        if len(self.left_players) < 2:
            return penalties, 0.0
        norm = max(1.0, math.hypot(float(SCREEN_WIDTH), float(self.pitch_height)))
        total = 0.0
        for player in self.left_players:
            nearest = min(
                self._distance(player.x, player.y, teammate.x, teammate.y)
                for teammate in self.left_players
                if teammate is not player
            )
            nearest_norm = float(nearest) / norm
            shortfall = max(0.0, float(TEAM_SHAPE_MIN_DIST_NORM) - nearest_norm)
            penalty = -float(
                np.clip(
                    float(TEAM_SHAPE_LINEAR_COEF) * shortfall
                    + float(TEAM_SHAPE_QUADRATIC_COEF) * shortfall * shortfall,
                    0.0,
                    float(TEAM_SHAPE_CLIP),
                )
            )
            penalties[int(player.slot_index)] = float(penalty)
            total += float(penalty)
        return penalties, float(total)

    def _score_reward(
        self,
        *,
        prev_left_positions: dict[int, tuple[float, float]],
        prev_ball_position: tuple[float, float],
    ) -> tuple[np.ndarray, dict[str, float]]:
        player_count = len(self.left_players)
        rewards = np.zeros((player_count,), dtype=np.float32)
        index_by_player_id = {int(player.slot_index): idx for idx, player in enumerate(self.left_players)}
        reward_breakdown = {
            "progress.reward_controlled": 0.0,
            "support.reward_ball_support": 0.0,
            "shape.penalty_team_shape": 0.0,
            "outcome.reward_score": 0.0,
            "outcome.penalty_concede": 0.0,
        }

        progress_reward, progress_owner_id = self._compute_controlled_progress_reward()
        if progress_owner_id is not None:
            progress_idx = index_by_player_id.get(int(progress_owner_id))
            if progress_idx is not None:
                rewards[int(progress_idx)] += float(progress_reward)
        reward_breakdown["progress.reward_controlled"] = float(progress_reward)

        ball_support_reward, support_player_id = self._compute_ball_support_reward(
            prev_left_positions=prev_left_positions,
            prev_ball_position=prev_ball_position,
        )
        if support_player_id is not None:
            support_idx = index_by_player_id.get(int(support_player_id))
            if support_idx is not None:
                rewards[int(support_idx)] += float(ball_support_reward)
        reward_breakdown["support.reward_ball_support"] = float(ball_support_reward)

        team_shape_penalties, team_shape_total = self._compute_team_shape_penalty()
        for player_id, team_shape_penalty in team_shape_penalties.items():
            player_idx = index_by_player_id.get(int(player_id))
            if player_idx is not None:
                rewards[int(player_idx)] += float(team_shape_penalty)
        reward_breakdown["shape.penalty_team_shape"] = float(team_shape_total)

        if self._goal_scored_team == self.TEAM_LEFT:
            if player_count > 0:
                rewards += float(REWARD_SCORE) / float(player_count)
            reward_breakdown["outcome.reward_score"] = float(REWARD_SCORE)
        elif self._goal_scored_team == self.TEAM_RIGHT:
            if player_count > 0:
                rewards += float(PENALTY_CONCEDE) / float(player_count)
            reward_breakdown["outcome.penalty_concede"] = float(PENALTY_CONCEDE)

        return rewards.astype(np.float32, copy=False), reward_breakdown

    def _auto_select_human_controlled_player(self) -> None:
        for idx, player in enumerate(self.left_players):
            if player.has_ball and self.ball_owner is player:
                self.controlled_index = int(idx)
                return
        if not self.left_players:
            self.controlled_index = 0
            return
        nearest_idx = min(
            range(len(self.left_players)),
            key=lambda idx: (
                self._distance(self.left_players[idx].x, self.left_players[idx].y, self.ball_x, self.ball_y),
                self.left_players[idx].slot_index,
            ),
        )
        self.controlled_index = int(nearest_idx)

    def _player_obs(self, controlled: KickPlayer) -> np.ndarray:
        width = max(1.0, float(SCREEN_WIDTH))
        height = max(1.0, float(self.pitch_height))
        player_vel_norm = max(1.0, self.max_player_speed)
        ball_vel_norm_per_sec = max(1.0, float(self.ball_max_speed) / max(1e-6, self.physics_dt))
        rel_ball_x = float(self.ball_x) - float(controlled.x)
        rel_ball_y = float(self.ball_y) - float(controlled.y)
        ball_dist = math.hypot(rel_ball_x, rel_ball_y)
        ball_angle = math.atan2(rel_ball_y, rel_ball_x) if ball_dist > 1e-9 else math.radians(float(controlled.angle))
        rel_angle = ball_angle - math.radians(float(controlled.angle))
        angle_rad = math.radians(float(controlled.angle))

        opp_goal_x = float(SCREEN_WIDTH)
        own_goal_x = 0.0
        opp_goal_y = float((self.right_goal_top + self.right_goal_bottom) * 0.5)
        own_goal_y = float((self.left_goal_top + self.left_goal_bottom) * 0.5)

        teammates = self._nearest_players(
            self.TEAM_LEFT,
            controlled.x,
            controlled.y,
            k=2,
            exclude=controlled,
        )
        opponents = self._nearest_players(
            self.TEAM_RIGHT,
            controlled.x,
            controlled.y,
            k=3,
        )

        ball_vx_per_sec = float(self.ball_vx) / max(1e-6, self.physics_dt)
        ball_vy_per_sec = float(self.ball_vy) / max(1e-6, self.physics_dt)
        owner_team = self.physical_owner_team()
        feature_values = {
            "self_x_norm": float(clip_signed((2.0 * (float(controlled.x) / width)) - 1.0)),
            "self_y_norm": float(clip_signed((2.0 * ((float(controlled.y) - float(self.pitch_top)) / height)) - 1.0)),
            "self_vx": float(clip_signed(float(controlled.vx) / player_vel_norm)),
            "self_vy": float(clip_signed(float(controlled.vy) / player_vel_norm)),
            "self_theta_cos": float(math.cos(angle_rad)),
            "self_theta_sin": float(math.sin(angle_rad)),
            "self_has_ball": 1.0 if controlled.has_ball and self.ball_owner is controlled else 0.0,
            "tgt_dx": float(clip_signed(rel_ball_x / width)),
            "tgt_dy": float(clip_signed(rel_ball_y / height)),
            "tgt_dist_norm": float(np.clip(ball_dist / max(1.0, math.hypot(width, height)), 0.0, 1.0)),
            "tgt_rel_ang_sin": float(math.sin(rel_angle)),
            "tgt_rel_ang_cos": float(math.cos(rel_angle)),
            "tgt_dvx": float(clip_signed((ball_vx_per_sec - float(controlled.vx)) / ball_vel_norm_per_sec)),
            "tgt_dvy": float(clip_signed((ball_vy_per_sec - float(controlled.vy)) / ball_vel_norm_per_sec)),
            "tgt_owner_left": 1.0 if owner_team == self.TEAM_LEFT else 0.0,
            "tgt_owner_right": 1.0 if owner_team == self.TEAM_RIGHT else 0.0,
            "land_opp_goal_dx": float(clip_signed((opp_goal_x - float(controlled.x)) / width)),
            "land_opp_goal_dy": float(clip_signed((opp_goal_y - float(controlled.y)) / height)),
            "land_own_goal_dx": float(clip_signed((own_goal_x - float(controlled.x)) / width)),
            "land_own_goal_dy": float(clip_signed((own_goal_y - float(controlled.y)) / height)),
        }

        def _encode_ally_neighbors(players: list[KickPlayer], count: int) -> None:
            for idx in range(1, int(count) + 1):
                if idx <= len(players):
                    player = players[idx - 1]
                    dx = float(clip_signed((float(player.x) - float(controlled.x)) / width))
                    dy = float(clip_signed((float(player.y) - float(controlled.y)) / height))
                else:
                    dx = dy = 0.0
                feature_values[f"ally{idx}_dx"] = dx
                feature_values[f"ally{idx}_dy"] = dy

        def _encode_opp_neighbors(players: list[KickPlayer], count: int) -> None:
            for idx in range(1, int(count) + 1):
                if idx <= len(players):
                    player = players[idx - 1]
                    dx = float(clip_signed((float(player.x) - float(controlled.x)) / width))
                    dy = float(clip_signed((float(player.y) - float(controlled.y)) / height))
                    dvx = float(clip_signed((float(player.vx) - float(controlled.vx)) / player_vel_norm))
                    dvy = float(clip_signed((float(player.vy) - float(controlled.vy)) / player_vel_norm))
                else:
                    dx = dy = dvx = dvy = 0.0
                feature_values[f"opp{idx}_dx"] = dx
                feature_values[f"opp{idx}_dy"] = dy
                feature_values[f"opp{idx}_dvx"] = dvx
                feature_values[f"opp{idx}_dvy"] = dvy

        _encode_ally_neighbors(teammates, 2)
        _encode_opp_neighbors(opponents, 3)

        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Kick observation expected {self.OBS_DIM} features, got {obs.shape[0]}.")
        return obs

    def _obs(self) -> np.ndarray:
        if self.mode == "human":
            return self._player_obs(self._controlled_player())
        obs = np.asarray([self._player_obs(player) for player in self.left_players], dtype=np.float32)
        if self.debug_sanity_checks and obs.shape != (len(self.left_players), int(self.OBS_DIM)):
            raise RuntimeError(
                f"Kick obs shape mismatch: expected {(len(self.left_players), int(self.OBS_DIM))}, got {tuple(obs.shape)}."
            )
        return obs

    def _central_player_values(self, player: KickPlayer | None) -> dict[str, float]:
        width = max(1.0, float(SCREEN_WIDTH))
        height = max(1.0, float(self.pitch_height))
        values = {
            "x_norm": 0.0,
            "y_norm": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "theta_cos": 0.0,
            "theta_sin": 0.0,
            "has_ball": 0.0,
            "active": 0.0,
        }
        if player is not None:
            angle_rad = math.radians(float(player.angle))
            values.update(
                {
                    "x_norm": float(clip_signed((2.0 * (float(player.x) / width)) - 1.0)),
                    "y_norm": float(clip_signed((2.0 * ((float(player.y) - float(self.pitch_top)) / height)) - 1.0)),
                    "vx": float(clip_signed(float(player.vx) / max(1.0, self.max_player_speed))),
                    "vy": float(clip_signed(float(player.vy) / max(1.0, self.max_player_speed))),
                    "theta_cos": float(math.cos(angle_rad)),
                    "theta_sin": float(math.sin(angle_rad)),
                    "has_ball": 1.0 if player.has_ball and self.ball_owner is player else 0.0,
                    "active": 1.0,
                }
            )
        return values

    @staticmethod
    def _score_norm(score: int) -> float:
        return float(np.clip(float(score) / 10.0, 0.0, 1.0))

    def get_centralized_state(self, _obs: object | None = None) -> np.ndarray:
        width = max(1.0, float(SCREEN_WIDTH))
        height = max(1.0, float(self.pitch_height))
        owner_team = self.physical_owner_team()
        feature_values: dict[str, float] = {}

        left_by_slot = {int(player.slot_index): player for player in self.left_players}
        for slot_index in range(int(self.MAX_TEAM_PLAYERS)):
            prefix = f"left{slot_index + 1}"
            values = self._central_player_values(left_by_slot.get(slot_index))
            for name, value in values.items():
                feature_values[f"{prefix}_{name}"] = float(value)

        right_by_slot = {int(player.slot_index): player for player in self.right_players}
        for slot_index in range(int(self.MAX_TEAM_PLAYERS)):
            prefix = f"right{slot_index + 1}"
            values = self._central_player_values(right_by_slot.get(slot_index))
            for name, value in values.items():
                feature_values[f"{prefix}_{name}"] = float(value)

        opp_goal_x = float(SCREEN_WIDTH)
        opp_goal_y = float((self.right_goal_top + self.right_goal_bottom) * 0.5)
        own_goal_x = 0.0
        own_goal_y = float((self.left_goal_top + self.left_goal_bottom) * 0.5)
        feature_values.update(
            {
                "tgt_x_norm": float(clip_signed((2.0 * (float(self.ball_x) / width)) - 1.0)),
                "tgt_y_norm": float(clip_signed((2.0 * ((float(self.ball_y) - float(self.pitch_top)) / height)) - 1.0)),
                "tgt_vx": float(clip_signed(float(self.ball_vx) / max(1.0, float(self.ball_max_speed)))),
                "tgt_vy": float(clip_signed(float(self.ball_vy) / max(1.0, float(self.ball_max_speed)))),
                "tgt_owner_left": 1.0 if owner_team == self.TEAM_LEFT else 0.0,
                "tgt_owner_right": 1.0 if owner_team == self.TEAM_RIGHT else 0.0,
                "tgt_owner_free": 1.0 if owner_team is None else 0.0,
                "land_ball_to_opp_goal_dx": float(clip_signed((opp_goal_x - float(self.ball_x)) / width)),
                "land_ball_to_opp_goal_dy": float(clip_signed((opp_goal_y - float(self.ball_y)) / height)),
                "land_ball_to_own_goal_dx": float(clip_signed((own_goal_x - float(self.ball_x)) / width)),
                "land_ball_to_own_goal_dy": float(clip_signed((own_goal_y - float(self.ball_y)) / height)),
                "state_time_norm": float(np.clip(float(self.steps) / max(1.0, float(self.max_steps)), 0.0, 1.0)),
                "state_left_score_norm": self._score_norm(int(self.left_score)),
                "state_right_score_norm": self._score_norm(int(self.right_score)),
                "state_level_norm": float(
                    np.clip(
                        (float(self._current_level) - float(MIN_LEVEL)) / max(1.0, float(MAX_LEVEL - MIN_LEVEL)),
                        0.0,
                        1.0,
                    )
                ),
                "state_team_size_norm": float(np.clip(float(self.team_size) / float(self.MAX_TEAM_PLAYERS), 0.0, 1.0)),
            }
        )

        state = np.asarray(ordered_feature_vector(self.CENTRAL_FEATURE_NAMES, feature_values), dtype=np.float32)
        if state.shape != (int(self.central_obs_dim),):
            raise RuntimeError(
                f"Kick centralized state shape mismatch: expected {(int(self.central_obs_dim),)}, got {tuple(state.shape)}."
            )
        return state

    def centralized_state(self, obs: object | None = None) -> np.ndarray:
        return self.get_centralized_state(obs)

    def reset(self) -> np.ndarray:
        self._apply_level_change(int(self._current_level))
        self._reset_team_scores()
        self.steps = 0
        self.done = False
        self.freeze_frames = 0
        self.controlled_index = 2 if len(self.left_players) >= 3 else 0
        self._prev_kick_down = False
        self.last_action_index = self.ACTION_STAY
        self._last_action_by_player_id = {}
        self._episode_reward_components.reset()
        self._restart_kickoff()
        self._reset_controlled_progress_state()
        obs = self._obs()
        if self.debug_sanity_checks and self.mode != "human" and obs.shape != (len(self.left_players), int(self.OBS_DIM)):
            raise RuntimeError(
                f"Kick reset obs shape mismatch: expected {(len(self.left_players), int(self.OBS_DIM))}, got {tuple(obs.shape)}."
            )
        return obs

    def _draw_pitch(self) -> None:
        pitch_h = self.pitch_height
        pitch_bottom = self.window_controller.top_left_to_bottom(self.pitch_top, pitch_h)
        line_width = float(PITCH_LINE_WIDTH)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_DARK_NEUTRAL)
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
        arcade.draw_lbwh_rectangle_outline(0, penalty_bottom, penalty_depth, penalty_height, COLOR_FOG_GRAY, line_width)
        arcade.draw_lbwh_rectangle_outline(
            SCREEN_WIDTH - penalty_depth,
            penalty_bottom,
            penalty_depth,
            penalty_height,
            COLOR_FOG_GRAY,
            line_width,
        )
        left_goal_h = self.left_goal_half_height * 2.0
        left_goal_bottom = self.window_controller.top_left_to_bottom(self.left_goal_top, left_goal_h)
        arcade.draw_lbwh_rectangle_outline(0, left_goal_bottom, TILE_SIZE, left_goal_h, COLOR_LIGHT_NEUTRAL, line_width)
        right_goal_h = self.right_goal_half_height * 2.0
        right_goal_bottom = self.window_controller.top_left_to_bottom(self.right_goal_top, right_goal_h)
        arcade.draw_lbwh_rectangle_outline(
            SCREEN_WIDTH - TILE_SIZE,
            right_goal_bottom,
            TILE_SIZE,
            right_goal_h,
            COLOR_LIGHT_NEUTRAL,
            line_width,
        )

    def _draw_player(self, player: KickPlayer, *, controlled_marker: bool) -> None:
        if player.team == self.TEAM_LEFT:
            outer = COLOR_AQUA
            inner = COLOR_DEEP_TEAL
        else:
            outer = COLOR_CORAL
            inner = COLOR_BRICK_RED
        draw_two_tone_cell(
            self.window_controller,
            top_left_x=player.x - self.player_half,
            top_left_y=player.y - self.player_half,
            tile_size=self.player_size,
            outer_color=outer,
            inner_color=inner,
            cell_inset=float(CELL_INSET),
        )
        if controlled_marker:
            draw_control_marker(
                self.window_controller,
                center_x=player.x,
                center_y_top_left=player.y,
                marker_size=max(3.0, self.player_size * 0.28),
                color=outer,
            )
        draw_facing_indicator(
            self.window_controller,
            center_x=player.x,
            center_y_top_left=player.y,
            angle_degrees=player.angle,
            length=self.player_size * 0.48,
            color=COLOR_LIGHT_NEUTRAL,
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
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outline_color,
            inner_color=fill_color,
            inset=status_icon_inset(float(CELL_INSET)),
        )

    def _remaining_time_ratio(self) -> float:
        return float(self.match_tracker.remaining_time_ratio(int(self.steps)))

    def _score_icon_items(self) -> list[str]:
        return [self.TEAM_LEFT] * max(0, int(self.left_score)) + [self.TEAM_RIGHT] * max(0, int(self.right_score))

    def _draw_score_icons(self, left: float, right: float, center_y: float) -> None:
        icon_size = self._status_icon_size()
        draw_status_icon_row(
            left=float(left),
            right=float(right),
            center_y=float(center_y),
            icon_size=float(icon_size),
            items=self._score_icon_items(),
            draw_item=lambda team, icon_center_x, row_center_y, size: self._draw_team_icon(
                str(team),
                float(icon_center_x),
                float(row_center_y),
                float(size),
            ),
        )

    def render(self) -> None:
        if self.window_controller.window is None:
            return
        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        self._draw_pitch()
        controlled = self._controlled_player() if self.mode == "human" else None
        for player in self.all_players:
            self._draw_player(player, controlled_marker=(controlled is not None and player is controlled))
        arcade.draw_circle_filled(self.ball_x, self.window_controller.to_arcade_y(self.ball_y), self.ball_radius, COLOR_FOG_GRAY)
        arcade.draw_circle_filled(
            self.ball_x,
            self.window_controller.to_arcade_y(self.ball_y),
            self.ball_radius * 0.62,
            COLOR_SLATE_GRAY,
        )
        bar_layout = draw_status_bar(
            width=float(SCREEN_WIDTH),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=float(CELL_INSET),
            include_clock=True,
        )
        draw_status_clock(layout=bar_layout, remaining_ratio=float(self._remaining_time_ratio()))
        self._draw_score_icons(float(bar_layout.score_left), float(bar_layout.score_right), float(bar_layout.center_y))
        self.window_controller.flip()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            done_reward_vec = np.zeros((len(self.left_players),), dtype=np.float32)
            return self._obs(), 0.0, True, {
                "win": bool(self.left_score > self.right_score),
                "success": int(self._last_episode_success),
                "score_left": int(self.left_score),
                "score_right": int(self.right_score),
                "time_left_ratio": float(self._remaining_time_ratio()),
                "controlled_slot": int(self._controlled_player().slot_index) if self.left_players else 0,
                "level": int(self._last_episode_level),
                "team_size": int(self.team_size),
                "reward_vec": done_reward_vec,
                "reward_breakdown": {},
                "reward_components": self._episode_reward_components.totals(),
            }

        self.window_controller.poll_events_or_raise()
        episode_level = int(self._current_level)
        parsed_action = self._decode_team_actions(action) if self.mode != "human" else self.ACTION_STAY
        compute_scored_breakdown = bool(self.mode != "human")
        reward_vec = np.zeros((len(self.left_players),), dtype=np.float32)
        reward_breakdown: dict[str, float] = {}
        repeat_frames = 1 if self.mode == "human" else int(self.rl_action_repeat_frames)

        for _ in range(max(1, int(repeat_frames))):
            prev_left_positions = (
                {int(player.slot_index): (float(player.x), float(player.y)) for player in self.left_players}
                if compute_scored_breakdown
                else {}
            )
            prev_ball_position = (float(self.ball_x), float(self.ball_y)) if compute_scored_breakdown else (0.0, 0.0)
            self._tick(parsed_action)
            if self.steps >= self.max_steps:
                self.done = True
            if compute_scored_breakdown:
                frame_rewards, frame_breakdown = self._score_reward(
                    prev_left_positions=prev_left_positions,
                    prev_ball_position=prev_ball_position,
                )
                reward_vec = reward_vec + np.asarray(frame_rewards, dtype=np.float32)
                for key, value in frame_breakdown.items():
                    reward_breakdown[key] = float(reward_breakdown.get(key, 0.0) + float(value))
                self._episode_reward_components.add_from_mapping(frame_breakdown, self.REWARD_COMPONENT_KEY_TO_CODE)
            self.render()
            self._tick_arcade_frame()
            if self.done:
                break

        reward = float(reward_vec.sum()) if self.mode != "human" else 0.0
        if self.mode == "human":
            reward_breakdown = {}
        if self.debug_sanity_checks:
            expected_shape = (len(self.left_players),)
            if reward_vec.shape != expected_shape:
                raise RuntimeError(f"Kick reward_vec shape mismatch: expected {expected_shape}, got {tuple(reward_vec.shape)}.")
            if not isinstance(reward, float):
                raise RuntimeError(f"Kick step reward must be scalar float, got {type(reward)!r}.")

        done = bool(self.done)
        win = bool(done and self.left_score > self.right_score)
        success = 1 if win else 0
        info = {
            "win": win,
            "success": int(success) if done else 0,
            "score_left": int(self.left_score),
            "score_right": int(self.right_score),
            "time_left_ratio": float(self._remaining_time_ratio()),
            "controlled_slot": int(self._controlled_player().slot_index) if self.left_players else 0,
            "level": int(episode_level),
            "team_size": int(self.team_size),
            "level_changed": False,
            "reward_vec": reward_vec,
            "reward_breakdown": reward_breakdown,
        }
        if done:
            info["reward_components"] = self._episode_reward_components.totals()
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(success)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(success),
                current_level=int(self._current_level),
                apply_level=self._apply_level_change,
            )
            info["level_changed"] = bool(level_changed)
        return self._obs(), float(reward), bool(done), info

    def close(self) -> None:
        super().close()
