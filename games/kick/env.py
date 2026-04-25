"""Football environment with human and RL control modes."""

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
from core.envs.base import Env
from core.ghost_overlay import ghost_color, update_ghost_overlay_toggle
from core.io_schema import clip_signed, ordered_feature_vector
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_control_marker,
    draw_facing_indicator,
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    status_icon_inset,
    draw_status_square_icon,
    draw_two_tone_tile,
    resolve_circle_collisions,
    status_icon_size,
)
from core.runtime import ArcadeFrameClock, ArcadeWindowController
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
    BALL_SUPPORT_CLIP,
    BALL_SUPPORT_SCALE,
    BALL_SUPPORT_TARGET_DIST_TILES,
    BALL_RADIUS_SCALE,
    CENTRAL_OBS_MASK_DIM,
    CENTRAL_OBS_BALL_FEATURES,
    CENTRAL_OBS_DIM,
    CONTROLLED_PROGRESS_STABLE_STEPS,
    CONTROLLED_PROGRESS_STEP_CLIP,
    CURRICULUM_PROMOTION,
    DEBUG_SANITY_CHECKS,
    GAME_SPEED_SCALE,
    GOALKEEPER_STATIC_ANCHOR_OUTSIDE_GOAL_MARGIN_TILES,
    INPUT_FEATURE_NAMES as KICK_INPUT_FEATURE_NAMES,
    LEVEL_SETTINGS,
    MAX_LEVEL,
    MAX_LEFT_PLAYERS,
    MIN_LEVEL,
    OBS_DIM as KICK_OBS_DIM,
    OBS_NEAREST_OUTFIELD_PLAYERS as KICK_OBS_NEAREST_OUTFIELD_PLAYERS,
    PENALTY_AREA_DEPTH_RATIO,
    PENALTY_AREA_WIDTH_RATIO,
    PENALTY_CONCEDE,
    PLAYER_A_MAX_PX_PER_SEC2,
    PLAYER_V_MAX_PX_PER_SEC,
    PPO_METRICS_LOG_ENABLED,
    PITCH_LINE_WIDTH,
    REWARD_PROGRESS_CONTROLLED,
    REWARD_SCORE,
    RL_ACTION_REPEAT_FRAMES,
    ROLE_ZONE_LINEAR_COEF,
    ROLE_ATTACK_SHIFT_TILES_BY_ROLE as KICK_ROLE_ATTACK_SHIFT_TILES_BY_ROLE,
    ROLE_FEATURE_NAME_BY_ROLE as KICK_ROLE_FEATURE_NAME_BY_ROLE,
    ROLE_NAMES as KICK_ROLE_NAMES,
    ROLE_ZONE_QUADRATIC_COEF,
    ROLE_ZONE_TOL_X,
    ROLE_ZONE_TOL_X_GK,
    ROLE_ZONE_TOL_Y,
    ROLE_ZONE_TOL_Y_GK,
    SHOW_ZONE_TARGET_CLONES,
    STAMINA_DRAIN_SECONDS,
    STAMINA_MAX,
    STAMINA_MIN,
    STAMINA_RECOVER_SECONDS,
    TEAM_SHAPE_CLIP,
    TEAM_SHAPE_LINEAR_COEF,
    TEAM_SHAPE_MIN_DIST_NORM,
    TEAM_SHAPE_QUADRATIC_COEF,
    WINDOW_TITLE,
    ZONE_TARGET_CLONE_ALPHA,
)


validate_curriculum_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
)


@dataclass
class KickPlayer:
    team: str
    role: str
    slot_index: int
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
    """Top-down football environment with curriculum-sized 7v7 teams.

    Human controls:
    - Move: WASD
    - Pass: Space
    - Shoot: Enter
    - Controlled player: auto-switch (left-ball-owner, else closest-to-ball)

    RL controls (Discrete 11):
    - 0: STAY
    - 1..8: MOVE_N, MOVE_NE, MOVE_E, MOVE_SE, MOVE_S, MOVE_SW, MOVE_W, MOVE_NW
    - 9: PASS
    - 10: SHOOT
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
    ACTION_PASS = 9
    ACTION_SHOOT = 10
    INPUT_FEATURE_NAMES = tuple(KICK_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(KICK_ACTION_NAMES)
    OBS_DIM = int(KICK_OBS_DIM)
    ACT_DIM = int(KICK_ACT_DIM)
    NUM_ACTIONS = ACT_DIM

    TEAM_LEFT = "left"
    TEAM_RIGHT = "right"
    ROLE_ORDER = tuple(KICK_ROLE_NAMES)
    ROLE_FEATURE_NAME_BY_ROLE = dict(KICK_ROLE_FEATURE_NAME_BY_ROLE)
    ROLE_GROUP_BY_ROLE = {
        "GK": "GK",
        "LB": "DEF",
        "RB": "DEF",
        "LM": "MID",
        "CM": "MID",
        "RM": "MID",
        "CS": "ATK",
    }
    ROLE_ATTACK_SHIFT_TILES_BY_ROLE = dict(KICK_ROLE_ATTACK_SHIFT_TILES_BY_ROLE)
    ROLE_Y_SHIFT_SCALE_BY_GROUP = {
        "GK": 0.15,
        "DEF": 0.40,
        "MID": 0.60,
        "ATK": 0.50,
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
    OBS_NEAREST_PLAYERS = int(KICK_OBS_NEAREST_OUTFIELD_PLAYERS)
    MATCH_DURATION_SECONDS = 60.0
    RL_ACTION_REPEAT_FRAMES = int(RL_ACTION_REPEAT_FRAMES)
    ANCHOR_SMOOTH_TAU_SECONDS = 0.35
    GOALKEEPER_TRACK_ALPHA = 0.22
    PASS_SPEED_SCALE = 8.5
    PASS_TARGET_MIN_ALIGNMENT = 0.25
    PASS_ASSIST_WEIGHT = 0.72
    SHOT_SPEED_SCALE = 12.0
    SHOT_QUALITY_DISTANCE_RATIO = 0.65
    SHOT_MIN_GOAL_ALIGNMENT = 0.35
    SHOT_LANE_ALIGNMENT_BLEND = 0.05
    SHOT_ASSIST_MIN = 0.66
    SHOT_ASSIST_MAX = 0.96
    SHOT_SPREAD_MIN_DEG = 2.0
    SHOT_SPREAD_MAX_DEG = 14.0
    SCRIPTED_SHOT_QUALITY_THRESHOLD = 0.55
    SCRIPTED_SHOT_THRESHOLD_MIN = 0.50
    SCRIPTED_SHOT_THRESHOLD_MAX = 0.68
    SCRIPTED_SHOT_INTENT_ENTER = 0.42
    SCRIPTED_SHOT_INTENT_EXIT = 0.28
    SCRIPTED_PASS_PRESSURE_RADIUS_TILES = 3.8
    SCRIPTED_PRESSURE_PASS_PROBABILITY = 0.75
    SCRIPTED_DRIBBLE_BURST_MIN_FRAMES = 18
    SCRIPTED_DRIBBLE_BURST_MAX_FRAMES = 34
    SCRIPTED_MOVE_DEADBAND_TILES = 0.35
    SCRIPTED_SUPPORT_X_OFFSET_TILES_BY_GROUP = {
        "DEF": -4.0,
        "MID": 1.5,
        "ATK": 4.2,
    }
    SCRIPTED_DEFENSE_DEPTH_BY_GROUP = {
        "DEF": 0.28,
        "MID": 0.46,
        "ATK": 0.62,
    }
    GOALKEEPER_HOME_DEPTH_TILES = 1.15
    GOALKEEPER_ARC_MIN_DEPTH_TILES = 0.55
    GOALKEEPER_ARC_BASE_DEPTH_TILES = 1.05
    GOALKEEPER_ARC_CLOSE_DEPTH_TILES = 1.45
    GOALKEEPER_HOLD_Y_TILES = 1.2
    GOALKEEPER_HOLD_X_TILES = 0.7
    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.log_ppo_metrics_line = bool(PPO_METRICS_LOG_ENABLED)
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
        self._left_roles = list(self.ROLE_ORDER)
        self._players_left = len(self._left_roles)
        self._opponent_roles = list(self.ROLE_ORDER)
        self._players_opponent = len(self._opponent_roles)
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
        self.base_goal_half_height = TILE_SIZE * 3.0
        self.left_goal_half_height = float(self.base_goal_half_height)
        self.right_goal_half_height = float(self.base_goal_half_height)
        self.left_goal_top = self.pitch_center_y - self.left_goal_half_height
        self.left_goal_bottom = self.pitch_center_y + self.left_goal_half_height
        self.right_goal_top = self.pitch_center_y - self.right_goal_half_height
        self.right_goal_bottom = self.pitch_center_y + self.right_goal_half_height
        self.goal_box_depth = TILE_SIZE * 6.0
        self.player_size = float(TILE_SIZE)
        self.player_half = self.player_size * 0.5
        self.speed_scale = float(GAME_SPEED_SCALE)
        self.ball_radius = max(3.0, TILE_SIZE * 0.2 * float(BALL_RADIUS_SCALE))
        self.ball_drag_offset = TILE_SIZE * 0.58
        self.physics_dt = float(PHYSICS_DT)
        self._anchor_smooth_alpha = 1.0 - math.exp(
            -float(self.physics_dt) / float(self.ANCHOR_SMOOTH_TAU_SECONDS)
        )
        self.player_vmax_base = float(PLAYER_V_MAX_PX_PER_SEC)
        self.player_amax_base = float(PLAYER_A_MAX_PX_PER_SEC2)
        self.ball_max_speed = max(1.0, 14.5 * self.speed_scale)
        self.ball_friction = 0.985
        self.pickup_range = TILE_SIZE * 0.7 * max(0.75, self.speed_scale)
        self.contest_range = TILE_SIZE * 1.45
        self.contest_cooldown_frames = max(1, int(FPS))
        self.freeze_after_restart = 20
        self.max_steps = int(FPS * float(self.MATCH_DURATION_SECONDS))
        self.rl_action_repeat_frames = max(1, int(self.RL_ACTION_REPEAT_FRAMES))
        self.match_tracker = MatchTracker[str](clock_duration_steps=int(self.max_steps))
        self.match_tracker.set_competitors((self.TEAM_LEFT, self.TEAM_RIGHT), preserve_existing=False)
        self.max_player_speed = max(1.0, self.player_vmax_base * STAMINA_MAX)
        self.stamina_drain_per_step = (STAMINA_MAX - STAMINA_MIN) / max(1.0, STAMINA_DRAIN_SECONDS * FPS)
        self.stamina_recover_per_step = (STAMINA_MAX - STAMINA_MIN) / max(1.0, STAMINA_RECOVER_SECONDS * FPS)
        self.player_contact_radius = self.player_size * 0.44
        self.contact_sep_strength = 0.5
        self.contact_overlap_cap = self.player_size * 0.02
        self.contact_damp = 0.08
        self.contact_accel_scale = 0.7
        self.enemy_stamina_scale = 1.0
        self.enemy_recovery_wait_ratio = 0.25
        self.enemy_recovery_resume_ratio = 0.70
        self.enemy_ball_turn_rate_deg = 7.5
        self._level_entropy_coef = 0.0
        self._start_possession_mode = "CEN"
        self.debug_sanity_checks = bool(DEBUG_SANITY_CHECKS)
        self.max_left_players = int(MAX_LEFT_PLAYERS)
        self.central_obs_mask_dim = int(CENTRAL_OBS_MASK_DIM)
        self.central_obs_ball_features = int(CENTRAL_OBS_BALL_FEATURES)
        self.central_obs_dim = int(CENTRAL_OBS_DIM)

        self.left_players: list[KickPlayer] = []
        self.right_players: list[KickPlayer] = []
        self.all_players: list[KickPlayer] = []
        self.left_goalkeeper: KickPlayer | None = None
        self.right_goalkeeper: KickPlayer | None = None
        self._enemy_recovery_mode: dict[int, bool] = {}
        self._keeper_track_y: dict[tuple[str, int], float] = {}
        self._scripted_shot_intent: dict[tuple[str, int], bool] = {}
        self._scripted_shot_threshold: dict[tuple[str, int], float] = {}
        self._scripted_dribble_until_step: dict[tuple[str, int], int] = {}
        self.controlled_index = 0

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_owner: KickPlayer | None = None
        self.ball_last_kick_type = 0
        self._anchor_x: dict[int, float] = {}
        self._anchor_y: dict[int, float] = {}

        self.last_touch_team: str | None = None
        self.last_touch_player_id: int | None = None
        self.left_score = int(self.match_tracker.score(self.TEAM_LEFT))
        self.right_score = int(self.match_tracker.score(self.TEAM_RIGHT))
        self.steps = 0
        self.done = False
        self.freeze_frames = 0
        self.last_action_index = self.ACTION_STAY
        self.show_zone_target_clones = bool(SHOW_ZONE_TARGET_CLONES)
        self._prev_ghost_overlay_toggle_down = False
        self.zone_target_clone_alpha = int(np.clip(int(ZONE_TARGET_CLONE_ALPHA), 0, 255))

        self._goal_scored_team: str | None = None
        self._controlled_progress_frontier = 0.0
        self._controlled_progress_owner_team: str | None = None
        self._controlled_progress_owner_id: int | None = None
        self._controlled_progress_frames = 0
        self._last_action_by_player_id: dict[int, int] = {}

        self._prev_shoot_down = False
        self._prev_pass_down = False
        self._apply_level_change(int(self._current_level))
        self.reset()

    def _default_controlled_index(self) -> int:
        if not self.left_players:
            return 0
        for preferred_role in ("CS", "CM", "LM", "RM", "GK"):
            for idx, player in enumerate(self.left_players):
                if str(player.role) == preferred_role:
                    return int(idx)
        return max(0, len(self.left_players) - 1)

    def get_entropy_coef_for_level(self, level: int | None = None) -> float | None:
        if level is None or int(level) == int(self._current_level):
            return float(self._level_entropy_coef)

        settings = LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Kick.")
        if "entropy_coef" not in settings:
            raise ValueError("Kick LEVEL_SETTINGS entries must define 'entropy_coef'.")
        try:
            return float(settings["entropy_coef"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Kick LEVEL_SETTINGS 'entropy_coef' must be numeric.") from exc

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Kick.")
        left_roles_raw = settings.get("left_roles", self.ROLE_ORDER)
        normalized_left_roles = [str(role) for role in left_roles_raw if str(role) in self.ROLE_ORDER]
        deduped_left_roles: list[str] = []
        for role in normalized_left_roles:
            if role not in deduped_left_roles:
                deduped_left_roles.append(role)
        if not deduped_left_roles:
            deduped_left_roles = list(self.ROLE_ORDER)
        players_left = settings.get("players_left", len(deduped_left_roles))
        if "players_opponent" not in settings:
            raise ValueError("Kick LEVEL_SETTINGS entries must define 'players_opponent'.")
        if "opponent_roles" not in settings:
            raise ValueError("Kick LEVEL_SETTINGS entries must define 'opponent_roles'.")

        opponent_roles = settings["opponent_roles"]
        normalized_roles = [str(role) for role in opponent_roles if str(role) in self.ROLE_ORDER]
        deduped_roles: list[str] = []
        for role in normalized_roles:
            if role not in deduped_roles:
                deduped_roles.append(role)
        if not deduped_roles:
            raise ValueError("Kick LEVEL_SETTINGS opponent_roles must include at least one valid role.")
        players_opponent = max(1, int(settings["players_opponent"]))
        goals_size_scale = settings.get("goals_size_scale", None)
        enemy_stamina_scale = settings.get("enemy_stamina_scale", 1.0)
        if "entropy_coef" not in settings:
            raise ValueError("Kick LEVEL_SETTINGS entries must define 'entropy_coef'.")
        raw_entropy_coef = settings["entropy_coef"]
        try:
            goals_scale = max(0.1, float(goals_size_scale))
        except (TypeError, ValueError):
            goals_scale = 1.0
        own_goal_scale = 1.0 / goals_scale
        opp_goal_scale = goals_scale
        try:
            enemy_stamina_scale_value = float(enemy_stamina_scale)
        except (TypeError, ValueError):
            enemy_stamina_scale_value = 1.0
        self.enemy_stamina_scale = self._clamp(enemy_stamina_scale_value, 0.0, 1.0)
        start_possession_mode = str(settings.get("start_possession", "CEN")).upper()
        if start_possession_mode not in {"CEN", "RND_LEFT", "RND_RIGHT"}:
            raise ValueError(
                "Kick LEVEL_SETTINGS 'start_possession' must be 'CEN', 'RND_LEFT', or 'RND_RIGHT'."
            )
        self._start_possession_mode = start_possession_mode
        try:
            parsed_entropy_coef = float(raw_entropy_coef)
        except (TypeError, ValueError) as exc:
            raise ValueError("Kick LEVEL_SETTINGS 'entropy_coef' must be numeric.") from exc
        self._level_entropy_coef = float(parsed_entropy_coef)
        self._set_goal_sizes(own_goal_scale=own_goal_scale, opp_goal_scale=opp_goal_scale)
        self._current_level = int(level)
        self._players_left = max(1, min(int(players_left), len(deduped_left_roles)))
        self._left_roles = list(deduped_left_roles[: self._players_left])
        self._players_opponent = max(1, min(players_opponent, len(self.ROLE_ORDER)))
        self._opponent_roles = list(deduped_roles[: self._players_opponent])

    def _apply_level_change(self, level: int) -> None:
        self._apply_level_settings(int(level))
        self._build_teams()

    def _build_teams(self) -> None:
        left_roles = list(self._left_roles[: max(1, int(self._players_left))])
        right_roles = list(self._opponent_roles[: max(1, int(self._players_opponent))])
        self.left_players = self._team_for_roles(self.TEAM_LEFT, left_roles)
        self.right_players = self._team_for_roles(self.TEAM_RIGHT, right_roles)
        self.all_players = [*self.left_players, *self.right_players]
        self._enemy_recovery_mode = {int(player.slot_index): False for player in self.right_players}
        self._scripted_shot_intent = {}
        self._scripted_shot_threshold = {}
        self._scripted_dribble_until_step = {}
        self.left_goalkeeper = next((player for player in self.left_players if player.role == "GK"), None)
        self.right_goalkeeper = next((player for player in self.right_players if player.role == "GK"), None)
        self._initialize_goalkeeper_track_state()
        self.controlled_index = int(self._default_controlled_index())

    def _team_for_roles(self, team: str, roles: list[str]) -> list[KickPlayer]:
        center_y = self.pitch_top + self.pitch_height * 0.50
        wide_top_y = self.pitch_top + self.pitch_height * 0.24
        wide_bottom_y = self.pitch_top + self.pitch_height * 0.76
        fullback_top_y = 0.5 * (wide_top_y + center_y)
        fullback_bottom_y = 0.5 * (center_y + wide_bottom_y)

        # Keep both teams organized inside their own half at kickoff, but give
        # each line enough vertical and horizontal spread to avoid clustering.
        goalkeeper_home_x = max(self.player_half, float(TILE_SIZE) * float(self.GOALKEEPER_HOME_DEPTH_TILES))
        line_x_left = {
            "GK": goalkeeper_home_x,
            "D": SCREEN_WIDTH * 0.20,
            "M": SCREEN_WIDTH * 0.34,
            "S": SCREEN_WIDTH * 0.47,
        }
        if team == self.TEAM_RIGHT:
            line_x = {key: SCREEN_WIDTH - value for key, value in line_x_left.items()}
            default_angle = 180.0
        else:
            line_x = line_x_left
            default_angle = 0.0

        placement = {
            "GK": (line_x["GK"], center_y),
            "LB": (line_x["D"], fullback_top_y),
            "RB": (line_x["D"], fullback_bottom_y),
            "LM": (line_x["M"], wide_top_y),
            "CM": (line_x["M"], center_y),
            "RM": (line_x["M"], wide_bottom_y),
            "CS": (line_x["S"], center_y),
        }

        players: list[KickPlayer] = []
        for slot_index, role in enumerate(roles):
            px, py = placement[role]
            players.append(
                KickPlayer(
                    team=team,
                    role=role,
                    slot_index=int(slot_index),
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

    def _set_goal_sizes(self, *, own_goal_scale: float, opp_goal_scale: float) -> None:
        min_half = max(1.0, self.player_half * 0.75)
        max_half = max(min_half, (self.pitch_height * 0.5) - self.player_half)
        self.left_goal_half_height = float(self._clamp(self.base_goal_half_height * own_goal_scale, min_half, max_half))
        self.right_goal_half_height = float(self._clamp(self.base_goal_half_height * opp_goal_scale, min_half, max_half))
        self.left_goal_top = self.pitch_center_y - self.left_goal_half_height
        self.left_goal_bottom = self.pitch_center_y + self.left_goal_half_height
        self.right_goal_top = self.pitch_center_y - self.right_goal_half_height
        self.right_goal_bottom = self.pitch_center_y + self.right_goal_half_height

    def _goal_bounds_for_defending_team(self, team: str) -> tuple[float, float, float]:
        if str(team) == self.TEAM_LEFT:
            return self.left_goal_top, self.left_goal_bottom, self.left_goal_half_height
        return self.right_goal_top, self.right_goal_bottom, self.right_goal_half_height

    def _stamina_cap_for(self, player: KickPlayer) -> float:
        if player.team == self.TEAM_RIGHT:
            return float(STAMINA_MAX) * float(self.enemy_stamina_scale)
        return float(STAMINA_MAX)

    def _enemy_should_recover(self, player: KickPlayer) -> bool:
        if player.team != self.TEAM_RIGHT:
            return False
        stamina_cap = max(0.0, float(self._stamina_cap_for(player)))
        player_key = int(player.slot_index)
        if stamina_cap <= 1e-8:
            self._enemy_recovery_mode[player_key] = True
            return True

        wait_ratio = self._clamp(float(self.enemy_recovery_wait_ratio), 0.0, 1.0)
        resume_ratio = self._clamp(float(self.enemy_recovery_resume_ratio), wait_ratio, 1.0)
        wait_threshold = stamina_cap * wait_ratio
        resume_threshold = stamina_cap * resume_ratio
        in_recovery = bool(self._enemy_recovery_mode.get(player_key, False))

        if in_recovery:
            if float(player.stamina) >= resume_threshold:
                self._enemy_recovery_mode[player_key] = False
                return False
            return True

        if float(player.stamina) <= wait_threshold:
            self._enemy_recovery_mode[player_key] = True
            return True
        return False

    def _left_has_possession(self) -> bool:
        return self.physical_owner_team() == self.TEAM_LEFT

    def physical_owner_team(self) -> str | None:
        if self.ball_owner is None:
            return None
        return str(self.ball_owner.team)

    def physical_owner_id(self) -> int | None:
        if self.ball_owner is None:
            return None
        return int(self.ball_owner.slot_index)

    def effective_possession_team(self) -> str | None:
        owner_team = self.physical_owner_team()
        if owner_team is not None:
            return owner_team
        if self.last_touch_team is not None:
            return str(self.last_touch_team)
        return None

    def _left_outfield_players(self) -> list[KickPlayer]:
        return [player for player in self.left_players if str(player.role) != "GK"]

    def _closest_left_ball_challenger_id(
        self,
        *,
        ball_x: float | None = None,
        ball_y: float | None = None,
        positions_by_player_id: dict[int, tuple[float, float]] | None = None,
    ) -> int | None:
        candidates = self._left_outfield_players()
        if not candidates:
            return None

        target_x = float(self.ball_x if ball_x is None else ball_x)
        target_y = float(self.ball_y if ball_y is None else ball_y)

        def _player_distance(player: KickPlayer) -> tuple[float, int]:
            player_id = int(player.slot_index)
            if positions_by_player_id is not None:
                px, py = positions_by_player_id.get(player_id, (float(player.x), float(player.y)))
            else:
                px, py = float(player.x), float(player.y)
            return float(self._distance(px, py, target_x, target_y)), int(player_id)

        challenger = min(candidates, key=_player_distance)
        return int(challenger.slot_index)

    def _set_ball_owner(self, owner: KickPlayer | None) -> None:
        previous_owner = self.ball_owner
        for player in self.all_players:
            player.has_ball = False
        self.ball_owner = owner
        if owner is not previous_owner:
            self._scripted_shot_intent.clear()
            self._scripted_shot_threshold.clear()
            self._scripted_dribble_until_step.clear()
        if owner is None:
            return
        self.ball_last_kick_type = 0
        owner.has_ball = True
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.last_touch_team = owner.team
        self.last_touch_player_id = int(owner.slot_index)

        if self.mode == "human" and owner.team == self.TEAM_LEFT and owner in self.left_players:
            new_index = self.left_players.index(owner)
            if new_index != self.controlled_index:
                self.controlled_index = new_index

        self._attach_ball_to_owner()

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

    @staticmethod
    def _turn_towards_angle(current_angle: float, target_angle: float, max_delta_degrees: float) -> float:
        max_delta = max(0.0, float(max_delta_degrees))
        current = (float(current_angle) + 360.0) % 360.0
        target = (float(target_angle) + 360.0) % 360.0
        delta = ((target - current + 540.0) % 360.0) - 180.0
        if abs(delta) <= max_delta:
            return target
        step = max_delta if delta > 0.0 else -max_delta
        return (current + step + 360.0) % 360.0

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
        exclude_goalkeeper: bool = False,
    ) -> list[KickPlayer]:
        pool = self.left_players if team == self.TEAM_LEFT else self.right_players
        candidates = [
            player
            for player in pool
            if player is not exclude and (not bool(exclude_goalkeeper) or str(player.role) != "GK")
        ]
        candidates.sort(
            key=lambda player: (
                float(self._distance(player.x, player.y, x, y)),
                int(player.slot_index),
            )
        )
        return candidates[: max(0, int(k))]

    def _debug_validate_nearest_order(
        self,
        *,
        controlled: KickPlayer,
        players: list[KickPlayer],
        label: str,
    ) -> None:
        if not self.debug_sanity_checks:
            return
        keys = [
            (
                float(self._distance(controlled.x, controlled.y, player.x, player.y)),
                int(player.slot_index),
            )
            for player in players
        ]
        if keys != sorted(keys, key=lambda item: (item[0], item[1])):
            raise RuntimeError(f"Kick nearest-player ordering for {label} is not stable by (distance, slot_index).")

    @classmethod
    def _role_group(cls, role: str) -> str:
        role_key = str(role).upper()
        return str(cls.ROLE_GROUP_BY_ROLE.get(role_key, "DEF"))

    @classmethod
    def _role_attack_shift_pixels(cls, role: str) -> float:
        role_key = str(role).upper()
        return float(cls.ROLE_ATTACK_SHIFT_TILES_BY_ROLE.get(role_key, 0.0)) * float(TILE_SIZE)

    @classmethod
    def _role_one_hot_feature_values(cls, role: str) -> dict[str, float]:
        role_key = str(role).upper()
        feature_name = cls.ROLE_FEATURE_NAME_BY_ROLE.get(role_key)
        if feature_name is None:
            raise ValueError(f"Unsupported Kick role '{role}'.")
        feature_values = {name: 0.0 for name in cls.ROLE_FEATURE_NAME_BY_ROLE.values()}
        feature_values[str(feature_name)] = 1.0
        return feature_values

    def _update_stamina(self, player: KickPlayer, moved: bool) -> None:
        previous = float(player.stamina)
        stamina_cap = max(0.0, float(self._stamina_cap_for(player)))
        stamina_floor = float(STAMINA_MIN) if player.team == self.TEAM_LEFT else 0.0
        if stamina_floor > stamina_cap:
            stamina_floor = stamina_cap
        if moved:
            player.stamina = self._clamp(previous - self.stamina_drain_per_step, stamina_floor, stamina_cap)
        else:
            player.stamina = self._clamp(previous + self.stamina_recover_per_step, stamina_floor, stamina_cap)
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

    def _decode_team_actions(self, actions) -> np.ndarray:
        team_size = len(self.left_players)
        if team_size <= 0:
            return np.zeros((0,), dtype=np.int64)

        if np.isscalar(actions):
            if team_size != 1:
                raise ValueError(f"Kick RL mode expects {team_size} actions, got scalar action.")
            action_idx = self._decode_action(actions)
            return np.asarray([action_idx], dtype=np.int64)

        action_array = np.asarray(actions).reshape(-1)
        if int(action_array.size) != int(team_size):
            raise ValueError(
                f"Kick RL mode expects {team_size} actions, got {int(action_array.size)}."
            )
        clipped = np.clip(action_array.astype(np.int64, copy=False), 0, self.NUM_ACTIONS - 1)
        return np.asarray(clipped, dtype=np.int64)

    def get_action_mask(self, _obs: object | None = None) -> np.ndarray:
        if self.mode == "human":
            return np.ones((self.NUM_ACTIONS,), dtype=np.bool_)

        team_size = len(self.left_players)
        if team_size <= 0:
            return np.ones((0, self.NUM_ACTIONS), dtype=np.bool_)

        mask = np.ones((team_size, self.NUM_ACTIONS), dtype=np.bool_)
        for idx, player in enumerate(self.left_players):
            if player.has_ball and self.ball_owner is player:
                continue
            mask[idx, self.ACTION_PASS : self.ACTION_SHOOT + 1] = False
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

    @classmethod
    def _attack_sign_for_team(cls, team: str) -> float:
        return 1.0 if str(team) == cls.TEAM_LEFT else -1.0

    def _attacking_goal_x_for_team(self, team: str) -> float:
        if str(team) == self.TEAM_LEFT:
            return float(SCREEN_WIDTH - self.ball_radius)
        return float(self.ball_radius)

    def _goalkeeper_for_team(self, team: str) -> KickPlayer | None:
        if str(team) == self.TEAM_LEFT:
            return self.left_goalkeeper
        return self.right_goalkeeper

    def _opponent_goalkeeper_for_player(self, player: KickPlayer) -> KickPlayer | None:
        defending_team = self.TEAM_RIGHT if player.team == self.TEAM_LEFT else self.TEAM_LEFT
        return self._goalkeeper_for_team(defending_team)

    def _own_goalkeeper_for_player(self, player: KickPlayer) -> KickPlayer | None:
        return self._goalkeeper_for_team(player.team)

    @staticmethod
    def _goalkeeper_track_key(keeper: KickPlayer) -> tuple[str, int]:
        return str(keeper.team), int(keeper.slot_index)

    def _initialize_goalkeeper_track_state(self) -> None:
        self._keeper_track_y = {}
        for keeper in (self.left_goalkeeper, self.right_goalkeeper):
            if keeper is None:
                continue
            self._keeper_track_y[self._goalkeeper_track_key(keeper)] = float(keeper.y)

    def _goalkeeper_reference_for_player(self, player: KickPlayer) -> tuple[float, float, float]:
        defending_team = self.TEAM_RIGHT if player.team == self.TEAM_LEFT else self.TEAM_LEFT
        goal_top, goal_bottom, _ = self._goal_bounds_for_defending_team(defending_team)
        goal_center_y = float((goal_top + goal_bottom) * 0.5)
        keeper = self._opponent_goalkeeper_for_player(player)
        if keeper is None:
            return self._attacking_goal_x_for_team(player.team), goal_center_y, 0.0
        return float(keeper.x), float(keeper.y), float(keeper.vy)

    def _own_goalkeeper_reference_for_player(self, player: KickPlayer) -> tuple[float, float]:
        keeper = self._own_goalkeeper_for_player(player)
        if keeper is not None:
            return float(keeper.x), float(keeper.y)
        goal_top, goal_bottom, _ = self._goal_bounds_for_defending_team(player.team)
        goal_x = 0.0 if player.team == self.TEAM_LEFT else float(SCREEN_WIDTH)
        return float(goal_x), float((goal_top + goal_bottom) * 0.5)

    def _own_goal_shot_features_for_player(self, player: KickPlayer) -> tuple[float, float]:
        defend_left = player.team == self.TEAM_LEFT
        moving_toward_goal = self.ball_vx < -0.05 if defend_left else self.ball_vx > 0.05
        goal_x = 0.0 if defend_left else float(SCREEN_WIDTH)
        goal_top, goal_bottom, goal_half = self._goal_bounds_for_defending_team(player.team)
        goal_center_y = float((goal_top + goal_bottom) * 0.5)
        if not moving_toward_goal:
            return 0.0, 0.0
        time_to_goal = abs((float(goal_x) - float(self.ball_x)) / max(1e-6, abs(float(self.ball_vx))))
        predicted_y = float(self.ball_y) + float(self.ball_vy) * time_to_goal
        shot_line_dy = float(clip_signed((predicted_y - goal_center_y) / max(1.0, float(goal_half))))
        shot_tti = float(np.clip(1.0 - (time_to_goal / max(1.0, float(FPS) * 2.0)), 0.0, 1.0))
        return shot_line_dy, shot_tti

    def _shape_target_for_player(self, player: KickPlayer) -> tuple[float, float]:
        team_attacking = self.effective_possession_team() == player.team
        return self._player_anchor_position(
            player,
            team_attacking=team_attacking,
            ball_y=float(self.ball_y),
            use_smoothed=(player.team == self.TEAM_LEFT),
        )

    def _scripted_support_target(self, player: KickPlayer, owner: KickPlayer) -> tuple[float, float]:
        attack_sign = self._attack_sign_for_team(player.team)
        role_group = self._role_group(player.role)
        x_offset_tiles = float(self.SCRIPTED_SUPPORT_X_OFFSET_TILES_BY_GROUP.get(role_group, 1.0))
        base_x, base_y = self._shape_target_for_player(player)
        support_x = float(owner.x) + attack_sign * float(TILE_SIZE) * x_offset_tiles

        lane_y = float(player.home_y) + self._ball_y_norm(owner.y) * float(self.pitch_height) * 0.045
        support_y = 0.72 * lane_y + 0.18 * float(owner.y) + 0.10 * float(base_y)
        if abs(support_y - float(owner.y)) < float(TILE_SIZE) * 1.8:
            lane_sign = -1.0 if float(player.home_y) < self.pitch_center_y else 1.0
            support_y += lane_sign * float(TILE_SIZE) * 1.7

        target_x = 0.62 * support_x + 0.38 * float(base_x)
        return (
            float(self._clamp(target_x, self.player_half, float(SCREEN_WIDTH) - self.player_half)),
            float(self._clamp(support_y, self.player_half, self.pitch_bottom - self.player_half)),
        )

    def _scripted_defense_target(self, player: KickPlayer, owner: KickPlayer) -> tuple[float, float]:
        base_x, base_y = self._shape_target_for_player(player)
        own_goal_x = 0.0 if player.team == self.TEAM_LEFT else float(SCREEN_WIDTH)
        role_group = self._role_group(player.role)
        depth = float(self.SCRIPTED_DEFENSE_DEPTH_BY_GROUP.get(role_group, 0.46))
        block_x = own_goal_x + (float(owner.x) - own_goal_x) * depth
        block_y = 0.54 * float(player.home_y) + 0.36 * float(owner.y) + 0.10 * float(self.ball_y)
        target_x = 0.70 * block_x + 0.30 * float(base_x)
        target_y = 0.82 * block_y + 0.18 * float(base_y)
        return (
            float(self._clamp(target_x, self.player_half, float(SCREEN_WIDTH) - self.player_half)),
            float(self._clamp(target_y, self.player_half, self.pitch_bottom - self.player_half)),
        )

    def _scripted_off_ball_target(self, player: KickPlayer, owner: KickPlayer | None) -> tuple[float, float, bool]:
        hunter = self._ball_hunter_for_team(player.team)
        if owner is None:
            if hunter is player:
                return float(self.ball_x), float(self.ball_y), True
            target_x, target_y = self._shape_target_for_player(player)
            return float(target_x), float(target_y), False

        if owner.team == player.team:
            if owner is player:
                return float(player.x), float(player.y), False
            target_x, target_y = self._scripted_support_target(player, owner)
            return float(target_x), float(target_y), False

        if hunter is player:
            return float(owner.x), float(owner.y), True
        target_x, target_y = self._scripted_defense_target(player, owner)
        return float(target_x), float(target_y), False

    def _ball_hunter_for_team(self, team: str) -> KickPlayer | None:
        pool = self.left_players if team == self.TEAM_LEFT else self.right_players
        if not pool:
            return None

        candidates = [player for player in pool if str(player.role) != "GK"]
        if not candidates:
            candidates = list(pool)

        if self.ball_owner is not None:
            target_x = float(self.ball_owner.x)
            target_y = float(self.ball_owner.y)
        else:
            target_x = float(self.ball_x)
            target_y = float(self.ball_y)

        return min(
            candidates,
            key=lambda player: self._distance(player.x, player.y, target_x, target_y),
        )

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
        if player.has_ball and self.ball_owner is player:
            min_y = -self.player_half
            max_y = self.pitch_bottom + self.player_half
        return min_x, max_x, min_y, max_y

    def _handle_ball_owner_touchline_exit(self) -> bool:
        owner = self.ball_owner
        if owner is None:
            return False
        if 0.0 <= float(owner.y) <= float(self.pitch_bottom):
            return False

        exit_x = float(np.clip(float(owner.x), -float(self.ball_radius), float(SCREEN_WIDTH) + float(self.ball_radius)))
        exit_y = float(owner.y)
        self._set_ball_owner(None)
        self.ball_x = exit_x
        self.ball_y = exit_y
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_last_kick_type = 0
        return True

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

    def _pass_speed(self) -> float:
        return float(self.PASS_SPEED_SCALE) * self.speed_scale

    def _shot_speed(self) -> float:
        return float(self.SHOT_SPEED_SCALE) * self.speed_scale

    @staticmethod
    def _facing_unit(player: KickPlayer) -> tuple[float, float]:
        return math.cos(math.radians(float(player.angle))), math.sin(math.radians(float(player.angle)))

    def _shot_alignment(self, player: KickPlayer) -> float:
        target_x, target_y = self._shot_target_point(player)
        to_target_x = float(target_x) - float(player.x)
        to_target_y = float(target_y) - float(player.y)
        distance = math.hypot(to_target_x, to_target_y)
        if distance <= 1e-8:
            return 1.0
        to_target_x /= distance
        to_target_y /= distance
        facing_x, facing_y = self._facing_unit(player)
        return float(np.clip(facing_x * to_target_x + facing_y * to_target_y, -1.0, 1.0))

    def _shot_goal_alignment(self, player: KickPlayer) -> float:
        facing_x, _ = self._facing_unit(player)
        alignment = self._attack_sign_for_team(player.team) * float(facing_x)
        return float(np.clip(alignment, -1.0, 1.0))

    def _forward_distance_to_goal(self, player: KickPlayer) -> float:
        goal_x = self._attacking_goal_x_for_team(player.team)
        return max(0.0, self._attack_sign_for_team(player.team) * (float(goal_x) - float(player.x)))

    def _shot_quality(self, player: KickPlayer) -> float:
        if (not player.has_ball) or self.ball_owner is not player:
            return 0.0

        max_distance = max(1.0, float(SCREEN_WIDTH) * float(self.SHOT_QUALITY_DISTANCE_RATIO))
        distance_score = 1.0 - (self._forward_distance_to_goal(player) / max_distance)
        distance_score = float(np.clip(distance_score, 0.0, 1.0))

        min_alignment = float(self.SHOT_MIN_GOAL_ALIGNMENT)
        goal_alignment = float(self._shot_goal_alignment(player))
        if goal_alignment <= min_alignment:
            return 0.0

        goal_alignment_score = float(
            np.clip((goal_alignment - min_alignment) / max(1e-6, 1.0 - min_alignment), 0.0, 1.0)
        )
        # Diagonal goal-facing directions should be viable; exact lane alignment only
        # gives a small bonus instead of gating the shot.
        goal_alignment_score = math.sqrt(goal_alignment_score)
        lane_alignment = float(self._shot_alignment(player))
        lane_alignment_score = float(np.clip((lane_alignment + 0.25) / 1.25, 0.0, 1.0))
        alignment_score = (
            (1.0 - float(self.SHOT_LANE_ALIGNMENT_BLEND)) * goal_alignment_score
            + float(self.SHOT_LANE_ALIGNMENT_BLEND) * lane_alignment_score
        )
        quality = distance_score * alignment_score
        return float(np.clip(quality, 0.0, 1.0))

    @staticmethod
    def _blend_angle(current_angle: float, target_angle: float, weight: float) -> float:
        current = (float(current_angle) + 360.0) % 360.0
        target = (float(target_angle) + 360.0) % 360.0
        delta = ((target - current + 540.0) % 360.0) - 180.0
        return (current + float(np.clip(weight, 0.0, 1.0)) * delta + 360.0) % 360.0

    def _shot_target_point(self, player: KickPlayer) -> tuple[float, float]:
        defending_team = self.TEAM_RIGHT if player.team == self.TEAM_LEFT else self.TEAM_LEFT
        goal_x = self._attacking_goal_x_for_team(player.team)
        goal_top, goal_bottom, goal_half = self._goal_bounds_for_defending_team(defending_team)
        goal_center_y = float((goal_top + goal_bottom) * 0.5)
        _, keeper_y, _ = self._goalkeeper_reference_for_player(player)
        keeper_offset = float(keeper_y) - goal_center_y
        center_band = float(goal_half) * 0.12
        if keeper_offset < -center_band:
            open_side = 1.0
        elif keeper_offset > center_band:
            open_side = -1.0
        else:
            _, facing_y = self._facing_unit(player)
            if abs(float(facing_y)) >= 0.25:
                open_side = 1.0 if float(facing_y) > 0.0 else -1.0
            else:
                open_side = 1.0 if float(player.y) <= goal_center_y else -1.0

        lane_offset = open_side * float(goal_half) * 0.58
        inset = max(float(self.ball_radius) * 0.9, float(goal_half) * 0.10)
        target_y = self._clamp(goal_center_y + lane_offset, float(goal_top) + inset, float(goal_bottom) - inset)
        return float(goal_x), float(target_y)

    def _shot_angle(self, player: KickPlayer) -> float:
        target_x, target_y = self._shot_target_point(player)
        target_angle = self._angle_degrees(player.x, player.y, target_x, target_y)
        quality = float(self._shot_quality(player))
        assist = 0.0
        if quality > 0.0:
            assist = float(self.SHOT_ASSIST_MIN) + (
                float(self.SHOT_ASSIST_MAX) - float(self.SHOT_ASSIST_MIN)
            ) * quality
        shot_angle = self._blend_angle(player.angle, target_angle, assist)
        spread = float(self.SHOT_SPREAD_MAX_DEG) - (
            float(self.SHOT_SPREAD_MAX_DEG) - float(self.SHOT_SPREAD_MIN_DEG)
        ) * quality
        return float((shot_angle + random.uniform(-spread, spread)) % 360.0)

    def _resolve_pass_action(self, player: KickPlayer) -> None:
        if (not player.has_ball) or self.ball_owner is not player:
            return
        pass_target = self._select_pass_target(player)
        if pass_target is not None:
            target_angle = self._angle_degrees(player.x, player.y, pass_target.x, pass_target.y)
            pass_angle = self._blend_angle(player.angle, target_angle, float(self.PASS_ASSIST_WEIGHT))
        else:
            pass_angle = float(player.angle)
        player.angle = float(pass_angle)
        self._kick_ball(
            player,
            speed=self._pass_speed(),
            angle_degrees=pass_angle,
            kick_type=1,
        )

    def _resolve_shoot_action(self, player: KickPlayer) -> None:
        if (not player.has_ball) or self.ball_owner is not player:
            return
        shot_angle = self._shot_angle(player)
        player.angle = float(shot_angle)
        self._kick_ball(player, speed=self._shot_speed(), angle_degrees=shot_angle, kick_type=2)

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

    def _kick_ball(
        self,
        player: KickPlayer,
        speed: float,
        angle_degrees: float | None = None,
        kick_type: int = 0,
    ) -> None:
        if not player.has_ball or self.ball_owner is not player:
            return
        # If no explicit angle is passed, kick in the current sticky facing direction.
        angle = float(player.angle if angle_degrees is None else angle_degrees) % 360.0
        radians = math.radians(angle)
        self._set_ball_owner(None)
        self.ball_x = player.x + math.cos(radians) * self.ball_drag_offset
        self.ball_y = player.y + math.sin(radians) * self.ball_drag_offset
        self.ball_vx = math.cos(radians) * float(speed)
        self.ball_vy = math.sin(radians) * float(speed)
        self.last_touch_team = player.team
        self.last_touch_player_id = int(player.slot_index)
        self.ball_last_kick_type = int(np.clip(int(kick_type), 0, 3))

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

    def _auto_select_human_controlled_player(self) -> None:
        if not self.left_players:
            return
        if self.ball_owner is not None and self.ball_owner.team == self.TEAM_LEFT and self.ball_owner in self.left_players:
            target_player = self.ball_owner
        else:
            target_player = self._nearest_player(self.TEAM_LEFT, self.ball_x, self.ball_y)
        new_index = self.left_players.index(target_player)
        if new_index != self.controlled_index:
            self.controlled_index = new_index

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
        # Theta is sticky across non-move actions: only meaningful movement updates facing.
        if math.hypot(move_x, move_y) > 1e-6:
            controlled.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)

        pass_down = self.window_controller.is_key_down(arcade.key.SPACE)
        shoot_down = self.window_controller.is_key_down(arcade.key.ENTER)
        if pass_down and not self._prev_pass_down and controlled.has_ball:
            self._resolve_pass_action(controlled)
            self.last_action_index = self.ACTION_PASS
        elif shoot_down and not self._prev_shoot_down and controlled.has_ball:
            self._resolve_shoot_action(controlled)
            self.last_action_index = self.ACTION_SHOOT

        self._prev_pass_down = bool(pass_down)
        self._prev_shoot_down = bool(shoot_down)

    def _apply_rl_action_to_player(self, player: KickPlayer, action_idx: int) -> None:
        action_idx = int(np.clip(int(action_idx), 0, self.NUM_ACTIONS - 1))
        self.last_action_index = int(action_idx)
        player_id = int(player.slot_index)
        self._last_action_by_player_id[player_id] = int(action_idx)

        if action_idx <= self.ACTION_MOVE_NW:
            move_x, move_y = self.ACTION_TO_DIRECTION.get(action_idx, (0.0, 0.0))
            self._move_player(player, move_x, move_y)
            # Sticky theta by design: kicks/stay keep last facing; only movement changes it.
            if math.hypot(move_x, move_y) > 1e-6:
                player.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)
            return

        # Kicks imply no movement; without possession they are ignored as STAY.
        self._move_player(player, 0.0, 0.0)
        if not player.has_ball or self.ball_owner is not player:
            return

        if action_idx == self.ACTION_PASS:
            self._resolve_pass_action(player)
        elif action_idx == self.ACTION_SHOOT:
            self._resolve_shoot_action(player)

    def _rl_team_step(self, actions) -> np.ndarray:
        action_indices = self._decode_team_actions(actions)
        if self.debug_sanity_checks and self.mode == "eval":
            action_mask = self.get_action_mask()
            for idx, action_idx in enumerate(action_indices):
                if int(action_idx) < self.ACTION_PASS:
                    continue
                if idx >= int(action_mask.shape[0]):
                    continue
                if bool(action_mask[idx, int(action_idx)]):
                    continue
                raise RuntimeError(
                    f"Kick eval produced invalid kick for player index {int(idx)} with mask disabled action {int(action_idx)}."
                )
        for player, action_idx in zip(self.left_players, action_indices):
            self._apply_rl_action_to_player(player, int(action_idx))
        return action_indices

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

    def _select_pass_target(self, carrier: KickPlayer) -> KickPlayer | None:
        attack_sign = 1.0 if carrier.team == self.TEAM_LEFT else -1.0
        facing_x, facing_y = self._facing_unit(carrier)
        teammates = [p for p in self.all_players if p.team == carrier.team and p is not carrier]
        defenders = [p for p in self.all_players if p.team != carrier.team]
        best_target: KickPlayer | None = None
        best_score = -1e9
        for teammate in teammates:
            rel_x = float(teammate.x) - float(carrier.x)
            rel_y = float(teammate.y) - float(carrier.y)
            distance = math.hypot(rel_x, rel_y)
            if distance <= TILE_SIZE * 0.8:
                continue
            alignment = (rel_x * facing_x + rel_y * facing_y) / max(1e-6, distance)
            if alignment < float(self.PASS_TARGET_MIN_ALIGNMENT):
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
            progress_norm = float(np.clip((rel_x * attack_sign) / max(1.0, float(SCREEN_WIDTH)), -1.0, 1.0))
            distance_norm = float(np.clip(distance / max(1.0, float(SCREEN_WIDTH)), 0.0, 1.5))
            clearance_score = float(np.clip(min_clearance / max(1.0, float(TILE_SIZE)), 0.0, 4.0))
            score = 1.8 * float(alignment) + 0.16 * clearance_score + 0.15 * progress_norm - 0.25 * distance_norm
            if score > best_score:
                best_score = score
                best_target = teammate
        return best_target

    def _carrier_pressure_count(self, carrier: KickPlayer, defenders: list[KickPlayer]) -> int:
        facing_x = math.cos(math.radians(float(carrier.angle)))
        facing_y = math.sin(math.radians(float(carrier.angle)))
        pressure_radius = float(self.SCRIPTED_PASS_PRESSURE_RADIUS_TILES) * float(TILE_SIZE)
        pressure_count = 0
        for defender in defenders:
            rel_x = float(defender.x) - float(carrier.x)
            rel_y = float(defender.y) - float(carrier.y)
            distance = math.hypot(rel_x, rel_y)
            if distance <= pressure_radius:
                pressure_count += 1
                continue
            forward = rel_x * facing_x + rel_y * facing_y
            if 0.0 < forward <= pressure_radius * 1.35:
                lateral = abs(rel_x * facing_y - rel_y * facing_x)
                if lateral <= TILE_SIZE * 1.9:
                    pressure_count += 1
        return int(pressure_count)

    @staticmethod
    def _scripted_player_key(player: KickPlayer) -> tuple[str, int]:
        return str(player.team), int(player.slot_index)

    def _scripted_shot_threshold_for(self, player: KickPlayer) -> float:
        key = self._scripted_player_key(player)
        threshold = self._scripted_shot_threshold.get(key)
        if threshold is None:
            threshold = random.uniform(
                float(self.SCRIPTED_SHOT_THRESHOLD_MIN),
                float(self.SCRIPTED_SHOT_THRESHOLD_MAX),
            )
            self._scripted_shot_threshold[key] = float(threshold)
        return float(threshold)

    def _scripted_ready_to_shoot(self, player: KickPlayer, shot_quality: float | None = None) -> bool:
        quality = float(self._shot_quality(player) if shot_quality is None else shot_quality)
        return bool(quality >= self._scripted_shot_threshold_for(player))

    def _scripted_wants_shot_lane(self, player: KickPlayer, shot_quality: float) -> bool:
        key = self._scripted_player_key(player)
        active = bool(self._scripted_shot_intent.get(key, False))
        if active:
            active = float(shot_quality) >= float(self.SCRIPTED_SHOT_INTENT_EXIT)
        else:
            active = float(shot_quality) >= float(self.SCRIPTED_SHOT_INTENT_ENTER)
        self._scripted_shot_intent[key] = bool(active)
        return bool(active)

    def _scripted_dribbles_under_pressure(self, player: KickPlayer) -> bool:
        key = self._scripted_player_key(player)
        if int(self.steps) < int(self._scripted_dribble_until_step.get(key, -1)):
            return True

        if random.random() < float(self.SCRIPTED_PRESSURE_PASS_PROBABILITY):
            self._scripted_dribble_until_step.pop(key, None)
            return False

        burst_frames = random.randint(
            int(self.SCRIPTED_DRIBBLE_BURST_MIN_FRAMES),
            int(self.SCRIPTED_DRIBBLE_BURST_MAX_FRAMES),
        )
        self._scripted_dribble_until_step[key] = int(self.steps) + int(burst_frames)
        return True

    def _penalty_area_bounds_for_defending_team(self, team: str) -> tuple[float, float, float, float]:
        penalty_depth = SCREEN_WIDTH * float(PENALTY_AREA_DEPTH_RATIO)
        penalty_height = self.pitch_height * float(PENALTY_AREA_WIDTH_RATIO)
        penalty_top = self.pitch_center_y - penalty_height * 0.5
        penalty_bottom = self.pitch_center_y + penalty_height * 0.5
        if str(team) == self.TEAM_LEFT:
            return 0.0, penalty_depth, penalty_top, penalty_bottom
        return SCREEN_WIDTH - penalty_depth, float(SCREEN_WIDTH), penalty_top, penalty_bottom

    def _goalkeeper_cover_target_y(self, keeper: KickPlayer) -> float:
        goal_top, goal_bottom, goal_half = self._goal_bounds_for_defending_team(keeper.team)
        goal_center_y = float((goal_top + goal_bottom) * 0.5)
        owner = self.ball_owner

        if owner is not None and owner.team != keeper.team:
            if float(self._shot_quality(owner)) >= float(self.SCRIPTED_SHOT_QUALITY_THRESHOLD):
                _, likely_target_y = self._shot_target_point(owner)
                desired_y = 0.65 * float(likely_target_y) + 0.35 * float(owner.y)
            else:
                desired_y = 0.55 * float(self.ball_y) + 0.45 * float(owner.y)
        else:
            desired_y = 0.65 * float(self.ball_y) + 0.35 * goal_center_y

        travel_band = max(float(self.player_half), float(goal_half) * 0.85)
        desired_y = self._clamp(desired_y, goal_center_y - travel_band, goal_center_y + travel_band)
        key = self._goalkeeper_track_key(keeper)
        prev_y = float(self._keeper_track_y.get(key, keeper.y))
        tracked_y = prev_y + float(self.GOALKEEPER_TRACK_ALPHA) * (float(desired_y) - prev_y)
        tracked_y = self._clamp(tracked_y, goal_top + self.player_half, goal_bottom - self.player_half)
        self._keeper_track_y[key] = float(tracked_y)
        return float(tracked_y)

    def _goalkeeper_threat_score(self, keeper: KickPlayer) -> float:
        owner = self.ball_owner
        if owner is not None and owner.team != keeper.team:
            distance_score = 1.0 - (
                self._forward_distance_to_goal(owner) / max(1.0, float(SCREEN_WIDTH) * 0.72)
            )
            distance_score = float(np.clip(distance_score, 0.0, 1.0))
            return float(np.clip(max(float(self._shot_quality(owner)), distance_score * 0.75), 0.0, 1.0))

        defending_left = keeper.team == self.TEAM_LEFT
        goal_x = 0.0 if defending_left else float(SCREEN_WIDTH)
        distance_from_goal = abs(float(self.ball_x) - goal_x)
        free_ball_threat = 1.0 - distance_from_goal / max(1.0, float(SCREEN_WIDTH) * 0.45)
        return float(np.clip(free_ball_threat * 0.45, 0.0, 1.0))

    def _goalkeeper_arc_target_x(self, keeper: KickPlayer, target_y: float) -> float:
        defend_left = keeper.team == self.TEAM_LEFT
        goal_x = 0.0 if defend_left else float(SCREEN_WIDTH)
        goal_top, goal_bottom, goal_half = self._goal_bounds_for_defending_team(keeper.team)
        goal_center_y = float((goal_top + goal_bottom) * 0.5)

        threat = float(self._goalkeeper_threat_score(keeper))
        max_depth_tiles = float(self.GOALKEEPER_ARC_BASE_DEPTH_TILES) + (
            float(self.GOALKEEPER_ARC_CLOSE_DEPTH_TILES) - float(self.GOALKEEPER_ARC_BASE_DEPTH_TILES)
        ) * threat
        max_depth = float(TILE_SIZE) * max_depth_tiles
        min_depth = float(TILE_SIZE) * float(self.GOALKEEPER_ARC_MIN_DEPTH_TILES)
        y_span = max(float(self.player_half), float(goal_half) * 0.95)
        offset_norm = float(np.clip(abs(float(target_y) - goal_center_y) / max(1.0, y_span), 0.0, 1.0))
        arc_scale = math.sqrt(max(0.0, 1.0 - offset_norm * offset_norm))
        depth = min_depth + (max_depth - min_depth) * arc_scale
        target_x = goal_x + (depth if defend_left else -depth)
        return float(self._clamp(target_x, self.player_half, float(SCREEN_WIDTH) - self.player_half))

    def _goalkeeper_holds_cover(self, keeper: KickPlayer, target_x: float, target_y: float) -> bool:
        owner = self.ball_owner
        threat_x = float(owner.x) if owner is not None and owner.team != keeper.team else float(self.ball_x)
        defend_left = keeper.team == self.TEAM_LEFT
        goal_x = 0.0 if defend_left else float(SCREEN_WIDTH)

        if defend_left:
            between_ball_and_goal = goal_x <= float(keeper.x) <= max(goal_x, threat_x)
            depth = float(keeper.x) - goal_x
        else:
            between_ball_and_goal = min(goal_x, threat_x) <= float(keeper.x) <= goal_x
            depth = goal_x - float(keeper.x)

        min_depth = float(TILE_SIZE) * max(0.25, float(self.GOALKEEPER_ARC_MIN_DEPTH_TILES) - 0.20)
        max_depth = float(TILE_SIZE) * (
            float(self.GOALKEEPER_ARC_CLOSE_DEPTH_TILES) + float(self.GOALKEEPER_HOLD_X_TILES)
        )
        y_close = abs(float(keeper.y) - float(target_y)) <= float(TILE_SIZE) * float(self.GOALKEEPER_HOLD_Y_TILES)
        x_close = abs(float(keeper.x) - float(target_x)) <= float(TILE_SIZE) * float(self.GOALKEEPER_HOLD_X_TILES)
        depth_ok = bool(min_depth <= depth <= max_depth)
        return bool(y_close and (x_close or (between_ball_and_goal and depth_ok)))

    def _ai_player_step(self, player: KickPlayer) -> None:
        if self.mode == "human":
            controlled = self._controlled_player()
            if player is controlled:
                return

        if self._enemy_should_recover(player):
            if self.ball_owner is not None and self.ball_owner is not player:
                player.angle = self._angle_degrees(player.x, player.y, self.ball_owner.x, self.ball_owner.y)
            self._move_player(player, 0.0, 0.0)
            return

        if player.role == "GK":
            self._ai_goalkeeper_step(player)
            return

        if player.has_ball:
            defenders = [candidate for candidate in self.all_players if candidate.team != player.team]
            blocker_in_front = self._has_defender_in_front(player, defenders)
            pressure_count = self._carrier_pressure_count(player, defenders)
            attack_sign = self._attack_sign_for_team(player.team)
            turn_rate = self.enemy_ball_turn_rate_deg if player.team == self.TEAM_RIGHT else 18.0
            shot_quality = float(self._shot_quality(player))
            wants_shot_lane = self._scripted_wants_shot_lane(player, shot_quality)
            shot_ready = self._scripted_ready_to_shoot(player, shot_quality)
            under_pressure = bool(blocker_in_front or pressure_count > 0)
            pass_target = self._select_pass_target(player) if under_pressure else None

            if shot_ready and not blocker_in_front:
                self._resolve_shoot_action(player)
                return

            dribble_under_pressure = bool(
                under_pressure and (pass_target is None or self._scripted_dribbles_under_pressure(player))
            )
            if under_pressure and pass_target is not None and not dribble_under_pressure:
                pass_angle = self._angle_degrees(player.x, player.y, pass_target.x, pass_target.y)
                player.angle = pass_angle
                self._resolve_pass_action(player)
                return

            if dribble_under_pressure:
                lane_sign = -1.0 if player.home_y > self.pitch_center_y else 1.0
                target_x = player.x + attack_sign * TILE_SIZE * (3.0 if wants_shot_lane else 3.6)
                target_y = self._clamp(player.y + lane_sign * TILE_SIZE * 2.8, TILE_SIZE, self.pitch_bottom - TILE_SIZE)
            else:
                carry_step = TILE_SIZE * (3.0 if wants_shot_lane else 4.4)
                target_x = self._clamp(player.x + attack_sign * carry_step, self.player_half, SCREEN_WIDTH - self.player_half)
                if wants_shot_lane:
                    _, shot_target_y = self._shot_target_point(player)
                    target_y = self._clamp(
                        0.55 * float(shot_target_y) + 0.45 * float(player.home_y),
                        TILE_SIZE,
                        self.pitch_bottom - TILE_SIZE,
                    )
                else:
                    target_y = self._clamp(
                        0.70 * float(player.home_y) + 0.30 * float(self.pitch_center_y),
                        TILE_SIZE,
                        self.pitch_bottom - TILE_SIZE,
                    )

            desired_angle = self._angle_degrees(player.x, player.y, target_x, target_y)
            player.angle = self._turn_towards_angle(player.angle, desired_angle, turn_rate)
            self._move_player(player, target_x - player.x, target_y - player.y)
            return

        owner = self.ball_owner
        target_x, target_y, should_contest = self._scripted_off_ball_target(player, owner)
        if should_contest and owner is not None and owner.team != player.team:
            if self._distance(player.x, player.y, self.ball_x, self.ball_y) < self.contest_range * 0.9:
                self._attempt_contest(player)

        distance = self._distance(player.x, player.y, target_x, target_y)
        if distance > float(TILE_SIZE) * float(self.SCRIPTED_MOVE_DEADBAND_TILES):
            player.angle = self._angle_degrees(player.x, player.y, target_x, target_y)
            self._move_player(player, target_x - player.x, target_y - player.y)
        else:
            self._move_player(player, 0.0, 0.0)

    def _ai_goalkeeper_step(self, keeper: KickPlayer) -> None:
        target_y = self._goalkeeper_cover_target_y(keeper)
        target_x = self._goalkeeper_arc_target_x(keeper, target_y)

        keeper.angle = self._angle_degrees(keeper.x, keeper.y, self.ball_x, self.ball_y)
        if self._goalkeeper_holds_cover(keeper, target_x, target_y):
            self._move_player(keeper, 0.0, 0.0)
        else:
            self._move_player(keeper, target_x - keeper.x, target_y - keeper.y)

        if keeper.has_ball and self.ball_owner is keeper:
            clear_target = self._select_pass_target(keeper)
            if clear_target is not None:
                keeper.angle = self._angle_degrees(keeper.x, keeper.y, clear_target.x, clear_target.y)
                self._resolve_pass_action(keeper)
            else:
                clear_x = SCREEN_WIDTH * 0.50
                clear_y = self.pitch_center_y
                keeper.angle = self._angle_degrees(keeper.x, keeper.y, clear_x, clear_y)
                self._resolve_pass_action(keeper)

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
            goal_top, goal_bottom, goal_half_height = self._goal_bounds_for_defending_team(keeper.team)
            if self.ball_y < goal_top - TILE_SIZE or self.ball_y > goal_bottom + TILE_SIZE:
                continue

            if self._distance(self.ball_x, self.ball_y, keeper.x, keeper.y) > TILE_SIZE * 1.8:
                continue

            ball_speed = math.hypot(self.ball_vx, self.ball_vy)
            offset_from_keeper = abs(self.ball_y - keeper.y)
            offset_ratio = self._clamp(offset_from_keeper / max(1.0, goal_half_height), 0.0, 1.0)
            speed_ratio = self._clamp(ball_speed / max(1.0, self.ball_max_speed * 1.1), 0.0, 1.0)
            catch_radius = TILE_SIZE * (1.0 + 0.55 * (1.0 - offset_ratio) - 0.20 * speed_ratio)
            if self._distance(self.ball_x, self.ball_y, keeper.x, keeper.y) > catch_radius:
                continue

            self._set_ball_owner(keeper)
            break

    def _restart_kickoff(self, _kickoff_team: str) -> None:
        self.last_touch_team = None
        self.last_touch_player_id = None
        for player in self.all_players:
            player.x = player.home_x
            player.y = player.home_y
            player.contest_cooldown = 0
            player.has_ball = False
            player.angle = 0.0 if player.team == self.TEAM_LEFT else 180.0
            player.vx = 0.0
            player.vy = 0.0
            player.stamina = self._stamina_cap_for(player)
            player.stamina_delta = 0.0
            player.in_contact = False

        self.ball_x = SCREEN_WIDTH * 0.5
        self.ball_y = self.pitch_center_y
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_last_kick_type = 0
        self._set_ball_owner(None)
        self._initialize_goalkeeper_track_state()
        self._reset_controlled_progress_state()

        self.freeze_frames = self.freeze_after_restart

    def _random_left_outfield_player(self) -> KickPlayer | None:
        candidates = [player for player in self.left_players if str(player.role) != "GK"]
        if candidates:
            return random.choice(candidates)
        if self.left_goalkeeper is not None:
            return self.left_goalkeeper
        if self.left_players:
            return self.left_players[0]
        return None

    def _random_right_outfield_player(self) -> KickPlayer | None:
        candidates = [player for player in self.right_players if str(player.role) != "GK"]
        if candidates:
            return random.choice(candidates)
        if self.right_goalkeeper is not None:
            return self.right_goalkeeper
        if self.right_players:
            return self.right_players[0]
        return None

    def _seed_easy_level_start_possession(self) -> None:
        if str(self._start_possession_mode) == "CEN":
            return
        player = (
            self._random_left_outfield_player()
            if str(self._start_possession_mode) == "RND_LEFT"
            else self._random_right_outfield_player()
        )
        if player is None:
            return
        player.vx = 0.0
        player.vy = 0.0
        target_x = SCREEN_WIDTH - self.ball_radius if player.team == self.TEAM_LEFT else self.ball_radius
        player.angle = self._angle_degrees(player.x, player.y, target_x, self.pitch_center_y)
        self._set_ball_owner(player)

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
        goal_x = SCREEN_WIDTH - self.ball_radius if team == self.TEAM_LEFT else self.ball_radius
        goal_y = self.pitch_center_y
        player.angle = self._angle_degrees(player.x, player.y, goal_x, goal_y)
        self._set_ball_owner(player)
        self._reset_controlled_progress_state()
        self.freeze_frames = self.freeze_after_restart

    def _restart_goal_kick(self, defending_team: str) -> None:
        if defending_team == self.TEAM_LEFT:
            keeper = self.left_goalkeeper
        else:
            keeper = self.right_goalkeeper
            if keeper is None and self.right_players:
                keeper = self._nearest_player(self.TEAM_RIGHT, SCREEN_WIDTH - TILE_SIZE * 1.8, self.pitch_center_y)

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
        self._keeper_track_y[self._goalkeeper_track_key(keeper)] = float(keeper.y)
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

    def _handle_ball_boundaries(self) -> None:
        if self.ball_owner is not None:
            return

        left_goal_top, left_goal_bottom, _ = self._goal_bounds_for_defending_team(self.TEAM_LEFT)
        right_goal_top, right_goal_bottom, _ = self._goal_bounds_for_defending_team(self.TEAM_RIGHT)
        if self.ball_x <= 0.0 and left_goal_top <= self.ball_y <= left_goal_bottom:
            self._increment_team_score(self.TEAM_RIGHT)
            self._goal_scored_team = self.TEAM_RIGHT
            self._restart_kickoff(self.TEAM_LEFT)
            return
        if self.ball_x >= SCREEN_WIDTH and right_goal_top <= self.ball_y <= right_goal_bottom:
            self._increment_team_score(self.TEAM_LEFT)
            self._goal_scored_team = self.TEAM_LEFT
            self._restart_kickoff(self.TEAM_RIGHT)
            return

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

    def _step_players(self, action):
        if self.mode == "human":
            self._human_controlled_step()
            for player in self.all_players:
                self._ai_player_step(player)
        else:
            action = self._rl_team_step(action)
            for player in self.right_players:
                self._ai_player_step(player)

        return action

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
        else:
            applied_action = self._step_players(action)
            self._resolve_player_contacts()
            self._run_auto_contests()
            self._decay_timers()
            if not self._handle_ball_owner_touchline_exit():
                self._step_ball()
                self._try_pickup_free_ball()
                self._try_goalkeeper_catch()
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

    # Reward only stable physical LEFT possession moving the ball forward.
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
        reward_value = float(REWARD_PROGRESS_CONTROLLED) * clipped_gain
        return float(reward_value), int(owner_id)

    # Encourage one LEFT outfielder to stay near the active play at a useful
    # support distance instead of collapsing directly onto the ball.
    def _compute_ball_support_reward(
        self,
        *,
        prev_left_positions: dict[int, tuple[float, float]],
        prev_ball_position: tuple[float, float],
    ) -> tuple[float, int | None]:
        candidates = self._left_outfield_players()
        owner = self.ball_owner
        if owner is not None and owner.team == self.TEAM_LEFT:
            candidates = [player for player in candidates if player is not owner]
        if not candidates:
            return 0.0, None

        support_player = min(
            candidates,
            key=lambda player: self._distance(float(player.x), float(player.y), float(self.ball_x), float(self.ball_y)),
        )
        support_player_id = int(support_player.slot_index)
        prev_position = prev_left_positions.get(support_player_id)
        if prev_position is None:
            return 0.0, None

        prev_ball_x, prev_ball_y = prev_ball_position
        target_dist = float(TILE_SIZE) * float(BALL_SUPPORT_TARGET_DIST_TILES)
        prev_dist = float(self._distance(prev_position[0], prev_position[1], prev_ball_x, prev_ball_y))
        curr_dist = float(self._distance(support_player.x, support_player.y, self.ball_x, self.ball_y))
        prev_error = abs(prev_dist - target_dist)
        curr_error = abs(curr_dist - target_dist)
        support_improve = float(prev_error - curr_error)
        reward_value = float(BALL_SUPPORT_SCALE) * float(
            np.clip(support_improve, -float(BALL_SUPPORT_CLIP), float(BALL_SUPPORT_CLIP))
        )
        return float(reward_value), int(support_player_id)

    def _ball_y_norm(self, ball_y: float) -> float:
        height = max(1.0, float(self.pitch_height))
        centered = (2.0 * ((float(ball_y) - float(self.pitch_top)) / height)) - 1.0
        return float(clip_signed(centered))

    def _raw_anchor_target(
        self,
        player: KickPlayer,
        *,
        team_attacking: bool,
        ball_y: float,
    ) -> tuple[float, float]:
        if str(player.role).upper() == "GK":
            goal_depth = float(TILE_SIZE)
            margin = float(GOALKEEPER_STATIC_ANCHOR_OUTSIDE_GOAL_MARGIN_TILES) * float(TILE_SIZE)
            tol_x, _ = self._role_zone_tolerances_for_player(player)
            safe_zone_half_width = 0.5 * max(self.player_size, 2.0 * float(tol_x) * float(SCREEN_WIDTH))
            if player.team == self.TEAM_LEFT:
                target_x = goal_depth + safe_zone_half_width + margin
            else:
                target_x = float(SCREEN_WIDTH) - goal_depth - safe_zone_half_width - margin
            return (
                float(self._clamp(target_x, self.player_half, SCREEN_WIDTH - self.player_half)),
                float(self._clamp(float(player.home_y), self.player_half, self.pitch_bottom - self.player_half)),
            )

        attack_direction = 1.0 if player.team == self.TEAM_LEFT else -1.0
        attack_shift = self._role_attack_shift_pixels(player.role) if bool(team_attacking) else 0.0
        target_x = float(player.home_x) + attack_direction * float(attack_shift)

        role_group = self._role_group(player.role)
        y_shift_scale = float(self.ROLE_Y_SHIFT_SCALE_BY_GROUP.get(role_group, 0.40))
        max_y_shift = float(self.pitch_height) * 0.08 * y_shift_scale
        target_y = float(player.home_y) + self._ball_y_norm(ball_y) * max_y_shift

        return (
            float(self._clamp(target_x, self.player_half, SCREEN_WIDTH - self.player_half)),
            float(self._clamp(target_y, self.player_half, self.pitch_bottom - self.player_half)),
        )

    def _role_zone_tolerances_for_player(self, player: KickPlayer) -> tuple[float, float]:
        if str(player.role) == "GK":
            return float(ROLE_ZONE_TOL_X_GK), float(ROLE_ZONE_TOL_Y_GK)
        return float(ROLE_ZONE_TOL_X), float(ROLE_ZONE_TOL_Y)

    def _anchor_offset_norm_for_player(
        self,
        player: KickPlayer,
        *,
        team_attacking: bool,
        ball_y: float,
        use_smoothed: bool = True,
    ) -> tuple[float, float]:
        target_x, target_y = self._player_anchor_position(
            player,
            team_attacking=team_attacking,
            ball_y=ball_y,
            use_smoothed=bool(use_smoothed),
        )
        width = max(1.0, float(SCREEN_WIDTH))
        height = max(1.0, float(self.pitch_height))
        return (
            float(clip_signed((float(target_x) - float(player.x)) / width)),
            float(clip_signed((float(target_y) - float(player.y)) / height)),
        )

    def _role_zone_distance_for_player(
        self,
        player: KickPlayer,
        *,
        team_attacking: bool,
        ball_y: float,
        use_smoothed: bool = True,
    ) -> float:
        dx, dy = self._anchor_offset_norm_for_player(
            player,
            team_attacking=team_attacking,
            ball_y=ball_y,
            use_smoothed=bool(use_smoothed),
        )
        tol_x, tol_y = self._role_zone_tolerances_for_player(player)
        return float(math.sqrt((float(dx) / tol_x) ** 2 + (float(dy) / tol_y) ** 2))

    def _role_zone_penalty_for_player(
        self,
        player: KickPlayer,
        *,
        team_attacking: bool,
        ball_y: float,
    ) -> float:
        zone_distance = self._role_zone_distance_for_player(
            player,
            team_attacking=team_attacking,
            ball_y=ball_y,
            use_smoothed=True,
        )
        if zone_distance <= 1.0:
            return 0.0
        excess = float(zone_distance) - 1.0
        return -(
            float(ROLE_ZONE_LINEAR_COEF) * excess
            + float(ROLE_ZONE_QUADRATIC_COEF) * excess * excess
        )

    def _should_skip_role_zone_penalty(self, player: KickPlayer, *, challenger_id: int | None) -> bool:
        if self.ball_owner is player:
            return True
        if self.effective_possession_team() != self.TEAM_LEFT and challenger_id is not None:
            return bool(int(player.slot_index) == int(challenger_id))
        return False

    # Penalize only the outfield players whose nearest teammate is too close,
    # which acts as a small anti-clumping regularizer.
    def _compute_team_shape_penalty(self) -> tuple[dict[int, float], float]:
        outfield_players = self._left_outfield_players()
        if len(outfield_players) < 2:
            return {}, 0.0

        pitch_width = max(1.0, float(SCREEN_WIDTH))
        penalties_by_player_id: dict[int, float] = {}
        total_penalty = 0.0
        for player in outfield_players:
            nearest_dist = min(
                self._distance(float(player.x), float(player.y), float(teammate.x), float(teammate.y))
                for teammate in outfield_players
                if teammate is not player
            )
            nearest_dist_norm = float(nearest_dist) / pitch_width
            shortfall = max(0.0, float(TEAM_SHAPE_MIN_DIST_NORM) - nearest_dist_norm)
            if shortfall <= 0.0:
                continue
            penalty_mag = (
                float(TEAM_SHAPE_LINEAR_COEF) * shortfall
                + float(TEAM_SHAPE_QUADRATIC_COEF) * shortfall * shortfall
            )
            penalty = -float(np.clip(penalty_mag, 0.0, float(TEAM_SHAPE_CLIP)))
            player_id = int(player.slot_index)
            penalties_by_player_id[player_id] = penalty
            total_penalty += penalty
        return penalties_by_player_id, float(total_penalty)

    # Reuse the anchor ellipse as a weak role prior, but keep it light enough
    # that direct play can still break shape when needed.
    def _compute_role_zone_penalty(self, *, challenger_id: int | None) -> tuple[dict[int, float], float]:
        if not self.left_players:
            return {}, 0.0

        left_attacking = self.effective_possession_team() == self.TEAM_LEFT
        ball_y = float(self.ball_y)
        penalties_by_player_id: dict[int, float] = {}
        total_penalty = 0.0
        for player in self.left_players:
            if self._should_skip_role_zone_penalty(player, challenger_id=challenger_id):
                continue
            role_zone_penalty = self._role_zone_penalty_for_player(
                player,
                team_attacking=left_attacking,
                ball_y=ball_y,
            )
            if role_zone_penalty == 0.0:
                continue
            player_id = int(player.slot_index)
            penalties_by_player_id[player_id] = float(role_zone_penalty)
            total_penalty += float(role_zone_penalty)
        return penalties_by_player_id, float(total_penalty)

    @staticmethod
    def _anchor_player_key(player: KickPlayer) -> int:
        return int(player.slot_index)

    def _initialize_anchor_state(self) -> None:
        if not self.left_players:
            self._anchor_x = {}
            self._anchor_y = {}
            return

        left_attacking = self.effective_possession_team() == self.TEAM_LEFT
        ball_y = float(self.ball_y)
        next_anchor_x: dict[int, float] = {}
        next_anchor_y: dict[int, float] = {}
        for player in self.left_players:
            key = self._anchor_player_key(player)
            target_x, target_y = self._raw_anchor_target(
                player,
                team_attacking=left_attacking,
                ball_y=ball_y,
            )
            next_anchor_x[key] = float(target_x)
            next_anchor_y[key] = float(target_y)
        self._anchor_x = next_anchor_x
        self._anchor_y = next_anchor_y

    def _update_anchor_state(self) -> None:
        if not self.left_players:
            self._anchor_x = {}
            self._anchor_y = {}
            return

        left_attacking = self.effective_possession_team() == self.TEAM_LEFT
        ball_y = float(self.ball_y)
        alpha = float(self._anchor_smooth_alpha)
        prev_anchor_x = self._anchor_x
        prev_anchor_y = self._anchor_y
        next_anchor_x: dict[int, float] = {}
        next_anchor_y: dict[int, float] = {}
        for player in self.left_players:
            key = self._anchor_player_key(player)
            target_x, target_y = self._raw_anchor_target(
                player,
                team_attacking=left_attacking,
                ball_y=ball_y,
            )
            prev_x = float(prev_anchor_x.get(key, target_x))
            prev_y = float(prev_anchor_y.get(key, target_y))
            next_anchor_x[key] = float(prev_x + alpha * (float(target_x) - prev_x))
            next_anchor_y[key] = float(prev_y + alpha * (float(target_y) - prev_y))
        self._anchor_x = next_anchor_x
        self._anchor_y = next_anchor_y

    def _player_anchor_position(
        self,
        player: KickPlayer,
        *,
        team_attacking: bool,
        ball_y: float,
        use_smoothed: bool,
    ) -> tuple[float, float]:
        target_x, target_y = self._raw_anchor_target(
            player,
            team_attacking=team_attacking,
            ball_y=ball_y,
        )
        if not bool(use_smoothed):
            return float(target_x), float(target_y)
        key = self._anchor_player_key(player)
        return (
            float(self._anchor_x.get(key, target_x)),
            float(self._anchor_y.get(key, target_y)),
        )

    def _can_toggle_visual_overlay(self) -> bool:
        return bool(self.show_game and self.mode in {"human", "eval"})

    def _update_visual_overlay_toggle(self) -> None:
        self.show_zone_target_clones, self._prev_ghost_overlay_toggle_down = update_ghost_overlay_toggle(
            window_controller=self.window_controller,
            visible=bool(self.show_zone_target_clones),
            previous_down=bool(self._prev_ghost_overlay_toggle_down),
            enabled=bool(self._can_toggle_visual_overlay()),
        )

    def _should_draw_zone_target_clones(self) -> bool:
        return bool(self.show_zone_target_clones and self.show_game and self.mode != "train")

    def _draw_zone_target_clones(self) -> None:
        if not self._should_draw_zone_target_clones():
            return
        if not self.left_players:
            return

        clone_color = ghost_color(int(self.zone_target_clone_alpha))
        safe_zone_color = ghost_color(128)
        left_attacking = self.effective_possession_team() == self.TEAM_LEFT
        ball_y = float(self.ball_y)
        inset = float(CELL_INSET)

        for player in self.left_players:
            target_x, target_y = self._player_anchor_position(
                player,
                team_attacking=left_attacking,
                ball_y=ball_y,
                use_smoothed=True,
            )
            tol_x, tol_y = self._role_zone_tolerances_for_player(player)
            safe_zone_width = max(self.player_size, 2.0 * float(tol_x) * float(SCREEN_WIDTH))
            safe_zone_height = max(self.player_size, 2.0 * float(tol_y) * float(self.pitch_height))
            arcade.draw_ellipse_filled(
                target_x,
                self.window_controller.to_arcade_y(target_y),
                safe_zone_width,
                safe_zone_height,
                safe_zone_color,
            )
            arcade.draw_line(
                player.x,
                self.window_controller.to_arcade_y(player.y),
                target_x,
                self.window_controller.to_arcade_y(target_y),
                clone_color,
                1.5,
            )
            draw_two_tone_tile(
                self.window_controller,
                top_left_x=target_x - self.player_half,
                top_left_y=target_y - self.player_half,
                size=self.player_size,
                outer_color=clone_color,
                inner_color=clone_color,
                inset=inset,
            )

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
            "shape.penalty_role_zone": 0.0,
            "outcome.reward_score": 0.0,
            "outcome.penalty_concede": 0.0,
        }

        active_challenger_id = None
        if self.effective_possession_team() != self.TEAM_LEFT:
            active_challenger_id = self._closest_left_ball_challenger_id()

        progress_reward, progress_owner_id = self._compute_controlled_progress_reward()
        progress_reward = float(progress_reward)
        if progress_owner_id is not None:
            progress_idx = index_by_player_id.get(int(progress_owner_id))
            if progress_idx is not None:
                rewards[int(progress_idx)] += progress_reward
        reward_breakdown["progress.reward_controlled"] = float(progress_reward)

        ball_support_reward, support_player_id = self._compute_ball_support_reward(
            prev_left_positions=prev_left_positions,
            prev_ball_position=prev_ball_position,
        )
        ball_support_reward = float(ball_support_reward)
        if ball_support_reward != 0.0 and support_player_id is not None:
            support_idx = index_by_player_id.get(int(support_player_id))
            if support_idx is not None:
                rewards[int(support_idx)] += ball_support_reward
        reward_breakdown["support.reward_ball_support"] = float(ball_support_reward)

        team_shape_penalties, team_shape_total = self._compute_team_shape_penalty()
        for player_id, team_shape_penalty in team_shape_penalties.items():
            player_idx = index_by_player_id.get(int(player_id))
            if player_idx is not None:
                rewards[int(player_idx)] += float(team_shape_penalty)
        reward_breakdown["shape.penalty_team_shape"] = float(team_shape_total)

        role_zone_penalties, role_zone_total = self._compute_role_zone_penalty(
            challenger_id=active_challenger_id
        )
        for player_id, role_zone_penalty in role_zone_penalties.items():
            player_idx = index_by_player_id.get(int(player_id))
            if player_idx is not None:
                rewards[int(player_idx)] += float(role_zone_penalty)
        reward_breakdown["shape.penalty_role_zone"] = float(role_zone_total)

        if self._goal_scored_team == self.TEAM_LEFT:
            team_reward = float(REWARD_SCORE)
            if player_count > 0:
                rewards += team_reward / float(player_count)
            reward_breakdown["outcome.reward_score"] = float(REWARD_SCORE)
        elif self._goal_scored_team == self.TEAM_RIGHT:
            team_penalty = float(PENALTY_CONCEDE)
            if player_count > 0:
                rewards += team_penalty / float(player_count)
            reward_breakdown["outcome.penalty_concede"] = float(PENALTY_CONCEDE)

        return rewards.astype(np.float32, copy=False), reward_breakdown

    def _player_obs(self, controlled: KickPlayer) -> np.ndarray:
        width = max(1.0, float(SCREEN_WIDTH))
        height = max(1.0, float(self.pitch_height))
        player_vel_norm = max(1.0, self.max_player_speed)
        ball_vel_norm = max(1.0, self.ball_max_speed)
        nearest_count = int(self.OBS_NEAREST_PLAYERS)
        teammates = self._nearest_players(
            self.TEAM_LEFT,
            controlled.x,
            controlled.y,
            k=nearest_count,
            exclude=controlled,
            exclude_goalkeeper=True,
        )
        opponents = self._nearest_players(
            self.TEAM_RIGHT,
            controlled.x,
            controlled.y,
            k=nearest_count,
            exclude_goalkeeper=True,
        )
        self._debug_validate_nearest_order(controlled=controlled, players=teammates, label="own")
        self._debug_validate_nearest_order(controlled=controlled, players=opponents, label="opp")

        angle_rad = math.radians(controlled.angle)
        self_theta_cos = float(math.cos(angle_rad))
        self_theta_sin = float(math.sin(angle_rad))
        last_action = int(self._last_action_by_player_id.get(int(controlled.slot_index), self.ACTION_STAY))
        last_move_x, last_move_y = self.ACTION_TO_DIRECTION.get(last_action, (0.0, 0.0))
        last_changed = 1.0 if int(last_action) != self.ACTION_STAY else 0.0
        tgt_dx = float(clip_signed((self.ball_x - controlled.x) / width))
        tgt_dy = float(clip_signed((self.ball_y - controlled.y) / height))
        tgt_dist_norm = float(np.clip(math.hypot(tgt_dx, tgt_dy), 0.0, 1.0))

        def _relative_heading(dx: float, dy: float) -> tuple[float, float]:
            rel_norm = math.hypot(float(dx), float(dy))
            if rel_norm <= 1e-8:
                return 0.0, 1.0
            rel_norm_eps = rel_norm + 1e-8
            rel_cos = float(clip_signed((self_theta_cos * float(dx) + self_theta_sin * float(dy)) / rel_norm_eps))
            rel_sin = float(clip_signed((self_theta_cos * float(dy) - self_theta_sin * float(dx)) / rel_norm_eps))
            return rel_sin, rel_cos

        tgt_rel_ang_sin, tgt_rel_ang_cos = _relative_heading(tgt_dx, tgt_dy)
        opp_goal_y = float((self.right_goal_top + self.right_goal_bottom) * 0.5)
        own_goal_y = float((self.left_goal_top + self.left_goal_bottom) * 0.5)
        goal_dx = float(clip_signed((float(SCREEN_WIDTH) - controlled.x) / width))
        goal_dy = float(clip_signed((opp_goal_y - controlled.y) / height))
        own_goal_dx = float(clip_signed((0.0 - controlled.x) / width))
        own_goal_dy = float(clip_signed((own_goal_y - controlled.y) / height))
        gk_x, gk_y, gk_vy = self._goalkeeper_reference_for_player(controlled)
        own_gk_x, own_gk_y = self._own_goalkeeper_reference_for_player(controlled)
        shot_line_dy, shot_tti = self._own_goal_shot_features_for_player(controlled)
        shot_quality = float(self._shot_quality(controlled))

        left_attacking = self.effective_possession_team() == self.TEAM_LEFT
        anchor_x, anchor_y = self._player_anchor_position(
            controlled,
            team_attacking=left_attacking,
            ball_y=float(self.ball_y),
            use_smoothed=True,
        )
        map_anchor_dx = float(clip_signed((float(anchor_x) - controlled.x) / width))
        map_anchor_dy = float(clip_signed((float(anchor_y) - controlled.y) / height))

        feature_values: dict[str, float] = {
            "self_x_norm": float(clip_signed((2.0 * (float(controlled.x) / width)) - 1.0)),
            "self_y_norm": float(clip_signed((2.0 * ((float(controlled.y) - float(self.pitch_top)) / height)) - 1.0)),
            "self_vx": float(clip_signed(controlled.vx / player_vel_norm)),
            "self_vy": float(clip_signed(controlled.vy / player_vel_norm)),
            "self_theta_cos": self_theta_cos,
            "self_theta_sin": self_theta_sin,
            "self_has_ball": 1.0 if controlled.has_ball else 0.0,
            "self_stamina": float(controlled.stamina),
            "self_stamina_delta": float(clip_signed(controlled.stamina_delta)),
            "self_last_move_x": float(last_move_x),
            "self_last_move_y": float(last_move_y),
            "self_action_changed": float(last_changed),
            "tgt_dx": tgt_dx,
            "tgt_dy": tgt_dy,
            "tgt_dist_norm": tgt_dist_norm,
            "tgt_rel_ang_sin": tgt_rel_ang_sin,
            "tgt_rel_ang_cos": tgt_rel_ang_cos,
            "tgt_dvx": float(clip_signed((self.ball_vx - controlled.vx) / ball_vel_norm)),
            "land_opp_goal_dx": goal_dx,
            "land_opp_goal_dy": goal_dy,
            "land_own_goal_dx": own_goal_dx,
            "land_own_goal_dy": own_goal_dy,
            "land_own_gk_dx": float(clip_signed((float(own_gk_x) - controlled.x) / width)),
            "land_own_gk_dy": float(clip_signed((float(own_gk_y) - controlled.y) / height)),
            "land_shot_line_dy": float(shot_line_dy),
            "land_shot_tti": float(shot_tti),
            "land_gk_dx": float(clip_signed((float(gk_x) - controlled.x) / width)),
            "land_gk_dy": float(clip_signed((float(gk_y) - controlled.y) / height)),
            "land_gk_dvy": float(clip_signed((float(gk_vy) - controlled.vy) / player_vel_norm)),
            "map_anchor_dx": map_anchor_dx,
            "map_anchor_dy": map_anchor_dy,
            "flag_shot_quality": shot_quality,
        }
        feature_values.update(self._role_one_hot_feature_values(controlled.role))

        def _encode_neighbors(prefix: str, players: list[KickPlayer]) -> None:
            for idx in range(1, nearest_count + 1):
                if idx <= len(players):
                    player = players[idx - 1]
                    dx = float(clip_signed((player.x - controlled.x) / width))
                    dy = float(clip_signed((player.y - controlled.y) / height))
                    dvx = float(clip_signed((player.vx - controlled.vx) / player_vel_norm))
                    dvy = float(clip_signed((player.vy - controlled.vy) / player_vel_norm))
                else:
                    dx = 0.0
                    dy = 0.0
                    dvx = 0.0
                    dvy = 0.0
                feature_values[f"{prefix}{idx}_dx"] = dx
                feature_values[f"{prefix}{idx}_dy"] = dy
                feature_values[f"{prefix}{idx}_dvx"] = dvx
                feature_values[f"{prefix}{idx}_dvy"] = dvy

        _encode_neighbors("ally", teammates)
        _encode_neighbors("opp", opponents)

        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Kick observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def _obs(self) -> np.ndarray:
        if self.mode == "human":
            return self._player_obs(self._controlled_player())

        if not self.left_players:
            return np.zeros((0, self.OBS_DIM), dtype=np.float32)

        team_obs = [self._player_obs(player) for player in self.left_players]
        obs = np.asarray(team_obs, dtype=np.float32)
        if self.debug_sanity_checks:
            expected_shape = (len(self.left_players), int(self.OBS_DIM))
            if obs.shape != expected_shape:
                raise RuntimeError(f"Kick obs shape mismatch: expected {expected_shape}, got {tuple(obs.shape)}.")
        return obs

    def get_centralized_state(self, obs: object | None = None) -> np.ndarray:
        if obs is None:
            obs_array = np.asarray(self._obs(), dtype=np.float32)
        else:
            obs_array = np.asarray(obs, dtype=np.float32)

        if obs_array.ndim == 1:
            obs_batch = obs_array.reshape(1, -1)
        elif obs_array.ndim == 2:
            obs_batch = obs_array
        else:
            raise ValueError(f"Kick centralized state expected obs ndim 1 or 2, got {obs_array.ndim}.")

        if int(obs_batch.shape[1]) != int(self.OBS_DIM):
            raise ValueError(
                f"Kick centralized state expected obs dim {int(self.OBS_DIM)}, got {int(obs_batch.shape[1])}."
            )
        if self.debug_sanity_checks and int(obs_batch.shape[0]) > int(self.max_left_players):
            raise RuntimeError(
                f"Kick centralized state got {int(obs_batch.shape[0])} LEFT players, max is {int(self.max_left_players)}."
            )

        padded_obs = np.zeros((int(self.max_left_players), int(self.OBS_DIM)), dtype=np.float32)
        central_mask = np.zeros((int(self.central_obs_mask_dim),), dtype=np.float32)
        present_count = min(int(obs_batch.shape[0]), int(self.max_left_players))
        if present_count > 0:
            padded_obs[:present_count, :] = obs_batch[:present_count, :]
            central_mask[:present_count] = 1.0
        obs_values = padded_obs.reshape(-1)

        width = max(1.0, float(SCREEN_WIDTH))
        height = max(1.0, float(self.pitch_height))
        ball_vel_norm = max(1.0, float(self.ball_max_speed))
        physical_team = self.physical_owner_team()
        effective_team = self.effective_possession_team()
        physical_owner_scalar = 0.0 if physical_team is None else (1.0 if physical_team == self.TEAM_LEFT else -1.0)
        effective_owner_scalar = 0.0 if effective_team is None else (1.0 if effective_team == self.TEAM_LEFT else -1.0)
        ball_features = np.asarray(
            [
                float(clip_signed((2.0 * (float(self.ball_x) / width)) - 1.0)),
                float(clip_signed((2.0 * ((float(self.ball_y) - float(self.pitch_top)) / height)) - 1.0)),
                float(clip_signed(float(self.ball_vx) / ball_vel_norm)),
                float(clip_signed(float(self.ball_vy) / ball_vel_norm)),
                float(physical_owner_scalar),
                float(effective_owner_scalar),
            ],
            dtype=np.float32,
        )

        state = np.concatenate((obs_values, central_mask, ball_features), axis=0).astype(np.float32, copy=False)
        if self.debug_sanity_checks and state.shape != (int(self.central_obs_dim),):
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
        self.controlled_index = int(self._default_controlled_index())
        self._prev_shoot_down = False
        self._prev_pass_down = False
        self.last_action_index = self.ACTION_STAY
        self._last_action_by_player_id = {}
        self._restart_kickoff(self.TEAM_LEFT)
        self._seed_easy_level_start_possession()
        self._initialize_anchor_state()
        self._reset_controlled_progress_state()
        obs = self._obs()
        if self.debug_sanity_checks and self.mode != "human":
            expected_shape = (len(self.left_players), int(self.OBS_DIM))
            if obs.shape != expected_shape:
                raise RuntimeError(f"Kick reset obs shape mismatch: expected {expected_shape}, got {tuple(obs.shape)}.")
        return obs

    def _draw_pitch(self) -> None:
        pitch_h = self.pitch_height
        pitch_bottom = self.window_controller.top_left_to_bottom(self.pitch_top, pitch_h)
        line_width = float(PITCH_LINE_WIDTH)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_DARK_NEUTRAL)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_DARK_NEUTRAL + (24,))

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

    def _draw_shot_quality_bar(
        self,
        player: KickPlayer,
        *,
        fill_color: tuple[int, int, int] | tuple[int, int, int, int],
    ) -> None:
        if (not player.has_ball) or self.ball_owner is not player:
            return

        quality = float(np.clip(self._shot_quality(player), 0.0, 1.0))
        bar_width = float(self.player_size)
        bar_height = max(2.0, float(self.player_size) * 0.12)
        bar_gap = max(1.0, float(self.player_size) * 0.10)
        left = float(player.x) - float(self.player_half)
        top = float(player.y) - float(self.player_half) - bar_gap - bar_height
        bottom = self.window_controller.top_left_to_bottom(top, bar_height)
        base_color = tuple(int(channel) for channel in fill_color[:3])

        arcade.draw_lbwh_rectangle_filled(left, bottom, bar_width, bar_height, base_color + (48,))
        fill_width = bar_width * quality
        if fill_width > 0.5:
            arcade.draw_lbwh_rectangle_filled(left, bottom, fill_width, bar_height, base_color + (128,))

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

        self._draw_shot_quality_bar(player, fill_color=inner)

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
        inset = status_icon_inset(float(CELL_INSET))
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outline_color,
            inner_color=fill_color,
            inset=float(inset),
        )

    def _remaining_time_ratio(self) -> float:
        return float(self.match_tracker.remaining_time_ratio(int(self.steps)))

    def _score_icon_items(self) -> list[str]:
        return (
            [self.TEAM_LEFT] * max(0, int(self.left_score))
            + [self.TEAM_RIGHT] * max(0, int(self.right_score))
        )

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
        self._draw_zone_target_clones()

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

        bar_layout = draw_status_bar(
            width=float(SCREEN_WIDTH),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=float(CELL_INSET),
            include_clock=True,
        )
        draw_status_clock(
            layout=bar_layout,
            remaining_ratio=float(self._remaining_time_ratio()),
        )
        self._draw_score_icons(
            float(bar_layout.score_left),
            float(bar_layout.score_right),
            float(bar_layout.center_y),
        )
        self.window_controller.flip()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        controlled_role = self._controlled_player().role if self.mode == "human" else "TEAM"
        if self.done:
            done_reward_vec = np.zeros((len(self.left_players),), dtype=np.float32)
            return self._obs(), 0.0, True, {
                "win": bool(self.left_score > self.right_score),
                "success": int(self._last_episode_success),
                "score_left": int(self.left_score),
                "score_right": int(self.right_score),
                "time_left_ratio": float(self._remaining_time_ratio()),
                "controlled_role": controlled_role,
                "level": int(self._last_episode_level),
                "reward_vec": done_reward_vec,
                "reward_breakdown": {},
            }

        self.window_controller.poll_events_or_raise()
        self._update_visual_overlay_toggle()

        episode_level = int(self._current_level)
        parsed_action = self._decode_team_actions(action) if self.mode != "human" else self.ACTION_STAY
        compute_scored_breakdown = bool(self.mode != "human")
        reward_vec = np.zeros((len(self.left_players),), dtype=np.float32)
        reward_breakdown: dict[str, float] = {}
        repeat_frames = 1 if self.mode == "human" else int(self.rl_action_repeat_frames)

        for _ in range(max(1, int(repeat_frames))):
            prev_left_positions = (
                {
                    int(player.slot_index): (float(player.x), float(player.y))
                    for player in self.left_players
                }
                if compute_scored_breakdown
                else {}
            )
            prev_ball_position = (
                (float(self.ball_x), float(self.ball_y))
                if compute_scored_breakdown
                else (0.0, 0.0)
            )
            self._tick(parsed_action)
            self._update_anchor_state()
            if self.steps >= self.max_steps:
                self.done = True

            frame_rewards = np.zeros((len(self.left_players),), dtype=np.float32)
            frame_breakdown: dict[str, float] = {}
            if compute_scored_breakdown:
                frame_rewards, frame_breakdown = self._score_reward(
                    prev_left_positions=prev_left_positions,
                    prev_ball_position=prev_ball_position,
                )
                reward_vec = reward_vec + np.asarray(frame_rewards, dtype=np.float32)
                for key, value in frame_breakdown.items():
                    reward_breakdown[key] = float(reward_breakdown.get(key, 0.0) + float(value))

            self.render()
            self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)
            if self.done:
                break

        if self.mode != "human":
            reward = float(reward_vec.sum())
        else:
            reward = 0.0
            reward_breakdown = {}

        if self.debug_sanity_checks:
            expected_shape = (len(self.left_players),)
            if reward_vec.shape != expected_shape:
                raise RuntimeError(
                    f"Kick reward_vec shape mismatch: expected {expected_shape}, got {tuple(reward_vec.shape)}."
                )
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
            "controlled_role": controlled_role,
            "level": int(episode_level),
            "level_changed": False,
            "reward_vec": reward_vec,
            "reward_breakdown": reward_breakdown,
        }
        if done:
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
        self.window_controller.close()
