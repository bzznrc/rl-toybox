"""Football environment with human and RL control modes."""

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
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_signed, ordered_feature_vector
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_control_marker,
    draw_facing_indicator,
    draw_status_square_icon,
    draw_time_pie_indicator,
    draw_two_tone_tile,
    resolve_circle_collisions,
    status_icon_size,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level
from games.kick.config import (
    ACTION_NAMES as KICK_ACTION_NAMES,
    ACT_DIM as KICK_ACT_DIM,
    BALL_RADIUS_SCALE,
    BB_HEIGHT,
    CELL_INSET,
    CURRICULUM_PROMOTION,
    ENTROPY_COEF,
    FPS,
    GAME_SPEED_SCALE,
    GK_HIGH_BYPASS_PROB_DEFAULT,
    INPUT_FEATURE_NAMES as KICK_INPUT_FEATURE_NAMES,
    LEVEL_SETTINGS,
    MAX_LEVEL,
    MIN_LEVEL,
    OBS_DIM as KICK_OBS_DIM,
    PHYSICS_DT,
    PENALTY_AREA_DEPTH_RATIO,
    PENALTY_AREA_WIDTH_RATIO,
    PENALTY_CONCEDE,
    PENALTY_TURNOVER,
    PENALTY_ZONE,
    PLAYER_A_MAX_PX_PER_SEC2,
    PLAYER_V_MAX_PX_PER_SEC,
    PITCH_LINE_WIDTH,
    REWARD_PROGRESS,
    REWARD_SCORE,
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


validate_curriculum_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
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
    """Top-down football environment with curriculum-sized teams.

    Human controls:
    - Move: WASD
    - Shoot: Space hold/release
    - Controlled player: auto-switch (left-ball-owner, else closest-to-ball)

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
    KICKOFF_GK = "GK"
    KICKOFF_CC = "CC"
    KICKOFF_MODES = {
        KICKOFF_GK,
        KICKOFF_CC,
    }
    ROLE_ORDER = ("GK", "LB", "LCB", "RCB", "RB", "LM", "LCM", "RCM", "RM", "ST1", "ST2")
    ROLE_SCALAR_BY_GROUP = {
        "GK": -1.0,
        "DEF": 0.0,
        "MID": 0.5,
        "ATK": 1.0,
    }
    ROLE_GROUP_BY_ROLE = {
        "GK": "GK",
        "LB": "DEF",
        "LCB": "DEF",
        "RCB": "DEF",
        "RB": "DEF",
        "LM": "MID",
        "LCM": "MID",
        "RCM": "MID",
        "RM": "MID",
        "ST1": "ATK",
        "ST2": "ATK",
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
    KICK_ACTION_TO_KIND = {
        ACTION_KICK_LOW: 1,
        ACTION_KICK_MID: 2,
        ACTION_KICK_HIGH: 3,
    }
    OBS_NEAREST_PLAYERS = 2
    MATCH_DURATION_SECONDS = 60.0
    REWARD_COMPONENT_ORDER = ("G", "C", "T", "P", "Z")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_score": "G",
        "outcome.penalty_concede": "C",
        "event.penalty_turnover": "T",
        "progress.reward_progress": "P",
        "event.penalty_zone": "Z",
    }

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
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
        self._left_roles = list(self.ROLE_ORDER)
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
        self.player_vmax_base = float(PLAYER_V_MAX_PX_PER_SEC)
        self.player_amax_base = float(PLAYER_A_MAX_PX_PER_SEC2)
        self.ball_max_speed = max(1.0, 14.5 * self.speed_scale)
        self.ball_friction = 0.985
        self.pickup_range = TILE_SIZE * 0.7 * max(0.75, self.speed_scale)
        self.contest_range = TILE_SIZE * 1.45
        self.contest_cooldown_frames = max(1, int(FPS))
        self.freeze_after_restart = 20
        self.max_steps = int(FPS * float(self.MATCH_DURATION_SECONDS))
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
        self.enemy_shot_error_choices: tuple[float, ...] = (0.0,)
        self.enemy_recovery_wait_ratio = 0.25
        self.enemy_recovery_resume_ratio = 0.70
        self.enemy_ball_turn_rate_deg = 7.5
        self.gk_high_bypass_prob = float(GK_HIGH_BYPASS_PROB_DEFAULT)
        self.kickoff_mode = self.KICKOFF_GK
        self._level_entropy_coef = float(ENTROPY_COEF)

        self.left_players: list[KickPlayer] = []
        self.right_players: list[KickPlayer] = []
        self.all_players: list[KickPlayer] = []
        self.left_goalkeeper: KickPlayer | None = None
        self.right_goalkeeper: KickPlayer | None = None
        self._enemy_recovery_mode: dict[int, bool] = {}
        self.controlled_index = 0

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_owner: KickPlayer | None = None
        self.ball_last_kick_type = 0

        self.last_touch_team = self.TEAM_LEFT
        self.left_score = int(self.match_tracker.score(self.TEAM_LEFT))
        self.right_score = int(self.match_tracker.score(self.TEAM_RIGHT))
        self.steps = 0
        self.done = False
        self.freeze_frames = 0
        self.last_action_index = self.ACTION_STAY

        self._goal_scored_team: str | None = None
        self._our_poss_best_goal_dist = 0.0
        self._prev_our_possession = False

        self._prev_space_down = False
        self._human_shot_hold_start: float | None = None
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        self._apply_level_change(int(self._current_level))
        self.reset()

    def _default_controlled_index(self) -> int:
        if not self.left_players:
            return 0
        for preferred_role in ("ST1", "ST2", "LCM", "RCM", "GK"):
            for idx, player in enumerate(self.left_players):
                if str(player.role) == preferred_role:
                    return int(idx)
        return max(0, len(self.left_players) - 1)

    @classmethod
    def _normalize_kickoff_mode(cls, value: object) -> str:
        normalized = str(value).strip().upper()
        if normalized not in cls.KICKOFF_MODES:
            valid = ", ".join(sorted(cls.KICKOFF_MODES))
            raise ValueError(f"Unsupported kickoff mode '{value}'. Valid modes: {valid}")
        return normalized

    def get_entropy_coef_for_level(self, level: int | None = None) -> float | None:
        if level is None or int(level) == int(self._current_level):
            return float(self._level_entropy_coef)

        settings = LEVEL_SETTINGS.get(int(level), LEVEL_SETTINGS[int(MIN_LEVEL)])
        try:
            return float(settings.get("entropy_coef", ENTROPY_COEF))
        except (TypeError, ValueError):
            return float(ENTROPY_COEF)

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level), LEVEL_SETTINGS[int(MIN_LEVEL)])
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
        enemy_shot_error_choices = settings.get("enemy_shot_error_choices", [0.0])
        if "kickoff" not in settings:
            raise ValueError("Kick LEVEL_SETTINGS entries must define 'kickoff' (GK or CC).")
        kickoff_mode = self._normalize_kickoff_mode(settings["kickoff"])
        raw_entropy_coef = settings.get("entropy_coef", ENTROPY_COEF)
        if goals_size_scale is None:
            # Backward compatibility with prior two-key format.
            own_goal_size_scale = settings.get("own_goal_size_scale", 1.0)
            opp_goal_size_scale = settings.get("opp_goal_size_scale", 1.0)
            try:
                own_goal_scale = max(0.1, float(own_goal_size_scale))
            except (TypeError, ValueError):
                own_goal_scale = 1.0
            try:
                opp_goal_scale = max(0.1, float(opp_goal_size_scale))
            except (TypeError, ValueError):
                opp_goal_scale = 1.0
        else:
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
        parsed_error_choices: list[float] = []
        if isinstance(enemy_shot_error_choices, (list, tuple)):
            for raw in enemy_shot_error_choices:
                try:
                    parsed_error_choices.append(float(raw))
                except (TypeError, ValueError):
                    continue
        if not parsed_error_choices:
            parsed_error_choices = [0.0]
        self.enemy_shot_error_choices = tuple(parsed_error_choices)
        self.gk_high_bypass_prob = self._clamp(float(GK_HIGH_BYPASS_PROB_DEFAULT), 0.0, 1.0)
        try:
            parsed_entropy_coef = float(raw_entropy_coef)
        except (TypeError, ValueError):
            parsed_entropy_coef = float(ENTROPY_COEF)
        self._level_entropy_coef = float(parsed_entropy_coef)
        self.kickoff_mode = str(kickoff_mode)
        self._set_goal_sizes(own_goal_scale=own_goal_scale, opp_goal_scale=opp_goal_scale)
        self._current_level = int(level)
        self._left_roles = list(self.ROLE_ORDER)
        self._players_opponent = max(1, min(players_opponent, len(self.ROLE_ORDER)))
        self._opponent_roles = list(deduped_roles[: self._players_opponent])

    def _apply_level_change(self, level: int) -> None:
        self._apply_level_settings(int(level))
        self._build_teams()

    def _build_teams(self) -> None:
        left_roles = list(self._left_roles[: len(self.ROLE_ORDER)])
        right_roles = list(self._opponent_roles[: max(1, int(self._players_opponent))])
        self.left_players = self._team_for_roles(self.TEAM_LEFT, left_roles)
        self.right_players = self._team_for_roles(self.TEAM_RIGHT, right_roles)
        self.all_players = [*self.left_players, *self.right_players]
        self._enemy_recovery_mode = {id(player): False for player in self.right_players}
        self.left_goalkeeper = next((player for player in self.left_players if player.role == "GK"), None)
        self.right_goalkeeper = next((player for player in self.right_players if player.role == "GK"), None)
        self.controlled_index = int(self._default_controlled_index())

    def _team_for_roles(self, team: str, roles: list[str]) -> list[KickPlayer]:
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
        for role in roles:
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
        player_key = int(id(player))
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
        return self.ball_owner is not None and self.ball_owner.team == self.TEAM_LEFT

    def _set_ball_owner(self, owner: KickPlayer | None) -> None:
        for player in self.all_players:
            player.has_ball = False
        self.ball_owner = owner
        if owner is None:
            return
        self.ball_last_kick_type = 0
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
        if self.mode == "human" and owner is not self._controlled_player():
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
    ) -> list[KickPlayer]:
        pool = self.left_players if team == self.TEAM_LEFT else self.right_players
        candidates = [player for player in pool if player is not exclude]
        candidates.sort(key=lambda player: self._distance(player.x, player.y, x, y))
        return candidates[: max(0, int(k))]

    def _role_scalar(self, role: str) -> float:
        role_key = str(role).upper()
        group = self.ROLE_GROUP_BY_ROLE.get(role_key)
        if group is None:
            group = "DEF"
        return float(self.ROLE_SCALAR_BY_GROUP[group])

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
            self._kick_ball(player, speed=speed, kick_type=kind)
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

    def _kick_ball(
        self,
        player: KickPlayer,
        speed: float,
        angle_degrees: float | None = None,
        kick_type: int = 0,
    ) -> None:
        if not player.has_ball or self.ball_owner is not player:
            return
        angle = float(player.angle if angle_degrees is None else angle_degrees) % 360.0
        radians = math.radians(angle)
        self._set_ball_owner(None)
        self.ball_x = player.x + math.cos(radians) * self.ball_drag_offset
        self.ball_y = player.y + math.sin(radians) * self.ball_drag_offset
        self.ball_vx = math.cos(radians) * float(speed)
        self.ball_vy = math.sin(radians) * float(speed)
        self.last_touch_team = player.team
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
        if move_x != 0.0 or move_y != 0.0:
            controlled.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)

        space_down = self.window_controller.is_key_down(arcade.key.SPACE)
        if space_down and not self._prev_space_down:
            if controlled.has_ball:
                self._human_shot_hold_start = time.perf_counter()

        if (not space_down) and self._prev_space_down:
            if controlled.has_ball and self._human_shot_hold_start is not None:
                hold = max(0.0, time.perf_counter() - self._human_shot_hold_start)
                kick_type = self._kick_type_from_hold_seconds(hold)
                self._apply_kick_or_contest(controlled, kick_type=kick_type)
                self.last_action_index = self._kick_action_from_kind(kick_type)
            self._human_shot_hold_start = None

        self._prev_space_down = space_down

    def _apply_rl_action_to_player(self, player: KickPlayer, action_idx: int) -> None:
        action_idx = int(np.clip(int(action_idx), 0, self.NUM_ACTIONS - 1))
        self.last_action_index = int(action_idx)

        if action_idx <= self.ACTION_MOVE_NW:
            move_x, move_y = self.ACTION_TO_DIRECTION.get(action_idx, (0.0, 0.0))
            self._move_player(player, move_x, move_y)
            if move_x != 0.0 or move_y != 0.0:
                player.angle = self._angle_degrees(0.0, 0.0, move_x, move_y)
            return

        # Kicks imply no movement; without possession they are ignored as STAY.
        self._move_player(player, 0.0, 0.0)
        if not player.has_ball or self.ball_owner is not player:
            return

        kick_type = self.KICK_ACTION_TO_KIND.get(action_idx, 0)
        if kick_type <= 0:
            return
        self._apply_kick_or_contest(player, kick_type=kick_type)

    def _rl_team_step(self, actions) -> np.ndarray:
        action_indices = self._decode_team_actions(actions)
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

    def _penalty_area_bounds_for_defending_team(self, team: str) -> tuple[float, float, float, float]:
        penalty_depth = SCREEN_WIDTH * float(PENALTY_AREA_DEPTH_RATIO)
        penalty_height = self.pitch_height * float(PENALTY_AREA_WIDTH_RATIO)
        penalty_top = self.pitch_center_y - penalty_height * 0.5
        penalty_bottom = self.pitch_center_y + penalty_height * 0.5
        if str(team) == self.TEAM_LEFT:
            return 0.0, penalty_depth, penalty_top, penalty_bottom
        return SCREEN_WIDTH - penalty_depth, float(SCREEN_WIDTH), penalty_top, penalty_bottom

    def _is_in_attacking_penalty_area(self, player: KickPlayer) -> bool:
        defending_team = self.TEAM_RIGHT if player.team == self.TEAM_LEFT else self.TEAM_LEFT
        left_x, right_x, top_y, bottom_y = self._penalty_area_bounds_for_defending_team(defending_team)
        return left_x <= float(player.x) <= right_x and top_y <= float(player.y) <= bottom_y

    def _attempt_penalty_area_shot(self, player: KickPlayer) -> bool:
        if (not player.has_ball) or self.ball_owner is not player:
            return False
        if not self._is_in_attacking_penalty_area(player):
            return False

        defending_team = self.TEAM_RIGHT if player.team == self.TEAM_LEFT else self.TEAM_LEFT
        goal_x = SCREEN_WIDTH - self.ball_radius if player.team == self.TEAM_LEFT else self.ball_radius
        goal_top, goal_bottom, goal_half = self._goal_bounds_for_defending_team(defending_team)
        corner_inset = max(self.ball_radius * 0.75, goal_half * 0.15)
        top_corner_y = goal_top + corner_inset
        center_y = self.pitch_center_y
        bottom_corner_y = goal_bottom - corner_inset

        roll = random.random()
        if float(player.y) <= self.pitch_center_y:
            if roll < 0.50:
                target_y = top_corner_y
            elif roll < 0.75:
                target_y = center_y
            else:
                target_y = bottom_corner_y
        else:
            if roll < 0.50:
                target_y = bottom_corner_y
            elif roll < 0.75:
                target_y = center_y
            else:
                target_y = top_corner_y

        shot_angle = self._angle_degrees(player.x, player.y, goal_x, target_y)
        if player.team == self.TEAM_RIGHT:
            shot_error = random.choice(self.enemy_shot_error_choices)
            shot_angle = float((float(shot_angle) + float(shot_error)) % 360.0)
        player.angle = float(shot_angle)
        self._kick_ball(player, speed=11.0 * self.speed_scale, angle_degrees=player.angle)
        return True

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

        target_x = player.home_x
        target_y = player.home_y

        if player.has_ball:
            goal_x = SCREEN_WIDTH - self.ball_radius if player.team == self.TEAM_LEFT else self.ball_radius
            if self._attempt_penalty_area_shot(player):
                return
            defenders = [candidate for candidate in self.all_players if candidate.team != player.team]
            blocker_in_front = self._has_defender_in_front(player, defenders)
            pass_target = self._select_progressive_pass_target(player) if blocker_in_front else None

            if pass_target is not None:
                pass_angle = self._angle_degrees(player.x, player.y, pass_target.x, pass_target.y)
                player.angle = pass_angle
                self._kick_ball(player, speed=self._shoot_speed(0.35), angle_degrees=pass_angle)
                return

            if blocker_in_front:
                lane_sign = -1.0 if player.home_y > self.pitch_center_y else 1.0
                target_x = player.x + (1.0 if player.team == self.TEAM_LEFT else -1.0) * TILE_SIZE * 4.0
                target_y = self._clamp(player.y + lane_sign * TILE_SIZE * 3.2, TILE_SIZE, self.pitch_bottom - TILE_SIZE)
            else:
                target_x = goal_x
                home_bias = (player.home_y - self.pitch_center_y) * 0.25
                target_y = self._clamp(self.pitch_center_y + home_bias, TILE_SIZE, self.pitch_bottom - TILE_SIZE)

            desired_angle = self._angle_degrees(player.x, player.y, target_x, target_y)
            if player.team == self.TEAM_RIGHT:
                player.angle = self._turn_towards_angle(player.angle, desired_angle, self.enemy_ball_turn_rate_deg)
            else:
                player.angle = desired_angle
            self._move_player(player, target_x - player.x, target_y - player.y)

            shoot_lane_blocked = self._is_lane_blocked(player.x, player.y, goal_x, self.pitch_center_y, defenders, margin=1.0)
            shoot_threshold = SCREEN_WIDTH * 0.27
            distance_to_goal = abs(goal_x - player.x)
            if distance_to_goal < shoot_threshold and (not shoot_lane_blocked) and random.random() < 0.12:
                if player.team == self.TEAM_RIGHT:
                    base_angle = self._angle_degrees(player.x, player.y, goal_x, self.pitch_center_y)
                    shot_error = random.choice(self.enemy_shot_error_choices)
                    player.angle = float((base_angle + float(shot_error)) % 360.0)
                else:
                    _, _, target_goal_half = self._goal_bounds_for_defending_team(self.TEAM_RIGHT)
                    variance = random.uniform(-target_goal_half * 0.7, target_goal_half * 0.7)
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
        goal_top, goal_bottom, _ = self._goal_bounds_for_defending_team(keeper.team)
        target_y = self._clamp(self.ball_y, goal_top + self.player_half, goal_bottom - self.player_half)
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
            goal_top, goal_bottom, goal_half_height = self._goal_bounds_for_defending_team(keeper.team)
            if self.ball_y < goal_top - TILE_SIZE or self.ball_y > goal_bottom + TILE_SIZE:
                continue

            if self._distance(self.ball_x, self.ball_y, keeper.x, keeper.y) > TILE_SIZE * 1.8:
                continue

            offset_from_keeper = abs(self.ball_y - keeper.y)
            catch_prob = self._clamp(1.0 - offset_from_keeper / max(1.0, goal_half_height), 0.0, 1.0)
            would_catch = random.random() < catch_prob
            if not would_catch:
                continue

            if int(self.ball_last_kick_type) == 3 and random.random() < float(self.gk_high_bypass_prob):
                continue

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
            player.stamina = self._stamina_cap_for(player)
            player.stamina_delta = 0.0
            player.in_contact = False

        self.ball_x = SCREEN_WIDTH * 0.5
        self.ball_y = self.pitch_center_y
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_last_kick_type = 0

        if kickoff_team == self.TEAM_RIGHT:
            self._set_ball_owner(None)
        else:
            kickoff_mode = self._normalize_kickoff_mode(self.kickoff_mode)
            if kickoff_mode == self.KICKOFF_CC:
                self._set_ball_owner(None)
            else:
                starter = self.left_goalkeeper
                if starter is None:
                    raise RuntimeError("Kickoff GK mode requires a left-team goalkeeper.")
                starter.angle = 0.0
                self._set_ball_owner(starter)

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
            self._step_ball()
            self._try_pickup_free_ball()
            self._try_goalkeeper_catch()
            self._handle_ball_boundaries()

        self.steps += 1
        return applied_action

    def _progress_reward(self) -> float:
        our_possession_now = bool(self._left_has_possession())
        goal_x, goal_y = self._opponent_goal_center()
        goal_dist = float(self._distance(self.ball_x, self.ball_y, goal_x, goal_y))

        if our_possession_now and (not self._prev_our_possession):
            self._our_poss_best_goal_dist = float(goal_dist)

        if our_possession_now:
            new_best = min(float(self._our_poss_best_goal_dist), float(goal_dist))
            improvement = max(0.0, float(self._our_poss_best_goal_dist) - float(new_best))
            self._our_poss_best_goal_dist = float(new_best)
            progress_norm = max(1.0, math.hypot(float(SCREEN_WIDTH), float(self.pitch_height)))
            progress_reward = float(REWARD_PROGRESS) * (float(improvement) / float(progress_norm))
        else:
            progress_reward = 0.0

        self._prev_our_possession = bool(our_possession_now)
        return float(progress_reward)

    def _opponent_goal_center(self) -> tuple[float, float]:
        goal_top, goal_bottom, _ = self._goal_bounds_for_defending_team(self.TEAM_RIGHT)
        return float(SCREEN_WIDTH), float((goal_top + goal_bottom) * 0.5)

    @staticmethod
    def _goal_axis_y_from_x(x_pos: float) -> float:
        width = max(1.0, float(SCREEN_WIDTH))
        return float(np.clip((float(x_pos) / width) * 100.0, 0.0, 100.0))

    def _zone_anchor_y(self, player: KickPlayer, *, has_possession: bool, ball_y: float) -> float:
        role_group = self.ROLE_GROUP_BY_ROLE.get(str(player.role).upper(), "DEF")
        if role_group == "GK":
            return 5.0

        base_anchor_by_group = {
            "DEF": 25.0,
            "MID": 45.0,
            "ATK": 65.0,
        }
        base_anchor = float(base_anchor_by_group.get(role_group, 45.0))
        phase_shift = 10.0 if bool(has_possession) else -10.0
        ball_shift = float(np.clip((float(ball_y) - 50.0) * 0.1, -5.0, 5.0))
        return float(self._clamp(base_anchor + phase_shift + ball_shift, 5.0, 85.0))

    def _zone_norm(self) -> float:
        if not self.left_players:
            return 0.0

        left_has_possession = self._left_has_possession()
        ball_y = self._goal_axis_y_from_x(self.ball_x)

        dy_norm_total = 0.0
        for player in self.left_players:
            player_y = self._goal_axis_y_from_x(player.x)
            anchor_y = self._zone_anchor_y(
                player,
                has_possession=left_has_possession,
                ball_y=ball_y,
            )
            dy_norm_total += abs(float(player_y) - float(anchor_y)) / 100.0

        return float(dy_norm_total / max(1, len(self.left_players)))

    def _score_reward(self, *, prev_owner_team: str | None, curr_owner_team: str | None) -> tuple[float, dict[str, float]]:
        reward = 0.0
        reward_breakdown = {
            "event.penalty_turnover": 0.0,
            "progress.reward_progress": 0.0,
            "event.penalty_zone": 0.0,
            "outcome.reward_score": 0.0,
            "outcome.penalty_concede": 0.0,
        }

        progress_reward = float(self._progress_reward())
        reward += progress_reward
        reward_breakdown["progress.reward_progress"] = progress_reward

        direct_turnover = (
            prev_owner_team == self.TEAM_LEFT
            and curr_owner_team == self.TEAM_RIGHT
            and self._goal_scored_team is None
        )
        if direct_turnover:
            reward += float(PENALTY_TURNOVER)
            reward_breakdown["event.penalty_turnover"] = float(PENALTY_TURNOVER)

        zone_norm = float(self._zone_norm())
        zone_penalty = float(PENALTY_ZONE) * zone_norm
        reward += zone_penalty
        reward_breakdown["event.penalty_zone"] = zone_penalty

        if self._goal_scored_team == self.TEAM_LEFT:
            reward += float(REWARD_SCORE)
            reward_breakdown["outcome.reward_score"] = float(REWARD_SCORE)
        elif self._goal_scored_team == self.TEAM_RIGHT:
            reward += float(PENALTY_CONCEDE)
            reward_breakdown["outcome.penalty_concede"] = float(PENALTY_CONCEDE)

        return reward, reward_breakdown

    def _player_obs(self, controlled: KickPlayer) -> np.ndarray:
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
        self_theta_cos = float(math.cos(angle_rad))
        self_theta_sin = float(math.sin(angle_rad))
        tgt_dx = float(clip_signed((self.ball_x - controlled.x) / width))
        tgt_dy = float(clip_signed((self.ball_y - controlled.y) / height))
        ball_rel_norm = math.hypot(tgt_dx, tgt_dy)
        if ball_rel_norm <= 1e-8:
            tgt_rel_angle_sin = 0.0
            tgt_rel_angle_cos = 1.0
        else:
            ball_rel_norm_eps = ball_rel_norm + 1e-8
            tgt_rel_angle_cos = float(
                clip_signed((self_theta_cos * tgt_dx + self_theta_sin * tgt_dy) / ball_rel_norm_eps)
            )
            tgt_rel_angle_sin = float(
                clip_signed((self_theta_cos * tgt_dy - self_theta_sin * tgt_dx) / ball_rel_norm_eps)
            )
        ball_is_free = 1.0 if self.ball_owner is None else 0.0
        if self.ball_owner is None:
            ball_owner_team = 0.0
        elif self.ball_owner.team == self.TEAM_LEFT:
            ball_owner_team = 1.0
        else:
            ball_owner_team = -1.0
        goal_dx = float(clip_signed((float(SCREEN_WIDTH) - controlled.x) / width))
        goal_dy = float(clip_signed((self.pitch_center_y - controlled.y) / height))
        goal_rel_norm = math.hypot(goal_dx, goal_dy)
        if goal_rel_norm <= 1e-8:
            goal_rel_angle_sin = 0.0
            goal_rel_angle_cos = 1.0
        else:
            goal_rel_norm_eps = goal_rel_norm + 1e-8
            goal_rel_angle_cos = float(
                clip_signed((self_theta_cos * goal_dx + self_theta_sin * goal_dy) / goal_rel_norm_eps)
            )
            goal_rel_angle_sin = float(
                clip_signed((self_theta_cos * goal_dy - self_theta_sin * goal_dx) / goal_rel_norm_eps)
            )

        feature_values: dict[str, float] = {
            "self_vx": float(clip_signed(controlled.vx / player_vel_norm)),
            "self_vy": float(clip_signed(controlled.vy / player_vel_norm)),
            "self_theta_cos": self_theta_cos,
            "self_theta_sin": self_theta_sin,
            "self_has_ball": 1.0 if controlled.has_ball else 0.0,
            "self_role": float(self._role_scalar(controlled.role)),
            "self_stamina": float(controlled.stamina),
            "self_stamina_delta": float(clip_signed(controlled.stamina_delta)),
            "tgt_dx": tgt_dx,
            "tgt_dy": tgt_dy,
            "tgt_rel_angle_sin": tgt_rel_angle_sin,
            "tgt_rel_angle_cos": tgt_rel_angle_cos,
            "tgt_dvx": float(clip_signed((self.ball_vx - controlled.vx) / ball_vel_norm)),
            "tgt_dvy": float(clip_signed((self.ball_vy - controlled.vy) / ball_vel_norm)),
            "tgt_is_free": float(ball_is_free),
            "tgt_owner_team": float(ball_owner_team),
            "goal_dx": goal_dx,
            "goal_dy": goal_dy,
            "goal_rel_angle_sin": goal_rel_angle_sin,
            "goal_rel_angle_cos": goal_rel_angle_cos,
        }

        while len(teammates) < nearest_count:
            teammates.append(controlled)
        while len(opponents) < nearest_count:
            opponents.append(self.right_players[0] if self.right_players else controlled)

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

    def _obs(self) -> np.ndarray:
        if self.mode == "human":
            return self._player_obs(self._controlled_player())

        if not self.left_players:
            return np.zeros((0, self.OBS_DIM), dtype=np.float32)

        team_obs = [self._player_obs(player) for player in self.left_players]
        return np.asarray(team_obs, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self._apply_level_change(int(self._current_level))
        self._reset_team_scores()
        self.steps = 0
        self.done = False
        self._episode_reward_components.reset()
        self.freeze_frames = 0
        self.controlled_index = int(self._default_controlled_index())
        self._prev_space_down = False
        self._human_shot_hold_start = None
        self.last_action_index = self.ACTION_STAY
        self._restart_kickoff(self.TEAM_LEFT)
        self._prev_our_possession = bool(self._left_has_possession())
        goal_x, goal_y = self._opponent_goal_center()
        self._our_poss_best_goal_dist = float(self._distance(self.ball_x, self.ball_y, goal_x, goal_y))
        return self._obs()

    def _draw_pitch(self) -> None:
        pitch_h = self.pitch_height
        pitch_bottom = self.window_controller.top_left_to_bottom(self.pitch_top, pitch_h)
        line_width = float(PITCH_LINE_WIDTH)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_CHARCOAL)
        arcade.draw_lbwh_rectangle_filled(0, pitch_bottom, SCREEN_WIDTH, pitch_h, COLOR_NEAR_BLACK + (24,))

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
        arcade.draw_lbwh_rectangle_outline(0, left_goal_bottom, TILE_SIZE, left_goal_h, COLOR_SOFT_WHITE, line_width)
        right_goal_h = self.right_goal_half_height * 2.0
        right_goal_bottom = self.window_controller.top_left_to_bottom(self.right_goal_top, right_goal_h)
        arcade.draw_lbwh_rectangle_outline(
            SCREEN_WIDTH - TILE_SIZE,
            right_goal_bottom,
            TILE_SIZE,
            right_goal_h,
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
        return float(self.match_tracker.remaining_time_ratio(int(self.steps)))

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

        controlled = self._controlled_player() if self.mode == "human" else None
        for player in self.all_players:
            self._draw_player(player, controlled_marker=(controlled is not None and player is controlled))

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
        controlled_role = self._controlled_player().role if self.mode == "human" else "TEAM"
        if self.done:
            return self._obs(), 0.0, True, {
                "win": bool(self.left_score > self.right_score),
                "success": int(self._last_episode_success),
                "score_left": int(self.left_score),
                "score_right": int(self.right_score),
                "time_left_ratio": float(self._remaining_time_ratio()),
                "controlled_role": controlled_role,
                "level": int(self._last_episode_level),
                "reward_components": self._episode_reward_components.totals(),
                "reward_breakdown": {},
            }

        episode_level = int(self._current_level)
        parsed_action = self._decode_team_actions(action) if self.mode != "human" else self.ACTION_STAY
        prev_owner_team = self.ball_owner.team if self.ball_owner is not None else None
        self._tick(parsed_action)
        curr_owner_team = self.ball_owner.team if self.ball_owner is not None else None
        if self.steps >= self.max_steps:
            self.done = True

        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        if self.mode != "human":
            reward, reward_breakdown = self._score_reward(
                prev_owner_team=prev_owner_team,
                curr_owner_team=curr_owner_team,
            )
            self._episode_reward_components.add_from_mapping(
                reward_breakdown,
                self.REWARD_COMPONENT_KEY_TO_CODE,
            )
        else:
            reward = 0.0
            reward_breakdown = {}

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
        self.window_controller.close()
