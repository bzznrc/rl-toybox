"""Cardz environment, rules, and minimal lane-card UI."""

from __future__ import annotations

from dataclasses import dataclass
import time

import arcade
import numpy as np

from assets.paths import resolve_font_path
from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
    DEFAULT_CELL_INSET,
    DEFAULT_TILE_SIZE,
    INTER_FONT_FILE,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_unit, ordered_feature_vector
from core.match_tracker import compact_count_to_icons
from core.primitives import draw_status_bar, draw_status_square_icon
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController, Rect, TextCache, load_font_once
from core.utils import resolve_play_level
from games.cardz import config


PLAYER_P1 = 0
PLAYER_P2 = 1
PLAYER_NAMES = ("P1", "P2")
TURN_PLAYER_NAMES = ("A", "B")
PLAYER_OUTERS = {
    PLAYER_P1: COLOR_AQUA,
    PLAYER_P2: COLOR_CORAL,
}
PLAYER_INNERS = {
    PLAYER_P1: COLOR_DEEP_TEAL,
    PLAYER_P2: COLOR_BRICK_RED,
}
PANEL_OUTER = COLOR_FOG_GRAY
PANEL_INNER = COLOR_SLATE_GRAY
WORLD_BG = COLOR_DARK_NEUTRAL
LANE_PLACEHOLDER_OUTER = PANEL_OUTER
LANE_PLACEHOLDER_INNER = PANEL_INNER
LANE_CONTROL_OUTERS = PLAYER_OUTERS
LANE_CONTROL_INNERS = PLAYER_INNERS
SELECTED_OUTLINE = COLOR_LIGHT_NEUTRAL
ACTIVE_OUTLINE = COLOR_FOG_GRAY
CARD_DISABLED_OUTER = PANEL_OUTER
CARD_DISABLED_INNER = PANEL_INNER
FRAME_INSET = 4.0
OUTLINE_WIDTH = 2.0


@dataclass(frozen=True)
class CardDefinition:
    key: str
    kind: str
    cost: int
    power: int
    value: int
    card_id: float


CARD_DEFS = {
    key: CardDefinition(
        key=str(key),
        kind=str(config.CARD_KINDS[str(key)]),
        cost=int(config.CARD_COSTS[str(key)]),
        power=int(config.CARD_POWERS[str(key)]),
        value=int(config.CARD_VALUES[str(key)]),
        card_id=float(config.CARD_IDS[str(key)]),
    )
    for key in config.CARD_DRAW_ORDER
}
CARD_DRAW_WEIGHTS = np.asarray(config.CARD_DRAW_WEIGHTS, dtype=np.float64)

validate_curriculum_level_settings(
    min_level=config.MIN_LEVEL,
    max_level=config.MAX_LEVEL,
    level_settings=config.LEVEL_SETTINGS,
)

OPPONENT_PASS_SCORE_THRESHOLD = 0.35


class CardzEnv(Env):
    """Tiny 2-player lane-control environment with a compact public P1 view."""

    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)
    REWARD_COMPONENT_ORDER = tuple(config.REWARD_COMPONENT_NAMES)

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.log_ppo_metrics_line = bool(getattr(config, "PPO_METRICS_LOG_ENABLED", True))
        curriculum_config = build_curriculum_config(
            min_level=int(config.MIN_LEVEL),
            max_level=int(config.MAX_LEVEL),
            promotion_settings=config.CURRICULUM_PROMOTION,
        )
        self._curriculum = (
            ThreeLevelCurriculum(config=curriculum_config, level_settings=config.LEVEL_SETTINGS)
            if self.mode == "train"
            else None
        )
        self._current_level = (
            int(self._curriculum.get_level())
            if self._curriculum is not None
            else resolve_play_level(
                level=level,
                min_level=config.MIN_LEVEL,
                max_level=config.MAX_LEVEL,
                default_level=3,
            )
        )
        self._last_episode_level = int(self._current_level)
        self._last_episode_success = 0
        self._episode_counter = 0
        self._level_entropy_coef = 0.0
        self._opp_max_hand = int(config.MAX_HAND_SIZE)
        self._opp_random_move_prob = 0.0

        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            config.WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=self.mode == "human",
            vsync=False,
        )
        load_font_once(resolve_font_path(INTER_FONT_FILE))
        self._text = TextCache()

        self._lane_rects = self._build_lane_rects()
        self._hand_rects = self._build_hand_rects()
        self._score_track_rects = self._build_score_track_rects()

        self._rng = np.random.default_rng(int(config.BASE_SEED))
        self._opening_lead_player = int(PLAYER_P1)
        self.turn = 0
        self.current_player = int(PLAYER_P1)
        self._terminal_actor = int(PLAYER_P1)
        self._scores = np.zeros((config.PLAYER_COUNT,), dtype=np.int32)
        self._lane_scores = np.zeros((config.PLAYER_COUNT, config.NUM_LANES), dtype=np.int32)
        self._energy = np.zeros((config.PLAYER_COUNT,), dtype=np.int32)
        self._temp_modifiers = np.zeros((config.PLAYER_COUNT, config.NUM_LANES), dtype=np.int32)
        self._lane_banners = np.zeros((config.PLAYER_COUNT, config.NUM_LANES), dtype=np.bool_)
        self._hands: list[list[str]] = [[], []]
        self._lane_units: list[list[list[str]]] = [[[], []] for _ in range(config.NUM_LANES)]
        self._passed = [False, False]
        self._turn_has_acted = [False, False]
        self._done = False
        self._winner: int | None = None
        self._last_turn_points = (0, 0)
        self._last_action_text = "Match start"
        self._last_turn_text = ""
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._turn_pause_active = False
        self._selected_hand_slot: int | None = None
        self._hover_hand_slot: int | None = None
        self._hover_lane: int | None = None
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        self._apply_level_settings(int(self._current_level))
        self.reset()

    @staticmethod
    def _other_player(player: int) -> int:
        return PLAYER_P2 if int(player) == PLAYER_P1 else PLAYER_P1

    @staticmethod
    def _player_name(player: int) -> str:
        return PLAYER_NAMES[int(player)]

    @staticmethod
    def _turn_player_name(player: int) -> str:
        return TURN_PLAYER_NAMES[int(player)]

    def _turn_lead_player(self, turn_number: int | None = None) -> int:
        number = int(self.turn if turn_number is None else turn_number)
        if number <= 0:
            return int(self._opening_lead_player)
        return int((int(self._opening_lead_player) + int(number) - 1) % config.PLAYER_COUNT)

    def _lane_leader(self, lane: int) -> int | None:
        p1_total = int(self._lane_total(int(PLAYER_P1), int(lane)))
        p2_total = int(self._lane_total(int(PLAYER_P2), int(lane)))
        if p1_total > p2_total:
            return int(PLAYER_P1)
        if p2_total > p1_total:
            return int(PLAYER_P2)
        return None

    def _build_lane_rects(self) -> list[Rect]:
        total_width = (
            float(config.NUM_LANES) * float(config.LANE_WIDTH)
            + float(config.NUM_LANES - 1) * float(config.LANE_GAP)
        )
        left = (float(config.WORLD_WIDTH) - total_width) * 0.5
        return [
            Rect(
                left=float(left) + lane_idx * (float(config.LANE_WIDTH) + float(config.LANE_GAP)),
                top=float(config.LANE_TOP),
                width=float(config.LANE_WIDTH),
                height=float(config.LANE_HEIGHT),
            )
            for lane_idx in range(config.NUM_LANES)
        ]

    def _build_hand_rects(self) -> list[Rect]:
        total_width = (
            float(config.MAX_HAND_SIZE) * float(config.HAND_CARD_WIDTH)
            + float(config.MAX_HAND_SIZE - 1) * float(config.HAND_CARD_GAP)
        )
        left = (float(config.WORLD_WIDTH) - total_width) * 0.5
        return [
            Rect(
                left=float(left) + slot * (float(config.HAND_CARD_WIDTH) + float(config.HAND_CARD_GAP)),
                top=float(config.HAND_TOP),
                width=float(config.HAND_CARD_WIDTH),
                height=float(config.HAND_CARD_HEIGHT),
            )
            for slot in range(config.MAX_HAND_SIZE)
        ]

    def _build_score_track_rects(self) -> dict[int, list[Rect]]:
        rects: dict[int, list[Rect]] = {
            int(PLAYER_P1): [],
            int(PLAYER_P2): [],
        }
        for lane_rect in self._lane_rects:
            rects[int(PLAYER_P2)].append(
                Rect(
                    left=float(lane_rect.left),
                    top=float(lane_rect.top) - float(config.SCORE_TRACK_GAP) - float(config.SCORE_TRACK_HEIGHT),
                    width=float(lane_rect.width),
                    height=float(config.SCORE_TRACK_HEIGHT),
                )
            )
            rects[int(PLAYER_P1)].append(
                Rect(
                    left=float(lane_rect.left),
                    top=float(lane_rect.bottom) + float(config.SCORE_TRACK_GAP),
                    width=float(lane_rect.width),
                    height=float(config.SCORE_TRACK_HEIGHT),
                )
            )
        return rects

    def get_entropy_coef_for_level(self, level: int | None = None) -> float | None:
        if level is None or int(level) == int(self._current_level):
            return float(self._level_entropy_coef)
        settings = config.LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Cardz.")
        if "entropy_coef" not in settings:
            raise ValueError("Cardz LEVEL_SETTINGS entries must define 'entropy_coef'.")
        return float(settings["entropy_coef"])

    def _apply_level_settings(self, level: int) -> None:
        settings = config.LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Cardz.")

        if "entropy_coef" not in settings:
            raise ValueError("Cardz LEVEL_SETTINGS entries must define 'entropy_coef'.")
        if "opp_max_hand" not in settings:
            raise ValueError("Cardz LEVEL_SETTINGS entries must define 'opp_max_hand'.")
        if "opp_random_move_prob" not in settings:
            raise ValueError("Cardz LEVEL_SETTINGS entries must define 'opp_random_move_prob'.")

        opp_max_hand = max(1, min(int(config.MAX_HAND_SIZE), int(settings["opp_max_hand"])))
        random_move_prob = float(max(0.0, min(1.0, float(settings["opp_random_move_prob"]))))

        self._current_level = int(level)
        self._level_entropy_coef = float(settings["entropy_coef"])
        self._opp_max_hand = int(opp_max_hand)
        self._opp_random_move_prob = float(random_move_prob)

    def _hand_limit_for_player(self, player: int) -> int:
        if int(player) == int(PLAYER_P2):
            return int(self._opp_max_hand)
        return int(config.MAX_HAND_SIZE)

    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {str(name): 0.0 for name in self.REWARD_COMPONENT_ORDER}

    def _sample_card(self) -> str:
        card_idx = int(self._rng.choice(np.arange(len(config.CARD_DRAW_ORDER)), p=CARD_DRAW_WEIGHTS))
        return str(config.CARD_DRAW_ORDER[card_idx])

    def _draw_card_to_hand(self, player: int) -> None:
        hand = self._hands[int(player)]
        if len(hand) >= int(self._hand_limit_for_player(int(player))):
            return
        hand.append(self._sample_card())

    def _reset_state(self) -> None:
        self.turn = 0
        self.current_player = int(PLAYER_P1)
        self._terminal_actor = int(PLAYER_P1)
        self._scores.fill(0)
        self._lane_scores.fill(0)
        self._energy.fill(0)
        self._temp_modifiers.fill(0)
        self._lane_banners.fill(False)
        self._hands = [[], []]
        self._lane_units = [[[], []] for _ in range(config.NUM_LANES)]
        self._passed = [False, False]
        self._turn_has_acted = [False, False]
        self._done = False
        self._winner = None
        self._last_turn_points = (0, 0)
        self._last_action_text = "Match start"
        self._last_turn_text = ""
        self._turn_pause_active = False
        self._selected_hand_slot = None
        self._hover_hand_slot = None
        self._hover_lane = None

    def _begin_turn(self) -> None:
        self.turn += 1
        energy_value = min(int(config.MAX_TURNS), max(1, int(self.turn)))
        self._energy[:] = int(energy_value)
        self._passed = [False, False]
        self._turn_has_acted = [False, False]
        self.current_player = int(self._turn_lead_player(int(self.turn)))
        self._selected_hand_slot = None

    def _draw_end_of_turn_cards(self) -> None:
        for player in range(config.PLAYER_COUNT):
            if len(self._hands[int(player)]) < int(self._hand_limit_for_player(int(player))):
                self._draw_card_to_hand(int(player))

    def reset(self) -> np.ndarray:
        self._episode_counter += 1
        seed = int(config.BASE_SEED) + int(self._episode_counter) * 9973
        self._rng = np.random.default_rng(seed)
        self._reset_state()
        self._opening_lead_player = int(self._rng.integers(0, int(config.PLAYER_COUNT)))
        self._episode_reward_components.reset()
        for player in range(config.PLAYER_COUNT):
            hand_target = int(min(int(config.STARTING_HAND_SIZE), int(self._hand_limit_for_player(int(player)))))
            for _ in range(hand_target):
                self._draw_card_to_hand(int(player))
        self._begin_turn()
        self._advance_opponent_until_player_turn(self._empty_reward_breakdown())
        self._last_obs = self._build_observation()
        if self.show_game:
            self.render()
        return np.asarray(self._last_obs, dtype=np.float32)

    def _card_in_slot(self, player: int, slot: int) -> str | None:
        hand = self._hands[int(player)]
        if 0 <= int(slot) < len(hand):
            return str(hand[int(slot)])
        return None

    def _lane_unit_count(self, player: int, lane: int) -> int:
        return len(self._lane_units[int(lane)][int(player)])

    def _lane_unit_power(self, player: int, lane: int) -> int:
        return int(sum(int(CARD_DEFS[str(card_key)].power) for card_key in self._lane_units[int(lane)][int(player)]))

    def _lane_has_banner(self, player: int, lane: int) -> bool:
        return bool(self._lane_banners[int(player), int(lane)])

    def _lane_banner_bonus(self, player: int, lane: int) -> int:
        if (not self._lane_has_banner(int(player), int(lane))) or self._lane_unit_count(int(player), int(lane)) <= 0:
            return 0
        return int(config.BAN_POWER)

    def _lane_has_attack(self, player: int, lane: int) -> bool:
        return int(self._temp_modifiers[int(player), int(lane)]) > 0

    def _lane_attack_bonus(self, player: int, lane: int) -> int:
        if not self._lane_has_attack(int(player), int(lane)):
            return 0
        return int(config.ATK_DELTA)

    def _persistent_lane_total(self, player: int, lane: int) -> int:
        return int(self._lane_unit_power(int(player), int(lane)) + self._lane_banner_bonus(int(player), int(lane)))

    def _lane_total(self, player: int, lane: int) -> int:
        return int(self._persistent_lane_total(int(player), int(lane)) + self._lane_attack_bonus(int(player), int(lane)))

    def _lane_margin(self, player: int, lane: int, *, include_temp: bool) -> int:
        opponent = int(self._other_player(int(player)))
        if include_temp:
            return int(self._lane_total(int(player), int(lane)) - self._lane_total(int(opponent), int(lane)))
        return int(
            self._persistent_lane_total(int(player), int(lane))
            - self._persistent_lane_total(int(opponent), int(lane))
        )

    def _lane_status_code(self, player: int, lane: int) -> float:
        # Public lane status bits: BAN=1, ATK=2.
        status_code = 0
        if self._lane_has_banner(int(player), int(lane)):
            status_code += 1
        if self._lane_has_attack(int(player), int(lane)):
            status_code += 2
        return float(status_code)

    def _phase_code(self) -> float:
        # Phase codes stay as compact public state ids rather than one-hot flags.
        if bool(self._passed[int(PLAYER_P2)]):
            return 2.0
        if int(self._turn_lead_player(int(self.turn))) == int(PLAYER_P1):
            return 0.0
        return 1.0

    def _build_observation(self) -> np.ndarray:
        feature_values: dict[str, float] = {
            "turn_norm": float(clip_unit(float(self.turn) / float(config.TURN_NORMALIZER))),
            "energy_p1_norm": float(clip_unit(float(self._energy[int(PLAYER_P1)]) / float(config.ENERGY_NORMALIZER))),
            "energy_p2_norm": float(clip_unit(float(self._energy[int(PLAYER_P2)]) / float(config.ENERGY_NORMALIZER))),
            "score_p1_norm": float(
                clip_unit(float(self._scores[int(PLAYER_P1)]) / float(config.MATCH_SCORE_NORMALIZER))
            ),
            "score_p2_norm": float(
                clip_unit(float(self._scores[int(PLAYER_P2)]) / float(config.MATCH_SCORE_NORMALIZER))
            ),
            "hand_count_p2_norm": float(
                clip_unit(float(len(self._hands[int(PLAYER_P2)])) / float(config.HAND_COUNT_NORMALIZER))
            ),
            "phase_code": float(self._phase_code()),
        }
        for lane in range(config.NUM_LANES):
            feature_values[f"lane_{lane}_power_p1_norm"] = float(
                clip_unit(float(self._lane_total(int(PLAYER_P1), int(lane))) / float(config.LANE_POWER_NORMALIZER))
            )
            feature_values[f"lane_{lane}_power_p2_norm"] = float(
                clip_unit(float(self._lane_total(int(PLAYER_P2), int(lane))) / float(config.LANE_POWER_NORMALIZER))
            )
            feature_values[f"lane_{lane}_unit_count_p1_norm"] = float(
                clip_unit(float(self._lane_unit_count(int(PLAYER_P1), int(lane))) / float(config.LANE_COUNT_NORMALIZER))
            )
            feature_values[f"lane_{lane}_unit_count_p2_norm"] = float(
                clip_unit(float(self._lane_unit_count(int(PLAYER_P2), int(lane))) / float(config.LANE_COUNT_NORMALIZER))
            )
            feature_values[f"lane_{lane}_status_p1"] = float(self._lane_status_code(int(PLAYER_P1), int(lane)))
            feature_values[f"lane_{lane}_status_p2"] = float(self._lane_status_code(int(PLAYER_P2), int(lane)))
        for slot in range(config.MAX_HAND_SIZE):
            card_key = self._card_in_slot(int(PLAYER_P1), int(slot))
            if card_key is None:
                feature_values[f"hand_{slot}_card_id"] = 0.0
                continue
            card = CARD_DEFS[str(card_key)]
            feature_values[f"hand_{slot}_card_id"] = float(card.card_id)
        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Cardz observation expected {self.OBS_DIM} features, got {obs.shape[0]}.")
        if not np.isfinite(obs).all():
            raise RuntimeError("Cardz observation contains non-finite values.")
        return obs

    def _can_play_card(self, player: int, card_key: str | None, lane: int) -> bool:
        if self._done or card_key is None:
            return False
        if not (0 <= int(lane) < int(config.NUM_LANES)):
            return False
        if bool(self._passed[int(player)]):
            return False
        card = CARD_DEFS[str(card_key)]
        if int(self._energy[int(player)]) < int(card.cost):
            return False
        if card.kind == "unit":
            return self._lane_unit_count(int(player), int(lane)) < int(config.MAX_UNITS_PER_LANE)
        if card.key == "Atk":
            return not self._lane_has_attack(int(player), int(lane))
        if card.key == "Ban":
            return self._lane_unit_count(int(player), int(lane)) > 0 and not self._lane_has_banner(int(player), int(lane))
        return False

    def _has_any_play_action(self, player: int) -> bool:
        if self._done or bool(self._passed[int(player)]):
            return False
        for slot in range(config.MAX_HAND_SIZE):
            card_key = self._card_in_slot(int(player), slot)
            if card_key is None:
                continue
            for lane in range(config.NUM_LANES):
                if self._can_play_card(int(player), str(card_key), int(lane)):
                    return True
        return False

    def _action_mask_for_player(self, player: int) -> np.ndarray:
        mask = np.zeros((self.ACT_DIM,), dtype=np.bool_)
        if self._done:
            return mask

        actor = int(player)
        mask[int(config.PASS_ACTION_INDEX)] = True
        if bool(self._passed[actor]):
            return mask
        for slot in range(config.MAX_HAND_SIZE):
            card_key = self._card_in_slot(actor, slot)
            if card_key is None:
                continue
            for lane in range(config.NUM_LANES):
                if self._can_play_card(actor, card_key, lane):
                    action_idx = int(slot * config.NUM_LANES + lane)
                    mask[action_idx] = True
        return mask

    def get_action_mask(self, _obs: object | None = None) -> np.ndarray:
        return self._action_mask_for_player(int(PLAYER_P1))

    def _legal_play_actions(self, player: int) -> np.ndarray:
        mask = self._action_mask_for_player(int(player))
        return np.flatnonzero(mask[: int(config.PASS_ACTION_INDEX)])

    @staticmethod
    def _lane_advantage_score(margin: int) -> float:
        clipped_margin = max(-6, min(6, int(margin)))
        if clipped_margin > 0:
            return 1.0 + 0.35 * float(clipped_margin)
        if clipped_margin == 0:
            return 0.15
        return -1.0 + 0.20 * float(clipped_margin)

    def _heuristic_action_score(self, player: int, action_idx: int) -> float:
        hand_slot, lane = self._decode_action(int(action_idx))
        if hand_slot is None or lane is None:
            return float("-inf")
        card_key = self._card_in_slot(int(player), int(hand_slot))
        if card_key is None:
            return float("-inf")
        card = CARD_DEFS[str(card_key)]
        turns_after_current = max(0, int(config.MAX_TURNS) - int(self.turn))
        turn_margin_before = int(self._lane_margin(int(player), int(lane), include_temp=True))
        persistent_margin_before = int(self._lane_margin(int(player), int(lane), include_temp=False))
        turn_margin_after = int(turn_margin_before)
        persistent_margin_after = int(persistent_margin_before)

        if card.kind == "unit":
            turn_margin_after += int(card.power)
            persistent_margin_after += int(card.power)
        elif card.key == "Atk":
            turn_margin_after += int(config.ATK_DELTA)
        elif card.key == "Ban":
            turn_margin_after += int(config.BAN_POWER)
            persistent_margin_after += int(config.BAN_POWER)

        immediate_delta = float(self._lane_advantage_score(turn_margin_after) - self._lane_advantage_score(turn_margin_before))
        future_delta = float(
            self._lane_advantage_score(persistent_margin_after)
            - self._lane_advantage_score(persistent_margin_before)
        )
        score = 2.0 * immediate_delta + 0.60 * float(turns_after_current) * future_delta

        if turn_margin_before <= 0 < turn_margin_after:
            score += 2.0
        elif turn_margin_before < 0 and turn_margin_after == 0:
            score += 0.75

        if card.kind == "unit":
            if self._lane_unit_count(int(player), int(lane)) == 0:
                score += 0.35
            if self._lane_has_banner(int(player), int(lane)):
                score += 0.40
            if turn_margin_before > 2:
                score -= 0.60
        elif card.key == "Ban":
            score += 0.20 * float(self._lane_unit_count(int(player), int(lane)))
            if turn_margin_before >= 2:
                score += 0.20
        elif card.key == "Atk":
            if turn_margin_before <= 0 < turn_margin_after:
                score += 0.75
            if persistent_margin_before < 0:
                score += 0.20

        score += 0.08 * float(card.value) - 0.12 * float(card.cost)
        score += float(self._rng.random()) * 1e-3
        return float(score)

    def _select_scripted_action(self, player: int) -> int:
        legal_play_actions = self._legal_play_actions(int(player))
        if legal_play_actions.size <= 0:
            return int(config.PASS_ACTION_INDEX)
        if float(self._rng.random()) < float(self._opp_random_move_prob):
            return int(self._rng.choice(legal_play_actions))

        best_action = int(config.PASS_ACTION_INDEX)
        best_score = float("-inf")
        for action_idx in legal_play_actions.tolist():
            score = float(self._heuristic_action_score(int(player), int(action_idx)))
            if score > best_score:
                best_score = float(score)
                best_action = int(action_idx)
        if best_score <= float(OPPONENT_PASS_SCORE_THRESHOLD):
            return int(config.PASS_ACTION_INDEX)
        return int(best_action)

    def _resolve_valid_action(self, action: object) -> int:
        mask = self.get_action_mask()
        legal_actions = np.flatnonzero(mask)
        if legal_actions.size <= 0:
            return int(config.PASS_ACTION_INDEX)
        try:
            action_idx = int(action)
        except (TypeError, ValueError):
            return int(legal_actions[0])
        if 0 <= int(action_idx) < self.ACT_DIM and bool(mask[int(action_idx)]):
            return int(action_idx)
        return int(legal_actions[0])

    def _decode_action(self, action_idx: int) -> tuple[int | None, int | None]:
        if int(action_idx) == int(config.PASS_ACTION_INDEX):
            return None, None
        hand_slot = int(action_idx) // int(config.NUM_LANES)
        lane = int(action_idx) % int(config.NUM_LANES)
        return int(hand_slot), int(lane)

    def _play_action_text(self, player: int, card_key: str, lane: int) -> str:
        return f"{self._turn_player_name(player)} {self._card_label_text(CARD_DEFS[str(card_key)])} -> L{int(lane) + 1}"

    def _apply_play(self, player: int, hand_slot: int, lane: int) -> None:
        hand = self._hands[int(player)]
        card_key = str(hand.pop(int(hand_slot)))
        card = CARD_DEFS[str(card_key)]
        self._energy[int(player)] -= int(card.cost)
        if card.kind == "unit":
            self._lane_units[int(lane)][int(player)].append(str(card.key))
        elif card.key == "Atk":
            self._temp_modifiers[int(player), int(lane)] = int(config.ATK_DELTA)
        elif card.key == "Ban":
            self._lane_banners[int(player), int(lane)] = True
        self._last_action_text = self._play_action_text(int(player), str(card.key), int(lane))

    def _score_turn(self) -> tuple[int, int]:
        turn_points = [0, 0]
        for lane in range(config.NUM_LANES):
            lane_p1 = int(self._lane_total(PLAYER_P1, lane))
            lane_p2 = int(self._lane_total(PLAYER_P2, lane))
            if lane_p1 > lane_p2:
                turn_points[int(PLAYER_P1)] += 1
                self._lane_scores[int(PLAYER_P1), int(lane)] += 1
            elif lane_p2 > lane_p1:
                turn_points[int(PLAYER_P2)] += 1
                self._lane_scores[int(PLAYER_P2), int(lane)] += 1
        self._scores[int(PLAYER_P1)] += int(turn_points[int(PLAYER_P1)])
        self._scores[int(PLAYER_P2)] += int(turn_points[int(PLAYER_P2)])
        self._last_turn_points = (int(turn_points[0]), int(turn_points[1]))
        self._temp_modifiers.fill(0)
        return int(turn_points[0]), int(turn_points[1])

    def _should_end_turn(self) -> bool:
        return bool(self._passed[int(PLAYER_P1)]) and bool(self._passed[int(PLAYER_P2)])

    def _pause_before_turn_resolution(self) -> None:
        if (not self.show_game) or float(config.TURN_RESOLUTION_PAUSE_SECONDS) <= 0.0:
            return
        self._turn_pause_active = True
        deadline = float(time.monotonic()) + float(config.TURN_RESOLUTION_PAUSE_SECONDS)
        try:
            while float(time.monotonic()) < deadline:
                self.window_controller.poll_events_or_raise()
                self.render()
                self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
        finally:
            self._turn_pause_active = False

    def _finish_turn(self, reward_breakdown: dict[str, float]) -> None:
        self._pause_before_turn_resolution()
        turn_number = int(self.turn)
        points_p1, points_p2 = self._score_turn()
        progress_reward = float(points_p1 - points_p2) * float(config.REWARD_PROGRESS_TURN_POINTS_PER_LANE)
        reward_breakdown["reward_progress_turn_points"] += float(progress_reward)
        self._last_turn_text = f"T{turn_number} scored {points_p1}-{points_p2}"

        if int(self.turn) >= int(config.MAX_TURNS):
            self._done = True
            self._terminal_actor = int(PLAYER_P1)
            self.current_player = int(PLAYER_P1)
            p1_score = int(self._scores[int(PLAYER_P1)])
            p2_score = int(self._scores[int(PLAYER_P2)])
            if p1_score > p2_score:
                self._winner = int(PLAYER_P1)
                reward_breakdown["reward_terminal_match_win"] += float(config.REWARD_TERMINAL_MATCH_WIN)
            elif p1_score < p2_score:
                self._winner = int(PLAYER_P2)
                reward_breakdown["reward_terminal_match_loss"] += float(config.REWARD_TERMINAL_MATCH_LOSS)
            else:
                self._winner = None
                reward_breakdown["reward_terminal_match_draw"] += float(config.REWARD_TERMINAL_MATCH_DRAW)
            return

        self._draw_end_of_turn_cards()
        self._begin_turn()

    def _advance_after_pass(self, actor: int) -> None:
        lead = int(self._turn_lead_player(int(self.turn)))
        follow = int(self._other_player(int(lead)))
        if int(actor) == int(lead):
            self.current_player = int(follow)
            return
        self.current_player = int(actor)

    def _advance_opponent_until_player_turn(self, reward_breakdown: dict[str, float]) -> None:
        while (not self._done) and int(self.current_player) == int(PLAYER_P2):
            action_idx = int(self._select_scripted_action(int(PLAYER_P2)))
            self._apply_action(int(action_idx), reward_breakdown)
            if self.show_game:
                self.render()
                self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)

    def _info(
        self,
        actor: int | None,
        reward_breakdown: dict[str, float] | None,
        *,
        episode_level: int | None = None,
        level_changed: bool = False,
    ) -> dict[str, object]:
        actor_value = int(PLAYER_P1 if actor is None else actor)
        opponent = int(self._other_player(actor_value))
        done = bool(self._done)
        info: dict[str, object] = {
            "level": int(self._current_level if episode_level is None else episode_level),
            "level_changed": bool(level_changed),
            "turn": int(self.turn),
            "actor": int(actor_value),
            "actor_label": self._player_name(int(actor_value)),
            "current_player": int(self.current_player),
            "turn_points_p1": int(self._last_turn_points[int(PLAYER_P1)]),
            "turn_points_p2": int(self._last_turn_points[int(PLAYER_P2)]),
            "score_p1": int(self._scores[int(PLAYER_P1)]),
            "score_p2": int(self._scores[int(PLAYER_P2)]),
            "winner": "draw" if self._winner is None else self._player_name(int(self._winner)),
            "win": bool(done and self._winner is not None and int(self._winner) == int(actor_value)),
            "draw": bool(done and self._winner is None),
            "success": 1 if done and self._winner is not None and int(self._winner) == int(actor_value) else 0,
            "score_self": int(self._scores[int(actor_value)]),
            "score_opp": int(self._scores[int(opponent)]),
            "energy_self": int(self._energy[int(actor_value)]),
            "energy_opp": int(self._energy[int(opponent)]),
            "hand_count_p1": int(len(self._hands[int(PLAYER_P1)])),
            "hand_count_p2": int(len(self._hands[int(PLAYER_P2)])),
            "lane_scores_p1": [int(value) for value in self._lane_scores[int(PLAYER_P1)].tolist()],
            "lane_scores_p2": [int(value) for value in self._lane_scores[int(PLAYER_P2)].tolist()],
        }
        if reward_breakdown is not None:
            info["reward_breakdown"] = dict(reward_breakdown) if self.mode != "human" else {}
        if done:
            info["reward_components"] = self._episode_reward_components.totals()
        return info

    def _settle_step(self, reward_breakdown: dict[str, float]) -> tuple[float, int, bool]:
        reward = float(sum(float(value) for value in reward_breakdown.values()))
        if self.mode != "human":
            for key, value in reward_breakdown.items():
                self._episode_reward_components.add(str(key), float(value))

        episode_level = int(self._current_level)
        level_changed = False
        if self._done:
            self._last_episode_level = int(episode_level)
            self._last_episode_success = 1 if self._winner is not None and int(self._winner) == int(PLAYER_P1) else 0
            if self._curriculum is not None:
                self._current_level, level_changed = advance_curriculum(
                    self._curriculum,
                    success=int(self._last_episode_success),
                    current_level=int(self._current_level),
                    apply_level=self._apply_level_settings,
                )
            self.current_player = int(PLAYER_P1)

        self._last_obs = self._build_observation()
        return float(reward), int(episode_level), bool(level_changed)

    def _complete_step_result(
        self,
        reward_breakdown: dict[str, float],
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        reward, episode_level, level_changed = self._settle_step(reward_breakdown)
        info = self._info(
            int(PLAYER_P1),
            reward_breakdown,
            episode_level=int(episode_level),
            level_changed=bool(level_changed),
        )
        return np.asarray(self._last_obs, dtype=np.float32), float(reward), bool(self._done), info

    def _apply_action(self, action_idx: int, reward_breakdown: dict[str, float]) -> None:
        actor = int(self.current_player)
        self._terminal_actor = int(PLAYER_P1)
        self._selected_hand_slot = None

        passed_this_action = bool(int(action_idx) == int(config.PASS_ACTION_INDEX))
        if int(action_idx) == int(config.PASS_ACTION_INDEX):
            self._passed[int(actor)] = True
            self._last_action_text = f"{self._turn_player_name(actor)} pass"
        else:
            hand_slot, lane = self._decode_action(int(action_idx))
            if hand_slot is None or lane is None:
                passed_this_action = True
                self._passed[int(actor)] = True
                self._last_action_text = f"{self._turn_player_name(actor)} pass"
            else:
                self._apply_play(int(actor), int(hand_slot), int(lane))
        self._turn_has_acted[int(actor)] = True

        if self._should_end_turn():
            self._finish_turn(reward_breakdown)
        elif passed_this_action and (not self._done):
            self._advance_after_pass(int(actor))

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        self.window_controller.poll_events_or_raise()

        if self.mode == "human":
            return self._step_human()

        if self._done:
            return np.asarray(self._last_obs, dtype=np.float32), 0.0, True, self._info(int(PLAYER_P1), None)

        reward_breakdown = self._empty_reward_breakdown()
        self._advance_opponent_until_player_turn(reward_breakdown)
        if self._done:
            return self._complete_step_result(reward_breakdown)

        action_idx = self._resolve_valid_action(action)
        self._apply_action(int(action_idx), reward_breakdown)
        self._advance_opponent_until_player_turn(reward_breakdown)
        obs, reward, done, info = self._complete_step_result(reward_breakdown)
        if self.show_game:
            self.render()
        self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
        return obs, float(reward), bool(done), info

    def _key_to_slot(self, key_code: int) -> int | None:
        key_map = {
            arcade.key.KEY_1: 0,
            arcade.key.NUM_1: 0,
            arcade.key.KEY_2: 1,
            arcade.key.NUM_2: 1,
            arcade.key.KEY_3: 2,
            arcade.key.NUM_3: 2,
            arcade.key.KEY_4: 3,
            arcade.key.NUM_4: 3,
            arcade.key.KEY_5: 4,
            arcade.key.NUM_5: 4,
        }
        return key_map.get(int(key_code))

    def _key_to_lane(self, key_code: int) -> int | None:
        key_map = {
            arcade.key.Q: 0,
            arcade.key.W: 1,
            arcade.key.E: 2,
        }
        return key_map.get(int(key_code))

    def _mouse_point_top_left(self) -> tuple[float, float] | None:
        mouse_pos = self.window_controller.mouse_position()
        if mouse_pos is None:
            return None
        return float(mouse_pos[0]), float(self.window_controller.to_top_left_y(float(mouse_pos[1])))

    @staticmethod
    def _point_in_rect(x: float, y: float, rect: Rect) -> bool:
        return float(rect.left) <= float(x) <= float(rect.right) and float(rect.top) <= float(y) <= float(rect.bottom)

    def _update_hover_state(self) -> None:
        self._hover_hand_slot = None
        self._hover_lane = None
        point = self._mouse_point_top_left()
        if point is None:
            return
        mouse_x, mouse_y = point
        for slot, rect in enumerate(self._hand_rects):
            if self._point_in_rect(float(mouse_x), float(mouse_y), rect):
                self._hover_hand_slot = int(slot)
                break
        for lane, rect in enumerate(self._lane_rects):
            if self._point_in_rect(float(mouse_x), float(mouse_y), rect):
                self._hover_lane = int(lane)
                break

    def _selected_action_for_lane(self, lane: int) -> int | None:
        if self._selected_hand_slot is None:
            return None
        action_idx = int(self._selected_hand_slot) * int(config.NUM_LANES) + int(lane)
        mask = self.get_action_mask()
        if 0 <= int(action_idx) < self.ACT_DIM and bool(mask[int(action_idx)]):
            return int(action_idx)
        return None

    def _consume_human_action(self) -> int | None:
        self._update_hover_state()
        mask = self.get_action_mask()
        for key_code in self.window_controller.consume_key_presses():
            if int(key_code) in {arcade.key.ESCAPE, arcade.key.BACKSPACE}:
                self._selected_hand_slot = None
                continue
            if int(key_code) == int(arcade.key.SPACE):
                self._selected_hand_slot = None
                if bool(mask[int(config.PASS_ACTION_INDEX)]):
                    return int(config.PASS_ACTION_INDEX)
                continue
            slot = self._key_to_slot(int(key_code))
            if slot is not None:
                if self._card_in_slot(int(PLAYER_P1), int(slot)) is not None:
                    self._selected_hand_slot = None if self._selected_hand_slot == int(slot) else int(slot)
                continue
            lane = self._key_to_lane(int(key_code))
            if lane is not None:
                action_idx = self._selected_action_for_lane(int(lane))
                if action_idx is not None:
                    return int(action_idx)

        for mouse_press in self.window_controller.consume_mouse_presses():
            mouse_x = float(mouse_press.x)
            mouse_y = float(self.window_controller.to_top_left_y(float(mouse_press.y)))
            clicked_hand = False
            for slot, rect in enumerate(self._hand_rects):
                if not self._point_in_rect(mouse_x, mouse_y, rect):
                    continue
                clicked_hand = True
                if self._card_in_slot(int(PLAYER_P1), int(slot)) is not None:
                    self._selected_hand_slot = None if self._selected_hand_slot == int(slot) else int(slot)
                break
            if clicked_hand:
                continue
            for lane, rect in enumerate(self._lane_rects):
                if not self._point_in_rect(mouse_x, mouse_y, rect):
                    continue
                action_idx = self._selected_action_for_lane(int(lane))
                if action_idx is not None:
                    return int(action_idx)
                break
        return None

    def _handle_human_terminal(self) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.window_controller.consume_mouse_presses():
            return self.reset(), 0.0, False, {"level": int(self._current_level)}
        for key_code in self.window_controller.consume_key_presses():
            if int(key_code) in {arcade.key.ENTER, arcade.key.SPACE}:
                return self.reset(), 0.0, False, {"level": int(self._current_level)}
        self.render()
        self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
        return np.asarray(self._last_obs, dtype=np.float32), 0.0, False, self._info(int(PLAYER_P1), None)

    def _step_human(self) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self._done:
            return self._handle_human_terminal()

        reward_breakdown = self._empty_reward_breakdown()
        self._advance_opponent_until_player_turn(reward_breakdown)
        if self._done:
            self._settle_step(reward_breakdown)
            return self._handle_human_terminal()

        action_idx = self._consume_human_action()
        if action_idx is None:
            self.render()
            self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
            return np.asarray(self._last_obs, dtype=np.float32), 0.0, False, self._info(int(PLAYER_P1), None)

        self._apply_action(int(action_idx), reward_breakdown)
        self._advance_opponent_until_player_turn(reward_breakdown)
        obs, _, done, info = self._complete_step_result(reward_breakdown)
        if done:
            return self._handle_human_terminal()
        self.render()
        self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
        return obs, 0.0, False, info

    def _draw_panel(
        self,
        rect: Rect,
        *,
        outer_color: tuple[int, int, int] | tuple[int, int, int, int],
        inner_color: tuple[int, int, int] | tuple[int, int, int, int],
        inset: float = FRAME_INSET,
    ) -> None:
        bottom = self.window_controller.top_left_to_bottom(float(rect.top), float(rect.height))
        arcade.draw_lbwh_rectangle_filled(float(rect.left), float(bottom), float(rect.width), float(rect.height), outer_color)
        inner_left = float(rect.left) + float(inset)
        inner_top = float(rect.top) + float(inset)
        inner_width = max(1.0, float(rect.width) - 2.0 * float(inset))
        inner_height = max(1.0, float(rect.height) - 2.0 * float(inset))
        inner_bottom = self.window_controller.top_left_to_bottom(float(inner_top), float(inner_height))
        arcade.draw_lbwh_rectangle_filled(inner_left, float(inner_bottom), inner_width, inner_height, inner_color)

    def _draw_outline(
        self,
        rect: Rect,
        color: tuple[int, int, int] | tuple[int, int, int, int],
        *,
        line_width: float = OUTLINE_WIDTH,
    ) -> None:
        bottom = self.window_controller.top_left_to_bottom(float(rect.top), float(rect.height))
        arcade.draw_lbwh_rectangle_outline(
            float(rect.left),
            float(bottom),
            float(rect.width),
            float(rect.height),
            color,
            float(line_width),
        )

    def _lane_panel_colors(self, lane: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        leader = self._lane_leader(int(lane))
        if leader is None:
            return PANEL_OUTER, PANEL_INNER
        return LANE_CONTROL_OUTERS[int(leader)], LANE_CONTROL_INNERS[int(leader)]

    def _draw_score_marker(
        self,
        *,
        center_x: float,
        center_y: float,
        size: float,
        outer_color: tuple[int, int, int],
        inner_color: tuple[int, int, int],
        packed: bool,
    ) -> None:
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(config.LANE_SCORE_ICON_INSET),
            packed=bool(packed),
            packed_marker_color=outer_color,
            packed_marker_size=max(2.0, round(float(size) * 0.3)),
        )

    def _draw_score_track(self, player: int, lane: int, rect: Rect) -> None:
        lane_value = max(0, int(self._lane_total(int(player), int(lane))))
        icons = compact_count_to_icons(int(lane_value), pack_size=5)
        if not icons:
            return
        icons = icons[: int(config.SCORE_TRACK_SLOTS)]
        outer_color = PLAYER_OUTERS[int(player)]
        inner_color = PLAYER_INNERS[int(player)]
        slot_width = float(rect.width) / float(config.SCORE_TRACK_SLOTS)
        icon_size = max(
            8.0,
            min(
                float(config.LANE_SCORE_ICON_SIZE),
                float(rect.height) - 4.0,
                float(slot_width) - 4.0,
            ),
        )
        center_y = float(self.window_controller.to_arcade_y(float(rect.top) + float(rect.height) * 0.5))
        for idx, packed in enumerate(icons):
            center_x = float(rect.left) + (float(idx) + 0.5) * slot_width
            self._draw_score_marker(
                center_x=float(center_x),
                center_y=float(center_y),
                size=float(icon_size),
                outer_color=outer_color,
                inner_color=inner_color,
                packed=bool(packed),
            )

    @staticmethod
    def _card_value_text(card: CardDefinition) -> str:
        if card.kind == "unit":
            return str(int(card.power))
        if card.key == "Atk":
            return f"+{int(config.ATK_DELTA)}"
        if card.key == "Ban":
            return f"+{int(config.BAN_POWER)}"
        return str(int(card.value))

    @staticmethod
    def _card_label_text(card: CardDefinition) -> str:
        if card.key == "Atk":
            return "ATK"
        if card.key == "Ban":
            return "BAN"
        return str(card.key)

    def _banner_marker_rect(self, lane_rect: Rect, player: int) -> Rect:
        anchor_rect = self._unit_slot_rect(lane_rect, int(player), 0)
        size = float(config.BANNER_MARKER_SIZE)
        if int(player) == int(PLAYER_P2):
            top = float(anchor_rect.top) - float(config.BANNER_MARKER_GAP) - size
        else:
            top = float(anchor_rect.bottom) + float(config.BANNER_MARKER_GAP)
        return Rect(
            left=float(lane_rect.left) + (float(lane_rect.width) - size) * 0.5,
            top=float(top),
            width=size,
            height=size,
        )

    def _draw_banner_marker(self, lane: int, player: int, lane_rect: Rect) -> None:
        if not self._lane_has_banner(int(player), int(lane)):
            return
        marker_rect = self._banner_marker_rect(lane_rect, int(player))
        self._draw_panel(
            marker_rect,
            outer_color=PLAYER_OUTERS[int(player)],
            inner_color=PLAYER_INNERS[int(player)],
            inset=FRAME_INSET,
        )

    def _attack_marker_rects(self, lane_rect: Rect, player: int) -> tuple[Rect, Rect]:
        left_slot_rect = self._unit_slot_rect(lane_rect, int(player), 0)
        right_slot_rect = self._unit_slot_rect(lane_rect, int(player), 1)
        size = float(config.BANNER_MARKER_SIZE)
        if int(player) == int(PLAYER_P2):
            top = float(left_slot_rect.bottom) + float(config.BANNER_MARKER_GAP)
        else:
            top = float(left_slot_rect.top) - float(config.BANNER_MARKER_GAP) - size
        return (
            Rect(
                left=float(left_slot_rect.right) - size,
                top=float(top),
                width=size,
                height=size,
            ),
            Rect(
                left=float(right_slot_rect.left),
                top=float(top),
                width=size,
                height=size,
            ),
        )

    def _draw_attack_markers(self, lane: int, player: int, lane_rect: Rect) -> None:
        if not self._lane_has_attack(int(player), int(lane)):
            return
        for marker_rect in self._attack_marker_rects(lane_rect, int(player)):
            self._draw_panel(
                marker_rect,
                outer_color=PLAYER_OUTERS[int(player)],
                inner_color=PLAYER_INNERS[int(player)],
                inset=FRAME_INSET,
            )

    def _draw_card_face(
        self,
        rect: Rect,
        *,
        card_key: str | None,
        outer_color: tuple[int, int, int],
        inner_color: tuple[int, int, int],
        outline_color: tuple[int, int, int] | None = None,
    ) -> None:
        self._draw_panel(rect, outer_color=outer_color, inner_color=inner_color, inset=FRAME_INSET)
        if outline_color is not None:
            self._draw_outline(rect, outline_color, line_width=OUTLINE_WIDTH)
        if card_key is None:
            return

        card = CARD_DEFS[str(card_key)]
        scale = max(0.55, float(rect.height) / float(config.HAND_CARD_HEIGHT))
        left_pad = 12.0 * scale
        right_pad = 14.0 * scale
        cost_y = float(rect.top) + 14.0 * scale
        name_y = float(rect.top) + 26.0 * scale
        value_y = float(rect.bottom) - 22.0 * scale

        self._text.draw(
            str(int(card.cost)),
            x=float(rect.right) - right_pad,
            y=float(self.window_controller.to_arcade_y(cost_y)),
            color=COLOR_FOG_GRAY,
            font_size=max(8, int(round(10.0 * scale))),
            font_name=config.UI_FONT_NAME,
            anchor_x="right",
            anchor_y="center",
        )
        self._text.draw(
            self._card_label_text(card),
            x=float(rect.left) + left_pad,
            y=float(self.window_controller.to_arcade_y(name_y)),
            color=COLOR_LIGHT_NEUTRAL,
            font_size=max(11, int(round(16.0 * scale))),
            font_name=config.TITLE_FONT_NAME,
            anchor_x="left",
            anchor_y="center",
        )
        self._text.draw(
            self._card_value_text(card),
            x=float(rect.right) - right_pad,
            y=float(self.window_controller.to_arcade_y(value_y)),
            color=COLOR_LIGHT_NEUTRAL,
            font_size=max(16, int(round(28.0 * scale))),
            font_name=config.TITLE_FONT_NAME,
            anchor_x="right",
            anchor_y="center",
        )

    def _unit_slot_rect(self, lane_rect: Rect, player: int, slot: int) -> Rect:
        total_width = 2.0 * float(config.UNIT_CARD_WIDTH) + float(config.UNIT_CARD_GAP)
        start_left = float(lane_rect.left) + (float(lane_rect.width) - total_width) * 0.5
        if int(player) == int(PLAYER_P2):
            top = float(lane_rect.top) + 54.0
        else:
            top = float(lane_rect.bottom) - float(config.UNIT_CARD_HEIGHT) - 54.0
        return Rect(
            left=float(start_left) + float(slot) * (float(config.UNIT_CARD_WIDTH) + float(config.UNIT_CARD_GAP)),
            top=float(top),
            width=float(config.UNIT_CARD_WIDTH),
            height=float(config.UNIT_CARD_HEIGHT),
        )

    def _draw_lane_unit_slots(self, lane: int, player: int, lane_rect: Rect) -> None:
        units = list(self._lane_units[int(lane)][int(player)])
        for slot in range(int(config.MAX_UNITS_PER_LANE)):
            rect = self._unit_slot_rect(lane_rect, int(player), int(slot))
            if int(slot) >= len(units):
                self._draw_card_face(
                    rect,
                    card_key=None,
                    outer_color=LANE_PLACEHOLDER_OUTER,
                    inner_color=LANE_PLACEHOLDER_INNER,
                )
                continue
            self._draw_card_face(
                rect,
                card_key=str(units[int(slot)]),
                outer_color=PLAYER_OUTERS[int(player)],
                inner_color=PLAYER_INNERS[int(player)],
            )

    def _draw_lane(self, lane: int, rect: Rect) -> None:
        outer_color, inner_color = self._lane_panel_colors(int(lane))
        self._draw_panel(
            rect,
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(config.LANE_INSET),
        )
        if self._selected_action_for_lane(int(lane)) is not None:
            self._draw_outline(rect, SELECTED_OUTLINE, line_width=OUTLINE_WIDTH)
        elif self._hover_lane == int(lane):
            self._draw_outline(rect, ACTIVE_OUTLINE, line_width=OUTLINE_WIDTH)

        self._draw_lane_unit_slots(int(lane), int(PLAYER_P2), rect)
        self._draw_banner_marker(int(lane), int(PLAYER_P2), rect)
        self._draw_attack_markers(int(lane), int(PLAYER_P2), rect)
        self._draw_lane_unit_slots(int(lane), int(PLAYER_P1), rect)
        self._draw_banner_marker(int(lane), int(PLAYER_P1), rect)
        self._draw_attack_markers(int(lane), int(PLAYER_P1), rect)

    def _slot_has_any_valid_lane(self, slot: int) -> bool:
        mask = self.get_action_mask()
        start = int(slot) * int(config.NUM_LANES)
        end = start + int(config.NUM_LANES)
        return bool(np.any(mask[start:end]))

    def _draw_hand_card(self, slot: int, rect: Rect) -> None:
        actor = int(PLAYER_P1)
        card_key = self._card_in_slot(int(actor), int(slot))
        outer_color = PLAYER_OUTERS[int(actor)]
        inner_color = PLAYER_INNERS[int(actor)]
        if card_key is None:
            outer_color = LANE_PLACEHOLDER_OUTER
            inner_color = LANE_PLACEHOLDER_INNER
        elif not self._slot_has_any_valid_lane(int(slot)):
            outer_color = CARD_DISABLED_OUTER
            inner_color = CARD_DISABLED_INNER
        outline_color: tuple[int, int, int] | None = None
        if self._selected_hand_slot == int(slot):
            outline_color = SELECTED_OUTLINE
        elif self._hover_hand_slot == int(slot):
            outline_color = ACTIVE_OUTLINE
        self._draw_card_face(
            rect,
            card_key=card_key,
            outer_color=outer_color,
            inner_color=inner_color,
            outline_color=outline_color,
        )

    def _draw_active_hand(self) -> None:
        for slot, rect in enumerate(self._hand_rects):
            self._draw_hand_card(int(slot), rect)

    def _playing_state_text(self) -> str:
        lead = int(self._turn_lead_player(int(self.turn)))
        follow = int(self._other_player(int(lead)))
        lead_label = self._turn_player_name(int(lead))
        follow_label = self._turn_player_name(int(follow))
        if int(self.current_player) == int(lead):
            return f"[{lead_label}]/{follow_label}"
        return f"{lead_label}/[{follow_label}]"

    def _status_entries(self) -> list[tuple[str, object]]:
        if self._done:
            result = "Draw" if self._winner is None else f"{self._turn_player_name(int(self._winner))} wins"
            return [
                ("Turn", f"{int(self.turn)}/{int(config.MAX_TURNS)}"),
                ("Result", result),
                ("Score", f"{int(self._scores[int(PLAYER_P1)])}/{int(self._scores[int(PLAYER_P2)])}"),
                ("Restart", "Enter"),
            ]
        if self._turn_pause_active:
            return [
                ("Turn", f"{int(self.turn)}/{int(config.MAX_TURNS)}"),
                ("Status", "Resolving turn..."),
                ("Score", f"{int(self._scores[int(PLAYER_P1)])}/{int(self._scores[int(PLAYER_P2)])}"),
            ]
        turn_cap = min(int(config.MAX_TURNS), max(1, int(self.turn)))
        return [
            ("Turn", f"{int(self.turn)}/{int(config.MAX_TURNS)}"),
            ("Playing", self._playing_state_text()),
            ("Energy", f"{int(self._energy[int(self.current_player)])}/{turn_cap}"),
            ("Score", f"{int(self._scores[int(PLAYER_P1)])}/{int(self._scores[int(PLAYER_P2)])}"),
        ]

    def _draw_bottom_bar(self) -> None:
        layout = draw_status_bar(
            width=float(config.SCREEN_WIDTH),
            bottom_bar_height=float(config.BB_HEIGHT),
            tile_size=float(DEFAULT_TILE_SIZE),
            cell_inset=float(DEFAULT_CELL_INSET),
            left_panel_width=max(0.0, float(config.SCREEN_WIDTH) - 16.0),
            include_clock=False,
            text_cache=self._text,
            left_text_entries=self._status_entries(),
            text_color=COLOR_LIGHT_NEUTRAL,
        )

    def render(self) -> None:
        if self.window_controller.window is None:
            return
        if self.mode == "human":
            self._update_hover_state()
        self.window_controller.clear(WORLD_BG)
        arcade.draw_lbwh_rectangle_filled(
            0.0,
            float(config.BB_HEIGHT),
            float(config.WORLD_WIDTH),
            float(config.WORLD_HEIGHT),
            WORLD_BG,
        )
        for lane, rect in enumerate(self._lane_rects):
            self._draw_lane(int(lane), rect)
        for lane, rect in enumerate(self._score_track_rects[int(PLAYER_P2)]):
            self._draw_score_track(int(PLAYER_P2), int(lane), rect)
        for lane, rect in enumerate(self._score_track_rects[int(PLAYER_P1)]):
            self._draw_score_track(int(PLAYER_P1), int(lane), rect)
        self._draw_active_hand()
        self._draw_bottom_bar()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
