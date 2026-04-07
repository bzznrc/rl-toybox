"""Wave-based tower defense environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_CORAL,
    COLOR_AQUA,
    COLOR_BLUE,
    COLOR_BRICK_RED,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_FOREST_GREEN,
    COLOR_LEAF_GREEN,
    COLOR_LIGHT_NEUTRAL,
    COLOR_NAVY,
    COLOR_OCHRE,
    COLOR_SAND,
    COLOR_SLATE_GRAY,
    DEFAULT_CELL_INSET,
    DEFAULT_TILE_SIZE,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_unit, ordered_feature_vector
from core.primitives import (
    draw_facing_indicator,
    draw_two_tone_tile,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController, TextCache
from core.utils import resolve_play_level
from games.tower import config


validate_curriculum_level_settings(
    min_level=config.MIN_LEVEL,
    max_level=config.MAX_LEVEL,
    level_settings=config.LEVEL_SETTINGS,
)


SIDE_LEFT = "left"
SIDE_RIGHT = "right"
ENTRY_BOTH = "both"
TOWER_KIND_TO_ID = {"arrow": 1.0, "cannon": 2.0, "tesla": 3.0}
SELL_REFUND_BY_LEVEL = {1: 0.90, 2: 0.75, 3: 0.60}
SPAWN_GAPS = {"swarm": 9, "armored": 15, "flying": 11}

WORLD_BG_TOP = (36, 39, 45)
WORLD_BG_BOTTOM = COLOR_DARK_NEUTRAL
GROUND_PATH_OUTER = COLOR_FOG_GRAY
GROUND_PATH_INNER = COLOR_SLATE_GRAY
ENTRY_OUTLINE = COLOR_CORAL
ENTRY_FILL = COLOR_BRICK_RED
EXIT_OUTLINE = COLOR_LEAF_GREEN
EXIT_FILL = COLOR_FOREST_GREEN
SLOT_OUTLINE = COLOR_FOG_GRAY
SLOT_FILL = COLOR_SLATE_GRAY
EMPTY_SLOT_FILL = WORLD_BG_TOP
SELECTED_SLOT_COLOR = COLOR_SAND
TILE_SIZE = float(DEFAULT_TILE_SIZE)
CELL_INSET = float(DEFAULT_CELL_INSET)
SLOT_SIZE = float(TILE_SIZE * 2.0)
SLOT_INSET = float(CELL_INSET)
ENEMY_SIZE = float(TILE_SIZE)
ENEMY_INSET = float(CELL_INSET)
ENTRY_SIZE = float(TILE_SIZE * 2.0)
TRACK_WIDTH_CELLS = 2


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def _lane_point(left_col: int, top_row: int) -> tuple[float, float]:
    return (float(left_col + 1) * TILE_SIZE, float(top_row + 1) * TILE_SIZE)


def _block_center(left_col: int, top_row: int) -> tuple[float, float]:
    return (float(left_col + 1) * TILE_SIZE, float(top_row + 1) * TILE_SIZE)


def _block_rect(left_col: int, top_row: int, width_cells: int, height_cells: int) -> tuple[float, float, float, float]:
    return (
        float(left_col) * TILE_SIZE,
        float(top_row) * TILE_SIZE,
        float(width_cells) * TILE_SIZE,
        float(height_cells) * TILE_SIZE,
    )


def _rect_cells(rects: tuple[tuple[int, int, int, int], ...]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for left_col, top_row, width_cells, height_cells in rects:
        for col in range(int(left_col), int(left_col) + int(width_cells)):
            for row in range(int(top_row), int(top_row) + int(height_cells)):
                cells.add((int(col), int(row)))
    return cells


class LanePath:
    """Polyline path used by ground and flying enemies."""

    def __init__(self, points: tuple[tuple[float, float], ...]) -> None:
        if len(points) < 2:
            raise ValueError("LanePath requires at least two points.")
        self.points = tuple((float(x), float(y)) for x, y in points)
        segment_lengths: list[float] = []
        cumulative_lengths: list[float] = [0.0]
        total_length = 0.0
        for start, end in zip(self.points[:-1], self.points[1:]):
            segment_length = _distance(start, end)
            segment_lengths.append(float(segment_length))
            total_length += float(segment_length)
            cumulative_lengths.append(float(total_length))
        self.segment_lengths = tuple(segment_lengths)
        self.cumulative_lengths = tuple(cumulative_lengths)
        self.total_length = float(max(1.0, total_length))

    def position_at(self, distance_along: float) -> tuple[float, float]:
        distance_value = float(max(0.0, min(float(distance_along), self.total_length)))
        if distance_value <= 0.0:
            return self.points[0]
        if distance_value >= self.total_length:
            return self.points[-1]

        for index, segment_length in enumerate(self.segment_lengths):
            start_len = float(self.cumulative_lengths[index])
            end_len = float(self.cumulative_lengths[index + 1])
            if distance_value > end_len and index < len(self.segment_lengths) - 1:
                continue
            if segment_length <= 1e-6:
                return self.points[index + 1]
            progress = (distance_value - start_len) / float(segment_length)
            x0, y0 = self.points[index]
            x1, y1 = self.points[index + 1]
            return (
                float(x0 + (x1 - x0) * progress),
                float(y0 + (y1 - y0) * progress),
            )
        return self.points[-1]


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: int
    name: str
    ground_paths: dict[str, LanePath]
    flying_paths: dict[str, LanePath]
    ground_rects: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class WaveSpec:
    entry_mode: str
    count_swarm: int
    count_armored: int
    count_flying: int

    def count_for(self, enemy_kind: str) -> int:
        if enemy_kind == "swarm":
            return int(self.count_swarm)
        if enemy_kind == "armored":
            return int(self.count_armored)
        if enemy_kind == "flying":
            return int(self.count_flying)
        raise KeyError(f"Unknown enemy kind '{enemy_kind}'.")

    @property
    def total_count(self) -> int:
        return int(self.count_swarm + self.count_armored + self.count_flying)


@dataclass(frozen=True)
class TowerStats:
    damage: float
    cooldown_ticks: int
    attack_range: float
    armor_pierce: float
    splash_radius: float = 0.0
    chain_count: int = 1
    chain_range: float = 0.0


@dataclass(frozen=True)
class TowerProfile:
    build_cost: int
    upgrade_costs: tuple[int, int]
    level_stats: tuple[TowerStats, TowerStats, TowerStats]

    def stats_for_level(self, level: int) -> TowerStats:
        level_value = max(1, min(3, int(level)))
        return self.level_stats[level_value - 1]

    def upgrade_cost_for_level(self, level: int) -> int:
        level_value = max(1, min(3, int(level)))
        if level_value >= 3:
            raise ValueError("Level-3 towers cannot upgrade further.")
        return int(self.upgrade_costs[level_value - 1])


@dataclass(frozen=True)
class EnemyProfile:
    max_hp: float
    speed: float
    armor: float
    bounty: int
    radius: float


@dataclass
class TowerState:
    kind: str
    level: int
    cooldown_ticks: int
    total_spent: int

    @property
    def profile(self) -> TowerProfile:
        return TOWER_PROFILES[str(self.kind)]

    @property
    def stats(self) -> TowerStats:
        return self.profile.stats_for_level(int(self.level))


@dataclass
class EnemyState:
    kind: str
    side: str
    path: LanePath
    hp: float
    speed: float
    armor: float
    bounty: int
    radius: float
    distance_along: float = 0.0
    alive: bool = True
    leaked: bool = False

    @property
    def position(self) -> tuple[float, float]:
        return self.path.position_at(float(self.distance_along))

    @property
    def progress_norm(self) -> float:
        return float(self.distance_along) / float(max(1e-6, self.path.total_length))


@dataclass
class AttackEffect:
    kind: str
    points: tuple[tuple[float, float], ...]
    ttl: int
    radius: float = 0.0


SLOT_CELLS = {
    "left": (6, 6),
    "upper": (16, 4),
    "mid": (20, 9),
    "lower": (16, 14),
    "right": (34, 6),
}
SLOT_POSITIONS = {
    slot_name: _block_center(*cell_top_left)
    for slot_name, cell_top_left in SLOT_CELLS.items()
}
ENTRY_CELLS = {
    SIDE_LEFT: (0, 8),
    SIDE_RIGHT: (40, 8),
}
ENTRY_POSITIONS = {
    side: _block_center(*cell_top_left)
    for side, cell_top_left in ENTRY_CELLS.items()
}
EXIT_CELL = (18, 19)
EXIT_POSITION = _block_center(*EXIT_CELL)

LAYOUTS = (
    LayoutSpec(
        layout_id=0,
        name="High Merge",
        ground_paths={
            SIDE_LEFT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_LEFT],
                    _lane_point(8, 8),
                    _lane_point(8, 6),
                    _lane_point(18, 6),
                    EXIT_POSITION,
                )
            ),
            SIDE_RIGHT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_RIGHT],
                    _lane_point(32, 8),
                    _lane_point(32, 6),
                    _lane_point(18, 6),
                    EXIT_POSITION,
                )
            ),
        },
        flying_paths={
            SIDE_LEFT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_LEFT],
                    _lane_point(8, 8),
                    _lane_point(8, 6),
                    _lane_point(18, 6),
                    EXIT_POSITION,
                )
            ),
            SIDE_RIGHT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_RIGHT],
                    _lane_point(32, 8),
                    _lane_point(32, 6),
                    _lane_point(18, 6),
                    EXIT_POSITION,
                )
            ),
        },
        ground_rects=(
            (0, 8, 10, 2),
            (8, 6, 2, 4),
            (8, 6, 12, 2),
            (18, 6, 2, 15),
            (32, 8, 10, 2),
            (32, 6, 2, 4),
            (20, 6, 14, 2),
        ),
    ),
    LayoutSpec(
        layout_id=1,
        name="Split Rise",
        ground_paths={
            SIDE_LEFT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_LEFT],
                    _lane_point(8, 8),
                    _lane_point(8, 12),
                    _lane_point(18, 12),
                    EXIT_POSITION,
                )
            ),
            SIDE_RIGHT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_RIGHT],
                    _lane_point(32, 8),
                    _lane_point(32, 4),
                    _lane_point(18, 4),
                    EXIT_POSITION,
                )
            ),
        },
        flying_paths={
            SIDE_LEFT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_LEFT],
                    _lane_point(8, 8),
                    _lane_point(8, 12),
                    _lane_point(18, 12),
                    EXIT_POSITION,
                )
            ),
            SIDE_RIGHT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_RIGHT],
                    _lane_point(32, 8),
                    _lane_point(32, 4),
                    _lane_point(18, 4),
                    EXIT_POSITION,
                )
            ),
        },
        ground_rects=(
            (0, 8, 10, 2),
            (8, 8, 2, 6),
            (8, 12, 12, 2),
            (18, 4, 2, 17),
            (32, 8, 10, 2),
            (32, 4, 2, 6),
            (20, 4, 14, 2),
        ),
    ),
    LayoutSpec(
        layout_id=2,
        name="Split Cross",
        ground_paths={
            SIDE_LEFT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_LEFT],
                    _lane_point(8, 8),
                    _lane_point(8, 6),
                    _lane_point(18, 6),
                    EXIT_POSITION,
                )
            ),
            SIDE_RIGHT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_RIGHT],
                    _lane_point(32, 8),
                    _lane_point(32, 12),
                    _lane_point(18, 12),
                    EXIT_POSITION,
                )
            ),
        },
        flying_paths={
            SIDE_LEFT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_LEFT],
                    _lane_point(8, 8),
                    _lane_point(8, 6),
                    _lane_point(18, 6),
                    EXIT_POSITION,
                )
            ),
            SIDE_RIGHT: LanePath(
                (
                    ENTRY_POSITIONS[SIDE_RIGHT],
                    _lane_point(32, 8),
                    _lane_point(32, 12),
                    _lane_point(18, 12),
                    EXIT_POSITION,
                )
            ),
        },
        ground_rects=(
            (0, 8, 10, 2),
            (8, 6, 2, 4),
            (8, 6, 12, 2),
            (18, 6, 2, 15),
            (32, 8, 10, 2),
            (32, 8, 2, 6),
            (20, 12, 14, 2),
        ),
    ),
)

TOWER_PROFILES = {
    "arrow": TowerProfile(
        build_cost=7,
        upgrade_costs=(6, 9),
        level_stats=(
            TowerStats(damage=1.10, cooldown_ticks=17, attack_range=150.0, armor_pierce=0.10),
            TowerStats(damage=1.35, cooldown_ticks=15, attack_range=160.0, armor_pierce=0.15),
            TowerStats(damage=1.70, cooldown_ticks=13, attack_range=170.0, armor_pierce=0.20),
        ),
    ),
    "cannon": TowerProfile(
        build_cost=8,
        upgrade_costs=(6, 10),
        level_stats=(
            TowerStats(damage=2.90, cooldown_ticks=31, attack_range=132.0, armor_pierce=0.90, splash_radius=34.0),
            TowerStats(damage=3.90, cooldown_ticks=28, attack_range=142.0, armor_pierce=1.20, splash_radius=42.0),
            TowerStats(damage=5.20, cooldown_ticks=24, attack_range=152.0, armor_pierce=1.55, splash_radius=50.0),
        ),
    ),
    "tesla": TowerProfile(
        build_cost=7,
        upgrade_costs=(6, 9),
        level_stats=(
            TowerStats(damage=0.85, cooldown_ticks=20, attack_range=142.0, armor_pierce=0.05, chain_count=2, chain_range=56.0),
            TowerStats(damage=1.00, cooldown_ticks=18, attack_range=152.0, armor_pierce=0.10, chain_count=3, chain_range=62.0),
            TowerStats(damage=1.20, cooldown_ticks=16, attack_range=162.0, armor_pierce=0.15, chain_count=4, chain_range=68.0),
        ),
    ),
}

TOWER_DAMAGE_MULTIPLIERS = {
    "arrow": {"swarm": 0.95, "armored": 0.45, "flying": 1.75},
    "cannon": {"swarm": 0.70, "armored": 1.55, "flying": 0.0},
    "tesla": {"swarm": 1.40, "armored": 0.80, "flying": 0.45},
}

ENEMY_PROFILES = {
    "swarm": EnemyProfile(max_hp=2.20, speed=3.20, armor=0.00, bounty=1, radius=8.0),
    "armored": EnemyProfile(max_hp=9.00, speed=1.55, armor=0.75, bounty=2, radius=10.0),
    "flying": EnemyProfile(max_hp=3.00, speed=3.45, armor=0.10, bounty=1, radius=9.0),
}

WAVE_TEMPLATES = (
    WaveSpec(entry_mode=SIDE_LEFT, count_swarm=6, count_armored=0, count_flying=0),
    WaveSpec(entry_mode=SIDE_RIGHT, count_swarm=3, count_armored=0, count_flying=4),
    WaveSpec(entry_mode=ENTRY_BOTH, count_swarm=7, count_armored=2, count_flying=0),
    WaveSpec(entry_mode=SIDE_LEFT, count_swarm=2, count_armored=3, count_flying=2),
    WaveSpec(entry_mode=SIDE_RIGHT, count_swarm=6, count_armored=2, count_flying=3),
    WaveSpec(entry_mode=ENTRY_BOTH, count_swarm=8, count_armored=4, count_flying=2),
    WaveSpec(entry_mode=SIDE_LEFT, count_swarm=4, count_armored=6, count_flying=4),
    WaveSpec(entry_mode=ENTRY_BOTH, count_swarm=10, count_armored=5, count_flying=5),
)

TOWER_COLORS = {
    "arrow": (COLOR_AQUA, COLOR_DEEP_TEAL),
    "cannon": (COLOR_SAND, COLOR_OCHRE),
    "tesla": (COLOR_BLUE, COLOR_NAVY),
}
ENEMY_COLORS = {
    "swarm": (COLOR_BLUE, COLOR_NAVY),
    "armored": (COLOR_SAND, COLOR_OCHRE),
    "flying": (COLOR_AQUA, COLOR_DEEP_TEAL),
}


class TowerEnv(Env):
    """Tiny wave-based tower-defense env with masked discrete actions."""

    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)
    REWARD_COMPONENT_ORDER = tuple(config.REWARD_COMPONENT_ORDER)

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)
        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            config.WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=self.mode == "human",
            vsync=False,
        )
        self._text_cache = TextCache(max_entries=512)

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
        self._last_outcome = ""
        self._level_settings = dict(config.LEVEL_SETTINGS[int(self._current_level)])
        self._episode_counter = 0

        self.layout: LayoutSpec = LAYOUTS[0]
        self.wave_plan: list[WaveSpec] = []
        self.wave_index = 0
        self.gold = 0
        self.lives = 0
        self.decision_budget = 0
        self.actions_remaining = 0
        self.slot_towers: dict[str, TowerState | None] = {slot_name: None for slot_name in config.SLOT_NAMES}
        self._selected_slot_index = 0
        self._selected_build_kind = "arrow"
        self._wave_in_progress = False
        self._active_enemies: list[EnemyState] = []
        self._attack_effects: list[AttackEffect] = []
        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)

        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _apply_level_settings(self, level: int) -> None:
        self._current_level = int(level)
        self._level_settings = dict(config.LEVEL_SETTINGS[int(self._current_level)])
        self.decision_budget = int(self._level_settings["decision_budget"])

    def _episode_seed(self) -> int:
        return int(config.BASE_SEED + self._episode_counter * 7919 + self._current_level * 101)

    @staticmethod
    def _empty_reward_breakdown() -> dict[str, float]:
        return {key: 0.0 for key in config.REWARD_COMPONENT_ORDER}

    @staticmethod
    def _safe_action(action: object) -> int:
        try:
            return int(action)
        except (TypeError, ValueError):
            return 0

    def _current_wave(self) -> WaveSpec | None:
        if 0 <= int(self.wave_index) < len(self.wave_plan):
            return self.wave_plan[int(self.wave_index)]
        return None

    def _build_wave_plan(self, rng: random.Random) -> list[WaveSpec]:
        wave_count = int(self._level_settings["num_waves"])
        wave_scale = float(self._level_settings["wave_scale"])
        plan: list[WaveSpec] = []
        for wave_idx in range(wave_count):
            base_wave = WAVE_TEMPLATES[int(wave_idx)]
            entry_mode = str(base_wave.entry_mode)
            if entry_mode in {SIDE_LEFT, SIDE_RIGHT} and rng.random() < 0.5:
                entry_mode = SIDE_RIGHT if entry_mode == SIDE_LEFT else SIDE_LEFT

            growth = 1.0 + 0.05 * float(wave_idx)
            counts: dict[str, int] = {}
            for kind in config.ENEMY_KINDS:
                base_count = int(base_wave.count_for(kind))
                if base_count <= 0:
                    counts[kind] = 0
                    continue
                scaled = float(base_count) * float(wave_scale) * float(growth)
                jitter = rng.choice((-1, 0, 0, 1))
                final_count = int(round(scaled)) + int(jitter)
                counts[kind] = max(1, int(final_count))

            plan.append(
                WaveSpec(
                    entry_mode=entry_mode,
                    count_swarm=int(counts["swarm"]),
                    count_armored=int(counts["armored"]),
                    count_flying=int(counts["flying"]),
                )
            )
        return plan

    def _tower_kind_value(self, tower_state: TowerState | None) -> float:
        if tower_state is None:
            return 0.0
        return float(TOWER_KIND_TO_ID.get(str(tower_state.kind), 0.0))

    @staticmethod
    def _tower_level_norm(tower_state: TowerState | None) -> float:
        if tower_state is None:
            return 0.0
        return clip_unit(float(tower_state.level) / 3.0)

    def _wave_number_norm(self) -> float:
        if not self.wave_plan:
            return 0.0
        next_wave_number = min(len(self.wave_plan), int(self.wave_index) + 1)
        if self._last_outcome:
            next_wave_number = len(self.wave_plan)
        return clip_unit(float(next_wave_number) / float(max(1, len(self.wave_plan))))

    def _build_observation(self) -> np.ndarray:
        wave = self._current_wave()
        feature_values = {
            "run_gold_norm": clip_unit(float(self.gold) / float(config.MAX_GOLD_NORMALIZER)),
            "run_lives_norm": clip_unit(float(self.lives) / float(config.MAX_LIVES_NORMALIZER)),
            "run_wave_norm": float(self._wave_number_norm()),
            "run_actions_left_norm": clip_unit(float(self.actions_remaining) / float(max(1, self.decision_budget))),
            "wave_entry_left": 1.0 if wave is not None and wave.entry_mode in {SIDE_LEFT, ENTRY_BOTH} else 0.0,
            "wave_entry_right": 1.0 if wave is not None and wave.entry_mode in {SIDE_RIGHT, ENTRY_BOTH} else 0.0,
            "wave_count_swarm_norm": clip_unit(
                float(0 if wave is None else wave.count_swarm) / float(config.MAX_WAVE_COUNT_NORMALIZER)
            ),
            "wave_count_armored_norm": clip_unit(
                float(0 if wave is None else wave.count_armored) / float(config.MAX_WAVE_COUNT_NORMALIZER)
            ),
            "wave_count_flying_norm": clip_unit(
                float(0 if wave is None else wave.count_flying) / float(config.MAX_WAVE_COUNT_NORMALIZER)
            ),
            "map_layout_id_norm": (
                0.0
                if len(LAYOUTS) <= 1
                else clip_unit(float(self.layout.layout_id) / float(len(LAYOUTS) - 1))
            ),
        }
        for slot_name in config.SLOT_NAMES:
            tower_state = self.slot_towers[str(slot_name)]
            feature_values[f"slot_{slot_name}_tower_kind"] = float(self._tower_kind_value(tower_state))
            feature_values[f"slot_{slot_name}_tower_level_norm"] = float(self._tower_level_norm(tower_state))

        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Tower observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        rng = random.Random(self._episode_seed())
        self.layout = LAYOUTS[int(rng.randrange(len(LAYOUTS)))]
        self.wave_plan = self._build_wave_plan(rng)
        self.wave_index = 0
        self.gold = int(self._level_settings["start_gold"])
        self.lives = int(self._level_settings["start_lives"])
        self.actions_remaining = int(self.decision_budget)
        self.slot_towers = {slot_name: None for slot_name in config.SLOT_NAMES}
        self._selected_slot_index = 0
        self._selected_build_kind = "arrow"
        self._wave_in_progress = False
        self._active_enemies = []
        self._attack_effects = []
        self._last_outcome = ""
        self._episode_reward_components.reset()
        self._episode_counter += 1
        self._last_obs = self._build_observation()
        return np.asarray(self._last_obs, dtype=np.float32)

    def _build_action_index(self, tower_kind: str, slot_name: str) -> int:
        tower_index = int(config.TOWER_KINDS.index(str(tower_kind)))
        slot_index = int(config.SLOT_NAMES.index(str(slot_name)))
        return 1 + tower_index * len(config.SLOT_NAMES) + slot_index

    def _upgrade_action_index(self, slot_name: str) -> int:
        return 1 + len(config.TOWER_KINDS) * len(config.SLOT_NAMES) + int(config.SLOT_NAMES.index(str(slot_name)))

    def _sell_action_index(self, slot_name: str) -> int:
        return self._upgrade_action_index(slot_name) + len(config.SLOT_NAMES)

    def _decode_action(self, action_idx: int) -> tuple[str, str | None, str | None]:
        if int(action_idx) == 0:
            return "start_wave", None, None

        build_span = len(config.TOWER_KINDS) * len(config.SLOT_NAMES)
        build_index = int(action_idx) - 1
        if 0 <= build_index < build_span:
            tower_kind = str(config.TOWER_KINDS[build_index // len(config.SLOT_NAMES)])
            slot_name = str(config.SLOT_NAMES[build_index % len(config.SLOT_NAMES)])
            return "build", tower_kind, slot_name

        upgrade_start = 1 + build_span
        if upgrade_start <= int(action_idx) < upgrade_start + len(config.SLOT_NAMES):
            slot_name = str(config.SLOT_NAMES[int(action_idx) - upgrade_start])
            return "upgrade", None, slot_name

        sell_start = upgrade_start + len(config.SLOT_NAMES)
        slot_name = str(config.SLOT_NAMES[int(action_idx) - sell_start])
        return "sell", None, slot_name

    def get_action_mask(self, _obs: object | None = None) -> np.ndarray:
        mask = np.zeros((self.ACT_DIM,), dtype=np.bool_)
        if self._last_outcome or self._wave_in_progress:
            return mask
        if self._current_wave() is None:
            return mask

        mask[0] = True
        if int(self.actions_remaining) <= 0:
            return mask

        for slot_name in config.SLOT_NAMES:
            tower_state = self.slot_towers[str(slot_name)]
            if tower_state is None:
                for tower_kind in config.TOWER_KINDS:
                    build_cost = int(TOWER_PROFILES[str(tower_kind)].build_cost)
                    if int(self.gold) >= build_cost:
                        mask[self._build_action_index(str(tower_kind), str(slot_name))] = True
                continue

            if int(tower_state.level) < 3:
                upgrade_cost = int(tower_state.profile.upgrade_cost_for_level(int(tower_state.level)))
                if int(self.gold) >= upgrade_cost:
                    mask[self._upgrade_action_index(str(slot_name))] = True
            mask[self._sell_action_index(str(slot_name))] = True

        return mask

    def _coerce_action(self, action: object) -> int:
        action_idx = self._safe_action(action)
        mask = self.get_action_mask()
        if 0 <= int(action_idx) < self.ACT_DIM and bool(mask[int(action_idx)]):
            return int(action_idx)
        valid_actions = np.flatnonzero(mask)
        if valid_actions.size <= 0:
            return 0
        return int(valid_actions[0])

    def _selected_slot_name(self) -> str:
        return str(config.SLOT_NAMES[int(self._selected_slot_index)])

    def _sell_value(self, slot_name: str) -> int:
        tower_state = self.slot_towers[str(slot_name)]
        if tower_state is None:
            return 0
        refund_ratio = float(SELL_REFUND_BY_LEVEL.get(int(tower_state.level), 0.0))
        return int(round(float(tower_state.total_spent) * float(refund_ratio)))

    def _slot_name_at(self, mouse_x: float, mouse_y_arcade: float) -> str | None:
        mouse_y = float(self.window_controller.to_top_left_y(float(mouse_y_arcade)))
        for slot_name, (center_x, center_y) in SLOT_POSITIONS.items():
            if (
                abs(float(mouse_x) - float(center_x)) <= float(SLOT_SIZE) * 0.5
                and abs(float(mouse_y) - float(center_y)) <= float(SLOT_SIZE) * 0.5
            ):
                return str(slot_name)
        return None

    def _human_action(self) -> int | None:
        for mouse_press in self.window_controller.consume_mouse_presses():
            slot_name = self._slot_name_at(float(mouse_press.x), float(mouse_press.y))
            if slot_name is None:
                continue
            self._selected_slot_index = int(config.SLOT_NAMES.index(str(slot_name)))
            if int(mouse_press.button) == int(arcade.MOUSE_BUTTON_RIGHT):
                return int(self._sell_action_index(str(slot_name)))

        for key in self.window_controller.consume_key_presses():
            if key == arcade.key.LEFT:
                self._selected_slot_index = (int(self._selected_slot_index) - 1) % len(config.SLOT_NAMES)
                continue
            if key == arcade.key.RIGHT:
                self._selected_slot_index = (int(self._selected_slot_index) + 1) % len(config.SLOT_NAMES)
                continue
            if key in (arcade.key.KEY_1, arcade.key.NUM_1):
                self._selected_slot_index = 0
                continue
            if key in (arcade.key.KEY_2, arcade.key.NUM_2):
                self._selected_slot_index = 1
                continue
            if key in (arcade.key.KEY_3, arcade.key.NUM_3):
                self._selected_slot_index = 2
                continue
            if key in (arcade.key.KEY_4, arcade.key.NUM_4):
                self._selected_slot_index = 3
                continue
            if key in (arcade.key.KEY_5, arcade.key.NUM_5):
                self._selected_slot_index = 4
                continue
            if key == arcade.key.A:
                self._selected_build_kind = "arrow"
                continue
            if key == arcade.key.C:
                self._selected_build_kind = "cannon"
                continue
            if key == arcade.key.T:
                self._selected_build_kind = "tesla"
                continue
            if key == arcade.key.U:
                return int(self._upgrade_action_index(self._selected_slot_name()))
            if key in (arcade.key.DELETE, arcade.key.BACKSPACE):
                return int(self._sell_action_index(self._selected_slot_name()))
            if key == arcade.key.SPACE:
                return int(self._build_action_index(self._selected_build_kind, self._selected_slot_name()))
            if key == arcade.key.ENTER:
                return 0
        return None

    def _spawn_enemy(self, side: str, enemy_kind: str) -> EnemyState:
        profile = ENEMY_PROFILES[str(enemy_kind)]
        path = self.layout.ground_paths[str(side)]
        return EnemyState(
            kind=str(enemy_kind),
            side=str(side),
            path=path,
            hp=float(profile.max_hp),
            speed=float(profile.speed),
            armor=float(profile.armor),
            bounty=int(profile.bounty),
            radius=float(profile.radius),
        )

    def _split_wave_counts(self, wave: WaveSpec, enemy_kind: str) -> dict[str, int]:
        total = int(wave.count_for(str(enemy_kind)))
        if wave.entry_mode == SIDE_LEFT:
            return {SIDE_LEFT: total, SIDE_RIGHT: 0}
        if wave.entry_mode == SIDE_RIGHT:
            return {SIDE_LEFT: 0, SIDE_RIGHT: total}

        left_count = total // 2
        right_count = total - left_count
        tie_break = (int(self.wave_index) + int(self.layout.layout_id) + int(config.ENEMY_KINDS.index(str(enemy_kind)))) % 2
        if tie_break == 1:
            left_count, right_count = right_count, left_count
        return {SIDE_LEFT: int(left_count), SIDE_RIGHT: int(right_count)}

    def _build_spawn_schedule(self, wave: WaveSpec) -> list[tuple[int, str, str]]:
        side_queues: dict[str, deque[str]] = {SIDE_LEFT: deque(), SIDE_RIGHT: deque()}
        for enemy_kind in ("swarm", "flying", "armored"):
            split_counts = self._split_wave_counts(wave, str(enemy_kind))
            for side in (SIDE_LEFT, SIDE_RIGHT):
                for _ in range(max(0, int(split_counts[str(side)]))):
                    side_queues[str(side)].append(str(enemy_kind))

        ordered_spawns: list[tuple[str, str]] = []
        while side_queues[SIDE_LEFT] or side_queues[SIDE_RIGHT]:
            if side_queues[SIDE_LEFT]:
                ordered_spawns.append((SIDE_LEFT, side_queues[SIDE_LEFT].popleft()))
            if side_queues[SIDE_RIGHT]:
                ordered_spawns.append((SIDE_RIGHT, side_queues[SIDE_RIGHT].popleft()))

        schedule: list[tuple[int, str, str]] = []
        tick_cursor = 0
        for side, enemy_kind in ordered_spawns:
            schedule.append((int(tick_cursor), str(side), str(enemy_kind)))
            tick_cursor += int(SPAWN_GAPS[str(enemy_kind)])
        return schedule

    @staticmethod
    def _enemy_priority(enemy: EnemyState, *, preferred_kind: str | None = None) -> tuple[float, float, float]:
        preferred = 1.0 if preferred_kind is not None and str(enemy.kind) == str(preferred_kind) else 0.0
        return (preferred, float(enemy.progress_norm), float(enemy.hp))

    def _in_range(self, origin: tuple[float, float], enemy: EnemyState, attack_range: float) -> bool:
        return _distance(origin, enemy.position) <= float(attack_range) + float(enemy.radius)

    def _damage_enemy(
        self,
        enemy: EnemyState,
        *,
        tower_kind: str,
        tower_stats: TowerStats,
        scale: float = 1.0,
    ) -> bool:
        multiplier = float(TOWER_DAMAGE_MULTIPLIERS[str(tower_kind)][str(enemy.kind)])
        if multiplier <= 0.0 or (not enemy.alive):
            return False
        raw_damage = float(tower_stats.damage) * float(multiplier) * float(scale)
        if raw_damage <= 0.0:
            return False
        mitigated_armor = max(0.0, float(enemy.armor) - float(tower_stats.armor_pierce))
        damage = max(0.05, float(raw_damage) - float(mitigated_armor))
        enemy.hp -= float(damage)
        if enemy.hp <= 0.0:
            enemy.alive = False
            return True
        return False

    def _attack_with_arrow(self, slot_name: str, tower_state: TowerState, enemies: list[EnemyState]) -> None:
        slot_pos = SLOT_POSITIONS[str(slot_name)]
        candidates = [
            enemy
            for enemy in enemies
            if enemy.alive and self._in_range(slot_pos, enemy, tower_state.stats.attack_range)
        ]
        if not candidates:
            return
        target = max(
            candidates,
            key=lambda enemy: self._enemy_priority(enemy, preferred_kind="flying"),
        )
        self._damage_enemy(target, tower_kind="arrow", tower_stats=tower_state.stats)
        tower_state.cooldown_ticks = int(tower_state.stats.cooldown_ticks)
        self._attack_effects.append(AttackEffect(kind="arrow", points=(slot_pos, target.position), ttl=4))

    def _attack_with_cannon(self, slot_name: str, tower_state: TowerState, enemies: list[EnemyState]) -> None:
        slot_pos = SLOT_POSITIONS[str(slot_name)]
        candidates = [
            enemy
            for enemy in enemies
            if enemy.alive
            and str(enemy.kind) != "flying"
            and self._in_range(slot_pos, enemy, tower_state.stats.attack_range)
        ]
        if not candidates:
            return
        target = max(
            candidates,
            key=lambda enemy: self._enemy_priority(enemy, preferred_kind="armored"),
        )
        blast_center = target.position
        for enemy in enemies:
            if not enemy.alive or str(enemy.kind) == "flying":
                continue
            distance_to_blast = _distance(blast_center, enemy.position)
            if distance_to_blast > float(tower_state.stats.splash_radius) + float(enemy.radius):
                continue
            splash_scale = 1.0 if enemy is target else 0.65
            self._damage_enemy(enemy, tower_kind="cannon", tower_stats=tower_state.stats, scale=float(splash_scale))
        tower_state.cooldown_ticks = int(tower_state.stats.cooldown_ticks)
        self._attack_effects.append(
            AttackEffect(
                kind="cannon",
                points=(slot_pos, blast_center),
                ttl=6,
                radius=float(tower_state.stats.splash_radius),
            )
        )

    def _attack_with_tesla(self, slot_name: str, tower_state: TowerState, enemies: list[EnemyState]) -> None:
        slot_pos = SLOT_POSITIONS[str(slot_name)]
        candidates = [
            enemy
            for enemy in enemies
            if enemy.alive and self._in_range(slot_pos, enemy, tower_state.stats.attack_range)
        ]
        if not candidates:
            return
        primary = max(
            candidates,
            key=lambda enemy: self._enemy_priority(enemy, preferred_kind="swarm"),
        )
        chained_targets = [primary]
        remaining = [enemy for enemy in candidates if enemy is not primary]
        anchor = primary
        while len(chained_targets) < int(tower_state.stats.chain_count) and remaining:
            chain_candidates = [
                enemy
                for enemy in remaining
                if _distance(anchor.position, enemy.position)
                <= float(tower_state.stats.chain_range) + float(enemy.radius)
            ]
            if not chain_candidates:
                break
            nxt = max(
                chain_candidates,
                key=lambda enemy: self._enemy_priority(enemy, preferred_kind="swarm"),
            )
            chained_targets.append(nxt)
            remaining.remove(nxt)
            anchor = nxt

        for chain_index, enemy in enumerate(chained_targets):
            chain_scale = max(0.55, 1.0 - 0.18 * float(chain_index))
            self._damage_enemy(enemy, tower_kind="tesla", tower_stats=tower_state.stats, scale=float(chain_scale))

        effect_points = [slot_pos]
        effect_points.extend(enemy.position for enemy in chained_targets)
        tower_state.cooldown_ticks = int(tower_state.stats.cooldown_ticks)
        self._attack_effects.append(AttackEffect(kind="tesla", points=tuple(effect_points), ttl=5))

    def _tick_towers(self, enemies: list[EnemyState]) -> None:
        for slot_name in config.SLOT_NAMES:
            tower_state = self.slot_towers[str(slot_name)]
            if tower_state is None:
                continue
            if int(tower_state.cooldown_ticks) > 0:
                tower_state.cooldown_ticks = max(0, int(tower_state.cooldown_ticks) - 1)
                continue
            if str(tower_state.kind) == "arrow":
                self._attack_with_arrow(str(slot_name), tower_state, enemies)
            elif str(tower_state.kind) == "cannon":
                self._attack_with_cannon(str(slot_name), tower_state, enemies)
            else:
                self._attack_with_tesla(str(slot_name), tower_state, enemies)

    def _collect_kills(self, enemies: list[EnemyState], reward_breakdown: dict[str, float]) -> None:
        for enemy in enemies:
            if enemy.alive or enemy.leaked:
                continue
            reward_breakdown["reward_progress_kill"] += float(config.REWARD_PROGRESS_KILL)
            self.gold += int(enemy.bounty)
            enemy.leaked = True

    def _move_enemies(self, enemies: list[EnemyState], reward_breakdown: dict[str, float]) -> None:
        for enemy in enemies:
            if not enemy.alive:
                continue
            enemy.distance_along += float(enemy.speed)
            if float(enemy.distance_along) >= float(enemy.path.total_length):
                enemy.alive = False
                enemy.leaked = True
                self.lives = max(0, int(self.lives) - int(config.ENEMY_LEAK_DAMAGE))
                reward_breakdown["reward_event_leak"] += float(config.REWARD_EVENT_LEAK)

    def _trim_effects(self) -> None:
        next_effects: list[AttackEffect] = []
        for effect in self._attack_effects:
            effect.ttl = max(0, int(effect.ttl) - 1)
            if int(effect.ttl) > 0:
                next_effects.append(effect)
        self._attack_effects = next_effects

    def _simulate_wave(self) -> dict[str, float]:
        reward_breakdown = self._empty_reward_breakdown()
        wave = self._current_wave()
        if wave is None:
            return reward_breakdown

        spawn_schedule = self._build_spawn_schedule(wave)
        self._wave_in_progress = True
        self._active_enemies = []
        self._attack_effects = []
        schedule_index = 0
        tick_count = 0

        while True:
            self.window_controller.poll_events_or_raise()

            while schedule_index < len(spawn_schedule) and int(spawn_schedule[schedule_index][0]) <= int(tick_count):
                _, side, enemy_kind = spawn_schedule[schedule_index]
                self._active_enemies.append(self._spawn_enemy(str(side), str(enemy_kind)))
                schedule_index += 1

            self._tick_towers(self._active_enemies)
            self._collect_kills(self._active_enemies, reward_breakdown)
            self._move_enemies(self._active_enemies, reward_breakdown)
            self._trim_effects()
            self._active_enemies = [enemy for enemy in self._active_enemies if enemy.alive]

            self.render()
            self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
            tick_count += 1

            if int(self.lives) <= 0:
                self._last_outcome = "loss"
                reward_breakdown["reward_terminal_loss"] += float(config.REWARD_TERMINAL_LOSS)
                break

            if schedule_index >= len(spawn_schedule) and not self._active_enemies:
                self.gold += int(config.WAVE_CLEAR_GOLD_BONUS)
                reward_breakdown["reward_progress_wave_clear"] += float(config.REWARD_PROGRESS_WAVE_CLEAR)
                self.wave_index += 1
                if int(self.wave_index) >= len(self.wave_plan):
                    self._last_outcome = "win"
                    reward_breakdown["reward_terminal_win"] += float(config.REWARD_TERMINAL_WIN)
                break

            if tick_count >= 2_000:
                self._last_outcome = "loss"
                reward_breakdown["reward_terminal_loss"] += float(config.REWARD_TERMINAL_LOSS)
                break

        self._wave_in_progress = False
        self._active_enemies = []
        self._attack_effects = []
        if not self._last_outcome:
            self.actions_remaining = int(self.decision_budget)
        return reward_breakdown

    def _build_tower(self, tower_kind: str, slot_name: str) -> None:
        profile = TOWER_PROFILES[str(tower_kind)]
        if self.slot_towers[str(slot_name)] is not None or int(self.gold) < int(profile.build_cost):
            return
        self.gold -= int(profile.build_cost)
        self.slot_towers[str(slot_name)] = TowerState(
            kind=str(tower_kind),
            level=1,
            cooldown_ticks=0,
            total_spent=int(profile.build_cost),
        )
        self.actions_remaining = max(0, int(self.actions_remaining) - 1)

    def _upgrade_tower(self, slot_name: str) -> None:
        tower_state = self.slot_towers[str(slot_name)]
        if tower_state is None:
            return
        upgrade_cost = int(tower_state.profile.upgrade_cost_for_level(int(tower_state.level)))
        if int(self.gold) < int(upgrade_cost):
            return
        self.gold -= int(upgrade_cost)
        tower_state.level = min(3, int(tower_state.level) + 1)
        tower_state.total_spent += int(upgrade_cost)
        tower_state.cooldown_ticks = 0
        self.actions_remaining = max(0, int(self.actions_remaining) - 1)

    def _sell_tower(self, slot_name: str) -> None:
        tower_state = self.slot_towers[str(slot_name)]
        if tower_state is None:
            return
        refund = self._sell_value(str(slot_name))
        self.gold += int(refund)
        self.slot_towers[str(slot_name)] = None
        self.actions_remaining = max(0, int(self.actions_remaining) - 1)

    def _apply_build_phase_action(self, action_idx: int) -> dict[str, float]:
        reward_breakdown = self._empty_reward_breakdown()
        action_kind, tower_kind, slot_name = self._decode_action(int(action_idx))
        if action_kind == "build" and tower_kind is not None and slot_name is not None:
            self._build_tower(str(tower_kind), str(slot_name))
        elif action_kind == "upgrade" and slot_name is not None:
            self._upgrade_tower(str(slot_name))
        elif action_kind == "sell" and slot_name is not None:
            self._sell_tower(str(slot_name))
        return reward_breakdown

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self._last_outcome:
            return np.asarray(self._last_obs, dtype=np.float32), 0.0, True, {
                "win": self._last_outcome == "win",
                "success": int(self._last_episode_success),
                "level": int(self._last_episode_level),
                "reward_components": self._episode_reward_components.totals(),
            }

        self.window_controller.poll_events_or_raise()
        episode_level = int(self._current_level)

        if self.mode == "human":
            proposed_action = self._human_action()
            if proposed_action is None:
                self._last_obs = self._build_observation()
                self.render()
                self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
                return np.asarray(self._last_obs, dtype=np.float32), 0.0, False, {
                    "win": False,
                    "success": 0,
                    "level": int(episode_level),
                    "level_changed": False,
                    "reward_breakdown": {},
                }
            mask = self.get_action_mask()
            if not (0 <= int(proposed_action) < self.ACT_DIM and bool(mask[int(proposed_action)])):
                self._last_obs = self._build_observation()
                self.render()
                self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)
                return np.asarray(self._last_obs, dtype=np.float32), 0.0, False, {
                    "win": False,
                    "success": 0,
                    "level": int(episode_level),
                    "level_changed": False,
                    "reward_breakdown": {},
                }
            action_idx = int(proposed_action)
        else:
            action_idx = self._coerce_action(action)

        if int(action_idx) == 0:
            reward_breakdown = self._simulate_wave()
        else:
            reward_breakdown = self._apply_build_phase_action(int(action_idx))

        reward = float(sum(float(value) for value in reward_breakdown.values()))
        if self.mode != "human":
            for key, value in reward_breakdown.items():
                if abs(float(value)) > 1e-12:
                    self._episode_reward_components.add(str(key), float(value))

        success = 1 if self._last_outcome == "win" else 0
        level_changed = False
        done = bool(self._last_outcome)
        if done:
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(success)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(success),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )

        self._last_obs = self._build_observation()
        if int(action_idx) != 0:
            self.render()
            self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)

        info: dict[str, object] = {
            "win": bool(self._last_outcome == "win") if done else False,
            "success": int(success) if done else 0,
            "level": int(episode_level),
            "level_changed": bool(level_changed),
            "reward_breakdown": reward_breakdown if self.mode != "human" else {},
        }
        if done:
            info["reward_components"] = self._episode_reward_components.totals()
        return np.asarray(self._last_obs, dtype=np.float32), float(reward), bool(done), info

    def _draw_block(
        self,
        left: float,
        top: float,
        width: float,
        height: float,
        outer_color: tuple[int, int, int],
        inner_color: tuple[int, int, int],
        inset: float = CELL_INSET,
    ) -> None:
        arcade.draw_lbwh_rectangle_filled(
            float(left),
            float(self.window_controller.to_arcade_y(float(top) + float(height))),
            float(width),
            float(height),
            outer_color,
        )
        inner_left = float(left) + float(inset)
        inner_top = float(top) + float(inset)
        inner_width = float(width) - float(inset) * 2.0
        inner_height = float(height) - float(inset) * 2.0
        if inner_width <= 0.0 or inner_height <= 0.0:
            return
        arcade.draw_lbwh_rectangle_filled(
            float(inner_left),
            float(self.window_controller.to_arcade_y(float(inner_top) + float(inner_height))),
            float(inner_width),
            float(inner_height),
            inner_color,
        )

    def _draw_paths(self) -> None:
        path_cells = _rect_cells(self.layout.ground_rects)
        border_width = float(CELL_INSET)
        for col, row in path_cells:
            left, top, width, height = _block_rect(int(col), int(row), 1, 1)
            arcade.draw_lbwh_rectangle_filled(
                float(left),
                float(self.window_controller.to_arcade_y(float(top) + float(height))),
                float(width),
                float(height),
                GROUND_PATH_INNER,
            )

        for col, row in path_cells:
            left, top, width, height = _block_rect(int(col), int(row), 1, 1)
            top_open = (int(col), int(row) - 1) not in path_cells
            bottom_open = (int(col), int(row) + 1) not in path_cells
            left_open = (int(col) - 1, int(row)) not in path_cells
            right_open = (int(col) + 1, int(row)) not in path_cells

            if top_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left),
                    float(self.window_controller.to_arcade_y(float(top) + border_width)),
                    float(width),
                    float(border_width),
                    GROUND_PATH_OUTER,
                )
            if bottom_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left),
                    float(self.window_controller.to_arcade_y(float(top) + float(height))),
                    float(width),
                    float(border_width),
                    GROUND_PATH_OUTER,
                )
            if left_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left),
                    float(self.window_controller.to_arcade_y(float(top) + float(height))),
                    float(border_width),
                    float(height),
                    GROUND_PATH_OUTER,
                )
            if right_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left + float(width) - border_width),
                    float(self.window_controller.to_arcade_y(float(top) + float(height))),
                    float(border_width),
                    float(height),
                    GROUND_PATH_OUTER,
                )
            if top_open or left_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left),
                    float(self.window_controller.to_arcade_y(float(top) + border_width)),
                    float(border_width),
                    float(border_width),
                    GROUND_PATH_OUTER,
                )
            if top_open or right_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left + float(width) - border_width),
                    float(self.window_controller.to_arcade_y(float(top) + border_width)),
                    float(border_width),
                    float(border_width),
                    GROUND_PATH_OUTER,
                )
            if bottom_open or left_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left),
                    float(self.window_controller.to_arcade_y(float(top) + float(height))),
                    float(border_width),
                    float(border_width),
                    GROUND_PATH_OUTER,
                )
            if bottom_open or right_open:
                arcade.draw_lbwh_rectangle_filled(
                    float(left + float(width) - border_width),
                    float(self.window_controller.to_arcade_y(float(top) + float(height))),
                    float(border_width),
                    float(border_width),
                    GROUND_PATH_OUTER,
                )

    def _draw_entries_and_exit(self) -> None:
        for side, cell_top_left in ENTRY_CELLS.items():
            x, y = ENTRY_POSITIONS[side]
            label = "L" if side == SIDE_LEFT else "R"
            self._draw_block(
                *_block_rect(int(cell_top_left[0]), int(cell_top_left[1]), TRACK_WIDTH_CELLS, TRACK_WIDTH_CELLS),
                outer_color=ENTRY_OUTLINE,
                inner_color=ENTRY_FILL,
            )
            self._text_cache.draw(
                label,
                x=float(x),
                y=float(self.window_controller.to_arcade_y(float(y))),
                color=COLOR_LIGHT_NEUTRAL,
                font_size=10,
                font_name=("Roboto", "Arial", "sans-serif"),
                anchor_x="center",
                anchor_y="center",
            )
        exit_x, exit_y = EXIT_POSITION
        self._draw_block(
            *_block_rect(int(EXIT_CELL[0]), int(EXIT_CELL[1]), TRACK_WIDTH_CELLS, TRACK_WIDTH_CELLS),
            outer_color=EXIT_OUTLINE,
            inner_color=EXIT_FILL,
        )
        self._text_cache.draw(
            "E",
            x=float(exit_x),
            y=float(self.window_controller.to_arcade_y(float(exit_y))),
            color=COLOR_LIGHT_NEUTRAL,
            font_size=10,
            font_name=("Roboto", "Arial", "sans-serif"),
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_slots(self) -> None:
        tower_labels = {"arrow": "A", "cannon": "C", "tesla": "T"}
        for slot_index, slot_name in enumerate(config.SLOT_NAMES):
            x, y = SLOT_POSITIONS[str(slot_name)]
            tower_state = self.slot_towers[str(slot_name)]
            selected = int(slot_index) == int(self._selected_slot_index)
            tower_outline, tower_fill = (SLOT_OUTLINE, SLOT_FILL)
            if tower_state is not None:
                tower_outline, tower_fill = TOWER_COLORS[str(tower_state.kind)]
            if tower_state is None:
                tower_fill = EMPTY_SLOT_FILL
            if selected and self.mode == "human":
                tower_outline = SELECTED_SLOT_COLOR
            self._draw_block(
                *_block_rect(int(SLOT_CELLS[str(slot_name)][0]), int(SLOT_CELLS[str(slot_name)][1]), 2, 2),
                outer_color=tower_outline,
                inner_color=tower_fill,
                inset=float(SLOT_INSET),
            )
            if tower_state is None:
                continue

            self._text_cache.draw(
                f"{tower_labels[str(tower_state.kind)]}{int(tower_state.level)}",
                x=float(x),
                y=float(self.window_controller.to_arcade_y(float(y))),
                color=COLOR_LIGHT_NEUTRAL,
                font_size=10,
                font_name=("Roboto", "Arial", "sans-serif"),
                anchor_x="center",
                anchor_y="center",
            )

    def _draw_enemy(self, enemy: EnemyState) -> None:
        x, y = enemy.position
        next_pos = enemy.path.position_at(min(float(enemy.distance_along) + max(2.0, float(enemy.speed) * 3.0), enemy.path.total_length))
        heading_degrees = math.degrees(math.atan2(float(next_pos[1]) - float(y), float(next_pos[0]) - float(x)))
        outline_color, fill_color = ENEMY_COLORS[str(enemy.kind)]
        draw_two_tone_tile(
            self.window_controller,
            top_left_x=float(x) - ENEMY_SIZE * 0.5,
            top_left_y=float(y) - ENEMY_SIZE * 0.5,
            size=float(ENEMY_SIZE),
            outer_color=outline_color,
            inner_color=fill_color,
            inset=float(ENEMY_INSET),
        )
        draw_facing_indicator(
            self.window_controller,
            center_x=float(x),
            center_y_top_left=float(y),
            angle_degrees=float(heading_degrees),
            length=float(ENEMY_SIZE * 0.5),
            color=COLOR_LIGHT_NEUTRAL,
            line_width=1.5,
        )

    def _draw_effects(self) -> None:
        for effect in self._attack_effects:
            if not effect.points:
                continue
            screen_points = [
                (float(x), float(self.window_controller.to_arcade_y(float(y))))
                for x, y in effect.points
            ]
            if str(effect.kind) == "arrow" and len(screen_points) >= 2:
                arcade.draw_line(
                    screen_points[0][0],
                    screen_points[0][1],
                    screen_points[1][0],
                    screen_points[1][1],
                    COLOR_AQUA,
                    2.0,
                )
            elif str(effect.kind) == "cannon" and len(screen_points) >= 2:
                arcade.draw_line(
                    screen_points[0][0],
                    screen_points[0][1],
                    screen_points[1][0],
                    screen_points[1][1],
                    COLOR_SAND,
                    2.0,
                )
                arcade.draw_circle_outline(
                    screen_points[1][0],
                    screen_points[1][1],
                    float(effect.radius),
                    COLOR_SAND,
                    2.0,
                )
            elif len(screen_points) >= 2:
                arcade.draw_line_strip(screen_points, COLOR_BLUE, 2.0)

    def _draw_world(self) -> None:
        arcade.draw_lbwh_rectangle_filled(
            0.0,
            float(config.BB_HEIGHT),
            float(config.SCREEN_WIDTH),
            float(config.WORLD_HEIGHT),
            WORLD_BG_BOTTOM,
        )
        self._draw_paths()
        self._draw_entries_and_exit()
        self._draw_slots()
        for enemy in self._active_enemies:
            self._draw_enemy(enemy)
        self._draw_effects()

    def _draw_tower_table(self) -> None:
        panel_width = 320.0
        panel_height = 52.0
        panel_left = float(config.SCREEN_WIDTH) - panel_width - 8.0
        panel_bottom = float(config.BB_HEIGHT) + 8.0
        panel_fill = COLOR_LIGHT_NEUTRAL + (24,)
        panel_line = COLOR_LIGHT_NEUTRAL + (128,)
        panel_text = COLOR_LIGHT_NEUTRAL + (128,)
        header_y = panel_bottom + panel_height - 12.0
        row_gap = 12.0
        columns = (
            ("Name", 8.0),
            ("Cst", 76.0),
            ("Dmg", 114.0),
            ("Area", 156.0),
            ("Rate", 206.0),
            ("L2/L3", 258.0),
        )

        arcade.draw_lbwh_rectangle_filled(panel_left, panel_bottom, panel_width, panel_height, panel_fill)
        arcade.draw_lbwh_rectangle_outline(panel_left, panel_bottom, panel_width, panel_height, panel_line, 1.0)

        for header, offset_x in columns:
            self._text_cache.draw(
                header,
                x=panel_left + offset_x,
                y=header_y,
                color=panel_text,
                font_size=10,
                font_name=("Roboto", "Arial", "sans-serif"),
                anchor_x="left",
                anchor_y="center",
            )

        tower_rows = (
            ("Arrow", TOWER_PROFILES["arrow"], TOWER_PROFILES["arrow"].level_stats[0]),
            ("Cannon", TOWER_PROFILES["cannon"], TOWER_PROFILES["cannon"].level_stats[0]),
            ("Tesla", TOWER_PROFILES["tesla"], TOWER_PROFILES["tesla"].level_stats[0]),
        )
        for row_index, (name, profile, stats) in enumerate(tower_rows, start=1):
            if float(stats.splash_radius) > 0.0:
                area_text = f"{int(round(float(stats.splash_radius) / float(TILE_SIZE)))}r"
            elif int(stats.chain_count) > 1:
                area_text = f"x{int(stats.chain_count)}"
            else:
                area_text = "1"
            row_y = header_y - float(row_index) * row_gap
            values = (
                name,
                str(int(profile.build_cost)),
                f"{float(stats.damage):.1f}",
                area_text,
                f"{30.0 / float(stats.cooldown_ticks):.1f}",
                f"{int(profile.upgrade_costs[0])}/{int(profile.upgrade_costs[1])}",
            )
            for (_, offset_x), value in zip(columns, values):
                self._text_cache.draw(
                    str(value),
                    x=panel_left + offset_x,
                    y=row_y,
                    color=panel_text,
                    font_size=10,
                    font_name=("Roboto", "Arial", "sans-serif"),
                    anchor_x="left",
                    anchor_y="center",
                )

    def _draw_hud(self) -> None:
        arcade.draw_lbwh_rectangle_filled(0, 0, config.SCREEN_WIDTH, config.BB_HEIGHT, COLOR_DARK_NEUTRAL)
        sell_value = self._sell_value(self._selected_slot_name())
        status_parts = [
            f"Budget {int(self.gold)}",
            f"Lives {int(self.lives)}",
        ]
        if int(sell_value) > 0:
            status_parts.append(f"Sell Value {int(sell_value)}")
        self._text_cache.draw(
            "   ".join(status_parts),
            x=14.0,
            y=float(config.BB_HEIGHT) - 16.0,
            color=COLOR_LIGHT_NEUTRAL,
            font_size=13,
            font_name=("Roboto", "Arial", "sans-serif"),
            anchor_x="left",
            anchor_y="center",
        )
        self._draw_tower_table()
        if self._last_outcome:
            result_text = "Victory" if self._last_outcome == "win" else "Defeat"
            self._text_cache.draw(
                result_text,
                x=float(config.SCREEN_WIDTH) - 286.0,
                y=12.0,
                color=COLOR_LIGHT_NEUTRAL,
                font_size=11,
                font_name=("Roboto", "Arial", "sans-serif"),
                anchor_x="right",
                anchor_y="center",
            )

    def render(self) -> None:
        if self.window_controller.window is None:
            return
        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        self._draw_world()
        self._draw_hud()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
