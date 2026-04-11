"""Wave-based tower defense environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random

import arcade
import numpy as np

from assets.paths import resolve_font_path
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
    COLOR_PURPLE,
    COLOR_SAND,
    COLOR_SLATE_GRAY,
    COLOR_DEEP_PURPLE,
    DEFAULT_CELL_INSET,
    DEFAULT_TILE_SIZE,
    GAME_TITLE_FONT_NAME,
    GAME_UI_FONT_NAME,
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
from core.primitives import (
    draw_cell_union_outline,
    draw_facing_indicator,
    draw_status_bar,
    draw_two_tone_tile,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController, TextCache, load_font_once
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
TOWER_KIND_TO_ID = {"fast": 1.0, "heavy": 2.0, "area": 3.0}
SELL_REFUND_BY_LEVEL = {1: 0.90, 2: 0.75, 3: 0.60}
SPAWN_GAPS = {"light": 8, "armored": 16, "flying": 10}

WORLD_BG_TOP = COLOR_SLATE_GRAY
WORLD_BG_BOTTOM = COLOR_DARK_NEUTRAL
GROUND_PATH_OUTER = COLOR_FOG_GRAY
GROUND_PATH_INNER = COLOR_SLATE_GRAY
ENTRY_OUTLINE = COLOR_CORAL
ENTRY_FILL = COLOR_BRICK_RED
EXIT_OUTLINE = COLOR_AQUA
EXIT_FILL = COLOR_DEEP_TEAL
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
MENU_FILL = COLOR_DARK_NEUTRAL + (128,)
MENU_BORDER = COLOR_LIGHT_NEUTRAL + (128,)
MENU_TEXT = COLOR_LIGHT_NEUTRAL + (128,)
MENU_TEXT_DISABLED = COLOR_FOG_GRAY + (96,)
MENU_ITEM_WIDTH = 84.0
MENU_ITEM_HEIGHT = 20.0
MENU_ITEM_GAP = 3.0
UI_FONT_NAME = GAME_UI_FONT_NAME
BLOCK_FONT_NAME = GAME_TITLE_FONT_NAME
GRID_COLS = int(round(float(config.WORLD_WIDTH) / float(TILE_SIZE)))
GRID_ROWS = int(round(float(config.WORLD_HEIGHT) / float(TILE_SIZE)))
MAP_MARGIN_CELLS = 2
CENTER_TRUNK_COL = GRID_COLS // 2 - TRACK_WIDTH_CELLS // 2
ENTRY_MARKER_DEPTH_CELLS = max(2, TRACK_WIDTH_CELLS // 2)
EXIT_MARKER_DEPTH_CELLS = max(2, TRACK_WIDTH_CELLS // 2)
OUTER_MARKER_MARGIN_CELLS = ENTRY_MARKER_DEPTH_CELLS
LEFT_ENTRY_COL = MAP_MARGIN_CELLS + OUTER_MARKER_MARGIN_CELLS
RIGHT_ENTRY_COL = GRID_COLS - TRACK_WIDTH_CELLS - MAP_MARGIN_CELLS - OUTER_MARKER_MARGIN_CELLS
EXIT_ROW = GRID_ROWS - TRACK_WIDTH_CELLS - MAP_MARGIN_CELLS - EXIT_MARKER_DEPTH_CELLS
SPAWN_ROW = MAP_MARGIN_CELLS + OUTER_MARKER_MARGIN_CELLS + 4

# Tower generation uses two compact handcrafted templates:
# - Soft S Merge: mirrored inward sweeps that meet at the center trunk.
# - Offset S: one side joins the center earlier while the other descends farther.
# Both keep the same fixed endpoints and the same five stable slot ids:
# left=left_corner, upper=left_shared, mid=center_trunk, lower=right_shared, right=right_corner.
PATH_TEMPLATES = (
    {"name": "Soft S Merge", "kind": "soft_s"},
    {"name": "Offset S", "kind": "offset_s"},
)


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def _lane_point(left_col: int, top_row: int) -> tuple[float, float]:
    return (
        (float(left_col) + float(TRACK_WIDTH_CELLS) * 0.5) * TILE_SIZE,
        (float(top_row) + float(TRACK_WIDTH_CELLS) * 0.5) * TILE_SIZE,
    )


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

    def tangent_at(self, distance_along: float) -> tuple[float, float]:
        distance_value = float(max(0.0, min(float(distance_along), self.total_length)))
        if distance_value <= 0.0:
            start, end = self.points[0], self.points[1]
        elif distance_value >= self.total_length:
            start, end = self.points[-2], self.points[-1]
        else:
            start = self.points[0]
            end = self.points[1]
            for index, segment_length in enumerate(self.segment_lengths):
                end_len = float(self.cumulative_lengths[index + 1])
                if distance_value <= end_len or index == len(self.segment_lengths) - 1:
                    start = self.points[index]
                    end = self.points[index + 1]
                    if segment_length > 1e-6:
                        break
        delta_x = float(end[0]) - float(start[0])
        delta_y = float(end[1]) - float(start[1])
        length = math.hypot(delta_x, delta_y)
        if length <= 1e-6:
            return (1.0, 0.0)
        return (delta_x / length, delta_y / length)


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: int
    layout_norm: float
    name: str
    ground_paths: dict[str, LanePath]
    flying_paths: dict[str, LanePath]
    ground_rects: tuple[tuple[int, int, int, int], ...]
    slot_cells: dict[str, tuple[int, int]]
    slot_positions: dict[str, tuple[float, float]]
    entry_cells: dict[str, tuple[int, int]]
    entry_positions: dict[str, tuple[float, float]]
    exit_cell: tuple[int, int]
    exit_position: tuple[float, float]


@dataclass(frozen=True)
class WaveSpec:
    entry_mode: str
    count_light: int
    count_armored: int
    count_flying: int

    def count_for(self, enemy_kind: str) -> int:
        if enemy_kind == "light":
            return int(self.count_light)
        if enemy_kind == "armored":
            return int(self.count_armored)
        if enemy_kind == "flying":
            return int(self.count_flying)
        raise KeyError(f"Unknown enemy kind '{enemy_kind}'.")

    @property
    def total_count(self) -> int:
        return int(self.count_light + self.count_armored + self.count_flying)


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
    lane_offset: float = 0.0
    distance_along: float = 0.0
    alive: bool = True
    leaked: bool = False

    @property
    def position(self) -> tuple[float, float]:
        center_x, center_y = self.path.position_at(float(self.distance_along))
        tangent_x, tangent_y = self.path.tangent_at(float(self.distance_along))
        normal_x = -float(tangent_y)
        normal_y = float(tangent_x)
        return (
            float(center_x) + normal_x * float(self.lane_offset),
            float(center_y) + normal_y * float(self.lane_offset),
        )

    @property
    def progress_norm(self) -> float:
        return float(self.distance_along) / float(max(1e-6, self.path.total_length))


@dataclass
class AttackEffect:
    kind: str
    points: tuple[tuple[float, float], ...]
    ttl: int
    radius: float = 0.0
    max_ttl: int = 0
    delay_ticks: int = 0


def _compress_cells(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    compressed: list[tuple[int, int]] = []
    for col, row in cells:
        cell = (int(col), int(row))
        if not compressed or cell != compressed[-1]:
            compressed.append(cell)
    return tuple(compressed)


def _segment_rect(start_cell: tuple[int, int], end_cell: tuple[int, int]) -> tuple[int, int, int, int]:
    start_col, start_row = (int(start_cell[0]), int(start_cell[1]))
    end_col, end_row = (int(end_cell[0]), int(end_cell[1]))
    if start_col == end_col:
        return (
            int(start_col),
            min(int(start_row), int(end_row)),
            int(TRACK_WIDTH_CELLS),
            abs(int(end_row) - int(start_row)) + int(TRACK_WIDTH_CELLS),
        )
    if start_row == end_row:
        return (
            min(int(start_col), int(end_col)),
            int(start_row),
            abs(int(end_col) - int(start_col)) + int(TRACK_WIDTH_CELLS),
            int(TRACK_WIDTH_CELLS),
        )
    raise ValueError("Tower paths must move orthogonally.")


def _path_rects(*paths: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
    rects: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for path_cells in paths:
        for start_cell, end_cell in zip(path_cells[:-1], path_cells[1:]):
            if tuple(start_cell) == tuple(end_cell):
                continue
            rect = _segment_rect(tuple(start_cell), tuple(end_cell))
            if rect not in seen:
                seen.add(rect)
                rects.append(rect)
    return tuple(rects)


def _lane_path_from_cells(path_cells: tuple[tuple[int, int], ...]) -> LanePath:
    return LanePath(tuple(_lane_point(int(col), int(row)) for col, row in path_cells))


def _slot_block_cells(cell_top_left: tuple[int, int]) -> set[tuple[int, int]]:
    left_col, top_row = (int(cell_top_left[0]), int(cell_top_left[1]))
    return {
        (int(left_col), int(top_row)),
        (int(left_col) + 1, int(top_row)),
        (int(left_col), int(top_row) + 1),
        (int(left_col) + 1, int(top_row) + 1),
    }


def _slot_touches_path(cell_top_left: tuple[int, int], path_cells: set[tuple[int, int]]) -> bool:
    for col, row in _slot_block_cells(cell_top_left):
        for delta_col, delta_row in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            if (int(col) + int(delta_col), int(row) + int(delta_row)) in path_cells:
                return True
    return False


def _pick_option(rng: random.Random, values: tuple[int | str, ...]) -> int | str:
    return values[int(rng.randrange(len(values)))]


def _build_layout_spec(
    *,
    template_index: int,
    template_name: str,
    left_path_cells: tuple[tuple[int, int], ...],
    right_path_cells: tuple[tuple[int, int], ...],
    slot_cells: dict[str, tuple[int, int]],
    entry_cells: dict[str, tuple[int, int]],
    exit_cell: tuple[int, int],
) -> LayoutSpec:
    adjusted_slot_cells = {str(slot_name): (int(cell_top_left[0]), int(cell_top_left[1])) for slot_name, cell_top_left in slot_cells.items()}
    mid_col, mid_row = adjusted_slot_cells["mid"]
    while True:
        mid_block = _slot_block_cells((int(mid_col), int(mid_row)))
        overlaps_other_slot = False
        for slot_name, cell_top_left in adjusted_slot_cells.items():
            if str(slot_name) == "mid":
                continue
            if mid_block.intersection(_slot_block_cells(cell_top_left)):
                overlaps_other_slot = True
                break
        if not overlaps_other_slot or int(mid_row) >= int(EXIT_ROW) - 4:
            break
        mid_row += 2
    adjusted_slot_cells["mid"] = (int(mid_col), int(mid_row))

    ground_rects = _path_rects(left_path_cells, right_path_cells)
    path_cells = _rect_cells(ground_rects)
    slot_blocks = {
        str(slot_name): _slot_block_cells(cell_top_left)
        for slot_name, cell_top_left in adjusted_slot_cells.items()
    }
    all_slot_cells: set[tuple[int, int]] = set()
    for block_cells in slot_blocks.values():
        if all_slot_cells.intersection(block_cells):
            raise RuntimeError("Tower layout template made slots overlap each other.")
        all_slot_cells.update(block_cells)
    if any(path_cells.intersection(block_cells) for block_cells in slot_blocks.values()):
        raise RuntimeError("Tower layout template placed a slot on the path.")
    if not all(_slot_touches_path(cell_top_left, path_cells) for cell_top_left in slot_cells.values()):
        raise RuntimeError("Tower layout template left a slot detached from the path.")

    slot_positions = {
        str(slot_name): _block_center(*cell_top_left)
        for slot_name, cell_top_left in adjusted_slot_cells.items()
    }
    entry_positions = {
        str(side): _lane_point(*cell_top_left)
        for side, cell_top_left in entry_cells.items()
    }

    max_layout_code = float(max(1, len(PATH_TEMPLATES) - 1))
    return LayoutSpec(
        layout_id=int(template_index),
        layout_norm=clip_unit(float(template_index) / float(max_layout_code)),
        name=str(template_name),
        ground_paths={
            SIDE_LEFT: _lane_path_from_cells(left_path_cells),
            SIDE_RIGHT: _lane_path_from_cells(right_path_cells),
        },
        flying_paths={
            SIDE_LEFT: _lane_path_from_cells(left_path_cells),
            SIDE_RIGHT: _lane_path_from_cells(right_path_cells),
        },
        ground_rects=ground_rects,
        slot_cells={str(slot_name): tuple(cell_top_left) for slot_name, cell_top_left in adjusted_slot_cells.items()},
        slot_positions=slot_positions,
        entry_cells={str(side): tuple(cell_top_left) for side, cell_top_left in entry_cells.items()},
        entry_positions=entry_positions,
        exit_cell=tuple(exit_cell),
        exit_position=_lane_point(*exit_cell),
    )


def _generate_soft_s_layout(
    rng: random.Random,
    *,
    template_index: int,
    entry_cells: dict[str, tuple[int, int]],
    exit_cell: tuple[int, int],
) -> LayoutSpec:
    # Template A: mirrored inward sweeps that meet at one shared center trunk.
    left_turn_col = int(_pick_option(rng, (9, 10, 11, 12)))
    right_turn_col = int(_pick_option(rng, (35, 36, 37, 38)))
    merge_row = int(_pick_option(rng, (14, 15, 16)))
    trunk_side = str(_pick_option(rng, ("left", "right")))

    left_path_cells = _compress_cells(
        (
            entry_cells[SIDE_LEFT],
            (int(left_turn_col), int(SPAWN_ROW)),
            (int(left_turn_col), int(merge_row)),
            (int(CENTER_TRUNK_COL), int(merge_row)),
            exit_cell,
        )
    )
    right_path_cells = _compress_cells(
        (
            entry_cells[SIDE_RIGHT],
            (int(right_turn_col), int(SPAWN_ROW)),
            (int(right_turn_col), int(merge_row)),
            (int(CENTER_TRUNK_COL), int(merge_row)),
            exit_cell,
        )
    )

    left_shared_col = max(int(left_turn_col) + int(TRACK_WIDTH_CELLS) + 1, int(CENTER_TRUNK_COL) - 6)
    right_shared_col = min(int(right_turn_col) - 3, int(CENTER_TRUNK_COL) + 2)
    trunk_row = min(int(EXIT_ROW) - 4, int(merge_row) + int(_pick_option(rng, (5, 6))))
    slot_cells = {
        "left": (int(left_turn_col) + int(TRACK_WIDTH_CELLS), int(SPAWN_ROW) + int(TRACK_WIDTH_CELLS)),
        "upper": (int(left_shared_col), int(merge_row) + int(TRACK_WIDTH_CELLS)),
        "mid": (
            int(CENTER_TRUNK_COL) - 2
            if trunk_side == "left"
            else int(CENTER_TRUNK_COL) + int(TRACK_WIDTH_CELLS),
            int(trunk_row),
        ),
        "lower": (int(right_shared_col), int(merge_row) + int(TRACK_WIDTH_CELLS)),
        "right": (int(right_turn_col) - 2, int(SPAWN_ROW) + int(TRACK_WIDTH_CELLS)),
    }
    return _build_layout_spec(
        template_index=int(template_index),
        template_name="Soft S Merge",
        left_path_cells=left_path_cells,
        right_path_cells=right_path_cells,
        slot_cells=slot_cells,
        entry_cells=entry_cells,
        exit_cell=exit_cell,
    )


def _generate_offset_s_layout(
    rng: random.Random,
    *,
    template_index: int,
    entry_cells: dict[str, tuple[int, int]],
    exit_cell: tuple[int, int],
) -> LayoutSpec:
    # Template B: one side joins the center earlier while the other side descends farther.
    early_side = str(_pick_option(rng, (SIDE_LEFT, SIDE_RIGHT)))
    left_turn_col = int(_pick_option(rng, (10, 11, 12)))
    right_turn_col = int(_pick_option(rng, (35, 36, 37)))
    early_join_row = int(_pick_option(rng, (12, 13, 14)))
    late_join_row = int(early_join_row + int(_pick_option(rng, (2, 3))))
    left_join_row = int(early_join_row if early_side == SIDE_LEFT else late_join_row)
    right_join_row = int(early_join_row if early_side == SIDE_RIGHT else late_join_row)
    trunk_side = "right" if early_side == SIDE_LEFT else "left"

    left_path_cells = _compress_cells(
        (
            entry_cells[SIDE_LEFT],
            (int(left_turn_col), int(SPAWN_ROW)),
            (int(left_turn_col), int(left_join_row)),
            (int(CENTER_TRUNK_COL), int(left_join_row)),
            exit_cell,
        )
    )
    right_path_cells = _compress_cells(
        (
            entry_cells[SIDE_RIGHT],
            (int(right_turn_col), int(SPAWN_ROW)),
            (int(right_turn_col), int(right_join_row)),
            (int(CENTER_TRUNK_COL), int(right_join_row)),
            exit_cell,
        )
    )

    left_shared_col = max(int(left_turn_col) + int(TRACK_WIDTH_CELLS) + 1, int(CENTER_TRUNK_COL) - 6)
    right_shared_col = min(int(right_turn_col) - 3, int(CENTER_TRUNK_COL) + 2)
    trunk_row = min(int(EXIT_ROW) - 4, int(max(left_join_row, right_join_row)) + int(_pick_option(rng, (4, 5))))
    slot_cells = {
        "left": (int(left_turn_col) + int(TRACK_WIDTH_CELLS), int(SPAWN_ROW) + int(TRACK_WIDTH_CELLS)),
        "upper": (int(left_shared_col), int(left_join_row) + int(TRACK_WIDTH_CELLS)),
        "mid": (
            int(CENTER_TRUNK_COL) - 2
            if trunk_side == "left"
            else int(CENTER_TRUNK_COL) + int(TRACK_WIDTH_CELLS),
            int(trunk_row),
        ),
        "lower": (int(right_shared_col), int(right_join_row) + int(TRACK_WIDTH_CELLS)),
        "right": (int(right_turn_col) - 2, int(SPAWN_ROW) + int(TRACK_WIDTH_CELLS)),
    }
    return _build_layout_spec(
        template_index=int(template_index),
        template_name="Offset S",
        left_path_cells=left_path_cells,
        right_path_cells=right_path_cells,
        slot_cells=slot_cells,
        entry_cells=entry_cells,
        exit_cell=exit_cell,
    )


def _generate_layout(rng: random.Random) -> LayoutSpec:
    template_index = int(rng.randrange(len(PATH_TEMPLATES)))
    template = dict(PATH_TEMPLATES[int(template_index)])
    entry_cells = {
        SIDE_LEFT: (int(LEFT_ENTRY_COL), int(SPAWN_ROW)),
        SIDE_RIGHT: (int(RIGHT_ENTRY_COL), int(SPAWN_ROW)),
    }
    exit_cell = (int(CENTER_TRUNK_COL), int(EXIT_ROW))
    if str(template["kind"]) == "soft_s":
        return _generate_soft_s_layout(
            rng,
            template_index=int(template_index),
            entry_cells=entry_cells,
            exit_cell=exit_cell,
        )
    return _generate_offset_s_layout(
        rng,
        template_index=int(template_index),
        entry_cells=entry_cells,
        exit_cell=exit_cell,
    )

TOWER_PROFILES = {
    "fast": TowerProfile(
        build_cost=5,
        upgrade_costs=(4, 7),
        level_stats=(
            TowerStats(damage=1.10, cooldown_ticks=13, attack_range=148.0, armor_pierce=0.05),
            TowerStats(damage=1.35, cooldown_ticks=11, attack_range=156.0, armor_pierce=0.10),
            TowerStats(damage=1.70, cooldown_ticks=9, attack_range=166.0, armor_pierce=0.15),
        ),
    ),
    "heavy": TowerProfile(
        build_cost=5,
        upgrade_costs=(4, 7),
        level_stats=(
            TowerStats(damage=4.20, cooldown_ticks=30, attack_range=136.0, armor_pierce=1.05),
            TowerStats(damage=5.60, cooldown_ticks=26, attack_range=144.0, armor_pierce=1.35),
            TowerStats(damage=7.10, cooldown_ticks=22, attack_range=152.0, armor_pierce=1.70),
        ),
    ),
    "area": TowerProfile(
        build_cost=5,
        upgrade_costs=(4, 7),
        level_stats=(
            TowerStats(damage=1.25, cooldown_ticks=20, attack_range=138.0, armor_pierce=0.05, splash_radius=30.0),
            TowerStats(damage=1.50, cooldown_ticks=18, attack_range=146.0, armor_pierce=0.10, splash_radius=38.0),
            TowerStats(damage=1.80, cooldown_ticks=16, attack_range=154.0, armor_pierce=0.15, splash_radius=46.0),
        ),
    ),
}

TOWER_DAMAGE_MULTIPLIERS = {
    "fast": {"light": 0.90, "armored": 0.45, "flying": 1.75},
    "heavy": {"light": 0.60, "armored": 1.60, "flying": 0.70},
    "area": {"light": 1.45, "armored": 0.80, "flying": 0.45},
}

ENEMY_PROFILES = {
    "light": EnemyProfile(max_hp=2.30, speed=3.35, armor=0.00, bounty=0, radius=8.0),
    "armored": EnemyProfile(max_hp=10.00, speed=1.55, armor=0.95, bounty=0, radius=10.0),
    "flying": EnemyProfile(max_hp=3.20, speed=3.55, armor=0.15, bounty=0, radius=9.0),
}

WAVE_TEMPLATES = (
    WaveSpec(entry_mode=SIDE_LEFT, count_light=6, count_armored=0, count_flying=0),
    WaveSpec(entry_mode=SIDE_RIGHT, count_light=3, count_armored=0, count_flying=4),
    WaveSpec(entry_mode=ENTRY_BOTH, count_light=7, count_armored=2, count_flying=0),
    WaveSpec(entry_mode=SIDE_LEFT, count_light=2, count_armored=3, count_flying=2),
    WaveSpec(entry_mode=SIDE_RIGHT, count_light=6, count_armored=2, count_flying=3),
    WaveSpec(entry_mode=ENTRY_BOTH, count_light=8, count_armored=4, count_flying=2),
    WaveSpec(entry_mode=SIDE_LEFT, count_light=4, count_armored=6, count_flying=4),
    WaveSpec(entry_mode=ENTRY_BOTH, count_light=10, count_armored=5, count_flying=5),
)

TOWER_COLORS = {
    "fast": (COLOR_BLUE, COLOR_NAVY),
    "heavy": (COLOR_SAND, COLOR_OCHRE),
    "area": (COLOR_PURPLE, COLOR_DEEP_PURPLE),
}
ENEMY_COLORS = {
    "light": (COLOR_PURPLE, COLOR_DEEP_PURPLE),
    "armored": (COLOR_SAND, COLOR_OCHRE),
    "flying": (COLOR_BLUE, COLOR_NAVY),
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
        load_font_once(resolve_font_path(INTER_FONT_FILE))

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
        self._session_seed = (
            int(config.BASE_SEED)
            if self.mode == "train"
            else int(random.SystemRandom().randrange(1 << 61))
        )

        self.layout: LayoutSpec = _generate_layout(random.Random(int(self._session_seed)))
        self.wave_plan: list[WaveSpec] = []
        self.wave_index = 0
        self.credits = 0
        self.lives = 0
        self.decision_budget = 0
        self.actions_remaining = 0
        self.slot_towers: dict[str, TowerState | None] = {slot_name: None for slot_name in config.SLOT_NAMES}
        self._menu_slot_name: str | None = None
        self._hovered_slot_name: str | None = None
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
        return int(self._session_seed + self._episode_counter * 7919 + self._current_level * 101)

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
                    count_light=int(counts["light"]),
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
            "run_gold_norm": clip_unit(float(self.credits) / float(config.MAX_CREDITS_NORMALIZER)),
            "run_lives_norm": clip_unit(float(self.lives) / float(config.MAX_LIVES_NORMALIZER)),
            "run_wave_norm": float(self._wave_number_norm()),
            "run_actions_left_norm": clip_unit(
                float(self.actions_remaining) / float(config.DECISION_BUDGET_NORMALIZER)
            ),
            "wave_entry_left": 1.0 if wave is not None and wave.entry_mode in {SIDE_LEFT, ENTRY_BOTH} else 0.0,
            "wave_entry_right": 1.0 if wave is not None and wave.entry_mode in {SIDE_RIGHT, ENTRY_BOTH} else 0.0,
            "wave_count_light_norm": clip_unit(
                float(0 if wave is None else wave.count_light) / float(config.MAX_WAVE_COUNT_NORMALIZER)
            ),
            "wave_count_armored_norm": clip_unit(
                float(0 if wave is None else wave.count_armored) / float(config.MAX_WAVE_COUNT_NORMALIZER)
            ),
            "wave_count_flying_norm": clip_unit(
                float(0 if wave is None else wave.count_flying) / float(config.MAX_WAVE_COUNT_NORMALIZER)
            ),
            "map_layout_id_norm": float(self.layout.layout_norm),
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
        self.layout = _generate_layout(rng)
        self.wave_plan = self._build_wave_plan(rng)
        self.wave_index = 0
        self.credits = int(self._level_settings["start_credits"])
        self.lives = int(self._level_settings["start_lives"])
        self.actions_remaining = int(self.decision_budget)
        self.slot_towers = {slot_name: None for slot_name in config.SLOT_NAMES}
        self._menu_slot_name = None
        self._hovered_slot_name = None
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
                    if int(self.credits) >= build_cost:
                        mask[self._build_action_index(str(tower_kind), str(slot_name))] = True
                continue

            if int(tower_state.level) < 3:
                upgrade_cost = int(tower_state.profile.upgrade_cost_for_level(int(tower_state.level)))
                if int(self.credits) >= upgrade_cost:
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

    def _sell_value(self, slot_name: str) -> int:
        tower_state = self.slot_towers[str(slot_name)]
        if tower_state is None:
            return 0
        refund_ratio = float(SELL_REFUND_BY_LEVEL.get(int(tower_state.level), 0.0))
        return int(round(float(tower_state.total_spent) * float(refund_ratio)))

    def _slot_name_at(self, mouse_x: float, mouse_y_arcade: float) -> str | None:
        mouse_y = float(self.window_controller.to_top_left_y(float(mouse_y_arcade)))
        for slot_name, (center_x, center_y) in self.layout.slot_positions.items():
            if (
                abs(float(mouse_x) - float(center_x)) <= float(SLOT_SIZE) * 0.5
                and abs(float(mouse_y) - float(center_y)) <= float(SLOT_SIZE) * 0.5
            ):
                return str(slot_name)
        return None

    def _update_hovered_slot(self) -> None:
        mouse_position = self.window_controller.mouse_position()
        if mouse_position is None:
            self._hovered_slot_name = None
            return
        self._hovered_slot_name = self._slot_name_at(float(mouse_position[0]), float(mouse_position[1]))

    def _menu_items_for_slot(self, slot_name: str) -> list[tuple[str, int, bool]]:
        mask = self.get_action_mask()
        tower_state = self.slot_towers[str(slot_name)]
        if tower_state is None:
            items = [
                ("Fast", self._build_action_index("fast", str(slot_name))),
                ("Heavy", self._build_action_index("heavy", str(slot_name))),
                ("Area", self._build_action_index("area", str(slot_name))),
            ]
        else:
            items = [
                ("Upgrade", self._upgrade_action_index(str(slot_name))),
                ("Sell", self._sell_action_index(str(slot_name))),
            ]
        return [(label, int(action_idx), bool(mask[int(action_idx)])) for label, action_idx in items]

    def _menu_item_rects(self, slot_name: str) -> list[tuple[str, int, bool, tuple[float, float, float, float]]]:
        items = self._menu_items_for_slot(str(slot_name))
        if not items:
            return []
        slot_left = float(self.layout.slot_cells[str(slot_name)][0]) * TILE_SIZE
        slot_top = float(self.layout.slot_cells[str(slot_name)][1]) * TILE_SIZE
        total_height = float(len(items)) * MENU_ITEM_HEIGHT + float(max(0, len(items) - 1)) * MENU_ITEM_GAP
        if slot_left + SLOT_SIZE + 6.0 + MENU_ITEM_WIDTH <= float(config.SCREEN_WIDTH) - 4.0:
            menu_left = slot_left + SLOT_SIZE + 6.0
        else:
            menu_left = slot_left - MENU_ITEM_WIDTH - 6.0
        menu_top = min(max(4.0, slot_top), float(config.WORLD_HEIGHT) - total_height - 4.0)
        rects: list[tuple[str, int, bool, tuple[float, float, float, float]]] = []
        for item_index, (label, action_idx, enabled) in enumerate(items):
            top = menu_top + float(item_index) * (MENU_ITEM_HEIGHT + MENU_ITEM_GAP)
            rects.append((label, int(action_idx), bool(enabled), (menu_left, top, MENU_ITEM_WIDTH, MENU_ITEM_HEIGHT)))
        return rects

    def _menu_action_at(self, mouse_x: float, mouse_y_arcade: float) -> tuple[int, bool] | None:
        if self._menu_slot_name is None:
            return None
        mouse_y = float(self.window_controller.to_top_left_y(float(mouse_y_arcade)))
        for _, action_idx, enabled, rect in self._menu_item_rects(str(self._menu_slot_name)):
            left, top, width, height = rect
            if left <= float(mouse_x) <= left + width and top <= float(mouse_y) <= top + height:
                return int(action_idx), bool(enabled)
        return None

    def _human_action(self) -> int | None:
        self._update_hovered_slot()

        for key in self.window_controller.consume_key_presses():
            if key == arcade.key.SPACE:
                self._menu_slot_name = None
                return 0

        for mouse_press in self.window_controller.consume_mouse_presses():
            if int(mouse_press.button) != int(arcade.MOUSE_BUTTON_LEFT):
                continue
            menu_action = self._menu_action_at(float(mouse_press.x), float(mouse_press.y))
            if menu_action is not None:
                action_idx, enabled = menu_action
                if enabled:
                    self._menu_slot_name = None
                    return int(action_idx)
                return None

            slot_name = self._slot_name_at(float(mouse_press.x), float(mouse_press.y))
            if slot_name is None:
                self._menu_slot_name = None
                continue
            self._menu_slot_name = str(slot_name)
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
            lane_offset=0.0,
            distance_along=0.0,
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
        for enemy_kind in ("light", "flying", "armored"):
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

    def _attack_with_fast(self, slot_name: str, tower_state: TowerState, enemies: list[EnemyState]) -> None:
        slot_pos = self.layout.slot_positions[str(slot_name)]
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
        self._damage_enemy(target, tower_kind="fast", tower_stats=tower_state.stats)
        tower_state.cooldown_ticks = int(tower_state.stats.cooldown_ticks)
        self._attack_effects.append(
            AttackEffect(
                kind="fast",
                points=(slot_pos, target.position),
                ttl=4,
                max_ttl=4,
            )
        )

    def _attack_with_heavy(self, slot_name: str, tower_state: TowerState, enemies: list[EnemyState]) -> None:
        slot_pos = self.layout.slot_positions[str(slot_name)]
        candidates = [
            enemy
            for enemy in enemies
            if enemy.alive
            and self._in_range(slot_pos, enemy, tower_state.stats.attack_range)
        ]
        if not candidates:
            return
        target = max(
            candidates,
            key=lambda enemy: self._enemy_priority(enemy, preferred_kind="armored"),
        )
        self._damage_enemy(target, tower_kind="heavy", tower_stats=tower_state.stats)
        tower_state.cooldown_ticks = int(tower_state.stats.cooldown_ticks)
        self._attack_effects.append(
            AttackEffect(
                kind="heavy",
                points=(slot_pos, target.position),
                ttl=4,
                max_ttl=4,
            )
        )

    def _attack_with_area(self, slot_name: str, tower_state: TowerState, enemies: list[EnemyState]) -> None:
        slot_pos = self.layout.slot_positions[str(slot_name)]
        candidates = [
            enemy
            for enemy in enemies
            if enemy.alive and self._in_range(slot_pos, enemy, tower_state.stats.attack_range)
        ]
        if not candidates:
            return
        target = max(
            candidates,
            key=lambda enemy: self._enemy_priority(enemy, preferred_kind="light"),
        )
        blast_center = target.position
        for enemy in enemies:
            if not enemy.alive:
                continue
            distance_to_blast = _distance(blast_center, enemy.position)
            if distance_to_blast > float(tower_state.stats.splash_radius) + float(enemy.radius):
                continue
            splash_scale = 1.0 if enemy is target else 0.70
            self._damage_enemy(enemy, tower_kind="area", tower_stats=tower_state.stats, scale=float(splash_scale))
        tower_state.cooldown_ticks = int(tower_state.stats.cooldown_ticks)
        self._attack_effects.append(
            AttackEffect(
                kind="area_flight",
                points=(slot_pos, blast_center),
                ttl=4,
                max_ttl=4,
                radius=5.0,
            )
        )
        self._attack_effects.append(
            AttackEffect(
                kind="area_splash",
                points=(blast_center,),
                ttl=9,
                max_ttl=9,
                delay_ticks=3,
                radius=float(tower_state.stats.splash_radius),
            )
        )

    def _tick_towers(self, enemies: list[EnemyState]) -> None:
        for slot_name in config.SLOT_NAMES:
            tower_state = self.slot_towers[str(slot_name)]
            if tower_state is None:
                continue
            if int(tower_state.cooldown_ticks) > 0:
                tower_state.cooldown_ticks = max(0, int(tower_state.cooldown_ticks) - 1)
                continue
            if str(tower_state.kind) == "fast":
                self._attack_with_fast(str(slot_name), tower_state, enemies)
            elif str(tower_state.kind) == "heavy":
                self._attack_with_heavy(str(slot_name), tower_state, enemies)
            else:
                self._attack_with_area(str(slot_name), tower_state, enemies)

    def _collect_kills(self, enemies: list[EnemyState], reward_breakdown: dict[str, float]) -> None:
        for enemy in enemies:
            if enemy.alive or enemy.leaked:
                continue
            reward_breakdown["reward_progress_kill"] += float(config.REWARD_PROGRESS_KILL)
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
            if int(effect.delay_ticks) > 0:
                effect.delay_ticks = max(0, int(effect.delay_ticks) - 1)
                next_effects.append(effect)
                continue
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
                self.credits += int(config.WAVE_CLEAR_CREDIT_BONUS)
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
        if self.slot_towers[str(slot_name)] is not None or int(self.credits) < int(profile.build_cost):
            return
        self.credits -= int(profile.build_cost)
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
        if int(self.credits) < int(upgrade_cost):
            return
        self.credits -= int(upgrade_cost)
        tower_state.level = min(3, int(tower_state.level) + 1)
        tower_state.total_spent += int(upgrade_cost)
        tower_state.cooldown_ticks = 0
        self.actions_remaining = max(0, int(self.actions_remaining) - 1)

    def _sell_tower(self, slot_name: str) -> None:
        tower_state = self.slot_towers[str(slot_name)]
        if tower_state is None:
            return
        refund = self._sell_value(str(slot_name))
        self.credits += int(refund)
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
        for col, row in path_cells:
            left, top, width, height = _block_rect(int(col), int(row), 1, 1)
            arcade.draw_lbwh_rectangle_filled(
                float(left),
                float(self.window_controller.to_arcade_y(float(top) + float(height))),
                float(width),
                float(height),
                GROUND_PATH_INNER,
            )
        draw_cell_union_outline(
            self.window_controller,
            cells=path_cells,
            top_left_x=0.0,
            top_left_y=0.0,
            cell_size=float(TILE_SIZE),
            border_width=float(CELL_INSET),
            color=GROUND_PATH_OUTER,
        )

    def _draw_entries_and_exit(self) -> None:
        for side, cell_top_left in self.layout.entry_cells.items():
            label = "L" if side == SIDE_LEFT else "R"
            marker_left = int(cell_top_left[0]) - int(ENTRY_MARKER_DEPTH_CELLS) if side == SIDE_LEFT else int(cell_top_left[0]) + int(TRACK_WIDTH_CELLS)
            marker_top = int(cell_top_left[1])
            marker_rect = _block_rect(
                int(marker_left),
                int(marker_top),
                int(ENTRY_MARKER_DEPTH_CELLS),
                int(TRACK_WIDTH_CELLS),
            )
            label_x = float(marker_rect[0]) + float(marker_rect[2]) * 0.5
            label_y = float(marker_rect[1]) + float(marker_rect[3]) * 0.5
            self._draw_block(
                *marker_rect,
                outer_color=ENTRY_OUTLINE,
                inner_color=ENTRY_FILL,
            )
            self._text_cache.draw(
                label,
                x=float(label_x),
                y=float(self.window_controller.to_arcade_y(float(label_y))),
                color=COLOR_LIGHT_NEUTRAL,
                font_size=12,
                font_name=BLOCK_FONT_NAME,
                anchor_x="center",
                anchor_y="center",
            )
        exit_rect = _block_rect(
            int(self.layout.exit_cell[0]),
            int(self.layout.exit_cell[1]) + int(TRACK_WIDTH_CELLS),
            int(TRACK_WIDTH_CELLS),
            int(EXIT_MARKER_DEPTH_CELLS),
        )
        exit_label_x = float(exit_rect[0]) + float(exit_rect[2]) * 0.5
        exit_label_y = float(exit_rect[1]) + float(exit_rect[3]) * 0.5
        self._draw_block(
            *exit_rect,
            outer_color=EXIT_OUTLINE,
            inner_color=EXIT_FILL,
        )
        self._text_cache.draw(
            "E",
            x=float(exit_label_x),
            y=float(self.window_controller.to_arcade_y(float(exit_label_y))),
            color=COLOR_LIGHT_NEUTRAL,
            font_size=12,
            font_name=BLOCK_FONT_NAME,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_slots(self) -> None:
        tower_labels = {"fast": "F", "heavy": "H", "area": "A"}
        for slot_name in config.SLOT_NAMES:
            x, y = self.layout.slot_positions[str(slot_name)]
            tower_state = self.slot_towers[str(slot_name)]
            highlighted = str(slot_name) in {str(self._hovered_slot_name), str(self._menu_slot_name)}
            tower_outline, tower_fill = (SLOT_OUTLINE, SLOT_FILL)
            if tower_state is not None:
                tower_outline, tower_fill = TOWER_COLORS[str(tower_state.kind)]
            if tower_state is None:
                tower_fill = EMPTY_SLOT_FILL
            if highlighted and self.mode == "human":
                tower_outline = SELECTED_SLOT_COLOR
            self._draw_block(
                *_block_rect(
                    int(self.layout.slot_cells[str(slot_name)][0]),
                    int(self.layout.slot_cells[str(slot_name)][1]),
                    2,
                    2,
                ),
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
                font_size=12,
                font_name=BLOCK_FONT_NAME,
                anchor_x="center",
                anchor_y="center",
            )

    def _draw_context_menu(self) -> None:
        if self.mode != "human" or self._menu_slot_name is None or self._wave_in_progress or self._last_outcome:
            return
        for label, _, enabled, rect in self._menu_item_rects(str(self._menu_slot_name)):
            left, top, width, height = rect
            arcade.draw_lbwh_rectangle_filled(
                float(left),
                float(self.window_controller.to_arcade_y(float(top) + float(height))),
                float(width),
                float(height),
                MENU_FILL,
            )
            arcade.draw_lbwh_rectangle_outline(
                float(left),
                float(self.window_controller.to_arcade_y(float(top) + float(height))),
                float(width),
                float(height),
                MENU_BORDER,
                1.0,
            )
            self._text_cache.draw(
                str(label),
                x=float(left) + 6.0,
                y=float(self.window_controller.to_arcade_y(float(top) + float(height) * 0.5)),
                color=MENU_TEXT if enabled else MENU_TEXT_DISABLED,
                font_size=12,
                font_name=UI_FONT_NAME,
                anchor_x="left",
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
            if int(effect.delay_ticks) > 0 or not effect.points:
                continue
            outer_color, _ = TOWER_COLORS.get(str(effect.kind).replace("_flight", "").replace("_splash", ""), (COLOR_LIGHT_NEUTRAL, COLOR_LIGHT_NEUTRAL))
            max_ttl = max(1, int(effect.max_ttl) if int(effect.max_ttl) > 0 else int(effect.ttl))
            progress = 1.0 - (float(effect.ttl) / float(max_ttl))
            if not effect.points:
                continue
            screen_points = [
                (float(x), float(self.window_controller.to_arcade_y(float(y))))
                for x, y in effect.points
            ]
            if str(effect.kind) == "fast" and len(screen_points) >= 2:
                start_x, start_y = screen_points[0]
                end_x, end_y = screen_points[1]
                shot_x = float(start_x + (end_x - start_x) * progress)
                shot_y = float(start_y + (end_y - start_y) * progress)
                arcade.draw_circle_filled(shot_x, shot_y, 5.0, outer_color)
            elif str(effect.kind) == "heavy" and len(screen_points) >= 2:
                start_x, start_y = screen_points[0]
                end_x, end_y = screen_points[1]
                arcade.draw_line(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    outer_color + (64,),
                    5.0,
                )
            elif str(effect.kind) == "area_flight" and len(screen_points) >= 2:
                start_x, start_y = screen_points[0]
                end_x, end_y = screen_points[1]
                shot_x = float(start_x + (end_x - start_x) * progress)
                shot_y = float(start_y + (end_y - start_y) * progress)
                arcade.draw_circle_filled(shot_x, shot_y, 5.0, outer_color + (128,))
            elif str(effect.kind) == "area_splash" and screen_points:
                center_x, center_y = screen_points[0]
                splash_radius = max(2.0, float(effect.radius) * progress)
                splash_color = outer_color + (128,)
                arcade.draw_circle_filled(center_x, center_y, splash_radius, splash_color)
                arcade.draw_circle_outline(
                    center_x,
                    center_y,
                    splash_radius,
                    splash_color,
                    2.0,
                )

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
        self._draw_context_menu()

    def _draw_hud(self) -> None:
        wave = self._current_wave()
        wave_label = "Done" if wave is None else f"{int(self.wave_index) + 1}/{len(self.wave_plan)}"
        preview_entry = "-"
        if wave is not None:
            preview_entry = {
                SIDE_LEFT: "Left",
                SIDE_RIGHT: "Right",
                ENTRY_BOTH: "Both",
            }.get(str(wave.entry_mode), str(wave.entry_mode).title())
        entries: list[tuple[str, object]] = [
            ("Credits", int(self.credits)),
            ("Lives", int(self.lives)),
            ("Wave", wave_label),
            ("NextWave", preview_entry),
            ("Light", 0 if wave is None else int(wave.count_light)),
            ("Armored", 0 if wave is None else int(wave.count_armored)),
            ("Flying", 0 if wave is None else int(wave.count_flying)),
        ]
        if self._last_outcome:
            entries.append(("Status", "Victory" if self._last_outcome == "win" else "Defeat"))
        draw_status_bar(
            width=float(config.SCREEN_WIDTH),
            bottom_bar_height=float(config.BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=float(CELL_INSET),
            left_panel_width=max(0.0, float(config.SCREEN_WIDTH) - 16.0),
            include_clock=False,
            text_cache=self._text_cache,
            left_text_entries=entries,
            text_color=COLOR_LIGHT_NEUTRAL,
        )

    def render(self) -> None:
        if self.window_controller.window is None:
            return
        if self.mode == "human":
            self._update_hovered_slot()
        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        self._draw_world()
        self._draw_hud()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
