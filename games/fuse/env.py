"""Fuse environment: compact bomb-duel / free-for-all survival."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import random

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BARK,
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
    COLOR_SLATE_GRAY,
    COLOR_WALNUT,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
)
from core.envs.base import Env
from core.io_schema import clip_signed, clip_unit, ordered_feature_vector
from core.match_tracker import MatchTracker
from core.primitives import (
    draw_status_bar,
    draw_status_clock,
    draw_status_icon_row,
    draw_status_square_icon,
    draw_two_tone_square_block,
    draw_two_tone_tile,
    status_icon_inset,
)
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level, validate_level_settings
from games.fuse.config import (
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_STOP,
    ACTION_MOVE_UP,
    ACTION_NAMES as FUSE_ACTION_NAMES,
    ACTION_PLACE_BOMB,
    ACT_DIM as FUSE_ACT_DIM,
    BB_HEIGHT,
    BLAST_RADIUS_TILES,
    BOARD_HEIGHT_TILES,
    BOARD_WIDTH_TILES,
    BOMB_FUSE_STEPS,
    CELL_INSET,
    CURRICULUM_PROMOTION,
    DEFAULT_HIDDEN_SIZES,
    ESCAPE_ROUTE_NORMALIZER,
    EXPLOSION_LIFETIME_STEPS,
    FPS,
    INPUT_FEATURE_NAMES as FUSE_INPUT_FEATURE_NAMES,
    LEVEL_SETTINGS,
    MAX_BOMBS_PER_PLAYER,
    MAX_EPISODE_STEPS,
    MAX_LEVEL,
    MIN_LEVEL,
    OBS_DIM as FUSE_OBS_DIM,
    PENALTY_LOSE,
    PENALTY_STEP,
    REWARD_CRATE,
    REWARD_ELIM,
    REWARD_WIN,
    SAFE_SEARCH_HORIZON_STEPS,
    SAFE_SPACE_NORMALIZER,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SENSE_RANGE_TILES,
    TILE_SIZE,
    TRAINING_FPS,
    WINDOW_TITLE,
    WIN_HISTORY_LIMIT,
)


ALL_PLAYER_ORDER = ("P1", "P2", "P3", "P4")
SUPPORTED_PLAYER_COUNTS = (2, 3, 4)
PLAYER_ORDER_INDEX = {player_id: idx for idx, player_id in enumerate(ALL_PLAYER_ORDER)}
DIRECTION_BY_ACTION = {
    ACTION_MOVE_UP: (0, -1),
    ACTION_MOVE_DOWN: (0, 1),
    ACTION_MOVE_LEFT: (-1, 0),
    ACTION_MOVE_RIGHT: (1, 0),
}
ORTHOGONAL_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
SCRIPTED_TARGET_STICKINESS_DISTANCE = 2
SCRIPTED_MOVE_COMMIT_STEPS = 4
SCRIPTED_RECENT_CELL_MEMORY = 8
SCRIPTED_RECENT_REVISIT_PENALTY = 0.40
SCRIPTED_BOMB_ESCAPE_MIN = 0.20
validate_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
    valid_player_counts=SUPPORTED_PLAYER_COUNTS,
)


PLAYER_STYLES = {
    "P1": {
        "render_fill": COLOR_DEEP_TEAL,
        "render_outline": COLOR_AQUA,
        "scripted": False,
    },
    "P2": {
        "render_fill": COLOR_BRICK_RED,
        "render_outline": COLOR_CORAL,
        "scripted": True,
    },
    "P3": {
        "render_fill": COLOR_NAVY,
        "render_outline": COLOR_BLUE,
        "scripted": True,
    },
    "P4": {
        "render_fill": COLOR_DEEP_PURPLE,
        "render_outline": COLOR_PURPLE,
        "scripted": True,
    },
}
HUMAN_MOVE_BINDINGS = (
    (ACTION_MOVE_UP, (arcade.key.W, arcade.key.UP)),
    (ACTION_MOVE_DOWN, (arcade.key.S, arcade.key.DOWN)),
    (ACTION_MOVE_LEFT, (arcade.key.A, arcade.key.LEFT)),
    (ACTION_MOVE_RIGHT, (arcade.key.D, arcade.key.RIGHT)),
)


def _resolve_player_order(num_players: int) -> tuple[str, ...]:
    count = int(num_players)
    if count not in SUPPORTED_PLAYER_COUNTS:
        raise ValueError(f"num_players must be one of {SUPPORTED_PLAYER_COUNTS}, got {count}")
    return ALL_PLAYER_ORDER[:count]


@dataclass(frozen=True, slots=True)
class Cell:
    x: int
    y: int


@dataclass
class PlayerState:
    player_id: str
    cell: Cell
    alive: bool = True
    bombs_available: int = MAX_BOMBS_PER_PLAYER
    pass_through_bomb_id: int | None = None
    last_action_index: int = ACTION_MOVE_STOP


@dataclass
class BombState:
    bomb_id: int
    owner_id: str
    cell: Cell
    fuse: int


@dataclass
class ExplosionState:
    owners_by_cell: dict[Cell, tuple[str, ...]] = field(default_factory=dict)
    ttl: int = EXPLOSION_LIFETIME_STEPS


@dataclass
class ScriptedPlayerState:
    target_id: str | None = None
    committed_action: int = ACTION_MOVE_STOP
    commit_steps_remaining: int = 0
    last_cell: Cell | None = None
    recent_cells: list[Cell] = field(default_factory=list)


class FuseEnv(Env):
    """Masked Bomberman-style free-for-all with scripted opponents."""

    INPUT_FEATURE_NAMES = tuple(FUSE_INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(FUSE_ACTION_NAMES)
    OBS_DIM = int(FUSE_OBS_DIM)
    ACT_DIM = int(FUSE_ACT_DIM)
    DEFAULT_HIDDEN_SIZES = tuple(DEFAULT_HIDDEN_SIZES)
    REWARD_COMPONENT_ORDER = ("W", "L", "E", "C", "S")
    REWARD_COMPONENT_KEY_TO_CODE = {
        "outcome.reward_win": "W",
        "outcome.penalty_lose": "L",
        "event.reward_elim": "E",
        "event.reward_crate": "C",
        "step.penalty_step": "S",
    }

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode).strip().lower()
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
            else resolve_play_level(level=level, min_level=MIN_LEVEL, max_level=MAX_LEVEL, default_level=MAX_LEVEL)
        )
        self._last_episode_level = int(self._current_level)
        self._last_episode_success = 0
        self._last_terminal_reward = 0.0

        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=False,
            vsync=False,
        )
        self.window = self.window_controller.window

        self.board_width = int(BOARD_WIDTH_TILES)
        self.board_height = int(BOARD_HEIGHT_TILES)
        self.tile_size = float(TILE_SIZE)
        self.max_episode_steps = int(MAX_EPISODE_STEPS)
        self.match_tracker = MatchTracker[str](
            history_limit=int(WIN_HISTORY_LIMIT),
            clock_duration_steps=int(self.max_episode_steps),
        )
        self.player_order: tuple[str, ...] = ()
        self.players_by_id: dict[str, PlayerState] = {}
        self.player: PlayerState | None = None
        self.solid_cells: set[Cell] = set()
        self.solid_block_origins: list[Cell] = []
        self.solid_block_cells: set[Cell] = set()
        self.crate_cells: set[Cell] = set()
        self.initial_crate_count = 1
        self.bombs: list[BombState] = []
        self.explosions: list[ExplosionState] = []
        self.scripted_states: dict[str, ScriptedPlayerState] = {}
        self.steps = 0
        self.done = False
        self.last_reward_breakdown = self._zero_reward_breakdown()
        self._episode_reward_components = RewardBreakdown(self.REWARD_COMPONENT_ORDER)
        self._bomb_key_was_down = False
        self._next_bomb_id = 1

        self.crate_density = 0.25
        self.bot_safety_weight = 1.0
        self.bot_bomb_weight = 1.0
        self.bot_trap_weight = 0.0
        self.bot_chase_weight = 0.0
        self.bot_random_action_prob = 0.0

        self._apply_level_settings(int(self._current_level))
        self.reset()

    def _zero_reward_breakdown(self) -> dict[str, float]:
        return {
            "outcome.reward_win": 0.0,
            "outcome.penalty_lose": 0.0,
            "event.reward_elim": 0.0,
            "event.reward_crate": 0.0,
            "step.penalty_step": 0.0,
        }

    def _apply_level_settings(self, level: int) -> None:
        settings = LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Fuse.")
        self._current_level = int(level)
        self.crate_density = float(settings.get("crate_density", 0.25))
        self.max_episode_steps = max(1, int(settings.get("max_episode_steps", MAX_EPISODE_STEPS)))
        self.bot_safety_weight = float(settings.get("bot_safety_weight", 1.0))
        self.bot_bomb_weight = float(settings.get("bot_bomb_weight", 1.0))
        self.bot_trap_weight = float(settings.get("bot_trap_weight", 0.0))
        self.bot_chase_weight = float(settings.get("bot_chase_weight", 0.0))
        self.bot_random_action_prob = float(settings.get("bot_random_action_prob", 0.0))
        self.match_tracker.set_clock_duration(int(self.max_episode_steps))
        self._set_player_count(int(settings["num_players"]))

    def _set_player_count(self, num_players: int) -> None:
        self.player_order = _resolve_player_order(int(num_players))
        self.match_tracker.set_competitors(self.player_order, preserve_existing=True)

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        self._setup_round()
        self._episode_reward_components.reset()
        self.last_reward_breakdown = self._zero_reward_breakdown()
        return self._obs()

    def _setup_round(self) -> None:
        self.solid_cells = self._build_solid_cells()
        self.solid_block_origins, self.solid_block_cells = self._resolve_solid_render_groups(self.solid_cells)
        spawn_positions = self._spawn_positions_by_player()
        safe_cells = self._spawn_safe_cells(spawn_positions.values())
        self.crate_cells = self._generate_crates(safe_cells)
        self.initial_crate_count = max(1, int(len(self.crate_cells)))
        self.players_by_id = {
            player_id: PlayerState(player_id=player_id, cell=spawn_positions[player_id])
            for player_id in self.player_order
        }
        self.player = self.players_by_id["P1"]
        self.scripted_states = {
            player_id: ScriptedPlayerState()
            for player_id in self.player_order
            if player_id != "P1"
        }
        self.bombs = []
        self.explosions = []
        self.steps = 0
        self.done = False
        self._last_terminal_reward = 0.0
        self._bomb_key_was_down = False
        self._next_bomb_id = 1

    def _spawn_positions_by_player(self) -> dict[str, Cell]:
        corners = (
            Cell(1, 1),
            Cell(self.board_width - 2, self.board_height - 2),
            Cell(1, self.board_height - 2),
            Cell(self.board_width - 2, 1),
        )
        return {
            player_id: corners[idx]
            for idx, player_id in enumerate(self.player_order)
        }

    def _spawn_safe_cells(self, spawn_cells: object) -> set[Cell]:
        safe_cells: set[Cell] = set()
        for spawn in tuple(spawn_cells):
            if not isinstance(spawn, Cell):
                continue
            step_x = 1 if int(spawn.x) <= int(self.board_width // 2) else -1
            step_y = 1 if int(spawn.y) <= int(self.board_height // 2) else -1
            candidates = (
                spawn,
                Cell(int(spawn.x) + step_x, int(spawn.y)),
                Cell(int(spawn.x) + 2 * step_x, int(spawn.y)),
                Cell(int(spawn.x), int(spawn.y) + step_y),
                Cell(int(spawn.x), int(spawn.y) + 2 * step_y),
                Cell(int(spawn.x) + step_x, int(spawn.y) + step_y),
            )
            for cell in candidates:
                if self._in_bounds(cell) and cell not in self.solid_cells:
                    safe_cells.add(cell)
        return safe_cells

    def _build_solid_cells(self) -> set[Cell]:
        cells: set[Cell] = set()
        for x in range(int(self.board_width)):
            cells.add(Cell(int(x), 0))
            cells.add(Cell(int(x), int(self.board_height) - 1))
        for y in range(int(self.board_height)):
            cells.add(Cell(0, int(y)))
            cells.add(Cell(int(self.board_width) - 1, int(y)))
        for top_left_x in range(2, int(self.board_width) - 1, 3):
            for top_left_y in range(2, int(self.board_height) - 1, 3):
                for dx in (0, 1):
                    for dy in (0, 1):
                        cell = Cell(int(top_left_x) + int(dx), int(top_left_y) + int(dy))
                        if (
                            0 < int(cell.x) < int(self.board_width) - 1
                            and 0 < int(cell.y) < int(self.board_height) - 1
                        ):
                            cells.add(cell)
        return cells

    def _resolve_solid_render_groups(self, solid_cells: set[Cell]) -> tuple[list[Cell], set[Cell]]:
        block_origins: list[Cell] = []
        block_cells: set[Cell] = set()
        for cell in sorted(solid_cells, key=lambda item: (int(item.y), int(item.x))):
            if cell in block_cells:
                continue
            if (
                int(cell.x) <= 0
                or int(cell.y) <= 0
                or int(cell.x) >= int(self.board_width) - 2
                or int(cell.y) >= int(self.board_height) - 2
            ):
                continue
            candidate_cells = {
                cell,
                Cell(int(cell.x) + 1, int(cell.y)),
                Cell(int(cell.x), int(cell.y) + 1),
                Cell(int(cell.x) + 1, int(cell.y) + 1),
            }
            if not candidate_cells.issubset(solid_cells):
                continue
            block_origins.append(cell)
            block_cells.update(candidate_cells)
        return block_origins, block_cells

    def _generate_crates(self, safe_cells: set[Cell]) -> set[Cell]:
        candidates = [
            Cell(x, y)
            for y in range(1, int(self.board_height) - 1)
            for x in range(1, int(self.board_width) - 1)
            if Cell(x, y) not in self.solid_cells and Cell(x, y) not in safe_cells
        ]
        crates = {
            cell
            for cell in candidates
            if random.random() < float(self.crate_density)
        }
        if not crates and candidates:
            crates.add(random.choice(candidates))
        return crates

    def _decode_action(self, action: object) -> int:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if int(values.size) <= 0:
            return int(ACTION_MOVE_STOP)
        if int(values.size) == int(self.ACT_DIM):
            return int(np.argmax(values))
        return int(np.clip(int(values[0]), 0, int(self.ACT_DIM) - 1))

    def _human_action(self) -> int:
        bomb_down = bool(self.window_controller.is_key_down(arcade.key.SPACE))
        if bomb_down and not bool(self._bomb_key_was_down):
            self._bomb_key_was_down = True
            return int(ACTION_PLACE_BOMB)
        self._bomb_key_was_down = bomb_down
        for action_idx, keys in HUMAN_MOVE_BINDINGS:
            if any(bool(self.window_controller.is_key_down(key)) for key in keys):
                return int(action_idx)
        return int(ACTION_MOVE_STOP)

    def get_action_mask(self, _obs: object | None = None) -> np.ndarray:
        mask = np.zeros((int(self.ACT_DIM),), dtype=np.bool_)
        player = self.player
        if player is None or not player.alive:
            mask[int(ACTION_MOVE_STOP)] = True
            return mask
        mask[int(ACTION_MOVE_STOP)] = True
        for action_idx, (dx, dy) in DIRECTION_BY_ACTION.items():
            if self._movement_target_for_player(player, int(dx), int(dy)) is not None:
                mask[int(action_idx)] = True
        mask[int(ACTION_PLACE_BOMB)] = bool(self._can_place_bomb(player))
        return mask

    def _alive_players(self) -> list[PlayerState]:
        return [
            self.players_by_id[player_id]
            for player_id in self.player_order
            if self.players_by_id[player_id].alive
        ]

    def _alive_opponents_for(self, player_id: str) -> list[PlayerState]:
        return [
            self.players_by_id[other_id]
            for other_id in self.player_order
            if other_id != str(player_id) and self.players_by_id[other_id].alive
        ]

    def _ordered_opponents_for(self, player_id: str) -> list[PlayerState]:
        player = self.players_by_id[str(player_id)]
        opponents = self._alive_opponents_for(str(player_id))
        opponents.sort(
            key=lambda other: (
                int(abs(int(player.cell.x) - int(other.cell.x)) + abs(int(player.cell.y) - int(other.cell.y))),
                int(PLAYER_ORDER_INDEX[other.player_id]),
            )
        )
        return opponents

    def _bomb_by_cell(self) -> dict[Cell, BombState]:
        return {bomb.cell: bomb for bomb in self.bombs}

    def _bomb_by_id(self) -> dict[int, BombState]:
        return {int(bomb.bomb_id): bomb for bomb in self.bombs}

    def _in_bounds(self, cell: Cell) -> bool:
        return 0 <= int(cell.x) < int(self.board_width) and 0 <= int(cell.y) < int(self.board_height)

    @staticmethod
    def _offset_cell(cell: Cell, dx: int, dy: int) -> Cell:
        return Cell(int(cell.x) + int(dx), int(cell.y) + int(dy))

    def _cell_is_board_blocked(self, cell: Cell) -> bool:
        return cell in self.solid_cells or cell in self.crate_cells

    def _cell_is_movement_blocked(self, cell: Cell, *, moving_player_id: str) -> bool:
        if not self._in_bounds(cell) or self._cell_is_board_blocked(cell):
            return True
        if cell in self._bomb_by_cell():
            return True
        for other_id, other in self.players_by_id.items():
            if other_id == str(moving_player_id) or not other.alive:
                continue
            if other.cell == cell:
                return True
        return False

    def _movement_target_for_player(self, player: PlayerState, dx: int, dy: int) -> Cell | None:
        target = self._offset_cell(player.cell, int(dx), int(dy))
        if self._cell_is_movement_blocked(target, moving_player_id=str(player.player_id)):
            return None
        return target

    def _can_place_bomb(self, player: PlayerState) -> bool:
        if not player.alive or int(player.bombs_available) <= 0:
            return False
        return player.cell not in self._bomb_by_cell()

    def _place_bombs(self, action_by_player: dict[str, int]) -> None:
        for player_id in self.player_order:
            player = self.players_by_id[player_id]
            if not player.alive:
                continue
            if int(action_by_player.get(player_id, ACTION_MOVE_STOP)) != int(ACTION_PLACE_BOMB):
                continue
            if not self._can_place_bomb(player):
                continue
            bomb = BombState(
                bomb_id=int(self._next_bomb_id),
                owner_id=str(player.player_id),
                cell=player.cell,
                fuse=int(BOMB_FUSE_STEPS),
            )
            self._next_bomb_id += 1
            self.bombs.append(bomb)
            player.bombs_available = max(0, int(player.bombs_available) - 1)
            player.pass_through_bomb_id = int(bomb.bomb_id)

    def _resolve_moves(self, action_by_player: dict[str, int]) -> None:
        raw_targets: dict[str, Cell] = {}
        for player_id in self.player_order:
            player = self.players_by_id[player_id]
            action_idx = int(action_by_player.get(player_id, ACTION_MOVE_STOP))
            player.last_action_index = int(action_idx)
            if not player.alive or action_idx not in DIRECTION_BY_ACTION:
                continue
            dx, dy = DIRECTION_BY_ACTION[int(action_idx)]
            target = self._offset_cell(player.cell, int(dx), int(dy))
            if not self._in_bounds(target) or self._cell_is_board_blocked(target) or target in self._bomb_by_cell():
                continue
            raw_targets[player_id] = target

        occupant_by_cell = {
            player.cell: player_id
            for player_id, player in self.players_by_id.items()
            if player.alive
        }
        valid_targets: dict[str, Cell] = {}
        for player_id, target in raw_targets.items():
            occupant_id = occupant_by_cell.get(target)
            if occupant_id is None or occupant_id == player_id:
                valid_targets[player_id] = target
                continue
            occupant_target = raw_targets.get(str(occupant_id))
            if occupant_target is None:
                continue
            if occupant_target == self.players_by_id[player_id].cell:
                continue
            valid_targets[player_id] = target

        target_counts = Counter(valid_targets.values())
        for player_id, target in valid_targets.items():
            if int(target_counts[target]) > 1:
                continue
            player = self.players_by_id[player_id]
            previous_cell = player.cell
            player.cell = target
            if player.pass_through_bomb_id is not None and target != previous_cell:
                player.pass_through_bomb_id = None

    def _active_explosion_owners(self) -> dict[Cell, set[str]]:
        combined: dict[Cell, set[str]] = {}
        for explosion in self.explosions:
            for cell, owners in dict(explosion.owners_by_cell).items():
                combined.setdefault(cell, set()).update(str(owner) for owner in owners)
        return combined

    def _apply_explosion_damage(self, owners_by_cell: dict[Cell, set[str]]) -> dict[str, int]:
        player_elims = 0
        for player_id in self.player_order:
            player = self.players_by_id[player_id]
            if not player.alive:
                continue
            owners = owners_by_cell.get(player.cell)
            if not owners:
                continue
            player.alive = False
            player.pass_through_bomb_id = None
            if str(player_id) != "P1" and "P1" in owners:
                player_elims += 1
        return {"player_elims": int(player_elims)}

    def _decay_explosions(self) -> None:
        remaining: list[ExplosionState] = []
        for explosion in self.explosions:
            explosion.ttl = max(0, int(explosion.ttl) - 1)
            if int(explosion.ttl) > 0:
                remaining.append(explosion)
        self.explosions = remaining

    def _trigger_bombs_from_cells(self, cells: object) -> None:
        trigger_cells = set(cell for cell in tuple(cells) if isinstance(cell, Cell))
        if not trigger_cells:
            return
        for bomb in self.bombs:
            if bomb.cell in trigger_cells:
                bomb.fuse = min(int(bomb.fuse), 0)

    def _blast_cells(self, origin: Cell, crates_snapshot: set[Cell]) -> list[Cell]:
        cells = [origin]
        for dx, dy in ORTHOGONAL_DIRS:
            for step in range(1, int(BLAST_RADIUS_TILES) + 1):
                cell = Cell(int(origin.x) + int(dx) * step, int(origin.y) + int(dy) * step)
                if not self._in_bounds(cell) or cell in self.solid_cells:
                    break
                cells.append(cell)
                if cell in crates_snapshot:
                    break
        return cells

    def _tick_bombs_and_explode(self) -> dict[str, int]:
        if not self.bombs:
            return {"player_elims": 0, "player_crates": 0}
        for bomb in self.bombs:
            bomb.fuse -= 1

        bombs_by_id = self._bomb_by_id()
        bombs_by_cell = self._bomb_by_cell()
        detonated_ids: set[int] = set()
        pending_ids = [int(bomb.bomb_id) for bomb in self.bombs if int(bomb.fuse) <= 0]
        if not pending_ids:
            return {"player_elims": 0, "player_crates": 0}

        crates_snapshot = set(self.crate_cells)
        destroyed_crates: dict[Cell, set[str]] = {}
        owners_by_cell: dict[Cell, set[str]] = {}
        while pending_ids:
            bomb_id = int(pending_ids.pop())
            if bomb_id in detonated_ids:
                continue
            bomb = bombs_by_id.get(int(bomb_id))
            if bomb is None:
                continue
            detonated_ids.add(int(bomb_id))
            for cell in self._blast_cells(bomb.cell, crates_snapshot):
                owners_by_cell.setdefault(cell, set()).add(str(bomb.owner_id))
                if cell in crates_snapshot:
                    destroyed_crates.setdefault(cell, set()).add(str(bomb.owner_id))
                chained_bomb = bombs_by_cell.get(cell)
                if chained_bomb is not None and int(chained_bomb.bomb_id) not in detonated_ids:
                    chained_bomb.fuse = min(int(chained_bomb.fuse), 0)
                    pending_ids.append(int(chained_bomb.bomb_id))

        if not owners_by_cell:
            return {"player_elims": 0, "player_crates": 0}

        self.bombs = [
            bomb
            for bomb in self.bombs
            if int(bomb.bomb_id) not in detonated_ids
        ]
        for bomb_id in tuple(detonated_ids):
            bomb = bombs_by_id.get(int(bomb_id))
            if bomb is None:
                continue
            owner = self.players_by_id.get(str(bomb.owner_id))
            if owner is not None:
                owner.bombs_available = min(int(MAX_BOMBS_PER_PLAYER), int(owner.bombs_available) + 1)
                if owner.pass_through_bomb_id == int(bomb_id):
                    owner.pass_through_bomb_id = None

        self.crate_cells.difference_update(destroyed_crates.keys())
        normalized_explosion = {
            cell: tuple(sorted(owners))
            for cell, owners in owners_by_cell.items()
        }
        self.explosions.append(
            ExplosionState(
                owners_by_cell=normalized_explosion,
                ttl=int(EXPLOSION_LIFETIME_STEPS),
            )
        )
        kill_events = self._apply_explosion_damage(
            {
                cell: set(owners)
                for cell, owners in normalized_explosion.items()
            }
        )
        player_crates = sum(1 for owners in destroyed_crates.values() if "P1" in owners)
        return {
            "player_elims": int(kill_events["player_elims"]),
            "player_crates": int(player_crates),
        }

    def _alive_count(self) -> int:
        return sum(1 for player in self.players_by_id.values() if player.alive)

    def _winner_id(self) -> str | None:
        alive = [player.player_id for player in self.players_by_id.values() if player.alive]
        if len(alive) == 1:
            return str(alive[0])
        return None

    def _record_round_result(self, winner_id: str | None) -> None:
        if winner_id is None:
            self.match_tracker.record_draw()
            return
        self.match_tracker.increment_score(str(winner_id))
        self.match_tracker.record_result(str(winner_id))

    @staticmethod
    def _normalized_dx(delta_x: int) -> float:
        return float(clip_signed(float(delta_x) / float(max(1, BOARD_WIDTH_TILES - 1))))

    @staticmethod
    def _normalized_dy(delta_y: int) -> float:
        return float(clip_signed(float(delta_y) / float(max(1, BOARD_HEIGHT_TILES - 1))))

    def _free_distance_norm(self, origin: Cell, dx: int, dy: int) -> float:
        distance = 0
        for step in range(1, int(SENSE_RANGE_TILES) + 1):
            cell = Cell(int(origin.x) + int(dx) * step, int(origin.y) + int(dy) * step)
            if self._cell_is_movement_blocked(cell, moving_player_id="P1"):
                break
            distance += 1
        return float(clip_unit(float(distance) / float(max(1, int(SENSE_RANGE_TILES)))))

    def _box_proximity_norm(self, origin: Cell, dx: int, dy: int) -> float:
        for step in range(1, int(SENSE_RANGE_TILES) + 1):
            cell = Cell(int(origin.x) + int(dx) * step, int(origin.y) + int(dy) * step)
            if not self._in_bounds(cell) or cell in self.solid_cells:
                break
            if cell in self.crate_cells:
                return float(clip_unit(1.0 - float(step - 1) / float(max(1, int(SENSE_RANGE_TILES)))))
        return 0.0

    def _player_bomb_cooldown_norm(self, player: PlayerState) -> float:
        if int(player.bombs_available) > 0:
            return 0.0
        own_fuses = [
            int(bomb.fuse)
            for bomb in self.bombs
            if str(bomb.owner_id) == str(player.player_id)
        ]
        if not own_fuses:
            return 0.0
        return float(clip_unit(float(max(0, min(own_fuses))) / float(max(1, int(BOMB_FUSE_STEPS)))))

    def _predict_hazard_times(self, extra_bombs: list[BombState] | None = None) -> dict[Cell, int]:
        copied_bombs = [
            BombState(
                bomb_id=int(bomb.bomb_id),
                owner_id=str(bomb.owner_id),
                cell=Cell(int(bomb.cell.x), int(bomb.cell.y)),
                fuse=int(bomb.fuse),
            )
            for bomb in self.bombs
        ]
        for extra in extra_bombs or []:
            copied_bombs.append(
                BombState(
                    bomb_id=int(extra.bomb_id),
                    owner_id=str(extra.owner_id),
                    cell=Cell(int(extra.cell.x), int(extra.cell.y)),
                    fuse=int(extra.fuse),
                )
            )

        impact_times: dict[Cell, int] = {}
        for cell in self._active_explosion_owners():
            impact_times.setdefault(cell, 0)

        if not copied_bombs:
            return impact_times

        bombs_by_id = {int(bomb.bomb_id): bomb for bomb in copied_bombs}
        bombs_by_cell = {bomb.cell: bomb for bomb in copied_bombs}
        crates_snapshot = set(self.crate_cells)
        detonated_ids: set[int] = set()
        current_time = 0
        while True:
            pending_ids = [
                int(bomb.bomb_id)
                for bomb in bombs_by_id.values()
                if int(bomb.bomb_id) not in detonated_ids and int(bomb.fuse) <= int(current_time)
            ]
            if not pending_ids:
                remaining_fuses = [
                    int(bomb.fuse)
                    for bomb in bombs_by_id.values()
                    if int(bomb.bomb_id) not in detonated_ids
                ]
                if not remaining_fuses:
                    break
                current_time = max(int(current_time) + 1, int(min(remaining_fuses)))
                continue

            destroyed_now: set[Cell] = set()
            while pending_ids:
                bomb_id = int(pending_ids.pop())
                if bomb_id in detonated_ids:
                    continue
                bomb = bombs_by_id.get(int(bomb_id))
                if bomb is None:
                    continue
                detonated_ids.add(int(bomb_id))
                for cell in self._blast_cells(bomb.cell, crates_snapshot):
                    impact_times[cell] = min(int(impact_times.get(cell, current_time)), int(current_time))
                    if cell in crates_snapshot:
                        destroyed_now.add(cell)
                    chained_bomb = bombs_by_cell.get(cell)
                    if chained_bomb is not None and int(chained_bomb.bomb_id) not in detonated_ids:
                        chained_bomb.fuse = min(int(chained_bomb.fuse), int(current_time))
                        pending_ids.append(int(chained_bomb.bomb_id))
            crates_snapshot.difference_update(destroyed_now)

        return impact_times

    @staticmethod
    def _cell_safe_at_time(cell: Cell, arrival_time: int, hazard_times: dict[Cell, int]) -> bool:
        impact_time = hazard_times.get(cell)
        return impact_time is None or int(impact_time) > int(arrival_time)

    def _blocked_cells_for_reachability(self, ignore_player_id: str, extra_blocked: set[Cell] | None = None) -> set[Cell]:
        blocked = set(self.solid_cells)
        blocked.update(self.crate_cells)
        blocked.update(bomb.cell for bomb in self.bombs)
        blocked.update(
            player.cell
            for player_id, player in self.players_by_id.items()
            if player.alive and str(player_id) != str(ignore_player_id)
        )
        if extra_blocked:
            blocked.update(extra_blocked)
        return blocked

    def _reachable_safe_space_norm_from(
        self,
        player_id: str,
        start_cell: Cell,
        *,
        start_time: int,
        hazard_times: dict[Cell, int],
        extra_blocked: set[Cell] | None = None,
    ) -> float:
        blocked = self._blocked_cells_for_reachability(str(player_id), extra_blocked=extra_blocked)
        if start_cell in blocked:
            return 0.0
        if not self._cell_safe_at_time(start_cell, int(start_time), hazard_times):
            return 0.0

        visited = {(start_cell, int(start_time))}
        queue: deque[tuple[Cell, int]] = deque([(start_cell, int(start_time))])
        safe_cells = {start_cell}
        while queue:
            cell, time_value = queue.popleft()
            if int(time_value) >= int(SAFE_SEARCH_HORIZON_STEPS):
                continue
            for dx, dy in ORTHOGONAL_DIRS:
                next_cell = Cell(int(cell.x) + int(dx), int(cell.y) + int(dy))
                arrival = int(time_value) + 1
                if not self._in_bounds(next_cell) or next_cell in blocked:
                    continue
                if not self._cell_safe_at_time(next_cell, int(arrival), hazard_times):
                    continue
                state = (next_cell, int(arrival))
                if state in visited:
                    continue
                visited.add(state)
                queue.append(state)
                safe_cells.add(next_cell)
        return float(clip_unit(float(len(safe_cells)) / float(SAFE_SPACE_NORMALIZER)))

    def _post_bomb_escape_norm(self, player: PlayerState) -> float:
        if not self._can_place_bomb(player):
            return 0.0
        hypothetical_bomb = BombState(
            bomb_id=-1,
            owner_id=str(player.player_id),
            cell=player.cell,
            fuse=int(BOMB_FUSE_STEPS),
        )
        hazard_times = self._predict_hazard_times(extra_bombs=[hypothetical_bomb])
        blocked = self._blocked_cells_for_reachability(str(player.player_id), extra_blocked={player.cell})
        safe_terminal_cells: set[Cell] = set()
        queue: deque[tuple[Cell, int]] = deque([(player.cell, 0)])
        visited = {(player.cell, 0)}
        while queue:
            cell, time_value = queue.popleft()
            if int(time_value) >= int(BOMB_FUSE_STEPS):
                if self._cell_safe_at_time(cell, int(time_value), hazard_times):
                    safe_terminal_cells.add(cell)
                continue
            for dx, dy in (*ORTHOGONAL_DIRS, (0, 0)):
                next_cell = Cell(int(cell.x) + int(dx), int(cell.y) + int(dy))
                arrival = int(time_value) + 1
                if not self._in_bounds(next_cell):
                    continue
                if cell != player.cell and next_cell == player.cell:
                    continue
                if next_cell != player.cell and next_cell in blocked:
                    continue
                if not self._cell_safe_at_time(next_cell, int(arrival), hazard_times):
                    continue
                state = (next_cell, int(arrival))
                if state in visited:
                    continue
                visited.add(state)
                queue.append(state)
        return float(clip_unit(float(len(safe_terminal_cells)) / float(ESCAPE_ROUTE_NORMALIZER)))

    def _open_neighbor_ratio(self, cell: Cell, *, moving_player_id: str) -> float:
        open_count = 0
        for dx, dy in ORTHOGONAL_DIRS:
            neighbor = Cell(int(cell.x) + int(dx), int(cell.y) + int(dy))
            if not self._cell_is_movement_blocked(neighbor, moving_player_id=str(moving_player_id)):
                open_count += 1
        return float(clip_unit(float(open_count) / 4.0))

    def _adjacent_crate_pressure_norm(self, cell: Cell) -> float:
        crate_count = 0
        for dx, dy in ORTHOGONAL_DIRS:
            if self._offset_cell(cell, int(dx), int(dy)) in self.crate_cells:
                crate_count += 1
        return float(clip_unit(float(crate_count) / 4.0))

    def _scripted_state(self, player_id: str) -> ScriptedPlayerState:
        return self.scripted_states.setdefault(str(player_id), ScriptedPlayerState())

    def _remember_scripted_cell(self, state: ScriptedPlayerState, cell: Cell) -> None:
        if state.recent_cells and state.recent_cells[-1] == cell:
            return
        state.recent_cells.append(cell)
        if len(state.recent_cells) > int(SCRIPTED_RECENT_CELL_MEMORY):
            del state.recent_cells[:-int(SCRIPTED_RECENT_CELL_MEMORY)]

    def _select_scripted_target(self, player: PlayerState, state: ScriptedPlayerState) -> PlayerState | None:
        opponents = self._ordered_opponents_for(str(player.player_id))
        if not opponents:
            state.target_id = None
            return None
        current = next((opp for opp in opponents if opp.player_id == state.target_id), None)
        preferred = opponents[0]
        if current is not None:
            current_distance = self._manhattan_distance(player.cell, current.cell)
            preferred_distance = self._manhattan_distance(player.cell, preferred.cell)
            if int(current_distance) <= int(preferred_distance) + int(SCRIPTED_TARGET_STICKINESS_DISTANCE):
                state.target_id = str(current.player_id)
                return current
        state.target_id = str(preferred.player_id)
        return preferred

    @staticmethod
    def _clear_scripted_plan(state: ScriptedPlayerState) -> None:
        state.committed_action = int(ACTION_MOVE_STOP)
        state.commit_steps_remaining = 0

    def _score_scripted_stop_action(
        self,
        player: PlayerState,
        target: PlayerState | None,
        *,
        hazard_times: dict[Cell, int],
        urgent_escape: bool,
    ) -> float:
        safe_here = float(
            self._reachable_safe_space_norm_from(
                str(player.player_id),
                player.cell,
                start_time=0,
                hazard_times=hazard_times,
            )
        )
        score = 0.30 * float(self.bot_safety_weight) * float(safe_here)
        if target is not None and (int(player.cell.x) == int(target.cell.x) or int(player.cell.y) == int(target.cell.y)):
            score += 0.04
        if bool(urgent_escape):
            score -= 0.70
        return float(score)

    def _score_scripted_move_action(
        self,
        player: PlayerState,
        target: PlayerState | None,
        action_idx: int,
        *,
        hazard_times: dict[Cell, int],
        state: ScriptedPlayerState,
        urgent_escape: bool,
    ) -> float | None:
        dx, dy = DIRECTION_BY_ACTION[int(action_idx)]
        target_cell = self._movement_target_for_player(player, int(dx), int(dy))
        if target_cell is None:
            return None

        safe_space = float(
            self._reachable_safe_space_norm_from(
                str(player.player_id),
                target_cell,
                start_time=1,
                hazard_times=hazard_times,
            )
        )
        if bool(urgent_escape) and float(safe_space) <= 0.0:
            return None

        score = float(self.bot_safety_weight) * float(safe_space)
        score += 0.18 * float(self._open_neighbor_ratio(target_cell, moving_player_id=str(player.player_id)))
        if bool(urgent_escape):
            score += 0.45 * float(safe_space)
        else:
            score += 0.10 * float(self._adjacent_crate_pressure_norm(target_cell))
            score += 0.08 * float(self._nearest_opponent_distance_score(player, target_cell))

        if target is not None:
            current_distance = self._manhattan_distance(player.cell, target.cell)
            next_distance = self._manhattan_distance(target_cell, target.cell)
            score += (0.45 + float(self.bot_chase_weight)) * float(int(current_distance) - int(next_distance))
            if int(target_cell.x) == int(target.cell.x) or int(target_cell.y) == int(target.cell.y):
                score += 0.10

        if target_cell in state.recent_cells[:-1]:
            score -= float(SCRIPTED_RECENT_REVISIT_PENALTY)
        if int(state.committed_action) == int(action_idx) and int(state.commit_steps_remaining) > 0:
            score += 0.08
        return float(score)

    def _should_scripted_place_bomb(
        self,
        player: PlayerState,
        target: PlayerState | None,
        *,
        under_pressure: bool,
    ) -> bool:
        if not self._can_place_bomb(player):
            return False
        escape_norm = float(self._post_bomb_escape_norm(player))
        if float(escape_norm) < float(SCRIPTED_BOMB_ESCAPE_MIN):
            return False

        bomb_value, can_hit_now = self._bomb_value_metrics(player, escape_norm=escape_norm)
        adjacent_crates = float(self._adjacent_crate_pressure_norm(player.cell))
        bomb_score = (
            float(self.bot_bomb_weight) * float(bomb_value)
            + float(self.bot_trap_weight) * float(can_hit_now)
            + 0.30 * float(adjacent_crates)
            + 0.20 * float(escape_norm)
        )
        threshold = 0.42 if float(adjacent_crates) <= 0.0 else 0.28
        if float(can_hit_now) > 0.0:
            threshold = min(float(threshold), 0.24)
        if target is not None and self._manhattan_distance(player.cell, target.cell) <= int(BLAST_RADIUS_TILES) + 1:
            threshold -= 0.04
        if bool(under_pressure):
            threshold += 0.20
        return float(bomb_score) >= float(threshold)

    def _bomb_value_metrics(self, player: PlayerState, *, escape_norm: float | None = None) -> tuple[float, float]:
        if not self._can_place_bomb(player):
            return 0.0, 0.0
        self_escape = float(self._post_bomb_escape_norm(player) if escape_norm is None else escape_norm)
        hypothetical_bomb = BombState(
            bomb_id=-1,
            owner_id=str(player.player_id),
            cell=player.cell,
            fuse=int(BOMB_FUSE_STEPS),
        )
        hazard_times = self._predict_hazard_times(extra_bombs=[hypothetical_bomb])
        blast_cells = set(self._blast_cells(player.cell, set(self.crate_cells)))
        opponents = self._alive_opponents_for(str(player.player_id))
        direct_hits = [opp for opp in opponents if opp.cell in blast_cells]
        crates_hit = sum(1 for cell in blast_cells if cell in self.crate_cells)
        pressure_score = 0.0
        for opponent in opponents:
            opponent_escape = self._reachable_safe_space_norm_from(
                str(opponent.player_id),
                opponent.cell,
                start_time=0,
                hazard_times=hazard_times,
            )
            pressure_score += max(0.0, 1.0 - float(opponent_escape))
            if opponent.cell in blast_cells:
                pressure_score += 1.0
        raw_score = (0.22 * float(crates_hit)) + (0.50 * float(len(direct_hits))) + (0.28 * float(pressure_score))
        discounted_score = raw_score * (0.35 + 0.65 * float(self_escape))
        return float(clip_unit(discounted_score / 3.0)), float(1.0 if direct_hits else 0.0)

    def _observation_for_player(self, player: PlayerState) -> np.ndarray:
        hazard_times = self._predict_hazard_times()
        ordered_opponents = self._ordered_opponents_for(str(player.player_id))
        feature_values = {
            "self_bombs_norm": float(clip_unit(float(player.bombs_available) / float(max(1, int(MAX_BOMBS_PER_PLAYER))))),
            "self_bomb_cd_norm": float(self._player_bomb_cooldown_norm(player)),
            "self_on_bomb": float(1.0 if player.cell in self._bomb_by_cell() else 0.0),
            "self_can_place_bomb": float(1.0 if self._can_place_bomb(player) else 0.0),
            "sens_free_up_norm": float(self._free_distance_norm(player.cell, 0, -1)),
            "sens_free_down_norm": float(self._free_distance_norm(player.cell, 0, 1)),
            "sens_free_left_norm": float(self._free_distance_norm(player.cell, -1, 0)),
            "sens_free_right_norm": float(self._free_distance_norm(player.cell, 1, 0)),
            "sens_box_up_norm": float(self._box_proximity_norm(player.cell, 0, -1)),
            "sens_box_down_norm": float(self._box_proximity_norm(player.cell, 0, 1)),
            "sens_box_left_norm": float(self._box_proximity_norm(player.cell, -1, 0)),
            "sens_box_right_norm": float(self._box_proximity_norm(player.cell, 1, 0)),
            "opp1_dx": 0.0,
            "opp1_dy": 0.0,
            "opp1_same_row": 0.0,
            "opp1_same_col": 0.0,
            "opp2_dx": 0.0,
            "opp2_dy": 0.0,
            "opp2_same_row": 0.0,
            "opp2_same_col": 0.0,
            "opp3_dx": 0.0,
            "opp3_dy": 0.0,
            "opp3_same_row": 0.0,
            "opp3_same_col": 0.0,
            "map_safe_up_norm": 0.0,
            "map_safe_down_norm": 0.0,
            "map_safe_left_norm": 0.0,
            "map_safe_right_norm": 0.0,
            "haz_here_tti_norm": 0.0,
            "haz_post_bomb_escape_norm": 0.0,
            "flag_bomb_value_norm": 0.0,
            "flag_can_hit_opp_now": 0.0,
            "flag_crates_left_norm": float(clip_unit(float(len(self.crate_cells)) / float(max(1, self.initial_crate_count)))),
            "flag_time_norm": float(clip_unit(float(self.steps) / float(max(1, int(self.max_episode_steps))))),
        }

        for slot_index, opponent in enumerate(ordered_opponents[:3], start=1):
            delta_x = int(opponent.cell.x) - int(player.cell.x)
            delta_y = int(opponent.cell.y) - int(player.cell.y)
            feature_values[f"opp{slot_index}_dx"] = float(self._normalized_dx(delta_x))
            feature_values[f"opp{slot_index}_dy"] = float(self._normalized_dy(delta_y))
            feature_values[f"opp{slot_index}_same_row"] = float(1.0 if int(opponent.cell.y) == int(player.cell.y) else 0.0)
            feature_values[f"opp{slot_index}_same_col"] = float(1.0 if int(opponent.cell.x) == int(player.cell.x) else 0.0)

        for action_idx, feature_name in (
            (ACTION_MOVE_UP, "map_safe_up_norm"),
            (ACTION_MOVE_DOWN, "map_safe_down_norm"),
            (ACTION_MOVE_LEFT, "map_safe_left_norm"),
            (ACTION_MOVE_RIGHT, "map_safe_right_norm"),
        ):
            dx, dy = DIRECTION_BY_ACTION[int(action_idx)]
            target = self._movement_target_for_player(player, int(dx), int(dy))
            if target is None:
                feature_values[str(feature_name)] = 0.0
                continue
            feature_values[str(feature_name)] = float(
                self._reachable_safe_space_norm_from(
                    str(player.player_id),
                    target,
                    start_time=1,
                    hazard_times=hazard_times,
                )
            )

        hazard_here = hazard_times.get(player.cell)
        if hazard_here is not None:
            if int(hazard_here) <= 0:
                feature_values["haz_here_tti_norm"] = 1.0
            else:
                feature_values["haz_here_tti_norm"] = float(
                    clip_unit(
                        1.0
                        - (
                            float(hazard_here)
                            / float(max(1, int(BOMB_FUSE_STEPS) + int(EXPLOSION_LIFETIME_STEPS)))
                        )
                    )
                )

        post_bomb_escape = float(self._post_bomb_escape_norm(player))
        bomb_value, can_hit_now = self._bomb_value_metrics(player, escape_norm=post_bomb_escape)
        feature_values["haz_post_bomb_escape_norm"] = float(post_bomb_escape)
        feature_values["flag_bomb_value_norm"] = float(bomb_value)
        feature_values["flag_can_hit_opp_now"] = float(can_hit_now)

        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (int(self.OBS_DIM),):
            raise RuntimeError(f"Fuse observation expected {int(self.OBS_DIM)} features, got {obs.shape[0]}")
        if not np.isfinite(obs).all():
            raise RuntimeError("Fuse observation contains non-finite values.")
        return obs

    def _obs(self) -> np.ndarray:
        if self.player is None:
            return np.zeros((int(self.OBS_DIM),), dtype=np.float32)
        return self._observation_for_player(self.player)

    def _manhattan_distance(self, cell_a: Cell, cell_b: Cell) -> int:
        return int(abs(int(cell_a.x) - int(cell_b.x)) + abs(int(cell_a.y) - int(cell_b.y)))

    def _nearest_opponent_distance_score(self, player: PlayerState, cell: Cell) -> float:
        opponents = self._alive_opponents_for(str(player.player_id))
        if not opponents:
            return 0.0
        nearest = min(self._manhattan_distance(cell, opponent.cell) for opponent in opponents)
        max_dist = max(1, int(self.board_width) + int(self.board_height))
        return float(clip_unit(1.0 - (float(nearest) / float(max_dist))))

    def _select_scripted_action(self, player: PlayerState) -> int:
        state = self._scripted_state(str(player.player_id))
        if state.last_cell == player.cell and int(state.committed_action) in DIRECTION_BY_ACTION:
            self._clear_scripted_plan(state)
        state.last_cell = player.cell
        self._remember_scripted_cell(state, player.cell)

        legal_actions = np.flatnonzero(self._action_mask_for_scripted_player(player))
        if int(legal_actions.size) <= 0:
            return int(ACTION_MOVE_STOP)
        if random.random() < float(self.bot_random_action_prob):
            self._clear_scripted_plan(state)
            return int(random.choice(legal_actions.tolist()))

        hazard_times = self._predict_hazard_times()
        target = self._select_scripted_target(player, state)
        current_hazard = hazard_times.get(player.cell)
        under_pressure = current_hazard is not None and int(current_hazard) <= int(BOMB_FUSE_STEPS)
        legal_set = {int(action_idx) for action_idx in legal_actions.tolist()}

        if int(state.commit_steps_remaining) > 0 and int(state.committed_action) in DIRECTION_BY_ACTION:
            committed_action = int(state.committed_action)
            if committed_action in legal_set:
                committed_score = self._score_scripted_move_action(
                    player,
                    target,
                    int(committed_action),
                    hazard_times=hazard_times,
                    state=state,
                    urgent_escape=bool(under_pressure),
                )
                if committed_score is not None:
                    state.commit_steps_remaining = max(0, int(state.commit_steps_remaining) - 1)
                    return int(committed_action)
            self._clear_scripted_plan(state)

        if int(ACTION_PLACE_BOMB) in legal_set and self._should_scripted_place_bomb(
            player,
            target,
            under_pressure=bool(under_pressure),
        ):
            self._clear_scripted_plan(state)
            return int(ACTION_PLACE_BOMB)

        best_action = int(ACTION_MOVE_STOP)
        best_score = float(
            self._score_scripted_stop_action(
                player,
                target,
                hazard_times=hazard_times,
                urgent_escape=bool(under_pressure),
            )
        )
        for action_idx in DIRECTION_BY_ACTION:
            if int(action_idx) not in legal_set:
                continue
            score = self._score_scripted_move_action(
                player,
                target,
                int(action_idx),
                hazard_times=hazard_times,
                state=state,
                urgent_escape=bool(under_pressure),
            )
            if score is None:
                continue
            if float(score) > float(best_score):
                best_score = float(score)
                best_action = int(action_idx)

        if int(best_action) in DIRECTION_BY_ACTION:
            state.committed_action = int(best_action)
            state.commit_steps_remaining = max(
                0,
                int(SCRIPTED_MOVE_COMMIT_STEPS) - (1 if bool(under_pressure) else 0),
            )
        else:
            self._clear_scripted_plan(state)
        return int(best_action)

    def _action_mask_for_scripted_player(self, player: PlayerState) -> np.ndarray:
        mask = np.zeros((int(self.ACT_DIM),), dtype=np.bool_)
        if not player.alive:
            mask[int(ACTION_MOVE_STOP)] = True
            return mask
        mask[int(ACTION_MOVE_STOP)] = True
        for action_idx, (dx, dy) in DIRECTION_BY_ACTION.items():
            if self._movement_target_for_player(player, int(dx), int(dy)) is not None:
                mask[int(action_idx)] = True
        mask[int(ACTION_PLACE_BOMB)] = bool(self._can_place_bomb(player))
        return mask

    def _terminal_info(self, *, reward: float) -> dict[str, object]:
        return {
            "win": bool(self.player is not None and self.player.alive and self._winner_id() == "P1"),
            "level": int(self._last_episode_level),
            "success": int(self._last_episode_success),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "reward_components": self._episode_reward_components.totals(),
            "crates_left": int(len(self.crate_cells)),
            "player_count": int(len(self.player_order)),
            "reward": float(reward),
        }

    def step(self, action: object) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._obs(), 0.0, True, self._terminal_info(reward=self._last_terminal_reward)

        assert self.player is not None
        self.window_controller.poll_events_or_raise()
        episode_level = int(self._current_level)
        player_action = int(self._human_action() if self.mode == "human" else self._decode_action(action))
        action_by_player = {"P1": int(player_action)}
        for player_id in self.player_order:
            if player_id == "P1":
                continue
            opponent = self.players_by_id[player_id]
            if opponent.alive:
                action_by_player[player_id] = int(self._select_scripted_action(opponent))
            else:
                action_by_player[player_id] = int(ACTION_MOVE_STOP)

        self._place_bombs(action_by_player)
        self._resolve_moves(action_by_player)

        active_explosion_owners = self._active_explosion_owners()
        current_explosion_events = self._apply_explosion_damage(active_explosion_owners)
        self._trigger_bombs_from_cells(active_explosion_owners.keys())
        self._decay_explosions()
        bomb_events = self._tick_bombs_and_explode()
        self.steps += 1

        reward = float(PENALTY_STEP)
        reward_breakdown = self._zero_reward_breakdown()
        reward_breakdown["step.penalty_step"] = float(PENALTY_STEP)

        total_player_elims = int(current_explosion_events["player_elims"]) + int(bomb_events["player_elims"])
        if int(total_player_elims) > 0:
            elim_reward = float(REWARD_ELIM) * float(total_player_elims)
            reward += float(elim_reward)
            reward_breakdown["event.reward_elim"] = float(elim_reward)
        if int(bomb_events["player_crates"]) > 0:
            crate_reward = float(REWARD_CRATE) * float(bomb_events["player_crates"])
            reward += float(crate_reward)
            reward_breakdown["event.reward_crate"] = float(crate_reward)

        winner_id = self._winner_id()
        alive_count = self._alive_count()
        timed_out = int(self.steps) >= int(self.max_episode_steps)
        done = False
        win = False
        if not self.player.alive:
            done = True
        elif winner_id is not None:
            done = True
            win = str(winner_id) == "P1"
        elif int(alive_count) <= 0:
            done = True
        elif bool(timed_out):
            done = True

        if done:
            if bool(win):
                reward += float(REWARD_WIN)
                reward_breakdown["outcome.reward_win"] = float(REWARD_WIN)
            elif not self.player.alive:
                reward += float(PENALTY_LOSE)
                reward_breakdown["outcome.penalty_lose"] = float(PENALTY_LOSE)
            self._record_round_result(winner_id)

        self.last_reward_breakdown = dict(reward_breakdown)
        self._episode_reward_components.add_from_mapping(
            reward_breakdown,
            self.REWARD_COMPONENT_KEY_TO_CODE,
        )

        self.done = bool(done)
        if self.done:
            self._last_terminal_reward = float(reward)
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(1 if win else 0)
            next_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(1 if win else 0),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )
            self._current_level = int(next_level)
        else:
            level_changed = False

        obs = self._obs()
        info: dict[str, object] = {
            "win": bool(win),
            "success": int(1 if win and self.done else 0),
            "level": int(episode_level),
            "level_changed": bool(level_changed),
            "player_count": int(len(self.player_order)),
            "crates_left": int(len(self.crate_cells)),
            "reward_breakdown": dict(reward_breakdown),
        }
        if self.done:
            info["reward_components"] = self._episode_reward_components.totals()

        self.render()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)
        return obs, float(reward), bool(self.done), info

    def _cell_top_left(self, cell: Cell) -> tuple[float, float]:
        return float(int(cell.x) * int(TILE_SIZE)), float(int(cell.y) * int(TILE_SIZE))

    def _draw_board_tile(self, cell: Cell, outer_color, inner_color) -> None:
        top_left_x, top_left_y = self._cell_top_left(cell)
        draw_two_tone_tile(
            self.window_controller,
            top_left_x=float(top_left_x),
            top_left_y=float(top_left_y),
            size=float(TILE_SIZE),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(CELL_INSET),
        )

    def _draw_solid_cell(self, cell: Cell) -> None:
        self._draw_board_tile(cell, COLOR_FOG_GRAY, COLOR_SLATE_GRAY)

    def _draw_solid_block(self, top_left_cell: Cell, *, tiles_per_side: int = 2) -> None:
        top_left_x, top_left_y = self._cell_top_left(top_left_cell)
        draw_two_tone_square_block(
            self.window_controller,
            top_left_x=float(top_left_x),
            top_left_y=float(top_left_y),
            tile_size=float(TILE_SIZE),
            tiles_per_side=int(tiles_per_side),
            outer_color=COLOR_FOG_GRAY,
            inner_color=COLOR_SLATE_GRAY,
            inset=float(CELL_INSET),
        )

    def _draw_crate_cell(self, cell: Cell) -> None:
        self._draw_board_tile(cell, COLOR_WALNUT, COLOR_BARK)

    def _draw_bomb(self, bomb: BombState) -> None:
        top_left_x, top_left_y = self._cell_top_left(bomb.cell)
        owner_outline = PLAYER_STYLES[str(bomb.owner_id)]["render_outline"]
        draw_two_tone_tile(
            self.window_controller,
            top_left_x=float(top_left_x),
            top_left_y=float(top_left_y),
            size=float(TILE_SIZE),
            outer_color=COLOR_SLATE_GRAY,
            inner_color=COLOR_DARK_NEUTRAL,
            inset=float(CELL_INSET),
        )
        center_x = float(top_left_x) + 0.5 * float(TILE_SIZE)
        center_y = float(top_left_y) + 0.5 * float(TILE_SIZE)
        arcade.draw_circle_filled(
            float(center_x),
            float(self.window_controller.to_arcade_y(center_y)),
            max(2.0, 0.18 * float(TILE_SIZE)),
            owner_outline,
        )
        fuse_ratio = float(clip_unit(float(max(0, int(bomb.fuse))) / float(max(1, int(BOMB_FUSE_STEPS)))))
        fuse_height = max(2.0, round(0.15 * float(TILE_SIZE)))
        fuse_bottom = self.window_controller.top_left_to_bottom(
            float(top_left_y) + float(TILE_SIZE) - float(CELL_INSET) - float(fuse_height),
            float(fuse_height),
        )
        arcade.draw_lbwh_rectangle_filled(
            float(top_left_x) + float(CELL_INSET),
            float(fuse_bottom),
            max(2.0, (float(TILE_SIZE) - 2.0 * float(CELL_INSET)) * fuse_ratio),
            float(fuse_height),
            owner_outline,
        )

    def _draw_explosion_cell(self, cell: Cell) -> None:
        self._draw_board_tile(cell, COLOR_CORAL, COLOR_BRICK_RED)

    def _draw_player(self, player: PlayerState) -> None:
        if not player.alive:
            return
        style = PLAYER_STYLES[str(player.player_id)]
        top_left_x, top_left_y = self._cell_top_left(player.cell)
        self._draw_board_tile(player.cell, style["render_outline"], style["render_fill"])
        if str(player.player_id) == "P1":
            marker_size = max(4.0, float(TILE_SIZE) * 0.22)
            marker_bottom = self.window_controller.top_left_to_bottom(
                float(top_left_y) + 0.18 * float(TILE_SIZE),
                marker_size,
            )
            arcade.draw_lbwh_rectangle_filled(
                float(top_left_x) + 0.5 * float(TILE_SIZE) - 0.5 * marker_size,
                float(marker_bottom),
                float(marker_size),
                float(marker_size),
                COLOR_LIGHT_NEUTRAL,
            )

    def _draw_winner_icon(self, winner_id: str | None, center_x: float, center_y: float, size: float) -> None:
        inset = status_icon_inset(float(CELL_INSET))
        if winner_id is None:
            draw_status_square_icon(
                center_x=float(center_x),
                center_y=float(center_y),
                size=float(size),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
                inset=float(inset),
            )
            return
        style = PLAYER_STYLES.get(str(winner_id), PLAYER_STYLES["P1"])
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=style["render_outline"],
            inner_color=style["render_fill"],
            inset=float(inset),
        )

    def _remaining_time_ratio(self) -> float:
        return float(self.match_tracker.remaining_time_ratio(int(self.steps)))

    def render(self) -> None:
        if self.window_controller.window is None:
            return

        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        for origin in self.solid_block_origins:
            self._draw_solid_block(origin)
        for cell in sorted(self.solid_cells - self.solid_block_cells, key=lambda item: (int(item.y), int(item.x))):
            self._draw_solid_cell(cell)
        for cell in sorted(self.crate_cells, key=lambda item: (int(item.y), int(item.x))):
            self._draw_crate_cell(cell)
        for bomb in self.bombs:
            self._draw_bomb(bomb)
        for explosion in self.explosions:
            for cell in explosion.owners_by_cell:
                self._draw_explosion_cell(cell)
        for player_id in self.player_order:
            self._draw_player(self.players_by_id[player_id])

        layout = draw_status_bar(
            width=float(SCREEN_WIDTH),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(TILE_SIZE),
            cell_inset=float(CELL_INSET),
            include_clock=True,
        )
        draw_status_clock(
            layout=layout,
            remaining_ratio=float(self._remaining_time_ratio()),
        )
        draw_status_icon_row(
            left=float(layout.score_left),
            right=float(layout.score_right),
            center_y=float(layout.center_y),
            icon_size=max(12.0, min(float(TILE_SIZE), float(BB_HEIGHT) - 8.0)),
            items=list(self.match_tracker.history),
            draw_item=lambda winner_id, center_x, center_y, size: self._draw_winner_icon(
                winner_id,
                float(center_x),
                float(center_y),
                float(size),
            ),
        )
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
        self.window = None
