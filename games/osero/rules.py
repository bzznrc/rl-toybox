"""Shared Osero rules and board encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from games.osero import config


STONE_EMPTY = 0
STONE_BLACK = 1
STONE_WHITE = -1
PLAYER_NAMES = {
    STONE_BLACK: "Black",
    STONE_WHITE: "White",
}
DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class OseroState:
    board: np.ndarray
    current_player: int
    consecutive_passes: int = 0
    move_count: int = 0


def validate_board_size(board_size: int) -> int:
    size = int(board_size)
    if size not in config.SUPPORTED_BOARD_SIZES:
        raise ValueError(
            f"Osero board size must be one of {config.SUPPORTED_BOARD_SIZES}, got {size}."
        )
    return int(size)


def pass_action_index(board_size: int) -> int:
    size = validate_board_size(board_size)
    return int(size * size)


def action_to_cell(action: int, board_size: int) -> tuple[int, int] | None:
    size = validate_board_size(board_size)
    action_index = int(action)
    if action_index == pass_action_index(size):
        return None
    if not (0 <= action_index < size * size):
        raise ValueError(f"Action index {action_index} is out of range for {size}x{size} Osero.")
    return divmod(action_index, size)


def cell_to_action(row: int, col: int, board_size: int) -> int:
    size = validate_board_size(board_size)
    row_index = int(row)
    col_index = int(col)
    if not (0 <= row_index < size and 0 <= col_index < size):
        raise ValueError(f"Cell {(row_index, col_index)} is out of bounds for {size}x{size} Osero.")
    return int(row_index * size + col_index)


def initial_state(board_size: int) -> OseroState:
    size = validate_board_size(board_size)
    board = np.zeros((size, size), dtype=np.int8)
    center_low = size // 2 - 1
    center_high = center_low + 1
    board[center_low, center_low] = STONE_WHITE
    board[center_high, center_high] = STONE_WHITE
    board[center_low, center_high] = STONE_BLACK
    board[center_high, center_low] = STONE_BLACK
    return OseroState(board=board, current_player=STONE_BLACK)


def in_bounds(row: int, col: int, board_size: int) -> bool:
    size = validate_board_size(board_size)
    return 0 <= int(row) < size and 0 <= int(col) < size


def _capture_line(
    board: np.ndarray,
    *,
    current_player: int,
    row: int,
    col: int,
    delta_row: int,
    delta_col: int,
) -> list[tuple[int, int]]:
    captures: list[tuple[int, int]] = []
    check_row = int(row) + int(delta_row)
    check_col = int(col) + int(delta_col)
    size = int(board.shape[0])

    while in_bounds(check_row, check_col, size) and int(board[check_row, check_col]) == -int(current_player):
        captures.append((int(check_row), int(check_col)))
        check_row += int(delta_row)
        check_col += int(delta_col)

    if not captures or not in_bounds(check_row, check_col, size):
        return []
    if int(board[check_row, check_col]) != int(current_player):
        return []
    return captures


def captures_for_move(board: np.ndarray, current_player: int, row: int, col: int) -> list[tuple[int, int]]:
    if int(board[row, col]) != STONE_EMPTY:
        return []
    captures: list[tuple[int, int]] = []
    for delta_row, delta_col in DIRECTIONS:
        captures.extend(
            _capture_line(
                board,
                current_player=int(current_player),
                row=int(row),
                col=int(col),
                delta_row=int(delta_row),
                delta_col=int(delta_col),
            )
        )
    return captures


def legal_cell_actions(board: np.ndarray, current_player: int) -> list[int]:
    board_array = np.asarray(board, dtype=np.int8)
    size = int(board_array.shape[0])
    actions: list[int] = []
    for row in range(size):
        for col in range(size):
            if captures_for_move(board_array, int(current_player), int(row), int(col)):
                actions.append(cell_to_action(row, col, size))
    return actions


def has_legal_placement(board: np.ndarray, current_player: int) -> bool:
    return bool(legal_cell_actions(board, int(current_player)))


def build_action_mask(board: np.ndarray, current_player: int) -> np.ndarray:
    board_array = np.asarray(board, dtype=np.int8)
    size = int(board_array.shape[0])
    mask = np.zeros((pass_action_index(size) + 1,), dtype=np.bool_)
    legal_actions = legal_cell_actions(board_array, int(current_player))
    if legal_actions:
        mask[np.asarray(legal_actions, dtype=np.int32)] = True
        return mask
    mask[pass_action_index(size)] = True
    return mask


def is_terminal_board(board: np.ndarray) -> bool:
    board_array = np.asarray(board, dtype=np.int8)
    if not np.any(board_array == STONE_EMPTY):
        return True
    if has_legal_placement(board_array, STONE_BLACK):
        return False
    if has_legal_placement(board_array, STONE_WHITE):
        return False
    return True


def is_terminal_state(state: OseroState) -> bool:
    if int(state.consecutive_passes) >= 2:
        return True
    return bool(is_terminal_board(state.board))


def apply_action(state: OseroState, action: int) -> OseroState:
    board = np.asarray(state.board, dtype=np.int8)
    size = int(board.shape[0])
    action_index = int(action)
    pass_index = pass_action_index(size)

    if action_index == pass_index:
        if has_legal_placement(board, int(state.current_player)):
            raise ValueError("Pass is only legal when no placement move is available.")
        return OseroState(
            board=board.copy(),
            current_player=-int(state.current_player),
            consecutive_passes=int(state.consecutive_passes) + 1,
            move_count=int(state.move_count) + 1,
        )

    row_col = action_to_cell(action_index, size)
    if row_col is None:
        raise ValueError("Expected a placement action, received pass.")
    row, col = row_col
    captures = captures_for_move(board, int(state.current_player), int(row), int(col))
    if not captures:
        raise ValueError(f"Action {action_index} is not legal in the current Osero position.")

    next_board = board.copy()
    next_board[row, col] = int(state.current_player)
    for capture_row, capture_col in captures:
        next_board[int(capture_row), int(capture_col)] = int(state.current_player)

    return OseroState(
        board=next_board,
        current_player=-int(state.current_player),
        consecutive_passes=0,
        move_count=int(state.move_count) + 1,
    )


def stone_counts(board: np.ndarray) -> tuple[int, int]:
    board_array = np.asarray(board, dtype=np.int8)
    black_count = int(np.count_nonzero(board_array == STONE_BLACK))
    white_count = int(np.count_nonzero(board_array == STONE_WHITE))
    return int(black_count), int(white_count)


def winner(board: np.ndarray) -> int:
    black_count, white_count = stone_counts(board)
    if black_count > white_count:
        return STONE_BLACK
    if white_count > black_count:
        return STONE_WHITE
    return STONE_EMPTY


def outcome_for_player(board: np.ndarray, player: int) -> float:
    winning_player = winner(board)
    if winning_player == STONE_EMPTY:
        return 0.0
    return 1.0 if int(winning_player) == int(player) else -1.0


def observation_from_board(board: np.ndarray, current_player: int) -> np.ndarray:
    board_array = np.asarray(board, dtype=np.int8)
    perspective_board = board_array.astype(np.float32) * float(current_player)
    return perspective_board.reshape(-1).astype(np.float32, copy=False)


def observation_from_state(state: OseroState) -> np.ndarray:
    return observation_from_board(state.board, int(state.current_player))


def canonical_board_from_observation(observation: np.ndarray, board_size: int) -> np.ndarray:
    size = validate_board_size(board_size)
    obs = np.asarray(observation, dtype=np.float32).reshape(size, size)
    return np.rint(obs).astype(np.int8, copy=False)


def action_mask_from_observation(observation: np.ndarray, board_size: int) -> np.ndarray:
    canonical_board = canonical_board_from_observation(observation, board_size)
    return build_action_mask(canonical_board, current_player=STONE_BLACK)


def apply_canonical_action(canonical_board: np.ndarray, action: int) -> np.ndarray:
    board = np.asarray(canonical_board, dtype=np.int8)
    state = OseroState(board=board, current_player=STONE_BLACK)
    next_state = apply_action(state, int(action))
    return (next_state.board * int(next_state.current_player)).astype(np.int8, copy=False)


def terminal_outcome_from_canonical(canonical_board: np.ndarray) -> float:
    board = np.asarray(canonical_board, dtype=np.int8)
    return float(outcome_for_player(board, STONE_BLACK))
