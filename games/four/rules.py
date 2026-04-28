"""Shared Four rules and board encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from games.four import config


STONE_EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1
PLAYER_NONE = 0
PLAYER_NAMES = {
    PLAYER_ONE: "P1",
    PLAYER_TWO: "P2",
    PLAYER_NONE: "draw",
}
DIRECTIONS = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


@dataclass(frozen=True)
class FourState:
    board: np.ndarray
    current_player: int
    move_count: int = 0


def validate_board_shape(rows: int | None = None, cols: int | None = None) -> tuple[int, int]:
    row_count = int(config.BOARD_ROWS if rows is None else rows)
    col_count = int(config.BOARD_COLS if cols is None else cols)
    if row_count != int(config.BOARD_ROWS) or col_count != int(config.BOARD_COLS):
        raise ValueError(
            f"Four board shape must be {config.BOARD_ROWS}x{config.BOARD_COLS}, "
            f"got {row_count}x{col_count}."
        )
    return int(row_count), int(col_count)


def initial_state(rows: int | None = None, cols: int | None = None) -> FourState:
    row_count, col_count = validate_board_shape(rows, cols)
    board = np.zeros((row_count, col_count), dtype=np.int8)
    return FourState(board=board, current_player=PLAYER_ONE)


def in_bounds(row: int, col: int, rows: int | None = None, cols: int | None = None) -> bool:
    row_count, col_count = validate_board_shape(rows, cols)
    return 0 <= int(row) < row_count and 0 <= int(col) < col_count


def drop_row_for_column(board: np.ndarray, col: int) -> int | None:
    board_array = np.asarray(board, dtype=np.int8)
    row_count, col_count = validate_board_shape(*board_array.shape)
    col_index = int(col)
    if not (0 <= col_index < col_count):
        raise ValueError(f"Column {col_index} is out of range for Four.")
    for row in range(row_count - 1, -1, -1):
        if int(board_array[row, col_index]) == STONE_EMPTY:
            return int(row)
    return None


def legal_actions(board: np.ndarray) -> list[int]:
    board_array = np.asarray(board, dtype=np.int8)
    _row_count, col_count = validate_board_shape(*board_array.shape)
    if winner(board_array) != PLAYER_NONE:
        return []
    return [col for col in range(col_count) if drop_row_for_column(board_array, col) is not None]


def build_action_mask(board: np.ndarray, current_player: int | None = None) -> np.ndarray:
    del current_player
    mask = np.zeros((int(config.BOARD_COLS),), dtype=np.bool_)
    actions = legal_actions(board)
    if actions:
        mask[np.asarray(actions, dtype=np.int32)] = True
    return mask


def has_four(board: np.ndarray, player: int) -> bool:
    board_array = np.asarray(board, dtype=np.int8)
    row_count, col_count = validate_board_shape(*board_array.shape)
    player_value = int(player)
    if player_value == STONE_EMPTY:
        return False

    for row in range(row_count):
        for col in range(col_count):
            if int(board_array[row, col]) != player_value:
                continue
            for delta_row, delta_col in DIRECTIONS:
                end_row = row + (int(config.CONNECT_N) - 1) * delta_row
                end_col = col + (int(config.CONNECT_N) - 1) * delta_col
                if not in_bounds(end_row, end_col, row_count, col_count):
                    continue
                if all(
                    int(board_array[row + step * delta_row, col + step * delta_col]) == player_value
                    for step in range(int(config.CONNECT_N))
                ):
                    return True
    return False


def winner(board: np.ndarray) -> int:
    board_array = np.asarray(board, dtype=np.int8)
    if has_four(board_array, PLAYER_ONE):
        return PLAYER_ONE
    if has_four(board_array, PLAYER_TWO):
        return PLAYER_TWO
    return PLAYER_NONE


def is_full(board: np.ndarray) -> bool:
    board_array = np.asarray(board, dtype=np.int8)
    validate_board_shape(*board_array.shape)
    return bool(np.all(board_array[0, :] != STONE_EMPTY))


def is_terminal_board(board: np.ndarray) -> bool:
    board_array = np.asarray(board, dtype=np.int8)
    return bool(winner(board_array) != PLAYER_NONE or is_full(board_array))


def is_terminal_state(state: FourState) -> bool:
    return bool(is_terminal_board(state.board))


def apply_action(state: FourState, action: int) -> FourState:
    board = np.asarray(state.board, dtype=np.int8)
    validate_board_shape(*board.shape)
    action_index = int(action)
    if not (0 <= action_index < int(config.BOARD_COLS)):
        raise ValueError(f"Action index {action_index} is out of range for Four.")
    if bool(is_terminal_board(board)):
        raise ValueError("Cannot apply an action to a terminal Four position.")

    row = drop_row_for_column(board, action_index)
    if row is None:
        raise ValueError(f"Column {action_index} is full in the current Four position.")

    next_board = board.copy()
    next_board[int(row), action_index] = int(state.current_player)
    return FourState(
        board=next_board,
        current_player=-int(state.current_player),
        move_count=int(state.move_count) + 1,
    )


def is_winning_action(state: FourState, action: int) -> bool:
    try:
        next_state = apply_action(state, int(action))
    except ValueError:
        return False
    return int(winner(next_state.board)) == int(state.current_player)


def outcome_for_player(board: np.ndarray, player: int) -> float:
    winning_player = winner(board)
    if winning_player == PLAYER_NONE:
        return 0.0
    return 1.0 if int(winning_player) == int(player) else -1.0


def reward_for_player(board: np.ndarray, player: int) -> float:
    outcome = float(outcome_for_player(board, int(player)))
    if outcome > 0.0:
        return float(config.REWARD_WIN)
    if outcome < 0.0:
        return float(config.PENALTY_LOSS)
    return float(config.REWARD_DRAW)


def observation_from_board(board: np.ndarray, current_player: int) -> np.ndarray:
    board_array = np.asarray(board, dtype=np.int8)
    validate_board_shape(*board_array.shape)
    perspective_board = board_array.astype(np.float32) * float(current_player)
    return perspective_board.reshape(-1).astype(np.float32, copy=False)


def observation_from_state(state: FourState) -> np.ndarray:
    return observation_from_board(state.board, int(state.current_player))


def canonical_board_from_observation(
    observation: np.ndarray,
    board_rows: int | None = None,
    board_cols: int | None = None,
) -> np.ndarray:
    row_count, col_count = validate_board_shape(board_rows, board_cols)
    obs = np.asarray(observation, dtype=np.float32).reshape(row_count, col_count)
    return np.rint(obs).astype(np.int8, copy=False)


def action_mask_from_observation(
    observation: np.ndarray,
    board_rows: int | None = None,
    board_cols: int | None = None,
) -> np.ndarray:
    canonical_board = canonical_board_from_observation(observation, board_rows, board_cols)
    return build_action_mask(canonical_board, current_player=PLAYER_ONE)


def apply_canonical_action(canonical_board: np.ndarray, action: int) -> np.ndarray:
    board = np.asarray(canonical_board, dtype=np.int8)
    state = FourState(board=board, current_player=PLAYER_ONE)
    next_state = apply_action(state, int(action))
    return (next_state.board * int(next_state.current_player)).astype(np.int8, copy=False)


def terminal_outcome_from_canonical(canonical_board: np.ndarray) -> float:
    board = np.asarray(canonical_board, dtype=np.int8)
    return float(outcome_for_player(board, PLAYER_ONE))


def symmetry_observation_policy_pairs(
    observation: np.ndarray,
    policy_target: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    row_count, col_count = validate_board_shape()
    board = np.asarray(observation, dtype=np.float32).reshape(row_count, col_count)
    policy = np.asarray(policy_target, dtype=np.float32).reshape(col_count)
    mirrored_board = np.fliplr(board).astype(np.float32, copy=False)
    mirrored_policy = policy[::-1].astype(np.float32, copy=False)
    return [
        (
            board.reshape(-1).astype(np.float32, copy=False),
            policy.astype(np.float32, copy=False),
        ),
        (
            mirrored_board.reshape(-1).astype(np.float32, copy=False),
            mirrored_policy,
        ),
    ]
