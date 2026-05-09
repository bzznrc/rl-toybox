"""Shared Flip rules and board encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from games.flip import config


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
class FlipState:
    board: np.ndarray
    current_player: int
    move_count: int = 0
    pass_count: int = 0


def validate_board_shape(rows: int | None = None, cols: int | None = None) -> tuple[int, int]:
    row_count = int(config.BOARD_ROWS if rows is None else rows)
    col_count = int(config.BOARD_COLS if cols is None else cols)
    if row_count != int(config.BOARD_ROWS) or col_count != int(config.BOARD_COLS):
        raise ValueError(
            f"Flip board shape must be {config.BOARD_ROWS}x{config.BOARD_COLS}, "
            f"got {row_count}x{col_count}."
        )
    return int(row_count), int(col_count)


def initial_state(rows: int | None = None, cols: int | None = None) -> FlipState:
    row_count, col_count = validate_board_shape(rows, cols)
    board = np.zeros((row_count, col_count), dtype=np.int8)
    top = row_count // 2 - 1
    left = col_count // 2 - 1
    board[top, left] = PLAYER_TWO
    board[top, left + 1] = PLAYER_ONE
    board[top + 1, left] = PLAYER_ONE
    board[top + 1, left + 1] = PLAYER_TWO
    return normalize_turn(FlipState(board=board, current_player=PLAYER_ONE))


def in_bounds(row: int, col: int, rows: int | None = None, cols: int | None = None) -> bool:
    row_count, col_count = validate_board_shape(rows, cols)
    return 0 <= int(row) < row_count and 0 <= int(col) < col_count


def action_to_row_col(action: int) -> tuple[int, int]:
    action_index = int(action)
    if not (0 <= action_index < int(config.ACT_DIM)):
        raise ValueError(f"Action index {action_index} is out of range for Flip.")
    return divmod(action_index, int(config.BOARD_COLS))


def row_col_to_action(row: int, col: int) -> int:
    if not in_bounds(int(row), int(col)):
        raise ValueError(f"Cell r{int(row)} c{int(col)} is out of range for Flip.")
    return int(row) * int(config.BOARD_COLS) + int(col)


def _validate_player(player: int) -> int:
    player_value = int(player)
    if player_value not in {PLAYER_ONE, PLAYER_TWO}:
        raise ValueError(f"Flip player must be {PLAYER_ONE} or {PLAYER_TWO}, got {player_value}.")
    return int(player_value)


def flips_for_cell(board: np.ndarray, row: int, col: int, player: int) -> list[tuple[int, int]]:
    board_array = np.asarray(board, dtype=np.int8)
    row_count, col_count = validate_board_shape(*board_array.shape)
    row_index = int(row)
    col_index = int(col)
    player_value = _validate_player(player)
    if not in_bounds(row_index, col_index, row_count, col_count):
        return []
    if int(board_array[row_index, col_index]) != STONE_EMPTY:
        return []

    opponent = -int(player_value)
    flipped: list[tuple[int, int]] = []
    for delta_row, delta_col in DIRECTIONS:
        line: list[tuple[int, int]] = []
        scan_row = row_index + int(delta_row)
        scan_col = col_index + int(delta_col)
        while in_bounds(scan_row, scan_col, row_count, col_count):
            stone = int(board_array[scan_row, scan_col])
            if stone == opponent:
                line.append((int(scan_row), int(scan_col)))
                scan_row += int(delta_row)
                scan_col += int(delta_col)
                continue
            if stone == player_value and line:
                flipped.extend(line)
            break
    return flipped


def flips_for_action(state: FlipState, action: int) -> list[tuple[int, int]]:
    row, col = action_to_row_col(int(action))
    return flips_for_cell(state.board, int(row), int(col), int(state.current_player))


def legal_actions(board: np.ndarray, current_player: int = PLAYER_ONE) -> list[int]:
    board_array = np.asarray(board, dtype=np.int8)
    row_count, col_count = validate_board_shape(*board_array.shape)
    player_value = _validate_player(current_player)
    actions: list[int] = []
    for row in range(row_count):
        for col in range(col_count):
            if flips_for_cell(board_array, int(row), int(col), int(player_value)):
                actions.append(row_col_to_action(int(row), int(col)))
    return actions


def build_action_mask(board: np.ndarray, current_player: int | None = None) -> np.ndarray:
    player_value = PLAYER_ONE if current_player is None else _validate_player(int(current_player))
    mask = np.zeros((int(config.ACT_DIM),), dtype=np.bool_)
    actions = legal_actions(board, int(player_value))
    if actions:
        mask[np.asarray(actions, dtype=np.int32)] = True
    return mask


def has_legal_move(board: np.ndarray, player: int) -> bool:
    return bool(legal_actions(board, int(player)))


def is_full(board: np.ndarray) -> bool:
    board_array = np.asarray(board, dtype=np.int8)
    validate_board_shape(*board_array.shape)
    return bool(np.all(board_array != STONE_EMPTY))


def is_terminal_board(board: np.ndarray) -> bool:
    board_array = np.asarray(board, dtype=np.int8)
    validate_board_shape(*board_array.shape)
    return bool(
        is_full(board_array)
        or (
            not has_legal_move(board_array, PLAYER_ONE)
            and not has_legal_move(board_array, PLAYER_TWO)
        )
    )


def normalize_turn(state: FlipState) -> FlipState:
    board = np.asarray(state.board, dtype=np.int8)
    validate_board_shape(*board.shape)
    player = _validate_player(int(state.current_player))
    if is_terminal_board(board) or has_legal_move(board, player):
        return FlipState(
            board=board,
            current_player=int(player),
            move_count=int(state.move_count),
            pass_count=int(state.pass_count),
        )
    return FlipState(
        board=board,
        current_player=-int(player),
        move_count=int(state.move_count),
        pass_count=int(state.pass_count) + 1,
    )


def is_terminal_state(state: FlipState) -> bool:
    return bool(is_terminal_board(state.board))


def apply_action(state: FlipState, action: int) -> FlipState:
    active_state = normalize_turn(state)
    board = np.asarray(active_state.board, dtype=np.int8)
    if bool(is_terminal_board(board)):
        raise ValueError("Cannot apply an action to a terminal Flip position.")

    action_index = int(action)
    row, col = action_to_row_col(action_index)
    flipped = flips_for_cell(board, int(row), int(col), int(active_state.current_player))
    if not flipped:
        raise ValueError(f"Action index {action_index} is not legal in the current Flip position.")

    next_board = board.copy()
    next_board[int(row), int(col)] = int(active_state.current_player)
    for flip_row, flip_col in flipped:
        next_board[int(flip_row), int(flip_col)] = int(active_state.current_player)
    return normalize_turn(
        FlipState(
            board=next_board,
            current_player=-int(active_state.current_player),
            move_count=int(active_state.move_count) + 1,
            pass_count=int(active_state.pass_count),
        )
    )


def is_winning_action(state: FlipState, action: int) -> bool:
    try:
        active_state = normalize_turn(state)
        next_state = apply_action(active_state, int(action))
    except ValueError:
        return False
    if not bool(is_terminal_state(next_state)):
        return False
    return outcome_for_player(next_state.board, int(active_state.current_player)) > 0.0


def disc_counts(board: np.ndarray) -> tuple[int, int]:
    board_array = np.asarray(board, dtype=np.int8)
    validate_board_shape(*board_array.shape)
    return int(np.sum(board_array == PLAYER_ONE)), int(np.sum(board_array == PLAYER_TWO))


def winner(board: np.ndarray) -> int:
    p1_count, p2_count = disc_counts(board)
    if int(p1_count) > int(p2_count):
        return PLAYER_ONE
    if int(p2_count) > int(p1_count):
        return PLAYER_TWO
    return PLAYER_NONE


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


def observation_from_state(state: FlipState) -> np.ndarray:
    active_state = normalize_turn(state)
    return observation_from_board(active_state.board, int(active_state.current_player))


def _auto_pass_canonical_board(canonical_board: np.ndarray) -> np.ndarray:
    board = np.asarray(canonical_board, dtype=np.int8)
    validate_board_shape(*board.shape)
    if bool(is_terminal_board(board)) or has_legal_move(board, PLAYER_ONE):
        return board
    if has_legal_move(board, PLAYER_TWO):
        return (-board).astype(np.int8, copy=False)
    return board


def canonical_board_from_observation(
    observation: np.ndarray,
    board_rows: int | None = None,
    board_cols: int | None = None,
) -> np.ndarray:
    row_count, col_count = validate_board_shape(board_rows, board_cols)
    obs = np.asarray(observation, dtype=np.float32).reshape(row_count, col_count)
    canonical = np.rint(obs).astype(np.int8, copy=False)
    return _auto_pass_canonical_board(canonical)


def action_mask_from_observation(
    observation: np.ndarray,
    board_rows: int | None = None,
    board_cols: int | None = None,
) -> np.ndarray:
    canonical_board = canonical_board_from_observation(observation, board_rows, board_cols)
    return build_action_mask(canonical_board, current_player=PLAYER_ONE)


def apply_canonical_action_with_turn(canonical_board: np.ndarray, action: int) -> tuple[np.ndarray, int]:
    board = _auto_pass_canonical_board(np.asarray(canonical_board, dtype=np.int8))
    state = FlipState(board=board, current_player=PLAYER_ONE)
    next_state = apply_action(state, int(action))
    turn_sign = 1 if int(next_state.current_player) == PLAYER_ONE else -1
    next_board = (next_state.board * int(next_state.current_player)).astype(np.int8, copy=False)
    return next_board, int(turn_sign)


def apply_canonical_action(canonical_board: np.ndarray, action: int) -> np.ndarray:
    next_board, _turn_sign = apply_canonical_action_with_turn(canonical_board, int(action))
    return next_board


def terminal_outcome_from_canonical(canonical_board: np.ndarray) -> float:
    board = np.asarray(canonical_board, dtype=np.int8)
    return float(outcome_for_player(board, PLAYER_ONE))


def symmetry_observation_policy_pairs(
    observation: np.ndarray,
    policy_target: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    row_count, col_count = validate_board_shape()
    board = np.asarray(observation, dtype=np.float32).reshape(row_count, col_count)
    policy = np.asarray(policy_target, dtype=np.float32).reshape(row_count, col_count)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[bytes, bytes]] = set()
    for rotation in range(4):
        rotated_board = np.rot90(board, int(rotation))
        rotated_policy = np.rot90(policy, int(rotation))
        for next_board, next_policy in (
            (rotated_board, rotated_policy),
            (np.fliplr(rotated_board), np.fliplr(rotated_policy)),
        ):
            obs_sample = next_board.reshape(-1).astype(np.float32, copy=False)
            policy_sample = next_policy.reshape(-1).astype(np.float32, copy=False)
            key = (obs_sample.tobytes(), policy_sample.tobytes())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((obs_sample, policy_sample))
    return pairs
