"""Osero environment, rendering, and human controls."""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SAND,
    COLOR_SLATE_GRAY,
)
from core.envs.arcade import ArcadeEnvMixin
from core.envs.base import Env
from core.primitives import (
    draw_status_bar,
    draw_status_icon_row,
    status_icon_inset,
    draw_status_square_icon,
    draw_two_tone_tile,
    status_icon_size,
)
from core.shared_config import BB_HEIGHT, FPS, SCREEN_HEIGHT, SCREEN_WIDTH, TRAINING_FPS, WORLD_HEIGHT
from games.osero import config
from games.osero.rules import (
    STONE_BLACK,
    STONE_EMPTY,
    STONE_WHITE,
    action_to_cell,
    apply_action,
    build_action_mask,
    initial_state,
    is_terminal_state,
    observation_from_state,
    outcome_for_player,
    pass_action_index,
    stone_counts,
    winner,
)


BOARD_FRAME_OUTER = COLOR_FOG_GRAY
BOARD_TILE_OUTER = COLOR_DARK_NEUTRAL
BOARD_TILE_INNER = COLOR_DEEP_TEAL
BLACK_STONE_OUTER = COLOR_SLATE_GRAY
BLACK_STONE_INNER = COLOR_DARK_NEUTRAL
WHITE_STONE_OUTER = COLOR_FOG_GRAY
WHITE_STONE_INNER = COLOR_LIGHT_NEUTRAL
LEGAL_HINT_COLOR = COLOR_SAND
HOVER_OUTLINE_COLOR = COLOR_AQUA


@dataclass(frozen=True)
class BoardLayout:
    left: float
    top: float
    tile_size: float
    board_pixels: float


class OseroEnv(ArcadeEnvMixin, Env):
    """AlphaZero-lite friendly Osero environment."""

    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.board_size = int(config.BOARD_SIZE)
        self._current_level = max(1, int(1 if level is None else level))
        self._init_arcade_runtime(
            width=SCREEN_WIDTH,
            height=SCREEN_HEIGHT,
            title=config.WINDOW_TITLE,
            render=bool(render),
            queue_input_events=self.mode == "human",
            vsync=False,
            render_fps=FPS,
            training_fps=TRAINING_FPS,
            eval_step_delay_seconds=float(config.AI_STEP_DELAY_SECONDS if self.mode == "eval" else 0.0),
        )
        self._board_layout = self._build_board_layout()
        self._state = initial_state(self.board_size)
        self._last_obs = observation_from_state(self._state)
        self._done = False
        self._last_winner = STONE_EMPTY
        self._hover_action: int | None = None

    @property
    def _pass_action(self) -> int:
        return int(pass_action_index(self.board_size))

    def _build_board_layout(self) -> BoardLayout:
        usable_width = float(SCREEN_WIDTH) - float(config.BOARD_SIDE_MARGIN) * 2.0
        usable_height = (
            float(WORLD_HEIGHT)
            - float(config.BOARD_TOP_MARGIN)
            - float(config.BOARD_BOTTOM_MARGIN)
        )
        board_pixels = float(max(self.board_size * 24, min(usable_width, usable_height)))
        tile_size = float(int(board_pixels // self.board_size))
        board_pixels = float(tile_size * self.board_size)
        left = (float(SCREEN_WIDTH) - board_pixels) * 0.5
        top = float(config.BOARD_TOP_MARGIN) + max(0.0, (usable_height - board_pixels) * 0.5)
        return BoardLayout(left=float(left), top=float(top), tile_size=float(tile_size), board_pixels=float(board_pixels))

    def reset(self) -> np.ndarray:
        self._state = initial_state(self.board_size)
        self._last_obs = observation_from_state(self._state)
        self._done = False
        self._last_winner = STONE_EMPTY
        self._hover_action = None
        if self.show_game:
            self.render()
        return np.asarray(self._last_obs, dtype=np.float32)

    def get_action_mask(self, _obs: object | None = None) -> np.ndarray:
        if self._done:
            return np.zeros((self.ACT_DIM,), dtype=np.bool_)
        return build_action_mask(self._state.board, int(self._state.current_player))

    def _resolve_valid_action(self, action: object) -> int:
        mask = self.get_action_mask()
        legal_actions = np.flatnonzero(mask)
        if legal_actions.size <= 0:
            return self._pass_action
        try:
            action_index = int(action)
        except (TypeError, ValueError):
            action_index = int(legal_actions[0])
        if 0 <= action_index < self.ACT_DIM and bool(mask[action_index]):
            return int(action_index)
        return int(legal_actions[0])

    def _state_info(self) -> dict[str, object]:
        black_count, white_count = stone_counts(self._state.board)
        return {
            "board_size": int(self.board_size),
            "black_stones": int(black_count),
            "white_stones": int(white_count),
            "winner": int(self._last_winner) if self._done else STONE_EMPTY,
            "level": int(self._current_level),
        }

    @staticmethod
    def _stone_colors(stone: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if int(stone) == STONE_BLACK:
            return BLACK_STONE_OUTER, BLACK_STONE_INNER
        return WHITE_STONE_OUTER, WHITE_STONE_INNER

    def _apply_action_and_collect(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        actor = int(self._state.current_player)
        self._state = apply_action(self._state, int(action_index))
        self._last_obs = observation_from_state(self._state)
        self._done = bool(is_terminal_state(self._state))
        self._last_winner = int(winner(self._state.board)) if self._done else STONE_EMPTY

        reward = 0.0
        reward_breakdown = {
            "reward_terminal_win": 0.0,
            "reward_terminal_draw": 0.0,
            "reward_terminal_loss": 0.0,
        }
        if self._done:
            reward = float(outcome_for_player(self._state.board, actor))
            if reward > 0.0:
                reward_breakdown["reward_terminal_win"] = 1.0
            elif reward < 0.0:
                reward_breakdown["reward_terminal_loss"] = -1.0
            else:
                reward_breakdown["reward_terminal_draw"] = 0.0

        info = self._state_info()
        info.update(
            {
                "win": bool(self._done and int(self._last_winner) == STONE_BLACK),
                "success": 1 if self._done and int(self._last_winner) == STONE_BLACK else 0,
                "reward_breakdown": reward_breakdown,
                "moves": int(self._state.move_count),
                "passed": bool(int(action_index) == int(self._pass_action)),
            }
        )
        if self._done:
            info["reward_components"] = {
                "W": float(reward_breakdown["reward_terminal_win"]),
                "D": float(reward_breakdown["reward_terminal_draw"]),
                "L": float(reward_breakdown["reward_terminal_loss"]),
            }
        return np.asarray(self._last_obs, dtype=np.float32), float(reward), bool(self._done), info

    def _mouse_to_action(self, x: float, y_arcade: float) -> int | None:
        top_left_y = self.window_controller.to_top_left_y(float(y_arcade))
        board = self._board_layout
        relative_x = float(x) - float(board.left)
        relative_y = float(top_left_y) - float(board.top)
        if relative_x < 0.0 or relative_y < 0.0:
            return None
        row = int(relative_y // float(board.tile_size))
        col = int(relative_x // float(board.tile_size))
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return None
        return int(row * self.board_size + col)

    def _update_hover_action(self) -> None:
        self._hover_action = None
        if not self.show_game:
            return
        mouse_pos = self.window_controller.mouse_position()
        if mouse_pos is None:
            return
        action_index = self._mouse_to_action(mouse_pos[0], mouse_pos[1])
        if action_index is None:
            return
        mask = self.get_action_mask()
        if 0 <= action_index < self.ACT_DIM and bool(mask[action_index]):
            self._hover_action = int(action_index)

    def _handle_human_terminal(self) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        for _mouse in self.window_controller.consume_mouse_presses():
            return self.reset(), 0.0, False, {"level": int(self._current_level)}
        for key_code in self.window_controller.consume_key_presses():
            if int(key_code) in {arcade.key.ENTER, arcade.key.SPACE}:
                return self.reset(), 0.0, False, {"level": int(self._current_level)}

        self.render()
        self._tick_arcade_frame(delay_seconds=0.0)
        return np.asarray(self._last_obs, dtype=np.float32), 0.0, False, self._state_info()

    def _step_human(self) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self._done:
            return self._handle_human_terminal()

        self._update_hover_action()
        mask = self.get_action_mask()
        if bool(mask[self._pass_action]) and int(mask[:-1].sum()) == 0:
            obs, _, done, info = self._apply_action_and_collect(self._pass_action)
            if self._done:
                return self._handle_human_terminal()
            self.render()
            self._tick_arcade_frame(delay_seconds=0.0)
            return obs, 0.0, bool(done), info

        for mouse_press in self.window_controller.consume_mouse_presses():
            action_index = self._mouse_to_action(mouse_press.x, mouse_press.y)
            if action_index is None or not bool(mask[action_index]):
                continue
            obs, _, done, info = self._apply_action_and_collect(action_index)
            if done:
                return self._handle_human_terminal()
            self.render()
            self._tick_arcade_frame(delay_seconds=0.0)
            return obs, 0.0, False, info

        self.render()
        self._tick_arcade_frame(delay_seconds=0.0)
        return np.asarray(self._last_obs, dtype=np.float32), 0.0, False, self._state_info()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        self.window_controller.poll_events_or_raise()

        if self.mode == "human":
            return self._step_human()

        if self._done:
            info = self._state_info()
            info.update(
                {
                    "win": bool(int(self._last_winner) == STONE_BLACK),
                    "success": 1 if int(self._last_winner) == STONE_BLACK else 0,
                }
            )
            return np.asarray(self._last_obs, dtype=np.float32), 0.0, True, info

        action_index = self._resolve_valid_action(action)
        obs, reward, done, info = self._apply_action_and_collect(action_index)
        if self.show_game:
            self.render()
        self._tick_arcade_frame()
        return obs, float(reward), bool(done), info

    def _draw_board(self) -> None:
        board = self._board_layout
        frame_left = float(board.left) - float(config.BOARD_FRAME_PADDING)
        frame_top = float(board.top) - float(config.BOARD_FRAME_PADDING)
        frame_size = float(board.board_pixels) + float(config.BOARD_FRAME_PADDING) * 2.0
        gridline_width = max(1.0, float(board.tile_size) * 0.08)
        frame_bottom = self.window_controller.to_arcade_y(float(frame_top) + float(frame_size))
        arcade.draw_lbwh_rectangle_outline(
            float(frame_left),
            float(frame_bottom),
            float(frame_size),
            float(frame_size),
            BOARD_FRAME_OUTER,
            float(gridline_width),
        )
        board_bottom = self.window_controller.to_arcade_y(float(board.top) + float(board.board_pixels))
        arcade.draw_lbwh_rectangle_outline(
            float(board.left),
            float(board_bottom),
            float(board.board_pixels),
            float(board.board_pixels),
            BOARD_TILE_OUTER,
            float(gridline_width),
        )
        for row in range(self.board_size):
            for col in range(self.board_size):
                cell_left = float(board.left) + float(col) * float(board.tile_size)
                cell_top = float(board.top) + float(row) * float(board.tile_size)
                draw_two_tone_tile(
                    self.window_controller,
                    top_left_x=float(cell_left),
                    top_left_y=float(cell_top),
                    size=float(board.tile_size),
                    outer_color=BOARD_TILE_OUTER,
                    inner_color=BOARD_TILE_INNER,
                    inset=max(1.0, float(board.tile_size) * 0.08),
                )

    def _draw_legal_hints(self) -> None:
        if self._done:
            return
        mask = self.get_action_mask()
        board = self._board_layout
        hint_size = max(6.0, float(board.tile_size) * float(config.LEGAL_HINT_RATIO))
        for action_index in np.flatnonzero(mask[:-1]):
            row_col = action_to_cell(int(action_index), self.board_size)
            if row_col is None:
                continue
            row, col = row_col
            center_x = float(board.left) + (float(col) + 0.5) * float(board.tile_size)
            center_y = float(board.top) + (float(row) + 0.5) * float(board.tile_size)
            bottom = self.window_controller.to_arcade_y(float(center_y) + hint_size * 0.5)
            arcade.draw_lbwh_rectangle_filled(
                float(center_x) - hint_size * 0.5,
                float(bottom),
                float(hint_size),
                float(hint_size),
                LEGAL_HINT_COLOR,
            )

    def _draw_hover_outline(self) -> None:
        if self._hover_action is None:
            return
        row_col = action_to_cell(int(self._hover_action), self.board_size)
        if row_col is None:
            return
        row, col = row_col
        board = self._board_layout
        left = float(board.left) + float(col) * float(board.tile_size)
        top = float(board.top) + float(row) * float(board.tile_size)
        bottom = self.window_controller.to_arcade_y(float(top) + float(board.tile_size))
        arcade.draw_lbwh_rectangle_outline(
            float(left),
            float(bottom),
            float(board.tile_size),
            float(board.tile_size),
            HOVER_OUTLINE_COLOR,
            float(config.HOVER_OUTLINE_WIDTH),
        )

    def _draw_stones(self) -> None:
        board = self._board_layout
        stone_size = float(board.tile_size) * (1.0 - float(config.STONE_INSET_RATIO) * 2.0)
        stone_inset = float(board.tile_size) * float(config.STONE_INSET_RATIO)
        inner_inset = max(2.0, stone_size * 0.18)
        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = int(self._state.board[row, col])
                if stone == STONE_EMPTY:
                    continue
                outer_color, inner_color = self._stone_colors(stone)
                draw_two_tone_tile(
                    self.window_controller,
                    top_left_x=float(board.left) + float(col) * float(board.tile_size) + float(stone_inset),
                    top_left_y=float(board.top) + float(row) * float(board.tile_size) + float(stone_inset),
                    size=float(stone_size),
                    outer_color=outer_color,
                    inner_color=inner_color,
                    inset=float(inner_inset),
                )

    def _draw_player_icon(self, stone: int, center_x: float, center_y: float, size: float) -> None:
        if stone == STONE_EMPTY:
            return
        outer_color, inner_color = self._stone_colors(stone)
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(status_icon_inset(float(self._board_layout.tile_size) * float(config.STONE_INSET_RATIO))),
        )

    def _draw_hud(self) -> None:
        layout = draw_status_bar(
            width=float(SCREEN_WIDTH),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(self._board_layout.tile_size),
            cell_inset=float(config.STONE_INSET_RATIO * self._board_layout.tile_size),
            include_clock=False,
        )
        stone = int(self._state.current_player) if not self._done else int(self._last_winner)
        if stone == STONE_EMPTY:
            return
        draw_status_icon_row(
            left=float(layout.score_left),
            right=float(layout.score_right),
            center_y=float(layout.center_y),
            icon_size=float(status_icon_size(float(BB_HEIGHT), float(self._board_layout.tile_size))),
            items=[stone],
            draw_item=lambda stone_value, center_x, row_center_y, size: self._draw_player_icon(
                int(stone_value),
                float(center_x),
                float(row_center_y),
                float(size),
            ),
        )

    def render(self) -> None:
        if self.window_controller.window is None:
            return
        if self.mode == "human":
            self._update_hover_action()
        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        self._draw_board()
        self._draw_legal_hints()
        self._draw_stones()
        self._draw_hover_outline()
        self._draw_hud()
        self.window_controller.flip()
