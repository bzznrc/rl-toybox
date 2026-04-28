"""Four environment, rendering, and human controls."""

from __future__ import annotations

from dataclasses import dataclass

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
    COLOR_SAND,
    COLOR_SLATE_GRAY,
)
from core.envs.arcade import ArcadeEnvMixin
from core.envs.base import Env
from core.primitives import (
    draw_status_bar,
    draw_status_icon_row,
    draw_status_square_icon,
    draw_two_tone_tile,
    status_icon_inset,
    status_icon_size,
)
from core.shared_config import BB_HEIGHT, FPS, SCREEN_HEIGHT, SCREEN_WIDTH, TRAINING_FPS, WORLD_HEIGHT
from games.four import config
from games.four.rules import (
    PLAYER_NONE,
    PLAYER_ONE,
    PLAYER_TWO,
    apply_action,
    build_action_mask,
    drop_row_for_column,
    initial_state,
    is_terminal_state,
    observation_from_state,
    outcome_for_player,
    reward_for_player,
    winner,
)


BOARD_FRAME_OUTER = COLOR_FOG_GRAY
BOARD_TILE_OUTER = COLOR_SLATE_GRAY
BOARD_TILE_INNER = COLOR_DARK_NEUTRAL
P1_TOKEN_OUTER = COLOR_AQUA
P1_TOKEN_INNER = COLOR_DEEP_TEAL
P2_TOKEN_OUTER = COLOR_CORAL
P2_TOKEN_INNER = COLOR_BRICK_RED
LEGAL_HINT_COLOR = COLOR_SAND
HOVER_OUTLINE_COLOR = COLOR_LIGHT_NEUTRAL


@dataclass(frozen=True)
class BoardLayout:
    left: float
    top: float
    tile_size: float
    board_width: float
    board_height: float


class FourEnv(ArcadeEnvMixin, Env):
    """AlphaZero-lite friendly Connect Four environment."""

    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.board_rows = int(config.BOARD_ROWS)
        self.board_cols = int(config.BOARD_COLS)
        self._current_level = 1
        del level
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
        self._state = initial_state(self.board_rows, self.board_cols)
        self._last_obs = observation_from_state(self._state)
        self._done = False
        self._last_winner = PLAYER_NONE
        self._hover_action: int | None = None
        self._ai_opponent: object | None = None
        self._human_player = PLAYER_ONE
        self._ai_player = PLAYER_TWO

    def set_ai_opponent(self, algorithm: object) -> None:
        """Attach a model opponent for human play."""
        self._ai_opponent = algorithm
        reset_policy_state = getattr(algorithm, "reset_policy_state", None)
        if callable(reset_policy_state):
            reset_policy_state()

    def _build_board_layout(self) -> BoardLayout:
        usable_width = float(SCREEN_WIDTH) - float(config.BOARD_SIDE_MARGIN) * 2.0
        usable_height = (
            float(WORLD_HEIGHT)
            - float(config.BOARD_TOP_MARGIN)
            - float(config.BOARD_BOTTOM_MARGIN)
        )
        tile_size = float(
            int(
                max(
                    24.0,
                    min(
                        usable_width / float(self.board_cols),
                        usable_height / float(self.board_rows),
                    ),
                )
            )
        )
        board_width = float(tile_size * self.board_cols)
        board_height = float(tile_size * self.board_rows)
        left = (float(SCREEN_WIDTH) - board_width) * 0.5
        top = float(config.BOARD_TOP_MARGIN) + max(0.0, (usable_height - board_height) * 0.5)
        return BoardLayout(
            left=float(left),
            top=float(top),
            tile_size=float(tile_size),
            board_width=float(board_width),
            board_height=float(board_height),
        )

    def reset(self) -> np.ndarray:
        self._state = initial_state(self.board_rows, self.board_cols)
        self._last_obs = observation_from_state(self._state)
        self._done = False
        self._last_winner = PLAYER_NONE
        self._hover_action = None
        reset_policy_state = getattr(self._ai_opponent, "reset_policy_state", None)
        if callable(reset_policy_state):
            reset_policy_state()
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
            return 0
        try:
            action_index = int(action)
        except (TypeError, ValueError):
            action_index = int(legal_actions[0])
        if 0 <= action_index < self.ACT_DIM and bool(mask[action_index]):
            return int(action_index)
        return int(legal_actions[0])

    def _state_info(self) -> dict[str, object]:
        return {
            "board_rows": int(self.board_rows),
            "board_cols": int(self.board_cols),
            "current_player": int(self._state.current_player),
            "winner": int(self._last_winner) if self._done else PLAYER_NONE,
            "level": int(self._current_level),
            "human_player": int(self._human_player),
            "ai_player": int(self._ai_player) if self._ai_opponent is not None else PLAYER_NONE,
        }

    @staticmethod
    def _token_colors(player: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if int(player) == PLAYER_TWO:
            return P2_TOKEN_OUTER, P2_TOKEN_INNER
        return P1_TOKEN_OUTER, P1_TOKEN_INNER

    def _apply_action_and_collect(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        actor = int(self._state.current_player)
        self._state = apply_action(self._state, int(action_index))
        self._last_obs = observation_from_state(self._state)
        self._done = bool(is_terminal_state(self._state))
        self._last_winner = int(winner(self._state.board)) if self._done else PLAYER_NONE

        reward = 0.0
        reward_breakdown = {
            "outcome.reward_win": 0.0,
            "outcome.reward_draw": 0.0,
            "outcome.penalty_loss": 0.0,
        }
        search_value = 0.0
        if self._done:
            reward = float(reward_for_player(self._state.board, actor))
            search_value = float(outcome_for_player(self._state.board, actor))
            if reward > 0.0:
                reward_breakdown["outcome.reward_win"] = float(config.REWARD_WIN)
            elif reward < 0.0:
                reward_breakdown["outcome.penalty_loss"] = float(config.PENALTY_LOSS)
            else:
                reward_breakdown["outcome.reward_draw"] = float(config.REWARD_DRAW)

        info = self._state_info()
        info.update(
            {
                "actor": int(actor),
                "action_col": int(action_index),
                "win": bool(self._done and int(self._last_winner) == PLAYER_ONE),
                "success": 1 if self._done and int(self._last_winner) == PLAYER_ONE else 0,
                "reward_breakdown": reward_breakdown,
                "moves": int(self._state.move_count),
                "search_value": float(search_value),
            }
        )
        if self._done:
            info["reward_components"] = {
                "W": float(reward_breakdown["outcome.reward_win"]),
                "D": float(reward_breakdown["outcome.reward_draw"]),
                "L": float(reward_breakdown["outcome.penalty_loss"]),
            }
        return np.asarray(self._last_obs, dtype=np.float32), float(reward), bool(self._done), info

    def _is_ai_turn(self) -> bool:
        return bool(
            self._ai_opponent is not None
            and not self._done
            and int(self._state.current_player) == int(self._ai_player)
        )

    def capture_pre_action_delay_seconds(self) -> float:
        if self._done:
            return 0.0
        if int(self._state.current_player) == PLAYER_TWO:
            return float(config.AI_STEP_DELAY_SECONDS)
        return 0.0

    def _select_ai_opponent_action(self) -> int:
        mask = self.get_action_mask()
        observation = np.asarray(self._last_obs, dtype=np.float32)
        act = getattr(self._ai_opponent, "act", None)
        if not callable(act):
            return self._resolve_valid_action(0)
        try:
            action = act(observation, explore=False, action_mask=mask)
        except TypeError:
            action = act(observation, explore=False)
        return self._resolve_valid_action(action)

    def _apply_ai_opponent_turn(self) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        action_index = self._select_ai_opponent_action()
        obs, reward, done, info = self._apply_action_and_collect(action_index)
        info["ai_action_col"] = int(action_index)
        return obs, float(reward), bool(done), info

    def _mouse_to_action(self, x: float, y_arcade: float) -> int | None:
        top_left_y = self.window_controller.to_top_left_y(float(y_arcade))
        board = self._board_layout
        relative_x = float(x) - float(board.left)
        relative_y = float(top_left_y) - float(board.top)
        if relative_x < 0.0 or relative_y < 0.0:
            return None
        col = int(relative_x // float(board.tile_size))
        if not (0 <= col < self.board_cols) or relative_y >= float(board.board_height):
            return None
        return int(col)

    def _update_hover_action(self) -> None:
        self._hover_action = None
        if not self.show_game:
            return
        if self._is_ai_turn():
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

        if self._is_ai_turn():
            self.render()
            self._tick_arcade_frame(delay_seconds=float(self.capture_pre_action_delay_seconds()))
            obs, _reward, done, info = self._apply_ai_opponent_turn()
            if done:
                return self._handle_human_terminal()
            self.render()
            self._tick_arcade_frame(delay_seconds=0.0)
            return obs, 0.0, False, info

        self._update_hover_action()
        mask = self.get_action_mask()
        for mouse_press in self.window_controller.consume_mouse_presses():
            if int(self._state.current_player) != int(self._human_player):
                continue
            action_index = self._mouse_to_action(mouse_press.x, mouse_press.y)
            if action_index is None or not bool(mask[action_index]):
                continue
            obs, _, done, info = self._apply_action_and_collect(action_index)
            if done:
                return self._handle_human_terminal()
            self.render()
            self._tick_arcade_frame(delay_seconds=0.0)

            if self._is_ai_turn():
                self._tick_arcade_frame(delay_seconds=float(self.capture_pre_action_delay_seconds()))
                obs, _ai_reward, done, info = self._apply_ai_opponent_turn()
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
                    "win": bool(int(self._last_winner) == PLAYER_ONE),
                    "success": 1 if int(self._last_winner) == PLAYER_ONE else 0,
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
        frame_width = float(board.board_width) + float(config.BOARD_FRAME_PADDING) * 2.0
        frame_height = float(board.board_height) + float(config.BOARD_FRAME_PADDING) * 2.0
        gridline_width = max(1.0, float(board.tile_size) * 0.08)
        frame_bottom = self.window_controller.to_arcade_y(float(frame_top) + float(frame_height))
        arcade.draw_lbwh_rectangle_outline(
            float(frame_left),
            float(frame_bottom),
            float(frame_width),
            float(frame_height),
            BOARD_FRAME_OUTER,
            float(gridline_width),
        )

        for row in range(self.board_rows):
            for col in range(self.board_cols):
                cell_left = float(board.left) + float(col) * float(board.tile_size)
                cell_top = float(board.top) + float(row) * float(board.tile_size)
                draw_two_tone_tile(
                    self.window_controller,
                    top_left_x=float(cell_left),
                    top_left_y=float(cell_top),
                    size=float(board.tile_size),
                    outer_color=BOARD_TILE_OUTER,
                    inner_color=BOARD_TILE_INNER,
                    inset=max(1.0, float(board.tile_size) * float(config.CELL_INSET_RATIO)),
                )

    def _draw_legal_hints(self) -> None:
        if self._done:
            return
        mask = self.get_action_mask()
        board = self._board_layout
        hint_size = max(6.0, float(board.tile_size) * float(config.LEGAL_HINT_RATIO))
        for action_index in np.flatnonzero(mask):
            row = drop_row_for_column(self._state.board, int(action_index))
            if row is None:
                continue
            center_x = float(board.left) + (float(action_index) + 0.5) * float(board.tile_size)
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
        board = self._board_layout
        left = float(board.left) + float(self._hover_action) * float(board.tile_size)
        top = float(board.top)
        bottom = self.window_controller.to_arcade_y(float(top) + float(board.board_height))
        arcade.draw_lbwh_rectangle_outline(
            float(left),
            float(bottom),
            float(board.tile_size),
            float(board.board_height),
            HOVER_OUTLINE_COLOR,
            float(config.HOVER_OUTLINE_WIDTH),
        )

    def _draw_tokens(self) -> None:
        board = self._board_layout
        token_size = float(board.tile_size) * (1.0 - float(config.STONE_INSET_RATIO) * 2.0)
        token_inset = float(board.tile_size) * float(config.STONE_INSET_RATIO)
        inner_inset = max(2.0, token_size * 0.18)
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                stone = int(self._state.board[row, col])
                if stone == PLAYER_NONE:
                    continue
                outer_color, inner_color = self._token_colors(stone)
                draw_two_tone_tile(
                    self.window_controller,
                    top_left_x=float(board.left) + float(col) * float(board.tile_size) + float(token_inset),
                    top_left_y=float(board.top) + float(row) * float(board.tile_size) + float(token_inset),
                    size=float(token_size),
                    outer_color=outer_color,
                    inner_color=inner_color,
                    inset=float(inner_inset),
                )

    def _draw_player_icon(self, player: int, center_x: float, center_y: float, size: float) -> None:
        if player == PLAYER_NONE:
            return
        outer_color, inner_color = self._token_colors(player)
        draw_status_square_icon(
            center_x=float(center_x),
            center_y=float(center_y),
            size=float(size),
            outer_color=outer_color,
            inner_color=inner_color,
            inset=float(status_icon_inset(float(size) * 0.18)),
        )

    def _draw_hud(self) -> None:
        layout = draw_status_bar(
            width=float(SCREEN_WIDTH),
            bottom_bar_height=float(BB_HEIGHT),
            tile_size=float(self._board_layout.tile_size),
            cell_inset=float(config.STONE_INSET_RATIO * self._board_layout.tile_size),
            include_clock=False,
        )
        player = int(self._state.current_player) if not self._done else int(self._last_winner)
        if player == PLAYER_NONE:
            return
        draw_status_icon_row(
            left=float(layout.score_left),
            right=float(layout.score_right),
            center_y=float(layout.center_y),
            icon_size=float(status_icon_size(float(BB_HEIGHT), float(self._board_layout.tile_size))),
            items=[player],
            draw_item=lambda player_value, center_x, row_center_y, size: self._draw_player_icon(
                int(player_value),
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
        self._draw_tokens()
        self._draw_hover_outline()
        self._draw_hud()
        self.window_controller.flip()
