"""Compact self-play training loop for search-play algorithms."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from statistics import mean

import numpy as np

from core.algorithms.base import Algorithm
from core.io.runs import RunPaths, write_metrics
from core.logging_utils import format_reward_components, log_arena_line, log_save_line, log_search_play_game_line
from core.runners.env_access import extract_action_mask
from core.search_play.interfaces import SearchPlayTrainConfig
from games.flip.rules import (
    PLAYER_NONE,
    PLAYER_ONE,
    PLAYER_TWO,
    FlipState,
    apply_action,
    build_action_mask,
    flips_for_action,
    initial_state,
    is_winning_action,
    is_terminal_state,
    legal_actions,
    observation_from_state,
    outcome_for_player,
)


def _aggregate_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = {key for row in metric_rows for key in row.keys()}
    return {
        str(key): float(mean([float(row[key]) for row in metric_rows if key in row]))
        for key in sorted(keys)
    }


def _winner_label(winner_value: object) -> str:
    try:
        winner_int = int(winner_value)
    except (TypeError, ValueError):
        return "Unknown"
    if winner_int == PLAYER_ONE:
        return "P1"
    if winner_int == PLAYER_TWO:
        return "P2"
    if winner_int == PLAYER_NONE:
        return "Draw"
    return str(winner_int)


def _opponent_action_random(state: FlipState) -> int:
    mask = build_action_mask(state.board, int(state.current_player))
    mask_actions = np.flatnonzero(mask)
    if int(mask_actions.size) <= 0:
        return 0
    return int(np.random.choice(mask_actions))


def _opponent_action_greedy(state: FlipState) -> int:
    actions = legal_actions(state.board, int(state.current_player))
    if not actions:
        return 0
    for action in actions:
        if is_winning_action(state, int(action)):
            return int(action)

    opponent_state = FlipState(
        board=state.board,
        current_player=-int(state.current_player),
        move_count=int(state.move_count),
        pass_count=int(state.pass_count),
    )
    for action in actions:
        if is_winning_action(opponent_state, int(action)):
            return int(action)

    center_row = (float(getattr(state.board, "shape", (6, 6))[0]) - 1.0) * 0.5
    center_col = (float(getattr(state.board, "shape", (6, 6))[1]) - 1.0) * 0.5
    return int(
        max(
            actions,
            key=lambda action: (
                len(flips_for_action(state, int(action))),
                -abs(float(action // int(state.board.shape[1])) - center_row)
                - abs(float(action % int(state.board.shape[1])) - center_col),
            ),
        )
    )


def _arena_opponent_action(state: FlipState, opponent: str) -> int:
    opponent_key = str(opponent).strip().lower()
    if opponent_key == "greedy":
        return _opponent_action_greedy(state)
    return _opponent_action_random(state)


def _play_arena_game(
    algorithm: Algorithm,
    *,
    board_rows: int,
    board_cols: int,
    agent_player: int,
    opponent: str,
) -> float:
    state = initial_state(int(board_rows), int(board_cols))
    algorithm.reset_policy_state()
    while not is_terminal_state(state):
        if int(state.current_player) == int(agent_player):
            obs = observation_from_state(state)
            mask = build_action_mask(state.board, int(state.current_player))
            try:
                action = algorithm.act(obs, explore=False, action_mask=mask)
            except TypeError:
                action = algorithm.act(obs, explore=False)
            try:
                state = apply_action(state, int(action))
            except ValueError:
                state = apply_action(state, int(np.flatnonzero(mask)[0]))
        else:
            state = apply_action(state, _arena_opponent_action(state, opponent))
    outcome = float(outcome_for_player(state.board, int(agent_player)))
    if outcome > 0.0:
        return 1.0
    if outcome < 0.0:
        return 0.0
    return 0.5


def _evaluate_arena(algorithm: Algorithm, *, games_per_opponent: int) -> dict[str, float]:
    config = getattr(algorithm, "config", None)
    board_rows = int(getattr(config, "board_rows", 6))
    board_cols = int(getattr(config, "board_cols", 6))
    games_each = max(1, int(games_per_opponent))
    scores: list[float] = []
    by_opponent: dict[str, float] = {}
    for opponent in ("random", "greedy"):
        opponent_scores: list[float] = []
        for game_idx in range(games_each):
            agent_player = PLAYER_ONE if game_idx % 2 == 0 else PLAYER_TWO
            score = _play_arena_game(
                algorithm,
                board_rows=int(board_rows),
                board_cols=int(board_cols),
                agent_player=int(agent_player),
                opponent=str(opponent),
            )
            opponent_scores.append(float(score))
            scores.append(float(score))
        by_opponent[f"{opponent}_score"] = float(mean(opponent_scores)) if opponent_scores else 0.0
    by_opponent["arena_score"] = float(mean(scores)) if scores else 0.0
    return by_opponent


def run_search_play_training(
    env: object,
    algorithm: Algorithm,
    run_paths: RunPaths,
    config: SearchPlayTrainConfig,
) -> dict[str, float | int | dict[str, int]]:
    total_steps = 0
    loss_window: deque[float] = deque(maxlen=20)
    length_window: deque[int] = deque(maxlen=20)
    first_player_results_window: deque[int] = deque(maxlen=50)
    draw_results_window: deque[int] = deque(maxlen=50)
    best_loss = float("inf")
    best_arena_score = float("-inf")
    last_metrics: dict[str, float] = {}
    last_arena_metrics: dict[str, float] = {}

    for game_index in range(1, int(config.max_games) + 1):
        algorithm.reset_policy_state()
        obs = env.reset()
        done = False
        episode_steps = 0
        last_info: dict[str, object] = {}

        while not done:
            action_mask = extract_action_mask(env, obs)
            try:
                action = algorithm.act(obs, explore=True, action_mask=action_mask)
            except TypeError:
                action = algorithm.act(obs, explore=True)
            next_obs, reward, done, info = env.step(action)
            algorithm.observe(
                {
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "done": bool(done),
                    "info": dict(info),
                }
            )
            obs = next_obs
            last_info = dict(info)
            total_steps += 1
            episode_steps += 1

        update_rows: list[dict[str, float]] = []
        if int(game_index) >= int(config.train_after_games):
            for _ in range(int(config.updates_per_game)):
                metrics = algorithm.update()
                if metrics:
                    update_rows.append({str(key): float(value) for key, value in metrics.items()})
        aggregated_metrics = _aggregate_metrics(update_rows)
        if "loss" in aggregated_metrics:
            loss_window.append(float(aggregated_metrics["loss"]))
            last_metrics = dict(aggregated_metrics)

        length_window.append(int(episode_steps))
        winner_value = last_info.get("winner", PLAYER_NONE)
        first_player_results_window.append(1 if int(winner_value) == PLAYER_ONE else 0)
        draw_results_window.append(1 if int(winner_value) == PLAYER_NONE else 0)

        rolling_loss = float(mean(loss_window)) if loss_window else None
        if rolling_loss is not None:
            best_loss = min(float(best_loss), float(rolling_loss))

        arena_score: float | None = None
        arena_every = max(0, int(config.arena_every_games))
        if arena_every > 0 and int(game_index) >= int(config.train_after_games) and int(game_index) % arena_every == 0:
            last_arena_metrics = _evaluate_arena(
                algorithm,
                games_per_opponent=int(config.arena_games_per_opponent),
            )
            arena_score = float(last_arena_metrics.get("arena_score", 0.0))
            log_arena_line(score=float(arena_score), metrics=last_arena_metrics)
        if arena_score is not None and float(arena_score) > float(best_arena_score):
            best_arena_score = float(arena_score)
            best_path = run_paths.model_path(level=1, kind="best")
            algorithm.save(str(best_path))
            log_save_line(kind="best", level=1, at=f"game {int(game_index)}", avg_reward=float(arena_score), path=best_path)

        if int(game_index) % int(config.checkpoint_every_games) == 0:
            checkpoint_path = run_paths.model_path(level=1, kind="check")
            algorithm.save(str(checkpoint_path))
            log_save_line(kind="check", level=1, at=f"game {int(game_index)}", path=checkpoint_path)

        log_search_play_game_line(
            game=int(game_index),
            moves=int(episode_steps),
            winner=_winner_label(winner_value),
            first_player_win_rate=(
                float(mean(first_player_results_window)) if first_player_results_window else None
            ),
            draw_rate=float(mean(draw_results_window)) if draw_results_window else None,
            avg_length=float(mean(length_window)) if length_window else None,
            loss=None if rolling_loss is None else float(rolling_loss),
            policy_loss=(
                float(aggregated_metrics["policy_loss"])
                if "policy_loss" in aggregated_metrics
                else None
            ),
            value_loss=(
                float(aggregated_metrics["value_loss"])
                if "value_loss" in aggregated_metrics
                else None
            ),
            reward_components=format_reward_components(last_info.get("reward_components")),
        )

    checkpoint_path = run_paths.model_path(level=1, kind="check")
    algorithm.save(str(checkpoint_path))
    log_save_line(kind="check", level=1, at=f"game {int(config.max_games)}", path=checkpoint_path)
    if best_arena_score == float("-inf"):
        best_path = run_paths.model_path(level=1, kind="best")
        algorithm.save(str(best_path))
        log_save_line(kind="best", level=1, at=f"game {int(config.max_games)}", path=best_path)

    final_metrics: dict[str, float | int | dict[str, int]] = {
        "total_games": int(config.max_games),
        "total_steps": int(total_steps),
        "best_loss": 0.0 if best_loss == float("inf") else float(best_loss),
        "best_arena_score": 0.0 if best_arena_score == float("-inf") else float(best_arena_score),
        "avg_length": float(mean(length_window)) if length_window else 0.0,
        "first_player_win_rate": float(mean(first_player_results_window)) if first_player_results_window else 0.0,
        "draw_rate": float(mean(draw_results_window)) if draw_results_window else 0.0,
        "last_metrics": {str(key): float(value) for key, value in last_metrics.items()},
        "last_arena_metrics": {str(key): float(value) for key, value in last_arena_metrics.items()},
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics
