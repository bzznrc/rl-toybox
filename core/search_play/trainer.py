"""Compact self-play training loop for search-play algorithms."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from statistics import mean

import numpy as np

from core.algorithms.base import Algorithm
from core.io.runs import RunPaths, write_metrics
from core.logging_utils import format_reward_components, log_save_line, log_search_play_game_line
from core.runners.env_access import extract_action_mask
from core.search_play.interfaces import SearchPlayTrainConfig
from games.osero.rules import (
    STONE_BLACK,
    STONE_EMPTY,
    STONE_WHITE,
    OseroState,
    apply_action,
    build_action_mask,
    initial_state,
    is_terminal_state,
    observation_from_state,
    outcome_for_player,
    stone_counts,
    winner,
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
        return "unknown"
    if winner_int == STONE_BLACK:
        return "black"
    if winner_int == STONE_WHITE:
        return "white"
    if winner_int == STONE_EMPTY:
        return "draw"
    return str(winner_int)


def _opponent_action_random(state: OseroState) -> int:
    mask = build_action_mask(state.board, int(state.current_player))
    legal_actions = np.flatnonzero(mask)
    if int(legal_actions.size) <= 0:
        return int(mask.size - 1)
    return int(np.random.choice(legal_actions))


def _opponent_action_greedy(state: OseroState) -> int:
    mask = build_action_mask(state.board, int(state.current_player))
    legal_actions = np.flatnonzero(mask)
    if int(legal_actions.size) <= 0:
        return int(mask.size - 1)
    best_action = int(legal_actions[0])
    best_score = float("-inf")
    for action in legal_actions:
        try:
            next_state = apply_action(state, int(action))
        except ValueError:
            continue
        black_count, white_count = stone_counts(next_state.board)
        score = (black_count - white_count) * int(state.current_player)
        if float(score) > float(best_score):
            best_score = float(score)
            best_action = int(action)
    return int(best_action)


def _arena_opponent_action(state: OseroState, opponent: str) -> int:
    opponent_key = str(opponent).strip().lower()
    if opponent_key == "greedy":
        return _opponent_action_greedy(state)
    return _opponent_action_random(state)


def _play_arena_game(algorithm: Algorithm, *, board_size: int, agent_player: int, opponent: str) -> float:
    state = initial_state(int(board_size))
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
    board_size = int(getattr(config, "board_size", 6))
    games_each = max(1, int(games_per_opponent))
    scores: list[float] = []
    by_opponent: dict[str, float] = {}
    for opponent in ("random", "greedy"):
        opponent_scores: list[float] = []
        for game_idx in range(games_each):
            agent_player = STONE_BLACK if game_idx % 2 == 0 else STONE_WHITE
            score = _play_arena_game(
                algorithm,
                board_size=int(board_size),
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
    black_results_window: deque[int] = deque(maxlen=50)
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
        winner_value = last_info.get("winner", STONE_EMPTY)
        black_results_window.append(1 if int(winner_value) == STONE_BLACK else 0)
        draw_results_window.append(1 if int(winner_value) == STONE_EMPTY else 0)

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
            black_win_rate=float(mean(black_results_window)) if black_results_window else None,
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
            arena_score=arena_score,
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
        "black_win_rate": float(mean(black_results_window)) if black_results_window else 0.0,
        "draw_rate": float(mean(draw_results_window)) if draw_results_window else 0.0,
        "last_metrics": {str(key): float(value) for key, value in last_metrics.items()},
        "last_arena_metrics": {str(key): float(value) for key, value in last_arena_metrics.items()},
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics
