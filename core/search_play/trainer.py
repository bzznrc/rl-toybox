"""Compact self-play training loop for search-play algorithms."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from statistics import mean

import numpy as np

from core.algorithms.base import Algorithm
from core.io.runs import RunPaths, write_metrics
from core.logging_utils import log_save_line, log_search_play_game_line
from core.search_play.interfaces import SearchPlayTrainConfig
from games.osero.rules import STONE_BLACK, STONE_EMPTY, STONE_WHITE


def _extract_action_mask(env: object, obs: object) -> np.ndarray | None:
    for method_name in ("get_action_mask", "action_mask"):
        getter = getattr(env, method_name, None)
        if not callable(getter):
            continue
        try:
            mask = getter(obs)
        except TypeError:
            mask = getter()
        if mask is None:
            return None
        return np.asarray(mask, dtype=np.bool_)
    return None


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
    last_metrics: dict[str, float] = {}

    for game_index in range(1, int(config.max_games) + 1):
        algorithm.reset_policy_state()
        obs = env.reset()
        done = False
        episode_steps = 0
        last_info: dict[str, object] = {}

        while not done:
            action_mask = _extract_action_mask(env, obs)
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
        if rolling_loss is not None and rolling_loss < float(best_loss):
            best_loss = float(rolling_loss)
            best_path = run_paths.model_path(level=1, kind="best")
            algorithm.save(str(best_path))
            log_save_line(kind="best", level=1, at=f"game {int(game_index)}", path=best_path)

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
        )

    checkpoint_path = run_paths.model_path(level=1, kind="check")
    algorithm.save(str(checkpoint_path))
    log_save_line(kind="check", level=1, at=f"game {int(config.max_games)}", path=checkpoint_path)
    if best_loss == float("inf"):
        best_path = run_paths.model_path(level=1, kind="best")
        algorithm.save(str(best_path))
        log_save_line(kind="best", level=1, at=f"game {int(config.max_games)}", path=best_path)

    final_metrics: dict[str, float | int | dict[str, int]] = {
        "total_games": int(config.max_games),
        "total_steps": int(total_steps),
        "best_loss": 0.0 if best_loss == float("inf") else float(best_loss),
        "avg_length": float(mean(length_window)) if length_window else 0.0,
        "black_win_rate": float(mean(black_results_window)) if black_results_window else 0.0,
        "draw_rate": float(mean(draw_results_window)) if draw_results_window else 0.0,
        "last_metrics": {str(key): float(value) for key, value in last_metrics.items()},
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics
