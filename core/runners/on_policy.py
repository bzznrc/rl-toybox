"""Generic on-policy training loop (ppo)."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean

import numpy as np

from core.algorithms.base import Algorithm
from core.algorithms.exploration import bump_epsilon_to_cap
from core.envs.base import Env
from core.io.runs import RunPaths, write_metrics
from core.logging_utils import (
    format_reward_components,
    log_on_policy_episode_line,
    log_ppo_metrics_line,
    log_ppo_update_line,
    log_save_line,
)
from core.runners.env_access import (
    act_with_optional_signals,
    broadcast_team_signal,
    curriculum_avg_success_for_level,
    extract_action_mask,
    extract_centralized_state,
    infer_current_level,
    reset_policy_state,
    reward_for_storage,
    reward_scalar,
    safe_level,
)


@dataclass
class OnPolicyConfig:
    max_iterations: int
    rollout_steps: int = 1024
    checkpoint_every_iterations: int = 10
    reward_window: int = 100
    min_episodes_for_stats: int = 100


def _apply_level_entropy_coef(algorithm: Algorithm, env: Env, level: int) -> float | None:
    getter = getattr(env, "get_entropy_coef_for_level", None)
    if not callable(getter):
        return None

    try:
        entropy_coef = getter(int(level))
    except TypeError:
        entropy_coef = getter()
    except Exception:
        return None

    if entropy_coef is None:
        return None

    config = getattr(algorithm, "config", None)
    if config is None or not hasattr(config, "entropy_coef"):
        return None

    try:
        entropy_value = float(entropy_coef)
    except (TypeError, ValueError):
        return None

    setattr(config, "entropy_coef", entropy_value)
    return entropy_value


def _metric_float(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(str(key))
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return None


def _should_log_ppo_debug_metrics(env: Env) -> bool:
    return bool(getattr(env, "log_ppo_metrics_line", False))


def run_on_policy_training(
    env: Env,
    algorithm: Algorithm,
    run_paths: RunPaths,
    config: OnPolicyConfig,
) -> dict[str, float | int]:
    reset_policy_state(algorithm)
    obs = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    reward_window: deque[float] = deque(maxlen=max(1, int(config.reward_window)))
    success_window_by_level: dict[int, deque[int]] = {}
    episodes_by_level: dict[int, int] = {}
    reward_window_by_level: dict[int, deque[float]] = {}
    min_episodes_for_stats = max(0, int(config.min_episodes_for_stats))
    best_avg_reward_by_level: dict[int, float] = {}
    best_avg_success_by_level: dict[int, float] = {}
    total_steps = 0
    total_episodes = 0
    total_updates = 0
    last_loss = 0.0
    current_level = infer_current_level(env, default=1)
    _apply_level_entropy_coef(algorithm, env, int(current_level))

    for iteration in range(1, int(config.max_iterations) + 1):
        for _ in range(int(config.rollout_steps)):
            action_mask = extract_action_mask(env, obs)
            central_obs = extract_centralized_state(env, obs)
            action = act_with_optional_signals(
                algorithm,
                obs,
                explore=True,
                action_mask=action_mask,
                central_obs=central_obs,
            )
            next_obs, reward, done, info = env.step(action)
            next_central_obs = extract_centralized_state(env, next_obs)
            storage_reward = reward_for_storage(obs, reward, info)
            done_for_storage = broadcast_team_signal(obs, bool(done), dtype=np.bool_)
            algorithm.observe(
                {
                    "obs": obs,
                    "central_obs": central_obs,
                    "action": action,
                    "action_mask": action_mask,
                    "reward": storage_reward,
                    "next_obs": next_obs,
                    "next_central_obs": next_central_obs,
                    "done": done_for_storage,
                    "info": dict(info),
                }
            )
            total_steps += 1
            episode_reward += reward_scalar(reward)
            episode_steps += 1
            obs = next_obs

            if done:
                total_episodes += 1
                reward_window.append(episode_reward)
                episode_level = safe_level(info.get("level", current_level), current_level)
                level_reward_window = reward_window_by_level.setdefault(
                    int(episode_level),
                    deque(maxlen=max(1, int(config.reward_window))),
                )
                level_reward_window.append(episode_reward)
                try:
                    episode_success = 1 if int(info.get("success", 0)) > 0 else 0
                except (TypeError, ValueError):
                    episode_success = 1 if bool(info.get("win", False)) else 0
                level_success_window = success_window_by_level.setdefault(
                    int(episode_level),
                    deque(maxlen=max(1, int(config.reward_window))),
                )
                level_success_window.append(int(episode_success))
                episodes_by_level[int(episode_level)] = int(episodes_by_level.get(int(episode_level), 0)) + 1
                if bool(info.get("level_changed", False)):
                    bump_epsilon_to_cap(algorithm)
                current_level = infer_current_level(env, default=episode_level)
                _apply_level_entropy_coef(algorithm, env, int(current_level))
                level_episode_count = int(episodes_by_level.get(int(episode_level), 0))
                stats_ready_level = level_episode_count >= int(min_episodes_for_stats)
                avg_reward_ep = float(mean(level_reward_window)) if stats_ready_level else None
                avg_success_ep = curriculum_avg_success_for_level(env, int(episode_level))
                if avg_success_ep is None and stats_ready_level:
                    avg_success_ep = float(mean(level_success_window)) if level_success_window else None

                if stats_ready_level:
                    avg_reward_level = float(mean(level_reward_window))
                    avg_success_level = (
                        float(avg_success_ep)
                        if avg_success_ep is not None
                        else float(mean(level_success_window)) if level_success_window else 0.0
                    )
                    best_success_level = best_avg_success_by_level.get(int(episode_level), float("-inf"))
                    best_reward_level = best_avg_reward_by_level.get(int(episode_level), float("-inf"))
                    if (
                        avg_success_level > float(best_success_level)
                        or (
                            avg_success_level == float(best_success_level)
                            and avg_reward_level > float(best_reward_level)
                        )
                    ):
                        best_avg_success_by_level[int(episode_level)] = float(avg_success_level)
                        best_avg_reward_by_level[int(episode_level)] = float(avg_reward_level)
                        best_path = run_paths.model_path(level=int(episode_level), kind="best")
                        algorithm.save(str(best_path))
                        log_save_line(
                            kind="best",
                            level=int(episode_level),
                            at=f"iter {int(iteration)}",
                            avg_reward=float(avg_reward_level),
                            path=best_path,
                        )

                components_text = format_reward_components(info.get("reward_components"))
                best_avg_for_level = best_avg_reward_by_level.get(int(episode_level))
                log_on_policy_episode_line(
                    episode=int(total_episodes),
                    level=int(episode_level),
                    ep_len=int(episode_steps),
                    reward=float(episode_reward),
                    avg_reward=avg_reward_ep,
                    best_avg=(
                        float(best_avg_for_level)
                        if stats_ready_level and best_avg_for_level is not None
                        else None
                    ),
                    success=int(episode_success),
                    avg_success=avg_success_ep,
                    best_avg_label=f"BR{int(episode_level)}",
                    reward_components=components_text,
                )
                obs = env.reset()
                reset_policy_state(algorithm)
                episode_reward = 0.0
                episode_steps = 0

        metrics = algorithm.update()
        if "loss" in metrics:
            last_loss = float(metrics["loss"])
        if metrics:
            total_updates += 1
            log_ppo_update_line(
                update=int(total_updates),
                level=int(current_level),
                steps=int(total_steps),
                policy_loss=_metric_float(metrics, "policy_loss"),
                value_loss=_metric_float(metrics, "value_loss"),
                explained_variance=_metric_float(metrics, "explained_variance"),
                entropy=_metric_float(metrics, "entropy"),
                approx_kl=_metric_float(metrics, "approx_kl"),
            )
            if _should_log_ppo_debug_metrics(env):
                log_ppo_metrics_line(
                    policy_loss=_metric_float(metrics, "policy_loss"),
                    value_loss=_metric_float(metrics, "value_loss"),
                    entropy=_metric_float(metrics, "entropy"),
                    approx_kl=_metric_float(metrics, "approx_kl"),
                    clip_frac=_metric_float(metrics, "clip_frac"),
                )

        if iteration % int(config.checkpoint_every_iterations) == 0:
            checkpoint_path = run_paths.model_path(level=int(current_level), kind="check")
            algorithm.save(str(checkpoint_path))
            log_save_line(
                kind="check",
                level=int(current_level),
                at=f"iter {int(iteration)}",
                path=checkpoint_path,
            )
    checkpoint_path = run_paths.model_path(level=int(current_level), kind="check")
    algorithm.save(str(checkpoint_path))
    log_save_line(
        kind="check",
        level=int(current_level),
        at=f"iter {int(config.max_iterations)}",
        path=checkpoint_path,
    )

    best_avg_reward = max(best_avg_reward_by_level.values()) if best_avg_reward_by_level else float("-inf")
    final_metrics: dict[str, float | int] = {
        "iterations": int(config.max_iterations),
        "updates": int(total_updates),
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "best_avg_reward": best_avg_reward if best_avg_reward > float("-inf") else 0.0,
        "best_avg_reward_by_level": {int(level): float(value) for level, value in best_avg_reward_by_level.items()},
        "best_avg_success_by_level": {int(level): float(value) for level, value in best_avg_success_by_level.items()},
        "last_loss": last_loss,
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics
