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
    log_episode_line,
    log_ppo_metrics_line,
    log_iteration_line,
    log_key_values,
    log_save_line,
)


@dataclass
class OnPolicyConfig:
    max_iterations: int
    rollout_steps: int = 1024
    checkpoint_every_iterations: int = 10
    reward_window: int = 100
    min_episodes_for_stats: int = 100
    log_every_iterations: int = 1
    log_heartbeat_steps: int = 0


def _safe_level(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _infer_current_level(env: Env, default: int = 1) -> int:
    level_value = getattr(env, "_current_level", None)
    if level_value is None:
        game = getattr(env, "game", None)
        level_value = getattr(game, "level", None)
    return _safe_level(level_value, default)


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


def _broadcast_team_signal(obs: object, value: float | bool, *, dtype: np.dtype):
    obs_array = np.asarray(obs)
    if obs_array.ndim == 2:
        return np.full((int(obs_array.shape[0]),), value, dtype=dtype)
    return value


def _extract_action_mask(env: Env, obs: object) -> np.ndarray | None:
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


def _extract_centralized_state(env: Env, obs: object) -> np.ndarray | None:
    for method_name in ("get_centralized_state", "centralized_state", "get_central_state", "central_state"):
        getter = getattr(env, method_name, None)
        if not callable(getter):
            continue
        try:
            state = getter(obs)
        except TypeError:
            state = getter()
        if state is None:
            return None
        return np.asarray(state, dtype=np.float32)
    return None


def _act_with_optional_mask(
    algorithm: Algorithm,
    obs: object,
    *,
    explore: bool,
    action_mask: np.ndarray | None,
    central_obs: np.ndarray | None,
):
    if action_mask is None and central_obs is None:
        return algorithm.act(obs, explore=explore)
    if action_mask is None:
        try:
            return algorithm.act(obs, explore=explore, central_obs=central_obs)
        except TypeError:
            return algorithm.act(obs, explore=explore)
    if central_obs is None:
        try:
            return algorithm.act(obs, explore=explore, action_mask=action_mask)
        except TypeError:
            return algorithm.act(obs, explore=explore)
    try:
        return algorithm.act(obs, explore=explore, action_mask=action_mask, central_obs=central_obs)
    except TypeError:
        try:
            return algorithm.act(obs, explore=explore, action_mask=action_mask)
        except TypeError:
            try:
                return algorithm.act(obs, explore=explore, central_obs=central_obs)
            except TypeError:
                return algorithm.act(obs, explore=explore)


def _reward_for_storage(obs: object, reward: object, info: object) -> np.ndarray | float:
    obs_array = np.asarray(obs)
    if isinstance(info, dict) and "reward_vec" in info:
        reward_vec = np.asarray(info.get("reward_vec"), dtype=np.float32).reshape(-1)
        if obs_array.ndim == 2:
            batch_size = int(obs_array.shape[0])
            if int(reward_vec.size) != int(batch_size):
                raise ValueError(
                    f"On-policy runner expected info['reward_vec'] batch size {int(batch_size)}, "
                    f"got {int(reward_vec.size)}."
                )
            return reward_vec.astype(np.float32, copy=False)
        if int(reward_vec.size) > 0:
            return float(reward_vec[0])

    if obs_array.ndim != 2:
        reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
        if int(reward_array.size) == 0:
            return 0.0
        return float(reward_array[0])

    batch_size = int(obs_array.shape[0])
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    if int(reward_array.size) == 1:
        return np.full((batch_size,), float(reward_array.item()), dtype=np.float32)
    if int(reward_array.size) != int(batch_size):
        raise ValueError(
            f"On-policy runner expected reward batch size {int(batch_size)}, got {int(reward_array.size)}."
        )
    return reward_array.astype(np.float32, copy=False)


def _reward_scalar(reward: object) -> float:
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    if int(reward_array.size) == 0:
        return 0.0
    return float(reward_array.sum())


def run_on_policy_training(
    env: Env,
    algorithm: Algorithm,
    run_paths: RunPaths,
    config: OnPolicyConfig,
) -> dict[str, float | int]:
    obs = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    reward_window: deque[float] = deque(maxlen=max(1, int(config.reward_window)))
    success_window_by_level: dict[int, deque[int]] = {}
    episodes_by_level: dict[int, int] = {}
    reward_window_by_level: dict[int, deque[float]] = {}
    min_episodes_for_stats = max(0, int(config.min_episodes_for_stats))
    best_avg_reward_by_level: dict[int, float] = {}
    total_steps = 0
    total_episodes = 0
    last_loss = 0.0
    last_ppo_update_metrics: dict[str, float] | None = None
    last_logged_step = 0
    current_level = _infer_current_level(env, default=1)
    _apply_level_entropy_coef(algorithm, env, int(current_level))

    for iteration in range(1, int(config.max_iterations) + 1):
        for _ in range(int(config.rollout_steps)):
            action_mask = _extract_action_mask(env, obs)
            central_obs = _extract_centralized_state(env, obs)
            action = _act_with_optional_mask(
                algorithm,
                obs,
                explore=True,
                action_mask=action_mask,
                central_obs=central_obs,
            )
            next_obs, reward, done, info = env.step(action)
            next_central_obs = _extract_centralized_state(env, next_obs)
            reward_for_storage = _reward_for_storage(obs, reward, info)
            done_for_storage = _broadcast_team_signal(obs, bool(done), dtype=np.bool_)
            algorithm.observe(
                {
                    "obs": obs,
                    "central_obs": central_obs,
                    "action": action,
                    "action_mask": action_mask,
                    "reward": reward_for_storage,
                    "next_obs": next_obs,
                    "next_central_obs": next_central_obs,
                    "done": done_for_storage,
                    "info": dict(info),
                }
            )
            total_steps += 1
            episode_reward += _reward_scalar(reward)
            episode_steps += 1
            obs = next_obs

            if done:
                total_episodes += 1
                reward_window.append(episode_reward)
                episode_level = _safe_level(info.get("level", current_level), current_level)
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
                current_level = _infer_current_level(env, default=episode_level)
                _apply_level_entropy_coef(algorithm, env, int(current_level))
                level_episode_count = int(episodes_by_level.get(int(episode_level), 0))
                stats_ready_level = level_episode_count >= int(min_episodes_for_stats)
                avg_reward_ep = float(mean(level_reward_window)) if stats_ready_level else None
                avg_success_ep = float(mean(level_success_window)) if stats_ready_level else None

                if stats_ready_level:
                    avg_reward_level = float(mean(level_reward_window))
                    best_avg_level = best_avg_reward_by_level.get(int(episode_level), float("-inf"))
                    if avg_reward_level > float(best_avg_level):
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
                        last_logged_step = int(total_steps)

                components_text = format_reward_components(info.get("reward_components"))
                best_avg_for_level = best_avg_reward_by_level.get(int(episode_level))
                log_episode_line(
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
                    epsilon=None,
                    success=int(episode_success),
                    avg_success=avg_success_ep,
                    best_avg_label=f"BR{int(episode_level)}",
                    reward_components=components_text,
                )
                cached_metrics = last_ppo_update_metrics or {}
                log_ppo_metrics_line(
                    policy_loss=(
                        float(cached_metrics["policy_loss"])
                        if "policy_loss" in cached_metrics
                        else None
                    ),
                    value_loss=(
                        float(cached_metrics["value_loss"])
                        if "value_loss" in cached_metrics
                        else None
                    ),
                    entropy=(
                        float(cached_metrics["entropy"])
                        if "entropy" in cached_metrics
                        else None
                    ),
                    approx_kl=(
                        float(cached_metrics["approx_kl"])
                        if "approx_kl" in cached_metrics
                        else None
                    ),
                    clip_frac=(
                        float(cached_metrics["clip_frac"])
                        if "clip_frac" in cached_metrics
                        else None
                    ),
                )
                last_logged_step = int(total_steps)
                obs = env.reset()
                episode_reward = 0.0
                episode_steps = 0

        metrics = algorithm.update()
        if "loss" in metrics:
            last_loss = float(metrics["loss"])
        if metrics:
            last_ppo_update_metrics = {
                str(key): float(value)
                for key, value in metrics.items()
                if str(key) in {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"}
                and isinstance(value, (int, float, np.floating))
            }

        avg_reward = float(mean(reward_window)) if reward_window else 0.0

        if iteration % int(config.checkpoint_every_iterations) == 0:
            checkpoint_path = run_paths.model_path(level=int(current_level), kind="check")
            algorithm.save(str(checkpoint_path))
            log_save_line(
                kind="check",
                level=int(current_level),
                at=f"iter {int(iteration)}",
                path=checkpoint_path,
            )
            last_logged_step = int(total_steps)

        if iteration % max(1, int(config.log_every_iterations)) == 0:
            # Iter-line logging disabled by request.
            # level_reward_window = reward_window_by_level.get(int(current_level))
            # if level_reward_window:
            #     avg_reward_level = float(mean(level_reward_window))
            # else:
            #     avg_reward_level = float(avg_reward)
            # best_avg_level = float(best_avg_reward_by_level.get(int(current_level), avg_reward_level))
            # log_iteration_line(
            #     iteration=int(iteration),
            #     steps=int(total_steps),
            #     avg_reward=float(avg_reward_level),
            #     best_avg=float(best_avg_level),
            #     best_avg_label=f"BR{int(current_level)}",
            # )
            # last_logged_step = int(total_steps)
            pass

        if int(config.log_heartbeat_steps) > 0 and (int(total_steps) - int(last_logged_step)) >= int(config.log_heartbeat_steps):
            log_key_values(
                "rl_toybox.train",
                {
                    "Heartbeat": "on",
                    "Steps": int(total_steps),
                    "Episodes": int(total_episodes),
                },
                key_value_separator=":",
            )
            last_logged_step = int(total_steps)

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
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "best_avg_reward": best_avg_reward if best_avg_reward > float("-inf") else 0.0,
        "best_avg_reward_by_level": {int(level): float(value) for level, value in best_avg_reward_by_level.items()},
        "last_loss": last_loss,
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics
