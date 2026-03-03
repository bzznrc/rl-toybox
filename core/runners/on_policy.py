"""Generic on-policy training loop (ppo)."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean

from core.algorithms.base import Algorithm
from core.algorithms.exploration import bump_epsilon_to_cap
from core.envs.base import Env
from core.io.runs import RunPaths, write_metrics
from core.logging_utils import (
    format_reward_components,
    log_episode_line,
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
    success_window: deque[int] = deque(maxlen=max(1, int(config.reward_window)))
    reward_window_by_level: dict[int, deque[float]] = {}
    min_episodes_for_stats = 100
    best_avg_reward_by_level: dict[int, float] = {}
    total_steps = 0
    total_episodes = 0
    last_loss = 0.0
    last_logged_step = 0
    current_level = _infer_current_level(env, default=1)

    for iteration in range(1, int(config.max_iterations) + 1):
        for _ in range(int(config.rollout_steps)):
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
            total_steps += 1
            episode_reward += float(reward)
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
                success_window.append(int(episode_success))
                if bool(info.get("level_changed", False)):
                    bump_epsilon_to_cap(algorithm)
                current_level = _infer_current_level(env, default=episode_level)
                stats_ready = total_episodes >= int(min_episodes_for_stats)
                avg_reward_ep = float(mean(reward_window)) if stats_ready else None
                avg_success_ep = float(mean(success_window)) if stats_ready else None

                if stats_ready:
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
                    best_avg=(float(best_avg_for_level) if stats_ready and best_avg_for_level is not None else None),
                    epsilon=None,
                    success=int(episode_success),
                    avg_success=avg_success_ep,
                    reward_components=components_text,
                )
                last_logged_step = int(total_steps)
                obs = env.reset()
                episode_reward = 0.0
                episode_steps = 0

        metrics = algorithm.update()
        if "loss" in metrics:
            last_loss = float(metrics["loss"])

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
            best_global = max(best_avg_reward_by_level.values()) if best_avg_reward_by_level else avg_reward
            log_iteration_line(
                iteration=int(iteration),
                steps=int(total_steps),
                episodes=int(total_episodes),
                avg_reward=float(avg_reward),
                best_avg=float(best_global),
            )
            last_logged_step = int(total_steps)

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

