"""Generic off-policy training loop (qlearn, dqn, future sac)."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import logging
from statistics import mean

from core.algorithms.base import Algorithm
from core.algorithms.exploration import bump_epsilon_to_cap
from core.envs.base import Env
from core.io.runs import RunPaths, write_metrics
from core.logging_utils import format_reward_components, log_episode_line, log_key_values, log_save_line


@dataclass
class OffPolicyConfig:
    max_steps: int
    max_episodes: int | None = None
    train_after_steps: int = 0
    update_every_steps: int = 1
    updates_per_step: int = 1
    checkpoint_every_steps: int = 50_000
    reward_window: int = 100
    min_episodes_for_stats: int = 100
    log_every_episodes: int = 1
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


def run_off_policy_training(
    env: Env,
    algorithm: Algorithm,
    run_paths: RunPaths,
    config: OffPolicyConfig,
) -> dict[str, float | int]:
    total_steps = 0
    total_episodes = 0
    obs = env.reset()

    episode_reward = 0.0
    episode_steps = 0
    reward_window_size = max(1, int(config.reward_window))
    exploration_window_size = reward_window_size
    algorithm_window = algorithm.exploration_avg_window()
    if algorithm_window is not None:
        exploration_window_size = max(1, int(algorithm_window))

    reward_window: deque[float] = deque(maxlen=reward_window_size)
    exploration_window: deque[float] = deque(maxlen=exploration_window_size)
    success_window: deque[int] = deque(maxlen=reward_window_size)
    reward_window_by_level: dict[int, deque[float]] = {}
    min_episodes_for_stats = max(0, int(config.min_episodes_for_stats))
    best_avg_reward_by_level: dict[int, float] = {}
    update_attempts = 0
    updates = 0
    last_loss = 0.0
    last_logged_step = 0
    current_level = _infer_current_level(env, default=1)

    while total_steps < int(config.max_steps):
        if config.max_episodes is not None and total_episodes >= int(config.max_episodes):
            break

        action = algorithm.act(obs, explore=True)
        next_obs, reward, done, info = env.step(action)

        transition = {
            "obs": obs,
            "action": action,
            "reward": float(reward),
            "next_obs": next_obs,
            "done": bool(done),
            "info": dict(info),
        }
        algorithm.observe(transition)

        total_steps += 1
        episode_reward += float(reward)
        episode_steps += 1
        obs = next_obs

        if total_steps >= int(config.train_after_steps) and total_steps % int(config.update_every_steps) == 0:
            for _ in range(int(config.updates_per_step)):
                update_attempts += 1
                metrics = algorithm.update()
                if "loss" in metrics:
                    last_loss = float(metrics["loss"])
                    updates += 1

        if total_steps % int(config.checkpoint_every_steps) == 0 and total_episodes >= min_episodes_for_stats:
            checkpoint_path = run_paths.model_path(level=int(current_level), kind="check")
            algorithm.save(str(checkpoint_path))
            log_save_line(
                kind="check",
                level=int(current_level),
                at=f"step {int(total_steps)}",
                path=checkpoint_path,
            )
            last_logged_step = int(total_steps)

        if done:
            total_episodes += 1
            reward_window.append(episode_reward)
            exploration_window.append(episode_reward)
            episode_level = _safe_level(info.get("level", current_level), current_level)
            level_reward_window = reward_window_by_level.setdefault(
                int(episode_level),
                deque(maxlen=reward_window_size),
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
            stats_ready = total_episodes >= min_episodes_for_stats

            avg_reward: float | None = None
            avg_success: float | None = None
            exploration_event: dict[str, float | int | str] | None = None
            if stats_ready:
                avg_reward = float(mean(reward_window))
                avg_reward_level = float(mean(level_reward_window))
                avg_success = float(mean(success_window)) if success_window else None
                exploration_avg_reward = float(mean(exploration_window))

                best_avg_level = best_avg_reward_by_level.get(int(episode_level), float("-inf"))
                if avg_reward_level > float(best_avg_level):
                    best_avg_reward_by_level[int(episode_level)] = float(avg_reward_level)
                    best_path = run_paths.model_path(level=int(episode_level), kind="best")
                    algorithm.save(str(best_path))
                    log_save_line(
                        kind="best",
                        level=int(episode_level),
                        at=f"step {int(total_steps)}",
                        avg_reward=float(avg_reward_level),
                        path=best_path,
                    )
                    last_logged_step = int(total_steps)

                exploration_event = algorithm.on_episode_end(float(exploration_avg_reward))
            if exploration_event is not None and str(exploration_event.get("bump", "off")).lower() == "on":
                epsilon = float(exploration_event.get("epsilon", 0.0))
                cooldown_steps = int(exploration_event.get("cooldown_steps", 0))
                reason = str(exploration_event.get("reason", "Plateau"))
                logging.getLogger("rl_toybox.train").info(
                    f"Explore\tBump: on\tEps: {epsilon:.2f}\tCooldown Steps: {cooldown_steps}\tReason: {reason}"
                )
                last_logged_step = int(total_steps)

            if total_episodes % max(1, int(config.log_every_episodes)) == 0:
                episode_epsilon: float | None = None
                algorithm_epsilon = getattr(algorithm, "epsilon", None)
                if algorithm_epsilon is not None:
                    episode_epsilon = float(algorithm_epsilon)
                elif exploration_event is not None and "epsilon" in exploration_event:
                    episode_epsilon = float(exploration_event["epsilon"])

                components_text = format_reward_components(info.get("reward_components"))
                best_avg_for_level = best_avg_reward_by_level.get(int(episode_level))

                log_episode_line(
                    episode=int(total_episodes),
                    level=int(episode_level),
                    ep_len=int(episode_steps),
                    reward=float(episode_reward),
                    avg_reward=avg_reward,
                    best_avg=(float(best_avg_for_level) if stats_ready and best_avg_for_level is not None else None),
                    epsilon=episode_epsilon,
                    success=int(episode_success),
                    avg_success=avg_success,
                    reward_components=components_text,
                )
                last_logged_step = int(total_steps)

            obs = env.reset()
            episode_reward = 0.0
            episode_steps = 0

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

    if total_episodes >= min_episodes_for_stats:
        current_level = _infer_current_level(env, default=current_level)
        checkpoint_path = run_paths.model_path(level=int(current_level), kind="check")
        algorithm.save(str(checkpoint_path))
        log_save_line(
            kind="check",
            level=int(current_level),
            at=f"step {int(total_steps)}",
            path=checkpoint_path,
        )

    if int(update_attempts) > 10_000 and int(updates) == 0:
        replay = getattr(algorithm, "replay", None)
        try:
            replay_len = len(replay) if replay is not None else -1
        except TypeError:
            replay_len = -1
        algo_config = getattr(algorithm, "config", None)
        batch_size = getattr(algo_config, "batch_size", "n/a")
        learn_start_steps = getattr(algo_config, "learn_start_steps", "n/a")
        if learn_start_steps == "n/a":
            learn_start_steps = int(config.train_after_steps)
        logging.getLogger("rl_toybox.train").warning(
            "Warn\tUpdates: 0\tUpdate Attempts: %s\tReplay Len: %s\tBatch Size: %s\tTrain After Steps: %s\tLearn Start Steps: %s\tUpdate Every Steps: %s",
            int(update_attempts),
            replay_len,
            batch_size,
            int(config.train_after_steps),
            learn_start_steps,
            int(config.update_every_steps),
        )

    best_avg_reward = max(best_avg_reward_by_level.values()) if best_avg_reward_by_level else float("-inf")
    final_metrics: dict[str, float | int] = {
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "update_attempts": update_attempts,
        "updates": updates,
        "best_avg_reward": best_avg_reward if best_avg_reward > float("-inf") else 0.0,
        "best_avg_reward_by_level": {int(level): float(value) for level, value in best_avg_reward_by_level.items()},
        "last_loss": last_loss,
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics

