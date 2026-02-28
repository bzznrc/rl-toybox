"""Generic off-policy training loop (qlearn, dqn, future sac)."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import logging
from statistics import mean

from core.algorithms.base import Algorithm
from core.envs.base import Env
from core.io.runs import RunPaths, write_metrics
from core.logging_utils import log_episode_line, log_key_values, log_save_line


@dataclass
class OffPolicyConfig:
    max_steps: int
    max_episodes: int | None = None
    train_after_steps: int = 0
    update_every_steps: int = 1
    updates_per_step: int = 1
    checkpoint_every_steps: int = 50_000
    reward_window: int = 100
    log_every_episodes: int = 1
    log_heartbeat_steps: int = 0


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
    best_avg_reward = float("-inf")
    update_attempts = 0
    updates = 0
    last_loss = 0.0
    last_logged_step = 0

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

        if total_steps % int(config.checkpoint_every_steps) == 0:
            algorithm.save(str(run_paths.checkpoint_path))
            log_save_line(
                kind="checkpoint",
                at=f"step {int(total_steps)}",
                path=run_paths.checkpoint_path,
            )
            last_logged_step = int(total_steps)

        if done:
            total_episodes += 1
            reward_window.append(episode_reward)
            exploration_window.append(episode_reward)

            avg_reward = float(mean(reward_window))
            exploration_avg_reward = float(mean(exploration_window))
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                algorithm.save(str(run_paths.best_path))
                log_save_line(
                    kind="best",
                    at=f"step {int(total_steps)}",
                    avg_reward=float(avg_reward),
                    path=run_paths.best_path,
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

                log_episode_line(
                    episode=int(total_episodes),
                    ep_len=int(episode_steps),
                    reward=float(episode_reward),
                    avg_reward=float(avg_reward),
                    best_avg=float(best_avg_reward if best_avg_reward > float("-inf") else avg_reward),
                    epsilon=episode_epsilon,
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

    algorithm.save(str(run_paths.checkpoint_path))
    log_save_line(
        kind="checkpoint",
        at=f"step {int(total_steps)}",
        path=run_paths.checkpoint_path,
    )

    final_metrics: dict[str, float | int] = {
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "update_attempts": update_attempts,
        "updates": updates,
        "best_avg_reward": best_avg_reward if best_avg_reward > float("-inf") else 0.0,
        "last_loss": last_loss,
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics
