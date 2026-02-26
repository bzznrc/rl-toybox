"""Generic off-policy training loop (qlearn, dqn, future sac)."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean

from core.algorithms.base import Algorithm
from core.envs.base import Env
from core.io.runs import RunPaths, write_metrics


@dataclass
class OffPolicyConfig:
    max_steps: int
    max_episodes: int | None = None
    train_after_steps: int = 0
    update_every_steps: int = 1
    updates_per_step: int = 1
    checkpoint_every_steps: int = 50_000
    reward_window: int = 100


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
    reward_window: deque[float] = deque(maxlen=max(1, int(config.reward_window)))
    best_avg_reward = float("-inf")
    update_attempts = 0
    updates = 0
    last_loss = 0.0

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

        if done:
            total_episodes += 1
            reward_window.append(episode_reward)

            avg_reward = mean(reward_window)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                algorithm.save(str(run_paths.best_path))

            obs = env.reset()
            episode_reward = 0.0

    algorithm.save(str(run_paths.checkpoint_path))

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
