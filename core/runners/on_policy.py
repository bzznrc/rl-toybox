"""Generic on-policy training loop (ppo)."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean

from core.algorithms.base import Algorithm
from core.envs.base import Env
from core.io.runs import RunPaths, write_metrics


@dataclass
class OnPolicyConfig:
    max_iterations: int
    rollout_steps: int = 1024
    checkpoint_every_iterations: int = 10
    reward_window: int = 100


def run_on_policy_training(
    env: Env,
    algorithm: Algorithm,
    run_paths: RunPaths,
    config: OnPolicyConfig,
) -> dict[str, float | int]:
    obs = env.reset()
    episode_reward = 0.0
    reward_window: deque[float] = deque(maxlen=max(1, int(config.reward_window)))
    best_avg_reward = float("-inf")
    total_steps = 0
    total_episodes = 0
    last_loss = 0.0

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
            obs = next_obs

            if done:
                total_episodes += 1
                reward_window.append(episode_reward)
                obs = env.reset()
                episode_reward = 0.0

        metrics = algorithm.update()
        if "loss" in metrics:
            last_loss = float(metrics["loss"])

        avg_reward = mean(reward_window) if reward_window else 0.0
        if reward_window and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            algorithm.save(str(run_paths.best_path))

        if iteration % int(config.checkpoint_every_iterations) == 0:
            algorithm.save(str(run_paths.checkpoint_path))

    algorithm.save(str(run_paths.checkpoint_path))

    final_metrics: dict[str, float | int] = {
        "iterations": int(config.max_iterations),
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "best_avg_reward": best_avg_reward if best_avg_reward > float("-inf") else 0.0,
        "last_loss": last_loss,
        "config": asdict(config),
    }
    write_metrics(run_paths.metrics_path, final_metrics)
    return final_metrics
