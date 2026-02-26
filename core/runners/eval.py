"""Shared evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from core.algorithms.base import Algorithm
from core.envs.base import Env


@dataclass
class EvalResult:
    episodes: int
    avg_reward: float
    avg_length: float
    wins: int


def run_eval(
    env: Env,
    algorithm: Algorithm,
    *,
    episodes: int = 10,
    max_steps_per_episode: int = 10_000,
) -> EvalResult:
    rewards: list[float] = []
    lengths: list[int] = []
    wins = 0

    for _ in range(int(episodes)):
        obs = env.reset()
        episode_reward = 0.0
        length = 0

        for _step in range(int(max_steps_per_episode)):
            action = algorithm.act(obs, explore=False)
            obs, reward, done, info = env.step(action)
            episode_reward += float(reward)
            length += 1
            if done:
                if bool(info.get("win", False)):
                    wins += 1
                break

        rewards.append(episode_reward)
        lengths.append(length)

    return EvalResult(
        episodes=int(episodes),
        avg_reward=mean(rewards) if rewards else 0.0,
        avg_length=mean(lengths) if lengths else 0.0,
        wins=int(wins),
    )
