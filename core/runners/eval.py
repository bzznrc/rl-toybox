"""Shared evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


from core.algorithms.base import Algorithm
from core.envs.base import Env
from core.runners.env_access import (
    act_with_optional_signals,
    extract_action_mask,
    extract_centralized_state,
    reset_policy_state,
    reward_scalar,
)


@dataclass
class EvalResult:
    episodes: int
    avg_reward: float
    avg_length: float
    wins: int


def reset_eval_policy_state(algorithm: Algorithm) -> None:
    reset_policy_state(algorithm)


def select_eval_action(env: Env, algorithm: Algorithm, obs: object):
    action_mask = extract_action_mask(env, obs)
    central_obs = extract_centralized_state(env, obs)
    return act_with_optional_signals(
        algorithm,
        obs,
        explore=False,
        action_mask=action_mask,
        central_obs=central_obs,
    )


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
        reset_eval_policy_state(algorithm)
        obs = env.reset()
        episode_reward = 0.0
        length = 0

        for _step in range(int(max_steps_per_episode)):
            action = select_eval_action(env, algorithm, obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward_scalar(reward)
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
