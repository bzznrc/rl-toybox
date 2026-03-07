"""Shared evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np

from core.algorithms.base import Algorithm
from core.envs.base import Env


@dataclass
class EvalResult:
    episodes: int
    avg_reward: float
    avg_length: float
    wins: int


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


def _reward_scalar(reward: object) -> float:
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    if int(reward_array.size) == 0:
        return 0.0
    return float(reward_array.sum())


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
            action_mask = _extract_action_mask(env, obs)
            central_obs = _extract_centralized_state(env, obs)
            action = _act_with_optional_mask(
                algorithm,
                obs,
                explore=False,
                action_mask=action_mask,
                central_obs=central_obs,
            )
            obs, reward, done, info = env.step(action)
            episode_reward += _reward_scalar(reward)
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
