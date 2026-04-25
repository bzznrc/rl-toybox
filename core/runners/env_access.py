"""Shared environment signal helpers used by runners and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_level(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def infer_current_level(env: object, default: int = 1) -> int:
    level_value = getattr(env, "_current_level", None)
    if level_value is None:
        game = getattr(env, "game", None)
        level_value = getattr(game, "level", None)
    return safe_level(level_value, default)


def curriculum_avg_success_for_level(env: object, level: int) -> float | None:
    curriculum = getattr(env, "_curriculum", None)
    if curriculum is None:
        return None
    episodes_in_level = getattr(curriculum, "episodes_in_level", None)
    avg_success_in_level = getattr(curriculum, "avg_success_in_level", None)
    curriculum_config = getattr(curriculum, "config", None)
    if not callable(episodes_in_level) or not callable(avg_success_in_level):
        return None
    min_episodes = max(1, int(getattr(curriculum_config, "min_episodes_per_level", 100)))
    if int(episodes_in_level(int(level))) < int(min_episodes):
        return None
    avg_success = avg_success_in_level(int(level))
    if avg_success is None:
        return None
    return float(avg_success)


def reset_policy_state(algorithm: object) -> None:
    reset_fn = getattr(algorithm, "reset_policy_state", None)
    if callable(reset_fn):
        reset_fn()


def extract_action_mask(env: object, obs: object, *, dtype: type[np.bool_] | None = np.bool_) -> Any:
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
        if dtype is None:
            return mask
        return np.asarray(mask, dtype=dtype)
    return None


def extract_centralized_state(env: object, obs: object) -> np.ndarray | None:
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


def act_with_optional_signals(
    algorithm: object,
    obs: object,
    *,
    explore: bool,
    action_mask: object | None = None,
    central_obs: object | None = None,
):
    if action_mask is None and central_obs is None:
        return algorithm.act(obs, explore=bool(explore))
    if action_mask is None:
        try:
            return algorithm.act(obs, explore=bool(explore), central_obs=central_obs)
        except TypeError:
            return algorithm.act(obs, explore=bool(explore))
    if central_obs is None:
        try:
            return algorithm.act(obs, explore=bool(explore), action_mask=action_mask)
        except TypeError:
            return algorithm.act(obs, explore=bool(explore))
    try:
        return algorithm.act(obs, explore=bool(explore), action_mask=action_mask, central_obs=central_obs)
    except TypeError:
        try:
            return algorithm.act(obs, explore=bool(explore), action_mask=action_mask)
        except TypeError:
            try:
                return algorithm.act(obs, explore=bool(explore), central_obs=central_obs)
            except TypeError:
                return algorithm.act(obs, explore=bool(explore))


def broadcast_team_signal(obs: object, value: float | bool, *, dtype: np.dtype) -> np.ndarray | float | bool:
    obs_array = np.asarray(obs)
    if obs_array.ndim == 2:
        return np.full((int(obs_array.shape[0]),), value, dtype=dtype)
    return value


def reward_scalar(reward: object) -> float:
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    if int(reward_array.size) == 0:
        return 0.0
    return float(reward_array.sum())


def reward_for_storage(obs: object, reward: object, info: object) -> np.ndarray | float:
    obs_array = np.asarray(obs)
    if isinstance(info, dict) and "reward_vec" in info:
        reward_vec = np.asarray(info.get("reward_vec"), dtype=np.float32).reshape(-1)
        if obs_array.ndim == 2:
            batch_size = int(obs_array.shape[0])
            if int(reward_vec.size) != int(batch_size):
                raise ValueError(
                    f"Runner expected info['reward_vec'] batch size {int(batch_size)}, "
                    f"got {int(reward_vec.size)}."
                )
            return reward_vec.astype(np.float32, copy=False)
        if int(reward_vec.size) > 0:
            return float(reward_vec[0])

    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    if obs_array.ndim != 2:
        if int(reward_array.size) == 0:
            return 0.0
        return float(reward_array[0])

    batch_size = int(obs_array.shape[0])
    if int(reward_array.size) == 1:
        return np.full((batch_size,), float(reward_array.item()), dtype=np.float32)
    if int(reward_array.size) != int(batch_size):
        raise ValueError(f"Runner expected reward batch size {int(batch_size)}, got {int(reward_array.size)}.")
    return reward_array.astype(np.float32, copy=False)

