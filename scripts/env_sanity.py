"""Fast runtime and environment smoke checks for rl-toybox."""

from __future__ import annotations

import importlib
import platform
from typing import Any

import numpy as np


SMOKE_STEPS = 3
RNG = np.random.default_rng(0)


def _version_for(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return "missing"
    except ImportError as exc:
        return f"unavailable ({exc})"
    return str(getattr(module, "__version__", "installed"))


def _first_legal_action(mask: object) -> int | np.ndarray:
    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.ndim == 1:
        legal = np.flatnonzero(mask_array)
        return int(RNG.choice(legal)) if legal.size else 0
    if mask_array.ndim == 2:
        actions = []
        for row in mask_array:
            legal = np.flatnonzero(row)
            actions.append(int(RNG.choice(legal)) if legal.size else 0)
        return np.asarray(actions, dtype=np.int64)
    raise ValueError(f"action mask expected ndim 1 or 2, got {mask_array.ndim}")


def _sample_action(env: object, action_space: object, obs: object) -> Any:
    action_mask_fn = getattr(env, "get_action_mask", None)
    if callable(action_mask_fn):
        return _first_legal_action(action_mask_fn(obs))

    from core.envs.spaces import Box, Discrete

    if isinstance(action_space, Discrete):
        obs_array = np.asarray(obs)
        if obs_array.ndim >= 2 and int(obs_array.shape[0]) > 1:
            return RNG.integers(0, int(action_space.n), size=(int(obs_array.shape[0]),), dtype=np.int64)
        return int(RNG.integers(0, int(action_space.n)))
    if isinstance(action_space, Box):
        return RNG.uniform(
            action_space.low_array,
            action_space.high_array,
            size=action_space.shape,
        ).astype(np.float32)
    sample = getattr(action_space, "sample", None)
    if callable(sample):
        return sample()
    raise TypeError(f"Unsupported action space {type(action_space).__name__}")


def _validate_obs(game_id: str, obs: object) -> tuple[int, ...]:
    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.size <= 0:
        raise RuntimeError(f"{game_id}: observation is empty")
    if not np.isfinite(obs_array).all():
        raise RuntimeError(f"{game_id}: observation contains non-finite values")
    return tuple(int(axis) for axis in obs_array.shape)


def _smoke_game(game_id: str) -> None:
    from core.game import get_game_spec

    spec = get_game_spec(game_id)
    env = spec.make_env(mode="train", render=False, level=1)
    try:
        obs = env.reset()
        shape = _validate_obs(game_id, obs)
        print(f"{game_id}\treset\tobs_shape={shape}")
        done = False
        for step_index in range(1, SMOKE_STEPS + 1):
            action = _sample_action(env, spec.action_space, obs)
            obs, reward, done, info = env.step(action)
            shape = _validate_obs(game_id, obs)
            if not np.isfinite(float(reward)):
                raise RuntimeError(f"{game_id}: reward is non-finite")
            if not isinstance(info, dict):
                raise RuntimeError(f"{game_id}: info must be a dict")
            print(f"{game_id}\tstep {step_index}\treward={float(reward):.3f}\tdone={bool(done)}")
            if done:
                obs = env.reset()
                _validate_obs(game_id, obs)
                done = False
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> int:
    print(f"Python\tVersion: {platform.python_version()}")
    for module_name in ("torch", "arcade", "numpy", "pyglet", "PIL"):
        print(f"{module_name}\tVersion: {_version_for(module_name)}")
    from core.game import ACTIVE_GAME_ORDER

    for game_id in ACTIVE_GAME_ORDER:
        _smoke_game(str(game_id))
    print("Smoke\tOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
