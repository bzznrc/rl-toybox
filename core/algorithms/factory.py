"""Algorithm factory helpers driven by GameSpec and algo id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from core.algorithms.base import Algorithm
from core.envs.spaces import Box, Discrete, Space


ON_POLICY_ALGO_IDS = frozenset({"ppo", "recurrent_ppo", "a2c", "mappo"})
OFF_POLICY_ALGO_IDS = frozenset({"qlearn", "dqn", "sac", "search_play"})


def is_on_policy_algo(algo_id: str) -> bool:
    return str(algo_id).strip().lower() in ON_POLICY_ALGO_IDS


class PlaceholderAlgorithm(Algorithm):
    """No-op algorithm used for scaffold-first entries that are not implemented yet."""

    def __init__(self, *, algo_id: str, action_space: Space):
        self.algo_id = str(algo_id)
        self._action_space = action_space

    def act(self, obs: np.ndarray, explore: bool) -> int | np.ndarray:
        del obs, explore
        if isinstance(self._action_space, Discrete):
            return 0
        return np.zeros(self._action_space.shape, dtype=np.float32)

    def observe(self, transition: dict[str, object]) -> None:
        del transition

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        payload = {"algo_id": self.algo_id, "placeholder": True}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        _ = Path(path).read_text(encoding="utf-8")


def build_algorithm(
    algo_id: str,
    obs_dim: int,
    action_space: Space,
    algo_config: Mapping[str, object],
) -> Algorithm:
    algo_key = str(algo_id).strip().lower()

    if algo_key == "dqn":
        from core.value_discrete.dqn.agent import DQNAlgorithm, DQNConfig

        if not isinstance(action_space, Discrete):
            raise TypeError("DQN requires Discrete action space.")
        config_data = dict(algo_config)
        # Centralize update cadence in the off-policy runner.
        config_data.pop("learn_start_steps", None)
        config_data.pop("train_every_steps", None)
        config = DQNConfig(
            obs_dim=int(obs_dim),
            action_dim=int(action_space.n),
            **config_data,
        )
        return DQNAlgorithm(config)

    if algo_key == "qlearn":
        from core.value_discrete.q_learning.trainer import QLearnAlgorithm, QLearnConfig

        if not isinstance(action_space, Discrete):
            raise TypeError("qlearn requires Discrete action space.")
        config = QLearnConfig(
            obs_dim=int(obs_dim),
            action_dim=int(action_space.n),
            **dict(algo_config),
        )
        return QLearnAlgorithm(config)

    if algo_key in {"ppo", "recurrent_ppo", "a2c", "mappo"}:
        from core.actor_critic.ppo.agent import PPOAlgorithm, PPOConfig

        config_data = dict(algo_config)
        if isinstance(action_space, Discrete):
            config = PPOConfig(
                obs_dim=int(obs_dim),
                action_dim=int(action_space.n),
                action_type="discrete",
                **config_data,
            )
            return PPOAlgorithm(config)
        if isinstance(action_space, Box):
            action_dim = 1
            for axis in action_space.shape:
                action_dim *= max(1, int(axis))
            config = PPOConfig(
                obs_dim=int(obs_dim),
                action_dim=int(action_dim),
                action_type="continuous",
                action_low=action_space.low_array.reshape(-1).copy(),
                action_high=action_space.high_array.reshape(-1).copy(),
                **config_data,
            )
            return PPOAlgorithm(config)
        raise TypeError("PPO requires Discrete or Box action space.")

    if algo_key == "sac":
        from core.actor_critic.sac.agent import SACAlgorithm, SACConfig

        if not isinstance(action_space, Box):
            raise TypeError("SAC requires Box action space.")
        action_dim = int(action_space.shape[0])
        config = SACConfig(
            obs_dim=int(obs_dim),
            action_dim=action_dim,
            action_low=action_space.low_array.reshape(-1).copy(),
            action_high=action_space.high_array.reshape(-1).copy(),
            **dict(algo_config),
        )
        return SACAlgorithm(config)

    if algo_key == "search_play":
        return PlaceholderAlgorithm(algo_id=algo_key, action_space=action_space)

    raise KeyError(f"Unsupported algorithm '{algo_id}'.")
