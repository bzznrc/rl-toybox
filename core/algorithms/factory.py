"""Algorithm factory helpers that dispatch into the family-specific implementations."""

from __future__ import annotations

from typing import Mapping

from core.algorithms.base import Algorithm
from core.envs.spaces import Box, Discrete, Space


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
        from core.search_play.agent import SearchPlayAlgorithm, SearchPlayConfig

        if not isinstance(action_space, Discrete):
            raise TypeError("search_play requires Discrete action space.")
        config = SearchPlayConfig(**dict(algo_config))
        expected_action_dim = int(config.action_dim)
        if int(action_space.n) != expected_action_dim:
            raise ValueError(
                "search_play action dim does not match game config. "
                f"Expected {expected_action_dim}, got {int(action_space.n)}."
            )
        expected_obs_dim = int(config.board_rows) * int(config.board_cols)
        if int(obs_dim) != expected_obs_dim:
            raise ValueError(
                "search_play observation dim does not match board shape. "
                f"Expected {expected_obs_dim}, got {int(obs_dim)}."
            )
        return SearchPlayAlgorithm(config)

    raise KeyError(f"Unsupported algorithm '{algo_id}'.")
