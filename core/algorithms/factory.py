"""Algorithm factory helpers driven by GameSpec and algo id."""

from __future__ import annotations

from typing import Mapping

from core.algorithms.base import Algorithm
from core.algorithms.dqn.agent import DQNAlgorithm, DQNConfig
from core.algorithms.ppo.agent import PPOAlgorithm, PPOConfig
from core.algorithms.qlearn.trainer import QLearnAlgorithm, QLearnConfig
from core.algorithms.sac.agent import SACAlgorithm, SACConfig
from core.envs.spaces import Box, Discrete, Space


def build_algorithm(
    algo_id: str,
    obs_dim: int,
    action_space: Space,
    algo_config: Mapping[str, object],
) -> Algorithm:
    algo_key = str(algo_id).strip().lower()

    if algo_key == "dqn":
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
        if not isinstance(action_space, Discrete):
            raise TypeError("qlearn requires Discrete action space.")
        config = QLearnConfig(
            obs_dim=int(obs_dim),
            action_dim=int(action_space.n),
            **dict(algo_config),
        )
        return QLearnAlgorithm(config)

    if algo_key == "ppo":
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
                action_low=float(action_space.low),
                action_high=float(action_space.high),
                **config_data,
            )
            return PPOAlgorithm(config)
        raise TypeError("PPO requires Discrete or Box action space.")

    if algo_key == "sac":
        if not isinstance(action_space, Box):
            raise TypeError("SAC requires Box action space.")
        action_dim = int(action_space.shape[0])
        config = SACConfig(
            obs_dim=int(obs_dim),
            action_dim=action_dim,
            action_low=float(action_space.low),
            action_high=float(action_space.high),
            **dict(algo_config),
        )
        return SACAlgorithm(config)

    raise KeyError(f"Unsupported algorithm '{algo_id}'.")
