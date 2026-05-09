"""Search-play algorithm wrapper for compact self-play games."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.algorithms.base import Algorithm
from core.io.checkpoint import load_torch_checkpoint, save_torch_checkpoint
from core.search_play.interfaces import MCTSConfig, ReplaySample
from core.search_play.mcts import run_mcts
from core.search_play.networks import build_policy_value_network
from games.flip.rules import (
    PLAYER_NONE,
    action_mask_from_observation,
    canonical_board_from_observation,
    symmetry_observation_policy_pairs,
)


@dataclass
class SearchPlayConfig:
    hidden_sizes: list[int]
    board_rows: int
    board_cols: int
    action_dim: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    replay_size: int = 20_000
    min_replay_to_train: int = 128
    value_loss_weight: float = 1.0
    grad_clip_norm: float = 5.0
    use_gpu: bool = False
    simulations_per_move: int = 48
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.35
    dirichlet_epsilon: float = 0.25
    temperature_sample_moves: int = 10


class SearchPlayAlgorithm(Algorithm):
    algo_id = "search_play"

    def __init__(self, config: SearchPlayConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        self.obs_dim = int(config.board_rows) * int(config.board_cols)
        self.action_dim = int(config.action_dim)
        self.model = build_policy_value_network(
            input_size=int(self.obs_dim),
            hidden_sizes=list(config.hidden_sizes),
            policy_size=int(self.action_dim),
        ).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.mcts_config = MCTSConfig(
            simulations_per_move=int(config.simulations_per_move),
            c_puct=float(config.c_puct),
            dirichlet_alpha=float(config.dirichlet_alpha),
            dirichlet_epsilon=float(config.dirichlet_epsilon),
        )
        self.replay: deque[ReplaySample] = deque(maxlen=int(config.replay_size))
        self._episode_history: list[dict[str, Any]] = []
        self._pending_policy_target: np.ndarray | None = None
        self._episode_step_index = 0
        self.completed_games = 0
        self.total_moves = 0
        self.training_steps = 0

    def _normalize_action_mask(self, action_mask: object | None, obs: np.ndarray) -> np.ndarray:
        if action_mask is None:
            return action_mask_from_observation(
                np.asarray(obs, dtype=np.float32),
                int(self.config.board_rows),
                int(self.config.board_cols),
            )
        mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
        if int(mask.size) != int(self.action_dim):
            raise ValueError(
                f"Search-play action mask expected {self.action_dim} values, got {int(mask.size)}."
            )
        if int(mask.sum()) <= 0:
            return action_mask_from_observation(
                np.asarray(obs, dtype=np.float32),
                int(self.config.board_rows),
                int(self.config.board_cols),
            )
        return mask.astype(np.bool_, copy=False)

    def _policy_value(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_logits, values = self.model(obs_tensor)
        return (
            policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
            float(values.squeeze(0).item()),
        )

    def _select_action(self, visit_policy: np.ndarray, action_mask: np.ndarray, *, explore: bool) -> int:
        legal_actions = np.flatnonzero(np.asarray(action_mask, dtype=np.bool_))
        if legal_actions.size <= 0:
            return 0
        if legal_actions.size == 1:
            return int(legal_actions[0])
        if (not bool(explore)) or int(self._episode_step_index) >= int(self.config.temperature_sample_moves):
            masked_policy = visit_policy.copy()
            masked_policy[~np.asarray(action_mask, dtype=np.bool_)] = -1.0
            return int(np.argmax(masked_policy))
        weights = visit_policy[legal_actions].astype(np.float64, copy=False)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            return int(random.choice(legal_actions.tolist()))
        probs = weights / weight_sum
        return int(np.random.choice(legal_actions, p=probs))

    def _symmetry_samples(
        self,
        observation: np.ndarray,
        policy_target: np.ndarray,
        value_target: float,
    ) -> list[ReplaySample]:
        samples: list[ReplaySample] = []
        seen: set[tuple[bytes, bytes]] = set()
        for obs_sample, policy_sample in symmetry_observation_policy_pairs(observation, policy_target):
            obs_array = np.asarray(obs_sample, dtype=np.float32).reshape(-1)
            policy_array = np.asarray(policy_sample, dtype=np.float32).reshape(-1)
            key = (obs_array.tobytes(), policy_array.tobytes())
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                ReplaySample(
                    observation=obs_array.astype(np.float32, copy=False),
                    policy_target=policy_array.astype(np.float32, copy=False),
                    value_target=float(value_target),
                )
            )
        return samples

    def act(self, obs: np.ndarray, explore: bool, action_mask: np.ndarray | None = None) -> int:
        observation = np.asarray(obs, dtype=np.float32).reshape(-1)
        if int(observation.size) != int(self.obs_dim):
            raise ValueError(f"Search-play observation expected {self.obs_dim} values, got {int(observation.size)}.")
        valid_mask = self._normalize_action_mask(action_mask, observation)
        canonical_board = canonical_board_from_observation(
            observation,
            int(self.config.board_rows),
            int(self.config.board_cols),
        )
        visit_policy, _root_value = run_mcts(
            canonical_board=canonical_board,
            root_action_mask=valid_mask,
            config=self.mcts_config,
            policy_value_fn=self._policy_value,
            add_root_noise=bool(explore),
        )
        self._pending_policy_target = np.asarray(visit_policy, dtype=np.float32)
        return self._select_action(visit_policy, valid_mask, explore=bool(explore))

    def observe(self, transition: dict[str, Any]) -> None:
        observation = np.asarray(transition["obs"], dtype=np.float32).reshape(-1)
        info = transition.get("info", {})
        actor_value: int | None = None
        if isinstance(info, dict) and info.get("actor") is not None:
            try:
                actor_value = int(info.get("actor"))
            except (TypeError, ValueError):
                actor_value = None
        policy_target = self._pending_policy_target
        if policy_target is None:
            policy_target = np.zeros((int(self.action_dim),), dtype=np.float32)
            try:
                fallback_action = int(transition["action"])
            except (TypeError, ValueError):
                fallback_action = 0
            if 0 <= fallback_action < int(self.action_dim):
                policy_target[fallback_action] = 1.0
        self._episode_history.append(
            {
                "observation": observation.astype(np.float32, copy=False),
                "policy_target": np.asarray(policy_target, dtype=np.float32),
                "actor": actor_value,
            }
        )
        self._pending_policy_target = None
        self.total_moves += 1
        self._episode_step_index += 1

        if not bool(transition.get("done", False)):
            return

        search_value = dict(info).get("search_value") if isinstance(info, dict) else None
        winner_value: int | None = None
        if isinstance(info, dict) and info.get("winner") is not None:
            try:
                winner_value = int(info.get("winner"))
            except (TypeError, ValueError):
                winner_value = None
        if winner_value is not None and all(sample.get("actor") is not None for sample in self._episode_history):
            for sample in self._episode_history:
                sample_actor = int(sample["actor"])
                if int(winner_value) == PLAYER_NONE:
                    sample_value = 0.0
                elif int(winner_value) == int(sample_actor):
                    sample_value = 1.0
                else:
                    sample_value = -1.0
                augmented_samples = self._symmetry_samples(
                    np.asarray(sample["observation"], dtype=np.float32),
                    np.asarray(sample["policy_target"], dtype=np.float32),
                    float(sample_value),
                )
                for replay_sample in augmented_samples:
                    self.replay.append(replay_sample)
        else:
            if search_value is None:
                reward_value = float(transition.get("reward", 0.0))
                search_value = 1.0 if reward_value > 0.0 else -1.0 if reward_value < 0.0 else 0.0
            outcome = float(search_value)
            for sample in reversed(self._episode_history):
                augmented_samples = self._symmetry_samples(
                    np.asarray(sample["observation"], dtype=np.float32),
                    np.asarray(sample["policy_target"], dtype=np.float32),
                    float(outcome),
                )
                for replay_sample in augmented_samples:
                    self.replay.append(replay_sample)
                outcome = -float(outcome)

        self.completed_games += 1
        self._episode_history.clear()
        self._episode_step_index = 0

    def update(self) -> dict[str, float]:
        required = max(int(self.config.batch_size), int(self.config.min_replay_to_train))
        if len(self.replay) < required:
            return {}

        batch_samples = random.sample(list(self.replay), int(self.config.batch_size))
        observations = np.asarray([sample.observation for sample in batch_samples], dtype=np.float32)
        policy_targets = np.asarray([sample.policy_target for sample in batch_samples], dtype=np.float32)
        value_targets = np.asarray([sample.value_target for sample in batch_samples], dtype=np.float32)

        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        policy_target_tensor = torch.as_tensor(policy_targets, dtype=torch.float32, device=self.device)
        value_target_tensor = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)

        policy_logits, values = self.model(obs_tensor)
        log_policy = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_target_tensor * log_policy).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_target_tensor)
        loss = policy_loss + float(self.config.value_loss_weight) * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
        self.optimizer.step()

        self.training_steps += 1
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "buffer_size": float(len(self.replay)),
        }

    def reset_policy_state(self) -> None:
        self._pending_policy_target = None
        self._episode_step_index = 0
        self._episode_history.clear()

    def save(self, path: str) -> None:
        save_torch_checkpoint(
            path,
            {
                "algo_id": self.algo_id,
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "completed_games": int(self.completed_games),
                "total_moves": int(self.total_moves),
                "training_steps": int(self.training_steps),
            },
        )

    def load(self, path: str) -> None:
        checkpoint = load_torch_checkpoint(path, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            self.completed_games = int(checkpoint.get("completed_games", self.completed_games))
            self.total_moves = int(checkpoint.get("total_moves", self.total_moves))
            self.training_steps = int(checkpoint.get("training_steps", self.training_steps))
            return
        self.model.load_state_dict(checkpoint)
