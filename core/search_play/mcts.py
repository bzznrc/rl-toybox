"""Minimal MCTS implementation for AlphaZero-lite self-play."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.search_play.interfaces import MCTSConfig
from games.flip.rules import (
    PLAYER_ONE,
    apply_canonical_action,
    apply_canonical_action_with_turn,
    build_action_mask,
    is_terminal_board,
    terminal_outcome_from_canonical,
)


PolicyValueFn = Callable[[np.ndarray], tuple[np.ndarray, float]]


@dataclass
class SearchNode:
    prior: float
    value_to_parent_sign: int = -1
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if int(self.visit_count) <= 0:
            return 0.0
        return float(self.value_sum) / float(self.visit_count)


def _normalize_mask(action_mask: np.ndarray, action_dim: int) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    if int(mask.size) != int(action_dim):
        raise ValueError(f"MCTS action mask expected {action_dim} values, got {int(mask.size)}.")
    if int(mask.sum()) <= 0:
        raise ValueError("MCTS received an all-zero action mask for a non-terminal position.")
    return mask.astype(np.bool_, copy=False)


def _masked_softmax(logits: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float32).reshape(-1)
    mask = _normalize_mask(action_mask, int(logits_array.size))
    legal_actions = np.flatnonzero(mask)
    probs = np.zeros_like(logits_array, dtype=np.float32)
    if legal_actions.size <= 0:
        probs.fill(1.0 / max(1, int(probs.size)))
        return probs

    legal_logits = logits_array[legal_actions]
    legal_logits = legal_logits - float(np.max(legal_logits))
    exp_logits = np.exp(legal_logits, dtype=np.float32)
    total = float(exp_logits.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs[legal_actions] = 1.0 / float(legal_actions.size)
        return probs
    probs[legal_actions] = exp_logits / total
    return probs


def _expand(node: SearchNode, canonical_board: np.ndarray, priors: np.ndarray, action_mask: np.ndarray) -> None:
    mask = _normalize_mask(action_mask, int(priors.size))
    for action_index in np.flatnonzero(mask):
        _next_board, turn_sign = apply_canonical_action_with_turn(canonical_board, int(action_index))
        node.children[int(action_index)] = SearchNode(
            prior=float(priors[int(action_index)]),
            value_to_parent_sign=int(turn_sign),
        )


def _add_root_noise(priors: np.ndarray, action_mask: np.ndarray, config: MCTSConfig) -> np.ndarray:
    mask = _normalize_mask(action_mask, int(priors.size))
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size <= 1:
        return priors
    noise = np.random.dirichlet(
        np.full((int(legal_actions.size),), float(config.dirichlet_alpha), dtype=np.float32)
    ).astype(np.float32, copy=False)
    mixed = priors.copy()
    mixed[legal_actions] = (
        (1.0 - float(config.dirichlet_epsilon)) * priors[legal_actions]
        + float(config.dirichlet_epsilon) * noise
    )
    return mixed


def _select_child(node: SearchNode, c_puct: float) -> tuple[int, SearchNode]:
    best_score = float("-inf")
    best_action = 0
    best_child: SearchNode | None = None
    root_visits = np.sqrt(float(max(1, node.visit_count)))
    for action_index, child in node.children.items():
        q_score = float(child.value_to_parent_sign) * float(child.value)
        u_score = float(c_puct) * float(child.prior) * root_visits / float(1 + int(child.visit_count))
        score = q_score + u_score
        if score > best_score:
            best_score = float(score)
            best_action = int(action_index)
            best_child = child
    if best_child is None:
        raise RuntimeError("MCTS select_child called before node expansion.")
    return int(best_action), best_child


def _backpropagate(search_path: list[SearchNode], value: float) -> None:
    backup_value = float(value)
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += float(backup_value)
        backup_value = float(node.value_to_parent_sign) * float(backup_value)


def run_mcts(
    *,
    canonical_board: np.ndarray,
    root_action_mask: np.ndarray,
    config: MCTSConfig,
    policy_value_fn: PolicyValueFn,
    add_root_noise: bool,
) -> tuple[np.ndarray, float]:
    board = np.asarray(canonical_board, dtype=np.int8)
    action_dim = int(root_action_mask.shape[0])
    root = SearchNode(prior=1.0, value_to_parent_sign=1)
    root_logits, root_value = policy_value_fn(board.reshape(-1))
    root_priors = _masked_softmax(root_logits, root_action_mask)
    if bool(add_root_noise):
        root_priors = _add_root_noise(root_priors, root_action_mask, config)
    _expand(root, board, root_priors, root_action_mask)

    for _ in range(int(config.simulations_per_move)):
        node = root
        rollout_board = board.copy()
        search_path = [root]

        while node.children:
            action_index, node = _select_child(node, float(config.c_puct))
            rollout_board = apply_canonical_action(rollout_board, int(action_index))
            search_path.append(node)

        if bool(is_terminal_board(rollout_board)):
            leaf_value = float(terminal_outcome_from_canonical(rollout_board))
        else:
            leaf_action_mask = build_action_mask(rollout_board, current_player=PLAYER_ONE)
            leaf_logits, leaf_value = policy_value_fn(rollout_board.reshape(-1))
            leaf_priors = _masked_softmax(leaf_logits, leaf_action_mask)
            _expand(node, rollout_board, leaf_priors, leaf_action_mask)

        _backpropagate(search_path, float(leaf_value))

    visit_policy = np.zeros((action_dim,), dtype=np.float32)
    for action_index, child in root.children.items():
        visit_policy[int(action_index)] = float(child.visit_count)
    total_visits = float(visit_policy.sum())
    if total_visits <= 0.0:
        legal_mask = _normalize_mask(root_action_mask, action_dim)
        visit_policy[legal_mask] = 1.0 / float(max(1, int(legal_mask.sum())))
    else:
        visit_policy /= total_visits
    return visit_policy.astype(np.float32, copy=False), float(root.value if root.visit_count > 0 else root_value)
