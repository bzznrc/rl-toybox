"""Shared defaults for off-policy game specs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from core.algorithms.exploration import ExplorationConfig, compute_eps_decay


OFF_POLICY_EXPLORATION_DEFAULTS = ExplorationConfig(
    eps_start=1.0,
    eps_min=0.05,
    avg_window_episodes=100,
    patience_episodes=30,
    min_improvement=0.20,
    eps_bump_cap=0.25,
    bump_hold_steps=50_000,
    bump_cooldown_episodes=30,
)

OFF_POLICY_TRAIN_DEFAULTS: dict[str, Any] = {
    "train_after_steps": 0,
    "update_every_steps": 1,
    "updates_per_step": 1,
    "reward_window": int(OFF_POLICY_EXPLORATION_DEFAULTS.avg_window_episodes),
}


def make_exploration_config(
    eps_start: float,
    eps_min: float,
    eps_decay_steps: int,
    eps_bump_cap: float = 0.25,
) -> dict[str, Any]:
    exploration = asdict(OFF_POLICY_EXPLORATION_DEFAULTS)
    exploration["eps_start"] = float(eps_start)
    exploration["eps_min"] = float(eps_min)
    exploration["eps_decay"] = compute_eps_decay(
        eps_start=float(eps_start),
        eps_min=float(eps_min),
        eps_decay_steps=int(eps_decay_steps),
    )
    exploration["eps_bump_cap"] = float(eps_bump_cap)
    return exploration
