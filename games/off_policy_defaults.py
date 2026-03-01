"""Shared defaults for off-policy game specs."""

from __future__ import annotations

from typing import Any

from core.algorithms.exploration import compute_eps_decay


EXPLORATION_AVG_WINDOW_EPISODES = 100
MIN_EPISODES_FOR_STATS = 100

OFF_POLICY_TRAIN_DEFAULTS: dict[str, Any] = {
    "train_after_steps": 0,
    "update_every_steps": 1,
    "updates_per_step": 1,
    "reward_window": int(EXPLORATION_AVG_WINDOW_EPISODES),
    "min_episodes_for_stats": int(MIN_EPISODES_FOR_STATS),
}


def make_exploration_config(
    eps_start: float,
    eps_min: float,
    eps_decay_steps: int,
    *,
    patience_episodes: int,
    min_improvement: float,
    eps_bump_cap: float,
    bump_cooldown_steps: int,
    avg_window_episodes: int = EXPLORATION_AVG_WINDOW_EPISODES,
) -> dict[str, Any]:
    return {
        "eps_start": float(eps_start),
        "eps_min": float(eps_min),
        "eps_decay": compute_eps_decay(
            eps_start=float(eps_start),
            eps_min=float(eps_min),
            eps_decay_steps=int(eps_decay_steps),
        ),
        "avg_window_episodes": int(avg_window_episodes),
        "patience_episodes": int(patience_episodes),
        "min_improvement": float(min_improvement),
        "eps_bump_cap": float(eps_bump_cap),
        "bump_cooldown_steps": int(bump_cooldown_steps),
    }
