"""Shared epsilon exploration scheduling for off-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def compute_eps_decay(eps_start: float, eps_min: float, eps_decay_steps: int) -> float:
    eps_start_value = float(eps_start)
    eps_min_value = float(eps_min)
    steps = int(eps_decay_steps)

    if eps_start_value <= 0.0:
        raise ValueError("eps_start must be > 0.")
    if eps_min_value <= 0.0:
        raise ValueError("eps_min must be > 0.")
    if steps <= 0:
        raise ValueError("eps_decay_steps must be > 0.")
    if eps_min_value >= eps_start_value:
        raise ValueError("eps_min must be < eps_start.")

    return (eps_min_value / eps_start_value) ** (1.0 / float(steps))


@dataclass(frozen=True)
class ExplorationConfig:
    eps_start: float
    eps_min: float
    eps_decay: float
    patience_episodes: int
    min_improvement: float
    eps_bump_cap: float
    bump_cooldown_steps: int
    avg_window_episodes: int = 100


@dataclass(frozen=True)
class ExplorationBumpEvent:
    epsilon: float
    cooldown_steps: int
    reason: str = "Plateau"


def resolve_exploration_config(
    value: ExplorationConfig | Mapping[str, object] | None,
) -> ExplorationConfig:
    if value is None:
        raise ValueError("exploration config is required for epsilon-based algorithms.")
    if isinstance(value, ExplorationConfig):
        return value

    config_data = dict(value)
    # Backward compatibility for older config payloads using decay steps.
    legacy_decay_steps = config_data.pop("eps_decay_steps", None)
    if "eps_decay" not in config_data and legacy_decay_steps is not None:
        config_data["eps_decay"] = compute_eps_decay(
            eps_start=float(config_data["eps_start"]),
            eps_min=float(config_data["eps_min"]),
            eps_decay_steps=int(legacy_decay_steps),
        )
    # Backward compatibility for older config payloads.
    legacy_bump = config_data.pop("bump_epsilon", None)
    if "eps_bump_cap" not in config_data and legacy_bump is not None:
        config_data["eps_bump_cap"] = float(legacy_bump)
    # Removed in favor of always-on decay with cooldown-only bump gating.
    config_data.pop("bump_hold_steps", None)
    # Backward compatibility for older cooldown key.
    legacy_cooldown_episodes = config_data.pop("bump_cooldown_episodes", None)
    if "bump_cooldown_steps" not in config_data and legacy_cooldown_episodes is not None:
        config_data["bump_cooldown_steps"] = int(legacy_cooldown_episodes)

    return ExplorationConfig(**config_data)


class EpsilonController:
    """Multiplicative epsilon decay with plateau-triggered bump and cooldown."""

    def __init__(
        self,
        config: ExplorationConfig,
        *,
        initial_epsilon: float | None = None,
    ):
        self.config = config
        self.epsilon = self._clamp(initial_epsilon if initial_epsilon is not None else float(config.eps_start))

        self._cooldown_steps_remaining = 0
        self._episodes_since_best = 0
        self._best_avg_reward: float | None = None

    @property
    def cooldown_steps_remaining(self) -> int:
        return int(self._cooldown_steps_remaining)

    def _clamp(self, epsilon: float) -> float:
        return max(float(self.config.eps_min), float(epsilon))

    def set_epsilon(self, epsilon: float) -> float:
        self.epsilon = self._clamp(float(epsilon))
        return float(self.epsilon)

    def advance_step(self) -> float:
        if self._cooldown_steps_remaining > 0:
            self._cooldown_steps_remaining -= 1

        self.epsilon = self._clamp(float(self.epsilon) * float(self.config.eps_decay))
        return float(self.epsilon)

    def on_episode_end(self, avg_reward: float) -> ExplorationBumpEvent | None:
        avg_reward = float(avg_reward)

        if self._best_avg_reward is None:
            self._best_avg_reward = avg_reward
            self._episodes_since_best = 0
        elif avg_reward > float(self._best_avg_reward) + float(self.config.min_improvement):
            self._best_avg_reward = avg_reward
            self._episodes_since_best = 0
        else:
            self._episodes_since_best += 1

        if self._cooldown_steps_remaining > 0:
            return None

        if self._episodes_since_best < int(self.config.patience_episodes):
            return None

        if float(self.epsilon) >= float(self.config.eps_bump_cap):
            return None

        self.epsilon = float(self.config.eps_bump_cap)
        self._cooldown_steps_remaining = int(self.config.bump_cooldown_steps)
        self._episodes_since_best = 0
        return ExplorationBumpEvent(
            epsilon=float(self.epsilon),
            cooldown_steps=int(self._cooldown_steps_remaining),
        )

    def state_dict(self) -> dict[str, float | int | None]:
        return {
            "epsilon": float(self.epsilon),
            "cooldown_steps_remaining": int(self._cooldown_steps_remaining),
            "episodes_since_best": int(self._episodes_since_best),
            "best_avg_reward": self._best_avg_reward,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.set_epsilon(float(state.get("epsilon", self.epsilon)))
        self._cooldown_steps_remaining = max(
            0,
            int(
                state.get(
                    "cooldown_steps_remaining",
                    state.get("cooldown_episodes_remaining", 0),
                )
            ),
        )
        self._episodes_since_best = max(
            0,
            int(state.get("episodes_since_best", state.get("episodes_since_improvement", 0))),
        )

        best = state.get("best_avg_reward", state.get("reference_avg_reward"))
        self._best_avg_reward = None if best is None else float(best)


def bump_epsilon_to_cap(algorithm: object) -> float | None:
    """Raise epsilon to eps_bump_cap when curriculum level increases."""
    exploration = getattr(algorithm, "_exploration", None)
    if exploration is None:
        return None

    config = getattr(exploration, "config", None)
    set_epsilon = getattr(exploration, "set_epsilon", None)
    current_epsilon = getattr(exploration, "epsilon", None)
    bump_cap = getattr(config, "eps_bump_cap", None)
    if bump_cap is None or current_epsilon is None or not callable(set_epsilon):
        return None

    updated_epsilon = float(set_epsilon(max(float(current_epsilon), float(bump_cap))))
    if hasattr(algorithm, "epsilon"):
        setattr(algorithm, "epsilon", updated_epsilon)
    return float(updated_epsilon)
