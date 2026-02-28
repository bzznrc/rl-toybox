"""Shared epsilon exploration scheduling for off-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ExplorationConfig:
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.999995
    avg_window_episodes: int = 100
    patience_episodes: int = 30
    min_improvement: float = 0.20
    bump_epsilon: float = 0.20
    bump_hold_steps: int = 50_000
    bump_cooldown_episodes: int = 30


@dataclass(frozen=True)
class ExplorationBumpEvent:
    epsilon: float
    hold_steps: int
    reason: str = "plateau"


def resolve_exploration_config(
    value: ExplorationConfig | Mapping[str, object] | None,
) -> ExplorationConfig:
    if value is None:
        return ExplorationConfig()
    if isinstance(value, ExplorationConfig):
        return value
    return ExplorationConfig(**dict(value))


class EpsilonController:
    """Multiplicative epsilon decay with plateau-triggered bump and hold."""

    def __init__(
        self,
        config: ExplorationConfig,
        *,
        initial_epsilon: float | None = None,
    ):
        self.config = config
        self.epsilon = self._clamp(initial_epsilon if initial_epsilon is not None else float(config.eps_start))

        self._hold_steps_remaining = 0
        self._cooldown_episodes_remaining = 0
        self._episodes_since_improvement = 0
        self._reference_avg_reward: float | None = None

    @property
    def hold_steps_remaining(self) -> int:
        return int(self._hold_steps_remaining)

    @property
    def cooldown_episodes_remaining(self) -> int:
        return int(self._cooldown_episodes_remaining)

    def _clamp(self, epsilon: float) -> float:
        return max(float(self.config.eps_min), float(epsilon))

    def set_epsilon(self, epsilon: float) -> float:
        self.epsilon = self._clamp(float(epsilon))
        return float(self.epsilon)

    def advance_step(self) -> float:
        if self._hold_steps_remaining > 0:
            self._hold_steps_remaining -= 1
            return float(self.epsilon)

        self.epsilon = self._clamp(float(self.epsilon) * float(self.config.eps_decay))
        return float(self.epsilon)

    def on_episode_end(self, avg_reward: float) -> ExplorationBumpEvent | None:
        avg_reward = float(avg_reward)

        if self._reference_avg_reward is None:
            self._reference_avg_reward = avg_reward
            self._episodes_since_improvement = 0
        elif avg_reward >= float(self._reference_avg_reward) + float(self.config.min_improvement):
            self._reference_avg_reward = avg_reward
            self._episodes_since_improvement = 0
        else:
            self._episodes_since_improvement += 1

        if self._cooldown_episodes_remaining > 0:
            self._cooldown_episodes_remaining -= 1
            return None

        if self._hold_steps_remaining > 0:
            return None

        if self._episodes_since_improvement < int(self.config.patience_episodes):
            return None

        self.epsilon = max(float(self.epsilon), float(self.config.bump_epsilon))
        self._hold_steps_remaining = int(self.config.bump_hold_steps)
        self._cooldown_episodes_remaining = int(self.config.bump_cooldown_episodes)
        self._episodes_since_improvement = 0
        self._reference_avg_reward = avg_reward
        return ExplorationBumpEvent(
            epsilon=float(self.epsilon),
            hold_steps=int(self._hold_steps_remaining),
        )

    def state_dict(self) -> dict[str, float | int | None]:
        return {
            "epsilon": float(self.epsilon),
            "hold_steps_remaining": int(self._hold_steps_remaining),
            "cooldown_episodes_remaining": int(self._cooldown_episodes_remaining),
            "episodes_since_improvement": int(self._episodes_since_improvement),
            "reference_avg_reward": self._reference_avg_reward,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.set_epsilon(float(state.get("epsilon", self.epsilon)))
        self._hold_steps_remaining = max(0, int(state.get("hold_steps_remaining", 0)))
        self._cooldown_episodes_remaining = max(0, int(state.get("cooldown_episodes_remaining", 0)))
        self._episodes_since_improvement = max(0, int(state.get("episodes_since_improvement", 0)))

        reference = state.get("reference_avg_reward")
        self._reference_avg_reward = None if reference is None else float(reference)
