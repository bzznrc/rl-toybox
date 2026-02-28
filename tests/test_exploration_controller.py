from __future__ import annotations

import pytest

from core.algorithms.exploration import EpsilonController, ExplorationConfig, compute_eps_decay


def test_epsilon_decays_multiplicatively_without_hold() -> None:
    controller = EpsilonController(
        ExplorationConfig(
            eps_start=1.0,
            eps_min=0.05,
            eps_decay=0.9,
            patience_episodes=999,
        )
    )

    assert controller.advance_step() == pytest.approx(0.9)
    assert controller.advance_step() == pytest.approx(0.81)


def test_compute_eps_decay_reaches_eps_min_at_target_steps() -> None:
    eps_start = 1.0
    eps_min = 0.05
    eps_decay_steps = 10_000
    eps_decay = compute_eps_decay(eps_start, eps_min, eps_decay_steps)

    controller = EpsilonController(
        ExplorationConfig(
            eps_start=eps_start,
            eps_min=eps_min,
            eps_decay=eps_decay,
            patience_episodes=999,
        )
    )

    for _ in range(eps_decay_steps):
        controller.advance_step()

    assert controller.epsilon == pytest.approx(eps_min, rel=1e-6, abs=1e-6)


def test_plateau_does_not_bump_above_cap() -> None:
    controller = EpsilonController(
        ExplorationConfig(
            eps_start=0.30,
            eps_min=0.05,
            eps_decay=1.0,
            patience_episodes=1,
            min_improvement=0.1,
            eps_bump_cap=0.25,
            bump_hold_steps=5,
            bump_cooldown_episodes=2,
        )
    )

    assert controller.on_episode_end(0.0) is None
    assert controller.on_episode_end(0.0) is None
    assert controller.epsilon == pytest.approx(0.30)
    assert controller.hold_steps_remaining == 0


def test_plateau_bumps_to_cap_when_epsilon_is_at_or_below_cap() -> None:
    controller = EpsilonController(
        ExplorationConfig(
            eps_start=0.20,
            eps_min=0.05,
            eps_decay=1.0,
            patience_episodes=1,
            min_improvement=0.1,
            eps_bump_cap=0.25,
            bump_hold_steps=7,
            bump_cooldown_episodes=3,
        )
    )

    assert controller.on_episode_end(0.0) is None
    bump = controller.on_episode_end(0.0)
    assert bump is not None
    assert bump.epsilon == pytest.approx(0.25)
    assert controller.epsilon == pytest.approx(0.25)
    assert controller.hold_steps_remaining == 7


def test_bump_hold_freezes_decay_for_hold_steps() -> None:
    controller = EpsilonController(
        ExplorationConfig(
            eps_start=0.20,
            eps_min=0.05,
            eps_decay=0.5,
            patience_episodes=1,
            min_improvement=0.1,
            eps_bump_cap=0.25,
            bump_hold_steps=3,
            bump_cooldown_episodes=1,
        )
    )

    assert controller.on_episode_end(0.0) is None
    bump = controller.on_episode_end(0.0)
    assert bump is not None
    assert controller.epsilon == pytest.approx(0.25)

    assert controller.advance_step() == pytest.approx(0.25)
    assert controller.advance_step() == pytest.approx(0.25)
    assert controller.advance_step() == pytest.approx(0.25)
    assert controller.hold_steps_remaining == 0

    assert controller.advance_step() == pytest.approx(0.125)
