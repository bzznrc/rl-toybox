"""Shared self-play placeholders for future AlphaZero-lite work."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfPlayMatchResult:
    moves: int
    winner: int
    outcome_value: float
