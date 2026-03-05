"""Shared score/history/clock tracking helpers for game status bars."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar


CompetitorT = TypeVar("CompetitorT")


def _optional_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    return max(1, int(value))


def compact_count_to_icons(count: int, *, pack_size: int = 5) -> list[bool]:
    """Return compact status icons where `True` means one packed icon."""

    total = max(0, int(count))
    unit = max(2, int(pack_size))
    packed = total // unit
    single = total % unit
    return ([True] * packed) + ([False] * single)


@dataclass
class MatchTracker(Generic[CompetitorT]):
    """Tracks match outcomes, optional score totals, and optional clock duration."""

    history_limit: int | None = None
    match_limit: int | None = None
    clock_duration_steps: int | None = None
    scores: dict[CompetitorT, int] = field(default_factory=dict)
    history: list[CompetitorT | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history_limit = _optional_positive_int(self.history_limit)
        self.match_limit = _optional_positive_int(self.match_limit)
        self.clock_duration_steps = _optional_positive_int(self.clock_duration_steps)
        self._trim_history()

    def set_history_limit(self, value: int | None) -> None:
        self.history_limit = _optional_positive_int(value)
        self._trim_history()

    def set_match_limit(self, value: int | None) -> None:
        self.match_limit = _optional_positive_int(value)

    def set_clock_duration(self, value: int | None) -> None:
        self.clock_duration_steps = _optional_positive_int(value)

    def set_competitors(self, competitors: Iterable[CompetitorT], *, preserve_existing: bool = True) -> None:
        ordered_unique: list[CompetitorT] = []
        seen: set[CompetitorT] = set()
        for competitor in competitors:
            if competitor in seen:
                continue
            seen.add(competitor)
            ordered_unique.append(competitor)

        old_scores = dict(self.scores)
        self.scores.clear()
        for competitor in ordered_unique:
            previous = int(old_scores.get(competitor, 0)) if preserve_existing else 0
            self.scores[competitor] = max(0, int(previous))

    def score(self, competitor: CompetitorT) -> int:
        return int(self.scores.get(competitor, 0))

    def set_score(self, competitor: CompetitorT, value: int) -> None:
        self.scores[competitor] = max(0, int(value))

    def increment_score(self, competitor: CompetitorT, amount: int = 1) -> None:
        self.scores[competitor] = max(0, int(self.scores.get(competitor, 0)) + int(amount))

    def reset_scores(self) -> None:
        for competitor in list(self.scores.keys()):
            self.scores[competitor] = 0

    def clear_history(self) -> None:
        self.history.clear()

    def record_result(self, winner: CompetitorT | None, *, increment_winner_score: bool = False) -> None:
        if increment_winner_score and winner is not None:
            self.increment_score(winner)
        self.history.append(winner)
        self._trim_history()

    def record_draw(self) -> None:
        self.record_result(None, increment_winner_score=False)

    def matches_played(self) -> int:
        return int(len(self.history))

    def match_limit_reached(self) -> bool:
        return self.match_limit is not None and int(len(self.history)) >= int(self.match_limit)

    def remaining_time_ratio(self, current_step: int) -> float:
        duration = self.clock_duration_steps
        if duration is None:
            return 1.0
        step = max(0, int(current_step))
        left = max(0, int(duration) - step)
        return float(left) / float(max(1, int(duration)))

    def _trim_history(self) -> None:
        limit = self.history_limit
        if limit is None:
            return
        overflow = int(len(self.history)) - int(limit)
        if overflow > 0:
            del self.history[:overflow]
