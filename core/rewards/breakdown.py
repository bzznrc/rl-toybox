"""Compact reward-component accumulation and formatting."""

from __future__ import annotations

from typing import Iterable, Mapping


class RewardBreakdown:
    def __init__(self, codes: Iterable[str] | None = None) -> None:
        self._values: dict[str, float] = {}
        self._codes: list[str] = []
        if codes is not None:
            for code in codes:
                self._register(str(code))
        self.reset()

    def _register(self, code: str) -> None:
        code_key = str(code)
        if code_key in self._values:
            return
        self._codes.append(code_key)
        self._values[code_key] = 0.0

    def reset(self) -> None:
        for code in self._codes:
            self._values[code] = 0.0

    def add(self, code: str, value: float) -> None:
        code_key = str(code)
        if code_key not in self._values:
            self._register(code_key)
        self._values[code_key] += float(value)

    def add_from_mapping(
        self,
        values_by_key: Mapping[str, object] | None,
        key_to_code: Mapping[str, str],
    ) -> None:
        if not isinstance(values_by_key, Mapping):
            return
        for key, code in key_to_code.items():
            raw_value = values_by_key.get(str(key), 0.0)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = 0.0
            self.add(str(code), float(value))

    def totals(self) -> dict[str, float]:
        return {code: float(self._values.get(code, 0.0)) for code in self._codes}

    @staticmethod
    def _format_value(value: float) -> str:
        rounded = 0.0 if abs(float(value)) < 5e-7 else float(value)
        text = f"{rounded:.2f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text in {"-0", "-0.0", "-0.00"}:
            return "0"
        return text

    def format(self, order: list[str]) -> str:
        parts: list[str] = []
        for code in order:
            code_key = str(code)
            value = float(self._values.get(code_key, 0.0))
            parts.append(f"{code_key}:{self._format_value(value)}")
        return " ".join(parts)
