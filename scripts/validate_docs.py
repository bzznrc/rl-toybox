"""Validate shared README/doc ordering and section structure."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_ORDER = ("snake", "vroom", "bang", "walk", "peek", "kick")
GAME_TITLES = {
    "snake": "Snake",
    "vroom": "Vroom",
    "bang": "Bang",
    "walk": "Walk",
    "peek": "Peek",
    "kick": "Kick",
}
GAME_README_HEADINGS = (
    "## Clip",
    "## Algorithm / Network",
    "## Controls (Human)",
    "## Observation / Actions",
    "## Environment Notes",
    "## Rewards (Training)",
    "## Curriculum (Train)",
    "## Run Commands",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _extract_section(text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing section '## {heading}'.")
    start = match.end()
    next_match = re.search(r"^## .+$", text[start:], flags=re.MULTILINE)
    if next_match is None:
        return text[start:]
    return text[start : start + next_match.start()]


def _extract_game_table_order(text: str) -> list[str]:
    section = _extract_section(text, "Games")
    return re.findall(r"^\| `([a-z]+)` \|", section, flags=re.MULTILINE)


def _extract_default_plan_order(text: str) -> list[str]:
    section = _extract_section(text, "Default Plans")
    return re.findall(r"^- `([a-z]+)` ->", section, flags=re.MULTILINE)


def _extract_clip_order(text: str) -> list[str]:
    section = _extract_section(text, "Clips")
    found = re.findall(r"media/([a-z]+)-demo\.(?:gif|mp4)", section)
    ordered: list[str] = []
    for game_id in found:
        if game_id not in ordered:
            ordered.append(game_id)
    return ordered


def _extract_docs_game_order(text: str) -> list[str]:
    return re.findall(r"^- \[games/([a-z]+)/README\.md\]", text, flags=re.MULTILINE)


def _top_level_headings(text: str) -> list[str]:
    return re.findall(r"^## .+$", text, flags=re.MULTILINE)


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _check_equal(label: str, actual: list[str] | tuple[str, ...], expected: tuple[str, ...], errors: list[str]) -> None:
    actual_list = list(actual)
    expected_list = list(expected)
    if actual_list != expected_list:
        errors.append(f"{label}: expected {expected_list}, got {actual_list}")


def validate_root_readme(errors: list[str]) -> None:
    text = _read_text(REPO_ROOT / "README.md")
    _check_equal("README.md games table order", _extract_game_table_order(text), GAME_ORDER, errors)
    _check_equal("README.md default plans order", _extract_default_plan_order(text), GAME_ORDER, errors)
    clip_order = _extract_clip_order(text)
    expected_clips = tuple(game_id for game_id in GAME_ORDER if game_id in set(clip_order))
    _check_equal("README.md clip order", clip_order, expected_clips, errors)


def validate_docs_index(errors: list[str]) -> None:
    text = _read_text(REPO_ROOT / "docs" / "README.md")
    _check_equal("docs/README.md game docs order", _extract_docs_game_order(text), GAME_ORDER, errors)


def validate_game_readmes(errors: list[str]) -> None:
    for game_id in GAME_ORDER:
        path = REPO_ROOT / "games" / game_id / "README.md"
        text = _read_text(path)
        expected_title = f"# {GAME_TITLES[game_id]}"
        actual_title = _first_nonempty_line(text)
        if actual_title != expected_title:
            errors.append(f"{path.as_posix()}: expected title '{expected_title}', got '{actual_title}'")
        _check_equal(f"{path.as_posix()} top-level headings", _top_level_headings(text), GAME_README_HEADINGS, errors)


def main() -> None:
    errors: list[str] = []
    validate_root_readme(errors)
    validate_docs_index(errors)
    validate_game_readmes(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("docs-ok")


if __name__ == "__main__":
    main()
