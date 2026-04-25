"""Lightweight validator for canonical repo docs and active game READMEs."""

from __future__ import annotations

import re
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_GAME_ORDER = ("snake", "bang", "jump", "vroom", "osero", "kick")
STRICT_LEVEL_MIRROR_GAMES = {"jump", "kick"}
REQUIRED_GAME_HEADINGS = (
    "Clip",
    "Algorithm / Network",
    "Controls (Human)",
    "Observation / Actions",
    "Environment Notes",
    "Rewards (Training)",
    "Curriculum (Train)",
    "Run Commands",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_table_game_order(text: str) -> list[str]:
    order: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\|\s*`([^`]+)`\s*\|", line.strip())
        if match:
            order.append(str(match.group(1)).strip())
    return order


def _extract_second_level_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            headings.append(str(match.group(1)).strip())
    return headings


def _config_module(game_id: str):
    return importlib.import_module(f"games.{game_id}.config")


def _require_snippet(errors: list[str], *, path: Path, text: str, snippet: str) -> None:
    if str(snippet) not in text:
        errors.append(f"{path.relative_to(REPO_ROOT)} is missing config mirror snippet: {snippet!r}")


def _validate_config_mirror(errors: list[str], *, game_id: str, readme_path: Path, readme_text: str) -> None:
    config = _config_module(game_id)
    obs_dim = int(getattr(config, "OBS_DIM"))
    act_dim = int(getattr(config, "ACT_DIM"))
    _require_snippet(errors, path=readme_path, text=readme_text, snippet=f"obs={obs_dim}")
    _require_snippet(errors, path=readme_path, text=readme_text, snippet=f"act={act_dim}")

    model_config = dict(getattr(config, "DEFAULT_MODEL_CONFIG", {}))
    hidden_sizes = model_config.get("hidden_sizes")
    if isinstance(hidden_sizes, list) and hidden_sizes:
        hidden_text = " -> ".join(str(int(size)) for size in hidden_sizes)
        _require_snippet(errors, path=readme_path, text=readme_text, snippet=hidden_text)

    critic_sizes = model_config.get("critic_hidden_sizes")
    if isinstance(critic_sizes, list) and critic_sizes:
        critic_text = " -> ".join(str(int(size)) for size in critic_sizes)
        _require_snippet(errors, path=readme_path, text=readme_text, snippet=critic_text)

    if game_id not in STRICT_LEVEL_MIRROR_GAMES:
        return

    level_settings = getattr(config, "LEVEL_SETTINGS", None)
    if isinstance(level_settings, dict):
        for level, settings in sorted(level_settings.items()):
            _require_snippet(errors, path=readme_path, text=readme_text, snippet=f"Level {int(level)}:")
            for key, value in dict(settings).items():
                if isinstance(value, list):
                    continue
                if isinstance(value, float):
                    value_text = f"{float(value):.2f}".rstrip("0").rstrip(".")
                else:
                    value_text = str(value)
                _require_snippet(
                    errors,
                    path=readme_path,
                    text=readme_text,
                    snippet=f"{str(key)}={value_text}",
                )


def validate() -> list[str]:
    errors: list[str] = []

    required_files = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "repo-guide.md",
        REPO_ROOT / "docs" / "rl-design-guide.md",
        REPO_ROOT / ".vscode" / "README.md",
    )
    for path in required_files:
        if not path.is_file():
            errors.append(f"Missing required doc: {path.relative_to(REPO_ROOT)}")

    root_readme = REPO_ROOT / "README.md"
    if root_readme.is_file():
        root_text = _read_text(root_readme)
        root_order = list(
            dict.fromkeys(game_id for game_id in _extract_table_game_order(root_text) if game_id in ACTIVE_GAME_ORDER)
        )
        if root_order != list(ACTIVE_GAME_ORDER):
            errors.append(
                "README.md active game order does not match ACTIVE_GAME_ORDER: "
                f"expected {list(ACTIVE_GAME_ORDER)}, got {root_order}"
            )
        required_snippets = (
            "shared `L1` to `L5` ladder",
            "OSERO_BOARD_SIZE",
        )
        for snippet in required_snippets:
            if snippet not in root_text:
                errors.append(f"README.md is missing expected snippet: {snippet!r}")

    repo_guide = REPO_ROOT / "docs" / "repo-guide.md"
    if repo_guide.is_file():
        repo_text = _read_text(repo_guide)
        repo_order = list(
            dict.fromkeys(game_id for game_id in _extract_table_game_order(repo_text) if game_id in ACTIVE_GAME_ORDER)
        )
        if repo_order != list(ACTIVE_GAME_ORDER):
            errors.append(
                "docs/repo-guide.md active lineup order does not match ACTIVE_GAME_ORDER: "
                f"expected {list(ACTIVE_GAME_ORDER)}, got {repo_order}"
            )
        for snippet in ("shared 5-level ladder", "board-size exception"):
            if snippet not in repo_text:
                errors.append(f"docs/repo-guide.md is missing expected snippet: {snippet!r}")

    rl_guide = REPO_ROOT / "docs" / "rl-design-guide.md"
    if rl_guide.is_file():
        rl_text = _read_text(rl_guide)
        for snippet in ("shared 5-level curriculum", "temporary exception"):
            if snippet not in rl_text:
                errors.append(f"docs/rl-design-guide.md is missing expected snippet: {snippet!r}")

    vscode_readme = REPO_ROOT / ".vscode" / "README.md"
    if vscode_readme.is_file():
        vscode_text = _read_text(vscode_readme)
        for snippet in ("curriculumTrainLevel", "curriculumPlayLevel", "oseroBoardSize"):
            if snippet not in vscode_text:
                errors.append(f".vscode/README.md is missing expected snippet: {snippet!r}")

    for game_id in ACTIVE_GAME_ORDER:
        readme_path = REPO_ROOT / "games" / game_id / "README.md"
        if not readme_path.is_file():
            errors.append(f"Missing game README: games/{game_id}/README.md")
            continue
        headings = _extract_second_level_headings(_read_text(readme_path))
        readme_text = _read_text(readme_path)
        if headings != list(REQUIRED_GAME_HEADINGS):
            errors.append(
                f"games/{game_id}/README.md has unexpected section order: "
                f"expected {list(REQUIRED_GAME_HEADINGS)}, got {headings}"
            )
        _validate_config_mirror(errors, game_id=game_id, readme_path=readme_path, readme_text=readme_text)

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Docs validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Docs validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
