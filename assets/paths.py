"""Shared asset path helpers for repo-wide runtime assets."""

from __future__ import annotations

from pathlib import Path


ASSETS_DIR = Path(__file__).resolve().parent
FONTS_DIR = ASSETS_DIR / "fonts"
AUDIO_DIR = ASSETS_DIR / "audio"
ICONS_DIR = ASSETS_DIR / "icons"


def resolve_asset_path(relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_absolute() and path.exists():
        return str(path)
    return str(ASSETS_DIR / relative_path.replace("\\", "/"))


def resolve_font_path(font_path_or_file: str) -> str:
    normalized = font_path_or_file.replace("\\", "/")
    if "/" not in normalized:
        normalized = f"fonts/{normalized}"
    return resolve_asset_path(normalized)

