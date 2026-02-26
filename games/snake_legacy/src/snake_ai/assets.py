"""Local asset path helpers."""

from __future__ import annotations

from pathlib import Path


_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def resolve_asset_path(relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_absolute() and path.exists():
        return str(path)

    normalized = relative_path.replace("\\", "/")
    candidate = _ASSETS_DIR / normalized
    if candidate.exists():
        return str(candidate)

    return str(candidate)


def resolve_font_path(font_path_or_file: str) -> str:
    normalized = font_path_or_file.replace("\\", "/")
    if "/" not in normalized:
        normalized = f"fonts/{normalized}"
    return resolve_asset_path(normalized)
