"""Report key runtime dependency availability and versions."""

from __future__ import annotations

import importlib
import platform


def _version_for(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return "missing"
    return str(getattr(module, "__version__", "installed"))


def main() -> int:
    print(f"Python\tVersion: {platform.python_version()}")
    for module_name in ("torch", "arcade", "numpy", "pyglet"):
        print(f"{module_name}\tVersion: {_version_for(module_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

