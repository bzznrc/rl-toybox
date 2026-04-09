# Docs Index

- [repo-architecture.md](./repo-architecture.md): shared repo structure, active lineup taxonomy, and ownership boundaries.
- [rl-design-guide.md](./rl-design-guide.md): cross-game RL and environment design rules.
- [migration-lineup-refactor.md](./migration-lineup-refactor.md): what changed in the lineup/taxonomy refactor and what was intentionally left for later.
- `core/game.py`: active game registry plus repo-wide lineup metadata.
- `games/<game>/README.md`: per-game snapshot or scaffold brief for active entries.

## Active Game Docs

Canonical active lineup order across repo docs is:
`snake`, `bang`, `tower`, `vroom`, `frogger`, `card`, `osero`, `kick`

- [games/snake/README.md](../games/snake/README.md)
- [games/bang/README.md](../games/bang/README.md)
- [games/tower/README.md](../games/tower/README.md)
- [games/vroom/README.md](../games/vroom/README.md)
- [games/frogger/README.md](../games/frogger/README.md)
- [games/card/README.md](../games/card/README.md)
- [games/osero/README.md](../games/osero/README.md)
- [games/kick/README.md](../games/kick/README.md)

Validate the shared README contract with:
`python -m scripts.validate_docs`
or
`rl-toybox-validate-docs`
