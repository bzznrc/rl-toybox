# Docs Index

- [repo-architecture.md](./repo-architecture.md): shared repo/codebase architecture and engineering conventions.
- [rl-design-guide.md](./rl-design-guide.md): cross-game RL/environment design rules.
- `core/game.py` and `core/io/`: shared game catalog/spec builders plus run/checkpoint/model-path handling used by the CLI entrypoints.
- `games/<game>/README.md`: source of truth for each game's current observation/action IO, model snapshot, rewards, and curriculum.
