# rl-toybox

[![CI](https://github.com/bzznrc/rl-toybox/actions/workflows/smoke.yml/badge.svg)](https://github.com/bzznrc/rl-toybox/actions/workflows/smoke.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyTorch](https://img.shields.io/badge/PyTorch-used-ee4c2c) ![Arcade](https://img.shields.io/badge/Arcade-used-2b7cff)

`rl-toybox` is a small reinforcement-learning playground built around short arcade games. Each game is meant to be readable from end to end. Shared code handles training, evaluation, rendering, configuration, logs, and checkpoints.

## Repo Layout

- `core/` contains the shared environments, algorithms, runners, rendering helpers, and run IO.
- `core/value_discrete/`, `core/actor_critic/`, and `core/search_play/` contain the three learning families.
- `games/<name>/` contains each game's config, environment, and README.
- `scripts/` contains the `train`, `play_ai`, `play_user`, `env_sanity`, and `capture_demo` entrypoints.
- `assets/` contains shared fonts, audio, and icons.
- `runs/` contains checkpoints and metrics. Best checkpoints are kept in git.
- `tests/` contains one small shared-contract smoke test.

## Docs

- [Repository guide](docs/repo-guide.md)
- [RL and environment design guide](docs/rl-design-guide.md)

## Clips

<p align="center">
  <img src="media/snake-demo.gif" alt="Snake demo clip" width="32%" />
  <img src="media/bang-demo.gif" alt="Bang demo clip" width="32%" />
  <img src="media/jump-demo.gif" alt="Jump demo clip" width="32%" />
</p>
<p align="center">
  <img src="media/vroom-demo.gif" alt="Vroom demo clip" width="32%" />
  <img src="media/flip-demo.gif" alt="Flip demo clip" width="32%" />
  <img src="media/kick-demo.gif" alt="Kick demo clip" width="32%" />
</p>

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Train, watch the model, or play yourself:

```bash
rl-toybox-train --game bang --mode team_arena
rl-toybox-play-ai --game bang --mode team_arena --render
rl-toybox-play-user --game bang --mode team_arena
```

The same commands work without installation:

```bash
python -m scripts.train --game bang --mode team_arena
python -m scripts.play_ai --game bang --mode team_arena --render
python -m scripts.play_user --game bang --mode team_arena
```

Training starts at level 1 by default. Play, evaluation, and capture default to level 5. `flip` has one fixed board, so it always uses level 1. AI play loads the best checkpoint unless another model source is requested.

## Games

| Game | Family | What it covers |
| --- | --- | --- |
| [`snake`](games/snake/README.md) | value-based | Small grid control, compact observations, and reward shaping |
| [`bang`](games/bang/README.md) | value-based | DQN combat in Duel, Arena, and shared-policy Team Arena modes |
| [`jump`](games/jump/README.md) | actor-critic | PPO timing and traversal in a short platformer |
| [`vroom`](games/vroom/README.md) | actor-critic | SAC and continuous control on procedural tracks |
| [`flip`](games/flip/README.md) | search + self-play | MCTS, legal-action masking, and policy/value learning |
| [`kick`](games/kick/README.md) | actor-critic / CTDE | Shared-policy football in 3v3, 5v5, and 7v7 modes |

## Suggested Learning Path

1. `snake` for Q-learning and the smallest environment.
2. `bang` for DQN, replay, and richer observations.
3. `jump` for PPO and on-policy learning.
4. `vroom` for SAC and continuous actions.
5. `flip` for planning and self-play.
6. `kick` for shared policies and a centralized critic.

## Training

Bang keeps one game ID and changes its combat format with `--mode`:

```bash
rl-toybox-train --game bang --mode duel
rl-toybox-train --game bang --mode arena
rl-toybox-train --game bang --mode team_arena
```

Kick uses `--team-size 3`, `5`, or `7`. The VS Code launch file exposes the same choices through simple prompts.

Training prints compact `Ep:` progress lines. PPO-style runs also print `Up:` lines with policy loss, value loss, explained variance, entropy, and approximate KL. Game configs own their observation names, action names, network sizes, algorithm overrides, and training budgets.

## Runs and Checkpoints

Ordinary checkpoints and metrics are generated under `runs/` and ignored by git. Files ending in `_best.pth` are deliberately kept so each game can ship with its best known model.

## Validation

```bash
pip install -e ".[dev]"
python -m ruff check .
python -m compileall core games scripts tests
python -m unittest discover -s tests -p "test_smoke.py" -q
rl-toybox-env-sanity
```

The sanity command resets every active game and takes a few legal steps. CI runs the same checks under a virtual display.

## Shared Conventions

All games use the same 12-color Arcade palette: four neutrals and four two-tone accents. Shared runtime code lives in `core/runtime.py`, reusable drawing code in `core/primitives.py`, and run/checkpoint helpers in `core/io/`.

Observation blocks follow a stable order where they apply: self, sensors or targets, terrain, allies, opponents, map or hazards, then flags. Exact fields remain in each game's `config.py` so the contract stays visible next to the environment.
