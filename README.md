# rl-toybox

A small RL playground organized around tiny games and a clearer algorithm taxonomy.

## Overview

- `core/value_discrete/` holds shared value-based infrastructure for `snake`, `bang`, and `tower`.
- `core/actor_critic/` holds shared actor-critic infrastructure for `vroom`, `frogger`, and `cardz`.
- `core/search_play/` holds the compact MCTS, policy/value, and self-play pieces used by `osero`.
- `core/marl_ctde/` isolates paused experimental multi-agent helpers for `kick`.
- `core/game.py` owns the active lineup registry, role metadata, and shared run preparation.
- `games/<name>/` owns per-game env logic, configs, specs, and README snapshots.

## Framework Docs

- Repo/codebase architecture: [docs/repo-architecture.md](docs/repo-architecture.md)
- Cross-game RL/environment design guide: [docs/rl-design-guide.md](docs/rl-design-guide.md)
- Docs index: [docs/README.md](docs/README.md)
- Migration note for this refactor: [docs/migration-lineup-refactor.md](docs/migration-lineup-refactor.md)

## Clips

<p>
  <img src="media/snake-demo.gif" width="32%">
  <img src="media/bang-demo.gif" width="32%">
  <img src="media/vroom-demo.gif" width="32%">
</p>

## Run

With package install:

```bash
pip install -e .
rl-toybox-train --game bang
rl-toybox-play-ai --game bang --model best --render
rl-toybox-play-user --game bang
```

Without installation, from repo root:

```bash
python -m scripts.train --game bang
python -m scripts.play_ai --game bang --model best --render
python -m scripts.play_user --game bang
```

## Games

| Game ID | Repo Role | Primary Family | Runtime Status | Notes | Docs |
| --- | --- | --- | --- | --- | --- |
| `snake` | Intro game | simple value-based | active | Keep; current runtime stays on linear Q / q-learning, with DQN as a future extension | [games/snake/README.md](games/snake/README.md) |
| `bang` | Flagship discrete RL game | advanced value-based | active | Keep; current DQN stack already maps well to the Rainbow-lite direction | [games/bang/README.md](games/bang/README.md) |
| `tower` | Delayed reward + action masking showcase | value-based | active | Tiny wave-based tower defense built around masked build-phase planning | [games/tower/README.md](games/tower/README.md) |
| `vroom` | Continuous control showcase | actor-critic | active | Continuous-control top-down racer with SAC-oriented defaults and compact vector observations | [games/vroom/README.md](games/vroom/README.md) |
| `frogger` | Memory / POMDP showcase | actor-critic | active | Compact Frogger-style road crossing with recurrent PPO-friendly partial observability | [games/frogger/README.md](games/frogger/README.md) |
| `cardz` | Simple stochastic hidden-info game | actor-critic | active | Tiny 3-lane hidden-info card duel with masked actions and A2C-oriented defaults | [games/cardz/README.md](games/cardz/README.md) |
| `osero` | Planning + self-play capstone | search + self-play | active | AlphaZero-lite Osero with flattened board IO, MCTS, and a small policy/value net | [games/osero/README.md](games/osero/README.md) |
| `kick` | Paused multi-agent CTDE project | MARL / CTDE | paused | Kept in repo, but explicitly outside the main active ladder | [games/kick/README.md](games/kick/README.md) |

## Default Plans

- `snake` -> linear Q / q-learning (`qlearn`, `obs=12`, `act=3`, hidden `[32]`)
- `bang` -> Rainbow-lite DQN direction (`dqn`, `obs=24`, `act=8`, hidden `[64, 64]`)
- `tower` -> masked Double DQN tower defense (`dqn`, `obs=20`, `act=26`, hidden `[64, 64]`)
- `vroom` -> SAC continuous-control racer (`sac`, `obs=20`, `act=3`, hidden `[128, 128]`)
- `frogger` -> recurrent PPO (`recurrent_ppo`, `obs=32`, `act=5`, encoder `[32]`, lstm `64`, heads `[32]`)
- `cardz` -> masked hidden-info lane-control card game (`a2c`, `obs=32`, `act=16`, shared trunk `[96, 96]`)
- `osero` -> AlphaZero-lite Osero (`search_play`, default `6x6`, `obs=36`, `act=37`, trunk `[128, 128]`; `8x8` supported with `[128, 128, 128]`)
- `kick` -> paused MAPPO-style PPO project (`ppo`, actor `[128, 128]`, critic `[256, 256]`)
