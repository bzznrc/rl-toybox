# rl-toybox

`rl-toybox` is a compact reinforcement-learning playground built around short arcade-style games, shared training infrastructure, and small, inspectable environments. The repo is organized so each game can stand on its own while still reusing common runtime, rendering, evaluation, and algorithm code.

## Repo Layout

- `core/value_discrete/` contains the shared value-based stack used by `snake`, `bang`, and `tower`.
- `core/actor_critic/` contains the shared actor-critic stack used by `vroom`, `frogger`, and `cardz`.
- `core/search_play/` contains the compact MCTS, policy/value, and self-play stack used by `osero`.
- `core/marl_ctde/` contains paused multi-agent CTDE helpers used by `kick`.
- `core/game.py` owns the active game registry, metadata, and shared run preparation.
- `games/<name>/` contains each game's environment, configuration, spec, and game-specific README.

## Docs

- Repo architecture: [docs/repo-architecture.md](docs/repo-architecture.md)
- RL and environment design guide: [docs/rl-design-guide.md](docs/rl-design-guide.md)
- Docs index: [docs/README.md](docs/README.md)
- Refactor/migration notes: [docs/migration-lineup-refactor.md](docs/migration-lineup-refactor.md)

## Clips

<p>
  <img src="media/snake-demo.gif" width="32%">
  <img src="media/bang-demo.gif" width="32%">
  <img src="media/vroom-demo.gif" width="32%">
</p>

## Quick Start

With package install:

```bash
pip install -e .
rl-toybox-train --game bang
rl-toybox-play-ai --game bang --model best --render
rl-toybox-play-user --game bang
```

Without installation, from the repo root:

```bash
python -m scripts.train --game bang
python -m scripts.play_ai --game bang --model best --render
python -m scripts.play_user --game bang
```

## Games

| Game ID | Role | Family | Summary | Docs |
| --- | --- | --- | --- | --- |
| `snake` | Intro grid-control game | value-based | Classic Snake with obstacle curriculum, compact vector observations, and lightweight shaping rewards | [games/snake/README.md](games/snake/README.md) |
| `bang` | Flagship discrete-control arena game | value-based | Top-down shooter focused on movement, line of sight, and timing shots under pressure | [games/bang/README.md](games/bang/README.md) |
| `tower` | Delayed-reward and action-masking showcase | value-based | Tiny wave-based tower defense where the policy acts only during build phases | [games/tower/README.md](games/tower/README.md) |
| `vroom` | Continuous-control showcase | actor-critic | One-lap top-down racer with procedural tracks, compact vector observations, and SAC-oriented defaults | [games/vroom/README.md](games/vroom/README.md) |
| `frogger` | Partial-observability / memory showcase | actor-critic | Compact road-crossing game designed around local sensing, timing, and recurrent-policy friendly observations | [games/frogger/README.md](games/frogger/README.md) |
| `cardz` | Stochastic hidden-information game | actor-critic | Two-player lane-control card game with masked actions and a scripted opponent | [games/cardz/README.md](games/cardz/README.md) |
| `osero` | Planning and self-play capstone | search + self-play | Small Osero/Reversi implementation using MCTS, self-play, and a compact policy/value network | [games/osero/README.md](games/osero/README.md) |
| `kick` | Experimental multi-agent football project | MARL / CTDE | Shared-policy left-team football environment kept in the repo as a paused experimental branch | [games/kick/README.md](games/kick/README.md) |

## Default Training Setups

- `snake`: linear Q-learning (`qlearn`, `obs=12`, `act=3`, hidden `[32]`)
- `bang`: DQN (`dqn`, `obs=24`, `act=8`, hidden `[64, 64]`)
- `tower`: masked Double DQN (`dqn`, `obs=20`, `act=26`, hidden `[64, 64]`)
- `vroom`: SAC (`sac`, `obs=20`, `act=3`, hidden `[64, 64]`)
- `frogger`: recurrent PPO (`recurrent_ppo`, `obs=32`, `act=5`, encoder `[32]`, lstm `64`, heads `[32]`)
- `cardz`: actor-critic (`a2c`, `obs=30`, `act=16`, shared trunk `[80, 80]`)
- `osero`: search/self-play (`search_play`, default `6x6`, `obs=36`, `act=37`, trunk `[96, 96]`; `8x8` also supported)
- `kick`: paused PPO/MAPPO-style project (`ppo`, actor `[128, 128]`, critic `[256, 256]`)
