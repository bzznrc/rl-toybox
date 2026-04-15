# rl-toybox

`rl-toybox` is a compact reinforcement-learning playground built around short arcade-style games, shared training infrastructure, and small, inspectable environments. The repo is organized so each game can stand on its own while still reusing common runtime, rendering, evaluation, and algorithm code.

## Repo Layout

- `core/value_discrete/` contains the shared value-based stack used by `snake`, `bang`, and `tower`.
- `core/actor_critic/` contains the shared actor-critic stack used by `vroom`, `frogger`, and `cardz`.
- `core/search_play/` contains the compact MCTS, policy/value, and self-play stack used by `osero`.
- `core/marl_ctde/` contains the shared multi-agent CTDE helpers used by `kick`.
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
| `kick` | Multi-agent football / CTDE showcase | MARL / CTDE | Shared-policy 7v7 left-team football environment with centralized-critic training | [games/kick/README.md](games/kick/README.md) |

## Observation Taxonomy

- Arcade / egocentric control: `SELF -> SENS -> TGT/LAND -> ALLY -> OPP -> MAP/MEM -> HAZ -> FLAG`
- Structured turn-based / masked decision: `GLOB -> PHASE -> BOARD/LANE/SLOT -> HAND/INV -> LEGAL`
- Board self-play / search: `BOARD` only; action masks stay outside the observation
- Blocks can be omitted when they do not apply. Compact canonical prefixes are `self_`, `sens_`, `tgt_`, `land_`, `ally_`, `opp_`, `map_`, `mem_`, `haz_`, `flag_`, `glob_`, `phase_`, `board_`, `lane_`, `slot_`, `hand_`, `inv_`, and `legal_`.
- Current active examples:
  - `snake`: `self_*`, `sens_*`, `tgt_*`
  - `vroom`: `self_*`, `sens_*`, `flag_*`
  - `frogger`: `sens_patch_*` plus `self_*`, `tgt_*`, `land_*`, `flag_*`
  - `kick`: `self_*`, `tgt_*`, `land_*`, `ally*_*`, `opp*_*`, `map_*`, `flag_*`
  - `osero`: `board_r*_c*`
- Per-game `config.py` owns the exact observation/action names, order, dimensions, and default network sizes; root docs and game READMEs should mirror that config truth.

## Default Plans

- `snake` -> `qlearn`, `obs=12`, `act=3`, Q-network `12 -> 32 -> 3`
- `bang` -> `dqn`, `obs=28`, `act=8`, Q-network `28 -> 64 -> 64 -> 8` with double-Q, dueling, and prioritized replay
- `tower` -> `dqn`, `obs=24`, `act=26`, masked Q-network `24 -> 64 -> 64 -> 26` with double-Q and a dueling head
- `vroom` -> `sac`, `obs=20`, `act=3`, actor `20 -> 64 -> 64 -> 3`, twin critics `(20 + 3) -> 64 -> 64 -> 1`
- `frogger` -> `recurrent_ppo`, `obs=32`, `act=5`, encoder `32 -> 32`, LSTM `64`, actor head `64 -> 32 -> 5`, critic head `64 -> 32 -> 1`
- `cardz` -> `a2c`, `obs=64`, `act=16`, shared actor-critic backbone `64 -> 64 -> 64` with direct policy/value heads
- `osero` -> `search_play`, default `6x6`, `obs=36`, `act=37`, policy/value net `36 -> 64 -> 64 -> (37 + 1)`; `4x4` uses `48 -> 48`, `8x8` uses `96 -> 96`
- `kick` -> `ppo`, `obs=56/player`, `act=12`, shared actor `56 -> 96 -> 96 -> 12`, centralized critic `405 -> 192 -> 192 -> 1`