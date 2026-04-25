# rl-toybox

`rl-toybox` is a compact reinforcement-learning playground built around short arcade-style games, one shared composition path, shared runtime/rendering/training infrastructure, and small, inspectable environments. The repo is organized so each game can stand on its own while still reusing common configuration, evaluation, algorithm, and runtime code.

## Repo Layout

- `core/value_discrete/` contains the shared value-based stack used by `snake` and `bang`.
- `core/actor_critic/` contains the shared PPO/SAC stack plus centralized-critic support used by `jump`, `vroom`, and `kick`.
- `core/search_play/` contains the compact MCTS, policy/value, and self-play stack used by `osero`.
- `core/algorithms/` contains the shared algorithm factory and thin common interfaces used by the composition layer.
- `core/shared_config.py` contains the shared runtime/window defaults used across the active games.
- `core/game.py` owns the active game registry, compatibility checks, config composition, and shared run preparation.
- `games/<name>/` contains each game's environment, configuration, and game-specific README.

## Docs

- Repo guide: [docs/repo-guide.md](docs/repo-guide.md)
- RL and environment design guide: [docs/rl-design-guide.md](docs/rl-design-guide.md)

## Clips

<p align="center">
  <img src="media/snake-demo.gif" alt="Snake demo clip" width="32%" />
  <img src="media/bang-demo.gif" alt="Bang demo clip" width="32%" />
  <img src="media/jump-demo.gif" alt="Jump demo clip" width="32%" />
</p>
<p align="center">
  <img src="media/vroom-demo.gif" alt="Vroom demo clip" width="32%" />
  <img src="media/osero-demo.gif" alt="Osero demo clip" width="32%" />
  <img src="media/kick-demo.gif" alt="Kick demo clip" width="32%" />
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

`play_ai` loads `best` by default, so `--model best` is shown only to make the artifact choice explicit. Curriculum-based games now use a shared `L1` to `L5` ladder, with training defaulting to `L1` and play/eval/capture defaulting to `L5`. `osero` remains the temporary exception: its board size is selected through `OSERO_BOARD_SIZE`, and `6x6` is the default.

## Games

| Game ID | Role | Family | Summary | Docs |
| --- | --- | --- | --- | --- |
| `snake` | Intro grid-control game | value-based | Classic Snake with obstacle curriculum, compact egocentric observations, and lightweight shaping rewards | [games/snake/README.md](games/snake/README.md) |
| `bang` | Flagship discrete-control arena game | value-based | Top-down arena shooter focused on movement, aiming, line of sight, and shot timing under pressure | [games/bang/README.md](games/bang/README.md) |
| `jump` | Traversal platformer | actor-critic | Compact side-view micro-platformer built around short procedural runs, timing windows, and simple left/right/jump control | [games/jump/README.md](games/jump/README.md) |
| `vroom` | Continuous-control racing game | actor-critic | One-lap top-down racer with procedural tracks, compact vector observations, and SAC-oriented defaults | [games/vroom/README.md](games/vroom/README.md) |
| `osero` | Planning + self-play capstone | search + self-play | Compact Osero/Reversi implementation using MCTS, self-play, and a small policy/value network | [games/osero/README.md](games/osero/README.md) |
| `kick` | Multi-agent football / CTDE showcase | actor-critic / CTDE | Shared-policy top-down `7v7` football environment with centralized-critic PPO training | [games/kick/README.md](games/kick/README.md) |

## Observation Taxonomy

- Arcade / egocentric control: `SELF -> SENS -> TGT/LAND/OPP -> HAZ -> FLAG`
- Team / CTDE control: `SELF -> TGT -> LAND -> ALLY -> OPP -> MAP -> FLAG`
- Board self-play / search: `BOARD` only; legal moves stay outside the observation via action masking
- Blocks can be omitted when they do not apply. Compact canonical prefixes are `self_`, `sens_`, `tgt_`, `land_`, `ally_`, `opp_`, `map_`, `haz_`, `flag_`, and `board_`.

Current active examples:

- `snake`: `self_*`, `sens_*`, `tgt_*`
- `bang`: `self_*`, `sens_*`, `opp*_*`, `haz_*`
- `jump`: `self_*`, `sens_*`, `land_*`, `opp*_*`, `haz_*`, `flag_*`
- `vroom`: `self_*`, `sens_*`, `flag_*`
- `kick`: `self_*`, `tgt_*`, `land_*`, `ally*_*`, `opp*_*`, `map_*`, `flag_*`
- `osero`: `board_r*_c*`

Per-game `config.py` owns the exact observation/action names, order, dimensions, model defaults, and training stop budget. The standard active-game template is `DEFAULT_ALGO`, `DEFAULT_MODEL_CONFIG`, `ALGO_CONFIG_OVERRIDES`, and `DEFAULT_TRAIN_CONFIG`. Change `DEFAULT_MODEL_CONFIG["hidden_sizes"]` to set one game-wide network size across supported models, and use `DEFAULT_MODEL_CONFIG["critic_hidden_sizes"]` when a game has a separate critic shape. Only use `ALGO_CONFIG_OVERRIDES[algo_id]` for true algo-specific deltas such as PPO entropy, DQN replay settings, or search-play simulations. Change `DEFAULT_TRAIN_CONFIG["budget"]` to change when a game's training run stops, including when you launch that game with a non-default compatible algo; the budget unit is total environment steps for value-based and actor-critic families, and self-play games for `search_play`. Runner-specific extras such as `rollout_steps` still only apply to runners that use them. The root docs and game READMEs should mirror that config truth.

## Default Plans

- `snake` -> `qlearn`, `obs=12`, `act=3`, Q-network `12 -> 32 -> 3`
- `bang` -> `dqn`, `obs=28`, `act=8`, Q-network `28 -> 64 -> 64 -> 8` with double-Q, a dueling head, and prioritized replay
- `jump` -> `ppo`, `obs=36`, `act=4`, actor `36 -> 32 -> 32 -> 4`, critic `36 -> 32 -> 32 -> 1`
- `vroom` -> `sac`, `obs=20`, `act=3`, actor `20 -> 64 -> 64 -> 3`, twin critics `(20 + 3) -> 64 -> 64 -> 1`
- `osero` -> `search_play`, default `6x6`, `obs=36`, `act=37`, policy/value net `36 -> 64 -> 64 -> (37 + 1)`; `4x4` uses `16 -> 48 -> 48`, `8x8` uses `64 -> 96 -> 96`
- `kick` -> `ppo`, `obs=63/player`, `act=11`, shared actor trunk `63 -> 64 -> 64` with 4 role heads `-> 11`, centralized critic `454 -> 128 -> 128 -> 1`; actions are 8 movement directions plus direction-respecting semantic `pass` / `shoot`

There is no post-config pair-override layer for the active games. Shared algorithm defaults provide the family baseline, and each active game's `config.py` is the final default source before explicit user overrides.
