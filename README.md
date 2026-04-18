# rl-toybox

Compact Arcade-based RL toybox with one shared composition path, one shared runtime layer, and eight small games.

## Current Games

| Game | Tuned Pairing | Notes | Game README |
| --- | --- | --- | --- |
| `snake` | `qlearn` | Intro discrete control | [games/snake/README.md](games/snake/README.md) |
| `bang` | `dqn` | Flagship discrete arena shooter | [games/bang/README.md](games/bang/README.md) |
| `fuse` | `dqn` | Masked bomb-duel survival and chain reactions | [games/fuse/README.md](games/fuse/README.md) |
| `vroom` | `sac` | Continuous-control racing | [games/vroom/README.md](games/vroom/README.md) |
| `trail` | `ppo` | Adversarial spatial control and territory pressure | [games/trail/README.md](games/trail/README.md) |
| `cardz` | `a2c` | Hidden-information masked card game | [games/cardz/README.md](games/cardz/README.md) |
| `osero` | `search_play` | Board self-play and search | [games/osero/README.md](games/osero/README.md) |
| `kick` | `ppo` | Centralized-critic football project | [games/kick/README.md](games/kick/README.md) |

Pair-specific tuning deltas only remain for:

- `bang + dqn`
- `cardz + a2c`
- `osero + search_play`
- `kick + ppo`

Everything else runs from shared game defaults plus shared algo defaults.

## Canonical Run Flow

Python entrypoints:

```bash
python -m scripts.train --game bang
python -m scripts.play_ai --game bang --render
python -m scripts.play_user --game bang
python -m scripts.capture_demo_ai --game bang --level 3
```

Installed console scripts:

```bash
pip install -e .
rl-toybox-train --game bang
rl-toybox-play-ai --game bang --render
rl-toybox-play-user --game bang
```

Common overrides:

```bash
python -m scripts.train --game fuse --algo dqn --steps 500000 --save-every 20000
python -m scripts.train --game trail --seed 7 --set algo.config.learning_rate=0.0001
python -m scripts.play_ai --game bang --episodes 10 --checkpoint runs/bang/dqn/64_64/L3_best.pth
```

`play_ai` loads the `best` model by default. Use `--model check` or `--checkpoint <path>` when you want a different artifact.

Osero board size is selected through `OSERO_BOARD_SIZE`:

```bash
$env:OSERO_BOARD_SIZE='8x8'; python -m scripts.train --game osero
```

Shared build flow in code:

1. `compose_run_config(...)`
2. `prepare_run(...)`
3. `build_env_from_config(...)`
4. `build_algo_from_config(...)`
5. `build_runner_from_config(...)`

Invalid game/algo combinations fail fast during composition.

## Repo Layout

- [core/game.py](core/game.py): shared game/algo composition, compatibility, run naming, and runtime builders
- [core/runtime.py](core/runtime.py): shared Arcade runtime/window helpers
- [core/pair_overrides.py](core/pair_overrides.py): tiny pair-specific tuning deltas only
- `games/<game>/`: per-game env, config, spec, and README
- `scripts/`: canonical train, play, and capture entrypoints
- [COMPOSITION_NOTES.md](COMPOSITION_NOTES.md): internal composition/build model
- [.vscode/launch.json](.vscode/launch.json): canonical local debug flow mapped onto the same scripts

Per-game gameplay, visuals, observations, rewards, and controls are documented only in each `games/<game>/README.md` plus that game's `config.py`.
