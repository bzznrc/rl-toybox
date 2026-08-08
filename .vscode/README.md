# VS Code Launch Notes

The launch configs mirror the canonical CLI flow:

- `scripts.train`
- `scripts.play_user`
- `scripts.play_ai`
- `scripts.capture_demo`

## How It Is Organized

There is one canonical launch flow for active games without game-specific selectors:

- `Run - Train`
- `Run - Play User`
- `Run - Play AI`
- `Run - Capture Demo`

The shared flow covers:
- `snake`
- `jump`
- `vroom`
- `flip`

Bang has its own launch flow because it has a mode selector:

- `Bang - Train`
- `Bang - Play User`
- `Bang - Play AI`
- `Bang - Capture Demo`

Kick has its own launch flow because it has a team-size selector:

- `Kick - Train`
- `Kick - Play User`
- `Kick - Play AI`
- `Kick - Capture Demo`

`flip` is fixed-mode. It can use the shared non-Kick launch configs safely because the CLI clamps fixed-mode games to `L1` and does not create a curriculum.
`bang` uses the additional `bangMode` selector for `Duel`, `Arena`, or `Team Arena`.
`kick` uses the additional `kickTeamSize` selector for `3 vs. 3`, `5 vs. 5`, or `7 vs. 7`.

## Levels And Modes

- The `trainLevel` and `playLevel` inputs expose `L1` through `L5`.
- The `bangMode` input appears only in the Bang launch configs and exposes Bang's `Duel`, `Arena`, and `Team Arena` modes.
- The `kickTeamSize` input appears only in the Kick launch configs and exposes Kick's `3 vs. 3`, `5 vs. 5`, and `7 vs. 7` modes.
- Training defaults to `L1`.
- `play-user`, `play-ai`, and `capture-demo` default to `L5`.
- Curriculum games use those levels normally.
- Flip ignores higher level selections and resolves to `L1`.
- `6x6` is the fixed Flip board shape.

## Algo Dropdowns

The train / play-ai / capture configs keep an algo dropdown because the CLI supports `--algo`.

Those dropdowns are intentionally static as well. VS Code will not auto-filter incompatible game/algo pairs. If you pick an invalid pair, the repo's compatibility checks will fail fast with a clear error.

The shared default is now:

- `Run - *` defaults to `auto`, which resolves to each game's `DEFAULT_ALGO` from `games/<game>/config.py`

## Adding A New Game Later

- If a new curriculum-based game follows the shared ladder and has no special launch options, add it to `activeGame` and make sure its config exposes a clear per-level settings table for `L1` through `L5`.
- If a new fixed-mode game is added and has no special launch options, include it in `activeGame` and add its id to `FIXED_LEVEL_GAME_IDS` in `core/game.py`.
- If a game needs its own launch-only selector, give it separate launch configs like Kick so unrelated games do not inherit its prompts.
