# VS Code Launch Notes

The launch configs mirror the canonical CLI flow:

- `scripts.train`
- `scripts.play_user`
- `scripts.play_ai`
- `scripts.capture_demo_ai`

## How It Is Organized

There is one canonical launch flow for active games:

- `Run - Train`
- `Run - Play User`
- `Run - Play AI`
- `Run - Capture Demo`

The shared flow covers:
- `snake`
- `bang`
- `jump`
- `vroom`
- `four`
- `kick`

`four` is fixed-mode. It can use the same launch configs safely because the CLI clamps fixed-mode games to `L1` and does not create a curriculum.

## Levels And Modes

- The `trainLevel` and `playLevel` inputs expose `L1` through `L5`.
- Training defaults to `L1`.
- `play-user`, `play-ai`, and `capture-demo` default to `L5`.
- Curriculum games use those levels normally.
- Four ignores higher level selections and resolves to `L1`.
- `7x6` is the fixed Four board shape.

## Algo Dropdowns

The train / play-ai / capture configs keep an algo dropdown because the CLI supports `--algo`.

Those dropdowns are intentionally static as well. VS Code will not auto-filter incompatible game/algo pairs. If you pick an invalid pair, the repo's compatibility checks will fail fast with a clear error.

The shared default is now:

- `Run - *` defaults to `auto`, which resolves to each game's `DEFAULT_ALGO` from `games/<game>/config.py`

## Adding A New Game Later

- If a new curriculum-based game follows the shared ladder, add it to `activeGame` and make sure its config exposes `LEVEL_SETTINGS[1..5]`.
- If a new fixed-mode game is added, include it in `activeGame` and add its id to `FIXED_LEVEL_GAME_IDS` in `core/game.py`.
