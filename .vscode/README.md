# VS Code Launch Notes

The launch configs mirror the canonical CLI flow:

- `scripts.train`
- `scripts.play_user`
- `scripts.play_ai`
- `scripts.capture_demo_ai`

## How It Is Organized

There are two canonical launch flows:

Curriculum-based games:
- `Run - Train`
- `Run - Play User`
- `Run - Play AI`
- `Run - Capture Demo`

Osero:
- `Run - Train (Osero)`
- `Run - Play User (Osero)`
- `Run - Play AI (Osero)`
- `Run - Capture Demo (Osero)`

The shared curriculum flow covers:
- `snake`
- `bang`
- `jump`
- `vroom`
- `kick`

`osero` stays separate because it still uses board-size modes rather than the repo's shared curriculum ladder.

## Levels And Modes

- Curriculum games expose `curriculumTrainLevel` and `curriculumPlayLevel` directly as `L1` through `L5`.
- Training defaults to `L1`.
- `play-user`, `play-ai`, and `capture-demo` default to `L5`.
- Osero uses `oseroBoardSize` instead of `--level`, with `4x4`, `6x6`, and `8x8` options.
- `6x6` is the default Osero board size.

## Algo Dropdowns

The train / play-ai / capture configs keep an algo dropdown because the CLI supports `--algo`. That applies to both the curriculum flow and the Osero-specific flow.

Those dropdowns are intentionally static as well. VS Code will not auto-filter incompatible game/algo pairs. If you pick an invalid pair, the repo's compatibility checks will fail fast with a clear error.

The shared default is now:

- `Run - *` defaults to `auto`, which resolves to each game's `DEFAULT_ALGO` from `games/<game>/config.py`

## Adding A New Game Later

- If a new curriculum-based game follows the shared ladder, add it to `curriculumGame` and make sure its config exposes `LEVEL_SETTINGS[1..5]`.
- If a game behaves more like Osero, keep a separate mode-specific launch block instead of forcing it into the curriculum inputs.
