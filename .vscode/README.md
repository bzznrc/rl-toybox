# VS Code Launch Notes

The launch configs mirror the canonical CLI flow:

- `scripts.train`
- `scripts.play_user`
- `scripts.play_ai`
- `scripts.capture_demo_ai`

## How It Is Organized

There are four everyday actions:

- `Run - Train`
- `Run - Play User`
- `Run - Play AI`
- `Run - Capture Demo`

All games now use the same shared launch shape:

- `snake`
- `bang`
- `vroom`
- `osero`
- `kick`

The generic `level3` input is remapped in shared runtime code:

- normal games: `1 -> 1`, `2 -> 2`, `3 -> 3`
- `kick`: `1 -> 1`, `2 -> 3`, `3 -> 5`
- `osero`: `1 -> 4x4`, `2 -> 6x6`, `3 -> 8x8`

## Algo Dropdowns

The train / play-ai / capture configs keep an algo dropdown because the CLI supports `--algo`.

Those dropdowns are intentionally static as well. VS Code will not auto-filter incompatible game/algo pairs. If you pick an invalid pair, the repo's compatibility checks will fail fast with a clear error.

The defaults are chosen to match the default launch game:

- `Run - *` defaults to `bang + dqn`

## Adding A New Game Later

If a new game can use the shared 3-level launch flow, add it to the `gameLevel3` input only.
