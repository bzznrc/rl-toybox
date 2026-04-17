# VS Code Launch Notes

The launch configs now mirror the canonical CLI flow:

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

Those cover the shared 1-3 level games:

- `snake`
- `bang`
- `tower`
- `vroom`
- `frogger`
- `cardz`

Two small sets of dedicated entries remain:

- `... - Kick`
- `... - Osero`

That is intentional.

## Why Kick And Osero Still Have Dedicated Entries

VS Code input dropdowns are static. They cannot change their options based on another dropdown.

We keep dedicated entries only where the runtime shape is genuinely different:

- `kick` uses a `1-5` level range instead of `1-3`
- `osero` does not use curriculum levels here and still needs the `OSERO_BOARD_SIZE` env input

Everything else shares the same canonical launch shape.

## Algo Dropdowns

The train / play-ai / capture configs keep an algo dropdown because the CLI supports `--algo`.

Those dropdowns are intentionally static as well. That means VS Code will not auto-filter incompatible game/algo pairs. If you pick an invalid pair, the repo's compatibility checks will fail fast with a clear error.

The defaults are chosen to match the default game in each launch group:

- `Run - *` defaults to `bang + dqn`
- `Run - * - Kick` defaults to `kick + ppo`
- `Run - * - Osero` defaults to `osero + search_play`

## Adding A New Game Later

If a new game:

- uses the normal `1-3` level flow
- does not need extra env vars

add it to the `gameLevel3` input only.

If a new game needs a different level range or extra env wiring, add one small dedicated set like `Kick` or `Osero` instead of bringing back per-game triplets for everything.
