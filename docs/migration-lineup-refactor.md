# Lineup Refactor Migration Note

## What Changed

- Active lineup is now:
  - `snake`
  - `bang`
  - `fuse`
  - `vroom`
  - `trail`
  - `cardz`
  - `osero`
  - `kick`
- `walk` was removed from the active repo structure.
- The bomb-duel slot now lives as `fuse`.
- Two older game folders were removed from the repo.
- Shared RL code was reorganized toward:
  - `core/value_discrete/`
  - `core/actor_critic/`
  - `core/search_play/`
  - `core/marl_ctde/`

## What Was Intentionally Not Done Yet

- This note now only records the current kept lineup and shared-family split.

## Compatibility Notes

- Existing kept games were preserved as much as practical.
- `vroom` now runs on the actor-critic / continuous-control branch with SAC-oriented defaults.
- The hidden-info actor-critic slot is filled by `cardz`.
