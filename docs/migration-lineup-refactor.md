# Lineup Refactor Migration Note

## What Changed

- Active lineup is now:
  - `snake`
  - `bang`
  - `tower`
  - `vroom`
  - `frogger`
  - `card`
  - `osero`
  - `kick`
- `walk` was removed from the active repo structure.
- `kick` remains present but is now explicitly marked paused / experimental.
- The memory / partial-observability slot is now owned by `frogger`.
- Shared RL code was reorganized toward:
  - `core/value_discrete/`
  - `core/actor_critic/`
  - `core/search_play/`
  - `core/marl_ctde/`

## What Was Intentionally Not Done Yet

- At the time of the refactor note, `tower`, `card`, and `osero` were still pending.
- Current repo state has `tower` and `osero` implemented; `card` remains scaffolded.

## Compatibility Notes

- Existing kept games were preserved as much as practical.
- `vroom` now runs on the actor-critic / continuous-control branch with SAC-oriented defaults.
- `frogger` is now the active recurrent PPO / memory game in the repo.
- Scaffold games still use placeholder envs where implementation has not started yet.
