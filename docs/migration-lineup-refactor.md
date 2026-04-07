# Lineup Refactor Migration Note

## What Changed

- Active lineup is now:
  - `snake`
  - `bang`
  - `tower`
  - `vroom`
  - `stealth`
  - `card`
  - `othello`
  - `kick`
- `peek` was retired from the active lineup and fully replaced by `stealth`.
- `walk` was removed from the active repo structure.
- `kick` remains present but is now explicitly marked paused / experimental.
- Shared RL code was reorganized toward:
  - `core/value_discrete/`
  - `core/actor_critic/`
  - `core/search_play/`
  - `core/marl_ctde/`

## What Was Intentionally Not Done Yet

- No `tower`, `card`, or `othello` implementations yet.

## Compatibility Notes

- Existing kept games were preserved as much as practical.
- `vroom` now runs on the actor-critic / continuous-control branch with SAC-oriented defaults.
- `stealth` is now the only active memory/stealth game in the repo.
- Scaffold games still use placeholder envs where implementation has not started yet.
