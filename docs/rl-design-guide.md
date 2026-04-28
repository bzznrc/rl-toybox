# RL Design Guide

This document defines cross-game RL/environment design practices.
It is intentionally game-agnostic.

Current implementation snapshots are owned by each `games/<game>/README.md`.

## Table of Contents

- [1) Runtime Contract](#1-runtime-contract)
- [2) Render and Performance Patterns](#2-render-and-performance-patterns)
- [3) Config Contract](#3-config-contract)
- [4) Observation Taxonomy](#4-observation-taxonomy)
- [5) Action Space Conventions](#5-action-space-conventions)
- [6) Reward Framework](#6-reward-framework)
- [7) Curriculum Framework](#7-curriculum-framework)
- [8) Game README Contract](#8-game-readme-contract)
- [9) Checklist for Environment Changes](#9-checklist-for-environment-changes)

## 1) Runtime Contract

### Step/reset

- `reset()` returns an `np.ndarray` observation (or a documented multi-agent equivalent).
- `step(action)` returns `(obs, reward, done, info)`.
- `reward` is scalar float unless a multi-agent game explicitly documents additional `info["reward_vec"]` behavior.
- Extra signals are exposed through `info[...]`.

### Frame pacing

- Use `ArcadeFrameClock.tick(...)` when a game renders in real time.
- New rendered games should use `ArcadeEnvMixin` for window setup, frame pacing, and close behavior unless they need a documented exception.
- Keep render FPS and training FPS separate.
- `TRAINING_FPS=0` means max-throughput training.
- Use explicit eval/play step-delay knobs when a game advances one full decision per rendered frame and the default FPS would make play unreadable.

### Terminal commit

- Success/failure should come from explicit terminal conditions.
- Render-only terminal holds should not change already-latched outcomes.

## 2) Render and Performance Patterns

- Reuse static visual assets when possible.
- Keep draw-call pressure intentional.
- Avoid expensive per-frame text churn in hot loops.
- Gate expensive debug-only work behind explicit flags.
- Performance changes must not alter observation, reward, or termination semantics.

### Visual building blocks

- Treat the repo's standard visual atom as the default `DEFAULT_TILE_SIZE` square block from `core/arcade_style.py`.
- This standard block is the same base unit used by Bang's actors and obstacles, and by Vroom's car body.
- Scale from that base deliberately:
  - `1x` is the default and should be the starting point for most gameplay-facing objects.
  - `2x` is acceptable for larger composite cells or overlays such as Jump's grid-style patches.
  - `0.5x` is acceptable for finer block-derived detail such as Vroom's rasterized track surface and lane markings.
- Prefer composing visuals from these square units instead of introducing unrelated bespoke shapes or gradients.
- Prefer the shared palette from `core/arcade_style.py`; new visuals should read as arrangements of standard blocks plus palette colors, not as independent art systems.
- When outlines are used, keep their thickness intentional relative to the standard block and avoid decorative inner grid noise unless it carries gameplay meaning.
- Optional observation ghosts use `X` as the standard rendered-mode toggle. Sensor rays, SENS grids, route probes, and role/area guides should use the shared light-neutral ghost color at roughly 50% alpha through `core.ghost_overlay`.

## 3) Config Contract

### Section order

Each game config should prefer this order:

1. `ENV`
2. `IO`
3. `CURRICULUM`
4. `REWARDS`
5. `TRAINING`

If a game needs extra sections, keep them clearly game-owned and place them deliberately rather than recreating shared runtime boilerplate.

### Ownership boundaries

- Keep `config.py` declarative and constants-only.
- Put shared training/runtime boilerplate in shared `core/` code.
- `core/shared_config.py` owns the default shared runtime/window constants:
  - render and training FPS
  - shared screen, playfield, and world geometry
  - shared tile sizing helpers
  - common marker sizing
- Per-game `config.py` files should import those defaults instead of redefining them, unless a game truly needs an override.
- Keep future placeholder configs lightweight rather than speculative.
- Treat `config.py` as the per-game source of truth for `INPUT_FEATURE_NAMES`, `ACTION_NAMES`, observation/action dimensions, default network sizes, and the default training stop budget in `DEFAULT_TRAIN_CONFIG["budget"]`.
- The standard active-game training/config template is `DEFAULT_ALGO`, `DEFAULT_MODEL_CONFIG`, `ALGO_CONFIG_OVERRIDES`, and `DEFAULT_TRAIN_CONFIG`.
- Put cross-algo model fundamentals such as `hidden_sizes` and `critic_hidden_sizes` in `DEFAULT_MODEL_CONFIG`.
- Use `ALGO_CONFIG_OVERRIDES[algo_id]` only for true algo-specific deltas such as PPO entropy, DQN replay/exploration settings, or search-play simulations.
- `DEFAULT_TRAIN_CONFIG["budget"]` is the common stop knob across active games; its unit is total environment steps for value-based and actor-critic families, and self-play games for search-play. It should still apply when a game is launched with a non-default compatible algo.
- Runner-specific train keys such as `rollout_steps`, `train_after_steps`, or `updates_per_game` should only affect runners that actually use them.
- When a game needs non-default action-space bounds, capability flags, env metadata, or default algo/train config, keep them in `config.py` and let `core/game.py` build the shared spec directly from `config.py` + `env.py`.
- Active games should not rely on a hidden post-config pair-override layer. Shared family defaults may supply the baseline, but `config.py` is the final default layer before explicit user overrides.

## 4) Observation Taxonomy

### Family block ordering

Use one of these family templates, and omit non-applicable blocks cleanly instead of inserting dummy features.

Arcade / egocentric control:

1. `SELF`
2. `SENS`
3. `TGT/LAND`
4. `ALLY`
5. `OPP`
6. `MAP/MEM`
7. `HAZ`
8. `FLAG`

Structured turn-based / masked decision:

1. `GLOB`
2. `PHASE`
3. `BOARD/LANE/SLOT`
4. `HAND/INV`
5. `LEGAL`

Board self-play / search:

1. `BOARD`

For board self-play, the action mask stays outside the observation.

### Naming rules

- Use compact canonical prefixes: `self_`, `sens_`, `tgt_`, `land_`, `ally_`, `opp_`, `map_`, `mem_`, `haz_`, `flag_`, `glob_`, `phase_`, `board_`, `lane_`, `slot_`, `hand_`, `inv_`, `legal_`
- Keep canonical prefixes to at most 5 characters before the underscore
- Use `_id` for categorical scalar codes
- Use `_norm` for normalized continuous scalars
- Keep `*_sin` paired with `*_cos`
- Keep `dx/dy` and `dvx/dvy` pairs adjacent
- Boolean features should be numeric `0.0/1.0`
- Keep block order intentional and documented in each game README

### Current repo examples

- `snake`: `self_*`, `sens_*`, `tgt_*`
- `bang`: `self_*`, `sens_*`, `opp_*`, `haz_*`
- `jump`: `self_*`, `sens_*`, `land_*`, `opp_*`, `flag_*`
- `vroom`: `self_*`, `sens_*`, `flag_*`
- `kick`: `self_*`, `tgt_*`, `land_*`, `ally*_*`, `opp*_*`
- `four`: `board_r*_c*`

## 5) Action Space Conventions

- Prefer discrete actions unless continuous control is core to the game.
- Keep action names explicit and verb-oriented.
- When action masking exists, apply it consistently in training, evaluation, and policy scoring.
- `vroom` and future continuous-control games are allowed to break the discrete-default rule when the control problem truly needs it.

## 6) Reward Framework

- Keep outcome magnitudes broadly comparable across games.
- Dense shaping should support, not dominate, outcomes.
- Prefer interpretable, bounded shaping terms.
- Log realized reward contributions, not just reward constants.
- Prefer declaring a compact `RewardSpec` from the game's config constants and mapping realized reward keys to short display codes.

## 7) Curriculum Framework

- Prefer a smooth shared 5-level curriculum for curriculum-based games.
- Use the previous anchor points as `L1 -> L1`, `L2 -> L3`, and `L3 -> L5`, then add bridge levels at `L2` and `L4`.
- Preserve prior top-end difficulty at the new `L5`; the goal is smoother interpolation, not a broader or harder overall ladder.
- fixed-mode board games such as `four` use a documented fixed `L1` slot instead of staged curriculum levels.
- Promote using success metrics when dense shaping could distort reward totals.
- Keep per-level knobs in clearly named config tables.

## 8) Game README Contract

### Canonical game order

When docs enumerate the active lineup, use:

1. `snake`
2. `bang`
3. `jump`
4. `vroom`
5. `four`
6. `kick`

### Required top-level section order

Each active `games/<game>/README.md` must use this top-level heading order:

1. `Clip`
2. `Algorithm / Network`
3. `Controls (Human)`
4. `Observation / Actions`
5. `Environment Notes`
6. `Rewards (Training)`
7. `Curriculum (Train)`
8. `Run Commands`

Games with no staged progression should still keep the same section structure and state that the curriculum is fixed.

### Config mirroring

- Each game README should copy observation/action dimensions, ordered feature names, and default network sizes from that game's `config.py`.
- Root-level summaries should also defer to per-game `config.py` rather than re-stating older values from memory.

## 9) Checklist for Environment Changes

Before merging environment changes:

- [ ] Config ownership remains clean.
- [ ] Observation ordering and naming stay intentional.
- [ ] Reward magnitudes remain balanced and interpretable.
- [ ] Action masking is consistent when used.
- [ ] README and docs still describe the implemented game accurately.
- [ ] `python -m scripts.validate_docs` passes after README/doc edits.

For code changes that intentionally move away from this guide, ask first, then update the guide in the same approved change so future work follows the new rule instead of the old one.
