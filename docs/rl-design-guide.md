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
- [8) Checklist for Environment Changes](#8-checklist-for-environment-changes)

## 1) Runtime Contract

### Step/reset
- `reset()` returns `np.ndarray` observation of shape `(OBS_DIM,)` (or game-documented multi-agent equivalent).
- `step(action)` returns `(obs, reward, done, info)`.
- `reward` is scalar float.
- Extra signals (for example reward vectors) are provided via `info[...]`.

### Frame pacing
- Use `ArcadeFrameClock.tick(...)`.
- Use render/training FPS split:
  - `frame_clock.tick(FPS if show_game else TRAINING_FPS)`
- `TRAINING_FPS=0` means no sleep and max throughput.

### Terminal commit
- Success/failure must come from explicit terminal conditions.
- If using render-only terminal holds, latch terminal semantics first so outcome does not drift during the hold.
- Configure hold durations in seconds and convert to frames via `FPS`.

## 2) Render and Performance Patterns

### Reuse static visual assets
- Build static masks/backgrounds/textures once and reuse them each frame.
- Prefer drawing textures over regenerating complex geometry every frame.

### Keep draw-call pressure intentional
- Decouple simulation fidelity from render mesh density where possible.
- Typical pattern:
  - finer sampling for physics/contact
  - coarser sampling for filled visual meshes
  - single strip/polyline for outlines

### Avoid unnecessary per-frame text overhead
- Avoid expensive per-frame text creation patterns in hot loops.
- Prefer reusable text objects (`arcade.Text`) or `TextCache`.
- If exact frame-by-frame updates are unnecessary, throttle display updates.

### Gate optional expensive logic
- Keep debug and display-only computations behind explicit mode/flag checks.

### Preserve runtime semantics
- Performance optimizations must not break observation, reward, or termination semantics.

## 3) Config Contract

### Section order
Each game config should keep this structure:
1. `RUNTIME`
2. `ENV`
3. `IO`
4. `CURRICULUM`
5. `REWARDS`
6. `TRAINING`

### Section expectations
- `RUNTIME`: `WINDOW_TITLE`, `FPS`, `TRAINING_FPS`, `USE_GPU`
- `ENV`: geometry/physics/limits
- `IO`: `INPUT_FEATURE_NAMES`, `ACTION_NAMES`, `OBS_DIM`, `ACT_DIM`
- `CURRICULUM`: level bounds, promotion settings, per-level settings
- `REWARDS`: reward magnitudes and taxonomy keys
- `TRAINING`: model and optimizer/training hyperparameters

### Ownership boundaries
- Do not duplicate a knob in multiple places.
- Keep path/artifact boilerplate out of game config when it belongs to shared code.
- Keep `config.py` modules constants-only: no `def` helpers inside config files.
- If a game needs derived settings, either precompute constant tables in `config.py` or move the operational logic into the owning runtime module (`env.py`, `spec.py`, shared core code).

## 4) Observation Taxonomy

### Feature block ordering
Use this order when applicable:
1. `SELF` (`self_*`)
2. `RAYS` (`ray_*`)
3. `TGT` (`tgt_*`)
4. `GOALS/LANDMARKS` (`goal_*`, `own_goal_*`)
5. `TEAMMATES` (`ownN_*`)
6. `OPPONENTS` (`oppN_*`)
7. `TRACK/MAP` (`trk_*`)
8. `HAZARDS` (`haz_*`)

### Naming and symmetry rules
- `*_sin` pairs with `*_cos`
- `dx` pairs with `dy`
- `dvx` pairs with `dvy`

### Normalization rules
- Normalize relative positions consistently.
- Prefer sin/cos for angles, unless a game intentionally documents a compact scalar-angle exception.
- Boolean features should be numeric `0.0/1.0`.

### Stable nearest ordering
- Nearest-N selections must be deterministic (for example `(distance, stable_slot_index)`).

## 5) Action Space Conventions

- Prefer discrete actions unless continuous control is core to the game.
- Keep action names explicit and verb-oriented.
- If some actions are state-invalid (for example kick without ball), mask consistently in:
  - sampling
  - logprob/entropy evaluation
  - evaluation/inference

## 6) Reward Framework

### Outcome scale philosophy
- Keep cross-game outcome magnitudes comparable:
  - win/score about `+10`
  - lose/concede about `-5`
- Dense shaping should support, not dominate, outcomes.

### Reward categories
1. Outcome
2. Event
3. Shaping

### Shaping discipline
- Prefer signed potential-difference or clipped delta shaping.
- Keep shaping interpretable and bounded.

### Logging discipline
- Log realized reward contributions, not reward parameter constants.
- Keep component naming compact and consistent.

## 7) Curriculum Framework

### Defaults
- Prefer a 3-level curriculum when applicable.
- Keep per-level knobs in `LEVEL_SETTINGS`.

### Promotion rule of thumb
- When dense shaping exists, promote using success metrics rather than raw reward.

### Typical promotion settings
- `min_episodes_per_level`
- `check_window`
- `success_threshold`
- `consecutive_checks_required`

## 8) Checklist for Environment Changes

Before merging environment changes:
- [ ] Config section ordering/ownership remains clean.
- [ ] Game `config.py` files remain constants-only with no function definitions.
- [ ] Observation ordering and symmetry rules are preserved.
- [ ] Reward magnitudes and shaping discipline remain balanced.
- [ ] Reward breakdown logs realized contributions only.
- [ ] Action masking is consistent across train/eval when used.
- [ ] Render/training frame pacing follows runtime contract.
- [ ] Render performance changes do not alter environment semantics.
- [ ] Game README snapshot is updated for any current behavior/value change.
