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
- Keep render FPS and training FPS separate.
- `TRAINING_FPS=0` means max-throughput training.

### Terminal commit

- Success/failure should come from explicit terminal conditions.
- Render-only terminal holds should not change already-latched outcomes.

## 2) Render and Performance Patterns

- Reuse static visual assets when possible.
- Keep draw-call pressure intentional.
- Avoid expensive per-frame text churn in hot loops.
- Gate expensive debug-only work behind explicit flags.
- Performance changes must not alter observation, reward, or termination semantics.

## 3) Config Contract

### Section order

Each game config should prefer this order:

1. `RUNTIME`
2. `ENV`
3. `IO`
4. `CURRICULUM`
5. `REWARDS`
6. `TRAINING`

### Ownership boundaries

- Keep `config.py` declarative and constants-only.
- Put shared training/runtime boilerplate in shared `core/` code.
- Keep future placeholder configs lightweight rather than speculative.

## 4) Observation Taxonomy

### Feature block ordering

Use this order when applicable:

1. `SELF`
2. `RAYS`
3. `TGT`
4. `GOALS/LANDMARKS`
5. `TEAMMATES`
6. `OPPONENTS`
7. `TRACK/MAP`
8. `HAZARDS`

### Naming rules

- `*_sin` pairs with `*_cos`
- `dx` pairs with `dy`
- `dvx` pairs with `dvy`
- Boolean features should be numeric `0.0/1.0`

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

## 7) Curriculum Framework

- Prefer a compact 3-level curriculum when applicable.
- Promote using success metrics when dense shaping could distort reward totals.
- Keep per-level knobs in clearly named config tables.

## 8) Game README Contract

### Canonical game order

When docs enumerate the active lineup, use:

1. `snake`
2. `bang`
3. `tower`
4. `vroom`
5. `frogger`
6. `cardz`
7. `osero`
8. `kick`

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

## 9) Checklist for Environment Changes

Before merging environment changes:

- [ ] Config ownership remains clean.
- [ ] Observation ordering and naming stay intentional.
- [ ] Reward magnitudes remain balanced and interpretable.
- [ ] Action masking is consistent when used.
- [ ] README and docs still describe the implemented game accurately.
- [ ] `python -m scripts.validate_docs` passes after README/doc edits.
