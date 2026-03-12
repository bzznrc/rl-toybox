# Repository Architecture

This document defines shared repo and codebase architecture:
- folder structure and shared module responsibilities
- logging and artifact conventions
- algorithm family placement
- Kick CTDE/MARL architecture notes

Cross-game RL/environment design rules are in [rl-design-guide.md](./rl-design-guide.md).
Current game snapshots live only in `games/<game>/README.md`.

## Table of Contents
- [1) Repo-Wide Goals](#1-repo-wide-goals)
- [2) Repository Layout and Shared Responsibilities](#2-repository-layout-and-shared-responsibilities)
- [3) Logging Framework](#3-logging-framework)
- [4) Model Saving and Run Naming](#4-model-saving-and-run-naming)
- [5) Algorithm Families](#5-algorithm-families)
- [6) Kick MARL Notes (MAPPO/CTDE)](#6-kick-marl-notes-mappoctde)
- [7) Repo-Level Checklist for New Changes](#7-repo-level-checklist-for-new-changes)

## 1) Repo-Wide Goals

### Consistency goals
- Keep shared scaffolding and runner behavior consistent across games.
- Keep interfaces stable so game internals can evolve without breaking scripts/trainers.
- Keep logging and artifact naming consistent so runs are comparable.

### Design goals
- Keep games arranged as a complexity ladder.
- Allow different algorithms per game when appropriate.
- Keep shared integration points strict and explicit.

## 2) Repository Layout and Shared Responsibilities

### Folder layout
- `games/<game_name>/`
  - `env.py`: environment logic
  - `config.py`: game knobs/constants only; do not put `def` helpers here
  - `spec.py`: game-specific `GameSpec` assembly and defaults
  - `README.md`: current game snapshot and run usage
- `core/`
  - game catalog/spec builders/run preparation helpers (`core/game.py`)
  - runtime/window/frame helpers
  - runner loops (`on_policy`, `off_policy`, `eval`)
  - curriculum helpers
  - logging and IO helpers

### Shared interface expectations
- `env.step(...)` returns scalar float reward.
- Extra outputs (for example per-agent reward vectors) are exposed through `info[...]`.
- Cross-game RL/environment design standards are defined in [rl-design-guide.md](./rl-design-guide.md).

### Metadata ownership
- `games/<game>/spec.py` should contain only game-specific metadata assembly.
- `games/<game>/config.py` should stay declarative and constants-only; derived behavior belongs in precomputed constants or in the owning runtime/spec module.
- Shared `GameSpec` types, registry lookup, shared train/play prep, and shared training-config builders live in `core/game.py`.
- Do not place cross-game helper modules under `games/`; `games/` is reserved for per-game code.

## 3) Logging Framework

### Shared cadence
- Train-progress logs are throttled centrally in `core/logging_utils.py`.
- Cadence is fixed at `0.5` seconds.

### Training header
- At run start, print one descriptor line:
  - `Train   Game:<g>  Algo:<a>  Run:<path>  Level:<k>  Resume:<...>  Render:<on/off>`

### Episode line
- Keep episode fields stable and tab-spaced:
  - `Ep:<n>  Lv:<k>  Len:<m>  R:<r>  AR:<avgR>  BR:<bestR>  E:<eps|n/a>  S:<0/1>  AS:<avgS>  <components>`

### Save lines
- Save logs are prefixed:
  - `>>> Save: Best ...`
  - `>>> Save: Check ...`

### On-policy metrics line
- For PPO/MAPPO-style training, emit a second line with optimizer metrics:
  - `> PPO\tPolicyLoss:<...>\tValueLoss:<...>\tEntropy:<...>\tApproxKl:<...>\tClipFrac:<...>`

## 4) Model Saving and Run Naming

- Save artifacts under `runs/<game>/...`.
- Filenames include algo + hidden sizes + curriculum level:
  - `..._L<k>_best.pth`
  - `..._L<k>_check.pth`
- Training at a given level reads/writes that level's artifacts.

## 5) Algorithm Families

### Off-policy
- Used by Snake, Vroom, Bang.
- Replay-buffer based.
- Exploration scheduling is shared and runner-driven.

### On-policy
- Used by Walk and Kick.
- Walk uses single-agent PPO with continuous `Box` actions.
- Kick uses MAPPO-style PPO with shared actor and centralized critic during training.

## 6) Kick MARL Notes (MAPPO/CTDE)

### Shared policy, per-player learning signal
- One shared actor network.
- Per-player rewards/advantages for cleaner credit assignment.

### Centralized critic
- Critic conditions on centralized context plus agent context.

### Action masking
- Invalid kick actions without ball are masked.
- Masking is applied consistently in training and inference.

### Ball-in-flight accounting
- Track physical owner separately from effective possession.
- Keep progress/turnover accounting robust across in-flight transitions.

## 7) Repo-Level Checklist for New Changes

Before merging:
- [ ] Shared architecture assumptions in this document still hold.
- [ ] Cross-game environment changes conform to [rl-design-guide.md](./rl-design-guide.md).
- [ ] Any game behavior change is reflected in that game's README snapshot.
- [ ] Logging format remains aligned with section 3.
- [ ] Artifact naming and run storage remain aligned with section 4.
