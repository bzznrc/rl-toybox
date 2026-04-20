# Repository Architecture

This document defines the current repo structure, the active game ladder, and the shared RL family layout.

Cross-game RL and environment design rules are in [rl-design-guide.md](./rl-design-guide.md).
Per-game snapshots live in `games/<game>/README.md`.

## Table of Contents

- [1) Repo-Wide Goals](#1-repo-wide-goals)
- [2) Active Lineup](#2-active-lineup)
- [3) Repository Layout and Shared Responsibilities](#3-repository-layout-and-shared-responsibilities)
- [4) Logging Framework](#4-logging-framework)
- [5) Model Saving and Run Naming](#5-model-saving-and-run-naming)
- [6) RL Family Layout](#6-rl-family-layout)
- [7) Special Areas](#7-special-areas)
- [8) Checklist for New Changes](#8-checklist-for-new-changes)

## 1) Repo-Wide Goals

- Keep the lineup small, polished, and easy to understand.
- Separate reusable RL systems from game-specific environment logic.
- Present a coherent progression across RL families and tradeoffs.
- Prefer compact, complete games over oversized systems.

## 2) Active Lineup

| Game | Role | Primary Family | Status |
| --- | --- | --- | --- |
| `snake` | Intro game | simple value-based | active |
| `bang` | Flagship discrete RL game | value-based / Rainbow-lite | active |
| `jump` | Reactive traversal platformer | actor-critic / PPO | active |
| `vroom` | Continuous control showcase | actor-critic / SAC | active |
| `osero` | Planning + self-play capstone | search + self-play | active |
| `kick` | Multi-agent CTDE experiment | actor-critic / PPO + CTDE | active |

## 3) Repository Layout and Shared Responsibilities

### Folder layout

- `games/<game_name>/`
- `env.py`: game-specific environment logic
- `config.py`: declarative game knobs, including game-owned model defaults, algo-specific overrides, train defaults, and any non-default spec metadata
- `README.md`: current implementation snapshot
- `core/game.py`
- active lineup registry
- shared metadata/spec building used by CLI entrypoints
- run-name builders and train-config builders
- `core/algorithms/`
- thin shared interface/factory layer plus exploration scheduling helpers
- `core/value_discrete/`
- shared tabular/linear-Q and DQN-family infrastructure
- home for value-based helpers such as Double DQN, PER, dueling heads, and action masking support
- `core/actor_critic/`
- shared PPO/A2C/recurrent PPO/SAC infrastructure
- target home for shared rollout, policy, critic, centralized-critic, and continuous-control helpers
- `core/search_play/`
- compact MCTS, self-play training, and policy/value helpers for `osero`
- `core/io/`, `core/runners/`, and `core/logging_utils.py`
- shared run IO, training loops, and logging behavior

### Ownership boundaries

- Reusable RL code belongs under the appropriate `core/<family>/` area.
- Keep `core/algorithms/` minimal and family-agnostic: shared interfaces, builders, and cross-family helpers only.
- Game-specific env logic stays under `games/<game>/`.
- Avoid introducing large new abstractions until a second pass actually needs them.

## 4) Logging Framework

- Train-progress logs are throttled centrally in `core/logging_utils.py`.
- Training emits a shared run-context header and stable episode/save lines.
- PPO-style runs keep the extra optimizer metrics line.
- Artifact output remains under `runs/<game>/...`.

## 5) Model Saving and Run Naming

- Save artifacts under `runs/<game>/`.
- Filenames keep the existing `<algo>_<net>_L<level>_<kind>.pth` convention.
- Existing kept games preserve their run tags where practical so older runs stay discoverable.
- Run tags should stay compact and reflect the active model shape.

## 6) RL Family Layout

### `core/value_discrete/`

- Used by: `snake`, `bang`
- Contains: linear-Q / q-learning, DQN family modules, replay, and value nets

### `core/actor_critic/`

- Used by: `jump`, `vroom`, `kick`
- Contains: PPO, recurrent PPO support, SAC, shared actor-critic rollout machinery, and centralized-critic support

### `core/search_play/`

- Used by: `osero`
- Contains: compact MCTS, search-play training, and small policy/value helpers

## 7) Special Areas

- `kick` stays in-tree and runnable as the repo's CTDE and centralized-critic game.
- `jump` sits on the shared PPO path as the repo's compact single-agent traversal showcase.
- Its centralized-critic support lives on the shared PPO path plus game-provided central observation metadata.
- `vroom` sits on the actor-critic and continuous-control branch with SAC-oriented defaults.

## 8) Checklist for New Changes

Before merging:

- [ ] Active lineup references still use the canonical order from section 2.
- [ ] Shared RL code is placed under the right `core/<family>/` area.
- [ ] Game-specific behavior changes are reflected in that game's README.
- [ ] `python -m scripts.validate_docs` passes after README/doc edits.
