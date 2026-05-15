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
- Standardize curriculum-based games on a shared 5-level ladder, with `flip` as the fixed-mode board game.
- Keep shared runtime and window defaults centralized in `core/shared_config.py`.
- Keep a coherent visual system built from the shared square-block language and shared palette.

## 2) Active Lineup

| Game | Role | Primary Family | Status |
| --- | --- | --- | --- |
| `snake` | Intro grid-control game | value-based | active |
| `bang` | Flagship discrete-control arena game with Duel / Arena / Team Arena modes | value-based | active |
| `jump` | Traversal platformer | actor-critic | active |
| `vroom` | Continuous-control racing game | actor-critic | active |
| `flip` | Planning + self-play capstone | search + self-play | active |
| `kick` | Scalable 3v3 / 5v5 / 7v7 football with one kick action | actor-critic / CTDE | active |

## 3) Repository Layout and Shared Responsibilities

### Folder layout

- `games/<game_name>/`
- `env.py`: game-specific environment logic
- `config.py`: declarative game knobs, including game-owned model defaults, algo-specific overrides, train defaults, and any non-default spec metadata
- `README.md`: current implementation snapshot
- `core/shared_config.py`
- shared runtime/window defaults such as FPS, geometry, tile sizing, and common marker sizing
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
- compact MCTS, self-play training, and policy/value helpers for `flip`
- `core/io/`, `core/runners/`, and `core/logging_utils.py`
- shared run IO, training loops, and logging behavior
- `core/runners/env_access.py`
- shared runner accessors for action masks, centralized state, reward storage, policy-state reset, and curriculum success stats
- `core/envs/arcade.py`
- opt-in Arcade mixin for common window setup, frame pacing, and close behavior
- `core/ghost_overlay.py` and `core/ray_viz.py`
- shared translucent observation/debug overlay primitives, including the standard `X` toggle path and ray drawing

### Ownership boundaries

- Reusable RL code belongs under the appropriate `core/<family>/` area.
- Keep `core/algorithms/` minimal and family-agnostic: shared interfaces, builders, and cross-family helpers only.
- Shared runtime/window constants belong in `core/shared_config.py`.
- Game-specific env logic stays under `games/<game>/`.
- Game configs should stay declarative and focused on `ENV`, `IO`, `CURRICULUM`, `REWARDS`, and `TRAINING`, keeping local overrides only when a game genuinely differs from the shared defaults.
- Avoid introducing large new abstractions until a second pass actually needs them.

### Shared visual language

- The default visual building block is the standard `DEFAULT_TILE_SIZE` square from `core/arcade_style.py`.
- Bang and most other games use that `1x` block directly; Vroom's car body also uses that same base block language.
- Allowed scale variants should stay simple and explicit:
  - `2x` for larger composite cells or overlays, such as Jump's grid-style patch visuals.
  - `0.5x` for small gameplay markers and compact UI accents.
- New visuals should prefer these square components plus the shared palette in `core/arcade_style.py` instead of introducing separate visual systems.
- Observation ghosts should use light-neutral 50% alpha overlays and the shared `X` toggle, so rays, SENS probes, and role/area guides behave consistently across games.

## 4) Logging Framework

- Episode progress logs are throttled centrally in `core/logging_utils.py`.
- Training emits a shared run-context header and stable single-line progress logs.
- Bang's compact episode line is the baseline format: short labels, padded values, tab-separated fields, and reward components at the end.
- Per-step / per-episode labels should stay compact. Prefer abbreviations such as `Len`, `Avg. Len`, `Policy L.`, and `Value L.` over long labels or collated words like `PolicyLoss`.
- Primary progress fields may pad after the colon to keep columns stable, for example `Game:  1879`, `Len:    13`, and `Win:    P1`.
- Letter-and-number indicators and categorical words are normalized in logs, for example `P1`, `P2`, and `Draw`.
- Path-like values are preserved as paths; other categorical values should avoid lowercase words such as `draw`, `scratch`, `on`, and `off`.
- Multi-word labels keep spaces for readability, for example `P1 Win` instead of `P1Win`.
- Reward components are appended at the end of the line as one- or two-letter codes separated by spaces, for example `W:10 D:0 L:0`.
- Periodic or occasional events use the `>>>` prefix, for example `>>> Save:`, `>>> Arena:`, `>>> Explore:`, and `>>> Warn:`.
- `Ep:` lines show environment performance: episode length, reward, rolling averages, success, and optional reward components.
- `Up:` lines show optimizer health for PPO / coach-critic methods and appear once per optimizer update, not once per episode.
- PPO-style `Up:` fields are `Up`, `Lv`, `Steps`, `Pi`, `V`, `EV`, `Ent`, and `KL`.
- SAC-style `Up:` fields are `Up`, `Lv`, `Steps`, `Pi`, `Q`, `Ent`, and `Alpha`, but SAC update logging is opt-in and Vroom keeps it disabled by default.
- `Pi` is actor / policy loss, `V` is value loss, `Q` is critic / Q loss, `Ent` is policy entropy, and `KL` is approximate KL when available.
- `EV` is critic explained variance, computed as `1 - Var(returns - values) / Var(returns)`: near `1.0` is excellent, around `0.0` means weak or no baseline improvement, and negative means worse than predicting the mean.
- Artifact output remains under `runs/<game>/...`.

## 5) Model Saving and Run Naming

- Save artifacts under `runs/<game>/`.
- Filenames keep the existing `<algo>_<net>_L<level>_<kind>.pth` convention.
- Trained checkpoints and run metrics are ignored by git; each run subfolder is kept with a zero-byte `.gitkeep`.
- Existing kept games preserve their run tags where practical so older runs stay discoverable. Shared-policy mode games such as Kick and Bang keep the model tag mode-neutral so the same checkpoint can be used across modes.
- Run tags should stay compact and reflect the active model shape.

## 6) RL Family Layout

### `core/value_discrete/`

- Used by: `snake`, `bang`
- Contains: linear-Q / q-learning, DQN family modules, replay, and value nets

### `core/actor_critic/`

- Used by: `jump`, `vroom`, `kick`
- Contains: PPO, A2C-style unclipped actor-critic updates, recurrent PPO support, SAC, shared actor-critic rollout machinery, and centralized-critic support

### `core/search_play/`

- Used by: `flip`
- Contains: compact MCTS, search-play training, and small policy/value helpers

### Default algorithm matrix

| Game | Default | Supported alternates |
| --- | --- | --- |
| `snake` | `qlearn` | `dqn`, `ppo`, `a2c` |
| `bang` | `dqn` | `qlearn`, `ppo`, `a2c` |
| `jump` | `ppo` | `a2c`, `dqn` |
| `vroom` | `sac` | `ppo`, `a2c` |
| `flip` | `search_play` | none; requires self-play/search support |
| `kick` | `ppo` | `mappo`, `a2c`; requires centralized critic and multi-agent support |

## 7) Special Areas

- `bang` stays in-tree as one value-based shooter game id, with `duel`, `arena`, and `team_arena` combat modes sharing the same DQN IO shape. `team_arena` controls two friendly agents with one shared DQN policy. Mode defines the maximum format; `LEVEL_SETTINGS[level]["active_enemies"]` owns the per-mode active enemy ramp inside that format.
- `kick` stays in-tree and runnable as the repo's single CTDE football game, with `3v3`, `5v5`, and `7v7` team-size modes.
- `jump` sits on the shared actor-critic path as the repo's compact single-agent traversal showcase.
- Its centralized-critic support lives on the shared actor-critic path plus game-provided central observation metadata.
- `vroom` sits on the actor-critic and continuous-control branch with SAC-oriented defaults.

## 8) Checklist for New Changes

Before merging:

- [ ] Active lineup references still use the canonical order from section 2.
- [ ] Shared RL code is placed under the right `core/<family>/` area.
- [ ] Curriculum-based games still follow the shared `L1` to `L5` convention, and fixed-mode board games clearly document their fixed `L1` slot.
- [ ] Game-specific behavior changes are reflected in that game's README.
