# RL-Toybox Framework (Repo Contract)

This document is the **source of truth** for how environments, agents, configs, rewards, curriculum, logging, and saved runs should look across this repo.

If you’re making changes in one game, use this to keep the whole repo consistent.

---

## Table of contents
- [1) Goals](#1-goals)
- [2) Repo conventions](#2-repo-conventions)
- [3) Config file contract](#3-config-file-contract)
- [4) Observation taxonomy](#4-observation-taxonomy)
- [5) Action space conventions](#5-action-space-conventions)
- [6) Reward framework](#6-reward-framework)
- [7) Curriculum framework](#7-curriculum-framework)
- [8) Logging framework](#8-logging-framework)
- [9) Model saving / run naming](#9-model-saving--run-naming)
- [10) Algorithm families](#10-algorithm-families)
- [11) Kick MARL notes (MAPPO/CTDE)](#11-kick-marl-notes-mappoctde)
- [12) Per-game spec snapshots](#12-pergame-spec-snapshots)
- [13) Checklist for new changes](#13-checklist-for-new-changes)

---

## 1) Goals

### Consistency goals
- Same **scaffolding** in every game’s config.
- Same **input feature taxonomy** across games (block order + naming).
- Same **reward breakdown style** and logging format.
- Same **3-level curriculum** pattern across games (when applicable).

### Design goals
- Games form a **complexity ladder**: simpler games use fewer inputs/outputs and smaller nets.
- Prefer **even OBS_DIM and ACT_DIM** where it doesn’t harm learnability.
- Project is a **demo repo**: different algorithms per game are fine, but interfaces must stay aligned.

---

## 2) Repo conventions

### Folder layout
- `games/<game_name>/` contains:
  - `env.py` (environment logic)
  - `config.py` (knobs only)
  - `README.md` (how to run + current IO/reward/net snapshot)
- `core/` contains shared utilities:
  - logging + formatting
  - curriculum promotion logic
  - save paths / run naming
  - exploration schedules (epsilon + bump)
  - shared RL runner loops (off-policy / on-policy)

### Cross-game contracts
- `env.step(...)` **returns scalar float reward**.
  - If a trainer needs more (e.g., per-agent reward vector), return it via `info[...]`.
- Observation vector ordering must follow the taxonomy in this doc.
- Reward component logging must use the repo’s compact breakdown keys.

---

## 3) Config file contract

Each game config follows the same section order and contains only meaningful knobs.

### Required sections (in this order)
1. **RUNTIME**
   - `WINDOW_TITLE`, `FPS`, `TRAINING_FPS`, `USE_GPU`
2. **ENV**
   - arena sizes, physics, constants, episode limits
3. **IO**
   - `INPUT_FEATURE_NAMES`, `ACTION_NAMES`, `OBS_DIM`, `ACT_DIM`
4. **CURRICULUM**
   - `MIN_LEVEL`, `MAX_LEVEL`
   - `CURRICULUM_PROMOTION` (see section 7)
   - `LEVEL_SETTINGS` (all per-level env knobs live here)
5. **REWARDS**
   - outcome magnitudes (+10 / -5 scale)
   - event/shaping weights
   - `REWARD_COMPONENTS` (for logging taxonomy; log realized contributions, not parameters)
6. **TRAINING**
   - model sizes (e.g. `HIDDEN_DIMENSIONS`)
   - algorithm hyperparameters (replay/rollout, lr, gamma, etc.)

### What **should NOT** live in game configs
- Derived model path boilerplate (run folder naming, filenames, retries, etc.)
  - This is centralized in core.
- Duplicate definitions of the same knob in two places (one source of truth).

---

## 4) Observation taxonomy

### 4.1 Block order (global rule)
Feature vectors must follow this block order (some blocks omitted per game):

1) **SELF** — `self_*`  
2) **RAYS** — `ray_*` (if used)  
3) **TGT** — `tgt_*` (primary target: enemy/ball/opponent)  
4) **GOALS / LANDMARKS** — `goal_*`, `own_goal_*` (Kick)  
5) **ALLIES** — `ally1_*`, `ally2_*`, ...  
6) **FOES** — `foe1_*`, `foe2_*`, ...  
7) **TRACK / MAP** — `trk_*` (Vroom)  
8) **HAZARDS** — `haz_*` (Bang)

### 4.2 Symmetry rules (do not break)
- If you add an `*_sin`, you add the matching `*_cos`.
- If you add a `dx`, you add the matching `dy`.
- If you add a `dvx`, you add the matching `dvy`.

### 4.3 Normalization rules
- Positions are relative (`dx`, `dy`) and normalized to a consistent scale.
- Angles should usually be **sin/cos pairs**.
  - Exception: `walk` intentionally uses compact normalized scalar angles to keep `OBS_DIM=18`.
- Booleans are floats `0.0/1.0`.
- Time/cooldowns are normalized to `[0,1]` where possible.

### 4.4 Stable ordering for “nearest N”
When choosing nearest allies/foes:
- select by distance
- sort deterministically by `(distance, stable_slot_index)`  
Never use Python object `id()` for tie-breaking.

---

## 5) Action space conventions

- Prefer discrete actions unless continuous control is the point of the game.
- Use clear verb-like names (`move_ne`, `kick_high`, `aim_left`, etc.)
- If some actions are invalid in a state (e.g., kicking without ball):
  - implement **action masking**
  - masking must be applied consistently in:
    - training sampling
    - training logprob/entropy evaluation
    - eval/play_ai inference

---

## 6) Reward framework

### 6.1 Scale rule (global)
Across games:
- **Win/Score ≈ +10**
- **Lose/Concede ≈ -5**
Shaping terms must not dwarf outcome terms.

### 6.2 Reward types
1) **Outcome** (either/or):
   - `win/lose`, `score/concede`
2) **Events**:
   - discrete events (food, kill, pass, turnover)
3) **Shaping**:
   - dense guidance, preferably as **signed ΔΦ** (potential difference) or clipped deltas

### 6.3 Logging vs parameters
- `REWARD_COMPONENTS` is for **taxonomy + logging keys**.
- Do **not** log shaping *parameters* as if they were reward contributions.
  - Example: don’t print `progress.scale: 2.5`
  - Do print realized `P:<accumulated_progress_reward>`

### 6.4 Compact component keys (repo standard)
Outcome / events / shaping should be logged as short letter keys:
- `W` win, `L` lose (Snake/Bang/Vroom where applicable)
- `G` goals scored, `C` conceded (Kick)
- `K` kill (Bang)
- `A` pass (Kick)
- `T` turnover (Kick)
- `P` progress (shaping)
- `Z` zone/formation (Kick)
- `S` step penalty (if used)

Format: `X:<value>` with no tabs inside the component list.

---

## 7) Curriculum framework

### 7.1 3 levels (repo standard)
Where curriculum is used, keep it to **3 levels** (clean + consistent).

### 7.2 Promotion gating (standard)
Promotion should be based on **AvgSuccess** (not AvgReward) when shaping exists.

Recommended structure:
- `min_episodes_per_level`
- `check_window`
- `success_threshold` (or per-level thresholds)
- `consecutive_checks_required`

### 7.3 What belongs in LEVEL_SETTINGS
All per-level env knobs belong in `LEVEL_SETTINGS`, e.g.:
- obstacle counts
- opponents count/strength
- goal size scaling
- speed caps / stamina scaling
- enemy accuracy / shot error choices
- entropy coefficient per level (Kick PPO family)

Avoid per-level knobs that add lots of branching logic unless the benefit is clear.

---

## 8) Logging framework

### 8.0 Shared cadence (global, fixed)
- Train-progress logs are throttled centrally in `core/logging_utils.py`.
- Cadence is fixed at **0.5 seconds** (`TRAIN_PROGRESS_LOG_INTERVAL_SECONDS = 0.5`).
- This cadence is intentionally **not configurable per game or CLI**.

### 8.1 Training header line
At training start, print a single descriptor line:
- `Train   Game:<g>  Algo:<a>  Run:<path>  Level:<k>  Resume:<...>  Render:<on/off>`

### 8.2 Episode line (tab spaced)
Main fields are tab-separated and aligned with fixed-width formatting:
- `Ep:<n>  Lv:<k>  Len:<m>  R:<r>  AR:<avgR>  BR:<bestR>  E:<eps|n/a>  S:<0/1>  AS:<avgS>  <components>`

### 8.3 Save lines
All save logs are prefixed/indented:
- `>>> Save: Best ...`
- `>>> Save: Check ...`

### 8.4 On-policy extra PPO/MAPPO metrics line
After the main episode line (Kick), print a second line:
- `> PPO\tPolicyLoss:<...>\tValueLoss:<...>\tEntropy:<...>\tApproxKl:<...>\tClipFrac:<...>`

---

## 9) Model saving / run naming

- Saved models live under `runs/<game>/...`
- Filenames include:
  - algo + hidden sizes + curriculum level
  - `..._L<k>_best.pth` and `..._L<k>_check.pth`
- When training at a given level:
  - load/resume from that level’s best/checkpoint as requested
  - write back to that level’s files
- No runtime code for migrating old files; do one-time manual move when needed.

---

## 10) Algorithm families

### 10.1 Off-policy (Snake / Vroom / Bang)
- DQN-style with replay buffer
- exploration:
  - multiplicative epsilon decay
  - bump logic (cap + patience + cooldown) shared across games
- update cadence is centralized in the runner (avoid double-gating inside the agent)

### 10.2 On-policy (Walk / Kick)
- Walk uses single-agent PPO with continuous `Box` actions.
- Kick uses MAPPO-style PPO (CTDE):
  - shared actor (decentralized execution)
  - centralized critic (training-only)
  - per-agent rewards/advantages to reduce credit leakage
- Outcome reward scaling should stay stable across environments and team sizes.

---

## 11) Kick MARL notes (MAPPO/CTDE)

### 11.1 Shared policy, per-player learning signal
- One actor network is shared.
- Training uses per-player rewards and advantages (critical for credit assignment).

### 11.2 Centralized critic
Best practice for Kick:
- critic conditioned on (central_state + agent_context) to fit agent-specific returns:
  - central state: padded up to MAX_LEFT_PLAYERS (e.g., 11)
  - agent context: agent obs and/or identity (role + lane)

### 11.3 Action masking
- Kick actions masked when player does not have ball.
- Masking required in training + eval.

### 11.4 Ball “in flight” handling
Define:
- physical owner: `ball_owner_team/id` (None if free)
- effective possession team: `ball_owner_team if not None else last_touch_team`
Use:
- effective possession for formation phase stability (Z)
- progress crediting should avoid kick-and-pray exploits:
  - use signed/clipped delta progress
  - apply turnover penalties on true loss of possession
  - ensure progress doesn’t reset incorrectly on LEFT->free->LEFT pass flight

---

## 12) Per-game spec snapshots

> These are the intended “current targets” (update when the repo changes).

### Snake
- Algo: DQN
- Net: `[32]`
- IO: `OBS_DIM=12`, `ACT_DIM=3`
- Reward scale: food +10, lose -5, small step penalty, progress shaping (signed ΔΦ)

### Vroom
- Algo: DQN
- Net: `[48,48]`
- IO: `OBS_DIM=20`, `ACT_DIM=6`
- Key features: track lookahead + forward rays (near/far + slight left/right)
- Rewards: win/lose, collision, small step penalty, progress shaping (signed ΔΦ)

### Bang
- Algo: Enhanced DQN (double/dueling/PER)
- Net: `[64,64]`
- IO: `OBS_DIM=24`, `ACT_DIM=8`
- Key features: symmetric aim/target angles, rays, hazards, LOS
- Rewards: win/lose (+10/-5), kill +2, shaping terms (engagement/hazard), small step penalty

### Walk
- Algo: PPO (continuous control)
- Net: `[64,64]`
- IO: `OBS_DIM=18`, `ACT_DIM=4` (`Box[-1,1]`)
- Key features: side-view biped, compact scalar-angle SELF block, 4 forward/down terrain rays
- Rewards: `P/L` only (`P = new best-x progress`, `L = fail penalty`)
- Curriculum: 3 levels (flat -> gentle variation -> broader bumps/steps)

### Kick
- Algo: MAPPO-style PPO (CTDE)
- Actor: `[128,128]`, Critic: `[256,256]`
- IO: `OBS_DIM=48`, `ACT_DIM=12` discrete
- Actions: `ACT_DIM=12` discrete
- Key systems: action masking, per-agent rewards, formation/zone shaping (Z), progress (P), pass (A), turnover (T)
- Curriculum: 3 levels (easy → medium → full)

---

## 13) Checklist for new changes

Before merging a change:
- [ ] Config follows scaffolding order (RUNTIME/ENV/IO/CURRICULUM/REWARDS/TRAINING)
- [ ] Observation names follow taxonomy + symmetry rules
- [ ] OBS_DIM and ACT_DIM are even if possible (or justified if not)
- [ ] Reward magnitudes respect +10/-5 scale; shaping doesn’t dominate
- [ ] Reward breakdown logs only realized contributions (not parameters)
- [ ] Action masking applies in training + eval if needed
- [ ] Curriculum settings live in LEVEL_SETTINGS
- [ ] Logs remain tab-aligned + save lines indented
- [ ] Game README updated with IO/reward/net/algo snapshot
