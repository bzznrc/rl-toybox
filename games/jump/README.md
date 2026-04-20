# Jump

Compact side-view micro-platformer built around short procedural runs, timing windows, and simple left/right/jump control. `jump` is the repo's PPO-first single-agent actor-critic game and a small traversal-focused counterpart to the other arcade environments.

## Clip

No clip is currently checked into the repo for `jump`.

## Algorithm / Network

- Default algorithm: `ppo`
- IO: `obs=24`, `act=4`
- Default actor: `24 -> 32 -> 32 -> 4`
- Default critic: `24 -> 32 -> 32 -> 1`

## Controls (Human)

- Move left: `A` or left arrow
- Move right: `D` or right arrow
- Jump: `W`, up arrow, or `Space`
- Stop horizontal movement: release left/right
- Jump keeps the current horizontal velocity, matching the RL action contract

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> SENS -> LAND -> OPP -> FLAG`
- Observation: `24` floats (`INPUT_FEATURE_NAMES`, exact order)

```python
[
    # SELF
    "self_vx_norm",
    "self_vy_norm",
    "self_grounded",
    # SENS
    "sens_floor_f1_norm",
    "sens_floor_f2_norm",
    "sens_floor_b1_norm",
    "sens_floor_b2_norm",
    "sens_arc_f1_norm",
    "sens_arc_f2_norm",
    "sens_up_clear_norm",
    "sens_down_ground_norm",
    # LAND
    "land_next_dx",
    "land_next_dy",
    "land_next2_dx",
    "land_next2_dy",
    # OPP
    "opp1_dx",
    "opp1_dy",
    "opp1_vx_norm",
    "opp2_dx",
    "opp2_dy",
    "opp2_vx_norm",
    # FLAG
    "flag_goal_dx",
    "flag_goal_dy",
    "flag_progress_norm",
]
```
- Actions: `Discrete(4)` (`ACTION_NAMES`, ordered)
  - `0 move_left`
  - `1 move_right`
  - `2 jump`
  - `3 move_stop`

- `opp*_*` slots are zero-padded when fewer than two enemies are active.
- Enemy slots are sorted by a stable relevance order that favors same-platform threats first, then smaller local distance, then spawn order.
- `sens_floor_*` probes are local standable-floor checks at near (`f1`) and mid (`f2`) offsets ahead/behind.
- `sens_arc_*` probes are short and mid forward jump / landing viability checks.
- `land_next_*` and `land_next2_*` are egocentric anchor deltas for the next two route platforms ahead, which improves forward lookahead on Level 2+ without expanding the state.
- `flag_goal_dx` and `flag_goal_dy` are egocentric signed goal deltas.

## Environment Notes

- Each reset builds one short deterministic side-scrolling level from procedural platform segments.
- The terrain uses exactly three equally spaced lanes:
  - baseline
  - one raised lane
  - one top lane
- Each curriculum level activates only the first `lane_count` lanes, so Level 1 stays baseline-only before the higher lanes appear later.
- Traversal platforms use exactly three standard widths:
  - short: `6` tiles
  - medium: `9` tiles
  - large: `12` tiles
- Platforms never stack vertically. If a raised platform occupies an `x` range, there is no baseline or middle platform directly underneath it.
- The route starts on the baseline, can move up through the higher lanes, and always returns to a flat baseline goal stretch at the end.
- Every inter-platform transition is a true jump. Gaps are never smaller than the player width, and generation validates each jump against the same movement envelope used by the player controller.
- Hazards are only gaps and simple walker enemies.
- Enemies are one type only in v1:
  - a Bang-sized single-tile walker that patrols the full usable length of one platform, reverses at the edges, and only kills on side contact.
- The player is a Bang-sized single-tile block with persistent horizontal velocity.
- Top-face contact on an enemy defeats it; side contact still fails the run.
- `jump` has immediate response and a small coyote window so late-edge jumps stay readable without adding extra actions.
- Training uses the shared discrete action-mask path to disable `jump` while airborne, which keeps PPO exploration focused on meaningful choices.
- Platforms render with the same neutral gray pair used for Bang obstacles.
- Episode timers are intentionally more generous so traversal and enemy timing are not overly rushed.

An episode counts as a success when the player reaches the flag.

## Rewards (Training)

- `REWARD_FINISH = +10.0` on flag reach
- `PENALTY_FAIL = -5.0` on enemy collision, falling into a gap, or timeout
- `combat.reward_stomp = +1.00` per stomp, capped at `+5.00` in a single step
- Forward shaping rewards only new furthest progress toward the flag, capped at `+0.10` per step, with `FORWARD_PROGRESS_SCALE = 2.5`
- Backtracking away from the best progress reached so far applies a small shaping penalty, clipped at `-0.02`
- `PENALTY_STEP = 0.0` every training step

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/jump/config.py` under `CURRICULUM_PROMOTION`
- Per-level curriculum settings stay compact:
  - `length_tiles`
  - `lane_count`
  - `enemy_frequency`
- Episode time budget is derived automatically from `length_tiles`.
- Levels:
  - Level 1: shorter baseline-only levels, gaps `2-3` tiles wide, and no enemies
  - Level 2: longer levels, all three lanes active, gaps `2-4` tiles wide, and `2-3` walkers
  - Level 3: longest short-form levels, denser lane changes, gaps `3-5` tiles wide, and `3-4` walkers

## Run Commands

```bash
rl-toybox-train --game jump
rl-toybox-play-ai --game jump --render
rl-toybox-play-user --game jump
python -m scripts.train --game jump
python -m scripts.play_ai --game jump --render
python -m scripts.play_user --game jump
```

See `games/jump/config.py` for the observation contract, level-generation knobs, rewards, and training defaults. Jump's game-wide net size lives in `DEFAULT_MODEL_CONFIG["hidden_sizes"]`, its PPO-specific extras live in `ALGO_CONFIG_OVERRIDES["ppo"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
