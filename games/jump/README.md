# Jump

Compact side-view micro-platformer built around short procedural runs, timing windows, and simple left/right/jump control. `jump` is the repo's PPO-first single-agent actor-critic game and a small traversal-focused counterpart to the other arcade environments.

## Clip

![Jump Demo](../../media/jump-demo.gif)

## Algorithm / Network

- Default algorithm: `ppo`
- IO: `obs=36`, `act=4`
- Default actor: `36 -> 32 -> 32 -> 4`
- Default critic: `36 -> 32 -> 32 -> 1`

## Controls (Human)

- Move left: `A` or left arrow
- Move right: `D` or right arrow
- Jump: `W`, up arrow, or `Space`
- Rendered overlays: `X` toggles the translucent route/SENS ghost overlay during `play-user` and `play-ai`
- Stop horizontal movement: release left/right
- Jump keeps the current horizontal velocity, matching the RL action contract

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> SENS -> LAND -> OPP -> HAZ -> FLAG`
- Observation: `36` floats (`INPUT_FEATURE_NAMES`, exact order)

```python
[
    # SELF
    "self_vx_norm",
    "self_vy_norm",
    "self_grounded",
    "self_lane_norm",
    # SENS
    "sens_ground_l2",
    "sens_ground_l1",
    "sens_ground_c0",
    "sens_ground_r1",
    "sens_ground_r2",
    "sens_gap_f1",
    "sens_gap_f2",
    "sens_gap_f3",
    # LAND
    "land_next_dx",
    "land_next_dy",
    "land_next_width",
    "land_next_lane_delta",
    "land_gap_dx",
    "land_gap_width",
    "land_move_dx",
    "land_move_dy",
    "land_move_vx_norm",
    "land_move_phase",
    # OPP
    "opp1_dx",
    "opp1_dy",
    "opp1_vx_norm",
    "opp1_tti",
    "opp2_dx",
    "opp2_dy",
    # HAZ
    "haz_route_dx",
    "haz_route_tti",
    "haz_lane_dx",
    "haz_lane_tti",
    # FLAG
    "flag_goal_dx",
    "flag_goal_dy",
    "flag_progress_norm",
    "flag_time_left",
]
```

- Actions: `Discrete(4)` (`ACTION_NAMES`, ordered)
  - `0 move_left`
  - `1 move_right`
  - `2 jump`
  - `3 move_stop`

- `SENS` is a compact symmetric route sensor: five local ground slots from left to right plus three forward gap probes. The `X` ghost overlay mirrors these current probes rather than the old patch grid.
- `LAND` tracks the next route platform, current gap, and nearest relevant moving platform including its travel phase.
- `OPP` tracks the two closest relevant enemies; the first enemy also includes velocity and time-to-impact.
- `HAZ` separates route-level and same-lane forward enemy threat.
- `flag_goal_dx` and `flag_goal_dy` are egocentric signed goal deltas.

## Environment Notes

- Each reset builds one short deterministic side-scrolling level from procedural platform segments.
- The terrain uses exactly three equally spaced lanes:
  - baseline
  - one raised lane
  - one top lane
- Each curriculum level activates only the first `lane_count` lanes, so Level 1 stays baseline-only before the higher lanes appear later.
- When a level is single-lane with both hazard frequencies at `0.0`, generation collapses the gaps so the early tutorial path becomes one contiguous flat run.
- Traversal platforms use exactly three standard widths:
  - short: `6` tiles
  - medium: `9` tiles
  - large: `12` tiles
- Moving platforms use the short/medium widths only:
  - short: `6` tiles
  - medium: `9` tiles
- Platforms never stack vertically. If a raised platform occupies an `x` range, there is no baseline or middle platform directly underneath it.
- The route starts on the baseline, can move up through the higher lanes, and always returns to a flat baseline goal stretch at the end.
- Every inter-platform transition is validated against the same movement envelope used by the player controller.
- Some gaps are widened into moving-platform transitions. Those gaps are intentionally too wide for a direct crossing, but a horizontally moving support platform exposes a normal reachable jump at each end of its travel.
- Moving platforms are simple in v1:
  - horizontal only
  - fixed-speed
  - deterministic per seed
  - solid for landing, standing, and jumping
- Hazards are gaps and simple walker enemies.
- Enemies are one type only in v1:
  - a Bang-sized single-tile walker that patrols the full usable length of one platform, reverses at the edges, and only kills on side contact
- The player is a Bang-sized single-tile block with persistent horizontal velocity.
- Top-face contact on an enemy defeats it; side contact still fails the run.
- Training uses the shared discrete action-mask path to disable `jump` while airborne, which keeps PPO exploration focused on meaningful choices.
- Platforms render with the same compact two-tone block language used elsewhere in the repo.
- Episode timers are intentionally more generous so traversal, enemy timing, and moving-platform timing are not overly rushed.

An episode counts as a success when the player reaches the flag.

## Rewards (Training)

- `REWARD_FINISH = +10.0` on flag reach
- `PENALTY_FAIL = -5.0` on enemy collision, falling into a gap, or timeout
- `combat.reward_stomp = +1.00` per stomp, capped at `+5.00` in a single step
- Progress shaping rewards only new furthest progress and penalizes backward steps, capped at `+/-0.10` per step, with `PROGRESS_SCALE = 2.5`
- `PENALTY_STALL = -0.005` only when progress is effectively flat for the step
- There is no generic step penalty; the pressure comes from new-progress reward, moving backward, standing still, timeout failure, and the finish bonus

## Curriculum (Train)

- Shared 5-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/jump/config.py` under `CURRICULUM_PROMOTION`
- Per-level curriculum settings stay compact:
  - `length_tiles`
  - `lane_count`
  - `enemy_frequency`
  - `moving_platform_frequency`
- Episode time budget is derived automatically from `length_tiles`.
- The ramp starts with a flat single-lane tutorial, then reintroduces regular gaps, enemies, extra lanes, and moving platforms before the unchanged `L5`.
- Levels:
  - Level 1: `length_tiles=48`, `lane_count=1`, `enemy_frequency=0.0`, `moving_platform_frequency=0.0`
  - Level 2: `length_tiles=64`, `lane_count=1`, `enemy_frequency=0.25`, `moving_platform_frequency=0.0`
  - Level 3: `length_tiles=80`, `lane_count=2`, `enemy_frequency=0.25`, `moving_platform_frequency=0.0`
  - Level 4: `length_tiles=104`, `lane_count=2`, `enemy_frequency=0.50`, `moving_platform_frequency=0.35`
  - Level 5: `length_tiles=128`, `lane_count=3`, `enemy_frequency=0.75`, `moving_platform_frequency=0.75`

## Run Commands

```bash
rl-toybox-train --game jump
rl-toybox-play-ai --game jump --render
rl-toybox-play-user --game jump
python -m scripts.train --game jump
python -m scripts.play_ai --game jump --render
python -m scripts.play_user --game jump
```

When `--level` is omitted, `train` starts at `L1` and `play-user` / `play-ai` default to `L5`.

See `games/jump/config.py` for the observation contract, level-generation knobs, rewards, and training defaults. Jump's game-wide net size lives in `DEFAULT_MODEL_CONFIG["hidden_sizes"]`, its PPO-specific extras live in `ALGO_CONFIG_OVERRIDES["ppo"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
