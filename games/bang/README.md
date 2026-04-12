# Bang

Top-down arena shooter focused on movement, aiming, line of sight, and timing shots under pressure. Among the value-based games in the repo, `bang` is the most combat-oriented and the closest to a full arcade-style duel.

## Clip

![Bang Demo](../../media/bang-demo.gif)

## Algorithm / Network

- Algorithm: DQN with double Q-learning, dueling heads, and prioritized replay
- Hidden sizes: `[96, 96]`

## Controls (Human)

- Move: `W/A/S/D`
- Aim: left/right arrows
- Shoot: `Space`
- If no movement key is held for a frame, movement intent becomes `move_stop`

## Observation / Actions

- Observation: `28` floats (`INPUT_FEATURE_NAMES`, exact order)

```python
[
    # SELF
    "self_ang_sin",
    "self_ang_cos",
    "self_move_x",
    "self_move_y",
    "self_shot_cd_norm",
    # SENS
    "sens_fwd",
    "sens_left",
    "sens_right",
    "sens_back",
    # OPP
    "opp1_dx",
    "opp1_dy",
    "opp1_los",
    "opp1_ang_sin",
    "opp1_ang_cos",
    "opp2_dx",
    "opp2_dy",
    "opp2_los",
    "opp2_ang_sin",
    "opp2_ang_cos",
    "opp3_dx",
    "opp3_dy",
    "opp3_los",
    "opp3_ang_sin",
    "opp3_ang_cos",
    "opp_near_dist_norm",
    # HAZ
    "haz_tti_norm",
    "haz_miss_norm",
    "haz_in_traj",
]
```
- Actions: `Discrete(8)` (`ACTION_NAMES`, ordered)
  - `0 move_up`
  - `1 move_down`
  - `2 move_left`
  - `3 move_right`
  - `4 move_stop`
  - `5 aim_left`
  - `6 aim_right`
  - `7 shoot`

- `sens_*` values are normalized free-space-before-hit values in `[0, 1]`. Hits include arena walls and square obstacles.
- Opponent slots are filled from alive opponents sorted by `(distance, fixed player order)`, so the slot order stays deterministic.
- `opp*_ang_sin` and `opp*_ang_cos` are derived from the same ego-relative angle for each slotted opponent.

## Environment Notes

- Scripted enemies retain the same target-selection and shot-error model used by earlier versions of the game.
- Enemy movement comes from a small local planner rather than random move attempts.
- Each replan scores relative move options such as hold, advance, retreat, strafe, and diagonal flank variants.
- The planner prioritizes:
  - regaining line of sight around cover
  - holding a useful engagement distance around `SAFE_RADIUS`
  - strafing when already in a favorable position
  - avoiding oscillation around recently visited cells
- `enemy_reposition_bias` is the main difficulty knob for enemy movement pressure.

An episode counts as a success when the player wins the match.

## Rewards (Training)

- `REWARD_WIN = +10.0` on match win
- `PENALTY_LOSE = -5.0` on match loss
- `REWARD_KILL = +2.0` per enemy elimination
- Engagement shaping: `clip(0.5 * (Phi_eng_next - Phi_eng_prev), -0.25, +0.25)` where `Phi_eng = (1 if tgt_in_los else 0) - tgt_dist_norm`
- Hazard shaping: `clip(0.5 * (Phi_haz_next - Phi_haz_prev), -0.25, +0.25)` where `Phi_haz = haz_dist_norm - 1.5 * haz_in_traj`
- `PENALTY_STEP = -0.005` every training step

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/bang/config.py` under `CURRICULUM_PROMOTION`
- Levels:
  - Level 1: `2` players, `4` obstacles, enemy reposition bias `0.25`, enemy shoot probability `0.025`
  - Level 2: `2` players, `8` obstacles, enemy reposition bias `0.60`, enemy shoot probability `0.05`
  - Level 3: `4` players, `12` obstacles, enemy reposition bias `1.00`, enemy shoot probability `0.10`

## Run Commands

```bash
rl-toybox-train --game bang
rl-toybox-play-ai --game bang --model best --render
rl-toybox-play-user --game bang
python -m scripts.train --game bang
python -m scripts.play_ai --game bang --model best --render
python -m scripts.play_user --game bang
```

See `games/bang/config.py` for the full reward constants, curriculum settings, and DQN hyperparameters.
