# Bang

Top-down arena shooter focused on movement, aiming, and timing shots under pressure.

## Quick Clip

<video src="../../media/bang-demo.mp4" width="600" controls preload="metadata"></video>
- Good clip idea: one clean duel that shows movement, aim correction, and a finish.

## Algorithm / Network

- Algo: enhanced DQN (double + dueling + prioritized replay)
- Hidden sizes: `[64, 64]`

## Controls (Human)

- Move: `W/A/S/D` (`move_up/move_left/move_down/move_right`)
- Aim: left/right arrows (`aim_left/aim_right`)
- Shoot: `Space` (`shoot`)
- If no movement key is pressed in a frame, movement becomes `move_stop`.

## Observation / Actions

- Observation: `24` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_angle_sin`
  - `self_angle_cos`
  - `self_move_intent_x`
  - `self_move_intent_y`
  - `self_shot_cd_norm`
  - `self_tgt_seen_norm`
  - `ray_fwd`
  - `ray_left`
  - `ray_right`
  - `ray_back`
  - `tgt_dx`
  - `tgt_dy`
  - `tgt_dvx`
  - `tgt_dvy`
  - `tgt_dist`
  - `tgt_in_los`
  - `tgt_rel_angle_sin`
  - `tgt_rel_angle_cos`
  - `haz_dx`
  - `haz_dy`
  - `haz_dvx`
  - `haz_dvy`
  - `haz_dist`
  - `haz_in_trajectory`
- Actions: `Discrete(8)` (`ACTION_NAMES`, ordered)
  - `0 move_up`
  - `1 move_down`
  - `2 move_left`
  - `3 move_right`
  - `4 move_stop`
  - `5 aim_left`
  - `6 aim_right`
  - `7 shoot`

Ray notes:
- `ray_*` are normalized distance-to-first-hit values in `[0,1]`.
- `1.0` means no hit within ray range.
- Hits include arena walls and obstacles.

## Rewards (Training)

- Outcome `REWARD_WIN`: `+10` on match win.
- Outcome `PENALTY_LOSE`: `-5` on match loss.
- Event `REWARD_KILL`: `+2` per enemy elimination.
- Engagement shaping: `r_eng = clip(0.2 * (Phi_eng_next - Phi_eng_prev), -0.1, +0.1)`, `Phi_eng = (1 if tgt_in_los else 0) - tgt_dist_norm`.
- Hazard shaping: `r_haz = clip(0.2 * (Phi_haz_next - Phi_haz_prev), -0.1, +0.1)`, `Phi_haz = haz_dist_norm - 1.5 * haz_in_trajectory`.
- Step `PENALTY_STEP`: `-0.005` every training step.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/bang/config.py` under `CURRICULUM_PROMOTION`.
- Levels:
  - Level 1: `2` players, low obstacles, easy enemy behavior
  - Level 2: `2` players, medium obstacles, medium enemy behavior
  - Level 3: `4` players, high obstacles, hard enemy behavior

Success per episode is `1` on win, else `0`.

## Run Commands

```bash
rl-toybox-train --game bang
rl-toybox-play-ai --game bang --model best --render
rl-toybox-play-user --game bang
```

Check `games/bang/config.py` for full hyperparameters.
