# Bang

Top-down arena shooter focused on movement, aiming, and timing shots under pressure.

## Clip

[![Bang Demo](../../media/bang-demo.gif)](../../media/bang-demo.mp4)

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
  - `ray_fwd`
  - `ray_left`
  - `ray_right`
  - `ray_back`
  - `opp1_dx`
  - `opp1_dy`
  - `opp1_los`
  - `opp1_rel_ang`
  - `opp2_dx`
  - `opp2_dy`
  - `opp2_los`
  - `opp2_rel_ang`
  - `opp3_dx`
  - `opp3_dy`
  - `opp3_los`
  - `opp3_rel_ang`
  - `haz_tti_norm`
  - `haz_miss_norm`
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
- `ray_*` are normalized free-space-before-hit values in `[0,1]`.
- `0.0` means the ray is blocked immediately.
- `1.0` means no hit within ray range.
- Hits include arena walls and obstacles.

## Environment Notes

### Scripted Enemies

- Scripted enemies keep the current target-selection and shot-error model.
- Movement is driven by a small local planner instead of random move attempts.
- Each replan scores a small set of relative moves: hold, advance, retreat, strafe, and diagonal flank variants.
- The planner prefers:
  - regaining line of sight around obstacles
  - holding a useful distance band around `SAFE_RADIUS`
  - strafing when already in a good engagement position
  - avoiding recently visited positions so enemies do not oscillate behind cover
- `enemy_reposition_bias` is the main movement difficulty knob.
  - Lower values make enemies less forceful about flanking cover.
  - Higher values make them push harder to regain line of sight.

Success per episode is `1` on win, else `0`.

## Rewards (Training)

- Outcome `REWARD_WIN`: `+10` on match win.
- Outcome `PENALTY_LOSE`: `-5` on match loss.
- Event `REWARD_KILL`: `+2` per enemy elimination.
- Engagement shaping: `r_eng = clip(0.5 * (Phi_eng_next - Phi_eng_prev), -0.25, +0.25)`, `Phi_eng = (1 if tgt_in_los else 0) - tgt_dist_norm`.
- Hazard shaping: `r_haz = clip(0.5 * (Phi_haz_next - Phi_haz_prev), -0.25, +0.25)`, `Phi_haz = haz_dist_norm - 1.5 * haz_in_trajectory`.
- Step `PENALTY_STEP`: `-0.005` every training step.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/bang/config.py` under `CURRICULUM_PROMOTION`.
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

Check `games/bang/config.py` for full hyperparameters.
