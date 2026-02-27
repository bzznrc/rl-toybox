# Bang

Top-down arena shooter environment.

## Algorithm / Network

- Algo: enhanced DQN (double + dueling + prioritized replay)
- Hidden sizes: `[64, 64]`

## Controls (Human)

- Move: `W/A/S/D`
- Aim: mouse (preferred) or `Q/E` (also left/right arrows)
- Shoot: `Space` or left mouse button

## Observation / Actions

- Observation: `24` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_angle_sin`
  - `self_angle_cos`
  - `self_move_intent_x`
  - `self_move_intent_y`
  - `self_aim_intent`
  - `self_last_action`
  - `self_time_since_shot`
  - `self_time_since_tgt_seen`
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

Ray semantics:

- `ray_*` are normalized distance-to-first-hit values in `[0,1]`
- `1.0` means no hit within ray range
- hits include arena walls and obstacles

## Rewards (Training)

- Time step: `-0.005`
- Bad shot (shoot with no LOS): `-0.1`
- Blocked move: `-0.1`
- Hit enemy: `+5.0` per hit
- Win: `+20.0`
- Lose: `-10.0`

## Training

Default algo is enhanced DQN (double + dueling + prioritized replay):

```bash
rl-toybox-train --game bang
```

Play AI:

```bash
rl-toybox-play-ai --game bang --model best --render
```

Play user:

```bash
rl-toybox-play-user --game bang
```
