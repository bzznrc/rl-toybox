# Vroom

Top-down one-lap racing environment with procedural closed-loop tracks.

## Algorithm / Network

- Algo: vanilla DQN
- Hidden sizes: `[48, 48]`

## Controls (Human)

- Steer: `A/D` or left/right arrows
- Throttle: `W` or up arrow
- Coast: no input

## Observation / Actions

- Observation: `20` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_lat_offset`
  - `self_lat_offset_delta`
  - `self_fwd_speed`
  - `self_fwd_speed_delta`
  - `self_heading_sin`
  - `self_heading_cos`
  - `self_in_contact`
  - `self_last_action`
  - `ray_fwd_near`
  - `ray_fwd_far`
  - `ray_fwd_left`
  - `ray_fwd_right`
  - `tgt_dx`
  - `tgt_dy`
  - `tgt_dvx`
  - `tgt_dvy`
  - `trk_lookahead_sin`
  - `trk_lookahead_cos`
  - `trk_lookahead_dist`
  - `trk_curvature_ahead`
- Actions: `Discrete(6)` (`ACTION_NAMES`, ordered)
  - `0 coast`
  - `1 throttle`
  - `2 left_coast`
  - `3 right_coast`
  - `4 left_throttle`
  - `5 right_throttle`

Ray semantics:

- `ray_*` are normalized distance-to-first-hit values in `[0,1]`
- `1.0` means no hit within ray range
- hits include walls and obstacles

## Race Rules

- Each race is exactly `1` lap.
- New random smooth closed-loop track at every reset/new race.
- `4` cars on the grid; player car is teal.
- If any car completes a lap, race ends and next reset starts a new one.

## Physics

- Car-like movement with heading, acceleration, drag, and lateral grip.
- Car-to-car contact prevents overlap and applies impact-based push/impulse.
- Track boundary contact pushes cars back inside the lane.
- Cars are rendered as two-tone square tiles.

## Rewards (Training)

- Step cost: `-0.002`
- Progress reward: `+0.01 * max(0, forward_speed_along_track)`
- Win race: `+10.0`
- Lose race: `-5.0`

## Training

Default algo is vanilla DQN:

```bash
rl-toybox-train --game vroom
```

Play AI:

```bash
rl-toybox-play-ai --game vroom --model best --render
```

Play user:

```bash
rl-toybox-play-user --game vroom
```
