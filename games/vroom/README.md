# Vroom

Top-down one-lap racing with procedural closed-loop tracks and continuous `steer / throttle / brake` control. `vroom` is the main continuous-control game in the repo and the smallest environment here that still feels like a proper racing game.

## Clip

![Vroom Demo](../../media/vroom-demo.gif)

## Default Algorithm / Network

- Algorithm: soft actor-critic (`sac`)
- Actor: `20 -> 64 -> 64 -> 3`
- Critic Q1/Q2: `(20 + 3) -> 64 -> 64 -> 1`

## Controls (Human)

- Steer: `A/D` or left/right arrows
- Throttle: `W` or up arrow
- Brake: `S` or down arrow
- Coast: release throttle and brake
- Rendered overlays: `X` toggles the sensor-ray visualization during `play-user` and `play-ai`

## Observation / Actions

- Observation: `20` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `track_lat_off`
  - `ego_spd_lat`
  - `ego_spd_fwd`
  - `ego_spd_delta`
  - `ego_yaw_rate`
  - `track_heading_err_sin`
  - `track_heading_err_cos`
  - `track_look_near_sin`
  - `track_look_near_cos`
  - `track_look_far_sin`
  - `track_look_far_cos`
  - `track_curve_near`
  - `track_curve_far`
  - `ray_f`
  - `ray_fl`
  - `ray_fr`
  - `ray_l`
  - `ray_r`
  - `flag_contact`
  - `flag_off_track`
- Actions: `Box(3)` (`ACTION_NAMES`, ordered)
  - `steer` in `[-1, 1]`
  - `throttle` in `[0, 1]`
  - `brake` in `[0, 1]`

The observation is intentionally vector-only and compact. `track_*` features encode track-relative geometry, `ray_*` features encode normalized free space before the road edge, and `flag_*` features expose binary control-state information.

## Environment Notes

- Each race is exactly `1` lap.
- A fresh smooth closed-loop track is generated at every reset.
- Cars spawn across the start strip with randomized row/lane ordering.
- If any car completes the lap, the race ends immediately.
- Episode length:
  - `train`: `1` race
  - `eval` / `human`: `10` races per set

### Vehicle Model

- Steering authority, drag, lateral damping, off-track slowdown, and contact response all come from the same underlying top-down car model.
- The policy and human runtime share the same continuous control interface.
- `flag_contact` and `flag_off_track` are binary observations; the optional rendered rays are visual-only and are not a separate observation channel.

### Track Generation

- Tracks stay in the same rounded-rectangle family across runs.
- The short sides are straight.
- Each long side independently samples one of:
  - `straight`
  - `bell`
  - `s_curve`
- Rounded corners and the start strip are built through the geometry-first pipeline in `games/vroom/track_geometry.py` and `games/vroom/trackgen.py`.

### Scripted Opponents

- Opponents use a compact lane-keeping controller rather than a learned policy.
- Speed targets are scaled by `opponent_speed_cap`.
- Each opponent samples a per-corner coasting timing error from `LEVEL_SETTINGS[*]["opponent_coast_error_choices"]`.
- After contact or off-line disturbances, opponents blend back toward their assigned lane instead of snapping instantly.

## Rewards (Training)

- `REWARD_WIN = +10.0` when the player wins
- `PENALTY_LOSE = -5.0` when another car wins or timeout resolves against the player
- Progress shaping: `clip(5.0 * (Phi_next - Phi_prev), -0.25, +0.25)` where `Phi = track_progress_norm`
- `PENALTY_COLLISION = -0.5` on collision-start events
- `PENALTY_STEP = -0.005` every training step

An episode counts as a success if the player wins the race set.

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/vroom/config.py` under `CURRICULUM_PROMOTION`
- Levels:
  - Level 1: `1` car, opponent speed cap `0.0`, coast error choices `[-40, 0, 40]`
  - Level 2: `2` cars, opponent speed cap `0.75`, coast error choices `[-20, 0, 20]`
  - Level 3: `4` cars, opponent speed cap `1.0`, coast error choices `[-10, 0, 10]`

## Run Commands

```bash
rl-toybox-train --game vroom
rl-toybox-play-ai --game vroom --model best --render
rl-toybox-play-user --game vroom
python -m scripts.train --game vroom
python -m scripts.play_ai --game vroom --model best --render
python -m scripts.play_user --game vroom
```

See `games/vroom/config.py` for the full set of physics parameters, track-generation settings, and curriculum knobs.
