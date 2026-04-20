# Vroom

Top-down one-lap racing with procedural closed-loop tracks and continuous `steer / throttle / brake` control. `vroom` is the main continuous-control game in the repo and the smallest environment here that still feels like a proper racing game.

## Clip

![Vroom Demo](../../media/vroom-demo.gif)

## Algorithm / Network

- Default algorithm: `sac`
- IO: `obs=20`, `act=3`
- Actor: `20 -> 64 -> 64 -> 3`
- Twin critics: `(20 + 3) -> 64 -> 64 -> 1`

## Controls (Human)

- Steer: `A/D` or left/right arrows
- Throttle: `W` or up arrow
- Brake: `S` or down arrow
- Coast: release throttle and brake
- Rendered overlays: `X` toggles the sensor-ray visualization during `play-user` and `play-ai`

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> SENS -> FLAG`
- Observation: `20` floats (`INPUT_FEATURE_NAMES`, exact order)
  - `SELF` (7): `self_lat_off self_spd_lat self_spd_fwd self_spd_delta self_yaw_rate self_head_err_sin self_head_err_cos`
  - `SENS` (11): `sens_look_near_sin sens_look_near_cos sens_look_far_sin sens_look_far_cos sens_curve_near sens_curve_far sens_fwd sens_left_front sens_right_front sens_left sens_right`
  - `FLAG` (2): `flag_contact flag_off_track`
- Actions: `Box(3)` (`ACTION_NAMES`, ordered)
  - `steer` in `[-1, 1]`
  - `throttle` in `[0, 1]`
  - `brake` in `[0, 1]`

The observation is intentionally vector-only and compact. `self_*` features encode car state in the local track frame, `sens_*` features encode look-ahead geometry plus road-edge clearance, and `flag_*` features expose binary control-state information.

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
rl-toybox-play-ai --game vroom --render
rl-toybox-play-user --game vroom
python -m scripts.train --game vroom
python -m scripts.play_ai --game vroom --render
python -m scripts.play_user --game vroom
```

See `games/vroom/config.py` for the physics parameters, track-generation settings, curriculum knobs, and training defaults. Vroom's game-wide net size lives in `DEFAULT_MODEL_CONFIG["hidden_sizes"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
