# Vroom

Top-down one-lap racing with procedural closed-loop tracks.
Vroom is the repo's compact continuous-control showcase: same arcade feel and core car physics, but now driven with true continuous `steer / throttle / brake` controls and SAC-oriented defaults.

## Clip

[![Vroom Demo](../../media/vroom-demo.gif)](../../media/vroom-demo.mp4)

## Algorithm / Network

- Algo: soft actor-critic (`sac`)
- Hidden sizes: `[128, 128]`

## Controls (Human)

- Steer: `A/D` or left/right arrows
- Throttle: `W` or up arrow
- Brake: `S` or down arrow
- Coast: no throttle / brake input

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

Observation notes:
- The observation stays vector-only and intentionally compact.
- `ego_*` names are reserved for car-state features.
- `track_*` names are reserved for track-relative and lookahead geometry features.
- `ray_*` names are reserved for normalized probe distances.
- `flag_*` names are reserved for binary state flags.
- `track_heading_err_*` encode the player's heading relative to the local track tangent.
- `track_look_*` encode heading relative to near/far lookahead tangents.
- `track_curve_*` encode how the track tangent itself changes over those lookaheads.
- The rendered ray lines now match the NN inputs exactly: forward, front-left, front-right, left, and right.
- `ray_*` values are normalized free space before the track edge; `0.0` means near-contact and `1.0` means no edge hit within probe range.
- `flag_contact` and `flag_off_track` are binary control-state flags.

## Environment Notes

### Race Rules

- Each race is exactly `1` lap.
- A new random smooth closed-loop track is created at every reset.
- Car spawn row and lane are randomized each race across the start strip.
- If any car completes a lap, the race ends.
- Race count per episode:
  - `train`: `1` race
  - `eval` / `human`: `10` races per set

### Vehicle Model

- The core top-down car handling is intentionally preserved from the previous Vroom runtime.
- Steering, speed cap, drag, lateral damping, off-track slowdown, and collision response still follow the same underlying physics path.
- The main change is the action interface: both the policy and human play now use the same continuous `steer / throttle / brake` control channels.

### Track Generation

- Tracks stay in the same rounded-rectangle family as before.
- The two short sides are always straight.
- The two long sides are each sampled independently from:
  - `straight`
  - `bell`: one smooth inward indentation
  - `s_curve`: two opposing smooth bends
- The long-side amplitudes are intentionally stronger now so those templates read more clearly while staying within the same family.
- Rounded corners and the start strip are still built by the same geometry-first pipeline, so the resulting tracks stay readable and raceable.

### Scripted Opponents

- Opponents follow a simple lane-keeping script.
- Speed targets are scaled by `opponent_speed_cap`, with lighter speed reductions on curved long-side templates and heavier slowdowns in the four rounded-corner zones.
- Each opponent samples a per-bend coasting timing error from `LEVEL_SETTINGS[*]["opponent_coast_error_choices"]`.
  - Negative values coast early.
  - `0` means the reference corner-entry point with no timing error.
  - Positive values coast late.
- After contact or an off-line shove, they blend back toward their assigned lane instead of snapping instantly.

## Rewards (Training)

- Outcome `REWARD_WIN`: `+10` when the player wins.
- Outcome `PENALTY_LOSE`: `-5` when another car wins or timeout resolves against player.
- Progress shaping: `r_progress = clip(5.0 * (Phi_next - Phi_prev), -0.25, +0.25)` with `Phi = track_progress_norm`.
- Event `PENALTY_COLLISION`: `-0.5` on collision-start events.
- Step `PENALTY_STEP`: `-0.005` every training step.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/vroom/config.py` under `CURRICULUM_PROMOTION`.
- Levels:
  - Level 1: `1` car, opponent speed cap `0.0`, coast error choices `[-40, 0, 40]`
  - Level 2: `2` cars, opponent speed cap `0.75`, coast error choices `[-20, 0, 20]`
  - Level 3: `4` cars, opponent speed cap `1.0`, coast error choices `[-10, 0, 10]`

Success per episode is `1` if player wins, else `0`.

## Run Commands

```bash
rl-toybox-train --game vroom
rl-toybox-play-ai --game vroom --model best --render
rl-toybox-play-user --game vroom
python -m scripts.train --game vroom
python -m scripts.play_ai --game vroom --model best --render
python -m scripts.play_user --game vroom
```

Check `games/vroom/config.py` for full hyperparameters.
