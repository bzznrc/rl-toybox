# Vroom

Top-down one-lap racing with procedural closed-loop tracks and continuous `steer / throttle / brake` control. `vroom` is the main continuous-control game in the repo and the smallest environment here that still feels like a proper racing game.

## Clip

![Vroom Demo](../../media/vroom-demo.gif)

## Algorithm / Network

- Default algorithm: `sac`
- IO: `obs=32`, `act=3`
- Actor: `32 -> 64 -> 64 -> 3`
- Twin critics: `(32 + 3) -> 64 -> 64 -> 1`

## Controls (Human)

- Steer: `A/D` or left/right arrows
- Throttle: `W` or up arrow
- Brake: `S` or down arrow
- Coast: release throttle and brake
- Rendered overlays: `X` toggles translucent route, edge, and car sensor ghosts during `play-user` and `play-ai`

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> SENS -> FLAG`
- Observation: `32` floats (`INPUT_FEATURE_NAMES`, exact order)
  - `SELF` (7): `self_lat_off self_spd_lat self_spd_fwd self_spd_delta self_yaw_rate self_head_err_sin self_head_err_cos`
  - `SENS / ROUTE` (14): 3 future centerline breadcrumbs plus near/far curve summaries
  - `SENS / EDGE` (5): continuous road-boundary clearance rays
  - `SENS / CAR` (4): continuous nearby-car clearances
  - `FLAG` (2): `flag_contact flag_off_track`
- Actions: `Box(3)` (`ACTION_NAMES`, ordered)
  - `steer` in `[-1, 1]`
  - `throttle` in `[0, 1]`
  - `brake` in `[0, 1]`

Ordered observation features:

```text
self_lat_off
self_spd_lat
self_spd_fwd
self_spd_delta
self_yaw_rate
self_head_err_sin
self_head_err_cos
sens_route1_fwd
sens_route1_lat
sens_route1_tan_sin
sens_route1_tan_cos
sens_route2_fwd
sens_route2_lat
sens_route2_tan_sin
sens_route2_tan_cos
sens_route3_fwd
sens_route3_lat
sens_route3_tan_sin
sens_route3_tan_cos
sens_curve_near
sens_curve_far
sens_edge_fwd
sens_edge_left_front
sens_edge_right_front
sens_edge_left
sens_edge_right
sens_car_fwd
sens_car_left
sens_car_right
sens_car_back
flag_contact
flag_off_track
```

The observation is intentionally vector-only and compact. `self_*` features encode car state in the local track frame. `sens_route*` samples future centerline points by track progress, which keeps lookahead meaningful on bendy, deformed, or folded playmat tracks where straight-line screen distance can point at the wrong road ribbon. `sens_curve_near` and `sens_curve_far` summarize upcoming turn strength, while `sens_edge_*` expose continuous road clearance.

`sens_car_*` values are continuous egocentric car clearances, not binary flags: `1.0` means no car is nearby in that sector, and `0.0` means touching or immediate collision risk. This gives the SAC policy a small signal for braking behind a car, avoiding occupied side space, and noticing pressure from behind without adding full opponent state.

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
- `flag_contact` and `flag_off_track` are binary observations.
- Ghost mode is visual-only and draws the 3 route breadcrumbs with tangent markers, the 5 edge rays, and the 4 car-clearance rays.

### Track Generation

- Tracks use one geometry-first deformed-loop generator.
- The lowest-complexity track is still the plain rounded rectangle.
- Higher complexity progressively deforms the top and bottom long sides inward, up to playmat/intestine-like folds.
- Track width is `75 px`.
- Each long side independently samples one of:
  - `straight`
  - `bell`
  - `s_curve`
  - `fold`
- At level 5, the long-side templates are sampled independently and evenly at `25%` each.
- The left and right sides can also sample `straight` or `bell`; at level 5 their mix is `50% / 50%`.
- Folds are validated with centerline clearance (`track_width + fold_gap`) so road ribbons do not overlap.
- Rounded corners and the start strip are built through the geometry-first pipeline in `games/vroom/track_geometry.py` and `games/vroom/trackgen.py`.
- Gameplay and rendering both follow the smooth analytical track geometry.
- The rendered road uses a supersampled smooth polygon with consistent smooth edge margins.
- The start mark is a single margin-colored line centered on the chosen side.

### Scripted Opponents

- Opponents use a compact lane-keeping controller rather than a learned policy.
- Scripted opponents use a curriculum max-speed cap from `0%` at level 1 to `100%` at level 5, then slow down through the bend/corner planner.
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

- Shared 5-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/vroom/config.py` under `CURRICULUM_PROMOTION`
- Levels:
  - Level 1: `1` car, opponent speed cap `0.0`, coast error choices `[0.0]`
  - Level 2: `2` cars, opponent speed cap `0.25`, coast error choices `[-40, 0, 40]`
  - Level 3: `2` cars, opponent speed cap `0.5`, coast error choices `[-30, 0, 30]`
  - Level 4: `3` cars, opponent speed cap `0.75`, coast error choices `[-20, 0, 20]`
  - Level 5: `4` cars, opponent speed cap `1.0`, coast error choices `[-10, 0, 10]`

## Run Commands

```bash
rl-toybox-train --game vroom
rl-toybox-play-ai --game vroom --render
rl-toybox-play-user --game vroom
python -m scripts.train --game vroom
python -m scripts.play_ai --game vroom --render
python -m scripts.play_user --game vroom
```

When `--level` is omitted, `train` starts at `L1` and `play-user` / `play-ai` default to `L5`.

See `games/vroom/config.py` for the physics parameters, track-generation settings, curriculum knobs, and training defaults. Vroom's game-wide net size lives in `DEFAULT_MODEL_CONFIG["hidden_sizes"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
