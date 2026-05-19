# Vroom

Top-down one-lap racing with procedural closed-loop tracks and continuous `steer / throttle / brake` control. `vroom` is the repo's main continuous-control game.

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
- Rendered overlays: `X` toggles translucent route, edge, and car-path sensor ghosts during `play-user` and `play-ai`

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> SENS -> FLAG`
- Observation: `32` floats (`INPUT_FEATURE_NAMES`, exact order)
  - `SELF` (7): `self_lat_off self_spd_lat self_spd_fwd self_spd_delta self_yaw_rate self_head_err_sin self_head_err_cos`
  - `SENS / ROUTE` (15): 3 speed-aware future centerline probes, each with local position, tangent, and bend severity
  - `SENS / EDGE` (5): continuous road-boundary clearance rays
  - `SENS / CAR-PATH` (3): continuous left / forward / right path-clearance probes
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
sens_route1_bend
sens_route2_fwd
sens_route2_lat
sens_route2_tan_sin
sens_route2_tan_cos
sens_route2_bend
sens_route3_fwd
sens_route3_lat
sens_route3_tan_sin
sens_route3_tan_cos
sens_route3_bend
sens_edge_fwd
sens_edge_left_front
sens_edge_right_front
sens_edge_left
sens_edge_right
sens_car_left
sens_car_fwd
sens_car_right
flag_contact
flag_off_track
```

The observation is intentionally vector-only and compact. `self_*` features encode car state in the local track frame. `sens_route*` samples future centerline points by track progress, which keeps lookahead meaningful on bendy, deformed, or folded playmat tracks where straight-line screen distance can point at the wrong road ribbon. Route probe lookaheads are speed-aware: low-speed recovery uses `45 / 90 / 180 px`, while high speed uses `75 / 135 / 270 px`. Each route probe also reports absolute local bend severity, so sharp S-curves stay visible even when left/right curvature cancels out over a longer window. `sens_edge_*` exposes continuous road clearance.

`sens_car_left`, `sens_car_fwd`, and `sens_car_right` are continuous egocentric path clearances, not binary flags: `1.0` means the path is clear, and `0.0` means touching or immediate collision risk. During contact, the relevant direction should stay near `0.0`, so `flag_contact` says contact is happening while `sens_car_*` gives the policy the directional escape signal.

## Environment Notes

- The finish goal is always the canonical start/finish line.
- Normal starts require one full valid lap; training random starts create shorter segments from the sampled spawn point to that canonical finish line.
- A fresh smooth closed-loop track is generated at every reset.
- Cars spawn across the start strip with randomized row/lane ordering.
- Training curriculum can randomize the whole race start; all cars move to the same sampled start region without overlapping, while evaluation and human play keep the normal start-line behavior.
- If an opponent reaches the canonical finish or the player reaches it with valid on-track progress, the race ends immediately.
- Episode length:
  - `train`: `1` race
  - `eval` / `human`: `10` races per set

### Vehicle Model

- Steering authority, drag, lateral damping, off-track slowdown, and contact response all come from the same underlying top-down car model.
- Low-speed steering is mildly reduced, with at least `25%` authority at near-zero speed and full low-speed scaling gone by normal driving speed.
- The policy and human runtime share the same continuous control interface.
- `flag_contact` and `flag_off_track` are binary observations; contact direction comes from `sens_car_*`.
- `flag_off_track` follows the off-track severity signal with simple hysteresis, without widening the rendered or physical road.
- Ghost mode is visual-only and draws the 3 speed-aware route probes with tangent and bend markers, the 5 edge rays, and the 3 car-path probes.

### Track Generation

- Tracks use one geometry-first deformed-loop generator.
- The lowest-complexity track is a plain rounded rectangle.
- Higher complexity progressively deforms the top and bottom long sides inward, up to playmat/intestine-like folds.
- Track width is `90 px`.
- Each long side independently samples one of:
  - `straight`
  - `bell`
  - `s_curve`
  - `fold`
- The left and right sides can sample `straight` or `bell`.
- Track complexity is sampled from the current curriculum level. Each reset samples `50%` from the full level range and `50%` from the upper third of that range, so hard tracks show up often without adding extra level knobs.
- Complexity below `0.20` allows only `straight`; `0.20+` can add `bell` / `s_curve`; `0.45+` can add `fold`.
- Folds are validated with centerline clearance (`track_width + fold_gap`) so road ribbons do not overlap.
- Rounded corners and the start strip are built through the geometry-first pipeline in `games/vroom/track_geometry.py` and `games/vroom/trackgen.py`.
- Generated centerlines keep the same template families but run a small deterministic smoothing pass over bend transitions for driveability across all curriculum levels.
- Gameplay, validity, and collision use the same raster road mask; rendered road antialiasing is clipped to that mask.
- Boundary and start/finish paint is clipped to the same mask, so the visible road shape does not extend beyond the playable road.
- The start/finish mark is a white band beginning at the canonical logical start and painted from the same track geometry.

### Scripted Opponents

- Opponents use a compact lane-keeping controller rather than a learned policy.
- Scripted opponents use `opponent_speed_cap` as the curriculum max-speed knob, from `0%` at level 1 to `100%` at level 5.
- Opponents use a bend-aware target speed: they drive near full allowed speed on straights, keep their lane through difficult bends, and brake when edge or steering demand gets high.
- Each opponent samples fixed per-race `speed_mult` and `bend_caution_mult` personality values so traffic is slightly varied without extra curriculum knobs.
- After contact or off-line disturbances, opponents blend back toward their assigned lane instead of snapping instantly.

## Rewards (Training)

- `REWARD_WIN = +10.0` when the player wins
- `PENALTY_LOSE = -5.0` when another car wins or timeout resolves against the player
- Progress shaping uses signed unwrapped valid route-progress deltas from spawn: on-track forward motion adds `P`, on-track backward motion subtracts `P`, and off-track motion gives zero `P`.
- A normal-start player win requires one valid full lap and settles cumulative `P` near `PROGRESS_SCALE` (`7.5`); random-start wins use the shorter valid distance from spawn to the canonical finish and can earn proportionally less `P`.
- `PENALTY_OFF_TRACK = -0.02` scaled by continuous off-track severity each training step; edge grazing is small, fully leaving the road reaches the full penalty.
- Sustained hard off-track driving for `45` steps terminates as a loss.
- `PENALTY_CONTACT = -0.005` each training step while the player car remains in contact; brief bumps are not punished as a separate event.
- `PENALTY_STEP = -0.0075` every training step
- No-progress episodes terminate as a loss if best valid unwrapped progress fails to improve by `0.01` for `240` steps.

An episode counts as a success if the player wins the race set.

## Curriculum (Train)

- Shared 5-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/vroom/config.py` under `CURRICULUM_PROMOTION`
- Promotion uses one rule: compare rolling AS over the last `100` level episodes with the shared `0.80` threshold.
- Training-only random starts decrease by level. When one triggers, the player and opponents spawn around the same sampled start region in different safe slots; direction is sampled within `+/-45` degrees of the local track tangent, speed starts between `0%` and `25%` of max speed, and positions with less than `25%` remaining progress to the canonical finish are rejected.
- Levels:
  - Level 1: `1` car, opponent speed cap `0.0`, complexity `0.00-0.30`, random starts `80%`
  - Level 2: `1` car, opponent speed cap `0.0`, complexity `0.20-0.50`, random starts `60%`
  - Level 3: `2` cars, opponent speed cap `0.50`, complexity `0.40-0.70`, random starts `40%`
  - Level 4: `3` cars, opponent speed cap `0.75`, complexity `0.50-0.80`, random starts `20%`
  - Level 5: `4` cars, opponent speed cap `1.0`, complexity `0.60-0.90`, random starts `0%`

Hard-track oversampling makes tougher tracks appear inside that same AS window, without adding a second promotion gate or per-band metrics.

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

See `games/vroom/config.py` for the physics parameters, track-generation settings, curriculum knobs, and training defaults. Vroom's game-wide net size lives in `DEFAULT_MODEL_CONFIG["hidden_sizes"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`. Vroom also uses conservative SAC overrides for stability on long procedural tracks.
