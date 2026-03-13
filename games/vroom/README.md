# Vroom

Top-down one-lap racing with procedural closed-loop tracks.

## Clip

[![Vroom Demo](../../media/vroom-demo.gif)](../../media/vroom-demo.mp4)

## Algorithm / Network

- Algo: vanilla DQN
- Hidden sizes: `[32, 32]`

## Controls (Human)

- Steer: `A/D` or left/right arrows
- Throttle: `W` or up arrow
- Coast: no input

## Observation / Actions

- Observation: `18` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `spd_fwd`
  - `spd_lat`
  - `yaw_rt`
  - `surf`
  - `trk_off`
  - `trk_ang`
  - `trk_ang_n`
  - `trk_ang_f`
  - `edg_fl`
  - `edg_fr`
  - `edg_l`
  - `edg_r`
  - `opp1_dx`
  - `opp1_dy`
  - `opp2_dx`
  - `opp2_dy`
  - `opp3_dx`
  - `opp3_dy`
- Actions: `Discrete(6)` (`ACTION_NAMES`, ordered)
  - `0 coast`
  - `1 throttle`
  - `2 left_coast`
  - `3 right_coast`
  - `4 left_throttle`
  - `5 right_throttle`

Opponent slot notes:
- `opp{1..3}_{dx,dy}` are ego-frame relative coordinates.
- Opponents are ordered deterministically each frame: ahead first by nearest longitudinal `dx`, then behind by nearest `|dx|`, with `dy` tie-break.
- Missing opponent slots are zero-filled.

Edge probe notes:
- `edg_*` are normalized free-space-before-track-edge values in `[0,1]`.
- `0.0` means the edge is effectively in contact with the probe origin and `1.0` means no edge within probe range.

## Environment Notes

### Race Rules

- Each race is exactly `1` lap.
- A new random smooth closed-loop track is created at every reset.
- Car spawn row and lane are randomized each race across the start strip.
- If any car completes a lap, the race ends.
- Race count per episode:
  - `train`: `1` race
  - `eval` / `human`: `10` races per set

### Scripted Opponents

- Opponents follow a simple lane-keeping script.
- Speed targets are scaled by `opponent_speed_cap`, with three section multipliers:
  - `1.0x` on plain sides
  - `0.75x` on bulged sides
  - `0.5x` in the corner-control zone for each of the four main corners
- They coast only for the four main rounded corners of the track.
- Bulges are driven at regular speed rather than corner-coasted.
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
