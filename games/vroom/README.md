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

## Race Rules

- Each race is exactly `1` lap.
- A new random smooth closed-loop track is created at every reset.
- If any car completes a lap, the race ends.
- Race count per episode:
  - `train` / `eval`: `1` race
  - `human` (`rl-toybox-play-user`): `10` races per set

## Rewards (Training)

- Outcome `REWARD_WIN`: `+10` when the player wins.
- Outcome `PENALTY_LOSE`: `-5` when another car wins or timeout resolves against player.
- Progress shaping: `r_progress = clip(5.0 * (Phi_next - Phi_prev), -0.25, +0.25)` with `Phi = track_progress_norm`.
- Event `PENALTY_COLLISION`: `-0.5` on collision-start events.
- Step `PENALTY_STEP`: `0` every training step.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/vroom/config.py` under `CURRICULUM_PROMOTION`.
- Levels:
  - Level 1: `1` car, opponent speed cap `0.0`
  - Level 2: `2` cars, opponent speed cap `0.75`
  - Level 3: `4` cars, opponent speed cap `1.0`

Success per episode is `1` if player wins, else `0`.

## Run Commands

```bash
rl-toybox-train --game vroom
rl-toybox-play-ai --game vroom --model best --render
rl-toybox-play-user --game vroom
```

Check `games/vroom/config.py` for full hyperparameters.
