# Frogger

Tiny one-crossing road-runner built as the repo's partial-observability / memory showcase.
Frogger replaces the old memory slot with a smaller, cleaner recurrent PPO target: top goal row, bottom safe start row, and only a few traffic lanes in between.

## Clip

No embedded clip yet.

## Algorithm / Network

- Algo: recurrent PPO (`recurrent_ppo`)
- Shared encoder: `[32]`
- Recurrent hidden size: `64`
- Actor head: `[32]`
- Critic head: `[32]`
- Action space: `Discrete(5)`

## Controls (Human)

- Move: arrow keys or `W/A/S/D`
- Wait: `Space`
- Goal: keep chaining crossings from the bottom safe row to the top goal row until a hit or timeout ends the run

## Observation / Actions

- Observation: exactly `32` floats (`INPUT_FEATURE_NAMES`, ordered)
- Local egocentric patch: `25` values, flattened row-major from top-left to bottom-right over a `5x5` window centered on the frog
  - `0 empty`
  - `1 boundary`
  - `2 safe row`
  - `3 road lane`
  - `4 car`
  - `5 goal row`
- Scalar supplement: `7` values
  - `run_steps_remaining_norm`: remaining step budget in `[0, 1]`
  - `frog_lane_id_norm`: normalized current traffic-lane id, `0.0` on safe rows
  - `frog_x_norm`: normalized frog column in `[0, 1]`
  - `goal_dy_norm`: normalized vertical distance to the goal row in `[0, 1]`
  - `lane_dir_here`: `-1` for left-moving traffic, `+1` for right-moving traffic, `0` on safe rows
  - `lane_speed_here_norm`: current lane speed divided by the configured max lane speed, `0` on safe rows
  - `flag_danger_now`: `1.0` when the current tile is unsafe now or on the next traffic advance, else `0.0`

Canonical feature order:

- `local_00` `local_01` `local_02` `local_03` `local_04`
- `local_05` `local_06` `local_07` `local_08` `local_09`
- `local_10` `local_11` `local_12` `local_13` `local_14`
- `local_15` `local_16` `local_17` `local_18` `local_19`
- `local_20` `local_21` `local_22` `local_23` `local_24`
- `run_steps_remaining_norm`
- `frog_lane_id_norm`
- `frog_x_norm`
- `goal_dy_norm`
- `lane_dir_here`
- `lane_speed_here_norm`
- `flag_danger_now`

- Actions: `Discrete(5)` (`ACTION_NAMES`, ordered)
  - `0 up`
  - `1 down`
  - `2 left`
  - `3 right`
  - `4 wait`

Why it is a POMDP:

- The policy never sees full-lane traffic state.
- The patch is local and egocentric, so cars just outside view matter.
- Each lane keeps a fixed per-crossing direction and speed, which the policy has to remember while timing crossings.

## Environment Notes

- Each run keeps going across repeated crossings until a car hit or timeout ends it.
- Layout is always:
  - top goal row
  - `1`, `3`, or `5` traffic lanes
  - bottom safe start row
- The board is rendered much larger, with wider columns and taller row spacing so it fills the window cleanly.
- Cars move horizontally, recycle across the board, and are clipped as soon as they cross the board boundary.
- Car colors map to traffic styles:
  - coral = slow
  - blue = medium
  - sand = fast
- The bottom bar keeps a clock plus compact win icons; each scored crossing adds one point and immediately starts the next crossing.
- Traffic density stays moderate and intentionally readable.
- The challenge is not hidden rules; it is local sensing, timing, and memory across lanes.

## Rewards (Training)

Realized reward components are logged with the canonical internal names:

- `reward_progress_forward`: `+0.05` for entering a new highest row reached
- `reward_event_goal`: `+1.0` for reaching the goal row on a crossing
- `reward_event_hit`: `-1.0` for getting hit by a car
- `reward_cost_step`: `-0.01` every step
- `reward_terminal_win`: `+2.0` scored-crossing bonus
- `reward_terminal_loss`: `-2.0` terminal loss penalty

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Level 1: `1` traffic lane
- Level 2: `3` traffic lanes
- Level 3: `5` traffic lanes

Success per run is `1` if the frog scores at least one crossing before the run ends.

## Run Commands

```bash
rl-toybox-train --game frogger
rl-toybox-play-ai --game frogger --model best --render
rl-toybox-play-user --game frogger
python -m scripts.train --game frogger
python -m scripts.play_ai --game frogger --model best --render
python -m scripts.play_user --game frogger
```

Check `games/frogger/config.py` for full hyperparameters.
