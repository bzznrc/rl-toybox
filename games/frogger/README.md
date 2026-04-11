# Frogger

Compact road-crossing game designed around local sensing, timing, and partial observability. `frogger` is the repo's recurrent-policy showcase: the agent only sees a small egocentric patch plus a few scalar hints, so memory matters.

## Default Algorithm / Network

- Algorithm: recurrent PPO (`recurrent_ppo`)
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

- Observation: `32` floats (`INPUT_FEATURE_NAMES`, ordered)
- Local egocentric patch: `25` values from a `5x5` window centered on the frog
  - `0 empty`
  - `1 boundary`
  - `2 safe row`
  - `3 road lane`
  - `4 car`
  - `5 goal row`
- Scalar supplement: `7` values
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

Canonical feature order is the 25 local patch cells in row-major order followed by the 7 scalar features listed above.

## Why It Is a POMDP

- The policy never sees the full lane state.
- Cars just outside the local patch can still matter.
- Lane direction and speed stay stable within a crossing, so the policy benefits from remembering them while crossing.

## Environment Notes

- A run continues across repeated crossings until a hit or timeout ends it.
- Board layout is always:
  - top goal row
  - `1`, `3`, or `5` traffic lanes
  - bottom safe start row
- Cars move horizontally, recycle across the board, and disappear once fully outside the board width.
- Traffic styles map to color and speed tiers:
  - coral = slow
  - blue = medium
  - sand = fast
- Each scored crossing immediately starts the next one.
- The challenge is local sensing and timing, not hidden rule exceptions.

## Rewards (Training)

- `reward_progress_forward = +0.05` for reaching a new highest row
- `reward_event_goal = +1.0` for reaching the goal row
- `reward_event_hit = -1.0` for being hit by a car
- `reward_cost_step = -0.01` every step
- `reward_terminal_win = +2.0` scored-crossing bonus
- `reward_terminal_loss = -2.0` terminal loss penalty

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Level 1: `1` traffic lane
- Level 2: `3` traffic lanes
- Level 3: `5` traffic lanes

A run counts as a success if the frog scores at least one crossing before the run ends.

## Run Commands

```bash
rl-toybox-train --game frogger
rl-toybox-play-ai --game frogger --model best --render
rl-toybox-play-user --game frogger
python -m scripts.train --game frogger
python -m scripts.play_ai --game frogger --model best --render
python -m scripts.play_user --game frogger
```

See `games/frogger/config.py` for the full lane-generation, reward, and recurrent PPO settings.
