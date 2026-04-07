# Stealth

Stealth is the active replacement for Peek: a tiny top-down POMDP where the only goal is to reach the exit without being seen.

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

- Move: `W/A/S/D`
- Wait: `Space` or no movement key

## Observation / Actions

Observation is exactly `32` floats:

- Local egocentric patch (`25` values): a flattened `5x5` patch centered on the player
  - `0 empty`
  - `1 wall`
  - `2 cover`
  - `3 exit`
  - `4 guard`
  - `5 danger tile`
- Scalar features (`7` values)
  - `has_exit_in_view`
  - `exit_dx_if_seen`
  - `exit_dy_if_seen`
  - `on_cover`
  - `danger_now`
  - `steps_remaining_norm`
  - `patrol_phase_norm`

Actions (`5`, ordered):

1. `move_up`
2. `move_down`
3. `move_left`
4. `move_right`
5. `wait`

## Environment Notes

- There is no key, no inventory, and no interaction button.
- The exit is present from the start of the episode.
- Guards patrol deterministic back-and-forth routes.
- Cover tiles block guard vision and create safe waiting spots.
- Maps stay small and readable; the challenge is timing, local visibility, and remembering patrol state rather than solving a large maze.

## Rewards (Training)

Realized reward components are logged as `W L P S B`:

- `W`: reach the exit `+10`
- `L`: get seen or run out of time `-5`
- `P`: small clipped progress reward toward the exit
- `S`: per-step penalty `-0.01`
- `B`: blocked movement penalty `-0.015`

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Level 1: one guard, short route, light cover
- Level 2: one guard, longer route, more branching
- Level 3: two guards, denser patrol timing, more cover interactions

Success per episode is `1` only when the player reaches the exit without being seen.

## Run Commands

```bash
rl-toybox-train --game stealth
rl-toybox-play-ai --game stealth --model best --render
rl-toybox-play-user --game stealth
python -m scripts.train --game stealth
python -m scripts.play_ai --game stealth --model best --render
python -m scripts.play_user --game stealth
```
