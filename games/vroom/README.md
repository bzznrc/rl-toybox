# Vroom

Top-down one-lap racing environment with procedural closed-loop tracks.

## Controls (Human)

- Steer left/right: `A/D` or left/right arrows
- Accelerate: `W` or up arrow
- Brake/reverse: `S` or down arrow

## Observation / Actions

- Observation: `6` floats
  - signed lateral offset from nearest track centerline
  - forward speed along local track tangent
  - heading alignment sine vs tangent
  - heading alignment cosine vs tangent
  - normalized nearest opponent distance
  - `in_contact` (0/1)
- Actions: `Discrete(5)`
  - `0 NOOP`
  - `1 LEFT`
  - `2 RIGHT`
  - `3 ACCEL`
  - `4 BRAKE`

## Race Rules

- Each race is exactly `1` lap.
- New random smooth closed-loop track at every reset/new race.
- `4` cars on the grid; player car is teal.
- If any car completes a lap, race ends and next reset starts a new one.

## Physics

- Car-like movement with heading, acceleration, drag, and lateral grip.
- Car-to-car contact prevents overlap and applies impact-based push/impulse.
- Track boundary contact pushes cars back inside the lane.
- Cars are rendered as two-tone square tiles.

## Rewards (Training)

- Step cost: `-0.002`
- Progress reward: `+0.01 * max(0, forward_speed_along_track)`
- Win race: `+10.0`
- Lose race: `-5.0`

## Training

Default algo is vanilla DQN:

```bash
rl-toybox-train --game vroom
```

Play AI:

```bash
rl-toybox-play-ai --game vroom --model best --render
```

Play user:

```bash
rl-toybox-play-user --game vroom
```
