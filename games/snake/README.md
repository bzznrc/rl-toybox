# Snake

Grid Snake environment with obstacles.

## Controls (Human)

- Move: `W/A/S/D`

## Observation / Actions

- Observation: `12` values
  - `danger_straight`, `danger_right`, `danger_left`
  - direction one-hot-ish flags (`dir_left/right/up/down`)
  - food direction flags (`food_left/right/up/down`)
  - `snake_length`
- Actions: `Discrete(3)`
  - `0 straight`
  - `1 turn_right`
  - `2 turn_left`

## Rewards (Training)

- Eat food: `+10`
- Death/timeout: `-5`
- Otherwise: `0` (the configured `REWARD_STEP` is currently not applied in `TrainingSnakeGame.play_step`)

## Training

Default algo is Q-learning with `LinearQNet`:

```bash
rl-toybox-train --game snake
```

Play AI:

```bash
rl-toybox-play-ai --game snake --model best --render
```

Play user:

```bash
rl-toybox-play-user --game snake
```
