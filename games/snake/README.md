# Snake

Classic grid Snake with a small obstacle curriculum, compact observations, and lightweight shaping rewards. It is the simplest entry point in the repo and the easiest environment to inspect end to end.

## Clip

![Snake Demo](../../media/snake-demo.gif)

## Default Algorithm / Network

- Algorithm: linear Q-learning (`qlearn`)
- Hidden sizes: `[32]`

## Controls (Human)

- Move: `W/A/S/D`

## Observation / Actions

- Observation: `12` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_heading_sin`
  - `self_heading_cos`
  - `self_length`
  - `self_last_action`
  - `ray_fwd`
  - `ray_left`
  - `ray_right`
  - `tgt_rel_angle_sin`
  - `tgt_rel_angle_cos`
  - `tgt_manhattan_dist`
  - `tgt_dist_delta`
  - `self_steps_since_food`
- Actions: `Discrete(3)`
  - `0 straight`
  - `1 turn_right`
  - `2 turn_left`

Ray features are normalized free-space-before-collision measurements in the snake's local frame. `0.0` means a collision is adjacent; `1.0` means no collision is found within the probe range.

## Environment Notes

- `WRAP_AROUND` is a global toggle in `games/snake/config.py`.
- Obstacles are static for the duration of an episode and are excluded from food spawn placement.
- The timeout budget scales with current snake length, so longer snakes are given more time to find food.

## Rewards (Training)

- `REWARD_FOOD = +1.0` when food is eaten
- `PENALTY_LOSE = -5.0` on death or timeout
- Progress shaping: `clip(1.0 * (Phi_next - Phi_prev), -0.05, +0.05)` where `Phi = -dist_food_norm - 0.5 * hunger_norm`
- `PENALTY_STEP = -0.005` every training step

`dist_food_norm` is normalized Manhattan head-to-food distance. When wrap-around is enabled it uses the shortest wrapped path. `hunger_norm` is `clamp(self_steps_since_food / hunger_cap_steps, 0, 1)`.

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/snake/config.py` under `CURRICULUM_PROMOTION`
- Levels:
  - Level 1: `0` obstacles, timeout `120 * snake_length`
  - Level 2: `6` obstacles, timeout `100 * snake_length`
  - Level 3: `12` obstacles, timeout `80 * snake_length`

An episode counts as a success if the snake eats at least `5` foods (`SUCCESS_FOODS_REQUIRED`).

## Run Commands

```bash
rl-toybox-train --game snake
rl-toybox-play-ai --game snake --model best --render
rl-toybox-play-user --game snake
python -m scripts.train --game snake
python -m scripts.play_ai --game snake --model best --render
python -m scripts.play_user --game snake
```

See `games/snake/config.py` for the full set of hyperparameters, reward constants, and curriculum thresholds.
