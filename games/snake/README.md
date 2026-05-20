# Snake

Classic grid Snake with a small obstacle curriculum, compact observations, and lightweight shaping rewards. It is the simplest entry point in the repo.

## Inspiration

This example was originally inspired by Nancy Zhou’s Medium walkthrough,
“Teaching an AI to Play the Snake Game Using Reinforcement Learning,”
which itself builds on Patrick Loeber’s Snake AI tutorial.

The implementation here is adapted to rl-toybox’s own environment,
observation, curriculum, rendering, and training structure.

References:
- Nancy Zhou, “Teaching an AI to Play the Snake Game Using Reinforcement Learning”
  https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c
- Patrick Loeber, snake-ai-pytorch
  https://github.com/patrickloeber/snake-ai-pytorch

## Clip

![Snake Demo](../../media/snake-demo.gif)

## Algorithm / Network

- Default algorithm: `qlearn`
- IO: `obs=12`, `act=3`
- Default Q-network: `12 -> 32 -> 3`

## Controls (Human)

- Move: `W/A/S/D`
- Rendered overlays: `X` toggles translucent `sens_*` ray ghosts during `play-user` and `play-ai`

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> SENS -> TGT`
- Observation: `12` floats (`INPUT_FEATURE_NAMES`, exact order)
  - `SELF` (5): `self_heading_sin self_heading_cos self_len_norm self_last_act_norm self_hunger_norm`
  - `SENS` (3): `sens_fwd sens_left sens_right`
  - `TGT` (4): `tgt_rel_angle_sin tgt_rel_angle_cos tgt_manhattan_norm tgt_dist_delta`
- Actions: `Discrete(3)`
  - `0 straight`
  - `1 turn_right`
  - `2 turn_left`

`sens_*` values are normalized free-space-before-collision measurements in the snake's local frame. `0.0` means a collision is adjacent; `1.0` means no collision is found within the probe range.

## Environment Notes

- `WRAP_AROUND` is a global toggle in `games/snake/config.py`.
- Obstacles are static for the duration of an episode and are excluded from food spawn placement.
- `FOOD_TIMEOUT_STEPS = 180`; hunger resets to `0` after food, and an episode times out if training goes that many steps without eating.

## Rewards (Training)

- `REWARD_FOOD = +1.0` when food is eaten
- `PENALTY_LOSE = -5.0` on death or timeout
- Progress shaping: `clip(1.0 * (Phi_next - Phi_prev), -0.05, +0.05)` where `Phi = -dist_food_norm - 0.5 * hunger_norm`
- `PENALTY_STEP = -0.01` every training step

`tgt_manhattan_norm` is normalized Manhattan head-to-food distance. When wrap-around is enabled it uses the shortest wrapped path. `self_hunger_norm` is `clamp(steps_since_food / FOOD_TIMEOUT_STEPS, 0, 1)`, so `0.0` means food was just eaten and `1.0` means the snake is about to starve.

## Curriculum (Train)

- Shared 5-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/snake/config.py` under `CURRICULUM_PROMOTION`
- Levels only increase obstacle count.
- Levels:
  - Level 1: `0` obstacles
  - Level 2: `2` obstacles
  - Level 3: `4` obstacles
  - Level 4: `6` obstacles
  - Level 5: `8` obstacles

An episode counts as a success if the snake eats at least `5` foods (`SUCCESS_FOODS_REQUIRED`).

## Run Commands

```bash
rl-toybox-train --game snake
rl-toybox-play-ai --game snake --render
rl-toybox-play-user --game snake
python -m scripts.train --game snake
python -m scripts.play_ai --game snake --render
python -m scripts.play_user --game snake
```

When `--level` is omitted, `train` starts at `L1` and `play-user` / `play-ai` default to `L5`.

See `games/snake/config.py` for the game constants, reward values, curriculum thresholds, and training defaults. Snake's game-wide net size lives in `DEFAULT_MODEL_CONFIG["hidden_sizes"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
