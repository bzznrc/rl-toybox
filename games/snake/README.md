# Snake

Classic grid Snake with obstacle curriculum and lightweight shaping rewards.

## Clip

[![Snake Demo](../../media/snake-demo.gif)](../../media/snake-demo.mp4)

## Algorithm / Network

- Algo: Q-learning (`qlearn`)
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

Ray notes:
- `ray_*` are normalized free-space-before-collision values in local snake directions.
- Values are in `[0,1]`; `0.0` means collision on the adjacent cell and `1.0` means no collision within ray range.

## Environment Notes

### Board Rules

- `WRAP_AROUND` is a global toggle in `games/snake/config.py`.
- Obstacles are static within an episode and excluded from food spawn placement.
- The timeout budget scales with current snake length.

## Rewards (Training)

- Event `REWARD_FOOD`: `+1.0` when food is eaten.
- Outcome `PENALTY_LOSE`: `-5` on death or timeout.
- Progress shaping: `r_progress = clip(1.0 * (Phi_next - Phi_prev), -0.05, +0.05)` with `Phi = -dist_food_norm - 0.5*hunger_norm`.
- Step `PENALTY_STEP`: `-0.005` every training step.

`dist_food_norm` is normalized Manhattan head-to-food distance (shortest wrapped path when wrap-around is enabled).
`hunger_norm` is `clamp(self_steps_since_food / hunger_cap_steps, 0, 1)`.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/snake/config.py` under `CURRICULUM_PROMOTION`.
- Levels:
  - Level 1: `0` obstacles, timeout `120 * snake_length`
  - Level 2: `6` obstacles, timeout `100 * snake_length`
  - Level 3: `12` obstacles, timeout `80 * snake_length`

Success per episode is `1` if at least `5` foods are eaten (`SUCCESS_FOODS_REQUIRED`), else `0`.

## Run Commands

```bash
rl-toybox-train --game snake
rl-toybox-play-ai --game snake --model best --render
rl-toybox-play-user --game snake
python -m scripts.train --game snake
python -m scripts.play_ai --game snake --model best --render
python -m scripts.play_user --game snake
```

Check `games/snake/config.py` for full hyperparameters and thresholds.
