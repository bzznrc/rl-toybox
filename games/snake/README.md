# Snake

Grid Snake environment with obstacles.

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
  - `tgt_dx`
  - `tgt_dy`
  - `tgt_manhattan_dist`
  - `tgt_dist_delta`
  - `self_steps_since_food`
- Actions: `Discrete(3)`
  - `0 straight`
  - `1 turn_right`
  - `2 turn_left`

Ray semantics:

- `ray_*` are normalized distance-to-first-collision in local snake directions.
- Values are in `[0,1]`; `1.0` means no collision within ray range.

## Rewards (Training)

- Event `REWARD_FOOD`: `+10` when food is eaten.
- Outcome `PENALTY_LOSE`: `-5` on death or timeout.
- Progress shaping: `r_progress = clip(1.0 * (Phi' - Phi), -0.2, +0.2)` with `Phi = -dist_food_norm`.
- Step `PENALTY_STEP`: `-0.01` every training step.

`dist_food_norm` is the normalized Manhattan head-to-food distance (shortest wrapped path when wrap-around is enabled), so moving toward food increases `Phi` and gives positive signed-ΔPhi shaping.

## Training

Default algo is Q-learning with `LinearQNet`:

```bash
rl-toybox-train --game snake
```

Key hyperparameters:

- Train: `max_steps=1_500_000`, `checkpoint_every_steps=100_000`
- Algo: `learning_rate=1e-3`, `gamma=0.95`, `max_memory=100_000`, `batch_size=512`
- Exploration: `eps_start=1.0`, `eps_min=0.05`, `eps_decay_steps=900_000`
- Plateau bump/cooldown: `avg_window=100`, `patience=50`, `min_improvement=0.10`, `eps_bump_cap=0.25`, `cooldown_steps=50_000`

Exploration uses multiplicative epsilon decay per env step: `eps = max(eps_min, eps * eps_decay)`,
with `eps_decay = (eps_min / eps_start) ** (1.0 / eps_decay_steps)`.
On plateau, if `eps < 0.25`, bump `eps` to `0.25` and start cooldown for `50_000` steps.
During cooldown, additional bumps are blocked, but epsilon continues normal multiplicative decay.
This avoids immediate collapse back to minimum exploration during overnight runs.

Play AI:

```bash
rl-toybox-play-ai --game snake --model best --render
```

Play user:

```bash
rl-toybox-play-user --game snake
```
