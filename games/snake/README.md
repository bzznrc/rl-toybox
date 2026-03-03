# Snake

Grid Snake environment with obstacles.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion parameters are configured in `games/snake/config.py` via `CURRICULUM_PROMOTION`:
  - `min_episodes_per_level`
  - `check_window`
  - `success_threshold`
  - `consecutive_checks_required`
- `WRAP_AROUND` is configured globally in `games/snake/config.py` (currently always enabled).
- Snake level settings:
  - Level 1: `0` obstacles, timeout `160 * snake_length`
  - Level 2: `4` obstacles, timeout `130 * snake_length`
  - Level 3: `8` obstacles, timeout `100 * snake_length`

Success (per episode): `1` if at least `5` foods were eaten (`SUCCESS_FOODS_REQUIRED`), else `0`.
Average Success (`AS`) is the rolling mean over the curriculum `check_window`.

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

Ray semantics:

- `ray_*` are normalized distance-to-first-collision in local snake directions.
- Values are in `[0,1]`; `1.0` means no collision within ray range.

## Rewards (Training)

- Event `REWARD_FOOD`: `+10` when food is eaten.
- Outcome `PENALTY_LOSE`: `-5` on death or timeout.
- Progress shaping: `r_progress = clip(1.0 * (Phi' - Phi), -0.2, +0.2)` with `Phi = -dist_food_norm - 0.5*hunger_norm`.
- Step `PENALTY_STEP`: `-0.005` every training step.

`dist_food_norm` is the normalized Manhattan head-to-food distance (shortest wrapped path when wrap-around is enabled).
`hunger_norm` is `clamp(self_steps_since_food / hunger_cap_steps, 0, 1)`.
Progress reward uses signed DeltaPhi with clipping, so moving toward food and reducing hunger pressure increases `Phi`.

## Training

Default algo is Q-learning with `LinearQNet`:

```bash
rl-toybox-train --game snake
```

Key hyperparameters:

- Train: `max_steps=1_500_000`, `checkpoint_every_steps=100_000`
- Algo: `learning_rate=1e-3`, `gamma=0.95`, `max_memory=100_000`, `batch_size=512`
- Exploration: `eps_start=1.0`, `eps_min=0.05`, `eps_decay_steps=300_000`
- Plateau bump/cooldown: `avg_window=100`, `patience=200`, `min_improvement=0.10`, `eps_bump_cap=0.20`, `cooldown_steps=200_000`
- Stats/saving gate: rolling avg/best tracking, plateau checks, checkpoint saves, and best-model saves start after `100` completed episodes.

Exploration uses multiplicative epsilon decay per env step: `eps = max(eps_min, eps * eps_decay)`,
with `eps_decay = (eps_min / eps_start) ** (1.0 / eps_decay_steps)`.
On plateau, if `eps < 0.20`, bump `eps` to `0.20` and start cooldown for `200_000` steps.
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

## Episode Log Fields

Training episode logs use compact tab-separated fields:

`Ep:<ep>\tLv:<level>\tLen:<len>\tR:<reward>\tAR:<avg_reward|n/a>\tBR<level>:<best_avg|n/a>\tE:<epsilon|n/a>\tS:<0/1>\tAS:<avg_success|n/a>\t<components>`

- `AR`, `BR<level>`, `AS` are level-scoped and shown as `n/a` until the minimum stats gate is met (`100` episodes) for that level.
- Reward components are appended as one space-separated blob (for Snake: `F L P S`).
