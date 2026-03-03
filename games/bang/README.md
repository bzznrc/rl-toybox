# Bang

Top-down arena shooter environment.

## Algorithm / Network

- Algo: enhanced DQN (double + dueling + prioritized replay)
- Hidden sizes: `[64, 64]`

## Controls (Human)

- Move: `W/A/S/D` (`move_up/move_left/move_down/move_right`)
- Aim: left/right arrows (`aim_left/aim_right`)
- Shoot: `Space` (`shoot`)
- If no `W/A/S/D` movement key is pressed in a frame, movement uses `move_stop`.

## Observation / Actions

- Observation: `24` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_angle_sin`
  - `self_angle_cos`
  - `self_move_intent_x`
  - `self_move_intent_y`
  - `self_shot_cd_norm`
  - `self_tgt_seen_norm`
  - `ray_fwd`
  - `ray_left`
  - `ray_right`
  - `ray_back`
  - `tgt_dx`
  - `tgt_dy`
  - `tgt_dvx`
  - `tgt_dvy`
  - `tgt_dist`
  - `tgt_in_los`
  - `tgt_rel_angle_sin`
  - `tgt_rel_angle_cos`
  - `haz_dx`
  - `haz_dy`
  - `haz_dvx`
  - `haz_dvy`
  - `haz_dist`
  - `haz_in_trajectory`
- Actions: `Discrete(8)` (`ACTION_NAMES`, ordered)
  - `0 move_up`
  - `1 move_down`
  - `2 move_left`
  - `3 move_right`
  - `4 move_stop`
  - `5 aim_left`
  - `6 aim_right`
  - `7 shoot`
- In human and training (`NN`) control, aim is per-step (non-sticky): heading rotates only on steps where action is `aim_left`/`aim_right`.

Ray semantics:

- `ray_*` are normalized distance-to-first-hit values in `[0,1]`
- `1.0` means no hit within ray range
- hits include arena walls and obstacles

Notes:

- `self_shot_cd_norm` is remaining shoot cooldown normalized to `[0,1]` (`cooldown_remaining / SHOOT_COOLDOWN_FRAMES`).
- `tgt_rel_angle_sin/cos` encode relative target bearing from current aim direction (stable angular signal for aiming).

## Rewards (Training)

- Outcome `REWARD_WIN`: `+10` on match win.
- Outcome `PENALTY_LOSE`: `-5` on match loss.
- Event `REWARD_KILL`: `+2` per enemy elimination.
- Engagement shaping: `r_eng = clip(0.2 * (Phi_eng' - Phi_eng), -0.1, +0.1)`, `Phi_eng = (1 if tgt_in_los else 0) - tgt_dist_norm`.
- Hazard shaping: `r_haz = clip(0.2 * (Phi_haz' - Phi_haz), -0.1, +0.1)`, `Phi_haz = haz_dist_norm - 1.5 * haz_in_trajectory`.
- Step `PENALTY_STEP`: `-0.005` every training step.
- `ENGAGEMENT_CLIP` / `HAZARD_CLIP` are shaping clamp parameters only; they are not standalone reward components.

The signed-ΔPhi terms are clipped and small, so terminal outcomes stay dominant while still rewarding better engagement and safer projectile states.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion parameters are configured in `games/bang/config.py` via `CURRICULUM_PROMOTION`:
  - `min_episodes_per_level`
  - `check_window`
  - `success_threshold`
  - `consecutive_checks_required`
- Bang level settings:
  - Level 1: `2` players, low obstacles, easy enemy behavior
  - Level 2: `2` players, medium obstacles, medium enemy behavior
  - Level 3: `4` players, high obstacles, hard enemy behavior

Success (per episode): `1` on match win, else `0`.
Average Success (`AS`) is the rolling mean over the curriculum `check_window`.

## Training

Default algo is enhanced DQN (double + dueling + prioritized replay):

```bash
rl-toybox-train --game bang
```

Key hyperparameters:

- Train: `max_steps=10_000_000`, `train_after_steps=50_000`, `update_every_steps=4`, `updates_per_step=1`, `checkpoint_every_steps=200_000`
- Algo: `learning_rate=2.5e-4`, `gamma=0.99`, `batch_size=256`, `replay_size=500_000`, `target_sync_every_steps=10_000`, `grad_clip_norm=10.0`
- DQN mode: `double_dqn=True`, `dueling=True`, `prioritized_replay=True`
- PER: `per_alpha=0.6`, `per_beta_start=0.4`, `per_beta_frames=10_000_000`, `per_epsilon=1e-4`
- Exploration: `eps_start=1.0`, `eps_min=0.05`, `eps_decay_steps=2_500_000`
- Plateau bump/cooldown: `avg_window=100`, `patience=300`, `min_improvement=0.10`, `eps_bump_cap=0.25`, `cooldown_steps=1_500_000`
- Stats/saving gate: rolling avg/best tracking, plateau checks, checkpoint saves, and best-model saves start after `100` completed episodes.

Exploration is multiplicative per env step: `eps = max(eps_min, eps * eps_decay)`,
with `eps_decay = (eps_min / eps_start) ** (1.0 / eps_decay_steps)`.
On plateau, if `eps < 0.25`, bump `eps` to `0.25` and start cooldown for `1_500_000` steps.
During cooldown, additional bumps are blocked, but epsilon continues normal multiplicative decay.
This cooldown window prevents rapid repeat bumps while preserving exploration decay.

Play AI:

```bash
rl-toybox-play-ai --game bang --model best --render
```

Play user:

```bash
rl-toybox-play-user --game bang
```

## Episode Log Fields

Training episode logs use compact tab-separated fields:

`Ep:<ep>\tLv:<level>\tLen:<len>\tR:<reward>\tAR:<avg_reward|n/a>\tBR<level>:<best_avg|n/a>\tE:<epsilon|n/a>\tS:<0/1>\tAS:<avg_success|n/a>\t<components>`

- `AR`, `BR<level>`, `AS` are level-scoped and shown as `n/a` until the minimum stats gate is met (`100` episodes) for that level.
- Reward components are appended as one space-separated blob (for Bang: `W L K E D S`).
