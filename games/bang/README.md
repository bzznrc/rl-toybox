# Bang

Top-down arena shooter environment.

## Algorithm / Network

- Algo: enhanced DQN (double + dueling + prioritized replay)
- Hidden sizes: `[64, 64]`

## Controls (Human)

- Move: `W/A/S/D`
- Aim: mouse (preferred) or `Q/E` (also left/right arrows)
- Shoot: `Space` or left mouse button

## Observation / Actions

- Observation: `24` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_angle_sin`
  - `self_angle_cos`
  - `self_move_intent_x`
  - `self_move_intent_y`
  - `self_aim_intent`
  - `self_last_action`
  - `self_time_since_shot`
  - `self_time_since_tgt_seen`
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

Ray semantics:

- `ray_*` are normalized distance-to-first-hit values in `[0,1]`
- `1.0` means no hit within ray range
- hits include arena walls and obstacles

## Rewards (Training)

- Outcome `REWARD_WIN`: `+10` on match win.
- Outcome `PENALTY_LOSE`: `-5` on match loss.
- Event `REWARD_KILL`: `+2` per enemy elimination.
- Engagement shaping: `r_eng = clip(0.2 * (Phi_eng' - Phi_eng), -0.1, +0.1)`, `Phi_eng = (1 if tgt_in_los else 0) - tgt_dist_norm`.
- Hazard shaping: `r_haz = clip(0.2 * (Phi_haz' - Phi_haz), -0.1, +0.1)`, `Phi_haz = haz_dist_norm - 1.5 * haz_in_trajectory`.
- Step `PENALTY_STEP`: `-0.005` every training step.

The signed-ΔPhi terms are clipped and small, so terminal outcomes stay dominant while still rewarding better engagement and safer projectile states.

## Training

Default algo is enhanced DQN (double + dueling + prioritized replay):

```bash
rl-toybox-train --game bang
```

Key hyperparameters:

- Train: `max_steps=10_000_000`, `learn_start_steps=50_000`, `train_every_steps=4`, `updates_per_train=1`, `checkpoint_every_steps=200_000`
- Algo: `learning_rate=2.5e-4`, `gamma=0.99`, `batch_size=256`, `replay_size=500_000`, `target_sync_every_steps=10_000`, `grad_clip_norm=10.0`
- DQN mode: `double_dqn=True`, `dueling=True`, `prioritized_replay=True`
- PER: `per_alpha=0.6`, `per_beta_start=0.4`, `per_beta_frames=10_000_000`, `per_epsilon=1e-4`
- Exploration: `eps_start=1.0`, `eps_min=0.05`, `eps_decay=0.9999995007`
- Plateau bump/hold: `avg_window=100`, `patience=30`, `min_improvement=0.20`, bump to `eps>=0.20`, `hold_steps=50_000`, `cooldown_episodes=30`

Exploration is multiplicative per env step: `eps = max(eps_min, eps * eps_decay)`.
If rolling avg reward plateaus, epsilon is bumped and held before decay resumes.
The hold window prevents repeated rapid drops back to minimum epsilon.

Play AI:

```bash
rl-toybox-play-ai --game bang --model best --render
```

Play user:

```bash
rl-toybox-play-user --game bang
```
