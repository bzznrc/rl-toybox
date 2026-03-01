# Vroom

Top-down one-lap racing environment with procedural closed-loop tracks.

## Algorithm / Network

- Algo: vanilla DQN
- Hidden sizes: `[48, 48]`

## Controls (Human)

- Steer: `A/D` or left/right arrows
- Throttle: `W` or up arrow
- Coast: no input

## Observation / Actions

- Observation: `20` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_lat_offset`
  - `self_lat_offset_delta`
  - `self_fwd_speed`
  - `self_fwd_speed_delta`
  - `self_heading_sin`
  - `self_heading_cos`
  - `self_in_contact`
  - `self_last_action`
  - `ray_fwd_near`
  - `ray_fwd_far`
  - `ray_fwd_left`
  - `ray_fwd_right`
  - `tgt_dx`
  - `tgt_dy`
  - `tgt_dvx`
  - `tgt_dvy`
  - `trk_lookahead_sin`
  - `trk_lookahead_cos`
  - `trk_lookahead_dist`
  - `trk_curvature_ahead`
- Actions: `Discrete(6)` (`ACTION_NAMES`, ordered)
  - `0 coast`
  - `1 throttle`
  - `2 left_coast`
  - `3 right_coast`
  - `4 left_throttle`
  - `5 right_throttle`

Ray semantics:

- `ray_*` are normalized distance-to-first-hit values in `[0,1]`
- `1.0` means no hit within ray range
- hits include walls and obstacles

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

- Outcome `REWARD_WIN`: `+10` when the player wins the race.
- Outcome `PENALTY_LOSE`: `-5` when another car wins or race timeout resolves against the player.
- Progress shaping: `r_progress = clip(1.0 * (Phi' - Phi), -0.2, +0.2)` with `Phi = track_progress_norm`.
- Event `PENALTY_COLLISION`: `-1.0` on collision-start events (transition from not-in-contact to in-contact).
- Step `PENALTY_STEP`: `-0.01` every training step.

`track_progress_norm` is normalized lap progress along the track/checkpoint ordering, so forward progress increases `Phi` and gives positive signed-ΔPhi shaping.

## Training

Default algo is vanilla DQN:

```bash
rl-toybox-train --game vroom
```

Key hyperparameters:

- Train: `max_steps=2_000_000`, `learn_start_steps=20_000`, `train_every_steps=1`, `updates_per_train=1`, `checkpoint_every_steps=100_000`
- Algo: `learning_rate=3e-4`, `gamma=0.99`, `batch_size=128`, `replay_size=200_000`, `target_sync_every_steps=2_000`, `grad_clip_norm=10.0`
- DQN mode: `double_dqn=False`, `dueling=False`, `prioritized_replay=False`
- Exploration: `eps_start=1.0`, `eps_min=0.05`, `eps_decay_steps=700_000`
- Plateau bump/cooldown: `avg_window=100`, `patience=200`, `min_improvement=0.10`, `eps_bump_cap=0.20`, `cooldown_steps=300_000`
- Stats/saving gate: rolling avg/best tracking, plateau checks, checkpoint saves, and best-model saves start after `100` completed episodes.

Exploration uses multiplicative epsilon decay per env step: `eps = max(eps_min, eps * eps_decay)`,
with `eps_decay = (eps_min / eps_start) ** (1.0 / eps_decay_steps)`.
On plateau, if `eps < 0.20`, bump `eps` to `0.20` and start cooldown for `300_000` steps.
During cooldown, additional bumps are blocked, but epsilon continues normal multiplicative decay.
This keeps exploration from snapping straight back to the minimum.

Play AI:

```bash
rl-toybox-play-ai --game vroom --model best --render
```

Play user:

```bash
rl-toybox-play-user --game vroom
```
