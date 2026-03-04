# Kick

Football environment (Arcade-style top-down) with PPO-oriented discrete actions.

## Algorithm / Network

- Algo: PPO (parameter-shared)
- Hidden sizes: `[96, 96]`

## Controls (Human)

- Move: `W/A/S/D` (diagonals via combinations, e.g. `W`+`D` -> `move_ne`)
- Shoot: hold/release `Space` (low/mid/high by hold time)
- Controlled player auto-switches: left ball-owner when in possession, otherwise closest left player to the ball.
- Facing direction follows movement direction.

## Observation / Actions

- Observation:
  - RL mode (`train` / `eval`): `(N_left, 36)` where each row is one left-player feature vector.
  - Human mode: single `(36,)` feature vector for the currently controlled player.
  - Per-player feature schema (`INPUT_FEATURE_NAMES`, ordered):
  - `self_vx`
  - `self_vy`
  - `self_theta_cos`
  - `self_theta_sin`
  - `self_has_ball`
  - `self_role`
  - `self_stamina`
  - `self_stamina_delta`
  - `tgt_dx`
  - `tgt_dy`
  - `tgt_rel_angle_sin`
  - `tgt_rel_angle_cos`
  - `tgt_dvx`
  - `tgt_dvy`
  - `tgt_is_free`
  - `tgt_owner_team`
  - `goal_opp_dx`
  - `goal_opp_dy`
  - `goal_own_dx`
  - `goal_own_dy`
  - `ally1_dx`
  - `ally1_dy`
  - `ally1_dvx`
  - `ally1_dvy`
  - `ally2_dx`
  - `ally2_dy`
  - `ally2_dvx`
  - `ally2_dvy`
  - `foe1_dx`
  - `foe1_dy`
  - `foe1_dvx`
  - `foe1_dvy`
  - `foe2_dx`
  - `foe2_dy`
  - `foe2_dvx`
  - `foe2_dvy`
- Actions: `Discrete(12)` (`ACTION_NAMES`, ordered)
  - RL mode applies one discrete action per left player each step: action vector shape `(N_left,)`.
  - `0 stay`
  - `1 move_n`
  - `2 move_ne`
  - `3 move_e`
  - `4 move_se`
  - `5 move_s`
  - `6 move_sw`
  - `7 move_w`
  - `8 move_nw`
  - `9 kick_low`
  - `10 kick_mid`
  - `11 kick_high`

Notes:

- Movement actions imply no kick.
- Kick actions only apply when the acting player has the ball; otherwise treated as `STAY`.
- `self_role` is a role-group scalar for the shared policy:
  - `GK = -1.0`
  - `DEF = 0.0`
  - `MID = 0.5`
  - `ATK = 1.0`
  - Detailed role mapping: `GK -> GK`, `LB/LCB/RCB/RB -> DEF`, `LM/LCM/RCM/RM -> MID`, `ST1/ST2 -> ATK`.
- Goalkeeper catch permeability:
  - Base keeper catch probability is positional (`1 - |ball_y - keeper_y| / goal_half_height`) after trajectory/box/range checks.
  - If a keeper would catch and the last kick was `kick_high`, catch can be bypassed with fixed probability `GK_HIGH_BYPASS_PROB_DEFAULT=0.25`.
  - This bypass applies only to goalkeepers, not other players.

## Rewards (Training)

Kick uses 5 reward terms (actual contributions):

- Outcome `REWARD_SCORE`: `+10` when left team scores.
- Outcome `PENALTY_CONCEDE`: `-5` when left team concedes.
- Event turnover penalty (`T`): `PENALTY_TURNOVER=-0.5` only on direct possession transfer `left -> right` within a step, and only when no goal event happened that step.
  Not counted as turnover: `left -> free`, `free -> left`, goal/reset transitions.
- Progress reward (`P`): while left has possession, reward only NEW maximum forward ball position:
  `r_prog = REWARD_PROGRESS * (delta_max / 100) = 2.5 * (delta_max / 100)`.
  Here `delta_max` is the increase in possession-local max forward ball position; if possession is lost, progress reward is `0` until possession is regained.
- Zone positioning penalty (`Z`): `r_zone = PENALTY_ZONE * zone_norm` with `PENALTY_ZONE=-0.0005` (applied every step).

Forward position is measured on a 0..100 axis (`0 = own goal`, `100 = opponent goal`) from internal coordinates.

Zone anchors use a simplified tactical axis (0 = own goal, 100 = opponent goal):
- Base anchors: `GK=5`, `DEF=25`, `MID=45`, `ATK=65`.
- Possession phase shift: `+10` for `DEF/MID/ATK` when left has possession, `-10` when not; `GK` remains fixed at `5`.
- Ball shift: `ball_shift = (ball_y - 50) * 0.1` (clamped to `[-5,+5]`).
- Final anchor: `anchor_y = clamp(base + phase_shift + ball_shift, 5, 85)` for `DEF/MID/ATK`; `GK=5`.
- Per-player distance: `dy_norm_i = abs(player_y - anchor_y_i) / 100`.
- `zone_norm = mean(dy_norm_i)` over all controlled left players (including ball owner).

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion parameters are configured in `games/kick/config.py` via `CURRICULUM_PROMOTION`:
  - `min_episodes_per_level`
  - `check_window`
  - `success_threshold`
  - `consecutive_checks_required`
- Kick level settings:
  - Level 1: `3v3`, `goals_size_scale=2.0` (own goal `1/2x`, opponent goal `2x`), enemy stamina max `0%`, enemy shot error choices `[-30, -20, 0, 20, 30]`.
  - Level 2: `7v7`, `goals_size_scale=1.5` (own goal `1/1.5x`, opponent goal `1.5x`), enemy stamina max `50%`, enemy shot error choices `[-20, -10, 0, 10, 20]`.
  - Level 3: `11v11`, `goals_size_scale=1.0` (both normal), enemy stamina max `100%`, enemy shot error choices `[-10, 0, 10]`.

Success (per episode): `1` if left team scores more than it concedes, else `0`.
Average Success (`AS`) is the rolling mean over the curriculum `check_window`.

## Training

Default algo is PPO (categorical policy):

```bash
rl-toybox-train --game kick --algo ppo
```

Key hyperparameters:

- Train: `max_iterations=1500`, `rollout_steps=2048`, `checkpoint_every_iterations=10`, `reward_window=100`, `min_episodes_for_stats=100`
- Algo: `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_ratio=0.2`, `update_epochs=4`, `minibatch_size=512`, `entropy_coef=0.01`, `value_coef=0.5`, `max_grad_norm=0.5`

Play AI:

```bash
rl-toybox-play-ai --game kick --model best --render
```

Play user:

```bash
rl-toybox-play-user --game kick
```

## Episode Log Fields

Training episode logs use compact tab-separated fields:

`Ep:<ep>\tLv:<level>\tLen:<len>\tR:<reward>\tAR:<avg_reward|n/a>\tBR<level>:<best_avg|n/a>\tE:<epsilon|n/a>\tS:<0/1>\tAS:<avg_success|n/a>\t<components>`

- `AR`, `BR<level>`, `AS` are level-scoped and shown as `n/a` until the minimum stats gate is met (`100` episodes) for that level.
- Reward components are appended as one space-separated blob (Kick: `G C T P Z`; `T` is turnover penalty).
