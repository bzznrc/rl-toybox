# Kick

Arcade-style top-down football with shared-team MAPPO training (CTDE): one shared actor for all LEFT players, centralized critic during training.

## Algorithm / Network

- Algo: PPO with MAPPO-style training (shared/decentralized actor + centralized critic)
- Actor MLP: `[128, 128]`
- Critic MLP: `[256, 256]`
- Actor output: `Discrete(12)` logits per LEFT player
- Critic input (per agent): centralized state + that agent's local observation

## Controls (Human)

- Move: `W/A/S/D` (diagonals via combinations, for example `W` + `D`)
- Shoot: hold/release `Space` (low/mid/high by hold time)
- Human mode keeps player switching behavior; RL mode controls all LEFT players each step.

## Observation / Actions

- Observation:
  - RL mode (`train` / `eval`): `(N_left, 48)` where each row is one LEFT-player feature vector.
  - Human mode: single `(48,)` vector for the currently controlled player.
- Feature blocks (ordered):
  - `SELF` (9): `self_vx self_vy self_theta_cos self_theta_sin self_has_ball self_role self_role_lane self_stamina self_stamina_delta`
  - `BALL` (7): `tgt_dx tgt_dy tgt_rel_angle_sin tgt_rel_angle_cos tgt_dvx tgt_dvy tgt_owner_team`
  - `GOAL (opponent)` (4): `goal_dx goal_dy goal_rel_angle_sin goal_rel_angle_cos`
  - `GOAL (own)` (4): `own_goal_dx own_goal_dy own_goal_rel_angle_sin own_goal_rel_angle_cos`
  - `ALLIES 1..3` (12): `ally{k}_{dx,dy,dvx,dvy}`
  - `FOES 1..3` (12): `foe{k}_{dx,dy,dvx,dvy}`
- Nearest ally/foe selection is deterministic and stable: sorted by `(distance, player.slot_index)`.
- Actions: `Discrete(12)` (`ACTION_NAMES`, ordered)
  - RL mode expects one action per LEFT player each step: `(N_left,)`
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

## Possession Semantics

- Physical owner:
  - `ball_owner_team`, `ball_owner_id` (stable slot id) from `ball_owner` (or `None` when free/in flight)
- Effective possession team:
  - `possession_team = ball_owner_team if owned else last_touch_team`
- Formation/zone phase uses effective possession (prevents snapping while ball is in flight).
- Progress shaping is based on physical ball motion and credited to a responsible LEFT player:
  - controlled ball: current LEFT owner
  - free ball after LEFT touch: `last_touch_player_id`
- Centralized critic state is fixed-size and team-size robust:
  - padded `central_obs` for up to `MAX_LEFT_PLAYERS=11`
  - `central_mask` marks present/padded LEFT slots

## Action Masking

- If `self_has_ball == 0`, `kick_low/mid/high` are invalid.
- Masking is applied in both training and eval.
- Eval policy selection uses masked argmax, so invalid kicks are not chosen.

## Rewards (Training)

Realized components logged as `G C T A P Z`:

- `G` outcome score: total `+10.0` normalized per player (`+10 / n_left` each).
- `C` outcome concede: total `-5.0` normalized per player (`-5 / n_left` each).
- `T` turnover: `-0.25` to the responsible LEFT player when possession changes LEFT -> RIGHT and RIGHT becomes physical owner.
  - Opponent GK catch in/near their box is excluded.
- `A` pass: `+0.25` to passer when a LEFT kick is next physically controlled by a different LEFT player.
- `P` progress (dense, player-specific):
  - `delta = prog_next - prog_prev` using ball depth toward opponent goal
  - `rP = 2.0 * clamp(delta, -0.01, +0.01)`
  - credited to responsible LEFT player (owner, or last toucher while free)
  - progress baseline resets on true RIGHT possession/reset, not LEFT -> free -> LEFT flight.
- `Z` zone discipline (per-player, no team averaging before assignment):
  - `d_i = normalized distance to role anchor (depth + lane)`
  - `excess = max(0, d_i - 0.05)`
  - `rZ_i = -0.01 * (excess^2)`

## Step Contract

- `env.step(...)` returns scalar team reward (`float`, sum of per-player rewards).
- Per-player rewards are always in `info["reward_vec"]` with length `N_left`.
- Reward breakdown in `info["reward_breakdown"]` contains realized step contributions by component key.

## PPO Debug Line

- After each episode line, training prints one extra tab-separated PPO line:
  - `PPO	PolicyLoss: ...	ValueLoss: ...	Entropy: ...	ApproxKl: ...	ClipFrac: ...`
- Before the first PPO update, these fields print as `n/a`.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/kick/config.py` under `CURRICULUM_PROMOTION`.
- LEFT team stays at 11 RL-controlled players.
- Opponent scaling by level:
  - Level 1: `11v3` (`RM, LM, LCS`)
  - Level 2: `11v7` (`GK, LB, RB, RM, LM, LCS, RCS`)
  - Level 3: `11v11` (`GK, LB, LCB, RCB, RB, LM, LCM, RCM, RM, LCS, RCS`)

Success per episode is `1` when LEFT scores more than it concedes, else `0`.

## Sanity Flag

- `KICK_DEBUG_SANITY=1` enables quick runtime checks for:
  - RL obs shape `(N_left, 48)`
  - stable nearest ordering for ally/foe blocks
  - masked invalid-kick prevention in eval
  - GK-catch turnover exclusion
  - scalar step reward + `reward_vec` length

## Run Commands

```bash
rl-toybox-train --game kick --algo ppo
rl-toybox-play-ai --game kick --model best --render
rl-toybox-play-user --game kick
```

Check `games/kick/config.py` for full PPO/MAPPO settings and curriculum parameters.
