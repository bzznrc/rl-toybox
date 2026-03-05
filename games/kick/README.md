# Kick

Arcade-style top-down football with shared-policy PPO across the full left team.

## Algorithm / Network

- Algo: PPO (parameter-shared across left-team players)
- Hidden sizes: `[96, 96]`

## Controls (Human)

- Move: `W/A/S/D` (diagonals via combinations, for example `W` + `D`)
- Shoot: hold/release `Space` (low/mid/high by hold time)
- Human mode keeps player switching behavior; RL mode does not switch and controls all left players each step.

## Observation / Actions

- Observation:
  - RL mode (`train` / `eval`): `(N_left, 36)` where each row is one left-player feature vector.
  - Human mode: single `(36,)` vector for the currently controlled player.
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
  - `goal_dx`
  - `goal_dy`
  - `goal_rel_angle_sin`
  - `goal_rel_angle_cos`
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
  - RL mode expects one action per left player each step: `(N_left,)`
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
- `goal_dx/goal_dy` is the player-to-opponent-goal-center vector.
- `goal_rel_angle_sin/cos` encodes the relative goal angle from player heading.
- Kick actions only apply if that player owns the ball.

## Rewards (Training)

Kick uses five active components:

- `G` score: `REWARD_SCORE = +10` when left scores.
- `C` concede: `PENALTY_CONCEDE = -5` when left concedes.
- `T` turnover: `PENALTY_TURNOVER = -0.5` only on direct `left -> right` possession transfer in a step, and not on goal/reset transitions.
- `P` progress: during left possession, reward only when the ball achieves a new best (minimum) distance to opponent goal center:
  `r_progress = REWARD_PROGRESS * (improvement / PROGRESS_NORM)`
- `Z` zone: per-step positioning penalty
  `r_zone = PENALTY_ZONE * zone_norm`

Zone anchors use a simplified tactical axis (0 = own goal, 100 = opponent goal):
- Base anchors: `GK=5`, `DEF=25`, `MID=45`, `ATK=65`
- Possession phase shift: `+10` for `DEF/MID/ATK` when left has possession, `-10` when not
- Ball shift: `(ball_y - 50) * 0.1`
- Final anchor clamped to `[5, 85]` for `DEF/MID/ATK`; `GK` stays at `5`

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/kick/config.py` under `CURRICULUM_PROMOTION`.
- Left team stays at 11 RL-controlled players.
- Opponent scaling by level:
  - Level 1: `11v3` (`RM, LM, ST1`)
  - Level 2: `11v7` (`GK, LB, RB, RM, LM, ST1, ST2`)
  - Level 3: `11v11`
- Kickoff behavior:
  - Your team uses the level `kickoff` setting (`GK` or `CC`).
  - Opponent restarts always begin from center circle (`CC`).

Success per episode is `1` when left scores more than it concedes, else `0`.

## Run Commands

```bash
rl-toybox-train --game kick --algo ppo
rl-toybox-play-ai --game kick --model best --render
rl-toybox-play-user --game kick
```

Check `games/kick/config.py` for full PPO settings and curriculum parameters.
