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
  - `self_role_lane`
  - `self_stamina`
  - `self_stamina_delta`
  - `tgt_dx`
  - `tgt_dy`
  - `tgt_rel_angle_sin`
  - `tgt_rel_angle_cos`
  - `tgt_dvx`
  - `tgt_dvy`
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
- `self_role_lane` uses 4 tactical lanes: left=`-1.0`, center-left=`-0.25`, center-right=`+0.25`, right=`+1.0`; `GK=0.0`.
- Kick tracks both physical owner (`ball_owner`) and effective possession (`ball_owner.team` if owned, else `last_touch_team` while free). `tgt_owner_team` reflects effective possession (`+1` left, `-1` right, `0` neutral reset state).
- During PPO training, kicks are action-masked out for players without ball possession.
- Eval/play uses the same action masking; turnover penalties exclude opponent GK catches in/near their penalty area.
- `env.step()` returns a scalar team reward (sum of per-player rewards); PPO training reads per-player rewards from `info["reward_vec"]`.

## Rewards (Training)

Kick uses per-player rewards with a shared policy (one row per left player). Team logs still show summed components.

Kick uses six active components:

- `G` score: `REWARD_SCORE = +10` when left scores.
- `C` concede: `PENALTY_CONCEDE = -5` when left concedes.
- `T` turnover: `PENALTY_TURNOVER = -0.25`, credited only to the responsible left player (last left physical owner, or pending passer if intercepted in flight), excluding opponent GK catches in/near their penalty area.
- `A` pass: `REWARD_PASS = +0.25`, credited only to the passer when the next owner is a different left player.
- `P` progress: computed only under left physical ball control, credited only to the current left ball owner:
  `r_progress = REWARD_PROGRESS * (improvement / PROGRESS_NORM)`
- `Z` zone: per-step per-player positioning penalty (no team-average before assignment) with dead-zone and quadratic growth
  `excess = max(0, zone_norm - Z_TOL)`
  `r_zone = PENALTY_ZONE * (excess^2)`

Zone norm is depth-only on the tactical X axis (`x -> [0,100]`):
- Base anchors by role-group: `GK=5`, `DEF=25`, `MID=45`, `ATK=65`
- Phase shift: `+10` in possession, `-10` out of possession (using effective possession for formation phase)
- Ball-depth shift: `clip((ball_depth_y - 50) * 0.1, -5, 5)`
- Formation anchors are smoothed over 2.5s.

## Curriculum (Train)

- Shared 3-level curriculum progression (`core/curriculum.py`) is used in train mode.
- Promotion settings live in `games/kick/config.py` under `CURRICULUM_PROMOTION`.
- Left team stays at 11 RL-controlled players.
- Opponent scaling by level:
  - Level 1: `11v3` (`RM, LM, LCS`)
  - Level 2: `11v7` (`GK, LB, RB, RM, LM, LCS, RCS`)
  - Level 3: `11v11` (`GK, LB, LCB, RCB, RB, LM, LCM, RCM, RM, LCS, RCS`)
- Kickoff behavior:
  - Kickoffs always begin from center circle (`CC`) with no immediate owner.

Success per episode is `1` when left scores more than it concedes, else `0`.

## Run Commands

```bash
rl-toybox-train --game kick --algo ppo
rl-toybox-play-ai --game kick --model best --render
rl-toybox-play-user --game kick
```

Check `games/kick/config.py` for full PPO settings and curriculum parameters.
