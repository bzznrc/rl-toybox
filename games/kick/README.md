# Kick

`kick` is the repo's single football environment. It supports `3v3`, `5v5`, and `7v7` modes under one public game id, with the same simplified design in every mode: no roles, no goalkeeper slots, no stamina, no map anchors, no shot-quality flag, and one semantic `kick` action. There is no separate 7v7 football game id.

## Clip

![Kick Demo](../../media/kick-demo.gif)

## Algorithm / Network

- Default algorithm: `ppo`
- Modes: `3v3`, `5v5`, `7v7`
- Per-player actor IO: `obs=36`, `act=10`
- Shared actor: `36 -> 64 -> 64 -> 10`
- Training-only coach critic: `128 -> 128 -> 128 -> 1`
- Shared run tag: `a64_64_c128_128`
- All active LEFT players share one policy.

## Controls (Human)

- Move: `W/A/S/D`
- Kick: `Space` or `Enter`
- Human mode keeps automatic player switching behavior: LEFT ball owner first, otherwise closest LEFT player to the ball.
- Idle LEFT teammates use the same dynamic scripted logic as RIGHT opponents.

## Observation / Actions

- Observation family: `SELF -> TGT -> LAND -> ALLY -> OPP`
- `TGT` means the ball.
- RL mode returns one row per active LEFT player: `(3, 36)`, `(5, 36)`, or `(7, 36)`.
- Human mode returns one `(36,)` vector for the currently controlled LEFT player.
- `ally1` / `ally2` are the nearest two active LEFT teammates, sorted by `(distance_to_observer, slot_index)`.
- `opp1` / `opp2` / `opp3` are the nearest three active RIGHT opponents, sorted by `(distance_to_observer, slot_index)`.
- Missing nearest slots are zero-padded.

Feature order:

1. `self_x_norm`
2. `self_y_norm`
3. `self_vx`
4. `self_vy`
5. `self_theta_cos`
6. `self_theta_sin`
7. `self_has_ball`
8. `tgt_dx`
9. `tgt_dy`
10. `tgt_dist_norm`
11. `tgt_rel_ang_sin`
12. `tgt_rel_ang_cos`
13. `tgt_dvx`
14. `tgt_dvy`
15. `tgt_owner_left`
16. `tgt_owner_right`
17. `land_opp_goal_dx`
18. `land_opp_goal_dy`
19. `land_own_goal_dx`
20. `land_own_goal_dy`
21. `ally1_dx`
22. `ally1_dy`
23. `ally2_dx`
24. `ally2_dy`
25. `opp1_dx`
26. `opp1_dy`
27. `opp1_dvx`
28. `opp1_dvy`
29. `opp2_dx`
30. `opp2_dy`
31. `opp2_dvx`
32. `opp2_dvy`
33. `opp3_dx`
34. `opp3_dy`
35. `opp3_dvx`
36. `opp3_dvy`

`tgt_owner_left` is `1.0` while LEFT has the ball, and `tgt_owner_right` is `1.0` while RIGHT has the ball. A free ball sets both owner flags to `0.0`; `self_has_ball` is `1.0` only for the actual carrier.

Actions: `Discrete(10)` (`ACTION_NAMES`, ordered)

- `0 stay`
- `1 move_n`
- `2 move_ne`
- `3 move_e`
- `4 move_se`
- `5 move_s`
- `6 move_sw`
- `7 move_w`
- `8 move_nw`
- `9 kick`

`kick` is valid only for the current ball owner. Action masking is applied in training and evaluation, with mask shape `(active_left_players, 10)`.

## Environment Notes

### Team Sizes

- `TEAM_SIZE_CHOICES = (3, 5, 7)`
- Default team size is `3`.
- Active LEFT players equal the selected team size.
- Active RIGHT players equal the selected team size at full curriculum level; earlier levels can use fewer active RIGHT players.
- Inactive slots exist only in the coach critic representation and have `active=0.0`.

### Starts

Fixed positions are used only at reset / kickoff. The spawn layouts are simple symmetric templates for 3, 5, and 7 players, named only by `slot_index`. After kickoff, players choose dynamic targets from the ball, owner, teammates, opponents, and goals.

### Scripted Players

The same dynamic scripted-team logic drives RIGHT opponents and idle LEFT teammates in human mode. The human-controlled LEFT player is never overridden.

Temporary jobs are recomputed from live state whenever the level reaction cadence allows it:

- `carrier`: current team player with the ball
- `support_a`: forward / diagonal support option
- `support_b`: level or safer support option
- `stopper`: closest scripted defender pressing the opponent carrier
- `cover_a`: lane cover between ball / carrier and own goal
- `cover_b`: dangerous-opponent mark or nearby passing-space cover

These are temporary jobs, not roles. The planner uses target points, teammate separation, mild opponent avoidance, and small possession-style variation (`direct`, `wide_upper`, `wide_lower`, `patient`).

### Kick

- `kick` is one semantic action. It can behave like a pass, shot, or clearance depending on facing and context.
- If a safe teammate is broadly aligned with the carrier's facing direction, the kick is biased toward that teammate.
- Otherwise, if the carrier is facing broadly toward the opponent goal, the kick is biased toward the opponent goal center.
- Otherwise, the ball is kicked along the carrier's current facing direction.

### Scoring

- A goal is scored when the ball crosses the opponent goal line inside the goal mouth.
- `kick` is useful but not required.
- Dribble goals are allowed.
- Kicks, deflections, loose balls, and dribbles can all score if the crossing is inside the goal mouth.

### Centralized Critic State

- The centralized critic receives a training-only `128` input coach view.
- The coach view is a compact global football snapshot, not a concatenation of actor observations.
- The actor still runs independently for each active LEFT player using only that player's local `36` input observation.
- LEFT is encoded as `7 x 8`.
- RIGHT is encoded as `7 x 8`.
- Each player slot uses: `x_norm`, `y_norm`, `vx`, `vy`, `theta_cos`, `theta_sin`, `has_ball`, `active`.
- Ball state contributes absolute normalized position, velocity, LEFT / RIGHT / free ownership flags, and ball-to-goal landmarks.
- Match state contributes normalized time, score, level, and team size.
- `CENTRAL_OBS_DIM = 128`.
- `state_team_size_norm = active_left_players / 7.0`.

Coach feature order:

- `left1..left7`: each slot in player-feature order above
- `right1..right7`: each slot in player-feature order above
- `tgt`: `x_norm`, `y_norm`, `vx`, `vy`, `owner_left`, `owner_right`, `owner_free`
- `land`: `ball_to_opp_goal_dx`, `ball_to_opp_goal_dy`, `ball_to_own_goal_dx`, `ball_to_own_goal_dy`
- `state`: `time_norm`, `left_score_norm`, `right_score_norm`, `level_norm`, `team_size_norm`

### Step Contract

- `env.step(...)` returns a scalar team reward.
- Per-player rewards are exposed in `info["reward_vec"]` with shape `(active_left_players,)`.
- Realized step contributions are exposed in `info["reward_breakdown"]`.
- Episode totals are exposed at terminal steps in `info["reward_components"]` as `G`, `C`, `P`, `B`, and `TS` for the training log.

### Diagnostics

- `KICK_DEBUG_SANITY=1` enables runtime checks for observation shape, masked invalid kicks, centralized state shape, and reward-vector consistency.

## Rewards (Training)

Reward terms:

- `G`: score bonus, total `+10.0`, normalized across active LEFT players
- `C`: concede penalty, total `-5.0`, normalized across active LEFT players
- `P`: controlled forward ball progress for the current LEFT owner after stable physical possession
- `B`: bounded ball-support shaping for one useful non-carrier support runner near the ball
- `TS`: team-shape anti-clumping penalty across LEFT players

There is no role-zone reward, map-anchor reward, explicit pass bonus, turnover penalty, stamina reward / penalty, goalkeeper exception, or shot-quality reward.

## Curriculum (Train)

- Shared 5-level curriculum progression from `core/curriculum.py`.
- `LEVEL_SCRIPTED_SETTINGS` is the single curriculum source of truth.
- Each level owns the scripted knobs plus `right_players`, where `right_players` maps team size to active RIGHT count.
- The env expands that table into the per-team-size structure required by the shared curriculum code; it is not a second place to tune curriculum.
- There is no separate difficulty object, enum, or easy / medium / hard branch.
- LEFT always uses the selected team size and level-5 scripted knobs for idle human-mode teammates.
- RIGHT scales by level up to the selected team size.

3v3 RIGHT counts by level: `1, 1, 2, 3, 3`

5v5 RIGHT counts by level: `2, 3, 4, 5, 5`

7v7 RIGHT counts by level: `3, 4, 5, 6, 7`

Kickoff starts from a free center ball, and both scripted teams begin moving immediately. An episode counts as a success if LEFT scores more than it concedes.

## Run Commands

```bash
rl-toybox-train --game kick --team-size 3
rl-toybox-train --game kick --team-size 5
rl-toybox-train --game kick --team-size 7

rl-toybox-play-ai --game kick --team-size 3 --render
rl-toybox-play-ai --game kick --team-size 5 --render
rl-toybox-play-ai --game kick --team-size 7 --render

rl-toybox-play-user --game kick --team-size 3
rl-toybox-play-user --game kick --team-size 5
rl-toybox-play-user --game kick --team-size 7
```

When `--team-size` is omitted, Kick defaults to `3`. The CLI also accepts readable labels such as `3v3`, `5v5`, and `"7 vs. 7"`. When `--level` is omitted, `train` starts at `L1` and `play-user` / `play-ai` default to `L5`.
