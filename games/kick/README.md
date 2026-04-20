# Kick

Top-down football environment with a shared LEFT-team policy and a centralized critic during training. `kick` uses a true `7v7` max setup on both sides and is the repo's active multi-agent / CTDE showcase.

## Clip

No clip is currently checked into the repo for `kick`.

## Algorithm / Network

- Default algorithm: `ppo`
- Teams: true `7v7` max on both sides
- Roles: `GK LB RB LM CM RM CS`
- Per-player IO: `obs=56`, `act=12`
- Shared actor: `56 -> 96 -> 96 -> 12`
- Centralized critic: `405 -> 192 -> 192 -> 1`
- Critic input: padded team state plus local agent context

## Controls (Human)

- Move: `W/A/S/D`
- Shoot: hold/release `Space` for low, mid, or high kicks
- Human mode keeps automatic player switching behavior
- Rendered overlays: `X` toggles the grey formation ghosts and safe-zone overlays during `play-user` and `play-ai`

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> TGT -> LAND -> ALLY -> OPP -> MAP -> FLAG`, emitted once per LEFT player
- Observation:
  - RL mode (`train` / `eval`): `(N_left, 56)` where each row is one LEFT-player feature vector
  - Human mode: single `(56,)` vector for the currently controlled player
- Feature blocks in exact order:
  - `SELF` (16): `self_x_norm self_y_norm self_vx self_vy self_theta_cos self_theta_sin self_has_ball self_stamina self_stamina_delta self_role_gk self_role_lb self_role_rb self_role_lm self_role_lcm self_role_rm self_role_lcs`
  - `TGT` (6): `tgt_dx tgt_dy tgt_dist_norm tgt_rel_ang_sin tgt_rel_ang_cos tgt_dvx`
  - `LAND` (7): `land_opp_goal_dx land_opp_goal_dy land_own_goal_dx land_own_goal_dy land_gk_dx land_gk_dy land_gk_dvy`
  - `ALLY` (12): `ally1_dx ally1_dy ally1_dvx ally1_dvy ally2_dx ally2_dy ally2_dvx ally2_dvy ally3_dx ally3_dy ally3_dvx ally3_dvy`
  - `OPP` (12): `opp1_dx opp1_dy opp1_dvx opp1_dvy opp2_dx opp2_dy opp2_dvx opp2_dvy opp3_dx opp3_dy opp3_dvx opp3_dvy`
  - `MAP` (2): `map_anchor_dx map_anchor_dy`
  - `FLAG` (1): `flag_shoot_mode`
- Role one-hot features keep the compact names `self_role_lcm` / `self_role_lcs`, which correspond to the gameplay roles `CM` / `CS`.
- Nearest teammate/opponent selection is deterministic: sort by `(distance, player.slot_index)`
- `ALLY` and `OPP` each encode exactly 3 nearest outfield players and always exclude the goalkeeper.
- Reduced-opponent curriculum levels keep those 3 slots by zero-padding any missing outfield opponents.
- Goalkeeper context stays in `LAND` through `land_gk_dx`, `land_gk_dy`, and `land_gk_dvy`.
- Actions: `Discrete(12)` (`ACTION_NAMES`, ordered)
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

In RL mode the environment expects one action per LEFT player each step.

## Environment Notes

### Possession and Credit Assignment

- Physical owner comes from `ball_owner`.
- Effective possession uses the current owner when owned and `last_touch_team` when the ball is free.
- Formation and anchor behavior use that same effective-possession interpretation so the team shape does not snap while the ball is airborne.
- Progress shaping is credited to the responsible LEFT player:
  - current LEFT owner while controlled
  - `last_touch_player_id` after a LEFT touch while the ball is free

### Ghost / Ideal Position

- Each LEFT player keeps a defensive base at its role `home_x` / `home_y`.
- When LEFT is effectively attacking, the ideal anchor shifts forward on `x` by a small role-specific amount; when LEFT is defending, the ideal `x` returns to `home_x`.
- `y` shifts only modestly with normalized ball lane so the whole shape can slide up or down without becoming twitchy.
- The displayed grey ghost and player-to-ghost line use the same smoothed anchor that also feeds `map_anchor_*` and the zone-discipline penalty.
- Zone discipline uses one shared outfield anchor ellipse plus a dedicated goalkeeper ellipse: no bonus or penalty inside the tolerance zone, then a progressive negative penalty outside it with more freedom on `x` than `y`.

### Shoot Mode

- `flag_shoot_mode` turns on only when the ball carrier has possession, is close enough to the opponent goal, and is facing sufficiently into a forward attack cone.
- Outside shoot mode, `kick_low`, `kick_mid`, and `kick_high` remain straight passes / clearances along the current facing direction.
- Inside shoot mode, those same three actions become goalkeeper-aware shots:
  - `kick_low`: fastest / safest finish toward center or open-center.
  - `kick_mid`: balanced finish toward an inner open lane.
  - `kick_high`: highest-power finish toward the outer open lane just inside the post.
- Shot targeting uses visible goal-mouth and opponent-goalkeeper geometry, with only modest spread; the old hidden random goalkeeper-bypass behavior is gone.
- When shoot mode is active, the player's normal fill stays in place and three small overlay squares appear across the body; while the human shot is charging, the low/mid/high segments light up one-by-one in the outline color.

### Centralized Critic State

- The centralized state is fixed-size and robust to team-size changes.
- It pads observations up to `MAX_LEFT_PLAYERS=7`.
- `central_mask` distinguishes present players from padded slots.
- `CENTRAL_OBS_DIM = (7 * 56) + 7 + 6 = 405`.

### Action Masking

- If `self_has_ball == 0`, `kick_low`, `kick_mid`, and `kick_high` are invalid.
- Masking is applied in both training and evaluation.
- Eval uses masked action selection, so invalid kicks are not chosen.

### Step Contract

- `env.step(...)` returns a scalar team reward.
- Per-player rewards are always exposed in `info["reward_vec"]`.
- Realized step contributions are exposed in `info["reward_breakdown"]`.

### Diagnostics

- Training prints PPO diagnostics after episodes.
- `KICK_DEBUG_SANITY=1` enables runtime checks for observation shape, masked invalid kicks, GK-catch exclusions, and reward-vector consistency.

## Rewards (Training)

Logged reward codes are `G C T A P B Z`:

- `G`: team score bonus, total `+10.0`, normalized across LEFT players
- `C`: team concede penalty, total `-5.0`, normalized across LEFT players
- `T`: turnover penalty for the responsible LEFT player
- `A`: pass reward for the passer when a LEFT teammate receives the kick
- `P`: dense ball-progress shaping toward the opponent goal while LEFT has attacking control
- `B`: dense ball-approach shaping for the single closest LEFT outfield challenger when LEFT does not have possession
  - `ball_improve = prev_ball_dist - curr_ball_dist`
  - `B = 0.02 * clip(ball_improve, -0.05, 0.05)`
- `Z`: zone-discipline penalty based on distance from the role anchor
  - disabled for the LEFT ball carrier, shoot-mode players, and the active closest LEFT outfield challenger when LEFT is out of possession
  - `d = sqrt((dx / tol_x)^2 + (dy / tol_y)^2)` using `map_anchor_*`-equivalent offsets
  - inside zone (`d <= 1.0`): `0.0`
  - outside zone: `-(0.0005 * e + 0.004 * e^2)` where `e = d - 1.0`

## Curriculum (Train)

- Shared 5-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/kick/config.py` under `CURRICULUM_PROMOTION`
- LEFT always stays at `7` RL-controlled players
- Opponent scaling:
  - Level 1: `7v1` vs `GK`, goals `2.5x`, enemy stamina `0.50`, `start_possession=RND_LEFT`
  - Level 2: `7v3` vs `GK LM RM`, goals `2.0x`, enemy stamina `0.50`, `start_possession=RND_LEFT`
  - Level 3: `7v5` vs `GK LB RB CM CS`, goals `1.5x`, enemy stamina `0.75`, `start_possession=CEN`
  - Level 4: `7v7` vs `GK LB RB LM CM RM CS`, goals `1.25x`, enemy stamina `0.75`, `start_possession=CEN`
  - Level 5: `7v7` vs `GK LB RB LM CM RM CS`, standard goals, enemy stamina `1.00`, `start_possession=CEN`

`start_possession` supports `CEN` for a free center-ball start, `RND_LEFT` for a random LEFT outfielder start, and `RND_RIGHT` for a random RIGHT outfielder start.
In levels `1` and `2`, `RND_LEFT` seeds possession on a random LEFT outfield player and excludes the goalkeeper.
Level `4` is the first full `7v7` game; level `5` keeps the same full role list with the regular goal and stamina modifiers.

An episode counts as a success if LEFT scores more than it concedes.

## Run Commands

```bash
rl-toybox-train --game kick
rl-toybox-play-ai --game kick --render
rl-toybox-play-user --game kick
python -m scripts.train --game kick
python -m scripts.play_ai --game kick --render
python -m scripts.play_user --game kick
```

See `games/kick/config.py` and `games/kick/env.py` for the CTDE runtime, rewards, curriculum settings, and training defaults. Kick's game-wide actor and critic sizes live in `DEFAULT_MODEL_CONFIG["hidden_sizes"]` and `DEFAULT_MODEL_CONFIG["critic_hidden_sizes"]`, its PPO-specific extras live in `ALGO_CONFIG_OVERRIDES["ppo"]`, its level-specific entropy schedule lives in `LEVEL_SETTINGS[*]["entropy_coef"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
