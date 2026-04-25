# Kick

Top-down football environment with a shared LEFT-team policy and a centralized critic during training. `kick` uses a true `7v7` max setup on both sides and is the repo's active multi-agent / CTDE showcase.

## Clip

![Kick Demo](../../media/kick-demo.gif)

## Algorithm / Network

- Default algorithm: `ppo`
- Teams: true `7v7` max on both sides
- Roles: `GK LB RB LM CM RM CS`
- Per-player IO: `obs=63`, `act=11`
- Shared actor trunk: `63 -> 64 -> 64` with 4 role heads (`GK`, `DEF`, `MID`, `ATK`) ending in `-> 11`
- Centralized critic: `454 -> 128 -> 128 -> 1`
- Critic input: padded team state plus local agent context

## Controls (Human)

- Move: `W/A/S/D`
- Pass: `Space`
- Shoot: `Enter`
- Human mode keeps automatic player switching behavior
- Rendered overlays: `X` toggles translucent grey formation ghosts and role-zone overlays during `play-user` and `play-ai`

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> TGT -> LAND -> ALLY -> OPP -> MAP -> FLAG`, emitted once per LEFT player
- Observation:
  - RL mode (`train` / `eval`): `(N_left, 63)` where each row is one LEFT-player feature vector
  - Human mode: single `(63,)` vector for the currently controlled player
- Feature blocks in exact order:
  - `SELF` (19): `self_x_norm self_y_norm self_vx self_vy self_theta_cos self_theta_sin self_has_ball self_stamina self_stamina_delta self_last_move_x self_last_move_y self_action_changed self_role_gk self_role_lb self_role_rb self_role_lm self_role_lcm self_role_rm self_role_lcs`
  - `TGT` (6): `tgt_dx tgt_dy tgt_dist_norm tgt_rel_ang_sin tgt_rel_ang_cos tgt_dvx`
  - `LAND` (11): `land_opp_goal_dx land_opp_goal_dy land_own_goal_dx land_own_goal_dy land_own_gk_dx land_own_gk_dy land_shot_line_dy land_shot_tti land_gk_dx land_gk_dy land_gk_dvy`
  - `ALLY` (12): `ally1_dx ally1_dy ally1_dvx ally1_dvy ally2_dx ally2_dy ally2_dvx ally2_dvy ally3_dx ally3_dy ally3_dvx ally3_dvy`
  - `OPP` (12): `opp1_dx opp1_dy opp1_dvx opp1_dvy opp2_dx opp2_dy opp2_dvx opp2_dvy opp3_dx opp3_dy opp3_dvx opp3_dvy`
  - `MAP` (2): `map_anchor_dx map_anchor_dy`
  - `FLAG` (1): `flag_shot_quality`
- Role one-hot features keep the compact names `self_role_lcm` / `self_role_lcs`, which correspond to the gameplay roles `CM` / `CS`.
- Nearest teammate/opponent selection is deterministic: sort by `(distance, player.slot_index)`
- `ALLY` and `OPP` each encode exactly 3 nearest outfield players and always exclude the goalkeeper.
- Reduced-opponent curriculum levels keep those 3 slots by zero-padding any missing outfield opponents.
- Goalkeeper context stays in `LAND`; opponent-goalkeeper geometry supports shooting, while own-goalkeeper and incoming-shot features help the goalkeeper learn coverage.
- PPO uses role-conditioned policy heads: `GK`, `DEF` (`LB/RB`), `MID` (`LM/CM/RM`), and `ATK` (`CS`).
- Actions: `Discrete(11)` (`ACTION_NAMES`, ordered)
  - `0 stay`
  - `1 move_n`
  - `2 move_ne`
  - `3 move_e`
  - `4 move_se`
  - `5 move_s`
  - `6 move_sw`
  - `7 move_w`
  - `8 move_nw`
  - `9 pass`
  - `10 shoot`

In RL mode the environment expects one action per LEFT player each decision. Each decision is held for `12` physics frames, so the policy acts five times per rendered second at the standard `60 FPS`.

## Environment Notes

### Possession and Credit Assignment

- Physical owner comes from `ball_owner`.
- Effective possession uses the current owner when owned and `last_touch_team` when the ball is free.
- Formation and anchor behavior use that same effective-possession interpretation so the team shape does not snap while the ball is airborne.
- Controlled progress pays only stable LEFT physical possession:
  - the same LEFT owner must hold the ball for `3` consecutive physics frames
  - only positive forward ball-depth gains are rewarded
  - the progress frontier resets on any owner or team change

### Ghost / Ideal Position

- Each LEFT player keeps a defensive base at its role `home_x` / `home_y`.
- When LEFT is effectively attacking, the ideal anchor shifts forward on `x` by a small role-specific amount; when LEFT is defending, the ideal `x` returns to `home_x`.
- `y` shifts only modestly with normalized ball lane so the whole shape can slide up or down without becoming twitchy.
- The goalkeeper is the exception: its ghost/role-zone anchor stays fixed just outside the goal itself, with the goalkeeper zone grazing the goal edge.
- The displayed grey ghost and player-to-ghost line use the same smoothed anchor that also feeds `map_anchor_*` and the role-zone penalty.
- Role zone uses one shared outfield anchor ellipse plus a dedicated goalkeeper ellipse: no penalty inside the tolerance zone, then a small progressive penalty outside it with more freedom on `x` than `y`.

### Pass / Shoot

- `flag_shot_quality` is a continuous `0..1` cue for the ball carrier: it rises when the carrier is closer to the opponent goal and broadly facing it. For LEFT, the viable facing band is the goal-facing diagonal/straight set (`NE`, `E`, `SE`); the mirrored band applies for RIGHT.
- `pass` chooses a safe teammate broadly aligned with the carrier's current facing, otherwise it kicks along the current facing direction. The pass assist can bend toward a teammate, but it no longer forces every pass forward.
- `shoot` is a single semantic action. The environment bends strongly toward an open goal-mouth lane using opponent-goalkeeper geometry whenever `flag_shot_quality` is non-zero, with higher quality tightening the angle and spread so diagonal goal-facing shots can still be angled into the net.
- The ball carrier shows a small shot-quality bar above the player: it spans the same width as the player and fills from `0..1` using the player's fill color at 50% transparency.

### Scripted Team

- Scripted outfield players use a compact rule set: advance in possession, pass when pressured, and shoot once shot quality clears a slightly randomized per-possession threshold.
- When pressured with a pass available, the scripted carrier passes most of the time and occasionally commits to a short diagonal dribble burst to break deadlocks.
- Off-ball teammates keep moving: in possession they spread into simple support lanes, and out of possession one hunter presses while the rest drop into compact defensive positions between the ball and their goal.
- Goalkeepers defend on a shallow semicircle in front of the goal: they step out most near the center line, move back toward the line near the posts, and track the expected shot lane with smoothing.
- If a goalkeeper is already in a good covering position between the ball and goal, it holds that position until the target moves outside a small hysteresis band.

### Centralized Critic State

- The centralized state is fixed-size and robust to team-size changes.
- It pads observations up to `MAX_LEFT_PLAYERS=7`.
- `central_mask` distinguishes present players from padded slots.
- `CENTRAL_OBS_DIM = (7 * 63) + 7 + 6 = 454`.

### Action Masking

- If `self_has_ball == 0`, `pass` and `shoot` are invalid.
- Masking is applied in both training and evaluation.
- Eval uses masked action selection, so invalid kicks are not chosen.

### Step Contract

- `env.step(...)` returns a scalar team reward.
- Per-player rewards are always exposed in `info["reward_vec"]`.
- Realized step contributions are exposed in `info["reward_breakdown"]`.

### Diagnostics

- Training prints PPO diagnostics after episodes without per-reward-term suffixes.
- `KICK_DEBUG_SANITY=1` enables runtime checks for observation shape, masked invalid kicks, and reward-vector consistency.

## Rewards (Training)

Reward terms are intentionally kept internal to the environment and are not rendered in the bottom bar or appended to training logs:

- `G`: team score bonus, total `+10.0`, normalized across LEFT players
- `C`: team concede penalty, total `-5.0`, normalized across LEFT players
- `P`: controlled forward ball progress for the current LEFT owner after stable physical possession
  - requires the same LEFT owner for `3` consecutive physics frames
  - only pays positive forward gains and resets its frontier on any owner/team change
  - `P = 1.0 * clip(max(0, ball_depth - frontier), 0.0, 0.01)`
- `B`: bounded ball-support shaping for one active LEFT outfielder near the ball / active play
  - picks the nearest LEFT outfielder to the ball; if LEFT owns the ball, the carrier is excluded so the reward goes to a support runner
  - target support distance is `2.5 * TILE_SIZE`, so collapsing directly onto the ball is not rewarded
  - `B = 0.01 * clip(prev_error - curr_error, -0.25, 0.25)` where `error = abs(dist_to_ball - target_dist)`
- `TS`: team-shape anti-clumping penalty for LEFT outfield players
  - uses each outfielder's nearest-teammate spacing shortfall against `TEAM_SHAPE_MIN_DIST_NORM = 0.065`
  - `TS_i = -clip(0.001 * s + 0.01 * s^2, 0.0, 0.00006)` where `s = max(0, min_dist_norm - nearest_teammate_dist_norm)`
- `RZ`: soft role-zone penalty based on distance from the role anchor
  - disabled for the LEFT ball carrier and the active closest LEFT outfield challenger when LEFT is out of possession
  - `d = sqrt((dx / tol_x)^2 + (dy / tol_y)^2)` using `map_anchor_*`-equivalent offsets
  - inside zone (`d <= 1.0`): `0.0`
  - outside zone: `-(0.000015 * e + 0.000015 * e^2)` where `e = d - 1.0`

## Curriculum (Train)

- Shared 5-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/kick/config.py` under `CURRICULUM_PROMOTION`
- LEFT always stays at `7` RL-controlled players
- Opponent scaling:
  - Level 1: `players_opponent=1`, `goals_size_scale=3`, `enemy_stamina_scale=0.25`, `entropy_coef=0.02`, `start_possession=RND_LEFT`
  - Level 2: `players_opponent=3`, `goals_size_scale=2.25`, `enemy_stamina_scale=0.5`, `entropy_coef=0.015`, `start_possession=RND_LEFT`
  - Level 3: `players_opponent=5`, `goals_size_scale=1.75`, `enemy_stamina_scale=0.625`, `entropy_coef=0.01`, `start_possession=CEN`
  - Level 4: `players_opponent=7`, `goals_size_scale=1.25`, `enemy_stamina_scale=0.75`, `entropy_coef=0.0075`, `start_possession=CEN`
  - Level 5: `players_opponent=7`, `goals_size_scale=1`, `enemy_stamina_scale=1`, `entropy_coef=0.005`, `start_possession=CEN`

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

When `--level` is omitted, `train` starts at `L1` and `play-user` / `play-ai` default to `L5`.

See `games/kick/config.py` and `games/kick/env.py` for the CTDE runtime, rewards, curriculum settings, and training defaults. Kick's game-wide actor and critic sizes live in `DEFAULT_MODEL_CONFIG["hidden_sizes"]` and `DEFAULT_MODEL_CONFIG["critic_hidden_sizes"]`, its PPO-specific extras live in `ALGO_CONFIG_OVERRIDES["ppo"]`, its level-specific entropy schedule lives in `LEVEL_SETTINGS[*]["entropy_coef"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
