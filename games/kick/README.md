# Kick

Top-down football environment with a shared LEFT-team policy and a centralized critic during training. `kick` is currently paused as an experimental branch, but the environment and documentation remain in the repo because it exercises the multi-agent and CTDE pieces more directly than the other games.

## Default Algorithm / Network

- Algorithm: PPO with MAPPO-style training
- Actor MLP: `[128, 128]`
- Critic MLP: `[256, 256]`
- Actor output: `Discrete(12)` logits per LEFT player
- Critic input: centralized state plus local agent context

## Controls (Human)

- Move: `W/A/S/D`
- Shoot: hold/release `Space` for low, mid, or high kicks
- Human mode keeps automatic player switching behavior
- Rendered overlays: `X` toggles the formation/zone target ghosts during `play-user` and `play-ai`

## Observation / Actions

- Observation:
  - RL mode (`train` / `eval`): `(N_left, 48)` where each row is one LEFT-player feature vector
  - Human mode: single `(48,)` vector for the currently controlled player
- Feature blocks:
  - `SELF` (9): `self_vx self_vy self_theta_cos self_theta_sin self_has_ball self_role self_role_lane self_stamina self_stamina_delta`
  - `BALL` (7): `tgt_dx tgt_dy tgt_rel_angle_sin tgt_rel_angle_cos tgt_dvx tgt_dvy tgt_owner_team`
  - `GOAL (opponent)` (4): `goal_dx goal_dy goal_rel_angle_sin goal_rel_angle_cos`
  - `GOAL (own)` (4): `own_goal_dx own_goal_dy own_goal_rel_angle_sin own_goal_rel_angle_cos`
  - `OWN 1..3` (12): `own{k}_{dx,dy,dvx,dvy}`
  - `OPP 1..3` (12): `opp{k}_{dx,dy,dvx,dvy}`
- Nearest teammate/opponent selection is deterministic: sort by `(distance, player.slot_index)`
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
- Effective possession uses `ball_owner_team` when owned and `last_touch_team` when the ball is free.
- Formation/zone behavior uses effective possession so the team shape does not snap while the ball is airborne.
- Progress shaping is credited to the responsible LEFT player:
  - current LEFT owner while controlled
  - `last_touch_player_id` after a LEFT touch while the ball is free

### Centralized Critic State

- The centralized state is fixed-size and robust to team-size changes.
- It pads observations up to `MAX_LEFT_PLAYERS=11`.
- `central_mask` distinguishes present players from padded slots.

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

Logged reward codes are `G C T A P Z`:

- `G`: team score bonus, total `+10.0`, normalized across LEFT players
- `C`: team concede penalty, total `-5.0`, normalized across LEFT players
- `T`: turnover penalty for the responsible LEFT player
- `A`: pass reward for the passer when a LEFT teammate receives the kick
- `P`: dense ball-progress shaping toward the opponent goal
- `Z`: zone-discipline penalty based on distance from the role anchor

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/kick/config.py` under `CURRICULUM_PROMOTION`
- LEFT always stays at `11` RL-controlled players
- Opponent scaling:
  - Level 1: `11v3`
  - Level 2: `11v7`
  - Level 3: `11v11`

An episode counts as a success if LEFT scores more than it concedes.

## Run Commands

```bash
rl-toybox-train --game kick
rl-toybox-play-ai --game kick --model best --render
rl-toybox-play-user --game kick
python -m scripts.train --game kick
python -m scripts.play_ai --game kick --model best --render
python -m scripts.play_user --game kick
```

See `games/kick/config.py` and `games/kick/env.py` for the full PPO, CTDE, reward, and curriculum settings.
