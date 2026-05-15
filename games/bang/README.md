# Bang

Top-down arena shooter focused on movement, aiming, line of sight, and timing shots under pressure. `bang` is one public game id with selectable combat modes, all using the same 36-input / 8-action DQN shape.

## Clip

![Bang Demo](../../media/bang-demo.gif)

## Modes

- Game id: `bang`
- `duel` (`Duel`): `1v1`, two teams, no ally for the controlled player
- `arena` (`Arena`): `1v1v1v1`, four solo teams, no ally for the controlled player
- `team_arena` (`Team Arena`): `2v2v2v2`, four teams of two, with two RL-controlled friendly players sharing one DQN policy
- Default mode: `team_arena`

Mode defines the maximum/target layout. Curriculum controls how many enemies from that layout are active during training. The actor network, observation order, action space, model size, rewards, and DQN algorithm stay shared across all modes.

`Team Arena` is the recommended training mode for a general Bang policy because it exposes the full ally/enemy/team structure. `Duel` and `Arena` are simpler subset cases of the same 36-input contract.

## Algorithm / Network

- Default algorithm: `dqn`
- IO: `obs=36`, `act=8`
- Default Q-network: `36 -> 64 -> 64 -> 8`
- Default model config: `{"hidden_sizes": [64, 64]}`
- Runtime shape: double-Q, dueling head, prioritized replay
- Controlled-agent shape: `Duel` and `Arena` emit one observation/action; `Team Arena` emits two observations/actions and stores both as per-agent samples in the same DQN replay buffer.

## Controls (Human)

- Move: `W/A/S/D`
- Aim: left/right arrows
- Shoot: `Space`
- Rendered overlays: `X` toggles translucent `sens_*` ray ghosts during `play-user` and `play-ai`
- If no movement key is held for a frame, movement intent becomes `move_stop`

## Observation / Actions

- Observation: `36` floats (`INPUT_FEATURE_NAMES`, exact order)

```python
[
    # SELF
    "self_ang_sin",
    "self_ang_cos",
    "self_move_x",
    "self_move_y",
    "self_shot_cd_norm",
    # SENS
    "sens_fwd",
    "sens_left",
    "sens_right",
    "sens_back",
    # ALLY
    "ally_dx",
    "ally_dy",
    "ally_dist_norm",
    "ally_los",
    "ally_ang_sin",
    "ally_ang_cos",
    "ally_shot_cd_norm",
    "ally_active",
    # OPP
    "opp1_dx",
    "opp1_dy",
    "opp1_los",
    "opp1_ang_sin",
    "opp1_ang_cos",
    "opp2_dx",
    "opp2_dy",
    "opp2_los",
    "opp2_ang_sin",
    "opp2_ang_cos",
    "opp3_dx",
    "opp3_dy",
    "opp3_los",
    "opp3_ang_sin",
    "opp3_ang_cos",
    # SUMMARY
    "opp_near_dist_norm",
    # HAZ
    "haz_tti_norm",
    "haz_miss_norm",
    "haz_in_traj",
]
```

- Actions: `Discrete(8)` (`ACTION_NAMES`, ordered)
  - `0 move_up`
  - `1 move_down`
  - `2 move_left`
  - `3 move_right`
  - `4 move_stop`
  - `5 aim_left`
  - `6 aim_right`
  - `7 shoot`

- `sens_*` values are normalized free-space-before-hit values in `[0, 1]`. Hits include arena walls and square obstacles.
- The `ALLY` block is zeroed in `Duel` and `Arena`. In `Team Arena`, each RL-controlled friendly observes the other RL-controlled friendly as its active ally.
- Opponent slots contain the nearest active enemy players only, sorted by `(distance_to_self, stable_player_id)`.
- Missing opponent slots are zero-padded, and `opp_near_dist_norm` is `1.0` when no active enemy remains.
- `haz_*` values consider projectiles from enemy teams, not allied projectiles.
- Inactive curriculum slots are omitted from targeting, collisions, rendering, hazards, and opponent-slot filling.

## Environment Notes

- Scripted players are enemies only. `Team Arena` has no scripted ally during training or AI evaluation; both friendly players are controlled by the shared DQN policy.
- Friendly hits are ignored through a simple projectile team check.
- In `Team Arena`, any RL-controlled friendly death is a loss.
- A win occurs when all active enemy players are eliminated.

## Rewards (Training)

- `REWARD_WIN = +10.0` on match win
- `PENALTY_LOSE = -5.0` on match loss
- `REWARD_KILL = +2.0` per enemy elimination by either RL-controlled friendly
- Engagement shaping: `clip(0.5 * (Phi_eng_next - Phi_eng_prev), -0.25, +0.25)` where `Phi_eng = (1 if tgt_in_los else 0) - tgt_dist_norm`
- Hazard shaping: `clip(0.5 * (Phi_haz_next - Phi_haz_prev), -0.25, +0.25)` where `Phi_haz = haz_dist_norm - 1.5 * haz_in_traj`
- `PENALTY_STEP = -0.005` every training step

## Curriculum (Train)

- Shared 5-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/bang/config.py` under `CURRICULUM_PROMOTION`
- `LEVEL_SETTINGS` is the single curriculum source of truth.
- Each level owns the scripted-pressure knobs plus `active_enemies`, where `active_enemies` maps Bang mode to active enemy count.
- Mode controls the maximum player/team layout; curriculum controls active enemy pressure inside that layout.
- Level settings:
  - Level 1: `active_enemies={duel: 1, arena: 1, team_arena: 1}`, `0` obstacles, `enemy_movement=0.0`, `enemy_repositioning=0.0`, enemy shoot probability `0.0`
  - Level 2: `active_enemies={duel: 1, arena: 1, team_arena: 2}`, `4` obstacles, `enemy_movement=0.25`, `enemy_repositioning=0.25`, enemy shoot probability `0.025`
  - Level 3: `active_enemies={duel: 1, arena: 2, team_arena: 3}`, `8` obstacles, `enemy_movement=0.50`, `enemy_repositioning=0.50`, enemy shoot probability `0.05`
  - Level 4: `active_enemies={duel: 1, arena: 3, team_arena: 4}`, `10` obstacles, `enemy_movement=0.75`, `enemy_repositioning=0.75`, enemy shoot probability `0.075`
  - Level 5: `active_enemies={duel: 1, arena: 3, team_arena: 6}`, `12` obstacles, `enemy_movement=1.00`, `enemy_repositioning=1.00`, enemy shoot probability `0.10`

In `Team Arena`, both friendly RL agents are active from Level 1. Level 5 is the full `2v2v2v2` format.

## Run Commands

```bash
rl-toybox-train --game bang --mode duel
rl-toybox-train --game bang --mode arena
rl-toybox-train --game bang --mode team_arena

rl-toybox-play-ai --game bang --mode duel --render
rl-toybox-play-ai --game bang --mode arena --render
rl-toybox-play-ai --game bang --mode team_arena --render

rl-toybox-play-user --game bang --mode duel
rl-toybox-play-user --game bang --mode arena
rl-toybox-play-user --game bang --mode team_arena
```

Without installation, use the same `--mode` values with `python -m scripts.train`, `python -m scripts.play_ai`, and `python -m scripts.play_user`.

When `--mode` is omitted, Bang defaults to `team_arena`. When `--level` is omitted, `train` starts at `L1` and `play-user` / `play-ai` default to `L5`. Like Kick, Bang uses one shared model tag across modes, such as `dqn_64_64_L5_best.pth`.

See `games/bang/config.py` for the game constants, modes, rewards, curriculum settings, and training defaults. Bang's game-wide net size lives in `DEFAULT_MODEL_CONFIG["hidden_sizes"]`, its DQN-specific extras live in `ALGO_CONFIG_OVERRIDES["dqn"]`, and its default training stop budget lives in `DEFAULT_TRAIN_CONFIG["budget"]`.
