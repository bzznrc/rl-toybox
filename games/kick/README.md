# Kick

11v11 football environment (Arcade-style top-down) with PPO-oriented discrete actions.

## Algorithm / Network

- Algo: PPO (parameter-shared)
- Hidden sizes: `[96, 96]`

## Controls (Human)

- Move: `W/A/S/D` (or arrows)
- Aim: mouse
- Shoot: hold/release left mouse button (low/mid/high by hold time)
- Switch controlled player: `Tab` (closest to ball)

## Observation / Actions

- Observation: `36` floats (`INPUT_FEATURE_NAMES`, ordered)
  - `self_vx`
  - `self_vy`
  - `self_theta_cos`
  - `self_theta_sin`
  - `self_has_ball`
  - `self_role`
  - `self_stamina`
  - `self_stamina_delta`
  - `self_in_contact`
  - `self_last_action`
  - `tgt_dx`
  - `tgt_dy`
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
- Kick actions only apply when the controlled player has the ball; otherwise treated as `STAY`.

## Rewards (Training)

- Outcome `REWARD_SCORE`: `+10` when left team scores.
- Outcome `PENALTY_CONCEDE`: `-5` when left team concedes.
- Ball-progress shaping: `r_prog = clip(1.0 * (Phi' - Phi), -0.2, +0.2)` with `Phi = -dist(ball, opp_goal)_norm`.
- Event possession change: `+0.5` on gain, `-0.5` on loss (team-shared).
- Event `PENALTY_KICK_COST`: `-0.01` when the chosen action is `kick_low`, `kick_mid`, or `kick_high`.
- Step `PENALTY_STEP`: `-0.001` every training step.

The progress potential is based on normalized ball distance to the opponent goal center; moving the ball toward goal increases `Phi` and yields positive signed-ΔPhi shaping.

## Training

Default algo is PPO (categorical policy):

```bash
rl-toybox-train --game kick --algo ppo
```

Key hyperparameters:

- Train: `max_iterations=1500`, `rollout_steps=2048`, `checkpoint_every_iterations=10`, `reward_window=100`
- Algo: `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_ratio=0.2`, `update_epochs=4`, `minibatch_size=512`, `entropy_coef=0.01`, `value_coef=0.5`, `max_grad_norm=0.5`

Play AI:

```bash
rl-toybox-play-ai --game kick --model best --render
```

Play user:

```bash
rl-toybox-play-user --game kick
```
