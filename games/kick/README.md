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

- Goal scored (left team): `+20.0`
- Goal conceded (left team): `-10.0`
- Step cost: `-0.005`
- Possession shaping:
  - while left has possession: `+0.01` per step
  - left loses possession: `-0.2`
  - left gains possession: `+0.1`
- Controlled kick outcome:
  - success (goal or teammate receives pass): `+0.2`
  - fail (opponent next touch or out of bounds): `-0.2`

## Training

Default algo is PPO (categorical policy):

```bash
rl-toybox-train --game kick --algo ppo
```

Play AI:

```bash
rl-toybox-play-ai --game kick --model best --render
```

Play user:

```bash
rl-toybox-play-user --game kick
```
