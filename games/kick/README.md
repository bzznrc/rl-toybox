# Kick

11v11 football environment (Arcade-style top-down) with PPO-oriented discrete actions.

## Controls (Human)

- Move: `W/A/S/D` (or arrows)
- Aim: mouse
- Shoot: hold/release left mouse button (low/mid/high by hold time)
- Switch controlled player: `Tab` (closest to ball)

## Observation / Actions

- Observation: `44` floats
  - Self (10): `vx, vy, cos(theta), sin(theta), has_ball, role_scalar, stamina, stamina_delta, in_contact, pad0`
  - Ball (6): `ball_dx, ball_dy, ball_vx, ball_vy, ball_is_free, ball_owner_team`
  - Goals (4): `opp_goal_dx, opp_goal_dy, own_goal_dx, own_goal_dy`
  - Teammates (12): nearest 3 x (`dx, dy, vx, vy`)
  - Opponents (12): nearest 3 x (`dx, dy, vx, vy`)
- Actions: `Discrete(12)`
  - Movement:
    - `0 STAY`
    - `1 MOVE_N`
    - `2 MOVE_NE`
    - `3 MOVE_E`
    - `4 MOVE_SE`
    - `5 MOVE_S`
    - `6 MOVE_SW`
    - `7 MOVE_W`
    - `8 MOVE_NW`
  - Kicks:
    - `9 KICK_LOW`
    - `10 KICK_MID`
    - `11 KICK_HIGH`

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
