# Bang

Top-down arena shooter environment.

## Controls (Human)

- Move: `W/A/S/D`
- Aim: mouse (preferred) or `Q/E` (also left/right arrows)
- Shoot: `Space` or left mouse button

## Observation / Actions

- Observation: `24` floats (`games/bang/config.py::INPUT_FEATURE_NAMES`)
- Actions: `Discrete(8)`
  - `0 move_up`
  - `1 move_down`
  - `2 move_left`
  - `3 move_right`
  - `4 stop_move`
  - `5 aim_left`
  - `6 aim_right`
  - `7 shoot`

## Rewards (Training)

- Time step: `-0.005`
- Bad shot (shoot with no LOS): `-0.1`
- Blocked move: `-0.1`
- Hit enemy: `+5.0` per hit
- Win: `+20.0`
- Lose: `-10.0`

## Training

Default algo is enhanced DQN (double + dueling + prioritized replay):

```bash
rl-toybox-train --game bang
```

Play AI:

```bash
rl-toybox-play-ai --game bang --model best --render
```

Play user:

```bash
rl-toybox-play-user --game bang
```
