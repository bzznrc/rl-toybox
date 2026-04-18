# Fuse

Compact bomb-duel / free-for-all survival on a tile arena with destructible crates, fixed-fuse bombs, and chain reactions. `fuse` is the repo's masked value-discrete showcase for delayed reward, trap-setting, destructible terrain, and bomb-timing pressure.

## Clip

No clip is currently checked into the repo for `fuse`.

## Algorithm / Network

- Default algorithm: `dqn`
- IO: `obs=34`, `act=6`
- Default Q-network: `34 -> 64 -> 64 -> 6`
- Runtime shape: masked Double DQN with a dueling head

## Controls (Human)

- Move: `W/A/S/D` or arrow keys
- Place bomb: `Space`
- If no move key is held for a frame, the player uses `move_stop`

## Observation / Actions

Ordered `INPUT_FEATURE_NAMES`:

```python
[
    "self_bombs_norm",
    "self_bomb_cd_norm",
    "self_on_bomb",
    "self_can_place_bomb",
    "sens_free_up_norm",
    "sens_free_down_norm",
    "sens_free_left_norm",
    "sens_free_right_norm",
    "sens_box_up_norm",
    "sens_box_down_norm",
    "sens_box_left_norm",
    "sens_box_right_norm",
    "opp1_dx",
    "opp1_dy",
    "opp1_same_row",
    "opp1_same_col",
    "opp2_dx",
    "opp2_dy",
    "opp2_same_row",
    "opp2_same_col",
    "opp3_dx",
    "opp3_dy",
    "opp3_same_row",
    "opp3_same_col",
    "map_safe_up_norm",
    "map_safe_down_norm",
    "map_safe_left_norm",
    "map_safe_right_norm",
    "haz_here_tti_norm",
    "haz_post_bomb_escape_norm",
    "flag_bomb_value_norm",
    "flag_can_hit_opp_now",
    "flag_crates_left_norm",
    "flag_time_norm",
]
```

- Opponents are slotted by nearest distance, then fixed player order.
- Missing opponent slots stay zero-filled on lower curriculum levels.
- `map_safe_*_norm` estimates directional access to survivable space under the current bomb timeline, not just immediate walkability.
- `haz_here_tti_norm` measures current-tile danger from existing bombs and flames only.
- `haz_post_bomb_escape_norm` estimates whether planting now still leaves an escape route.

Ordered `ACTION_NAMES`:

```python
[
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_stop",
    "place_bomb",
]
```

Action masking disables:

- moves into solid walls, crates, bombs, and blocked occupied tiles
- `place_bomb` when the player has no bomb available or is already standing on a bomb

## Environment Notes

- The arena uses two permanent obstacle types:
- solid border walls and fixed interior blocks
- destructible crates that block movement and stop blasts
- Bomb blasts are cross-shaped, stop on solid walls, destroy crates, kill players, and chain-react nearby bombs immediately.
- The player can stand on the bomb they just planted until they step off it, but cannot walk back onto bombs afterward.
- Episodes end on last-player-standing or timeout.
- The learned policy always controls `P1`; the remaining players are scripted FFA opponents that scale with the curriculum.

## Rewards (Training)

- `REWARD_WIN = +10.0`
- `PENALTY_LOSE = -8.0`
- `REWARD_ELIM = +1.5` when `P1`'s bomb explosion eliminates an opponent
- `REWARD_CRATE = +0.05` per crate broken by `P1`'s bombs
- `PENALTY_STEP = -0.0025` each step

Terminal outcome dominates the episode total. Crate and elimination rewards stay supportive only, and episode reward components are logged through the shared breakdown path.

## Curriculum (Train)

- Level 1: `2` players total, more open arena, lower crate density, simpler scripted pressure
- Level 2: `3` players total, more crates and choke points, smarter FFA scripting
- Level 3: `4` players total, densest arena, strongest trap-focused scripting, more chain-reaction pressure

## Run Commands

```bash
rl-toybox-train --game fuse
rl-toybox-play-ai --game fuse --render
rl-toybox-play-user --game fuse
python -m scripts.train --game fuse
python -m scripts.play_ai --game fuse --render
python -m scripts.play_user --game fuse
```

See `games/fuse/config.py` for IO, rewards, runtime constants, and curriculum settings. Shared DQN defaults live in `core/game.py`.
