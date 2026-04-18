# Trail

Compact top-down light-cycles duel with continuous forward motion, solid trails, and immediate terminal punishment for bad space control. `trail` is meant to showcase adversarial spatial control, territory pressure, and a game shape that can later grow into fuller self-play.

## Clip

No clip is currently checked into the repo for `trail`.

## Algorithm / Network

- Default algorithm: `ppo`
- IO: `obs=20`, `act=3` by default
- Recommended MLP: `20 -> 32 -> 32 -> 3`
- Tiny preset recommendation: `16 -> 24 -> 24 -> 3`

## Controls (Human)

- Steer absolute with `W/A/S/D` or arrow keys
- Reverse requests are ignored, so human control stays aligned with the RL-relative action model

## Observation / Actions

- Observation family: arcade / egocentric `SELF -> SENS -> OPP -> MAP -> FLAG`
- Observation: compact vector-only local duel state (`INPUT_FEATURE_NAMES`, exact order)
- Default preset (`TRAIL_OBS_PRESET=default`, `obs=20`)
  - `SELF` (2): `self_dir_x self_dir_y`
  - `SENS` (6): `sens_fwd sens_left sens_right sens_back sens_fwd_left sens_fwd_right`
  - `OPP` (6): `opp_dx opp_dy opp_dir_x opp_dir_y opp_dist_norm opp_fwd_align`
  - `MAP` (5): `map_area_left_norm map_area_straight_norm map_area_right_norm map_area_adv_norm map_fill_ratio_norm`
  - `FLAG` (1): `flag_time_norm`
- Tiny preset (`TRAIL_OBS_PRESET=tiny`, `obs=16`) drops:
  - `sens_fwd_left`
  - `sens_fwd_right`
  - `map_area_adv_norm`
  - `flag_time_norm`
- Actions: `Discrete(3)` (`ACTION_NAMES`, ordered)
  - `0 turn_left`
  - `1 go_straight`
  - `2 turn_right`

The observation stays compact by mixing only three kinds of signal: immediate collision geometry, relative opponent state, and one-step action-conditioned territory estimates. The `map_area_*_norm` features are flood-fill reachability estimates after each immediate candidate turn, so the policy gets a small amount of spatial look-ahead without needing pixels or large handcrafted state.

## Environment Notes

- Both riders move one cell every step and always leave a persistent solid trail.
- Collision with the arena wall or any occupied trail cell is terminal for that rider.
- Moving into the same empty cell on the same frame is treated as a draw.
- The scripted opponent is deterministic per seed and uses the same immediate space-control helpers as the player observation pipeline.
- Reset randomness only changes spawn lane offsets and opponent opening choices; step-to-step game dynamics are deterministic given actions and seed.
- Arena footprint matches the larger Bang / Vroom window size.
- `train` runs `1` duel per episode.
- `eval` and `human` run `10` duels per episode, with winner history shown in the bottom bar.

## Rewards (Training)

- `REWARD_WIN = +1.0` on win
- `PENALTY_LOSE = -1.0` on loss
- `REWARD_DRAW = 0.0` on simultaneous crash or timeout draw
- No shaping reward is added between terminal events

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Difficulty comes from a stronger deterministic opponent evaluation, not from extra mechanics
- Levels:
  - Level 1: basic space-first opponent
  - Level 2: stronger area-advantage opponent
  - Level 3: strongest pressure-aware opponent

An episode counts as a success if the player wins the duel.

## Run Commands

```bash
rl-toybox-train --game trail
rl-toybox-play-ai --game trail --render
rl-toybox-play-user --game trail
python -m scripts.train --game trail
python -m scripts.play_ai --game trail --render
python -m scripts.play_user --game trail
```

PowerShell preset switch:

```powershell
$env:TRAIL_OBS_PRESET='tiny'; python -m scripts.train --game trail
```

See `games/trail/config.py` for the observation presets, opponent weights, and reward constants. Shared PPO defaults live in `core/game.py`.
