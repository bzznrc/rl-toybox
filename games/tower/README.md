# Tower

Wave-based tower defense built around build-phase planning, action masking, and delayed rewards. `tower` is the repo's clearest example of a value-based game where the policy acts at decision points instead of every frame.

## Clip

No clip is currently checked into the repo for `tower`.

## Algorithm / Network

- Default algorithm: `dqn`
- IO: `obs=24`, `act=26`
- Default Q-network: `24 -> 64 -> 64 -> 26`
- Runtime shape: masked Double DQN with a dueling head

## Controls (Human)

- `Mouse Left` on an empty slot: open a menu with `Fast`, `Heavy`, and `Area`
- `Mouse Left` on an occupied slot: open a menu with `Upgrade` and `Sell`
- `Mouse Left` on a menu item: apply it if valid
- `Mouse Left` elsewhere: close the active menu
- `Space`: start the previewed wave
- `X`: toggle turret-range ghosts for deployed towers

## Observation / Actions

Canonical `INPUT_FEATURE_NAMES` order:

```python
[
    "glob_gold_norm",
    "glob_lives_norm",
    "glob_wave_norm",
    "glob_acts_left_norm",
    "wave_n_light_norm",
    "wave_n_armored_norm",
    "wave_n_flying_norm",
    "route_shortcut_upper_active",
    "route_shortcut_lower_active",
    "slot_0_kind_id",
    "slot_0_lvl_norm",
    "slot_0_exposure_norm",
    "slot_1_kind_id",
    "slot_1_lvl_norm",
    "slot_1_exposure_norm",
    "slot_2_kind_id",
    "slot_2_lvl_norm",
    "slot_2_exposure_norm",
    "slot_3_kind_id",
    "slot_3_lvl_norm",
    "slot_3_exposure_norm",
    "slot_4_kind_id",
    "slot_4_lvl_norm",
    "slot_4_exposure_norm",
]
```

- `kind_id` encoding: `0=empty`, `1=fast`, `2=heavy`, `3=area`
- `lvl_norm`: `0.0` for empty, otherwise `level / 3`
- `glob_acts_left_norm` is kept as a compatibility feature slot; because Tower no longer uses a build-phase action budget, it is `1.0` during playable build phases.
- `route_shortcut_upper_active` and `route_shortcut_lower_active` show which route setup is active for the next wave.
- `slot_i_exposure_norm` is the main topology feature: it is the normalized route length inside that slot's coverage for the upcoming wave route.
- Runtime slots are always ordered from earliest to latest exposure timing on the base S path.
- `slot_0` is the earliest slot and `slot_4` is the latest cleanup slot.

Action space:

```python
[
    "start_wave",
    "build_fast_0", "build_fast_1", "build_fast_2", "build_fast_3", "build_fast_4",
    "build_heavy_0", "build_heavy_1", "build_heavy_2", "build_heavy_3", "build_heavy_4",
    "build_area_0", "build_area_1", "build_area_2", "build_area_3", "build_area_4",
    "upgrade_0", "upgrade_1", "upgrade_2", "upgrade_3", "upgrade_4",
    "sell_0", "sell_1", "sell_2", "sell_3", "sell_4",
]
```

Action masking is central: invalid builds, upgrades, sells, and unaffordable actions are masked, and `start_wave` is only valid during build phases.

## Environment Notes

The gameplay loop is:

1. Preview the next wave.
2. Build, upgrade, or sell during the build phase.
3. Start the wave.
4. Let the wave auto-simulate.
5. Return to the next build phase.

The policy never acts during the live wave itself.
Build phases no longer use an action budget: you can keep building, upgrading, and selling until you start the wave or run out of credits.

Tower now uses one compact soft-S lane:

- top entry into an upper sweep
- right-side bend into a middle sweep back
- left-side bend into a lower sweep
- short final trunk to the exit

Two shortcuts are always visible on the board:

- `Upper`: drops vertically from the first corner into the fourth, bypassing corners `2` and `3`
- `Lower`: drops vertically from the third corner into the sixth, bypassing corners `4` and `5`

At most one shortcut is active for a wave. Inactive shortcuts are drawn with the normal path styling at reduced opacity; the active shortcut is drawn exactly like the base path, with the fork rendered as one continuous outlined union so the branch reads as a smooth bifurcation.

Like the other repo games with optional helper overlays, Tower also supports an `X`-toggled ghost view during rendered play and eval: each deployed tower shows a restrained translucent circle for its current attack range.

Each wave uses exactly one route mode for all enemies:

- `none`
- `upper`
- `lower`

The next wave's route is chosen before the build phase and previewed in the HUD.

The board still uses 7 candidate tower pads, with 5 active in a run and 2 shown as faded unavailable pads:

- `upper_left_bend`
- `upper_mid_inner`
- `upper_right_bend`
- `mid_left_bend`
- `mid_center_inner`
- `mid_right_bend`
- `lower_trunk`

At run start, the active 5 candidate pads are sorted by path order / exposure timing and remapped to the 5 runtime slot ids:

- `slot_0`
- `slot_1`
- `slot_2`
- `slot_3`
- `slot_4`

### Roles

Enemies:

- `Light`: fast, fragile pressure
- `Armored`: slow, durable pressure
- `Flying`: fast air pressure

Towers:

- `Fast`: strongest into `Flying`, weakest into `Armored`
- `Heavy`: strongest into `Armored`, weakest into `Light`
- `Area`: strongest into `Light`, weakest into `Flying`

Each tower has levels `1` to `3`.

### Economy

- Build cost: `5`
- Level 2 upgrade cost: `5`
- Level 3 upgrade cost: `5`
- Start credits: `12`
- Start lives: `12`
- Wave-clear credits: `+5`
- Kills do not grant credits
- Leak damage: `1` life per enemy

Sell values:

- Level 1 tower: `4`
- Level 2 tower: `8`
- Level 3 tower: `12`

Tower damage no longer uses armor piercing. Damage is now:

```python
raw_damage = base_damage * matchup_multiplier
final_damage = max(0.05, raw_damage - enemy_armor)
```

Spawn order is interleaved instead of grouped by type: Tower cycles remaining pools in `light -> flying -> armored` order and skips exhausted types.

## Rewards (Training)

- `reward_progress_kill = +0.05`
- `reward_event_leak = -0.25`
- `reward_progress_wave_clear = +0.50`
- `reward_terminal_win = +2.00`
- `reward_terminal_loss = -2.00`

Build, upgrade, and sell actions do not receive direct reward. Episode totals are logged through the internal reward breakdown.

## Curriculum (Train)

- Level 1: `start_credits=12`, `start_lives=12`, `num_waves=5`
- Level 2: `start_credits=12`, `start_lives=12`, `num_waves=7`
- Level 3: `start_credits=12`, `start_lives=12`, `num_waves=9`

Wave templates are fully authored in `games/tower/config.py`; Tower no longer scales wave counts procedurally and no longer adds per-wave count jitter.

Authored wave list:

1. `6 light`
2. `4 light, 3 flying`
3. `6 light, 2 armored`
4. `4 light, 2 armored, 2 flying`
5. `8 light, 3 armored, 3 flying`
6. `6 light, 5 armored, 4 flying`
7. `10 light, 5 armored, 5 flying`
8. `12 light, 6 armored, 6 flying`
9. `8 light, 8 armored, 6 flying`

## Run Commands

```bash
rl-toybox-train --game tower
rl-toybox-play-ai --game tower --render
rl-toybox-play-user --game tower
python -m scripts.train --game tower
python -m scripts.play_ai --game tower --render
python -m scripts.play_user --game tower
```

See `games/tower/config.py` and `games/tower/env.py` for the wave plan, rewards, and map-layout logic. Shared DQN defaults live in `core/game.py`.
