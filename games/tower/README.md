# Tower

Wave-based tower defense built around build-phase planning, action masking, and delayed rewards. `tower` is the repo's clearest example of a value-based game where the policy acts at decision points instead of every frame.

## Default Algorithm / Network

- Algorithm family: value-based discrete control
- Default algorithm: `dqn`
- Recommended runtime shape: masked Double DQN with a dueling head
- Hidden sizes: `[64, 64]`
- Observation size: `20`
- Action count: `26`

## Controls (Human)

- `Mouse Left` on an empty slot: open a menu with `Fast`, `Heavy`, and `Area`
- `Mouse Left` on an occupied slot: open a menu with `Upgrade` and `Sell`
- `Mouse Left` on a menu item: apply it if valid
- `Mouse Left` elsewhere: close the active menu
- `Space`: start the previewed wave

## Observation / Actions

Canonical `INPUT_FEATURE_NAMES` order:

```python
[
    "run_gold_norm",
    "run_lives_norm",
    "run_wave_norm",
    "run_actions_left_norm",
    "wave_entry_left",
    "wave_entry_right",
    "wave_count_light_norm",
    "wave_count_armored_norm",
    "wave_count_flying_norm",
    "map_layout_id_norm",
    "slot_left_tower_kind",
    "slot_left_tower_level_norm",
    "slot_upper_tower_kind",
    "slot_upper_tower_level_norm",
    "slot_mid_tower_kind",
    "slot_mid_tower_level_norm",
    "slot_lower_tower_kind",
    "slot_lower_tower_level_norm",
    "slot_right_tower_kind",
    "slot_right_tower_level_norm",
]
```

- `tower_kind` encoding: `0=empty`, `1=fast`, `2=heavy`, `3=area`
- `tower_level_norm`: `0.0` for empty, otherwise `level / 3`
- `run_gold_norm` is the legacy feature name; the UI surfaces the same value as `Credits`

Action space:

```python
[
    "start_wave",
    "build_fast_left", "build_fast_upper", "build_fast_mid", "build_fast_lower", "build_fast_right",
    "build_heavy_left", "build_heavy_upper", "build_heavy_mid", "build_heavy_lower", "build_heavy_right",
    "build_area_left", "build_area_upper", "build_area_mid", "build_area_lower", "build_area_right",
    "upgrade_left", "upgrade_upper", "upgrade_mid", "upgrade_lower", "upgrade_right",
    "sell_left", "sell_upper", "sell_mid", "sell_lower", "sell_right",
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

Each run picks one of two handcrafted layouts:

- `Soft S Merge`
- `Offset S`

The five slot ids keep the same semantic roles across both layouts:

- `left`
- `upper`
- `mid`
- `lower`
- `right`

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
- Level 2 upgrade cost: `4`
- Level 3 upgrade cost: `7`
- Start credits: `12`
- Wave-clear credits: `+6`
- Kills do not grant credits

Sell refund rates:

- Level 1: `90%`
- Level 2: `75%`
- Level 3: `60%`

## Rewards (Training)

- `reward_progress_kill = +0.05`
- `reward_event_leak = -0.25`
- `reward_progress_wave_clear = +0.50`
- `reward_terminal_win = +2.00`
- `reward_terminal_loss = -2.00`

Build, upgrade, and sell actions do not receive direct reward. Episode totals are logged through the internal reward breakdown.

## Curriculum (Train)

- Level 1: `start_credits=12`, `start_lives=10`, `num_waves=6`, `wave_scale=1.00`
- Level 2: `start_credits=12`, `start_lives=10`, `num_waves=7`, `wave_scale=1.10`
- Level 3: `start_credits=12`, `start_lives=10`, `num_waves=8`, `wave_scale=1.20`
- All levels use `decision_budget=6`

## Run Commands

```bash
rl-toybox-train --game tower
rl-toybox-play-ai --game tower --model best --render
rl-toybox-play-user --game tower
python -m scripts.train --game tower
python -m scripts.play_ai --game tower --model best --render
python -m scripts.play_user --game tower
```

See `games/tower/config.py` and `games/tower/env.py` for the full wave plan, reward constants, and map-layout logic.
