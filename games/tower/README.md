# Tower

Tiny wave-based tower defense built as the repo's masked-DQN showcase.

## Clip

No embedded clip yet.

## Algorithm / Network

- Primary family: value-based discrete control
- Default algorithm: `dqn`
- Recommended runtime shape: masked Double DQN with dueling head
- Default hidden sizes: `[64, 64]`
- Observation size: `20`
- Action count: `26`

## Controls (Human)

- `Mouse Left` on an empty slot: open a tiny build menu with `Fast`, `Heavy`, and `Area`
- `Mouse Left` on an occupied slot: open a tiny action menu with `Upgrade` and `Sell`
- `Mouse Left` on a menu item: apply that action if it is currently valid
- `Mouse Left` elsewhere: close the current menu
- `Space`: start the previewed wave

The human UI is mouse-first and uses small contextual menus instead of keyboard navigation.

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
- `run_gold_norm` keeps its legacy feature name for compatibility, but the user-facing economy is shown as `Credits`

The discrete action space is exactly `26` actions:

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

Action masking is central:

- invalid builds on occupied slots are masked
- invalid upgrades on empty or level-3 slots are masked
- invalid sells on empty slots are masked
- actions that exceed current credits are masked
- once the build-phase action budget is exhausted, only `start_wave` remains valid
- `start_wave` is only valid during build phases

## Environment Notes

1. Preview the next wave.
2. Build, upgrade, or sell during the build phase.
3. Press `Space` to start the wave.
4. The wave auto-simulates.
5. Return to the next build phase.

Tower never asks the agent to act per frame during the active wave.

Each run picks one of two compact handcrafted map templates:

- `Soft S Merge`: mirrored inward sweeps that meet at a shared center trunk
- `Offset S`: one side joins the center earlier while the other descends farther before merging

The five stable slot ids keep the same semantic roles across both templates:

- `left`: left corner control
- `upper`: left shared / merge coverage
- `mid`: center trunk cleanup
- `lower`: right shared / merge coverage
- `right`: right corner control

The extra input is `run_actions_left_norm`, which exposes the remaining build-phase action budget to the policy.

### Roles

Enemies:

- `Light`: fast, fragile pressure
- `Armored`: slow, durable pressure
- `Flying`: fast air pressure

Towers:

- `Fast`: fast single-target, strongest into `Flying`, weak into `Armored`
- `Heavy`: slow hard-hitting single-target, strongest into `Armored`, weak into `Light`
- `Area`: splash damage, strongest into `Light`, weak into `Flying`

Each tower has levels `1` to `3`.

Selling matters because wave entry side and enemy mix shift between waves, while the lane geometry and slot pressure shift between runs.

Refunds:

- level 1: `90%`
- level 2: `75%`
- level 3: `60%`

### Economy / Balance

Tower uses one shared economy ladder across all tower types:

- build cost: `5`
- level 2 upgrade cost: `4`
- level 3 upgrade cost: `7`
- start credits: `12` at every curriculum level
- wave clear credits: `+6` after every cleared wave
- kill rewards do not grant credits, so the economy stays fixed from run to run

This means each build phase roughly funds one new tower or one upgrade, while higher curriculum levels get harder through extra waves and stronger enemy counts rather than reduced income.

## Rewards (Training)

Named internal reward components:

- `reward_progress_kill = +0.05`
- `reward_event_leak = -0.25`
- `reward_progress_wave_clear = +0.50`
- `reward_terminal_win = +2.00`
- `reward_terminal_loss = -2.00`

Notes:

- build, upgrade, and sell actions do not get direct reward
- masked invalid actions do not need separate penalties
- episode totals are logged through the internal reward breakdown

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
