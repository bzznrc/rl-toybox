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

- `1-5` or `Left/Right`: select `left`, `upper`, `mid`, `lower`, `right`
- `Mouse Left`: select a slot
- `Mouse Right`: sell the clicked slot
- `A`: select Arrow as the active build blueprint
- `C`: select Cannon as the active build blueprint
- `T`: select Tesla as the active build blueprint
- `Space`: allocate/place the selected tower type on the selected slot
- `U`: upgrade the selected slot
- `Delete` or `Backspace`: delete/sell the selected slot
- `Enter`: start the previewed wave

## Observation / Actions

Tower uses a compact `20`-float observation. The extra feature beyond the original `19`-feature target is `run_actions_left_norm`, which keeps the build-phase action budget explicit for the agent.

Canonical `INPUT_FEATURE_NAMES` order:

```python
[
    "run_gold_norm",
    "run_lives_norm",
    "run_wave_norm",
    "run_actions_left_norm",
    "wave_entry_left",
    "wave_entry_right",
    "wave_count_swarm_norm",
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

- `tower_kind` encoding: `0=empty`, `1=arrow`, `2=cannon`, `3=tesla`
- `tower_level_norm`: `0.0` for empty, otherwise `level / 3`
- `map_layout_id_norm` identifies which of the fixed-layout variants is active this run

The discrete action space is exactly `26` actions:

```python
[
    "start_wave",
    "build_arrow_left", "build_arrow_upper", "build_arrow_mid", "build_arrow_lower", "build_arrow_right",
    "build_cannon_left", "build_cannon_upper", "build_cannon_mid", "build_cannon_lower", "build_cannon_right",
    "build_tesla_left", "build_tesla_upper", "build_tesla_mid", "build_tesla_lower", "build_tesla_right",
    "upgrade_left", "upgrade_upper", "upgrade_mid", "upgrade_lower", "upgrade_right",
    "sell_left", "sell_upper", "sell_mid", "sell_lower", "sell_right",
]
```

Action masking is central:

- invalid builds on occupied slots are masked
- invalid upgrades on empty or level-3 slots are masked
- invalid sell actions on empty slots are masked
- actions that exceed current gold are masked
- once the build-phase action budget is exhausted, only `start_wave` remains valid

## Environment Notes

- The map always uses `2` entry points, `1` shared exit, and the same `5` fixed slots for stable IO.
- Each run samples one of `3` small layouts. The slot coordinates stay fixed; only the lane bends change.
- Waves are previewed before the agent commits. `start_wave` simulates the entire wave internally; the agent never acts per frame.
- Each build phase has a small action budget. That keeps planning meaningful and prevents endless build/sell loops.

Tower and enemy roles are intentionally asymmetric:

- Arrow: fast single-target tower, strongest into `flying`, still weak into `armored`
- Cannon: slow splash / armor-piercing tower, strongest into `armored`, cannot hit `flying`
- Tesla: chain chip-damage tower, strongest into `swarm`, weak into `flying`
- Swarm: fast, low-HP pressure that rewards chaining
- Armored: slow, tanky lane pressure that rewards armor-breaking
- Flying: fast direct pressure that rewards precise anti-air coverage

Selling matters because lane pressure changes between `left`, `right`, and `both`, layout coverage shifts between runs, and the refund decays with commitment:

- level 1 refund: `90%`
- level 2 refund: `75%`
- level 3 refund: `60%`

## Rewards (Training)

Tower keeps a simple scalar reward with named internal components:

- `reward_progress_kill = +0.05`
- `reward_event_leak = -0.25`
- `reward_progress_wave_clear = +0.50`
- `reward_terminal_win = +2.00`
- `reward_terminal_loss = -2.00`

Notes:

- build / upgrade / sell actions do not get direct positive reward
- the reward signal is returned on the wave-resolution transition after `start_wave`
- episode totals are logged through the internal reward breakdown

## Curriculum (Train)

Tower uses the shared 3-level curriculum:

- Level 1: `start_gold=18`, `start_lives=10`, `num_waves=6`, `wave_scale=0.92`
- Level 2: `start_gold=17`, `start_lives=10`, `num_waves=7`, `wave_scale=1.00`
- Level 3: `start_gold=16`, `start_lives=10`, `num_waves=8`, `wave_scale=1.10`

All levels keep the same `6`-action build-phase budget and the same fixed action/feature taxonomy.

## Run Commands

```bash
rl-toybox-train --game tower
rl-toybox-play-ai --game tower --model best --render
rl-toybox-play-user --game tower
python -m scripts.train --game tower
python -m scripts.play_ai --game tower --model best --render
python -m scripts.play_user --game tower
```
