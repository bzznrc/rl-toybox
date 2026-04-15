# Cardz

Two-player lane-control card game built around hidden information, scripted opposition, and masked actions. `cardz` is the repo's smallest stochastic card game and a useful actor-critic example for turn-based decisions with legality constraints.

## Clip

No clip is currently checked into the repo for `cardz`.

## Algorithm / Network

- Default algorithm: `a2c`
- IO: `obs=64`, `act=16`
- Default network: shared actor-critic backbone `64 -> 64 -> 64` with direct policy/value heads
- Backbone sharing: `SHARE_BACKBONE=True`

## Controls (Human)

- You control `P1`; `P2` is a scripted opponent
- Click a hand card, then click a lane to play it
- Keyboard shortcuts:
  - `1..5` select hand slot
  - `Q/W/E` target lane
  - `Space` pass
- After the match ends, click anywhere or press `Enter` / `Space` to restart

## Observation / Actions

- Structured taxonomy fit: `GLOB -> PHASE -> LANE -> HAND -> LEGAL`
- The current implementation keeps `LEGAL` in the action mask rather than the observation.
- Observation: `64` floats (`INPUT_FEATURE_NAMES`, ordered)
- Categorical public fields were expanded into explicit flags / one-hot blocks:
  - `PHASE`: one-hot over `phase_open`, `phase_resp`, `phase_free`
  - lane status: explicit `lane_<i>_p<player>_has_ban` and `lane_<i>_p<player>_has_atk`
  - each `HAND` slot: one-hot over `empty`, `u1`, `u2`, `u3`, `atk`, `ban`
- Ordered features:
  - `GLOB (1-7)`: `glob_turn_norm`, `glob_energy_p1_norm`, `glob_energy_p2_norm`, `glob_score_p1_norm`, `glob_score_p2_norm`, `glob_hand_count_p1_norm`, `glob_hand_count_p2_norm`
  - `PHASE (8-10)`: `phase_open`, `phase_resp`, `phase_free`
  - `LANE numeric (11-22)`: `lane_0_power_p1_norm`, `lane_0_power_p2_norm`, `lane_0_unit_count_p1_norm`, `lane_0_unit_count_p2_norm`, `lane_1_power_p1_norm`, `lane_1_power_p2_norm`, `lane_1_unit_count_p1_norm`, `lane_1_unit_count_p2_norm`, `lane_2_power_p1_norm`, `lane_2_power_p2_norm`, `lane_2_unit_count_p1_norm`, `lane_2_unit_count_p2_norm`
  - `LANE status (23-34)`: `lane_0_p1_has_ban`, `lane_0_p1_has_atk`, `lane_0_p2_has_ban`, `lane_0_p2_has_atk`, `lane_1_p1_has_ban`, `lane_1_p1_has_atk`, `lane_1_p2_has_ban`, `lane_1_p2_has_atk`, `lane_2_p1_has_ban`, `lane_2_p1_has_atk`, `lane_2_p2_has_ban`, `lane_2_p2_has_atk`
  - `HAND (35-64)`: `hand_0_empty`, `hand_0_u1`, `hand_0_u2`, `hand_0_u3`, `hand_0_atk`, `hand_0_ban`, `hand_1_empty`, `hand_1_u1`, `hand_1_u2`, `hand_1_u3`, `hand_1_atk`, `hand_1_ban`, `hand_2_empty`, `hand_2_u1`, `hand_2_u2`, `hand_2_u3`, `hand_2_atk`, `hand_2_ban`, `hand_3_empty`, `hand_3_u1`, `hand_3_u2`, `hand_3_u3`, `hand_3_atk`, `hand_3_ban`, `hand_4_empty`, `hand_4_u1`, `hand_4_u2`, `hand_4_u3`, `hand_4_atk`, `hand_4_ban`
- Perspective: `P1` only
- Phase semantics:
  - `phase_open`: P1 opens the turn exchange
  - `phase_resp`: P1 is the responder while P2 may still act later in the turn
  - `phase_free`: P1 acts uncontested because P2 already passed
- Actions: `Discrete(16)` (`ACTION_NAMES`, ordered)
  - `0..14`: `play_hand_<slot>_lane_<lane>`
  - `15`: `pass`

The observation stays public-information only: P1 sees board state, scores, energy, both public hand counts, and P1's own hand contents, but not P2's hand contents. Action masking still handles legal-play validity, so there are no per-card playable-now flags in the observation.

Action masking disables empty hand slots, unaffordable cards, unit plays into full lanes, `Atk` on lanes where that side already has an active attack buff, and `Ban` on lanes with no friendly units or an existing friendly banner.

## Environment Notes

- Match structure: `2` players, `3` lanes, `5` turns
- Energy rises from `1` to `5`
- Training, eval, and human play are all `P1` versus scripted `P2`
- Turn order alternates between two opening patterns:
  - `AB / BA / AB / BA / AB`
  - `BA / AB / BA / AB / BA`

### Cards

- Units:
  - `U1`: cost `1`, power `2`
  - `U2`: cost `2`, power `3`
  - `U3`: cost `3`, power `4`
- `ATK`: cost `1`, gives `+2` temporary lane power for the current turn and is capped at one active `ATK` per lane per side
- `BAN`: cost `1`, requires at least one friendly unit in the lane, can be placed once per lane, and contributes `+1` persistent lane power while that side still has units there

### Turn Flow

- Units persist on board.
- `BAN` is a persistent lane investment.
- `ATK` only affects the current turn's scoring.
- Each side can occupy at most `2` unit slots per lane.
- The active player may make multiple legal plays before passing.
- Lane scoring happens only after both players pass.
- In rendered play there is a short pause after both players pass so temporary attack modifiers remain visible before resolution.

### Draw System

- The deck is an infinite probabilistic pool:
  - `U1`: `0.30`
  - `U2`: `0.20`
  - `U3`: `0.15`
  - `ATK`: `0.175`
  - `BAN`: `0.175`
- `P1` starts with `5` cards and refills by `1` up to `5`
- `P2` uses the curriculum hand cap for both opening hand size and refills

## Rewards (Training)

- `reward_progress_turn_points`: `+0.2` per lane point scored that turn and `-0.2` per lane point conceded
- `reward_terminal_match_win = +1.0`
- `reward_terminal_match_draw = 0.0`
- `reward_terminal_match_loss = -1.0`
- There is no extra reward for simply playing cards or spending energy

## Curriculum (Train)

- Shared 3-level curriculum progression from `core/curriculum.py`
- Promotion settings live in `games/cardz/config.py` under `CURRICULUM_PROMOTION`
- Levels:
  - Level 1: `entropy_coef=0.015`, `opp_max_hand=3`, `opp_random_move_prob=0.75`
  - Level 2: `entropy_coef=0.010`, `opp_max_hand=4`, `opp_random_move_prob=0.35`
  - Level 3: `entropy_coef=0.005`, `opp_max_hand=5`, `opp_random_move_prob=0.10`

The scripted opponent uses a simple lane-value heuristic, with `opp_random_move_prob` controlling how often it replaces its preferred move with a random legal action.

## Run Commands

```bash
rl-toybox-train --game cardz
rl-toybox-play-ai --game cardz --model best --render
rl-toybox-play-user --game cardz
python -m scripts.train --game cardz
python -m scripts.play_ai --game cardz --model best --render
python -m scripts.play_user --game cardz
```

See `games/cardz/config.py` for the full card tables, reward constants, and curriculum settings.
