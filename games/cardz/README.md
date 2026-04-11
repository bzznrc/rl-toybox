# Cardz

Two-player lane-control card game built around hidden information, scripted opposition, and masked actions. `cardz` is the repo's smallest stochastic card game and a useful actor-critic example for turn-based decisions with legality constraints.

## Clip

No demo clip published yet.

## Default Algorithm / Network

- Algorithm: `a2c` on the shared on-policy actor-critic stack
- Network: shared trunk `[80, 80]`, actor head `16`, critic head `1`

## Controls (Human)

- You control `P1`; `P2` is a scripted opponent
- Click a hand card, then click a lane to play it
- Keyboard shortcuts:
  - `1..5` select hand slot
  - `Q/W/E` target lane
  - `Space` pass
- After the match ends, click anywhere or press `Enter` / `Space` to restart

## Observation / Actions

- Observation: `30` floats (`INPUT_FEATURE_NAMES`, ordered)
  - Global: `turn_norm`, `energy_p1_norm`, `energy_p2_norm`, `score_p1_norm`, `score_p2_norm`, `hand_count_p2_norm`, `phase_code`
  - Lanes: `lane_<i>_power_p1_norm`, `lane_<i>_power_p2_norm`, `lane_<i>_unit_count_p1_norm`, `lane_<i>_unit_count_p2_norm`, `lane_<i>_status_p1`, `lane_<i>_status_p2` for `i in {0,1,2}`
  - Hand: `hand_<j>_card_id` for `j in {0,1,2,3,4}`
- Perspective: `P1` only
- Card id encoding:
  - `0` empty
  - `1` `U1`
  - `2` `U2`
  - `3` `U3`
  - `4` `Atk`
  - `5` `Ban`
- Lane status encoding:
  - `0` none
  - `1` `BAN` only
  - `2` `ATK` only
  - `3` `BAN + ATK`
- `phase_code` encoding:
  - `0` P1 opens the turn exchange
  - `1` P1 is the responder while P2 may still act later in the turn
  - `2` P1 acts uncontested because P2 already passed
- Actions: `Discrete(16)` (`ACTION_NAMES`, ordered)
  - `0..14`: `play_hand_<slot>_lane_<lane>`
  - `15`: `pass`

The observation stays compact and public-information only: P1 sees board state, scores, energy, P2 hand size, and P1's own hand ids, but not P2's hand contents. Action masking still handles legal-play validity, so there are no per-card playable-now flags in the observation.

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
