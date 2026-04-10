# Cardz

Tiny 2-player lane-control card game built as the repo's compact stochastic hidden-info actor-critic showcase.

## Clip

No embedded clip yet.

## Algorithm / Network

- Algo: `a2c` on the shared on-policy actor-critic stack
- Role: compact hidden-information / masked-action showcase
- Network: shared trunk `32 -> 96 -> 96`, actor head `16`, critic head `1`

## Controls (Human)

- Click a hand card, then click a lane to play it.
- Keyboard: `1..5` select hand slot, `Q/W/E` target lane, `Space` passes.
- After the match ends, click anywhere or press `Enter` / `Space` to restart.

## Observation / Actions

- Observation: `32` floats (`INPUT_FEATURE_NAMES`, ordered)
  - Global: `turn_norm`, `energy_self_norm`, `energy_opp_norm`, `score_self_norm`, `score_opp_norm`
  - Lanes: `lane_<i>_power_self_norm`, `lane_<i>_power_opp_norm`, `lane_<i>_count_self_norm`, `lane_<i>_count_opp_norm` for `i in {0,1,2}`
  - Hand: `hand_<j>_type`, `hand_<j>_cost_norm`, `hand_<j>_value_norm` for `j in {0,1,2,3,4}`
- Perspective: current player only
- Card type encoding: `0` empty, `1` `U1`, `2` `U2`, `3` `U3`, `4` `Atk`, `5` `Ban`
- Lane power features use current scoring totals, including this turn's temporary tactic modifiers, normalized by `18`
- Actions: `Discrete(16)` (`ACTION_NAMES`, ordered)
  - `0..14`: `play_hand_<slot>_lane_<lane>` with slots `0..4` and lanes `0..2`
  - `15`: `pass`
- Action masking disables empty hand slots, unaffordable cards, unit plays into full lanes, `Atk` on lanes with no enemy units, and `Ban` on lanes with no friendly units or an existing friendly banner in that lane

## Environment Notes

- Match structure: `2` players, `3` lanes, `5` turns, energy rising from `1` to `5`, with the opening pattern randomized between `AB / BA / AB / BA / AB` and `BA / AB / BA / AB / BA`
- Cards:
  - Units: `U1` cost `1`, power `2`; `U2` cost `2`, power `3`; `U3` cost `3`, power `4`
  - `ATK`: cost `1`, applies `-2` enemy lane power for the current turn only; requires an enemy unit in the lane
  - `BAN`: cost `1`, can be placed in any friendly lane once per lane, and contributes `+1` permanent lane power only while that side has at least one unit there
- Units persist on board; `BAN` persists as a permanent lane investment; `ATK` only affects the current turn's scoring
- Each side has at most `2` unit slots in each lane
- Each turn is split into two contiguous subturns: the lead player acts until pressing `Space`, then the other player acts until pressing `Space`
- Players may play multiple cards during their active subturn as long as they still have energy, legal plays, and have not passed
- Control does not auto-switch after card plays; `Space` is what ends the current player's subturn
- Lane scoring happens only after both players have passed: higher current lane total gets `1` point, tied lane gives `0` points to both players
- End-of-turn order is: score lanes, clear temporary effects, draw `1` card for each player if hand size is below `5`, then begin the next turn
- Draw system is an infinite probabilistic pool:
  - `U1`: `0.30`
  - `U2`: `0.20`
  - `U3`: `0.15`
  - `ATK`: `0.175`
  - `BAN`: `0.175`
- Hands start at `5` cards and refill by `1` only after turn-end scoring and cleanup if below max hand size

## Rewards (Training)

- `reward_progress_turn_points`: `+0.2` per lane point scored that turn and `-0.2` per lane point conceded
- `reward_terminal_match_win`: `+1.0`
- `reward_terminal_match_draw`: `0.0`
- `reward_terminal_match_loss`: `-1.0`
- No extra reward is given for playing cards, spending energy, or other intermediate events

## Curriculum (Train)

- Fixed single ruleset only
- `level=1` is the full game
- Entropy coefficient for the default run is `0.015`

## Run Commands

```bash
rl-toybox-train --game cardz
rl-toybox-play-ai --game cardz --model best --render
rl-toybox-play-user --game cardz
python -m scripts.train --game cardz
python -m scripts.play_ai --game cardz --model best --render
python -m scripts.play_user --game cardz
```
