# Osero

Compact Osero/Reversi built as the repo's planning + self-play capstone.
The implementation stays intentionally small: flattened board input, one compact policy/value MLP, MCTS, and self-play.

## Clip

No embedded clip yet.

## Algorithm / Network

- Algo: `search_play`
- Search: lightweight PUCT MCTS with policy priors and value rollouts from the shared net
- Training shape: self-play games -> MCTS visit-count policy targets + final outcome value targets
- Default board: `6x6`
- Supported boards: `4x4` (`easy`), `6x6`, `8x8`
- Default network:
  - `4x4`: input `16`, trunk `[64, 64]`, policy `17`, value `1`
  - `6x6`: input `36`, trunk `[96, 96]`, policy `37`, value `1`
  - `8x8`: input `64`, trunk `[128, 128]`, policy `65`, value `1`

## Controls (Human)

- `Mouse Left` on a highlighted legal square: place a stone there
- Passes are automatic in human mode when no legal move exists
- `Space` or `Enter` after the game ends: restart
- Winner, turn, and stone counts stay visible in the HUD

## Observation / Actions

- Observation is the flattened board only, row-major, from the current player to move
- Cell encoding:
  - `0` empty
  - `1` current player's stone
  - `-1` opponent stone
- Legal moves are not encoded into the observation; they are exposed through action masking

Canonical feature order comes from `INPUT_FEATURE_NAMES`:

- `4x4`: `cell_r0_c0` through `cell_r3_c3` -> `16` inputs
- `6x6`: `cell_r0_c0` through `cell_r5_c5` -> `36` inputs
- `8x8`: `cell_r0_c0` through `cell_r7_c7` -> `64` inputs

Action space is one action per board cell plus one pass action:

- `4x4`: `16 + 1 = 17`
- `6x6`: `36 + 1 = 37`
- `8x8`: `64 + 1 = 65`

Action names follow row-major board order plus `pass`.

## Environment Notes

- Rules are standard Osero/Reversi:
  - legal move generation
  - directional flipping
  - forced pass when no move exists
  - terminal detection when neither side can move
  - winner from final stone counts
- Black moves first.
- Human rendering stays in the repo's restrained square-tile language rather than classic glossy discs.
- Stones are square Bang-like markers built from the same two-tone tile treatment used elsewhere in the repo.

## Rewards (Training)

- Non-terminal reward: `0`
- Terminal win: `+1`
- Terminal draw: `0`
- Terminal loss: `-1`

The search trainer mainly learns from:

- policy target = normalized MCTS visit counts
- value target = final game outcome from the acting player's perspective

## Curriculum (Train)

- No environment curriculum is used.
- `4x4` is the explicit `easy` board-size option and uses its own `16`-input / `[64, 64]` model.
- The default recommended setup remains `6x6` because it keeps search, replay, and model size small while staying closer to standard play.
- `4x4`, `6x6`, and `8x8` are separate board-size choices, not curriculum levels.
- Choose the target board before running train/play commands:

```bash
OSERO_BOARD_SIZE=4 rl-toybox-train --game osero
OSERO_BOARD_SIZE=6 rl-toybox-train --game osero
OSERO_BOARD_SIZE=8 rl-toybox-train --game osero
```

## Run Commands

```bash
rl-toybox-train --game osero
rl-toybox-play-ai --game osero --model best --render
rl-toybox-play-user --game osero
python -m scripts.train --game osero
python -m scripts.play_ai --game osero --model best --render
python -m scripts.play_user --game osero
```
