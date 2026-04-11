# Osero

Compact Osero/Reversi implementation built around self-play, MCTS, and a small policy/value network. It is the planning-oriented game in the repo and the clearest example of search-guided training.

## Default Algorithm / Network

- Algorithm family: `search_play`
- Search: lightweight PUCT MCTS with policy priors and value estimates from the shared network
- Training flow: self-play games produce MCTS visit-count policy targets and final-outcome value targets
- Default board: `6x6`
- Supported boards: `4x4`, `6x6`, `8x8`
- Default networks:
  - `4x4`: input `16`, trunk `[64, 64]`, policy `17`, value `1`
  - `6x6`: input `36`, trunk `[96, 96]`, policy `37`, value `1`
  - `8x8`: input `64`, trunk `[128, 128]`, policy `65`, value `1`

## Controls (Human)

- `Mouse Left` on a highlighted legal square: place a stone
- Passes are automatic when no legal move exists
- `Space` or `Enter` after the game ends: restart

## Observation / Actions

- Observation is the flattened board only, row-major, from the current player to move
- Cell encoding:
  - `0` empty
  - `1` current player's stone
  - `-1` opponent stone
- Legal moves are not embedded into the observation; they are exposed through action masking

Canonical feature order follows `INPUT_FEATURE_NAMES`:

- `4x4`: `cell_r0_c0` through `cell_r3_c3`
- `6x6`: `cell_r0_c0` through `cell_r5_c5`
- `8x8`: `cell_r0_c0` through `cell_r7_c7`

Action space is one action per board cell plus one pass action:

- `4x4`: `17`
- `6x6`: `37`
- `8x8`: `65`

Action names follow row-major board order plus `pass`.

## Environment Notes

- Rules follow standard Osero/Reversi:
  - legal move generation
  - directional flipping
  - forced pass when no legal move exists
  - terminal detection when neither side can move
  - winner from final stone counts
- Black moves first.
- Rendering uses the same square-tile visual language as the rest of the repo instead of glossy discs.

## Rewards (Training)

- Non-terminal reward: `0`
- Terminal win: `+1`
- Terminal draw: `0`
- Terminal loss: `-1`

The search trainer primarily learns from normalized MCTS visit counts for the policy target and final game outcome for the value target.

## Board Size Selection

There is no environment curriculum for `osero`. Board sizes are separate runtime choices:

- `4x4` is the smallest option and a useful easy/debug board
- `6x6` is the default recommended setup
- `8x8` is supported with a larger model

Choose the board before running train or play commands:

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

See `games/osero/config.py` and `games/osero/rules.py` for board-size selection, reward settings, and the compact rules implementation.
