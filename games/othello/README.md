# Othello

Scaffold entry for the repo's intentionally more separate planning and self-play capstone.

## Clip

No embedded clip yet.

## Algorithm / Network

- Planned primary family: MCTS + policy/value net + self-play
- Current scaffold registry exists for taxonomy and future integration only.
- Placeholder policy/value hidden sizes: `[128, 128]`

## Controls (Human)

- Human controls are not implemented yet.
- Future controls should stay board-centric and minimal.

## Observation / Actions

- Placeholder observation: `64` floats
- Placeholder actions: `Discrete(65)`
- Target direction: board-state encoding plus legal-move masking and pass support.

## Environment Notes

- `othello` is intentionally more separate than the arcade-style games.
- It is the capstone entry for planning + self-play.
- This pass only establishes its package, config, README, and search/self-play scaffolding.

## Rewards (Training)

- Placeholder only.
- Final training should center on self-play outcomes and search-guided policy/value targets rather than arcade-style shaping.

## Curriculum (Train)

- Placeholder only.
- Future progression may come from search budget, network size, or self-play league staging rather than a standard environment curriculum.

## Run Commands

```bash
rl-toybox-train --game othello
rl-toybox-play-ai --game othello --model best --render
rl-toybox-play-user --game othello
python -m scripts.train --game othello
python -m scripts.play_ai --game othello --model best --render
python -m scripts.play_user --game othello
```
