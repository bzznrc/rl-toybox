# Card

Scaffold entry for a compact stochastic hidden-information actor-critic game.

## Clip

No embedded clip yet.

## Algorithm / Network

- Planned primary family: A2C
- Current scaffold default: `a2c`
- Placeholder hidden sizes: `[64, 64]`

## Controls (Human)

- Human controls are not implemented yet.
- The future design should remain simple enough for quick turn-based interaction.

## Observation / Actions

- Placeholder observation: `14` floats
- Placeholder actions: `Discrete(4)`
- Target direction: small hidden-info state encoding with a short discrete betting/decision action set.

## Environment Notes

- This is scaffold-only in this pass.
- Future role: simple stochastic hidden-info actor-critic showcase.
- Keep the eventual game tiny and legible rather than simulation-heavy.

## Rewards (Training)

- Placeholder only.
- Final reward design should focus on outcome value, risk/reward clarity, and stochastic decision quality.

## Curriculum (Train)

- Placeholder only.
- Likely progression: deck simplification, opponent complexity, or horizon length.

## Run Commands

```bash
rl-toybox-train --game card
rl-toybox-play-ai --game card --model best --render
rl-toybox-play-user --game card
python -m scripts.train --game card
python -m scripts.play_ai --game card --model best --render
python -m scripts.play_user --game card
```
