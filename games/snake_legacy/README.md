# Snake AI

## Overview
Minimal, local-only Snake reinforcement learning project using `LinearQNet`.

## Quickstart
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

## Run
```bash
python -m snake_ai
python -m snake_ai.play_user
python -m snake_ai.play_ai
python -m snake_ai.train_ai
```

## Project Layout
- `src/snake_ai/config.py`: central configuration and constants
- `src/snake_ai/game.py`: game logic and human/training game modes
- `src/snake_ai/model.py`: `LinearQNet` and `QTrainer`
- `src/snake_ai/train_ai.py`: RL training loop
- `src/snake_ai/play_user.py`: human play entrypoint
- `src/snake_ai/play_ai.py`: greedy model play entrypoint
- `src/snake_ai/runtime.py`: arcade runtime primitives

## RL Inputs/Outputs
- State input size: `12`
- State feature 1: `danger_straight`
- State feature 2: `danger_right`
- State feature 3: `danger_left`
- State feature 4: `dir_left`
- State feature 5: `dir_right`
- State feature 6: `dir_up`
- State feature 7: `dir_down`
- State feature 8: `food_left`
- State feature 9: `food_right`
- State feature 10: `food_up`
- State feature 11: `food_down`
- State feature 12: `snake_length`
- Action output size: `3` (one-hot)
- Action 1: `[1, 0, 0]` (straight)
- Action 2: `[0, 1, 0]` (turn right)
- Action 3: `[0, 0, 1]` (turn left)
- Reward: `+10` food eaten
- Reward: `-10` collision/timeout
- Reward: `0` otherwise
- Model input tensor shape: `(..., 12)`
- Model output tensor shape: `(..., 3)`
