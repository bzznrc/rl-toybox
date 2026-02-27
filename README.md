# rl-toybox

Toy RL monorepo with shared `core/` infrastructure and multiple Arcade-style games under `games/`.

## Overview

- Shared runtime/env interfaces in `core/`.
- Game-specific environments and specs in `games/`.
- Unified CLI scripts in `scripts/` for train, AI play, and human play.

## Setup

```bash
pip install -e .
```

## Run

Train:

```bash
rl-toybox-train --game bang
```

Play with trained model:

```bash
rl-toybox-play-ai --game bang --model best --render
```

Play as human:

```bash
rl-toybox-play-user --game bang
```

Without installation, from repo root:

```bash
python -m scripts.train --game bang
python -m scripts.play_ai --game bang --model best --render
python -m scripts.play_user --game bang
```

## Games

| Game ID | Default Algo | Obs / Action | Notes | Docs |
| --- | --- | --- | --- | --- |
| `snake` | `qlearn` | 12-dim / Discrete(3) | Classic snake baseline | [games/snake/README.md](games/snake/README.md) |
| `bang` | `dqn` | 24-dim / Discrete(8) | Shooter with enhanced DQN defaults | [games/bang/README.md](games/bang/README.md) |
| `vroom` | `dqn` | 6-dim / Discrete(5) | Procedural one-lap loop racer | [games/vroom/README.md](games/vroom/README.md) |
| `kick` | `ppo` | 44-dim / Discrete(12) | Football env, current discrete PPO setup | [games/kick/README.md](games/kick/README.md) |
| `stomp` | `sac` | 6-dim / Box(2) | Continuous-control placeholder | [games/stomp/README.md](games/stomp/README.md) |

## Plans: Games / Algos / IO / Nets

- `snake` -> Q-learning + `LinearQNet`
  - IO: obs 12, actions 3 discrete
  - Net size: `[32]`
- `bang` -> enhanced DQN (double + dueling + prioritized replay)
  - IO: obs 24, actions 8 discrete
  - Net size: `[64, 64]`
- `vroom` -> vanilla DQN
  - IO: obs 6, actions 5 discrete
  - Net size: `[48, 48]`
- `kick` -> PPO, parameter-shared across 11 players (start discrete)
  - IO: obs 44, actions 12 discrete
  - Net size: `[96, 96]`
- `walk` -> SAC (continuous), later
  - Current placeholder in repo: `stomp`
  - IO target: continuous control
  - Placeholder net size: `[128, 128]`
