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

### Training log lines

Training output is tab-separated and compact:

- Header (once): `Train\tGame: ...\tAlgo: ...\tRun: ...\tResume: ...\tRender: off/on`
- Off-policy episode line: `Episode: N\tLength: M\tReward: r\tAverage: ar\tBest: bar\tEpsilon: eee`
- On-policy iteration line: `Iter: I\tSteps: s\tEpisodes: e\tAvg Reward: ar\tBest Avg: bar`
- Save line: `Save: best|checkpoint\tAt: step/iter ...\tAvg Reward: ar\tPath: ...`

CLI cadence flags:

- `--log-every-episodes` (default `1`, off-policy)
- `--log-every-iterations` (default `1`, on-policy)
- `--log-heartbeat-steps` (default `0`, disabled; only logs when no episode/iter line has been printed in that interval)

## Games

| Game ID | Default Algo | Obs / Action | Notes | Docs |
| --- | --- | --- | --- | --- |
| `snake` | `qlearn` | 12-dim / Discrete(3) | Classic grid snake survival game | [games/snake/README.md](games/snake/README.md) |
| `vroom` | `dqn` | 20-dim / Discrete(6) | Top-down one-lap racing on procedural tracks | [games/vroom/README.md](games/vroom/README.md) |
| `bang` | `dqn` | 24-dim / Discrete(8) | Top-down arena shooter with movement, aim, and firing | [games/bang/README.md](games/bang/README.md) |
| `kick` | `ppo` | 36-dim / Discrete(12) | Top-down football match with movement and kick actions | [games/kick/README.md](games/kick/README.md) |
| `stomp` | `sac` | 6-dim / Box(2) | Continuous-control testbed environment | [games/stomp/README.md](games/stomp/README.md) |

## Plans: Games / Algos / IO / Nets

- `snake` -> Q-learning + `LinearQNet`
  - IO: obs 12, actions 3 discrete
  - Net size: `[32]`
- `vroom` -> vanilla DQN
  - IO: obs 20, actions 6 discrete
  - Net size: `[48, 48]`
- `bang` -> enhanced DQN (double + dueling + prioritized replay)
  - IO: obs 24, actions 8 discrete
  - Net size: `[64, 64]`
- `kick` -> PPO, parameter-shared across 11 players (start discrete)
  - IO: obs 36, actions 12 discrete
  - Net size: `[96, 96]`
- `walk` -> SAC (continuous), later
  - Current placeholder in repo: `stomp`
  - IO target: continuous control
  - Placeholder net size: `[128, 128]`
