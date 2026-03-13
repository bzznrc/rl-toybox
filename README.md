# rl-toybox

A small RL playground with shared infrastructure and arcade-style environments.

## Overview

- Shared game catalog, spec builders, and run preparation live in `core/game.py`; runtime helpers, runners, curriculum, and logging live in `core/`.
- Shared run, checkpoint, and model-path IO lives in `core/io/`.
- Game implementations, per-game configs/specs, and game snapshot READMEs live in `games/<name>/`.
- CLI entry points in `scripts/` cover training, AI play, and human play.

## Framework Docs

- Repo/codebase architecture: [docs/repo-architecture.md](docs/repo-architecture.md)
- Cross-game RL/environment design guide: [docs/rl-design-guide.md](docs/rl-design-guide.md)
- Docs index: [docs/README.md](docs/README.md)

## Clips

<p>
  <img src="media/snake-demo.gif" width="32%">
  <img src="media/vroom-demo.gif" width="32%">
  <img src="media/bang-demo.gif" width="32%">
</p>

## Run

With package install (recommended):

```bash
pip install -e .
rl-toybox-train --game bang
rl-toybox-play-ai --game bang --model best --render
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
| `snake` | `qlearn` | 12-dim / Discrete(3) | Classic grid snake with wrap-around and obstacle curriculum | [games/snake/README.md](games/snake/README.md) |
| `vroom` | `dqn` | 18-dim / Discrete(6) | One-lap procedural racing with scripted lane-keeping opponents and randomized starts | [games/vroom/README.md](games/vroom/README.md) |
| `bang` | `dqn` | 24-dim / Discrete(8) | Top-down arena shooter with cover, aiming, and obstacle-aware scripted enemies | [games/bang/README.md](games/bang/README.md) |
| `walk` | `ppo` | 18-dim / Box(4) | Side-view continuous-control biped walker over flat, stair, and roller terrain | [games/walk/README.md](games/walk/README.md) |
| `peek` | `ppo` | 18-dim / Discrete(5) | Top-down stealth navigation with procedural rooms, patrol guards, sparse route markers, and recurrent partial observability | [games/peek/README.md](games/peek/README.md) |
| `kick` | `ppo` | 48-dim / Discrete(12) | Top-down football with a shared LEFT-team actor and centralized critic training | [games/kick/README.md](games/kick/README.md) |

## Default Plans

- `snake` -> Q-learning + `LinearQNet` (`obs=12`, `act=3`, hidden `[32]`)
- `vroom` -> vanilla DQN (`obs=18`, `act=6`, hidden `[32, 32]`)
- `bang` -> enhanced DQN (`obs=24`, `act=8`, hidden `[64, 64]`)
- `walk` -> PPO continuous control (`obs=18`, `act=4`, hidden `[32, 32]`)
- `peek` -> recurrent PPO-LSTM (`obs=18`, `act=5`, encoder `[32]`, lstm `64`, heads `[32]`)
- `kick` -> MAPPO-style PPO (`actor obs=48`, `act=12`, actor `[128, 128]`, critic `[256, 256]`)

See each game README for the ordered observation/action IO, reward breakdown, and current curriculum snapshot.
