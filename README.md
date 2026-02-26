# rl-toybox

Minimal monorepo for toy RL games with a shared core.

## Quick Start

```bash
pip install -e .

rl-toybox-train --game bang
rl-toybox-train --game snake
rl-toybox-play-ai --game bang --model best --render
rl-toybox-play-user --game bang
```

Without installation, run module entrypoints from repo root:
`python -m scripts.train --game bang`

Games available: `bang`, `snake`, `vroom`, `kick`, `stomp`.
