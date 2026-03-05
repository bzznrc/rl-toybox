# Stomp (Walk Placeholder)

Continuous-control placeholder environment for future walker work.

## Controls (Human)

- No dedicated human gameplay controls yet.
- `render()` is currently a stub.

## Observation / Actions

- Observation: `6` floats
  - `pos_x`, `pos_y`, `vel_x`, `vel_y`, `target_dx`, `target_dy`
- Actions: `Box(shape=(2,), low=-1.0, high=1.0)`
  - Continuous 2D control vector

## Rewards (Training)

- Step reward: negative distance to target (`-distance`)
- Episode ends at max steps or when target is reached (`distance < 0.05`)

## Run Commands

```bash
rl-toybox-train --game stomp --algo sac
```

This environment is intentionally minimal and serves as a stepping stone toward a fuller `walk` task.
