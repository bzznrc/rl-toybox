# Walk

Arcade-style side-view continuous-control biped walker demo for PPO.

`walk` is intentionally lightweight: simple articulated kinematics, terrain contact springs, and a compact observation space designed for stable toy-repo training.

## Algorithm / Network

- Algo: PPO (shared on-policy runner)
- Actor/Critic MLP: `[32, 32]`
- Action space: `Box(shape=(4,), low=-1.0, high=1.0)`
- Continuous policy: diagonal Gaussian (`mean + learned log_std`)

## Observation / Actions

Observation is exactly `18` floats in this order:

- `SELF` (14)
  - `self_torso_tilt`
  - `self_torso_ang_vel`
  - `self_vx`
  - `self_vy`
  - `self_left_hip_angle`
  - `self_left_hip_speed`
  - `self_left_knee_angle`
  - `self_left_knee_speed`
  - `self_right_hip_angle`
  - `self_right_hip_speed`
  - `self_right_knee_angle`
  - `self_right_knee_speed`
  - `self_left_foot_contact`
  - `self_right_foot_contact`
- `RAYS` (4)
  - `ray_ground_10`
  - `ray_ground_25`
  - `ray_ground_40`
  - `ray_ground_55`

Actions (`4`, ordered):

1. `left_hip_torque`
2. `left_knee_torque`
3. `right_hip_torque`
4. `right_knee_torque`

Important walk-specific exception:

- Joint/torso angles are normalized scalar angles (not sin/cos pairs) on purpose.
- This keeps `OBS_DIM=18` exactly and is intentional for this compact demo.

## Rays / Sensing

- Four forward/down ground rays from the upper torso region.
- Ray directions are `+15°`, `+30°`, `+45°`, `+60°` from straight down toward forward.
- Values are normalized hit distances in `[0,1]`.
- `1.0` means no terrain hit within max ray range.

## Physics Model

- Semi-implicit Euler integration at fixed `dt`.
- State includes torso pose/velocity and per-joint angles/velocities.
- Legs use forward kinematics (thigh + shin + rigid foot).
- Foot-ground interaction uses simple spring-damper normal force + capped friction.
- Torso-ground contact latches horizontal distance and locks torso `x` in place (no post-fall sliding).
- Extreme torso tilt very near the ground is treated as a fall trigger to avoid unrealistic near-ground hovering.
- Poor control falls; stable control can learn walk-like forward gait.

## Rewards

Realized reward components are logged as `P L`:

- `P`: forward progress from episode-best distance only.
  - Let `best_x` be the farthest x reached so far in the episode.
  - Per step: `progress_reward = max(0, best_x_now - best_x_prev)`.
  - This is unscaled meters walked (`1m = 1 reward`).
- `L`: lose penalty (`-5`) when the episode ends without success.
  - In rendered play, fall failures keep the scene alive briefly (~2s) before final terminal commit.

No win bonus, no per-step penalty, and no signed-`dx`/velocity shaping are used.

## Curriculum

Three levels, success-based promotion:

- Level 1: perfectly flat terrain.
- Level 2: obstacle-by-obstacle random mix of substantial right-angle stairs (`1-2-1`, `1-2-3-2-1`) and substantial rollers (bumps/depressions).
- Level 3: same random mix, but larger/wider terrain features and cutoff stairs (`1-2` or `1-2-3` followed by a straight drop).

`LEVEL_SETTINGS` keeps only a small set of per-level terrain controls:

- `terrain_variants` (flat / stairs / rollers)
- `terrain_scale`
- `stairs_cutoff`
- `goal_distance` (single success/termination distance)
- `entropy_coef`

Terrain spacing and base geometry are fixed in shared constants (feature spacing/start offset, stair tread/patterns/base step height, roller base width/amplitude), keeping generation intentionally simple and consistent.

Terrain length and episode step budget are derived globally from `goal_distance` and shared viewport/runtime constants.

Promotion uses shared 3-level curriculum (`CURRICULUM_PROMOTION`) and episode `success`, not raw reward.

## Human Mode

- Minimal keyboard torque control is available:
  - Left hip: `A` (negative), `D` (positive)
  - Left knee: `S` (negative), `W` (positive)
  - Right hip: `J` (negative), `L` (positive)
  - Right knee: `K` (negative), `I` (positive)
- AI train/eval flow is the primary target.

Rendering notes:

- Terrain rays remain part of the 18-dim observation.
- Ray lines are shown by default (`WALK_SHOW_PLAYER_RAYS=1`) and can be disabled with that env flag.
- Pavement distance markers are drawn every 1m from spawn; every 10m marker includes an outline.
- Foot rendering is visual-clamped to the terrain outline; on contact, feet pivot to stay flat on the surface.
- The torso render body is also clamped above the same terrain outline.

## Run Commands

```bash
rl-toybox-train --game walk
rl-toybox-play-ai --game walk --model best --render
rl-toybox-play-user --game walk
python -m scripts.train --game walk
python -m scripts.play_ai --game walk --model best --render
python -m scripts.play_user --game walk
```

See `games/walk/config.py` for terrain/reward/training defaults.
