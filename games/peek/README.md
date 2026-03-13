# Peek

Top-down partial-observability stealth/navigation game built around memory: find the key, then reach the door without stepping into a guard's forward line of sight.

## Clip

No embedded clip yet.

## Algorithm / Network

- Algo: recurrent PPO (`lstm`)
- Shared encoder: `[32]`
- Recurrent hidden size: `64`
- Actor head: `[32]`
- Critic head: `[32]`
- Action space: `Discrete(5)`

## Controls (Human)

- Move: `W/A/S/D`
- Wait: `Space` or no movement key

## Observation / Actions

Observation is exactly `18` floats in this order:

- `SELF` (3)
  - `self_has_key`
  - `self_time_left`
  - `self_here_revisited`
- `RAYS` (4)
  - `ray_wall_up`
  - `ray_wall_down`
  - `ray_wall_left`
  - `ray_wall_right`
- `OBJ` (3)
  - `obj1_dx`
  - `obj1_dy`
  - `obj1_type`
- `MEM` (4)
  - `mem_visited_up`
  - `mem_visited_down`
  - `mem_visited_left`
  - `mem_visited_right`
- `OPP` (4)
  - `opp1_dx`
  - `opp1_dy`
  - `opp1_facing_dx`
  - `opp1_facing_dy`

Observation notes:

- Only current perception plus explicit episode-local revisit memory is exposed.
- `self_here_revisited` uses `0.0` for the first visit to the current tile, `0.5` for the second or third visit, and `1.0` for the fourth visit or later.
- `ray_wall_*` is normalized free space before a wall in that direction: `0.0` means the adjacent tile is blocked and `1.0` means no wall within ray range.
- `obj1_*` is a single deterministic visible objective slot chosen from the visible world state. Before key pickup it prefers the visible key, otherwise the visible door. After key pickup it prefers the visible door. The door is never hidden when visible.
- `mem_visited_*` is the clipped revisit level for the adjacent walkable tile in that direction: `0.0` never visited, `0.5` visited `1-2` times already, `1.0` visited `3+` times already. Non-walkable or out-of-bounds neighbors use `0.0`.
- Visibility is local and wall-blocked; no global coordinates, discovery flags, or hidden world info are exposed.
- `obj1_type`: `0 empty`, `1 key`, `2 door`
- `opp1_facing_*` is one of `up=(0,-1)`, `down=(0,1)`, `left=(-1,0)`, `right=(1,0)`, or `(0,0)` when no guard is visible.

Actions (`5`, ordered):

1. `move_up`
2. `move_down`
3. `move_left`
4. `move_right`
5. `wait`

Key and door interaction are automatic.

## Environment Notes

### Layout / Guards

- Each episode generates a connected room-and-corridor layout.
- Start, key, and door are placed in distinct rooms using walkable-path distance.
- Key uses the sand pair and door uses the brown pair.
- Guards patrol deterministic straight room lanes, reverse at endpoints, and see only forward up to `4` tiles until a wall blocks their line.
- Layout generation rejects guard placements unless the main route still has a valid timing window for a wait-and-go traversal.

### Render Notes

- Walls use the same two-tone block tile family as the repo's obstacle-heavy top-down games.
- Playfield background is light and wall tiles use dark outlines to match the shared neutral palette roles.
- Player, guards, key, and door all stay in the shared arcade palette.
- Guard vision can be rendered as a light overlay when `PEEK_DRAW_GUARD_VISION=1`.

## Rewards (Training)

Realized components are logged as `W L K P B`:

- `W`: win reward `+10`
- `L`: lose penalty `-5`
- `K`: key pickup `+2.5`
- `P`: first visit to a walkable tile that episode `+0.02`
- `B`: blocked movement attempt `-0.01`

Timeout and capture both count as loss. Success per episode is `1` only when the agent reaches the door while carrying the key.

## Curriculum (Train)

Three levels, success-based promotion:

- Level 1: three rooms, no extra loop, no guards
- Level 2: six rooms, one extra loop, two guards
- Level 3: eight rooms, one extra loop, four guards

Route-distance targets, room-size range, and episode step budget are derived from the shared board size and current level rather than being repeated inside the per-level table.

## Run Commands

```bash
rl-toybox-train --game peek
rl-toybox-play-ai --game peek --model best --render
rl-toybox-play-user --game peek
python -m scripts.train --game peek
python -m scripts.play_ai --game peek --model best --render
python -m scripts.play_user --game peek
```
