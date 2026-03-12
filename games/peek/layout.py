"""Peek layout generation and grid-geometry helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


CARDINAL_DIRS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
)


@dataclass(frozen=True, order=True)
class Cell:
    x: int
    y: int

    def moved(self, dx: int, dy: int) -> "Cell":
        return Cell(int(self.x) + int(dx), int(self.y) + int(dy))


@dataclass(frozen=True)
class Room:
    room_id: int
    x: int
    y: int
    w: int
    h: int

    def center(self) -> Cell:
        return Cell(int(self.x + self.w // 2), int(self.y + self.h // 2))

    def contains(self, cell: Cell) -> bool:
        return int(self.x) <= int(cell.x) < int(self.x + self.w) and int(self.y) <= int(cell.y) < int(self.y + self.h)

    def interior_cells(self) -> list[Cell]:
        cells: list[Cell] = []
        for y in range(int(self.y + 1), int(self.y + self.h - 1)):
            for x in range(int(self.x + 1), int(self.x + self.w - 1)):
                cells.append(Cell(x, y))
        return cells if cells else [self.center()]


@dataclass(frozen=True)
class GuardLane:
    guard_id: int
    start: Cell
    end: Cell
    facing_dx: int
    facing_dy: int


@dataclass
class PatrolState:
    lane: GuardLane
    position: Cell
    step_dir: int
    facing_dx: int
    facing_dy: int

    @classmethod
    def from_lane(cls, lane: GuardLane) -> "PatrolState":
        return cls(
            lane=lane,
            position=lane.start,
            step_dir=1,
            facing_dx=int(lane.facing_dx),
            facing_dy=int(lane.facing_dy),
        )

    def signature(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.position.x),
            int(self.position.y),
            int(self.step_dir),
            int(self.facing_dx),
            int(self.facing_dy),
        )

    def advance(self) -> None:
        target = self.lane.end if int(self.step_dir) > 0 else self.lane.start
        if self.position == target:
            self.step_dir *= -1
            target = self.lane.end if int(self.step_dir) > 0 else self.lane.start
        dx = int(np.sign(int(target.x) - int(self.position.x)))
        dy = int(np.sign(int(target.y) - int(self.position.y)))
        self.position = self.position.moved(dx, dy)
        self.facing_dx = int(dx if dx != 0 else self.facing_dx)
        self.facing_dy = int(dy if dy != 0 else self.facing_dy)


@dataclass(frozen=True)
class PeekLayout:
    walkable: np.ndarray
    rooms: tuple[Room, ...]
    start: Cell
    key: Cell
    door: Cell
    guards: tuple[GuardLane, ...]
    key_distance_map: np.ndarray
    door_distance_map: np.ndarray


def in_bounds(walkable: np.ndarray, cell: Cell) -> bool:
    rows, cols = walkable.shape
    return 0 <= int(cell.x) < int(cols) and 0 <= int(cell.y) < int(rows)


def is_walkable(walkable: np.ndarray, cell: Cell) -> bool:
    return bool(in_bounds(walkable, cell) and walkable[int(cell.y), int(cell.x)])


def walkable_neighbors(walkable: np.ndarray, cell: Cell) -> list[Cell]:
    neighbors: list[Cell] = []
    for dx, dy in CARDINAL_DIRS:
        nxt = cell.moved(dx, dy)
        if is_walkable(walkable, nxt):
            neighbors.append(nxt)
    return neighbors


def manhattan(a: Cell, b: Cell) -> int:
    return abs(int(a.x) - int(b.x)) + abs(int(a.y) - int(b.y))


def bfs_distance_map(walkable: np.ndarray, origin: Cell) -> np.ndarray:
    dist = np.full(walkable.shape, -1, dtype=np.int32)
    if not is_walkable(walkable, origin):
        return dist
    queue: deque[Cell] = deque([origin])
    dist[int(origin.y), int(origin.x)] = 0
    while queue:
        cell = queue.popleft()
        base = int(dist[int(cell.y), int(cell.x)])
        for nxt in walkable_neighbors(walkable, cell):
            if int(dist[int(nxt.y), int(nxt.x)]) >= 0:
                continue
            dist[int(nxt.y), int(nxt.x)] = int(base + 1)
            queue.append(nxt)
    return dist


def shortest_path(walkable: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
    if not (is_walkable(walkable, start) and is_walkable(walkable, goal)):
        return []
    queue: deque[Cell] = deque([start])
    parents: dict[Cell, Cell | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for nxt in walkable_neighbors(walkable, cell):
            if nxt in parents:
                continue
            parents[nxt] = cell
            queue.append(nxt)
    if goal not in parents:
        return []
    path: list[Cell] = []
    cursor: Cell | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    return path


def line_of_sight_clear(walkable: np.ndarray, origin: Cell, target: Cell) -> bool:
    x0 = int(origin.x)
    y0 = int(origin.y)
    x1 = int(target.x)
    y1 = int(target.y)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x = x0
    y = y0
    while True:
        current = Cell(x, y)
        if current != origin and (not is_walkable(walkable, current)):
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def is_visible(walkable: np.ndarray, origin: Cell, target: Cell, max_range: int) -> bool:
    delta_x = abs(int(target.x) - int(origin.x))
    delta_y = abs(int(target.y) - int(origin.y))
    if max(delta_x, delta_y) > int(max_range):
        return False
    return bool(line_of_sight_clear(walkable, origin, target))


def _rooms_overlap(candidate: Room, other: Room, *, padding: int = 1) -> bool:
    return not (
        int(candidate.x + candidate.w + padding) <= int(other.x)
        or int(other.x + other.w + padding) <= int(candidate.x)
        or int(candidate.y + candidate.h + padding) <= int(other.y)
        or int(other.y + other.h + padding) <= int(candidate.y)
    )


def _carve_room(walkable: np.ndarray, room: Room) -> None:
    walkable[int(room.y) : int(room.y + room.h), int(room.x) : int(room.x + room.w)] = True


def _carve_corridor(walkable: np.ndarray, start: Cell, end: Cell) -> None:
    x = int(start.x)
    y = int(start.y)
    while x != int(end.x):
        walkable[int(y), int(x)] = True
        x += 1 if int(end.x) > int(x) else -1
    while y != int(end.y):
        walkable[int(y), int(x)] = True
        y += 1 if int(end.y) > int(y) else -1
    walkable[int(end.y), int(end.x)] = True


def _room_anchor(room: Room, rng: random.Random) -> Cell:
    cells = room.interior_cells()
    cells.sort(key=lambda cell: (manhattan(cell, room.center()), int(cell.y), int(cell.x)))
    if len(cells) <= 1:
        return cells[0]
    return cells[int(rng.randrange(len(cells[: min(4, len(cells))])))]


def _place_rooms(
    *,
    rng: random.Random,
    rows: int,
    cols: int,
    room_count: int,
    room_size_range: tuple[int, int],
    room_place_attempts: int,
) -> tuple[np.ndarray, list[Room]]:
    walkable = np.zeros((rows, cols), dtype=np.bool_)
    rooms: list[Room] = []
    min_size, max_size = room_size_range
    for _ in range(int(room_place_attempts)):
        if len(rooms) >= int(room_count):
            break
        width = int(rng.randint(int(min_size), int(max_size)))
        height = int(rng.randint(int(min_size), int(max_size)))
        width = min(width, int(cols) - 2)
        height = min(height, int(rows) - 2)
        if width < 4 or height < 4:
            continue
        x = int(rng.randint(1, max(1, int(cols) - width - 1)))
        y = int(rng.randint(1, max(1, int(rows) - height - 1)))
        room = Room(room_id=len(rooms), x=x, y=y, w=width, h=height)
        if any(_rooms_overlap(room, other, padding=1) for other in rooms):
            continue
        rooms.append(room)
        _carve_room(walkable, room)
    return walkable, rooms


def _connect_rooms(walkable: np.ndarray, rooms: list[Room], *, rng: random.Random, extra_connection: bool) -> None:
    ordered_rooms = sorted(rooms, key=lambda room: (int(room.center().x), int(room.center().y), int(room.room_id)))
    for left, right in zip(ordered_rooms[:-1], ordered_rooms[1:]):
        start = left.center()
        end = right.center()
        if bool(rng.random() < 0.5):
            _carve_corridor(walkable, start, Cell(end.x, start.y))
            _carve_corridor(walkable, Cell(end.x, start.y), end)
        else:
            _carve_corridor(walkable, start, Cell(start.x, end.y))
            _carve_corridor(walkable, Cell(start.x, end.y), end)

    if extra_connection and len(ordered_rooms) >= 4:
        left = ordered_rooms[0]
        right = ordered_rooms[-1]
        _carve_corridor(walkable, left.center(), Cell(right.center().x, left.center().y))
        _carve_corridor(walkable, Cell(right.center().x, left.center().y), right.center())


def _farthest_room_pair(walkable: np.ndarray, rooms: list[Room]) -> tuple[Room, Room, dict[tuple[int, int], int]]:
    best_pair = (rooms[0], rooms[1])
    best_dist = -1
    distances: dict[tuple[int, int], int] = {}
    anchors = {room.room_id: room.center() for room in rooms}
    for room in rooms:
        dist_map = bfs_distance_map(walkable, anchors[room.room_id])
        for other in rooms:
            if int(other.room_id) <= int(room.room_id):
                continue
            dist_value = int(dist_map[int(anchors[other.room_id].y), int(anchors[other.room_id].x)])
            distances[(int(room.room_id), int(other.room_id))] = int(dist_value)
            distances[(int(other.room_id), int(room.room_id))] = int(dist_value)
            if dist_value > best_dist:
                best_dist = int(dist_value)
                best_pair = (room, other)
    return best_pair[0], best_pair[1], distances


def _distance_between(distance_lookup: dict[tuple[int, int], int], left_id: int, right_id: int) -> int:
    return int(distance_lookup.get((int(left_id), int(right_id)), -1))


def _build_guard_lane(room: Room, guard_id: int) -> GuardLane:
    center = room.center()
    horizontal_len = max(1, int(room.w) - 2)
    vertical_len = max(1, int(room.h) - 2)
    if horizontal_len >= vertical_len:
        start = Cell(int(room.x + 1), int(center.y))
        end = Cell(int(room.x + room.w - 2), int(center.y))
        return GuardLane(guard_id=int(guard_id), start=start, end=end, facing_dx=1, facing_dy=0)
    start = Cell(int(center.x), int(room.y + 1))
    end = Cell(int(center.x), int(room.y + room.h - 2))
    return GuardLane(guard_id=int(guard_id), start=start, end=end, facing_dx=0, facing_dy=1)


def _guard_observed_cells(
    walkable: np.ndarray,
    state: PatrolState,
    *,
    vision_range: int,
) -> set[Cell]:
    observed = {state.position}
    for step in range(1, max(0, int(vision_range)) + 1):
        cell = state.position.moved(int(state.facing_dx) * step, int(state.facing_dy) * step)
        if not is_walkable(walkable, cell):
            break
        observed.add(cell)
    return observed


def _combined_guard_observation_cycle(
    walkable: np.ndarray,
    guards: tuple[GuardLane, ...],
    *,
    vision_range: int,
    move_period: int,
) -> list[set[Cell]]:
    if not guards:
        return [set()]

    states = [PatrolState.from_lane(lane) for lane in guards]
    period = max(1, int(move_period))
    observed_cycle: list[set[Cell]] = []
    seen: dict[tuple[int, tuple[tuple[int, int, int, int, int], ...]], int] = {}
    env_step = 0

    while True:
        key = (
            int(env_step % period),
            tuple(state.signature() for state in states),
        )
        if key in seen:
            break
        seen[key] = len(observed_cycle)
        observed: set[Cell] = set()
        for state in states:
            observed.update(_guard_observed_cells(walkable, state, vision_range=int(vision_range)))
        observed_cycle.append(observed)
        env_step += 1
        if int(env_step % period) == 0:
            for state in states:
                state.advance()
    return observed_cycle


def _path_has_timing_window(path: list[Cell], observed_cycle: list[set[Cell]]) -> bool:
    if not path:
        return False
    cycle_len = max(1, len(observed_cycle))
    queue: deque[tuple[int, int]] = deque()
    seen: set[tuple[int, int]] = set()

    for start_phase in range(cycle_len):
        if path[0] in observed_cycle[start_phase]:
            continue
        state = (0, int(start_phase))
        seen.add(state)
        queue.append(state)

    while queue:
        path_idx, phase = queue.popleft()
        if int(path_idx) >= int(len(path) - 1):
            return True
        next_phase = int((phase + 1) % cycle_len)

        if path[path_idx] not in observed_cycle[next_phase]:
            wait_state = (int(path_idx), next_phase)
            if wait_state not in seen:
                seen.add(wait_state)
                queue.append(wait_state)

        next_idx = int(path_idx + 1)
        if path[next_idx] not in observed_cycle[next_phase]:
            move_state = (next_idx, next_phase)
            if move_state not in seen:
                seen.add(move_state)
                queue.append(move_state)
    return False


def _min_required_room_count(guard_count: int) -> int:
    # Peek always needs distinct start/key/door rooms, plus separate guard rooms.
    return 3 + max(0, int(guard_count))


def generate_layout(
    *,
    seed: int,
    layout_attempts: int,
    rows: int,
    cols: int,
    room_count: int,
    room_size_range: tuple[int, int],
    room_place_attempts: int,
    extra_connection: bool,
    guard_count: int,
    guard_vision_range: int,
    guard_move_period: int,
    min_start_key_dist: int,
    min_key_door_dist: int,
) -> PeekLayout:
    requested_room_count = max(0, int(room_count))
    requested_guard_count = max(0, int(guard_count))
    required_room_count = _min_required_room_count(requested_guard_count)
    effective_room_count = max(requested_room_count, required_room_count)
    failure_counts = {
        "insufficient_rooms": 0,
        "missing_door_room": 0,
        "short_route": 0,
        "insufficient_guard_rooms": 0,
        "blocked_start_key_route": 0,
        "blocked_key_door_route": 0,
    }
    for attempt in range(max(1, int(layout_attempts))):
        rng = random.Random(int(seed) + attempt * 9_973)
        walkable, rooms = _place_rooms(
            rng=rng,
            rows=int(rows),
            cols=int(cols),
            room_count=effective_room_count,
            room_size_range=room_size_range,
            room_place_attempts=int(room_place_attempts),
        )
        if len(rooms) < required_room_count:
            failure_counts["insufficient_rooms"] += 1
            continue
        _connect_rooms(walkable, rooms, rng=rng, extra_connection=bool(extra_connection))

        start_room, key_room, distance_lookup = _farthest_room_pair(walkable, rooms)
        remaining_rooms = [room for room in rooms if room.room_id not in {start_room.room_id, key_room.room_id}]
        remaining_rooms.sort(
            key=lambda room: (
                -_distance_between(distance_lookup, key_room.room_id, room.room_id),
                -_distance_between(distance_lookup, start_room.room_id, room.room_id),
                int(room.room_id),
            )
        )
        if not remaining_rooms:
            failure_counts["missing_door_room"] += 1
            continue
        door_room = remaining_rooms[0]

        start = _room_anchor(start_room, rng)
        key = _room_anchor(key_room, rng)
        door = _room_anchor(door_room, rng)
        start_key_path = shortest_path(walkable, start, key)
        key_door_path = shortest_path(walkable, key, door)
        if len(start_key_path) <= max(1, int(min_start_key_dist)) or len(key_door_path) <= max(1, int(min_key_door_dist)):
            failure_counts["short_route"] += 1
            continue

        route_cells = {cell for cell in start_key_path + key_door_path}
        eligible_rooms = [room for room in rooms if room.room_id not in {start_room.room_id, key_room.room_id, door_room.room_id}]
        eligible_rooms.sort(
            key=lambda room: (
                min(manhattan(room.center(), cell) for cell in route_cells),
                -int(room.w * room.h),
                int(room.room_id),
            )
        )
        if len(eligible_rooms) < requested_guard_count:
            failure_counts["insufficient_guard_rooms"] += 1
            continue
        guards = tuple(_build_guard_lane(room, idx) for idx, room in enumerate(eligible_rooms[:requested_guard_count]))
        observed_cycle = _combined_guard_observation_cycle(
            walkable,
            guards,
            vision_range=int(guard_vision_range),
            move_period=int(guard_move_period),
        )
        if not _path_has_timing_window(start_key_path, observed_cycle):
            failure_counts["blocked_start_key_route"] += 1
            continue
        if not _path_has_timing_window(key_door_path, observed_cycle):
            failure_counts["blocked_key_door_route"] += 1
            continue

        return PeekLayout(
            walkable=walkable,
            rooms=tuple(rooms),
            start=start,
            key=key,
            door=door,
            guards=guards,
            key_distance_map=bfs_distance_map(walkable, key),
            door_distance_map=bfs_distance_map(walkable, door),
        )

    raise RuntimeError(
        "Failed to generate a valid Peek layout. "
        f"requested_room_count={requested_room_count}, "
        f"effective_room_count={effective_room_count}, "
        f"guard_count={requested_guard_count}, "
        f"min_start_key_dist={int(min_start_key_dist)}, "
        f"min_key_door_dist={int(min_key_door_dist)}, "
        f"failure_counts={failure_counts}"
    )


def iter_walkable_cells(walkable: np.ndarray) -> Iterable[Cell]:
    rows, cols = walkable.shape
    for y in range(int(rows)):
        for x in range(int(cols)):
            cell = Cell(x, y)
            if is_walkable(walkable, cell):
                yield cell
