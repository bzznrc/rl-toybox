"""Stealth layout generation and grid helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

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


@dataclass(frozen=True)
class StealthLayout:
    walkable: np.ndarray
    covers: frozenset[Cell]
    rooms: tuple[Room, ...]
    start: Cell
    exit: Cell
    guards: tuple[GuardLane, ...]
    exit_distance_map: np.ndarray
    main_path: tuple[Cell, ...]


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


def _cell_blocks_vision(walkable: np.ndarray, covers: frozenset[Cell], cell: Cell) -> bool:
    return (not is_walkable(walkable, cell)) or (cell in covers)


def line_of_sight_clear(walkable: np.ndarray, covers: frozenset[Cell], origin: Cell, target: Cell) -> bool:
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
        if current != origin and current != target and _cell_blocks_vision(walkable, covers, current):
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


def is_visible(walkable: np.ndarray, covers: frozenset[Cell], origin: Cell, target: Cell, max_range: int) -> bool:
    delta_x = abs(int(target.x) - int(origin.x))
    delta_y = abs(int(target.y) - int(origin.y))
    if max(delta_x, delta_y) > int(max_range):
        return False
    return bool(line_of_sight_clear(walkable, covers, origin, target))


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


def _farthest_room_pair(walkable: np.ndarray, rooms: list[Room]) -> tuple[Room, Room]:
    best_pair = (rooms[0], rooms[1])
    best_dist = -1
    anchors = {room.room_id: room.center() for room in rooms}
    for room in rooms:
        dist_map = bfs_distance_map(walkable, anchors[room.room_id])
        for other in rooms:
            if int(other.room_id) <= int(room.room_id):
                continue
            dist_value = int(dist_map[int(anchors[other.room_id].y), int(anchors[other.room_id].x)])
            if dist_value > best_dist:
                best_dist = int(dist_value)
                best_pair = (room, other)
    return best_pair


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


def guard_lane_cells(lane: GuardLane) -> list[Cell]:
    dx = int(np.sign(int(lane.end.x) - int(lane.start.x)))
    dy = int(np.sign(int(lane.end.y) - int(lane.start.y)))
    cells = [lane.start]
    cursor = lane.start
    while cursor != lane.end:
        cursor = cursor.moved(dx, dy)
        cells.append(cursor)
    return cells


def _guard_observed_cells(
    walkable: np.ndarray,
    covers: frozenset[Cell],
    lane: GuardLane,
    position: Cell,
    facing_dx: int,
    facing_dy: int,
    *,
    vision_range: int,
) -> set[Cell]:
    observed = {position}
    for step in range(1, max(0, int(vision_range)) + 1):
        cell = position.moved(int(facing_dx) * step, int(facing_dy) * step)
        if not is_walkable(walkable, cell):
            break
        if cell in covers:
            break
        observed.add(cell)
    return observed


def _combined_guard_observation_cycle(
    walkable: np.ndarray,
    covers: frozenset[Cell],
    guards: tuple[GuardLane, ...],
    *,
    vision_range: int,
    move_period: int,
) -> list[set[Cell]]:
    if not guards:
        return [set()]
    states = [
        {
            "lane": lane,
            "cells": tuple(guard_lane_cells(lane)),
            "index": 0,
            "step_dir": 1,
            "facing_dx": int(lane.facing_dx),
            "facing_dy": int(lane.facing_dy),
        }
        for lane in guards
    ]
    observed_cycle: list[set[Cell]] = []
    seen: set[tuple[int, tuple[tuple[int, int, int, int, int], ...]]] = set()
    env_step = 0
    period = max(1, int(move_period))
    while True:
        signature = (
            int(env_step % period),
            tuple(
                (
                    int(state["cells"][int(state["index"])].x),
                    int(state["cells"][int(state["index"])].y),
                    int(state["step_dir"]),
                    int(state["facing_dx"]),
                    int(state["facing_dy"]),
                )
                for state in states
            ),
        )
        if signature in seen:
            break
        seen.add(signature)
        observed: set[Cell] = set()
        for state in states:
            position = state["cells"][int(state["index"])]
            observed.update(
                _guard_observed_cells(
                    walkable,
                    covers,
                    state["lane"],
                    position,
                    int(state["facing_dx"]),
                    int(state["facing_dy"]),
                    vision_range=int(vision_range),
                )
            )
        observed_cycle.append(observed)
        env_step += 1
        if int(env_step % period) == 0:
            for state in states:
                cells = state["cells"]
                if len(cells) <= 1:
                    continue
                next_index = int(state["index"]) + int(state["step_dir"])
                if next_index >= len(cells) or next_index < 0:
                    state["step_dir"] = -int(state["step_dir"])
                    next_index = int(state["index"]) + int(state["step_dir"])
                current = cells[int(state["index"])]
                state["index"] = int(next_index)
                nxt = cells[int(state["index"])]
                state["facing_dx"] = int(np.sign(int(nxt.x) - int(current.x)))
                state["facing_dy"] = int(np.sign(int(nxt.y) - int(current.y)))
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


def _choose_cover_cells(
    *,
    rng: random.Random,
    rooms: list[Room],
    route_cells: set[Cell],
    guard_lanes: tuple[GuardLane, ...],
    cover_count: int,
    start: Cell,
    exit: Cell,
) -> frozenset[Cell]:
    lane_cells = {cell for lane in guard_lanes for cell in guard_lane_cells(lane)}
    candidates: list[tuple[int, float, int, int, Cell]] = []
    for room in rooms:
        for cell in room.interior_cells():
            if cell in lane_cells or cell in {start, exit}:
                continue
            route_dist = min(manhattan(cell, route_cell) for route_cell in route_cells)
            if route_dist > 3:
                continue
            candidates.append((int(route_dist), float(rng.random()), int(cell.y), int(cell.x), cell))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return frozenset(cell for _, _, _, _, cell in candidates[: max(0, int(cover_count))])


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
    cover_count: int,
    guard_vision_range: int,
    guard_move_period: int,
    min_start_exit_dist: int,
) -> StealthLayout:
    requested_guard_count = max(0, int(guard_count))
    required_rooms = 2 + requested_guard_count
    for attempt in range(max(1, int(layout_attempts))):
        rng = random.Random(int(seed) + attempt * 9_973)
        walkable, rooms = _place_rooms(
            rng=rng,
            rows=int(rows),
            cols=int(cols),
            room_count=max(int(room_count), required_rooms),
            room_size_range=room_size_range,
            room_place_attempts=int(room_place_attempts),
        )
        if len(rooms) < required_rooms:
            continue
        _connect_rooms(walkable, rooms, rng=rng, extra_connection=bool(extra_connection))
        start_room, exit_room = _farthest_room_pair(walkable, rooms)
        start = start_room.center()
        exit_cell = exit_room.center()
        path = shortest_path(walkable, start, exit_cell)
        if len(path) <= max(1, int(min_start_exit_dist)):
            continue
        route_cells = set(path)
        eligible_rooms = [room for room in rooms if room.room_id not in {start_room.room_id, exit_room.room_id}]
        eligible_rooms.sort(
            key=lambda room: (
                min(manhattan(room.center(), route_cell) for route_cell in route_cells),
                -int(room.w * room.h),
                int(room.room_id),
            )
        )
        if len(eligible_rooms) < requested_guard_count:
            continue
        guards = tuple(_build_guard_lane(room, idx) for idx, room in enumerate(eligible_rooms[:requested_guard_count]))
        covers = _choose_cover_cells(
            rng=rng,
            rooms=rooms,
            route_cells=route_cells,
            guard_lanes=guards,
            cover_count=int(cover_count),
            start=start,
            exit=exit_cell,
        )
        observed_cycle = _combined_guard_observation_cycle(
            walkable,
            covers,
            guards,
            vision_range=int(guard_vision_range),
            move_period=int(guard_move_period),
        )
        if not _path_has_timing_window(path, observed_cycle):
            continue
        return StealthLayout(
            walkable=walkable,
            covers=covers,
            rooms=tuple(rooms),
            start=start,
            exit=exit_cell,
            guards=guards,
            exit_distance_map=bfs_distance_map(walkable, exit_cell),
            main_path=tuple(path),
        )
    raise RuntimeError("Failed to generate a valid Stealth layout.")
