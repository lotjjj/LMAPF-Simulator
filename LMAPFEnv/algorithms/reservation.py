"""Reservation table and SIPP search for RHCR-family planners.

Following the official RHCR design (Jiaoyang Li et al., AAAI 2021):
- CT (Constraint Table): stores forbidden time intervals per location
- SIT (Safe Interval Table): lazily computed safe intervals per location
- CAT (Conflict Avoidance Table): soft constraints for conflict counting
- Supports k-robust constraints (expand forbidden intervals by k steps)
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..envs.entities import Position
from .base import _HAS_CXX_SIPP, _cxx_sipp_search


_INTERVAL_MAX = 10**9


class ReservationTable:
    """Interval-based reservation table with Safe Interval Table (SIT).

    Following the official RHCR design:
    - CT: location_key -> list of (t_min, t_max) forbidden intervals
    - SIT: location_key -> list of (t_min, t_max, num_conflicts) safe intervals
    - CAT: timestep -> set of location_keys (soft constraints)
    - Supports k-robust constraints (expand forbidden intervals by k steps)
    """

    def __init__(self, map_size: int = 0, k_robust: int = 0, window: int = 100,
                 k_robust_ct: Optional[int] = None):
        self.map_size = map_size
        self.k_robust = k_robust
        # CT (hard constraints) can use a different k_robust than CAT (soft constraints)
        self.k_robust_ct = k_robust_ct if k_robust_ct is not None else k_robust
        self.window = window
        self.ct: Dict[int, List[Tuple[int, int]]] = {}
        self.sit: Dict[int, List[Tuple[int, int, int]]] = {}
        self.cat: Dict[int, Set[int]] = {}
        self.hold_endpoints = False

    def clear(self):
        self.ct.clear()
        self.sit.clear()
        self.cat.clear()

    @staticmethod
    def _loc_key(pos: Position, map_width: int) -> int:
        return pos.y * map_width + pos.x

    @staticmethod
    def _edge_key(from_pos: Position, to_pos: Position, map_width: int) -> int:
        return -(to_pos.y * map_width + to_pos.x) * 10000 - (from_pos.y * map_width + from_pos.x)

    def insert_path_constraints(self, path: List[Position], map_width: int):
        """Insert hard vertex and edge constraints from a path, with k-robust expansion.

        Also fills gap constraints: when an agent leaves a position and later
        returns, the intermediate timesteps are blocked so other agents cannot
        slip into the gap and conflict with the agent's return.
        """
        if not path:
            return
        kr = self.k_robust_ct
        last_visit: Dict[int, int] = {}  # loc_key -> last timestep visited
        for t, pos in enumerate(path):
            loc = self._loc_key(pos, map_width)
            t_min = max(0, t - kr)
            t_max = t + 1 + kr
            self.ct.setdefault(loc, []).append((t_min, t_max))
            # Gap-filling: if this location was visited before and the agent
            # left in between, fill the gap to prevent other agents from
            # entering during the absence and conflicting with the return.
            if loc in last_visit:
                last_t = last_visit[loc]
                if t - last_t > 1:
                    gap_min = last_t + 1 + kr
                    gap_max = t - kr
                    if gap_min < gap_max:
                        self.ct.setdefault(loc, []).append((gap_min, gap_max))
            last_visit[loc] = t
            if t > 0 and path[t - 1] != pos:
                prev_pos = path[t - 1]
                edge = self._edge_key(prev_pos, pos, map_width)
                self.ct.setdefault(edge, []).append((t - 1 - kr, t + 1 + kr))
                reverse_edge = self._edge_key(pos, prev_pos, map_width)
                self.ct.setdefault(reverse_edge, []).append((t - 1 - kr, t + 1 + kr))
        if self.hold_endpoints and path:
            last_loc = self._loc_key(path[-1], map_width)
            self.ct.setdefault(last_loc, []).append((len(path), _INTERVAL_MAX))

    def insert_path_to_cat(self, path: List[Position], map_width: int):
        """Insert path into Conflict Avoidance Table (soft constraints)."""
        for t, pos in enumerate(path):
            if t > self.window + self.k_robust:
                break
            loc = self._loc_key(pos, map_width)
            for dt in range(-self.k_robust, self.k_robust + 1):
                ct = t + dt
                if 0 <= ct <= self.window + self.k_robust:
                    self.cat.setdefault(ct, set()).add(loc)
            if t > 0 and path[t - 1] != pos:
                edge_key = self._edge_key(path[t - 1], pos, map_width)
                for dt in range(-self.k_robust, self.k_robust + 1):
                    ct = t + dt
                    if 0 <= ct <= self.window + self.k_robust:
                        self.cat.setdefault(ct, set()).add(edge_key)

    def _build_sit(self, loc: int):
        """Build Safe Interval Table from CT (always rebuilds to reflect latest constraints)."""
        constraints = self.ct.get(loc, [])
        if not constraints:
            self.sit[loc] = [(0, _INTERVAL_MAX, 0)]
            return

        sorted_cons = sorted(constraints)
        intervals: List[Tuple[int, int, int]] = []
        current_start = 0

        for c_min, c_max in sorted_cons:
            c_max = min(c_max, self.window + 1)
            if c_min > current_start:
                conflicts = self._count_cat_conflicts(loc, current_start, c_min)
                intervals.append((current_start, c_min, conflicts))
            current_start = max(current_start, c_max)

        if current_start < _INTERVAL_MAX:
            conflicts = self._count_cat_conflicts(loc, current_start, _INTERVAL_MAX)
            intervals.append((current_start, _INTERVAL_MAX, conflicts))

        if not intervals:
            intervals.append((0, _INTERVAL_MAX, 0))

        self.sit[loc] = intervals

    def _count_cat_conflicts(self, loc: int, t_min: int, t_max: int) -> int:
        """Count soft conflicts in a time range from CAT."""
        count = 0
        for t in range(t_min, min(t_max, self.window + self.k_robust + 1)):
            if loc in self.cat.get(t, set()):
                count += 1
        return count

    def get_safe_intervals(self, loc: int, t_min: int, t_max: int) -> List[Tuple[int, int, int]]:
        """Query safe intervals at location within [t_min, t_max)."""
        self._build_sit(loc)
        result = []
        for si_min, si_max, conflicts in self.sit.get(loc, []):
            if si_min >= t_max:
                break
            if si_max <= t_min:
                continue
            result.append((max(si_min, t_min), min(si_max, t_max), conflicts))
        return result

    def get_first_safe_interval(self, loc: int) -> Tuple[int, int, int]:
        """Get the first safe interval at location."""
        self._build_sit(loc)
        sit = self.sit.get(loc, [])
        return sit[0] if sit else (0, _INTERVAL_MAX, 0)

    def get_safe_intervals_for_edge(
        self, from_pos: Position, to_pos: Position, map_width: int,
        t_min: int, t_max: int
    ) -> List[Tuple[int, int, int]]:
        """Get safe intervals for both vertex (to_pos) and edge (from->to)."""
        to_loc = self._loc_key(to_pos, map_width)
        vertex_intervals = self.get_safe_intervals(to_loc, t_min, t_max)

        edge = self._edge_key(from_pos, to_pos, map_width)
        self._build_sit(edge)
        edge_intervals = self.get_safe_intervals(edge, t_min, t_max)

        result = []
        it1 = iter(vertex_intervals)
        it2 = iter(edge_intervals)
        try:
            vi = next(it1)
            ei = next(it2)
            while True:
                t_lo = max(vi[0], ei[0])
                t_hi = min(vi[1], ei[1])
                if t_lo < t_hi:
                    result.append((t_lo, t_hi, vi[2] + ei[2]))
                if vi[1] <= ei[1]:
                    vi = next(it1)
                else:
                    ei = next(it2)
        except StopIteration:
            pass
        return result


def _serialize_ct(rt: ReservationTable) -> list:
    """Flatten ReservationTable CT to list of (loc_key, t_min, t_max) for C++."""
    flat = []
    for loc, intervals in rt.ct.items():
        for t_min, t_max in intervals:
            flat.append((loc, t_min, t_max))
    return flat


def _serialize_cat(rt: ReservationTable) -> list:
    """Flatten ReservationTable CAT to list of (loc_key, timestep) for C++."""
    flat = []
    for t, locs in rt.cat.items():
        for loc in locs:
            flat.append((loc, t))
    return flat


def sipp_search(
    rt: ReservationTable,
    start: Position,
    goal_locations: List[Tuple[Position, int]],  # [(pos, release_time), ...]
    passable_grid: np.ndarray,
    shelf_grid: Optional[np.ndarray],
    map_width: int,
    max_time: int = 200,
    shelf_penalty: float = 0.0,
    deadline: Optional[float] = None,
) -> List[Position]:
    """Safe Interval Path Planning (SIPP) search.

    Following the official RHCR SIPP.cpp design:
    - Search state: (position, safe_interval) instead of (position, time)
    - Wait actions jump to next safe interval instead of expanding step-by-step
    - Multi-goal support via goal_id tracking
    - Standard A* on open list (f-val ordering), no focal search

    Args:
        deadline: Optional wall-clock deadline (time.time()). If exceeded,
                  returns empty list immediately.

    Returns: list of positions forming the path, or empty list if no path found.
    """
    if deadline is not None and time.time() >= deadline:
        return []
    if _HAS_CXX_SIPP and passable_grid is not None:
        goals_cpp = [(p.x, p.y, rt_val) for p, rt_val in goal_locations]
        ct_flat = _serialize_ct(rt)
        cat_flat = _serialize_cat(rt) if rt.cat else []
        _shelf = shelf_grid if shelf_grid is not None else np.zeros_like(passable_grid, dtype=np.int8)
        result = _cxx_sipp_search(
            start_x=start.x, start_y=start.y,
            goals=goals_cpp,
            passable_grid=passable_grid,
            shelf_grid=_shelf,
            map_width=map_width,
            k_robust=rt.k_robust,
            window=rt.window,
            max_time=max_time,
            shelf_penalty=shelf_penalty,
            hold_endpoints=rt.hold_endpoints,
            ct_data=ct_flat,
            existing_paths=[],
            cat_data=cat_flat,
        )
        return [Position(x, y) for x, y in result] if result else []

    return []  # C++ SIPP not available or no grid data
