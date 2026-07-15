"""Rolling-Horizon Collision Resolution (RHCR) planner.

Following the official RHCR design (Jiaoyang Li et al., AAAI 2021):
- Planning window W: conflict-free guarantee within [0, W)
- Horizon H: search depth cap (H < W)
- Reservation-based sequential planning with random restarts
- Supports both space-time A* and SIPP (Safe Interval Path Planning) backends
- K-robust constraints for expanded conflict detection
"""

import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..envs.entities import Position, AGV
from .base import PathPlannerBase, _HAS_CXX_ASTAR, _cpp_astar_nocopy, _sequential_sipp_attempt, _estimate_chain_distance, _HAS_CXX_BATCH, _cxx_batch_plan
from .conflict_utils import find_first_conflict


class RHCRPlanner(PathPlannerBase):
    """Rolling-Horizon Collision Resolution planner.

    Parameters
    ----------
    planning_window : int
        Conflict-free guarantee window W.  Paths are conflict-free within [0, W).
    horizon : int
        Search depth cap (H < W).  Must be less than planning_window.
    max_low_level_steps : int
        Maximum A* search steps per agent.
    use_sipp : bool
        If True, use SIPP backend instead of space-time A*.
    k_robust : int
        K-robust constraint expansion factor.
    suboptimal_bound : float
        Suboptimality bound (>= 1.0).
    """

    MAX_PLANNING_TIME = 5.0

    def __init__(
        self,
        grid_map,
        planning_window: int = 10,
        horizon: int = 4,
        max_low_level_steps: int = 500,
        random_seed: Optional[int] = 42,
        use_sipp: bool = False,
        k_robust: int = 2,
        suboptimal_bound: float = 1.0,
        tie_breaker_by_depth: Optional[bool] = None,
        use_closed_set: Optional[bool] = None,
    ):
        super().__init__(grid_map)
        self.planning_window = int(max(1, planning_window))
        self.horizon = int(max(1, horizon))
        if self.horizon >= self.planning_window:
            self.horizon = max(1, self.planning_window - 1)
        self.max_low_level_steps = int(max(1, max_low_level_steps))
        self._rng = random.Random(random_seed)
        # Initial constraints from already-executed path portions
        # Following official RHCR's BasicSystem::update_initial_constraints
        self._initial_constraints: Dict[int, List[Tuple[Position, int]]] = {}
        # SIPP configuration
        self.use_sipp = bool(use_sipp)
        self.k_robust = int(max(0, k_robust))
        self.suboptimal_bound = float(max(1.0, suboptimal_bound))
        # A* low-level search tunables (None = use defaults for A*/SIPP)
        self._tie_breaker_by_depth = tie_breaker_by_depth
        self._use_closed_set = use_closed_set

    def set_initial_constraints(self, constraints: Dict[int, List[Tuple[Position, int]]]):
        """Set constraints from already-executed path portions.

        These prevent new paths from conflicting with movements that have
        already been committed but not yet fully traversed.
        """
        self._initial_constraints = constraints

    def _plan_multi_agent_impl(self, agvs: Dict[int, AGV], target_positions: Dict[int, Position],
                                deadline: Optional[float] = None) -> Dict[int, List[Position]]:
        start_time = time.time()
        own_deadline = start_time + self.MAX_PLANNING_TIME
        if deadline is not None:
            own_deadline = min(own_deadline, deadline)
        # Check if SIPP multi-goal planning will be used
        self._multi_goal_agents = set()
        if self._goal_sequences:
            for aid, gs in self._goal_sequences.items():
                if len(gs) > 1:
                    self._multi_goal_agents.add(aid)
        has_multi_goals = bool(self._multi_goal_agents)
        self._used_multi_goal_sipp = self.use_sipp and has_multi_goals
        self._used_multi_goal = has_multi_goals

        def sort_key(aid: int):
            agv = agvs[aid]
            goal = target_positions.get(aid)
            if goal is None:
                return (0, aid)
            return (-(abs(agv.position.x - goal.x) + abs(agv.position.y - goal.y)), aid)

        active_ids = [aid for aid in agvs.keys() if aid in target_positions]

        # Timing accumulators (only used when self.timing_enabled)
        _astar_time = 0.0
        _conflict_time = 0.0
        _reservation_time = 0.0
        _num_restarts = 0

        def attempt(order: List[int]) -> Optional[Dict[int, List[Position]]]:
            nonlocal _astar_time, _conflict_time, _reservation_time
            solution: Dict[int, List[Position]] = {}

            if self.use_sipp:
                # SIPP-based planning with interval reservation table
                if self._passable_grid is None:
                    return None
                if self.timing_enabled:
                    _t = time.perf_counter()
                sol = _sequential_sipp_attempt(
                    passable_grid=self._passable_grid,
                    shelf_grid=self._shelf_grid,
                    shelf_penalty=self.SHELF_PENALTY,
                    planning_window=self.planning_window,
                    k_robust=self.k_robust,
                    max_low_level_steps=self.max_low_level_steps,
                    initial_constraints=self._initial_constraints,
                    order=order,
                    agvs=agvs,
                    target_positions=target_positions,
                    goal_sequences=self._goal_sequences,
                    deadline=own_deadline,
                    horizon=max(self.planning_window, self.horizon),
                )
                if self.timing_enabled:
                    self.last_timing.setdefault('astar_search_time', 0.0)
                    self.last_timing['astar_search_time'] += time.perf_counter() - _t
                return sol

            # A* mode: use C++ batch kernel if available
            if _HAS_CXX_BATCH and self._passable_grid is not None:
                if self.timing_enabled:
                    _t = time.perf_counter()
                sol = _cxx_batch_plan(
                    agvs=agvs,
                    target_positions=target_positions,
                    order=order,
                    passable_grid=self._passable_grid,
                    shelf_grid=self._shelf_grid,
                    mode="astar",
                    planning_window=self.planning_window,
                    max_low_level_steps=self.max_low_level_steps,
                    goal_sequences=self._goal_sequences,
                    initial_constraints=None,  # A* RHCR excludes initial constraints
                    k_robust=0,
                    horizon_mode=True,
                    shelf_penalty=self.SHELF_PENALTY,
                    deadline=own_deadline,
                )
                if self.timing_enabled:
                    self.last_timing.setdefault('astar_search_time', 0.0)
                    self.last_timing['astar_search_time'] += time.perf_counter() - _t
                return sol

            # Initial constraints are agent-specific: they prevent the same agent
            # from backtracking into recently-occupied positions.  They must NOT
            # be added to the global reservations set, as that would block OTHER
            # agents from using those positions at t=0.
            reservations_v: Set[Tuple[Position, int]] = set()
            reservations_e: Set[Tuple[Position, Position, int]] = set()
            for agent_id in order:
                agv = agvs[agent_id]
                goal = target_positions[agent_id]
                goals_list = None
                if agent_id in self._goal_sequences and self._goal_sequences[agent_id]:
                    goals_list = self._goal_sequences[agent_id]

                edge_constraints: Set[Tuple[Position, Position, int]] = set()
                for (p_from, p_to, t) in reservations_e:
                    edge_constraints.add((p_to, p_from, t))

                # Build per-agent vertex constraints: global reservations only.
                # Initial constraints (agent's own previous positions) are intentionally
                # excluded for A* RHCR: they are too restrictive on narrow maps and
                # only marginally useful since A* doesn't use k-robust expansion.
                agent_vc: Set[Tuple[Position, int]] = set(reservations_v)

                if self.timing_enabled:
                    _t_astar = time.perf_counter()
                full_path = self._windowed_space_time_search(
                    agv.position,
                    goal,
                    agv,
                    agent_vc,
                    edge_constraints,
                    goals_list=goals_list,
                )
                if self.timing_enabled:
                    _astar_time += time.perf_counter() - _t_astar
                if not full_path:
                    return None

                # Store full path — planner returns variable-length paths (>= planning_window).
                # Truncation for info/visualization happens in env post-processing.
                pw = self.planning_window
                if len(full_path) < pw + 1:
                    full_path = full_path + [full_path[-1]] * (pw + 1 - len(full_path))
                solution[agent_id] = full_path

                if self.timing_enabled:
                    _t_res = time.perf_counter()
                # Reservation insert covers only planning_window [0, pw]
                res_path = full_path[:pw + 1]
                for t in range(pw + 1):
                    reservations_v.add((res_path[t], t))
                for t in range(pw):
                    p0 = res_path[t]
                    p1 = res_path[t + 1]
                    if p0 != p1:
                        reservations_e.add((p0, p1, t))
                        reservations_e.add((p1, p0, t))
                if self.timing_enabled:
                    _reservation_time += time.perf_counter() - _t_res

            if self.timing_enabled:
                _t_conflict = time.perf_counter()
            # Check conflicts only within planning_window
            has_conflict = find_first_conflict(solution, conflict_horizon=self.planning_window) is not None
            if self.timing_enabled:
                _conflict_time += time.perf_counter() - _t_conflict
            if has_conflict:
                return None
            return solution

        solution = attempt(sorted(active_ids, key=sort_key))
        n_active = len(active_ids)
        effective_restarts = 3 * n_active if n_active > 3 else 2
        if solution is None and effective_restarts > 0:
            base = list(active_ids)
            for restart_idx in range(effective_restarts):
                if time.time() >= own_deadline:
                    break
                # Reset timing for this restart attempt
                _astar_time = 0.0
                _conflict_time = 0.0
                _reservation_time = 0.0
                _num_restarts = restart_idx + 1
                self._rng.shuffle(base)
                solution = attempt(base)
                if solution is not None:
                    break

        if solution is None:
            # Planning failed or timed out — return empty paths.
            self._planning_success = False
            return {}

        # Record sub-timings (base class wrapper will add total_planning_time and num_agents)
        if self.timing_enabled:
            self.last_timing.update({
                "astar_search_time": _astar_time,
                "conflict_resolution_time": _conflict_time,
                "reservation_update_time": _reservation_time,
                "num_restarts": _num_restarts,
            })

        return solution

    def _windowed_space_time_search(
        self,
        start: Position,
        goal: Position,
        agv: AGV,
        vertex_constraints: Set[Tuple[Position, int]],
        edge_constraints: Set[Tuple[Position, Position, int]],
        goals_list: Optional[List[Position]] = None,
    ) -> List[Position]:
        """Space-time A* search for a single agent.

        Returns the full path (variable length, >= planning_window).
        Search depth is determined by goal-chain distance for multi-goal,
        or ``planning_window * 2`` for single-goal.
        """
        if not _HAS_CXX_ASTAR:
            return []  # C++ backend required
        if start == goal and (not goals_list or len(goals_list) <= 1):
            return [start]
        if not self._is_passable(start):
            return []
        if (start, 0) in vertex_constraints:
            return []

        # Build flat numpy arrays directly (avoids pybind11 tuple->struct conversion)
        vc_count = len(vertex_constraints)
        ec_count = len(edge_constraints)
        if vc_count > 0:
            vc_arr = np.empty(vc_count * 3, dtype=np.int32)
            i = 0
            for pos, t in vertex_constraints:
                vc_arr[i] = pos.x; vc_arr[i + 1] = pos.y; vc_arr[i + 2] = t
                i += 3
        else:
            vc_arr = np.empty(0, dtype=np.int32)

        if ec_count > 0:
            ec_arr = np.empty(ec_count * 5, dtype=np.int32)
            i = 0
            for p1, p2, t in edge_constraints:
                ec_arr[i] = p1.x; ec_arr[i + 1] = p1.y
                ec_arr[i + 2] = p2.x; ec_arr[i + 3] = p2.y
                ec_arr[i + 4] = t
                i += 5
        else:
            ec_arr = np.empty(0, dtype=np.int32)

        # Determine search depth: chain distance for multi-goal, window * 2 for single
        if goals_list and len(goals_list) > 1:
            chain_dist = _estimate_chain_distance(start, goals_list)
            search_depth = min(chain_dist + 20, self.max_low_level_steps)
        else:
            search_depth = min(self.planning_window * 2, self.max_low_level_steps)
        max_time = max(search_depth, self.planning_window)
        if goals_list:
            goals_tuples = [(pos.x, pos.y) for pos in goals_list]
        else:
            goals_tuples = [(goal.x, goal.y)]

        _use_cs = self._use_closed_set if self._use_closed_set is not None else False
        _tie_bd = self._tie_breaker_by_depth if self._tie_breaker_by_depth is not None else True

        path = _cpp_astar_nocopy(
            start=(start.x, start.y),
            goals=goals_tuples,
            passable_grid=self._passable_grid,
            shelf_grid=self._shelf_grid,
            vc_flat=vc_arr,
            ec_flat=ec_arr,
            max_time=max_time,
            horizon_mode=True,
            use_closed_set=_use_cs,
            tie_breaker_by_depth=_tie_bd,
            shelf_penalty=self.SHELF_PENALTY,
        )
        if path:
            return [Position(x, y) for x, y in path]
        # Fallback: if multi-goal search failed, retry with single goal.
        if goals_list and len(goals_list) > 1:
            fallback = _cpp_astar_nocopy(
                start=(start.x, start.y),
                goals=[(goal.x, goal.y)],
                passable_grid=self._passable_grid,
                shelf_grid=self._shelf_grid,
                vc_flat=vc_arr,
                ec_flat=ec_arr,
                max_time=max_time,
                horizon_mode=True,
                use_closed_set=_use_cs,
                tie_breaker_by_depth=_tie_bd,
                shelf_penalty=self.SHELF_PENALTY,
            )
            return [Position(x, y) for x, y in fallback] if fallback else []
        return []
