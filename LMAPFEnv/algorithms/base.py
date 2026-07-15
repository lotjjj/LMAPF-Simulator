"""Planner Interface — base class and shared infrastructure for all path planners.

Interface Contract
------------------
Every planner subclass MUST implement ``_plan_multi_agent_impl``.
The base class ``plan_multi_agent`` orchestrates the planning lifecycle:

1. Reset ``_planning_success = True``.
2. Call ``_plan_multi_agent_impl`` (single-pass multi-goal planning).
3. Store the result in ``self.planned_paths`` and reset ``self._path_heads``.

``_plan_multi_agent_impl`` contract:
  - Return ``Dict[int, List[Position]]`` mapping agent_id → planned path.
  - Return ``{}`` and set ``self._planning_success = False`` on failure/timeout.
  - Do NOT assign ``self.planned_paths`` or ``self._path_heads`` (base class handles this).
  - Use ``self._goal_sequences`` for multi-goal planning (goal_id as state dimension).

Low-Level Search Backends
-------------------------
Both A* and SIPP are available as peer-level module functions:

- ``_cpp_astar_nocopy``  — Zero-copy C++ space-time A* with flat constraint arrays.
- ``_cpp_sipp``   — C++ Safe Interval Path Planning with reservation table.
- ``_sequential_sipp_attempt`` — Sequential multi-agent SIPP planning helper.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set

import numpy as np

from ..envs.entities import Position, AGV, Shelf

# ── C++ backend import guards ────────────────────────────────────────────────

try:
    from LMAPFEnv.fast_graph import cxx_astar_nocopy as _cxx_astar_nocopy
    _HAS_CXX_ASTAR = True
except ImportError:
    _cxx_astar_nocopy = None
    _HAS_CXX_ASTAR = False

try:
    from LMAPFEnv.fast_graph import cxx_sipp_search as _cxx_sipp_search
    _HAS_CXX_SIPP = True
except ImportError:
    _cxx_sipp_search = None
    _HAS_CXX_SIPP = False

try:
    from LMAPFEnv.fast_graph import (
        cxx_batch_sequential_plan as _cxx_batch_sequential_plan,
        BatchAgent as _BatchAgent,
    )
    _HAS_CXX_BATCH = True
except ImportError:
    _cxx_batch_sequential_plan = None
    _BatchAgent = None
    _HAS_CXX_BATCH = False



def _cpp_astar_nocopy(start, goals, passable_grid, shelf_grid,
                      vc_flat, ec_flat,
                      max_time, horizon_mode, use_closed_set,
                      tie_breaker_by_depth, shelf_penalty):
    """Zero-copy wrapper: accepts numpy arrays directly (no data copy).

    vc_flat: 1D np.int32 array [x0,y0,t0, x1,y1,t1, ...] (length=3*vc_count)
    ec_flat: 1D np.int32 array [fx0,fy0,tx0,ty0,t0, ...] (length=5*ec_count)
    """
    if passable_grid is None:
        return []
    if shelf_grid is None:
        shelf_grid = np.zeros_like(passable_grid, dtype=np.int8)
    return _cxx_astar_nocopy(
        start=start, goals=goals,
        passable_grid=passable_grid,
        shelf_grid=shelf_grid,
        vc_flat=vc_flat,
        ec_flat=ec_flat,
        max_time=max_time,
        horizon_mode=horizon_mode,
        use_closed_set=use_closed_set,
        tie_breaker_by_depth=tie_breaker_by_depth,
        shelf_penalty=float(shelf_penalty),
    )


def _estimate_chain_distance(start, goals) -> int:
    """Estimate total Manhattan distance through a goal chain.

    Parameters
    ----------
    start : Position or tuple
        Start position (supports .x/.y or (x, y)).
    goals : list
        Ordered goal positions (Position or (x, y)).
    """
    if not goals:
        return 0
    if hasattr(start, 'x'):
        sx, sy = start.x, start.y
    else:
        sx, sy = start
    dist = 0
    px, py = sx, sy
    for g in goals:
        gx, gy = (g.x, g.y) if hasattr(g, 'x') else g
        dist += abs(px - gx) + abs(py - gy)
        px, py = gx, gy
    return dist


def _cxx_batch_plan(
    agvs,
    target_positions,
    order,
    passable_grid,
    shelf_grid,
    mode,
    planning_window,
    max_low_level_steps,
    goal_sequences=None,
    initial_constraints=None,
    k_robust=0,
    horizon_mode=False,
    shelf_penalty=3.0,
    deadline=None,
):
    """Wrapper around C++ cxx_batch_sequential_plan.

    Converts Python objects to C++ format, calls the batch kernel, and
    converts the result back.

    Parameters
    ----------
    agvs : dict
        ``{agent_id: AGV}``
    target_positions : dict
        ``{agent_id: Position}``
    order : list of int
        Planning order (agent_ids).
    passable_grid, shelf_grid : np.ndarray
        Grid data.
    mode : str
        ``'astar'`` or ``'sipp'``.
    planning_window : int
        Conflict-free guarantee window.
    max_low_level_steps : int
        Maximum search depth per agent.
    goal_sequences : dict or None
        ``{agent_id: [Position, ...]}`` multi-goal sequences.
    initial_constraints : dict or None
        ``{agent_id: [(Position, t), ...]}`` for SIPP mode.
    k_robust : int
        K-robust constraint expansion (SIPP mode).
    horizon_mode : bool
        A* horizon mode (A* mode only).
    shelf_penalty : float
        Shelf traversal penalty.
    deadline : float or None
        Wall-clock deadline (time.time()).

    Returns
    -------
    dict or None
        ``{agent_id: [Position, ...]}`` on success, ``None`` on failure.
    """
    if not _HAS_CXX_BATCH:
        return None
    if passable_grid is None:
        return None

    import numpy as np

    map_h, map_w = passable_grid.shape

    # Build BatchAgent list: indexed by position in `order`
    # We create a mapping: agent_id -> index
    id_to_idx = {aid: i for i, aid in enumerate(order)}
    agents = []
    for aid in order:
        agv = agvs[aid]
        ba = _BatchAgent()
        ba.start_x = agv.position.x
        ba.start_y = agv.position.y
        # Goals
        goals = []
        if goal_sequences and aid in goal_sequences and goal_sequences[aid]:
            for pos in goal_sequences[aid]:
                goals.append((pos.x, pos.y))
        else:
            goal = target_positions[aid]
            goals.append((goal.x, goal.y))
        ba.goals = goals
        agents.append(ba)

    # per_agent_max_time: indexed by position
    per_agent_max_time = []
    for aid in order:
        agv = agvs[aid]
        goal = target_positions[aid]
        # Use chain distance for multi-goal, window * 2 for single
        if goal_sequences and aid in goal_sequences and len(goal_sequences[aid]) > 1:
            chain_dist = _estimate_chain_distance(agv.position, goal_sequences[aid])
            mt = min(chain_dist + 20, max_low_level_steps)
        else:
            mt = min(max(planning_window * 2, 500), max_low_level_steps)
        per_agent_max_time.append(mt)

    # Initial CT constraints (SIPP mode): per-agent list of (loc_key, t_min, t_max)
    initial_ct_flat = []
    if mode == "sipp" and initial_constraints:
        k_robust_ct = 0
        for aid in order:
            agent_ct = []
            if aid in initial_constraints:
                for pos, t in initial_constraints[aid]:
                    loc = pos.y * map_w + pos.x
                    t_min = max(0, t - k_robust_ct)
                    t_max = t + 1 + k_robust_ct
                    agent_ct.append((loc, t_min, t_max))
            initial_ct_flat.append(agent_ct)
    else:
        initial_ct_flat = [[] for _ in order]

    _shelf = shelf_grid if shelf_grid is not None else np.zeros_like(passable_grid, dtype=np.int8)
    _deadline = deadline if deadline is not None else 0.0

    result = _cxx_batch_sequential_plan(
        agents=agents,
        order=list(range(len(order))),  # 0,1,2,... since agents is already in order
        per_agent_max_time=per_agent_max_time,
        passable_grid=passable_grid,
        shelf_grid=_shelf,
        mode=mode,
        horizon_mode=horizon_mode,
        shelf_penalty=float(shelf_penalty),
        planning_window=planning_window,
        k_robust=k_robust,
        initial_ct_flat=initial_ct_flat,
        deadline=_deadline,
    )

    if not result["success"]:
        return None

    # Convert paths: index -> agent_id
    solution = {}
    for idx, path in result["paths"].items():
        aid = order[idx]
        solution[aid] = [Position(x, y) for x, y in path]
    return solution


def _cpp_sipp(start, goals, passable_grid, shelf_grid, rt, max_time, shelf_penalty, deadline=None):
    """Wrapper around C++ cxx_sipp_search that handles None grids.

    Parameters
    ----------
    start : Position
        Start position.
    goals : list of (Position, int)
        Goal locations with release times: ``[(pos, release_time), ...]``.
    passable_grid : np.ndarray or None
        Passability grid.
    shelf_grid : np.ndarray or None
        Shelf grid.
    rt : ReservationTable
        Reservation table with CT/CAT constraints.
    max_time : int
        Maximum search depth.
    shelf_penalty : float
        Shelf traversal penalty.
    deadline : float or None
        Wall-clock deadline (time.time()).

    Returns
    -------
    list of Position
        Planned path, or empty list on failure.
    """
    if passable_grid is None:
        return []
    if deadline is not None and time.time() >= deadline:
        return []
    if not _HAS_CXX_SIPP:
        return []
    from .reservation import _serialize_ct, _serialize_cat
    map_w = passable_grid.shape[1]
    goals_cpp = [(p.x, p.y, rt_val) for p, rt_val in goals]
    ct_flat = _serialize_ct(rt)
    cat_flat = _serialize_cat(rt) if rt.cat else []
    _shelf = shelf_grid if shelf_grid is not None else np.zeros_like(passable_grid, dtype=np.int8)
    result = _cxx_sipp_search(
        start_x=start.x, start_y=start.y,
        goals=goals_cpp,
        passable_grid=passable_grid,
        shelf_grid=_shelf,
        map_width=map_w,
        k_robust=rt.k_robust,
        window=rt.window,
        max_time=max_time,
        shelf_penalty=float(shelf_penalty),
        hold_endpoints=rt.hold_endpoints,
        ct_data=ct_flat,
        existing_paths=[],
        cat_data=cat_flat,
    )
    return [Position(x, y) for x, y in result] if result else []


def _sequential_sipp_attempt(
    passable_grid: np.ndarray,
    shelf_grid: Optional[np.ndarray],
    shelf_penalty: float,
    planning_window: int,
    k_robust: int,
    max_low_level_steps: int,
    initial_constraints,
    order: List[int],
    agvs,
    target_positions,
    goal_sequences=None,
    deadline: Optional[float] = None,
    horizon: Optional[int] = None,
):
    """Sequential multi-agent SIPP planning with interval reservation table.

    Plans agents one by one in *order*, inserting each agent's path as hard
    constraints (CT) and soft constraints (CAT) for subsequent agents.

    Returns a solution dict ``{agent_id: [Position, ...]}`` or ``None`` on failure.
    """
    # Fast path: use C++ batch kernel if available
    if _HAS_CXX_BATCH:
        return _cxx_batch_plan(
            agvs=agvs,
            target_positions=target_positions,
            order=order,
            passable_grid=passable_grid,
            shelf_grid=shelf_grid,
            mode="sipp",
            planning_window=planning_window,
            max_low_level_steps=max_low_level_steps,
            goal_sequences=goal_sequences,
            initial_constraints=initial_constraints,
            k_robust=k_robust,
            shelf_penalty=shelf_penalty,
            deadline=deadline,
        )

    from .reservation import ReservationTable
    from .conflict_utils import find_first_conflict

    if passable_grid is None:
        return None
    map_h, map_w = passable_grid.shape
    k_robust_ct = 0
    rt = ReservationTable(map_size=max(map_h, map_w), k_robust=k_robust,
                         window=planning_window, k_robust_ct=k_robust_ct)

    if deadline is None:
        deadline = time.time() + 5.0  # safety fallback

    solution: Dict[int, List[Position]] = {}
    for agent_id in order:
        if time.time() >= deadline:
            return None
        agv = agvs[agent_id]
        goal = target_positions[agent_id]

        # Insert agent-specific initial constraints into CT (prevent backtracking)
        ic_keys = []
        if agent_id in initial_constraints:
            for pos, t in initial_constraints[agent_id]:
                loc = rt._loc_key(pos, map_w)
                interval = (max(0, t - k_robust_ct), t + 1 + k_robust_ct)
                rt.ct.setdefault(loc, []).append(interval)
                ic_keys.append((loc, interval))

        # Use full goal sequence if available, otherwise single goal
        if goal_sequences and agent_id in goal_sequences and goal_sequences[agent_id]:
            goal_list = [(pos, 0) for pos in goal_sequences[agent_id]]
        else:
            goal_list = [(goal, 0)]
        # Use goal-chain distance to determine search depth, with buffer
        if goal_sequences and agent_id in goal_sequences and len(goal_sequences[agent_id]) > 1:
            chain_dist = _estimate_chain_distance(agv.position, goal_sequences[agent_id])
            sipp_budget = chain_dist + 20
        else:
            sipp_budget = max(planning_window * 4, horizon or 0)
        sipp_max_time = min(sipp_budget, max_low_level_steps)
        path = _cpp_sipp(
            start=agv.position,
            goals=goal_list,
            passable_grid=passable_grid,
            shelf_grid=shelf_grid,
            rt=rt,
            max_time=sipp_max_time,
            shelf_penalty=shelf_penalty,
            deadline=deadline,
        )
        # Fallback: if multi-goal SIPP failed, retry with single goal
        if not path and len(goal_list) > 1:
            path = _cpp_sipp(
                start=agv.position,
                goals=[(goal, 0)],
                passable_grid=passable_grid,
                shelf_grid=shelf_grid,
                rt=rt,
                max_time=sipp_max_time,
                shelf_penalty=shelf_penalty,
                deadline=deadline,
            )
        if not path:
            return None

        # Store the full path — planner returns variable-length paths (>= planning_window).
        # Path truncation for info/visualization happens in the env post-processing layer.
        # Ensure minimum length: pad to planning_window + 1 only if shorter.
        pw = planning_window
        if len(path) < pw + 1:
            path = path + [path[-1]] * (pw + 1 - len(path))
        solution[agent_id] = path
        # Reservation / CT / CAT only cover the planning_window portion.
        res_path = path[:pw + 1]
        rt.insert_path_constraints(res_path, map_w)
        rt.insert_path_to_cat(res_path, map_w)

        # Remove agent-specific initial constraints
        for loc, interval in ic_keys:
            try:
                rt.ct[loc].remove(interval)
            except ValueError:
                pass

    # Verify conflict-free within planning_window, with repair attempts
    from .conflict_utils import force_wait
    for _repair_round in range(10):
        conflict = find_first_conflict(solution, conflict_horizon=planning_window)
        if conflict is None:
            break
        # Try to repair by forcing one agent to wait
        a1, a2 = conflict['agents']
        t = conflict['time']
        repaired = False
        for aid in (a2, a1):
            path = solution.get(aid, [])
            if t > 0 and t < len(path) and path[t] != path[t - 1]:
                force_wait(solution, aid, t)
                repaired = True
                break
        if not repaired:
            return None
    else:
        # Exhausted repair rounds
        conflict = find_first_conflict(solution, conflict_horizon=planning_window)
        if conflict is not None:
            return None
    return solution


class PathPlannerBase(ABC):
    """Abstract base class for all path planners.

    Subclasses implement ``_plan_multi_agent_impl`` with their specific
    algorithm.  The base class provides:

    * **Planning lifecycle** — ``plan``, ``replan``, ``plan_multi_agent``
    * **Path management** — get/set/clear paths and path heads
    * **State queries** — success check, deviation detection
    * **Grid utilities** — passability, heuristic, shelf detection

    State managed by the base class (do NOT reassign in subclasses):
        ``planned_paths``  — Dict[int, List[Position]]
        ``_path_heads``    — Dict[int, int]
        ``_planning_success`` — bool
    """

    SHELF_PENALTY = 3.0

    def __init__(self, grid_map: List[List]):
        self.grid_map = grid_map
        self.height = len(grid_map)
        self.width = len(grid_map[0]) if self.height > 0 else 0
        self.planned_paths: Dict[int, List[Position]] = {}
        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self._path_heads: Dict[int, int] = {}
        self._used_multi_goal_sipp = False
        self._used_multi_goal = False
        self._multi_goal_agents: Set[int] = set()
        self._goal_sequences: Dict[int, List[Position]] = {}
        self._planning_success: bool = True
        self.timing_enabled: bool = False
        self.last_timing: dict = {}
        self._passable_grid: Optional[np.ndarray] = None
        self._shelf_grid: Optional[np.ndarray] = None
        self._planning_window: Optional[int] = None

    # ====================================================================
    #  1. Configuration — called by env before planning
    # ====================================================================

    def set_goal_sequences(self, goal_sequences: Dict[int, List[Position]]):
        """Set ordered task goals for multi-goal low-level search.

        Stores a defensive copy.  Called by the environment before each
        planning round when goal sequences are available.

        Parameters
        ----------
        goal_sequences : dict
            ``{agent_id: [goal1, goal2, ...]}``.  The first element is the
            primary target; subsequent elements are future targets for
            chained multi-goal planning.
        """
        self._goal_sequences = (
            {k: list(v) for k, v in goal_sequences.items()}
            if goal_sequences else {}
        )

    def set_grid_data(self, passable_grid: np.ndarray, shelf_grid: np.ndarray):
        """Attach numpy grid arrays for fast access in A* hot paths.

        Must be called after environment initialisation and whenever the
        grid layout changes.
        """
        self._passable_grid = passable_grid
        self._shelf_grid = shelf_grid

    def set_planning_window(self, window: int):
        """Set the planning window size (number of conflict-free steps)."""
        self._planning_window = int(max(1, window))

    def get_planning_window(self) -> int:
        """Get the planning window size.

        Falls back to ``self.planning_window`` attribute or 10.
        """
        if self._planning_window is not None:
            return self._planning_window
        return getattr(self, 'planning_window', 10)

    def enable_timing(self):
        """Enable timing instrumentation for plan_multi_agent calls."""
        self.timing_enabled = True

    def disable_timing(self):
        """Disable timing instrumentation."""
        self.timing_enabled = False

    # ====================================================================
    #  2. Planning Lifecycle — called by env to trigger planning
    # ====================================================================

    def plan(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        deadline: Optional[float] = None,
    ) -> Dict[int, List[Position]]:
        """Initial planning: compute conflict-free paths for all agents.

        Convenience wrapper around ``plan_multi_agent``.
        Called from ``env.reset()``.
        """
        return self.plan_multi_agent(agvs, target_positions, deadline=deadline)

    def replan(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        deadline: Optional[float] = None,
    ) -> Dict[int, List[Position]]:
        """Replanning: recompute paths from agents' current positions.

        Convenience wrapper around ``plan_multi_agent``.
        Called from ``env.step()``.
        """
        return self.plan_multi_agent(agvs, target_positions, deadline=deadline)

    def plan_multi_agent(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        deadline: Optional[float] = None,
    ) -> Dict[int, List[Position]]:
        """Plan paths for multiple agents (single-pass multi-goal).

        This is the main orchestrator.  It:

        1. Resets ``_planning_success = True``.
        2. Calls ``_plan_multi_agent_impl``.
        3. Stores result in ``self.planned_paths`` and resets ``self._path_heads``.

        Multi-goal planning is handled natively by the low-level search
        backends (A*/SIPP) using goal_id as a state dimension.  The full
        goal sequence is provided via ``set_goal_sequences`` before planning.

        Parameters
        ----------
        deadline : float or None
            Absolute wall-clock deadline passed to ``_plan_multi_agent_impl``.

        Returns
        -------
        dict
            ``{agent_id: [Position, ...]}`` — empty dict on failure.
        """
        if self.timing_enabled:
            self.last_timing = {}
            _t0 = time.perf_counter()

        # Reset planning success flag before each planning round
        self._planning_success = True

        # Single-pass planning through full goal sequences
        result = self._plan_multi_agent_impl(agvs, target_positions, deadline=deadline)

        # Finalise internal state
        self.planned_paths = result
        self._path_heads = {aid: 0 for aid in result}

        if self.timing_enabled:
            self.last_timing.update({
                "total_planning_time": time.perf_counter() - _t0,
                "num_agents": len(agvs),
            })
        return result

    @abstractmethod
    def _plan_multi_agent_impl(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        deadline: Optional[float] = None,
    ) -> Dict[int, List[Position]]:
        """Plan paths for multiple agents (subclass MUST override).

        Contract
        --------
        * Return ``{agent_id: [Position, ...]}`` on success.
        * Return ``{}`` and set ``self._planning_success = False`` on
          failure or timeout.
        * Do NOT assign ``self.planned_paths`` or ``self._path_heads``
          — the base class ``plan_multi_agent`` handles this.

        Parameters
        ----------
        agvs : dict
            ``{agent_id: AGV}`` — agents to plan for.
        target_positions : dict
            ``{agent_id: Position}`` — primary targets.
        deadline : float or None
            Absolute wall-clock deadline.  Abort when exceeded.

        Returns
        -------
        dict
            ``{agent_id: [Position, ...]}`` or ``{}`` on failure.
        """
        ...

    # ====================================================================
    #  3. Result Query — called by env after planning
    # ====================================================================

    def is_last_plan_successful(self) -> bool:
        """Return whether the last plan/replan call succeeded.

        The environment checks this after every ``replan()`` call.
        ``False`` means the planner could not find valid paths.
        """
        return self._planning_success

    # ====================================================================
    #  4. Path Management — read/write planned paths and heads
    # ====================================================================

    def get_path(self, agv_id: int) -> List[Position]:
        """Return the planned path for a single agent (empty list if none)."""
        return self.planned_paths.get(agv_id, [])

    def get_paths(self) -> Dict[int, List[Position]]:
        """Return a copy of all planned paths."""
        return dict(self.planned_paths)

    def set_path(self, agv_id: int, path: List[Position]):
        """Set planned path for a single agent."""
        self.planned_paths[agv_id] = path

    def set_paths(self, paths: Dict[int, List[Position]]):
        """Replace all planned paths."""
        self.planned_paths = dict(paths)

    def get_all_path_ids(self) -> Set[int]:
        """Return set of agent IDs that have planned paths."""
        return set(self.planned_paths.keys())

    def clear_path(self, agv_id: int):
        """Remove planned path and head for a single agent."""
        self.planned_paths.pop(agv_id, None)
        self._path_heads.pop(agv_id, None)

    def clear_all_paths(self):
        """Remove all planned paths and heads."""
        self.planned_paths.clear()
        self._path_heads.clear()

    # ====================================================================
    #  5. Path Head & Step Tracking
    # ====================================================================

    def get_path_head(self, agv_id: int) -> int:
        """Return current path head index for an agent (0 if not tracked)."""
        return self._path_heads.get(agv_id, 0)

    def get_path_heads(self) -> Dict[int, int]:
        """Return a copy of all path heads."""
        return dict(self._path_heads)

    def set_path_head(self, agv_id: int, head: int):
        """Set path head for a single agent."""
        self._path_heads[agv_id] = head

    def set_path_heads(self, heads: Dict[int, int]):
        """Replace all path heads."""
        self._path_heads = dict(heads)

    def advance_step(self, agv_id: int):
        """Advance path head by 1 (called after agent follows its path)."""
        if agv_id in self._path_heads:
            self._path_heads[agv_id] += 1

    # ====================================================================
    #  6. State Query
    # ====================================================================

    def is_deviated(self, agv_id: int, current_pos: Position) -> bool:
        """Check if an AGV has deviated from its planned path.

        An AGV is deviated if:
        1. It has a planned path, and
        2. Its current position is not the next expected position (head+1).
        """
        path = self.get_path(agv_id)
        head = self.get_path_head(agv_id)

        if not path or head + 1 >= len(path):
            return False

        return current_pos != path[head + 1]

    # ====================================================================
    #  7. Grid Utilities — helpers for subclass low-level search
    # ====================================================================

    def _is_valid_position(self, pos: Position) -> bool:
        """Check if a position is within grid bounds."""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def _is_passable(self, pos: Position) -> bool:
        """Check if a position is passable (not a wall)."""
        if not self._is_valid_position(pos):
            return False
        if self._passable_grid is not None:
            return bool(self._passable_grid[pos.y, pos.x])
        return self.grid_map[pos.y][pos.x].passable

    def _is_shelf_to_shelf(self, current_pos: Position, neighbor_pos: Position) -> bool:
        """Check if moving from current_pos to neighbor_pos is a shelf→shelf move."""
        if self._shelf_grid is not None:
            return bool(self._shelf_grid[current_pos.y, current_pos.x] and self._shelf_grid[neighbor_pos.y, neighbor_pos.x])
        current_grid = self.grid_map[current_pos.y][current_pos.x]
        neighbor_grid = self.grid_map[neighbor_pos.y][neighbor_pos.x]
        return isinstance(current_grid, Shelf) and isinstance(neighbor_grid, Shelf) and current_grid is not neighbor_grid

    def _heuristic(self, pos: Position, goal: Position) -> float:
        """Manhattan distance heuristic."""
        return abs(pos.x - goal.x) + abs(pos.y - goal.y)
