"""RHCR-family solvers that combine RHCR windowing with CBS/PBS/ECBS high-level search.

These planners wrap a high-level MAPF solver (CBS, PBS, or ECBS) inside
the RHCR rolling-horizon framework:

1. Compute horizon targets by planning independent A* paths to goals.
2. Use the high-level solver (CBS/PBS/ECBS) to plan conflict-free paths
   to the horizon targets within the planning window.
3. Extend the windowed paths with A* paths beyond the planning window
   for a complete (but conflict-free only within W) solution.
4. If the solver fails, apply fallback conflict resolution inherited
   from CBSPlanner.

All three concrete classes share the same planning flow, differing only
in which solver class is instantiated (via ``_create_solver``).
"""

import time
from typing import Dict, List, Optional, Set, Tuple

from ..envs.entities import Position, AGV
from .base import PathPlannerBase, _sequential_sipp_attempt
from .astar_planner import AStarPlanner
from .cbs_planner import CBSPlanner
from .pbs_planner import PBSPlanner
from .ecbs_planner import ECBSPlanner
from .conflict_utils import find_first_conflict


class _RHCRSolverBase(PathPlannerBase):
    """Base class for RHCR-family solvers wrapping a high-level MAPF solver.

    Subclasses must implement :meth:`_create_solver` to return the
    appropriate solver instance (CBSPlanner, PBSPlanner, or ECBSPlanner).
    All shared planning logic — SIPP fast path, horizon-target
    computation, window-path extension, and fallback conflict resolution —
    lives here.
    """

    MAX_PLANNING_TIME = 5.0

    def __init__(
        self,
        grid_map,
        planning_window: int = 10,
        horizon: int = 1,
        max_solver_nodes: int = 2000,
        max_low_level_steps: int = 500,
        use_sipp: bool = False,
        k_robust: int = 0,
        suboptimal_bound: float = 1.0,
    ):
        super().__init__(grid_map)
        self.planning_window = int(max(1, planning_window))
        self.horizon = int(max(1, horizon))
        if self.horizon >= self.planning_window:
            self.horizon = max(1, self.planning_window - 1)
        self.max_solver_nodes = int(max(1, max_solver_nodes))
        self.max_low_level_steps = int(max(1, max_low_level_steps))
        self.use_sipp = bool(use_sipp)
        self.k_robust = int(max(0, k_robust))
        self.suboptimal_bound = float(max(1.0, suboptimal_bound))
        self._initial_constraints: Dict[int, List[Tuple[Position, int]]] = {}
        self._goal_sequences: Dict[int, List[Position]] = {}

    # ── Public setters (consistent defensive-copy contract) ────────────

    def set_initial_constraints(self, constraints: Dict[int, List[Tuple[Position, int]]]):
        self._initial_constraints = constraints

    # ── Hook: subclasses create their specific solver ──────────────────

    def _create_solver(self, grid_map) -> CBSPlanner:
        """Create and return the high-level solver instance.

        Override in subclass.  The returned solver must be a CBSPlanner
        subclass so that fallback methods are available.
        """
        raise NotImplementedError

    # ── Shared planning helpers ────────────────────────────────────────

    def _compute_horizon_targets(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
    ) -> Tuple[Dict[int, Position], Dict[int, List[Position]]]:
        """Compute intermediate horizon targets via independent A* planning.

        Returns (horizon_targets, astar_paths) where *astar_paths* stores
        the full A* paths for later extension beyond the planning window.
        """
        horizon_targets: Dict[int, Position] = {}
        astar_paths: Dict[int, List[Position]] = {}
        astar = AStarPlanner(self.grid_map)
        astar.SHELF_PENALTY = self.SHELF_PENALTY
        if self._passable_grid is not None:
            astar.set_grid_data(self._passable_grid, self._shelf_grid)
        for aid, agv in agvs.items():
            if aid not in target_positions:
                continue
            goal = target_positions[aid]
            path = astar._single_astar(agv.position, goal, agv)
            if not path:
                horizon_targets[aid] = agv.position
                continue
            astar_paths[aid] = path
            idx = min(self.planning_window, len(path) - 1)
            horizon_targets[aid] = path[idx]
        return horizon_targets, astar_paths

    def _extend_window_paths(
        self,
        window_paths: Dict[int, List[Position]],
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        astar_paths: Dict[int, List[Position]],
    ) -> Dict[int, List[Position]]:
        """Extend window paths with A* tails for complete paths.

        Window paths from the high-level solver cover [0, planning_window].
        We extend them with the independent A* path tail to produce a
        variable-length path that reaches the goal.
        """
        pw = self.planning_window
        solution: Dict[int, List[Position]] = {}
        for aid, agv in agvs.items():
            if aid not in target_positions:
                continue
            window = window_paths.get(aid) or [agv.position]
            # Pad to at least planning_window + 1
            if len(window) < pw + 1:
                window = window + [window[-1]] * (pw + 1 - len(window))
            # Extend with A* path beyond the window.
            astar_full = astar_paths.get(aid)
            if astar_full and len(astar_full) > pw + 1:
                # Simple extension: use A* path from index pw+1 onward.
                # The window endpoint and A*[pw] may differ, so we append
                # the A* tail directly (the agent will follow the window
                # path within the conflict-free zone, then the A* tail).
                solution[aid] = window[:pw + 1] + astar_full[pw + 1:]
            else:
                solution[aid] = window
        return solution

    def _apply_fallback(
        self,
        solver: CBSPlanner,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        horizon_targets: Dict[int, Position],
        astar_paths: Dict[int, List[Position]],
    ) -> Dict[int, List[Position]]:
        """Run the 4-stage fallback conflict resolution and extend paths."""
        fallback_sol = solver._run_fallback_chain(
            agvs, horizon_targets,
            agent_ids=[aid for aid in agvs if aid in target_positions],
            horizon=self.planning_window,
            max_resolve_rounds=5,
        )
        return self._extend_window_paths(
            fallback_sol, agvs, target_positions, astar_paths,
        )

    @staticmethod
    def _distance_sort_key(
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
    ):
        """Return a sort key that prioritises agents farthest from their targets."""
        def key(aid: int):
            agv = agvs[aid]
            goal = target_positions[aid]
            return (-(abs(agv.position.x - goal.x)
                       + abs(agv.position.y - goal.y)), aid)
        return key

    # ── Main planning ───────────────────────────────────────────────────

    def _plan_multi_agent_impl(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        deadline: Optional[float] = None,
    ) -> Dict[int, List[Position]]:
        active_ids = [aid for aid in agvs.keys() if aid in target_positions]

        # Track multi-goal agents
        self._multi_goal_agents = set()
        if self._goal_sequences:
            for aid, gs in self._goal_sequences.items():
                if len(gs) > 1:
                    self._multi_goal_agents.add(aid)
        has_multi_goals = bool(self._multi_goal_agents)
        self._used_multi_goal = has_multi_goals

        # ── SIPP fast path ──────────────────────────────────────────────
        if self.use_sipp:
            self._used_multi_goal_sipp = has_multi_goals
            sort_key = self._distance_sort_key(agvs, target_positions)
            order = sorted(active_ids, key=sort_key)
            sipp_sol = _sequential_sipp_attempt(
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
                deadline=deadline,
                horizon=max(self.planning_window, self.horizon),
            )
            if sipp_sol is not None:
                return sipp_sol
            self._used_multi_goal_sipp = False  # SIPP failed, falling back

        # ── High-level solver path ─────────────────────────────────────
        horizon_targets, astar_paths = self._compute_horizon_targets(
            agvs, target_positions,
        )

        solver = self._create_solver(self.grid_map)
        solver.SHELF_PENALTY = self.SHELF_PENALTY
        solver.MAX_LOW_LEVEL_STEPS = self.max_low_level_steps
        # Give inner solver a small time budget so the outer fallback
        # chain has time to run if the solver fails.
        outer_budget = float(getattr(self, "MAX_PLANNING_TIME", 5.0))
        solver.MAX_PLANNING_TIME = min(outer_budget * 0.3, 3.0)
        solver.conflict_horizon = self.planning_window
        if self.timing_enabled:
            solver.enable_timing()
        # Propagate goal sequences for chain heuristic multi-goal planning.
        solver.set_goal_sequences(self._goal_sequences)
        if self._passable_grid is not None:
            solver.set_grid_data(self._passable_grid, self._shelf_grid)

        window_paths = solver.plan_multi_agent(
            agvs, horizon_targets, deadline=None,
        )

        # Propagate inner solver timing to outer planner
        if self.timing_enabled and hasattr(solver, 'last_timing'):
            self.last_timing['inner_nodes_expanded'] = solver.last_timing.get('nodes_expanded', 0)
            self.last_timing['inner_planning_time'] = solver.last_timing.get('total_planning_time', 0.0)

        solution = self._extend_window_paths(
            window_paths, agvs, target_positions, astar_paths,
        )

        if find_first_conflict(solution, conflict_horizon=self.planning_window) is not None:
            solution = self._apply_fallback(
                solver, agvs, target_positions, horizon_targets, astar_paths,
            )

        # Check if solution is conflict-free within planning_window.
        # If not, return empty paths to signal planning failure.
        remaining = find_first_conflict(solution, conflict_horizon=self.planning_window)
        if remaining is not None:
            self._planning_success = False
            return {}

        return solution


# ── Concrete solver classes ──────────────────────────────────────────────


class RHCRCBSPlanner(_RHCRSolverBase):
    """RHCR with CBS high-level solver.

    Combines rolling-horizon windowing with Conflict-Based Search.
    Conflict-free guarantee within planning_window W.
    """

    def __init__(
        self,
        grid_map,
        planning_window: int = 10,
        horizon: int = 1,
        max_cbs_nodes: int = 2000,
        max_low_level_steps: int = 500,
        use_sipp: bool = False,
        k_robust: int = 0,
        suboptimal_bound: float = 1.0,
    ):
        super().__init__(
            grid_map,
            planning_window=planning_window,
            horizon=horizon,
            max_solver_nodes=max_cbs_nodes,
            max_low_level_steps=max_low_level_steps,
            use_sipp=use_sipp,
            k_robust=k_robust,
            suboptimal_bound=suboptimal_bound,
        )

    def _create_solver(self, grid_map) -> CBSPlanner:
        solver = CBSPlanner(grid_map)
        solver.MAX_CBS_NODES = self.max_solver_nodes
        return solver


class RHCRPBSPlanner(_RHCRSolverBase):
    """RHCR with PBS high-level solver.

    Combines rolling-horizon windowing with Priority-Based Search.
    Conflict-free guarantee within planning_window W.
    """

    def __init__(
        self,
        grid_map,
        planning_window: int = 10,
        horizon: int = 1,
        max_pbs_nodes: int = 2000,
        max_low_level_steps: int = 500,
        use_sipp: bool = False,
        k_robust: int = 0,
        suboptimal_bound: float = 1.0,
    ):
        super().__init__(
            grid_map,
            planning_window=planning_window,
            horizon=horizon,
            max_solver_nodes=max_pbs_nodes,
            max_low_level_steps=max_low_level_steps,
            use_sipp=use_sipp,
            k_robust=k_robust,
            suboptimal_bound=suboptimal_bound,
        )

    def _create_solver(self, grid_map) -> PBSPlanner:
        solver = PBSPlanner(grid_map)
        solver.MAX_PBS_NODES = self.max_solver_nodes
        return solver


class RHCRECBSPlanner(_RHCRSolverBase):
    """RHCR with ECBS high-level solver.

    Combines rolling-horizon windowing with Enhanced Conflict-Based Search.
    Conflict-free guarantee within planning_window W.
    Bounded suboptimal with factor *w*.
    """

    def __init__(
        self,
        grid_map,
        planning_window: int = 10,
        horizon: int = 1,
        w: float = 1.5,
        max_cbs_nodes: int = 2000,
        max_low_level_steps: int = 500,
        use_sipp: bool = False,
        k_robust: int = 0,
        suboptimal_bound: float = 1.5,
    ):
        super().__init__(
            grid_map,
            planning_window=planning_window,
            horizon=horizon,
            max_solver_nodes=max_cbs_nodes,
            max_low_level_steps=max_low_level_steps,
            use_sipp=use_sipp,
            k_robust=k_robust,
            suboptimal_bound=suboptimal_bound,
        )
        self.w = float(w)

    def _create_solver(self, grid_map) -> ECBSPlanner:
        solver = ECBSPlanner(grid_map, w=self.w)
        solver.MAX_CBS_NODES = self.max_solver_nodes
        return solver
