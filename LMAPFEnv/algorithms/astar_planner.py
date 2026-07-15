"""Single-agent A* planners (independent, no inter-agent coordination)."""

from typing import Dict, List, Optional, Set, Tuple

from ..envs.entities import Position, AGV
from .base import PathPlannerBase, _HAS_CXX_ASTAR, _cpp_astar_nocopy, _estimate_chain_distance
import numpy as np


class AStarPlanner(PathPlannerBase):
    """Single-agent A* planner (independent, no inter-agent coordination).

    Uses C++ cxx_astar backend exclusively.  No Python fallback.
    """
    MAX_LOW_LEVEL_STEPS = 500

    def _plan_multi_agent_impl(self, agvs: Dict[int, AGV], target_positions: Dict[int, Position],
                                deadline: Optional[float] = None) -> Dict[int, List[Position]]:
        """Plan independently for each agent using C++ A*"""
        self._multi_goal_agents = set()
        paths = {}
        for agent_id, agv in agvs.items():
            if agent_id in target_positions:
                goals_list = None
                if agent_id in self._goal_sequences and self._goal_sequences[agent_id]:
                    goals_list = self._goal_sequences[agent_id]
                    if len(goals_list) > 1:
                        self._multi_goal_agents.add(agent_id)
                path = self._single_astar(agv.position, target_positions[agent_id], agv, goals_list)
                paths[agent_id] = path

        self._used_multi_goal = bool(self._multi_goal_agents)
        # Mark failure if any agent has no path
        if any(not p for p in paths.values()):
            self._planning_success = False
        return paths

    def _single_astar(self, start: Position, goal: Position, agv: AGV,
                      goals_list: Optional[List[Position]] = None) -> List[Position]:
        """Run C++ A* search for a single agent. Returns path including start position.

        When *goals_list* is provided, performs multi-goal chained planning
        via the C++ A* with goal_id support.
        Uses the zero-copy API (_cpp_astar_nocopy) with the same parameter
        defaults as RHCRPlanner for maximum performance.
        """
        if not _HAS_CXX_ASTAR:
            return []  # C++ backend required

        if start == goal and (not goals_list or len(goals_list) <= 1):
            return [start]
        if goals_list and len(goals_list) > 1:
            goals_tuples = [(p.x, p.y) for p in goals_list]
        else:
            goals_tuples = [(goal.x, goal.y)]

        # Empty constraint arrays (no inter-agent coordination)
        vc_arr = np.empty(0, dtype=np.int32)
        ec_arr = np.empty(0, dtype=np.int32)

        # Search depth: chain distance for multi-goal, reasonable cap for single
        if goals_list and len(goals_list) > 1:
            chain_dist = _estimate_chain_distance(start, goals_list)
            max_time = min(chain_dist + 20, self.MAX_LOW_LEVEL_STEPS)
        else:
            max_time = 500

        path = _cpp_astar_nocopy(
            start=(start.x, start.y),
            goals=goals_tuples,
            passable_grid=self._passable_grid,
            shelf_grid=self._shelf_grid,
            vc_flat=vc_arr,
            ec_flat=ec_arr,
            max_time=max_time,
            horizon_mode=False,
            use_closed_set=False,
            tie_breaker_by_depth=True,
            shelf_penalty=self.SHELF_PENALTY,
        )
        if path:
            return [Position(x, y) for x, y in path]
        # Fallback: if multi-goal search failed, retry with single goal
        if goals_list and len(goals_list) > 1:
            fallback = _cpp_astar_nocopy(
                start=(start.x, start.y),
                goals=[(goal.x, goal.y)],
                passable_grid=self._passable_grid,
                shelf_grid=self._shelf_grid,
                vc_flat=vc_arr,
                ec_flat=ec_arr,
                max_time=max_time,
                horizon_mode=False,
                use_closed_set=False,
                tie_breaker_by_depth=True,
                shelf_penalty=self.SHELF_PENALTY,
            )
            return [Position(x, y) for x, y in fallback] if fallback else []
        return []


class EnhancedAStarPlanner(AStarPlanner):
    """A* with avoidance of currently occupied nodes of FOV-visible AGVs.

    Uses C++ A* backend exclusively.  When penalty positions are present,
    first plans a base path, then if the path crosses penalty positions,
    replans with those positions as hard vertex constraints at the
    estimated arrival times to route around visible AGVs.
    """

    def __init__(self, grid_map: List[List], visible_agv_penalty: float = 5.0):
        super().__init__(grid_map)
        self.visible_agv_penalty = float(max(0.0, visible_agv_penalty))

    def _plan_multi_agent_impl(self, agvs: Dict[int, AGV], target_positions: Dict[int, Position],
                                deadline: Optional[float] = None) -> Dict[int, List[Position]]:
        self._multi_goal_agents = set()
        paths = {}
        for agent_id, agv in agvs.items():
            if agent_id not in target_positions:
                continue
            goal = target_positions[agent_id]
            goals_list = None
            if agent_id in self._goal_sequences and self._goal_sequences[agent_id]:
                goals_list = self._goal_sequences[agent_id]
                if len(goals_list) > 1:
                    self._multi_goal_agents.add(agent_id)
            penalty_positions = self._get_visible_agv_positions(agv, agvs, goal)
            paths[agent_id] = self._single_astar_with_penalty(
                agv.position, goal, agv, penalty_positions, goals_list)

        self._used_multi_goal = bool(self._multi_goal_agents)
        if any(not p for p in paths.values()):
            self._planning_success = False
        return paths

    def _single_astar_with_penalty(
        self,
        start: Position,
        goal: Position,
        agv: AGV,
        penalty_positions: Set[Position],
        goals_list: Optional[List[Position]] = None,
    ) -> List[Position]:
        """C++ A* with AGV avoidance via vertex constraints.

        Strategy:
        1. Plan base path with C++ A*.
        2. If path crosses penalty positions, add vertex constraints at
           estimated arrival times and replan to route around them.
        3. If constrained path fails, fall back to base path.
        """
        if not _HAS_CXX_ASTAR:
            return []  # C++ backend required

        if self.visible_agv_penalty <= 0.0 or not penalty_positions:
            return self._single_astar(start, goal, agv, goals_list)

        if start == goal and (not goals_list or len(goals_list) <= 1):
            return [start]

        # Step 1: Plan base path
        base_path = self._single_astar(start, goal, agv, goals_list)
        if not base_path:
            return []

        # Step 2: Check if base path crosses penalty positions
        crossing_times: List[Tuple[int, Position]] = []
        for t, pos in enumerate(base_path):
            if t > 0 and pos in penalty_positions and pos != goal:
                crossing_times.append((t, pos))

        if not crossing_times:
            return base_path  # No crossing, use base path

        # Step 3: Add vertex constraints at crossing times and replan
        vertex_constraints = [(pos.x, pos.y, t) for t, pos in crossing_times]
        vc_arr = np.empty(len(vertex_constraints) * 3, dtype=np.int32)
        for i, (x, y, t) in enumerate(vertex_constraints):
            vc_arr[i * 3] = x; vc_arr[i * 3 + 1] = y; vc_arr[i * 3 + 2] = t
        ec_arr = np.empty(0, dtype=np.int32)
        if goals_list and len(goals_list) > 1:
            goals_tuples = [(p.x, p.y) for p in goals_list]
        else:
            goals_tuples = [(goal.x, goal.y)]
        if goals_list and len(goals_list) > 1:
            chain_dist = _estimate_chain_distance(start, goals_list)
            max_time = min(chain_dist + 20, self.MAX_LOW_LEVEL_STEPS)
        else:
            max_time = 500
        path = _cpp_astar_nocopy(
            start=(start.x, start.y),
            goals=goals_tuples,
            passable_grid=self._passable_grid,
            shelf_grid=self._shelf_grid,
            vc_flat=vc_arr,
            ec_flat=ec_arr,
            max_time=max_time,
            horizon_mode=False,
            use_closed_set=False,
            tie_breaker_by_depth=True,
            shelf_penalty=self.SHELF_PENALTY,
        )
        if path:
            return [Position(x, y) for x, y in path]
        return base_path  # Fallback to base path if constrained search fails

    @staticmethod
    def _get_visible_agv_positions(agv: AGV, agvs: Dict[int, AGV], goal: Position) -> Set[Position]:
        radius = int(max(0, agv.fov_size // 2))
        if radius <= 0:
            return set()

        penalty_positions: Set[Position] = set()
        for other_id, other_agv in agvs.items():
            if other_id == agv.id:
                continue
            if abs(other_agv.x - agv.x) <= radius and abs(other_agv.y - agv.y) <= radius:
                if other_agv.position != goal:
                    penalty_positions.add(other_agv.position)
        return penalty_positions
