"""Priority-Based Search (PBS) planner.

PBS uses a partial priority graph instead of a full constraint tree.
When a conflict is found, PBS branches by adding a priority ordering
between the two conflicting agents.  The lower-priority agent is
replanned with constraints derived from the higher-priority agent's path.

PBS typically explores fewer nodes than CBS because the branching factor
is 2 (priority orderings) rather than creating explicit constraints.
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

from ..envs.entities import Position, AGV
from .cbs_planner import CBSPlanner
from .astar_planner import AStarPlanner
from .conflict_utils import find_first_conflict


class PBSNode:
    """Priority-Based Search node."""
    __slots__ = ('cost', 'depth', 'counter', 'priorities', 'solution')

    def __init__(self, cost: int, depth: int, counter: int, priorities, solution):
        self.cost = cost
        self.depth = depth
        self.counter = counter
        self.priorities = priorities
        self.solution = solution

    def __lt__(self, other):
        return (self.cost, self.depth, self.counter) < (other.cost, other.depth, other.counter)


class PBSPlanner(CBSPlanner):
    """Priority-Based Search using a partial priority graph and constrained replanning."""

    MAX_PBS_NODES = 10000
    MAX_PLANNING_TIME = 20.0

    def _plan_multi_agent_impl(self, agvs: Dict[int, AGV], target_positions: Dict[int, Position],
                                deadline: Optional[float] = None) -> Dict[int, List[Position]]:
        self._multi_goal_agents = set()
        for aid, gs in (self._goal_sequences or {}).items():
            if len(gs) > 1:
                self._multi_goal_agents.add(aid)
        self._used_multi_goal = bool(self._multi_goal_agents)
        active_ids = [aid for aid in agvs.keys() if aid in target_positions]
        root_solution = self._plan_independent_paths(agvs, target_positions, active_ids)
        if not active_ids:
            return {}

        root_conflict = find_first_conflict(root_solution, conflict_horizon=self._get_conflict_horizon())
        if root_conflict is None:
            return root_solution

        counter = 0
        open_list: list[PBSNode] = [PBSNode(self._solution_cost(root_solution, target_positions), 0, counter, set(), root_solution)]
        counter += 1
        visited = {self._priority_signature(set())}
        nodes_expanded = 0
        start_time = time.time()
        effective_deadline = deadline if deadline is not None else (start_time + self.MAX_PLANNING_TIME)

        while open_list and nodes_expanded < self.MAX_PBS_NODES and time.time() < effective_deadline:
            node = heapq.heappop(open_list)
            nodes_expanded += 1

            conflict = find_first_conflict(node.solution, conflict_horizon=self._get_conflict_horizon())
            if conflict is None:
                if self.timing_enabled:
                    self.last_timing['nodes_expanded'] = nodes_expanded
                return dict(node.solution)

            agent_i, agent_j = conflict['agents']
            for higher, lower in ((agent_i, agent_j), (agent_j, agent_i)):
                child_priorities = set(node.priorities)
                child_priorities.add((higher, lower))
                signature = self._priority_signature(child_priorities)
                if signature in visited:
                    continue
                visited.add(signature)

                child_solution = self._plan_with_priorities(agvs, target_positions, active_ids, child_priorities)
                if child_solution is None:
                    continue

                heapq.heappush(
                    open_list,
                    PBSNode(
                        self._solution_cost(child_solution, target_positions),
                        node.depth + 1,
                        counter,
                        child_priorities,
                        child_solution,
                    ),
                )
                counter += 1

        # PBS failed to find a conflict-free priority ordering within limits.
        # Fallback: unified 4-stage conflict resolution chain.
        if self.timing_enabled:
            self.last_timing['nodes_expanded'] = nodes_expanded
        fallback = self._run_fallback_chain(
            agvs, target_positions,
            agent_ids=active_ids,
            horizon=self._get_conflict_horizon(),
        )
        if find_first_conflict(fallback, conflict_horizon=self._get_conflict_horizon()) is not None:
            self._planning_success = False
            return {}
        return fallback

    def _plan_independent_paths(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        active_ids: List[int],
    ) -> Dict[int, List[Position]]:
        astar = AStarPlanner(self.grid_map)
        astar.SHELF_PENALTY = self.SHELF_PENALTY
        if self._passable_grid is not None:
            astar.set_grid_data(self._passable_grid, self._shelf_grid)

        solution: Dict[int, List[Position]] = {}
        for aid in active_ids:
            agv = agvs[aid]
            goal = target_positions[aid]
            goals_list = None
            if aid in self._goal_sequences and self._goal_sequences[aid]:
                goals_list = self._goal_sequences[aid]
            path = astar._single_astar(agv.position, goal, agv, goals_list)
            solution[aid] = path if path else [agv.position]
        return solution

    def _plan_with_priorities(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        active_ids: List[int],
        priorities: Set[Tuple[int, int]],
    ) -> Optional[Dict[int, List[Position]]]:
        order = self._topological_order(active_ids, priorities)
        if order is None:
            return None

        ancestors = self._compute_ancestors(active_ids, priorities)
        solution: Dict[int, List[Position]] = {}
        for aid in order:
            agv = agvs[aid]
            goal = target_positions[aid]
            higher_ids = ancestors.get(aid, set())
            vertex_constraints, edge_constraints = self._build_priority_constraints(solution, higher_ids)
            goals_list = None
            if aid in self._goal_sequences and self._goal_sequences[aid]:
                goals_list = self._goal_sequences[aid]
            path = self._low_level_search(agv.position, goal, agv, vertex_constraints, edge_constraints, goals_list)
            if not path:
                return None
            solution[aid] = path

        return solution

    def _build_priority_constraints(
        self,
        solution: Dict[int, List[Position]],
        higher_ids: Set[int],
    ) -> Tuple[Set[Tuple[Position, int]], Set[Tuple[Position, Position, int]]]:
        vertex_constraints: Set[Tuple[Position, int]] = set()
        edge_constraints: Set[Tuple[Position, Position, int]] = set()
        if not higher_ids:
            return vertex_constraints, edge_constraints

        conflict_horizon = self._get_conflict_horizon()
        # Only constrain up to the actual path length or conflict horizon.
        # Previously this was max_planned_time + MAX_LOW_LEVEL_STEPS (up to
        # ~500 extra timesteps), generating massive constraint sets with
        # clamped final positions that slowed down the low-level A*.
        if conflict_horizon is not None:
            vertex_limit = conflict_horizon
        else:
            vertex_limit = max(
                (len(solution[aid]) for aid in higher_ids if aid in solution),
                default=0,
            )

        for aid in higher_ids:
            path = solution.get(aid)
            if not path:
                continue
            # Vertex constraints: one per timestep in the path
            for t in range(min(len(path), vertex_limit + 1)):
                vertex_constraints.add((path[t], t))
            # If path is shorter than vertex_limit, hold the endpoint
            if len(path) <= vertex_limit:
                end_pos = path[-1]
                for t in range(len(path), vertex_limit + 1):
                    vertex_constraints.add((end_pos, t))

            # Edge constraints: one per transition in the path
            for t in range(min(len(path) - 1, vertex_limit)):
                pos = path[t]
                pos_next = path[t + 1]
                if pos != pos_next:
                    edge_constraints.add((pos, pos_next, t))
                    edge_constraints.add((pos_next, pos, t))

        return vertex_constraints, edge_constraints

    @staticmethod
    def _priority_signature(priorities: Set[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted(priorities))

    @staticmethod
    def _topological_order(active_ids: List[int], priorities: Set[Tuple[int, int]]) -> Optional[List[int]]:
        active = set(active_ids)
        indegree = {aid: 0 for aid in active_ids}
        outgoing: Dict[int, Set[int]] = {aid: set() for aid in active_ids}

        for higher, lower in priorities:
            if higher not in active or lower not in active or higher == lower:
                continue
            if lower not in outgoing[higher]:
                outgoing[higher].add(lower)
                indegree[lower] += 1

        ready = [aid for aid, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        order: List[int] = []
        while ready:
            aid = heapq.heappop(ready)
            order.append(aid)
            for nxt in sorted(outgoing[aid]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(ready, nxt)

        if len(order) != len(active_ids):
            return None
        return order

    @staticmethod
    def _compute_ancestors(active_ids: List[int], priorities: Set[Tuple[int, int]]) -> Dict[int, Set[int]]:
        parents: Dict[int, Set[int]] = {aid: set() for aid in active_ids}
        active = set(active_ids)
        for higher, lower in priorities:
            if higher in active and lower in active and higher != lower:
                parents[lower].add(higher)

        ancestors: Dict[int, Set[int]] = {}
        for aid in active_ids:
            seen: Set[int] = set()
            stack = list(parents[aid])
            while stack:
                parent = stack.pop()
                if parent in seen:
                    continue
                seen.add(parent)
                stack.extend(parents.get(parent, ()))
            ancestors[aid] = seen
        return ancestors
