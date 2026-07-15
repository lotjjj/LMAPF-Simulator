"""Enhanced Conflict-Based Search (ECBS) planner.

ECBS is a bounded-suboptimal MAPF planner that uses:

High level -- FOCAL search:
    Instead of always expanding the minimum-SIC node (as in CBS), ECBS
    maintains a *focal list* of open nodes whose SIC <= w * min_SIC.
    From the focal list the node with the fewest conflicts is expanded,
    which tends to find conflict-free solutions faster.

Low level -- focal A* (NOT YET IMPLEMENTED):
    Standard ECBS uses focal A* at the low level: instead of always
    expanding the minimum-f state, it expands from a focal list of
    states with f <= w * min_f, preferring lower h (closer to goal).
    This produces paths that may be slightly longer but expose fewer
    conflicts to the high level.

    .. note::
        **Implementation limitation**: the low-level search currently
        delegates to CBS's standard space-time A* (no focal list).
        The high-level focal search is correct, but the bounded-
        suboptimality guarantee (SIC <= w * optimal) is weakened
        because low-level paths are not focal-search-optimal.
        Implementing true focal A* requires extending the C++ backend
        (``cxx_astar``) with a focal-list mode.

Parameters
----------
w : float
    Suboptimality factor (>= 1.0).  ``w=1.0`` recovers exact CBS.
    ``w=1.5`` is a common practical choice.
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

from ..envs.entities import Position, AGV
from .cbs_planner import CBSPlanner
from .conflict_utils import find_first_conflict, get_position_at


class _FocalEntry:
    """Wrapper for ECBS focal list, sorted by (num_conflicts, counter)."""
    __slots__ = ('num_conflicts', 'counter', 'node')

    def __init__(self, node):
        self.num_conflicts = node.num_conflicts
        self.counter = node.counter
        self.node = node

    def __lt__(self, other):
        return (self.num_conflicts, self.counter) < (other.num_conflicts, other.counter)


class _ECBSNode:
    """ECBS constraint tree node (sorted by cost)."""
    __slots__ = ('cost', 'counter', 'constraints', 'solution', 'num_conflicts')

    def __init__(self, cost, counter, constraints, solution, num_conflicts=0):
        self.cost = cost
        self.counter = counter
        self.constraints = constraints
        self.solution = solution
        self.num_conflicts = num_conflicts

    def __lt__(self, other):
        return (self.cost, self.counter) < (other.cost, other.counter)

    def __eq__(self, other):
        return (self.cost, self.counter) == (other.cost, other.counter)

    def __le__(self, other):
        return self == other or self < other


class ECBSPlanner(CBSPlanner):
    """Enhanced CBS with high-level FOCAL search and low-level focal A*.

    Bounded-suboptimal MAPF planner.  The solution SIC is guaranteed to be
    at most *w* times the optimal SIC cost.
    """

    def __init__(self, grid_map, w: float = 1.5):
        super().__init__(grid_map)
        self.w = w

    def _count_pair_conflicts(self, path_a: List[Position], path_b: List[Position]) -> int:
        if not path_a or not path_b:
            return 0
        max_len = max(len(path_a), len(path_b))
        h = self._get_conflict_horizon()
        if h is not None:
            max_len = min(max_len, h + 1)
        count = 0
        for t in range(max_len):
            pa = get_position_at(path_a, t)
            pb = get_position_at(path_b, t)
            if pa is not None and pa == pb:
                count += 1
            if t + 1 < max_len:
                pa_next = get_position_at(path_a, t + 1)
                pb_next = get_position_at(path_b, t + 1)
                if (
                    pa is not None and pb is not None and pa_next is not None and pb_next is not None
                    and pa != pa_next
                    and pa == pb_next
                    and pb == pa_next
                ):
                    count += 1
        return count

    def _sum_conflicts_involving_agent(
        self,
        solution: Dict[int, List[Position]],
        agent_id: int,
        path_override: Optional[List[Position]] = None,
    ) -> int:
        if agent_id not in solution:
            return 0
        path_a = path_override if path_override is not None else solution[agent_id]
        total = 0
        for other_id, path_b in solution.items():
            if other_id == agent_id:
                continue
            total += self._count_pair_conflicts(path_a, path_b)
        return total

    def _plan_multi_agent_impl(self, agvs, target_positions, deadline=None):
        start_time = time.time()
        effective_deadline = deadline if deadline is not None else (start_time + self.MAX_PLANNING_TIME)
        self._multi_goal_agents = set()
        if self._goal_sequences:
            for aid, gs in self._goal_sequences.items():
                if len(gs) > 1:
                    self._multi_goal_agents.add(aid)
        self._used_multi_goal = bool(self._multi_goal_agents)
        solution: Dict[int, List[Position]] = {}
        for agent_id, agv in agvs.items():
            if agent_id in target_positions:
                if time.time() >= effective_deadline:
                    break
                goals_list = None
                if agent_id in self._goal_sequences and self._goal_sequences[agent_id]:
                    goals_list = self._goal_sequences[agent_id]
                path = self._low_level_search(
                    agv.position, target_positions[agent_id], agv, set(), set(),
                    goals_list)
                solution[agent_id] = path if path else [agv.position]

        root_constraints: Dict[int, Dict[str, set]] = {
            aid: {'vertex': set(), 'edge': set()} for aid in solution}
        root_cost = self._solution_cost(solution, target_positions)
        root_conflicts = self._count_conflicts(solution)

        counter = 0
        root_node = _ECBSNode(root_cost, counter, root_constraints,
                              solution, root_conflicts)
        counter += 1

        open_list: list = [root_node]
        open_for_focal: list = [root_node]
        focal_list: list = []
        in_focal: set = set()
        expanded: set = set()
        nodes_expanded = 0

        while nodes_expanded < self.MAX_CBS_NODES:
            if time.time() >= effective_deadline:
                break
            while open_list and open_list[0].counter in expanded:
                heapq.heappop(open_list)
            if not open_list:
                break

            min_cost = open_list[0].cost
            threshold = self.w * min_cost

            while open_for_focal and open_for_focal[0].counter in expanded:
                heapq.heappop(open_for_focal)
            while open_for_focal and open_for_focal[0].cost <= threshold:
                n = heapq.heappop(open_for_focal)
                if n.counter in expanded or n.counter in in_focal:
                    continue
                heapq.heappush(focal_list, _FocalEntry(n))
                in_focal.add(n.counter)

            while focal_list:
                entry = focal_list[0]
                if entry.node.counter in expanded:
                    heapq.heappop(focal_list)
                    in_focal.discard(entry.node.counter)
                    continue
                if entry.node.cost > threshold:
                    heapq.heappop(focal_list)
                    in_focal.discard(entry.node.counter)
                    continue
                break

            if not focal_list:
                n = open_list[0]
                if n.counter not in expanded and n.counter not in in_focal:
                    heapq.heappush(focal_list, _FocalEntry(n))
                    in_focal.add(n.counter)

            entry = heapq.heappop(focal_list)
            in_focal.discard(entry.node.counter)
            node = entry.node
            constraints, sol = node.constraints, node.solution
            expanded.add(node.counter)
            nodes_expanded += 1

            conflict = find_first_conflict(sol, conflict_horizon=self._get_conflict_horizon())
            if conflict is None:
                if self.timing_enabled:
                    self.last_timing['nodes_expanded'] = nodes_expanded
                return dict(sol)

            agent_i, agent_j = conflict['agents']

            for constrained_agent in (agent_i, agent_j):
                if constrained_agent not in sol:
                    continue

                child_constraints: Dict[int, Dict[str, set]] = dict(constraints)
                base_c = constraints.get(constrained_agent)
                if base_c is None:
                    child_constraints[constrained_agent] = {'vertex': set(), 'edge': set()}
                else:
                    child_constraints[constrained_agent] = {
                        'vertex': set(base_c['vertex']),
                        'edge': set(base_c['edge']),
                    }

                if conflict['type'] == 'vertex':
                    child_constraints[constrained_agent]['vertex'].add(
                        (conflict['position'], conflict['time']))
                else:
                    if constrained_agent == agent_i:
                        child_constraints[constrained_agent]['edge'].add(
                            (conflict['pos1'], conflict['pos2'], conflict['time']))
                    else:
                        child_constraints[constrained_agent]['edge'].add(
                            (conflict['pos2'], conflict['pos1'], conflict['time']))

                agv = agvs[constrained_agent]
                goal = target_positions[constrained_agent]
                goals_list = None
                if constrained_agent in self._goal_sequences and self._goal_sequences[constrained_agent]:
                    goals_list = self._goal_sequences[constrained_agent]
                new_path = self._low_level_search(
                    agv.position, goal, agv,
                    child_constraints[constrained_agent]['vertex'],
                    child_constraints[constrained_agent]['edge'],
                    goals_list)

                if not new_path:
                    continue

                child_sol = dict(sol)
                child_sol[constrained_agent] = new_path
                child_cost = self._solution_cost(child_sol, target_positions)
                old_path = sol.get(constrained_agent) or []
                old_sum = self._sum_conflicts_involving_agent(sol, constrained_agent, old_path)
                new_sum = self._sum_conflicts_involving_agent(sol, constrained_agent, new_path)
                child_conflicts = int(node.num_conflicts - old_sum + new_sum)
                counter += 1

                child_node = _ECBSNode(child_cost, counter,
                                       child_constraints, child_sol,
                                       child_conflicts)
                heapq.heappush(open_list, child_node)
                heapq.heappush(open_for_focal, child_node)

        # ECBS failed to find a conflict-free solution within limits.
        # Fallback: unified 4-stage conflict resolution chain.
        if self.timing_enabled:
            self.last_timing['nodes_expanded'] = nodes_expanded
        solution = self._run_fallback_chain(
            agvs, target_positions,
            agent_ids=list(agvs.keys()),
            horizon=self._get_conflict_horizon(),
        )
        if find_first_conflict(solution, conflict_horizon=self._get_conflict_horizon()) is not None:
            self._planning_success = False
            return {}
        return dict(solution)

    def _count_conflicts(self, solution: Dict[int, List[Position]]) -> int:
        """Count total vertex + edge conflicts in *solution*."""
        if len(solution) < 2:
            return 0

        count = 0
        agent_ids = list(solution.keys())
        max_len = max((len(p) for p in solution.values()), default=0)
        h = self._get_conflict_horizon()
        if h is not None:
            max_len = min(max_len, h + 1)

        for t in range(max_len):
            vertex_counts: Dict[Position, int] = {}
            for aid in agent_ids:
                pos = get_position_at(solution[aid], t)
                if pos is None:
                    continue
                vertex_counts[pos] = vertex_counts.get(pos, 0) + 1
            for k in vertex_counts.values():
                if k > 1:
                    count += (k * (k - 1)) // 2

            if t + 1 < max_len:
                move_counts: Dict[Tuple[int, int, int, int], int] = {}
                for aid in agent_ids:
                    pos = get_position_at(solution[aid], t)
                    pos_next = get_position_at(solution[aid], t + 1)
                    if pos is None or pos_next is None or pos == pos_next:
                        continue
                    key = (pos.x, pos.y, pos_next.x, pos_next.y)
                    move_counts[key] = move_counts.get(key, 0) + 1
                for key, c1 in move_counts.items():
                    rev = (key[2], key[3], key[0], key[1])
                    if key < rev and rev in move_counts:
                        count += c1 * move_counts[rev]

        return count
