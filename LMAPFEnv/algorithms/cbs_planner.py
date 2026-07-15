"""Conflict-Based Search (CBS) planner with conflict resolution fallbacks.

CBS operates in two levels:
- High level: searches a constraint tree, branching on conflicts
- Low level: constrained A* in space-time for individual agents

Only vertex conflicts (two agents at same position at same time) and
edge conflicts (two agents swapping positions) are handled.

When CBS fails to find a solution within node/time limits, a multi-stage
fallback chain is applied:
1. Prioritized planning (sequential with constraints)
2. Iterative conflict resolution (replan conflicting agents)
3. Greedy conflict patching (force wait)
4. Runtime conflict validation (simulate step-by-step)
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..envs.entities import Position, AGV
from .base import PathPlannerBase, _HAS_CXX_ASTAR, _cpp_astar_nocopy
from .conflict_utils import find_first_conflict, get_position_at, force_wait


class CBSNode:
    """CBS constraint tree node with deterministic ordering for heapq."""
    __slots__ = ('cost', 'counter', 'constraints', 'solution')

    def __init__(self, cost: int, counter: int, constraints, solution):
        self.cost = cost
        self.counter = counter
        self.constraints = constraints
        self.solution = solution

    def __lt__(self, other):
        return (self.cost, self.counter) < (other.cost, other.counter)

    def __eq__(self, other):
        return (self.cost, self.counter) == (other.cost, other.counter)

    def __le__(self, other):
        return self == other or self < other


class CBSPlanner(PathPlannerBase):
    """Conflict-Based Search planner for multi-agent path finding.

    CBS operates in two levels:
    - High level: searches a constraint tree, branching on conflicts
    - Low level: constrained A* in space-time for individual agents

    Only vertex conflicts (two agents at same position at same time) and
    edge conflicts (two agents swapping positions) are handled.
    """

    MAX_LOW_LEVEL_STEPS = 500
    MAX_CBS_NODES = 100000
    MAX_PLANNING_TIME = 20.0
    conflict_horizon: Optional[int] = None

    # ── Conflict horizon helpers ────────────────────────────────────────

    def _get_conflict_horizon(self) -> Optional[int]:
        h = getattr(self, "conflict_horizon", None)
        if h is None:
            return None
        return int(max(0, h))

    def _get_max_low_level_time(self) -> int:
        """Return the time-horizon cap for low level space-time A*.

        When a conflict_horizon is active, the search cap is set to
        horizon + 40 -- enough slack for obstacle bypasses without
        exploding the search space.  Without a conflict horizon the
        full MAX_LOW_LEVEL_STEPS budget is available.
        """
        horizon = self._get_conflict_horizon()
        if horizon is None:
            return self.MAX_LOW_LEVEL_STEPS
        return min(self.MAX_LOW_LEVEL_STEPS, horizon + 40)

    def _solution_cost(self, solution: Dict[int, List[Position]],
                       target_positions: Optional[Dict[int, Position]] = None) -> int:
        h = self._get_conflict_horizon()
        if h is None:
            return sum(max(0, len(p) - 1) for p in solution.values())
        total = 0
        for aid, p in solution.items():
            steps = max(0, len(p) - 1)
            if steps == 0:
                if target_positions and aid in target_positions and p and p[-1] == target_positions[aid]:
                    continue
                total += h + 1
            else:
                total += min(steps, h)
        return total

    # ── Main planning ───────────────────────────────────────────────────

    def _plan_multi_agent_impl(self, agvs: Dict[int, AGV], target_positions: Dict[int, Position],
                                deadline: Optional[float] = None) -> Dict[int, List[Position]]:
        self._multi_goal_agents = set()
        solution: Dict[int, List[Position]] = {}
        for agent_id, agv in agvs.items():
            if agent_id in target_positions:
                goals_list = None
                if agent_id in self._goal_sequences and self._goal_sequences[agent_id]:
                    goals_list = self._goal_sequences[agent_id]
                    if len(goals_list) > 1:
                        self._multi_goal_agents.add(agent_id)
                path = self._low_level_search(agv.position, target_positions[agent_id], agv, set(), set(), goals_list)
                solution[agent_id] = path if path else [agv.position]
        self._used_multi_goal = bool(self._multi_goal_agents)

        root_constraints: Dict[int, Dict[str, set]] = {
            aid: {'vertex': set(), 'edge': set()} for aid in solution}
        root_cost = self._solution_cost(solution, target_positions)

        counter = 0
        open_list: list = [CBSNode(root_cost, counter, root_constraints, solution)]
        counter += 1
        nodes_expanded = 0
        start_time = time.time()
        effective_deadline = deadline if deadline is not None else (start_time + self.MAX_PLANNING_TIME)

        while open_list and nodes_expanded < self.MAX_CBS_NODES and time.time() < effective_deadline:
            node = heapq.heappop(open_list)
            constraints, sol = node.constraints, node.solution
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
                    goals_list,
                )

                if not new_path:
                    continue

                child_sol = dict(sol)
                child_sol[constrained_agent] = new_path
                child_cost = self._solution_cost(child_sol, target_positions)
                counter += 1
                heapq.heappush(open_list, CBSNode(child_cost, counter, child_constraints, child_sol))

        # CBS failed to find a conflict-free solution within node / time limits.
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

    # ── Unified fallback chain ─────────────────────────────────────────

    def _run_fallback_chain(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        agent_ids: Optional[List[int]] = None,
        horizon: Optional[int] = None,
        max_resolve_rounds: int = 20,
    ) -> Dict[int, List[Position]]:
        """Run the 4-stage fallback conflict resolution chain.

        Stages:
        1. Prioritized planning (sequential with constraints)
        2. Iterative conflict resolution (replan conflicting agents)
        3. Greedy conflict patching (force wait)
        4. Runtime conflict validation (simulate step-by-step)
        """
        if agent_ids is None:
            agent_ids = list(agvs.keys())
        sol = self._prioritized_fallback(agvs, target_positions, agent_ids, horizon=horizon)
        sol = self._resolve_fallback_conflicts(sol, agvs, target_positions, max_rounds=max_resolve_rounds)
        sol = self._patch_remaining_conflicts(sol, agvs)
        sol = self._validate_runtime_conflicts(sol, agvs)
        return sol

    # ── Fallback conflict resolution methods ────────────────────────────

    def _resolve_fallback_conflicts(
        self,
        solution: Dict[int, List[Position]],
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        max_rounds: int = 5,
    ) -> Dict[int, List[Position]]:
        """Iteratively replan conflicting agents until conflict-free or max_rounds."""
        horizon = self._get_conflict_horizon()
        for _ in range(max_rounds):
            conf_agents: set = set()
            ids = sorted(solution.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if ids[i] not in solution or ids[j] not in solution:
                        continue
                    pair = {ids[i]: solution[ids[i]], ids[j]: solution[ids[j]]}
                    c = find_first_conflict(pair, conflict_horizon=horizon)
                    if c is not None:
                        conf_agents.add(ids[i])
                        conf_agents.add(ids[j])

            if not conf_agents:
                return solution

            replanned_this_round: set = set()
            for aid in sorted(conf_agents):
                if aid not in agvs or aid not in target_positions:
                    continue
                agv = agvs[aid]
                goal = target_positions[aid]
                vertex: set = set()
                edge: set = set()
                for other_id, other_path in solution.items():
                    if other_id == aid:
                        continue
                    if other_id in conf_agents and other_id not in replanned_this_round:
                        continue
                    max_t = horizon if horizon else len(other_path)
                    other_len = len(other_path)
                    if other_len == 1 and horizon:
                        for t in range(min(horizon + 1, max_t)):
                            vertex.add((other_path[0], t))
                    else:
                        for t in range(min(other_len, max_t)):
                            vertex.add((other_path[t], t))
                    for t in range(min(other_len - 1, max_t)):
                        edge.add((other_path[t], other_path[t + 1], t))
                        edge.add((other_path[t + 1], other_path[t], t))
                goals_list = None
                if aid in self._goal_sequences and self._goal_sequences[aid]:
                    goals_list = self._goal_sequences[aid]
                new_path = self._low_level_search(
                    agv.position, goal, agv, vertex, edge, goals_list)
                if new_path:
                    if horizon and len(new_path) > horizon + 1:
                        new_path = new_path[: horizon + 1]
                    solution[aid] = new_path
                else:
                    solution[aid] = [agv.position]
                replanned_this_round.add(aid)

        return solution

    def _patch_remaining_conflicts(
        self,
        solution: Dict[int, List[Position]],
        agvs: Dict[int, AGV],
    ) -> Dict[int, List[Position]]:
        """Greedy conflict patching: force lower-priority agent to wait."""
        horizon = self._get_conflict_horizon()
        max_len = max((len(p) for p in solution.values()), default=0)
        if horizon is not None:
            max_len = min(max_len, horizon + 1)
        max_iterations = max(1, max_len * max(1, len(solution)) * 4)

        for _ in range(max_iterations):
            conflict = find_first_conflict(solution, conflict_horizon=horizon)
            if conflict is None:
                break

            if conflict["type"] == "vertex":
                t = int(conflict["time"])
                pos = conflict["position"]
                agents = list(conflict["agents"])
                if t <= 0:
                    break

                movers = []
                for aid in agents:
                    prev = get_position_at(solution[aid], t - 1)
                    cur = get_position_at(solution[aid], t)
                    if prev is not None and cur == pos and prev != cur:
                        movers.append(aid)
                victim = max(movers) if movers else max(agents)
                before = list(solution.get(victim, []))
                force_wait(solution, victim, t)
                if solution.get(victim, []) == before and movers:
                    for alt in sorted(movers, reverse=True):
                        if alt == victim:
                            continue
                        before_alt = list(solution.get(alt, []))
                        force_wait(solution, alt, t)
                        if solution.get(alt, []) != before_alt:
                            break
            else:
                t = int(conflict["time"])
                victim = max(conflict["agents"])
                force_wait(solution, victim, t + 1)
        return solution

    def _validate_runtime_conflicts(
        self,
        solution: Dict[int, List[Position]],
        agvs: Dict[int, AGV],
    ) -> Dict[int, List[Position]]:
        """Simulate the solution through a proxy of the environment's
        graph-based step() conflict detector and insert wait steps for
        any merge-point collisions the planner's static checks missed.
        """
        horizon = self._get_conflict_horizon()
        max_len = max((len(p) for p in solution.values()), default=0)
        if horizon is not None:
            max_len = min(max_len, horizon + 1)

        for t in range(max_len):
            curr: Dict[Position, list] = {}
            for aid in sorted(solution.keys()):
                pos = get_position_at(solution[aid], t)
                if pos is not None:
                    curr.setdefault(pos, []).append(aid)

            edges: Dict[Tuple[Position, Position], int] = {}
            nodes: set = set()
            for aid in sorted(solution.keys()):
                pos = get_position_at(solution[aid], t)
                nxt = get_position_at(solution[aid], t + 1)
                if pos is None:
                    continue
                if nxt is None or nxt == pos:
                    edges[(pos, pos)] = aid
                    nodes.add(pos)
                else:
                    edges[(pos, nxt)] = aid
                    nodes.add(pos)
                    nodes.add(nxt)

            swap_handled: set = set()
            for (src, dst), aid_a in edges.items():
                if src == dst:
                    continue
                rev = (dst, src)
                if rev in edges:
                    aid_b = edges[rev]
                    if aid_a != aid_b:
                        pair = (min(aid_a, aid_b), max(aid_a, aid_b))
                        if pair not in swap_handled:
                            swap_handled.add(pair)
                            force_wait(solution, pair[1], t + 1)

            stationary_at: Dict[Position, int] = {
                src: aid for (src, dst), aid in edges.items() if src == dst
            }
            for (src, dst), aid in list(edges.items()):
                if src == dst:
                    continue
                stayer = stationary_at.get(dst)
                if stayer is not None and stayer != aid:
                    force_wait(solution, aid, t + 1)

            in_deg: Dict[Position, int] = {}
            for (src, dst), _aid in edges.items():
                in_deg.setdefault(dst, 0)
                if src != dst:
                    in_deg[dst] = in_deg.get(dst, 0) + 1

            merges: Dict[Position, list] = {}
            for dst, deg in in_deg.items():
                if deg > 1:
                    incoming = []
                    for (src, d), aid in edges.items():
                        if d == dst and src != dst:
                            incoming.append(aid)
                    if len(incoming) > 1:
                        merges[dst] = incoming

            if not merges:
                continue

            for dst, incoming in merges.items():
                incoming.sort()
                for victim in incoming[1:]:
                    force_wait(solution, victim, t + 1)

            for t2 in range(t, min(t + 2, max_len)):
                edges2: Dict[Tuple[Position, Position], int] = {}
                in_deg2: Dict[Position, int] = {}
                for aid in sorted(solution.keys()):
                    p = get_position_at(solution[aid], t2)
                    n = get_position_at(solution[aid], t2 + 1)
                    if p is None or n is None:
                        continue
                    edges2[(p, n)] = aid
                    in_deg2.setdefault(n, 0)
                    if p != n:
                        in_deg2[n] = in_deg2.get(n, 0) + 1
                for (src2, dst2), aid_a2 in edges2.items():
                    if src2 == dst2:
                        continue
                    rev2 = (dst2, src2)
                    if rev2 in edges2:
                        aid_b2 = edges2[rev2]
                        if aid_a2 != aid_b2:
                            force_wait(solution, max(aid_a2, aid_b2), t2 + 1)
                stationary2: Dict[Position, int] = {
                    src: aid for (src, dst), aid in edges2.items() if src == dst
                }
                for (src2, dst2), aid2 in list(edges2.items()):
                    if src2 == dst2:
                        continue
                    stayer2 = stationary2.get(dst2)
                    if stayer2 is not None and stayer2 != aid2:
                        force_wait(solution, aid2, t2 + 1)
                for dst2, deg2 in in_deg2.items():
                    if deg2 > 1:
                        incoming2 = []
                        for (src, d), aid in edges2.items():
                            if d == dst2 and src != d:
                                incoming2.append(aid)
                        if len(incoming2) > 1:
                            incoming2.sort()
                            for victim in incoming2[1:]:
                                force_wait(solution, victim, t2 + 1)

        return solution

    def _prioritized_fallback(
        self,
        agvs: Dict[int, AGV],
        target_positions: Dict[int, Position],
        agent_ids: list,
        horizon: Optional[int] = None,
    ) -> Dict[int, List[Position]]:
        """Fallback planner: assign paths sequentially with priority order."""
        solution: Dict[int, List[Position]] = {}
        for aid in agent_ids:
            if aid not in agvs or aid not in target_positions:
                continue
            agv = agvs[aid]
            goal = target_positions[aid]
            goals_list = None
            if aid in self._goal_sequences and self._goal_sequences[aid]:
                goals_list = self._goal_sequences[aid]
            vertex_constraints: set = set()
            edge_constraints: set = set()
            for other_id, other_path in solution.items():
                max_t = horizon if horizon else len(other_path)
                for t in range(min(len(other_path), max_t)):
                    vertex_constraints.add((other_path[t], t))
                for t in range(min(len(other_path) - 1, max_t)):
                    edge_constraints.add((other_path[t], other_path[t + 1], t))
                    edge_constraints.add((other_path[t + 1], other_path[t], t))
            path = self._low_level_search(agv.position, goal, agv,
                                          vertex_constraints, edge_constraints,
                                          goals_list)
            if path:
                if horizon and len(path) > horizon + 1:
                    path = path[: horizon + 1]
            solution[aid] = path if path else [agv.position]
        return solution

    # ── Low level search ────────────────────────────────────────────────

    def _low_level_search(
        self,
        start: Position,
        goal: Position,
        agv: AGV,
        vertex_constraints: Set[Tuple[Position, int]],
        edge_constraints: Set[Tuple[Position, Position, int]],
        goals_list: Optional[List[Position]] = None,
    ) -> List[Position]:
        """Space-time A* respecting vertex and edge constraints.

        Supports multi-goal planning via goals_list parameter.
        Returns the path as a list of Positions (one per time step).
        """
        if not _HAS_CXX_ASTAR:
            return []  # C++ backend required
        if not self._is_passable(start):
            return []
        if (start, 0) in vertex_constraints:
            return []

        # Build flat constraint arrays for zero-copy API
        vc_arr = np.empty(len(vertex_constraints) * 3, dtype=np.int32)
        for i, (pos, t) in enumerate(vertex_constraints):
            vc_arr[i * 3] = pos.x; vc_arr[i * 3 + 1] = pos.y; vc_arr[i * 3 + 2] = t
        ec_arr = np.empty(len(edge_constraints) * 5, dtype=np.int32)
        for i, (p1, p2, t) in enumerate(edge_constraints):
            ec_arr[i * 5] = p1.x; ec_arr[i * 5 + 1] = p1.y
            ec_arr[i * 5 + 2] = p2.x; ec_arr[i * 5 + 3] = p2.y
            ec_arr[i * 5 + 4] = t
        max_time = self._get_max_low_level_time()

        if goals_list:
            goals_tuples = [(p.x, p.y) for p in goals_list]
        else:
            goals_tuples = [(goal.x, goal.y)]

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
