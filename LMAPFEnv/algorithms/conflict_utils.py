"""Standalone conflict detection and resolution utilities for MAPF planners.

These functions were originally static/instance methods on ``CBSPlanner``.
They are extracted here so that any planner (not just CBS subclasses) can
reuse them without inheritance coupling.
"""

from typing import Dict, List, Optional, Tuple

from ..envs.entities import Position


def get_position_at(path: List[Position], t: int) -> Optional[Position]:
    """Return the position at timestep *t*, clamping to the last position."""
    if not path:
        return None
    if t < len(path):
        return path[t]
    return path[-1]


def find_first_conflict(
    solution: Dict[int, List[Position]],
    conflict_horizon: Optional[int] = None,
) -> Optional[dict]:
    """Find the first vertex or edge conflict in *solution*.

    Returns a dict describing the conflict, or ``None`` if conflict-free.
    """
    if len(solution) < 2:
        return None

    agent_ids = sorted(solution.keys())
    max_len = max((len(p) for p in solution.values()), default=0)
    if conflict_horizon is not None:
        max_len = min(max_len, int(max(0, conflict_horizon)) + 1)

    for t in range(max_len):
        occupied: Dict[Position, int] = {}
        for aid in agent_ids:
            pos = get_position_at(solution[aid], t)
            if pos is None:
                continue
            other = occupied.get(pos)
            if other is not None:
                return {
                    'type': 'vertex',
                    'agents': (other, aid),
                    'position': pos,
                    'time': t,
                }
            occupied[pos] = aid

        if t + 1 < max_len:
            moves: Dict[Tuple[Position, Position], int] = {}
            for aid in agent_ids:
                pos = get_position_at(solution[aid], t)
                pos_next = get_position_at(solution[aid], t + 1)
                if pos is None or pos_next is None or pos == pos_next:
                    continue
                rev = (pos_next, pos)
                other = moves.get(rev)
                if other is not None:
                    return {
                        'type': 'edge',
                        'agents': (other, aid),
                        'pos1': pos_next,
                        'pos2': pos,
                        'time': t,
                    }
                moves[(pos, pos_next)] = aid

    return None


def force_wait(solution: Dict[int, List[Position]], agent_id: int, t: int):
    """Force *agent_id* to stay at its position at time *t*.

    The path is extended/spliced so that ``path[t] == path[t-1]``
    (the agent stays in place at time t).  If the path is shorter than
    ``t``, it is first padded with the last position.
    """
    path = solution.get(agent_id)
    if not path:
        return
    if t >= len(path):
        # Pad with last position
        path.extend([path[-1]] * (t + 1 - len(path)))
    elif t > 0:
        path[t] = path[t - 1]
    # If t == 0 we can't "stay before moving" - nothing to do.
