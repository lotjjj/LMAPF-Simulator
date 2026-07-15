"""PlannerPolicy: converts planned paths into discrete actions for active agents."""

from typing import Dict, List

from ..envs.entities import AGV, Action


class PlannerPolicy:
    """Policy that converts planned paths into discrete actions for active agents.

    PlannerPolicy reads path information from the ``info`` dict returned by
    ``env.step()`` rather than accessing the planner's internal state
    directly.  This ensures the policy can operate using only the
    publicly-exposed path data (``info[agent]['planner_paths']``).
    """

    def __init__(self, planner=None):
        self.planner = planner  # kept for backward compat; not used for path data
        self._last_info: Dict[str, Dict] = {}

    def update_info(self, info: Dict[str, Dict]):
        """Store the latest info dict from env.step() / env.reset()."""
        self._last_info = info or {}

    def select_actions(self, agvs: Dict[str, AGV], active_agents: List[str],
                       info: Dict[str, Dict] = None) -> Dict[str, int]:
        """Select actions for active agents based on planner_paths from info.

        Parameters
        ----------
        agvs : dict
            Mapping of agent name -> AGV object.
        active_agents : list
            List of active agent names.
        info : dict, optional
            The info dict from the last ``env.step()`` or ``env.reset()``.
            If provided, it is stored via ``update_info`` automatically.
            If not provided, the previously stored info is used.
        """
        if info is not None:
            self._last_info = info

        actions: Dict[str, int] = {}
        for agent_name in active_agents:
            if agent_name not in agvs:
                actions[agent_name] = Action.STAY
                continue

            agv = agvs[agent_name]

            # Read path from info['planner_paths']
            agent_info = self._last_info.get(agent_name, {})
            planner_paths = agent_info.get('planner_paths', {})
            path_abs = planner_paths.get('path_abs', None)
            has_path = planner_paths.get('has_path', False)

            if path_abs is None or not has_path or len(path_abs) < 2:
                actions[agent_name] = Action.STAY
                continue

            # path_abs[0] = current position, path_abs[1] = next target
            cur_x, cur_y = float(path_abs[0, 0]), float(path_abs[0, 1])
            nxt_x, nxt_y = float(path_abs[1, 0]), float(path_abs[1, 1])
            dx = round(nxt_x - cur_x)
            dy = round(nxt_y - cur_y)

            if dx == 1 and dy == 0:
                actions[agent_name] = Action.RIGHT
            elif dx == -1 and dy == 0:
                actions[agent_name] = Action.LEFT
            elif dx == 0 and dy == 1:
                actions[agent_name] = Action.DOWN
            elif dx == 0 and dy == -1:
                actions[agent_name] = Action.UP
            else:
                actions[agent_name] = Action.STAY

        return actions
