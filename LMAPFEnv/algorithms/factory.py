"""Factory function for creating path planners by type string.

This module centralizes planner instantiation so that the environment
and CLI tools can create planners without importing every class.
"""

from typing import List, Literal

from .base import PathPlannerBase
from .astar_planner import AStarPlanner, EnhancedAStarPlanner
from .cbs_planner import CBSPlanner
from .pbs_planner import PBSPlanner
from .ecbs_planner import ECBSPlanner
from .rhcr_planner import RHCRPlanner
from .rhcr_solvers import RHCRCBSPlanner, RHCRPBSPlanner, RHCRECBSPlanner


def create_path_planner(
    planner_type: Literal['CBS', 'ECBS', 'PBS', 'AStar', 'EnhancedAStar',
                          'RHCR', 'RHCR_CBS', 'RHCR_ECBS', 'RHCR_PBS'],
    grid_map: List[List],
    **kwargs,
) -> PathPlannerBase:
    """Create path planner based on type.

    Parameters
    ----------
    planner_type : str
        One of 'CBS', 'ECBS', 'PBS', 'AStar', 'EnhancedAStar',
        'RHCR', 'RHCR_CBS', 'RHCR_ECBS', 'RHCR_PBS'.
    grid_map : list of lists
        The grid map for the planner.
    **kwargs
        Additional keyword arguments.  Constructor-specific kwargs
        (e.g. ``w`` for ECBS, ``planning_window`` for RHCR) are passed
        to the planner constructor.  Common settings like
        ``shelf_penalty``, ``max_planning_time``, ``max_low_level_steps``,
        ``conflict_horizon``, ``max_cbs_nodes``, and ``max_pbs_nodes``
        are applied as post-construction attribute assignments so they
        work uniformly across all planner types.

    Returns
    -------
    PathPlannerBase
        The configured planner instance.
    """
    # ── Pop global settings (applied to all planners after construction) ──
    shelf_penalty = kwargs.pop("shelf_penalty", None)
    max_planning_time = kwargs.pop("max_planning_time", None)

    # ── Pop common multi-agent settings (applied as attributes post-construction) ──
    max_low_level_steps = kwargs.pop("max_low_level_steps", None)
    conflict_horizon = kwargs.pop("conflict_horizon", None)
    max_cbs_nodes = kwargs.pop("max_cbs_nodes", None)
    max_pbs_nodes = kwargs.pop("max_pbs_nodes", None)

    # ── For planners that take max_low_level_steps as constructor param, re-inject ──
    _ctor_max_steps_types = ('RHCR', 'RHCR_CBS', 'RHCR_ECBS', 'RHCR_PBS')
    if max_low_level_steps is not None and planner_type in _ctor_max_steps_types:
        kwargs["max_low_level_steps"] = int(max_low_level_steps)

    # ── Create planner with only constructor-specific kwargs ──────────────
    if planner_type == 'AStar':
        planner = AStarPlanner(grid_map)
    elif planner_type == 'EnhancedAStar':
        visible_agv_penalty = kwargs.pop("visible_agv_penalty", 5.0)
        planner = EnhancedAStarPlanner(grid_map, visible_agv_penalty=visible_agv_penalty)
    elif planner_type == 'CBS':
        planner = CBSPlanner(grid_map)
    elif planner_type == 'ECBS':
        w = kwargs.pop("w", 1.5)
        planner = ECBSPlanner(grid_map, w=w)
    elif planner_type == 'PBS':
        planner = PBSPlanner(grid_map)
    elif planner_type == 'RHCR':
        planner = RHCRPlanner(grid_map, **kwargs)
    elif planner_type == 'RHCR_CBS':
        planner = RHCRCBSPlanner(grid_map, **kwargs)
    elif planner_type == 'RHCR_ECBS':
        planner = RHCRECBSPlanner(grid_map, **kwargs)
    elif planner_type == 'RHCR_PBS':
        planner = RHCRPBSPlanner(grid_map, **kwargs)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    # ── Apply common settings as post-construction attributes ─────────────
    if max_cbs_nodes is not None and hasattr(planner, "MAX_CBS_NODES"):
        planner.MAX_CBS_NODES = int(max_cbs_nodes)
    if max_pbs_nodes is not None and hasattr(planner, "MAX_PBS_NODES"):
        planner.MAX_PBS_NODES = int(max_pbs_nodes)
    if max_low_level_steps is not None and hasattr(planner, "MAX_LOW_LEVEL_STEPS"):
        planner.MAX_LOW_LEVEL_STEPS = int(max_low_level_steps)
    if conflict_horizon is not None and hasattr(planner, "conflict_horizon"):
        planner.conflict_horizon = int(conflict_horizon)
    if shelf_penalty is not None:
        planner.SHELF_PENALTY = float(shelf_penalty)
    if max_planning_time is not None and hasattr(planner, "MAX_PLANNING_TIME"):
        planner.MAX_PLANNING_TIME = float(max_planning_time)

    return planner
