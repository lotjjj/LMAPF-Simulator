"""Backward-compatible re-export shim.

All planner code has been split into individual modules:

- base.py              : PathPlannerBase, _cpp_astar_nocopy, C++ import guards
- conflict_utils.py    : find_first_conflict, get_position_at, force_wait
- reservation.py       : ReservationTable, sipp_search, _serialize_ct
- planner_policy.py     : PlannerPolicy
- astar_planner.py      : AStarPlanner, EnhancedAStarPlanner
- cbs_planner.py        : CBSNode, CBSPlanner
- pbs_planner.py        : PBSNode, PBSPlanner
- ecbs_planner.py       : _FocalEntry, _ECBSNode, ECBSPlanner
- rhcr_planner.py       : RHCRPlanner
- rhcr_solvers.py       : RHCRCBSPlanner, RHCRPBSPlanner, RHCRECBSPlanner
- factory.py            : create_path_planner

This file re-exports all public names so that existing imports like
``from LMAPFEnv.algorithms.path_planners import CBSPlanner`` continue to work.
"""

# Base
from .base import (
    PathPlannerBase,
    _cpp_astar_nocopy,
    _cpp_sipp,
    _sequential_sipp_attempt,
    _HAS_CXX_ASTAR,
    _HAS_CXX_SIPP,
)

# Conflict utilities
from .conflict_utils import (
    find_first_conflict,
    get_position_at,
    force_wait,
)

# Reservation / SIPP
from .reservation import (
    ReservationTable,
    sipp_search,
    _serialize_ct,
    _INTERVAL_MAX,
)

# Policy
from .planner_policy import PlannerPolicy

# Single-agent planners
from .astar_planner import AStarPlanner, EnhancedAStarPlanner

# Multi-agent planners
from .cbs_planner import CBSNode, CBSPlanner
from .pbs_planner import PBSNode, PBSPlanner
from .ecbs_planner import _FocalEntry, _ECBSNode, ECBSPlanner

# RHCR family
from .rhcr_planner import RHCRPlanner
from .rhcr_solvers import RHCRCBSPlanner, RHCRPBSPlanner, RHCRECBSPlanner

# Factory
from .factory import create_path_planner

__all__ = [
    # Base
    'PathPlannerBase',
    '_cpp_astar_nocopy',
    '_cpp_sipp',
    '_sequential_sipp_attempt',
    # Conflict utilities
    'find_first_conflict',
    'get_position_at',
    'force_wait',
    # Reservation / SIPP
    'ReservationTable',
    'sipp_search',
    '_serialize_ct',
    '_INTERVAL_MAX',
    # Policy
    'PlannerPolicy',
    # Single-agent planners
    'AStarPlanner',
    'EnhancedAStarPlanner',
    # Multi-agent planners
    'CBSNode',
    'CBSPlanner',
    'PBSNode',
    'PBSPlanner',
    '_FocalEntry',
    '_ECBSNode',
    'ECBSPlanner',
    # RHCR family
    'RHCRPlanner',
    'RHCRCBSPlanner',
    'RHCRPBSPlanner',
    'RHCRECBSPlanner',
    # Factory
    'create_path_planner',
    # C++ guards
    '_HAS_CXX_ASTAR',
    '_HAS_CXX_SIPP',
]
