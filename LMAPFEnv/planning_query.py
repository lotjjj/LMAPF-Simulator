"""Public result types for side-effect-free planner queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class PathQueryResult:
    """Result of :meth:`WarehouseEnv.query_paths`.

    The returned coordinates are detached from both the environment and the
    temporary planner, so callers may freely mutate the nested ``paths``
    dictionary and lists without affecting either one.
    """

    planner_type: str
    current_step: int
    success: bool
    paths: Dict[str, List[Coordinate]]
    planning_time_ms: float
    timed_out: bool = False
    failure_reason: Optional[str] = None
    timing_detail: Dict[str, Any] = field(default_factory=dict)

