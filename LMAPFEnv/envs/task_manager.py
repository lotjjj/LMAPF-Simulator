"""Task generation and assignment for the lifelong warehouse environment.

This module centralizes all task lifecycle logic behind a single
:class:`TaskManager` facade so that the environment no longer scatters task
sampling, completion detection and refilling across several private methods.

Design overview
----------------
``MAEnv_base`` owns one :class:`TaskManager`.  The manager owns:

* one :class:`TaskQueue` per AGV (FIFO with a position-uniqueness invariant),
* a shared :class:`TargetPool` that pre-computes every candidate target cell
  for both target modes once at construction (so ``reset`` never rescans the
  grid), and
* a :class:`TargetSampler` strategy (only :class:`RandomTargetSampler` today)
  used to draw distinct targets.

Class relationships::

    MAEnv_base ──has-a──> TaskManager ──has-a──> TargetPool
                              |        \\-has-a--> TargetSampler (Strategy)
                              \\-has-many--> TaskQueue --has-many--> Task

Key invariants
--------------
* Every ACTIVE task across ALL AGV queues targets a globally distinct cell
  (global task flow).  The manager maintains ``_global_active_positions`` to
  enforce this system-wide uniqueness.
* New tasks are *pushed* by the manager when a task completes; AGVs never pull
  their own tasks (req. 2), which keeps the uniqueness invariant enforceable.
* Only random assignment is supported (req. 3).
* Each queue length is exactly ``num_visible_tasks`` after ``reset`` and after
  every completion; queues are never temporarily extended (req. 5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .entities import AGV, Position, Task, TaskStatus


class TargetMode(Enum):
    """Where task targets may be generated.

    ``SHELF_ONLY`` restricts targets to shelf cells (historical warehouse
    behavior).  ``ANY_PASSABLE`` allows any non-wall cell, i.e. shelves plus
    corridors.
    """

    SHELF_ONLY = "shelf_only"
    ANY_PASSABLE = "any_passable"

    @classmethod
    def from_targets_only_on_shelf(cls, targets_only_on_shelf: bool) -> "TargetMode":
        """Map the legacy ``targets_only_on_shelf`` flag onto a mode."""
        return cls.SHELF_ONLY if targets_only_on_shelf else cls.ANY_PASSABLE


class TaskQueue:
    """Per-AGV FIFO task queue enforcing target-position uniqueness.

    A parallel ``set`` mirrors the deque so membership checks (used to keep
    every active target distinct) stay O(1).
    """

    __slots__ = ("_items", "_positions")

    def __init__(self) -> None:
        self._items: "deque[Task]" = deque()
        self._positions: Set[Position] = set()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def head(self) -> Optional[Task]:
        """Current (front) task, or ``None`` when empty."""
        return self._items[0] if self._items else None

    @property
    def positions(self) -> Set[Position]:
        """Copy of the set of target positions currently queued."""
        return set(self._positions)

    def contains(self, pos: Position) -> bool:
        return pos in self._positions

    def append(self, task: Task) -> None:
        """Append ``task``; raises if its target duplicates a queued one."""
        pos = task.target_pos
        if pos in self._positions:
            raise ValueError(
                f"duplicate target {pos} violates queue uniqueness invariant"
            )
        self._items.append(task)
        self._positions.add(pos)

    def popleft(self) -> Optional[Task]:
        if not self._items:
            return None
        task = self._items.popleft()
        self._positions.discard(task.target_pos)
        return task

    def clear(self) -> None:
        self._items.clear()
        self._positions.clear()

    def snapshot(self) -> List[Task]:
        """Return the queued tasks in order (front first)."""
        return list(self._items)


class TargetPool:
    """Pre-computed candidate target cells for each :class:`TargetMode`.

    Both candidate lists are extracted once from the tile masks at environment
    construction.  ``reset`` then samples straight from these cached lists,
    avoiding a full grid rescan on every episode (req. 6).
    """

    __slots__ = ("_by_mode",)

    def __init__(self, shelf_mask: np.ndarray, passable_mask: np.ndarray) -> None:
        self._by_mode: Dict[TargetMode, List[Position]] = {
            TargetMode.SHELF_ONLY: self._extract(shelf_mask),
            TargetMode.ANY_PASSABLE: self._extract(passable_mask),
        }

    @staticmethod
    def _extract(mask: np.ndarray) -> List[Position]:
        # mask is indexed [y, x]; np.nonzero returns (ys, xs).
        ys, xs = np.nonzero(mask)
        return [Position(int(x), int(y)) for x, y in zip(xs, ys)]

    def candidates(self, mode: TargetMode) -> List[Position]:
        """Return the cached candidate list for ``mode`` (do not mutate)."""
        return self._by_mode[mode]

    def size(self, mode: TargetMode) -> int:
        return len(self._by_mode[mode])


class TargetSampler(ABC):
    """Strategy interface for drawing distinct target positions."""

    @abstractmethod
    def sample_distinct(
        self,
        candidates: Sequence[Position],
        forbidden: Set[Position],
        rng: np.random.Generator,
        need: int,
    ) -> List[Position]:
        """Return up to ``need`` distinct positions not in ``forbidden``.

        The returned positions are mutually distinct and each excluded from
        ``forbidden``.  Fewer than ``need`` may be returned only when the
        candidate pool cannot satisfy the request; callers relax constraints
        and retry in that case.
        """
        raise NotImplementedError


class RandomTargetSampler(TargetSampler):
    """Uniform random sampler.

    Draws a single batch of distinct indices sized so that, by the pigeonhole
    principle, at least ``need`` of them fall outside ``forbidden`` whenever the
    pool is large enough.  This replaces the previous up-to-64-try rejection
    loop with one vectorized draw per fill.
    """

    def sample_distinct(
        self,
        candidates: Sequence[Position],
        forbidden: Set[Position],
        rng: np.random.Generator,
        need: int,
    ) -> List[Position]:
        n = len(candidates)
        if need <= 0 or n == 0:
            return []
        # Drawing (need + |forbidden|) distinct candidates guarantees at least
        # `need` of them are outside `forbidden` (at most |forbidden| overlap).
        draw = min(n, need + len(forbidden))
        if draw >= n:
            indices = rng.permutation(n)
        else:
            indices = rng.choice(n, size=draw, replace=False)
        out: List[Position] = []
        for idx in indices:
            pos = candidates[int(idx)]
            if pos in forbidden:
                continue
            out.append(pos)
            if len(out) == need:
                break
        return out


class TaskManager:
    """Facade coordinating task queues, the target pool and the sampler.

    The manager holds a live reference to the environment's ``agvs`` mapping so
    it can read current positions and write ``agv.target_pos`` directly (the
    push model).  It never scans the grid: candidate targets come from the
    pre-computed :class:`TargetPool`.

    A global task flow (``_global_active_positions``) tracks every active task
    target across ALL agents so that newly generated tasks never duplicate an
    existing active task system-wide.
    """

    def __init__(
        self,
        agvs: Dict[str, AGV],
        pool: TargetPool,
        sampler: TargetSampler,
        num_visible_tasks: int,
        mode: TargetMode,
    ) -> None:
        self._agvs = agvs
        self._pool = pool
        self._sampler = sampler
        self._k = int(num_visible_tasks)
        self._mode = mode
        self._queues: Dict[int, TaskQueue] = {
            agv.id: TaskQueue() for agv in agvs.values()
        }
        # Global task flow: all active task target positions across all agents.
        self._global_active_positions: Set[Position] = set()
        available = pool.size(mode)
        if available < self._k:
            raise ValueError(
                f"num_visible_tasks={self._k} exceeds available target cells "
                f"({available}) for mode {mode.value}; cannot guarantee "
                f"distinct targets per queue"
            )

    # -- lifecycle --------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        """Clear every queue and fill each AGV to ``num_visible_tasks``."""
        for queue in self._queues.values():
            queue.clear()
        self._global_active_positions.clear()
        for agent_name in sorted(self._agvs.keys()):
            agv = self._agvs[agent_name]
            self._fill(agv, rng)
            self._sync_target(agv)

    def process_completions(self, rng: np.random.Generator) -> Set[str]:
        """Detect AGVs sitting on their current target, advance and refill.

        Returns the set of agent names that completed a task this step.  Task
        completion pops the head, then the manager immediately pushes a fresh
        distinct target so the queue length returns to ``num_visible_tasks``.
        """
        completed: Set[str] = set()
        for agent_name, agv in self._agvs.items():
            queue = self._queues[agv.id]
            head = queue.head
            if head is None:
                continue
            if agv.position != head.target_pos:
                continue
            done_task = queue.popleft()
            # Release the completed target from the global task flow.
            if done_task is not None:
                self._global_active_positions.discard(done_task.target_pos)
            self._fill(agv, rng)
            self._sync_target(agv)
            completed.add(agent_name)
        return completed

    # -- read APIs (used by planner input collection / replan checks) -----
    def current_task(self, agv_id: int) -> Optional[Task]:
        return self._queues[agv_id].head

    def task_count(self, agv_id: int) -> int:
        return len(self._queues[agv_id])

    def goal_sequence(self, agv_id: int) -> List[Position]:
        """Active target positions (front first), capped at ``num_visible_tasks``."""
        queue = self._queues.get(agv_id)
        if queue is None:
            return []
        return [
            Position(t.target_pos.x, t.target_pos.y)
            for t in queue.snapshot()[: self._k]
        ]

    def snapshot_all(self) -> Dict[int, List[Task]]:
        """Return a shallow snapshot of every AGV queue (front first)."""
        return {agv_id: q.snapshot() for agv_id, q in self._queues.items()}

    # -- internals --------------------------------------------------------
    def _fill(self, agv: AGV, rng: np.random.Generator) -> None:
        """Top the AGV's queue back up to ``num_visible_tasks`` distinct targets.

        New tasks are drawn from the global task flow: they must not duplicate
        any currently active task across ALL agents (global uniqueness), nor
        the AGV's current cell (soft constraint).
        """
        queue = self._queues[agv.id]
        need = self._k - len(queue)
        if need <= 0:
            return
        candidates = self._pool.candidates(self._mode)
        # Hard constraint: avoid all globally active targets + own queue.
        forbidden = self._global_active_positions | {agv.position}
        sampled = self._sampler.sample_distinct(candidates, forbidden, rng, need)
        if len(sampled) < need:
            # Pool too small to also skip the current cell: relax the soft
            # position constraint but keep global uniqueness.
            forbidden = self._global_active_positions | set(sampled)
            sampled += self._sampler.sample_distinct(
                candidates, forbidden, rng, need - len(sampled)
            )
        for pos in sampled:
            queue.append(Task(pos, TaskStatus.ACTIVE))
            self._global_active_positions.add(pos)

    def _sync_target(self, agv: AGV) -> None:
        """Write the head target back onto the AGV (push model)."""
        head = self._queues[agv.id].head
        agv.target_pos = (
            Position(head.target_pos.x, head.target_pos.y) if head else None
        )
