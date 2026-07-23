from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple, List
from typing import Literal
from collections import OrderedDict
import time
import warnings
import numpy as np
from gymnasium.utils import seeding

from .entities import AGV, Wall, Shelf, Corridor, Action, TaskStatus, Position
from .task_manager import TaskManager, TargetPool, TargetMode, RandomTargetSampler
from ..algorithms.path_planners import create_path_planner, PathPlannerBase
from ..planning_query import PathQueryResult
from ..configBase import (
    LegacyRewardConfig,
    PlannerConfigBase,
    get_default_planner_config,
    get_legacy_reward_config,
)

# ── C++ FastGraph engine (mandatory) ──────────────────────────────────────
from ..fast_graph import FastGraph as _CppFastGraph
from ..fast_graph import bfs_distance_grid as _cpp_bfs_distance_grid
from ..fast_graph import validate_agent_actions as _cpp_validate_actions

@dataclass
class MapConfig:
    shelf_cols: int = 3
    shelf_rows: int = 3
    shelf_width: int = 3
    shelf_height: int = 2
    corridor_width: int = 1
    corridor_out_width: int = 2

PRESET_MAPS: dict[str, MapConfig] = {
    'small': MapConfig(shelf_cols=4, shelf_rows=4, shelf_width=3, shelf_height=2,
                       corridor_width=1, corridor_out_width=2),
    'medium': MapConfig(shelf_cols=8, shelf_rows=8, shelf_width=3, shelf_height=2,
                        corridor_width=1, corridor_out_width=2),
    'large': MapConfig(shelf_cols=16, shelf_rows=16, shelf_width=3, shelf_height=2,
                       corridor_width=1, corridor_out_width=2),
    'long': MapConfig(shelf_cols=2, shelf_rows=2, shelf_width=12, shelf_height=2,
                      corridor_width=1, corridor_out_width=1),
    'test': MapConfig(shelf_cols=1, shelf_rows=1, shelf_width=5, shelf_height=5,
                      corridor_width=1, corridor_out_width=4),
    'custom': MapConfig(shelf_cols=2, shelf_rows=2, shelf_width=12, shelf_height=2,
                        corridor_width=2, corridor_out_width=2)
}

# Backward-compatible aliases: old single-letter names → full-word names
_LEGACY_MAP_ALIASES = {'s': 'small', 'm': 'medium', 'l': 'large'}

class WarehouseEnvBase:
    def __init__(self, num_agvs=6, fov_size=5, render_mode=None, map_size: str = 'medium',
                 max_episode_steps=500,
                 path_planner: Optional[
                     Literal[
                         'CBS',
                         'ECBS',
                         'PBS',
                         'AStar',
                         'EnhancedAStar',
                         'RHCR',
                         'RHCR_CBS',
                         'RHCR_ECBS',
                         'RHCR_PBS',
                     ]
                 ] = None,
                 path_window: int = 5,
                 planner_args: Optional[Dict] = None,
                 num_visible_tasks: Optional[int] = None,
                 kstep_conflict_check: Optional[int] = None,
                 targets_only_on_shelf: bool = True,
                 padding_path_enable: bool = True,
                 ):

        # Backward compat: kstep_conflict_check is deprecated alias for path_window
        if kstep_conflict_check is not None:
            warnings.warn(
                "kstep_conflict_check is deprecated, use path_window instead",
                DeprecationWarning, stacklevel=2)
            path_window = kstep_conflict_check

        # Resolve legacy single-letter alias
        map_size = _LEGACY_MAP_ALIASES.get(str(map_size).strip().lower(), map_size)
        if map_size not in PRESET_MAPS:
            raise ValueError(f"map_size must be one of {list(PRESET_MAPS.keys())}, got '{map_size}'")

        self.targets_only_on_shelf = bool(targets_only_on_shelf)
        # padding_path_enable: when a planner path is shorter than
        # path_window + 1, controls whether info['planner_paths']['path_abs']
        # is padded (by repeating the last position) up to the fixed length.
        # When disabled, the raw (shorter) path is exposed as-is.
        self.padding_path_enable = bool(padding_path_enable)
        self.num_agvs = num_agvs
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.path_planner_type = path_planner
        self.path_window = int(path_window)
        self.planner_args = dict(planner_args) if planner_args else {}
        self.obs_path_planner_type = self.planner_args.get("obs_planner_type", None)
        raw_obs_planner_args = self.planner_args.get("obs_planner_args", {})
        self.obs_planner_args = dict(raw_obs_planner_args) if isinstance(raw_obs_planner_args, dict) else {}
        # num_visible_tasks: task queue depth visible to the planner.
        # Default 2 (current task + next task), matching typical warehouse semantics.
        self.num_visible_tasks = int(max(1, num_visible_tasks)) if num_visible_tasks is not None else 2

        self.map_config = PRESET_MAPS[map_size]

        self.shelf_cols = self.map_config.shelf_cols
        self.shelf_rows = self.map_config.shelf_rows
        self.shelf_width = self.map_config.shelf_width
        self.shelf_height = self.map_config.shelf_height
        self.corridor_width = self.map_config.corridor_width
        self.corridor_out_width = self.map_config.corridor_out_width

        if fov_size % 2 == 0:
            raise ValueError(f"fov_size must be odd, right now it is {fov_size}")
        self.fov_size = fov_size
        self.fov_radius = fov_size // 2

        self.width, self.height = self._calculate_map_size()

        self.agvs = {}
        self.possible_agents = [f"agv_{i}" for i in range(num_agvs)]
        for i in range(num_agvs):
            agv = AGV(i, (0, 0), fov_size=self.fov_size, path_window=self.path_window)
            self.agvs[f"agv_{i}"] = agv

        self.grid_map = self._create_initial_map()
        self._build_tile_codes()
        
        self.path_planner: Optional[PathPlannerBase] = None
        self.obs_path_planner: Optional[PathPlannerBase] = None
        self.path_planner = self._build_planner(self.path_planner_type, self.planner_args)
        self.obs_path_planner = self._build_planner(self.obs_path_planner_type, self.obs_planner_args)

        if self.path_planner and self.render_mode:
            self.path_planner.enable_timing()
        if self.obs_path_planner and self.render_mode:
            self.obs_path_planner.enable_timing()

        self._agv_positions = {}
        self._position_to_agv = {}

        self._target_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._agv_grid = np.full((self.height, self.width), -1, dtype=np.int32)
        self._agent_at = np.full(self.height * self.width, -1, dtype=np.int32)

        self._target_pool = TargetPool(self._shelf_mask, self._passable_mask)

        r = self.fov_radius
        pad_h = self.height + 2 * r
        pad_w = self.width + 2 * r
        self._padded_tile_codes = np.full((pad_h, pad_w), 1, dtype=np.uint8)
        self._padded_tile_codes[r:r + self.height, r:r + self.width] = self._tile_codes
        self._padded_agv_grid = np.full((pad_h, pad_w), -1, dtype=np.int32)
        self._padded_agv_grid[r:r + self.height, r:r + self.width] = self._agv_grid
        self._padded_target_grid = np.zeros((pad_h, pad_w), dtype=np.float32)
        self._padded_target_grid[r:r + self.height, r:r + self.width] = self._target_grid

        self._fov_buffers: Dict[str, np.ndarray] = {}
        self._path_info_bufs: Dict[str, np.ndarray] = {}
        _target_len = self.path_window + 1
        for agent in self.possible_agents:
            self._fov_buffers[agent] = np.zeros((5, self.fov_size, self.fov_size), dtype=np.float32)
            self._path_info_bufs[agent] = np.zeros((_target_len, 2), dtype=np.float32)

        self._episode_count = 0
        self._current_step = 0

        self.np_random = None
        self.seed_val = None
        self._seed()

        self.legacy_reward_config = LegacyRewardConfig(**vars(get_legacy_reward_config()))

        # Mirror the active legacy reward config on the environment instance.
        self.task_completion_reward = float(self.legacy_reward_config.task_completion_reward)
        self.each_step_reward = float(self.legacy_reward_config.each_step_reward)
        self.conflict_penalty = float(self.legacy_reward_config.conflict_penalty)
        self.progress_shaping_weight = float(self.legacy_reward_config.progress_shaping_weight)
        self.invalid_action_penalty = float(self.legacy_reward_config.invalid_action_penalty)

        self._distance_cache_size = 512
        self._distance_cache = OrderedDict()

        self._planner_disabled = False
        self._planner_disabled_reason = ""
        self._planner_last_plan_time_ms = 0.0
        self._planner_last_timed_out = False
        self._planner_timeout_cooldown = 0

        self._agent_dones = {}
        self._agent_rewards = {}
        self._agent_terminations = {}
        self._agent_truncations = {}
        self._conflicted_agents = None
        
        # Instance-level task manager (not global singleton)
        self.task_manager = TaskManager(
            agvs=self.agvs,
            pool=self._target_pool,
            sampler=RandomTargetSampler(),
            num_visible_tasks=self.num_visible_tasks,
            mode=TargetMode.from_targets_only_on_shelf(self.targets_only_on_shelf),
        )

        self._initialize_agvs()
        self._render_prev_positions = self._snapshot_positions() if self.render_mode else {}
        self._render_current_positions = self._snapshot_positions() if self.render_mode else {}

        self.action_spaces = {agent: agv.action_space for agent, agv in self.agvs.items()}
        
        self.observation_spaces = {}
        for agent, agv in self.agvs.items():
            self.observation_spaces[agent] = agv.observation_space()

        self.agents = self.possible_agents[:]

        self._fov_tensors_cache = {}
        self._self_states_cache = {}
        self._info_cache = {}
        self._planner_paths_info_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._planner_goal_sequences_snapshot: Dict[int, Tuple[Position, ...]] = {}

        fov_num_channels = 5
        fov_shape = (fov_num_channels, self.fov_size, self.fov_size)

        for agent in self.possible_agents:
            self_states = {
                "position": np.zeros(2, dtype=np.float32),
                "fov_density": np.zeros(1, dtype=np.float32),
                "target_rel": np.zeros(2, dtype=np.float32),
                "target_visible": np.zeros(1, dtype=np.float32),
                "target_dist_norm": np.zeros(1, dtype=np.float32),
            }

            self._self_states_cache[agent] = self_states

        for agent in self.possible_agents:
            self._info_cache[agent] = {
                "action_mask": np.ones(5, dtype=np.int8),
                "conflicted": False,
                "invalid_action": False,
                "task_completed": False,
                "progress_target_pos": None,
                "progress_distance_prev": None,
                "progress_distance_now": None,
                "planner_skipped": False,
                "planner_timed_out": False,
                "planner_time_ms": 0.0,
                "planner_timing_detail": {},
                "obs_planner_time_ms": 0.0,
                "obs_planner_timing_detail": {},
                "planner_disabled": False,
                "planner_disable_reason": "",
                "planner_replanned": False,
                "planner_partial_replan": False,
                "planner_replanned_agents": [],
            }

        self._profile_data = [0] * 9

        self._graph_capacity = max(2 * self.num_agvs + 8, 1024)
        self._graph_engine = _CppFastGraph(self._graph_capacity)

        self._held_endpoints: Dict[Position, str] = {}  # pos → agent_name
        self._step_distance_cache: Dict[Tuple[int, int, int, int], Optional[int]] = {}

    def _build_planner(self, planner_type, planner_args: Optional[Dict] = None) -> Optional[PathPlannerBase]:
        if not planner_type:
            return None
        cfg = get_default_planner_config(planner_type)
        cfg_overrides: Dict = {}
        if "shelf_penalty" not in (planner_args or {}):
            cfg_overrides["shelf_penalty"] = 4.0

        cfg = cfg.with_overrides(cfg_overrides).with_overrides(planner_args or {})
        planner_kwargs = cfg.to_planner_kwargs()
        planner = create_path_planner(planner_type, self.grid_map, **planner_kwargs)
        if planner is not None:
            planner.set_grid_data(self._passable_mask, self._shelf_mask)
            planner.set_planning_window(self.path_window)
        return planner

    def query_paths(
        self,
        planner_type: str,
        planner_args: Optional[Dict] = None,
        *,
        timeout: Optional[float] = None,
        use_current_constraints: bool = True,
    ) -> PathQueryResult:
        """Run one isolated planning query against the current environment state.

        A fresh planner and detached AGV snapshots are used for every call.
        Consequently this method does not change the environment, the main
        path planner, or the optional observation planner.  It is intended to
        be called synchronously between ``step`` calls.

        Parameters
        ----------
        planner_type:
            Registered planner name, such as ``"AStar"`` or ``"RHCR"``.
        planner_args:
            Overrides accepted by that planner's config. Unknown keys raise
            ``ValueError`` so misspelled experimental settings are not ignored.
        timeout:
            Optional per-query wall-clock limit in seconds. The planner's own
            smaller limit, when present, still applies.
        use_current_constraints:
            For a query planner with ``k_robust > 0``, copy initial constraints
            derived from the main planner's already-executed path prefix.

        Returns
        -------
        PathQueryResult
            Detached paths keyed by public agent name plus query diagnostics.

        Raises
        ------
        RuntimeError
            If the environment has not been reset yet.
        ValueError, TypeError
            If the planner name or query arguments are invalid.

        Notes
        -----
        This method is side-effect-free but not thread-safe with concurrent
        ``step`` or ``reset`` calls; those could change the sampled state while
        the temporary planner is running.
        """
        if self._episode_count <= 0:
            raise RuntimeError("query_paths() requires reset() to be called first")
        if not isinstance(planner_type, str) or not planner_type:
            raise ValueError("planner_type must be a non-empty planner name")
        if planner_args is not None and not isinstance(planner_args, dict):
            raise TypeError("planner_args must be a dict or None")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number of seconds or None")
        if not isinstance(use_current_constraints, bool):
            raise TypeError("use_current_constraints must be a bool")

        cfg = get_default_planner_config(planner_type)
        query_args = dict(planner_args or {})
        allowed_args = {item.name for item in fields(cfg)} - {"planner_type"}
        unknown_args = sorted(set(query_args) - allowed_args)
        if unknown_args:
            raise ValueError(
                f"Unknown {planner_type} planner_args: {', '.join(unknown_args)}"
            )

        planner = self._build_planner(planner_type, query_args)
        if planner is None:  # Defensive; validated planner names always build one.
            raise ValueError(f"Unable to create planner: {planner_type}")

        # Do not expose the environment-owned numpy arrays to query planners.
        planner.set_grid_data(self._passable_mask.copy(), self._shelf_mask.copy())
        planner.enable_timing()

        source_agvs, target_positions, goal_sequences = self._collect_planner_inputs()
        query_agvs: Dict[int, AGV] = {}
        id_to_name: Dict[int, str] = {}
        for agent_name, source in self.agvs.items():
            if source.id not in source_agvs:
                continue
            snapshot = AGV(
                source.id,
                source.position.to_tuple(),
                fov_size=source.fov_size,
                path_window=source.path_window,
            )
            snapshot.target_pos = source.target_pos
            snapshot.req_action = source.req_action
            snapshot.status = source.status
            query_agvs[source.id] = snapshot
            id_to_name[source.id] = agent_name

        planner.set_goal_sequences(goal_sequences)
        query_k_robust = int(getattr(planner, "k_robust", 0))
        if (
            use_current_constraints
            and query_k_robust > 0
            and hasattr(planner, "set_initial_constraints")
        ):
            planner.set_initial_constraints(
                self._build_initial_constraints(query_k_robust)
            )

        deadline = None if timeout is None else time.time() + float(timeout)
        started = time.perf_counter()
        raw_paths = planner.plan_multi_agent(
            query_agvs, target_positions, deadline=deadline
        ) if target_positions else {}
        planning_time_ms = (time.perf_counter() - started) * 1000.0

        expected_ids = set(target_positions)
        success = bool(planner.is_last_plan_successful()) and all(
            raw_paths.get(agent_id) for agent_id in expected_ids
        )

        planner_budget = getattr(planner, "MAX_PLANNING_TIME", None)
        effective_budget = float(timeout) if timeout is not None else None
        if planner_budget is not None:
            planner_budget = float(planner_budget)
            effective_budget = (
                planner_budget
                if effective_budget is None
                else min(effective_budget, planner_budget)
            )
        timed_out = bool(
            not success
            and effective_budget is not None
            and planning_time_ms >= effective_budget * 900.0
        )
        failure_reason = None if success else ("timed_out" if timed_out else "no_paths")

        detached_paths = {
            id_to_name[agent_id]: [(int(pos.x), int(pos.y)) for pos in path]
            for agent_id, path in raw_paths.items()
            if agent_id in id_to_name
        }
        return PathQueryResult(
            planner_type=planner_type,
            current_step=int(self._current_step),
            success=success,
            paths=detached_paths,
            planning_time_ms=float(planning_time_ms),
            timed_out=timed_out,
            failure_reason=failure_reason,
            timing_detail=dict(planner.last_timing or {}),
        )


    def _get_observation_planner(self) -> Optional[PathPlannerBase]:
        return self.obs_path_planner if self.obs_path_planner is not None else self.path_planner

    @staticmethod
    def _planner_uses_periodic_replan(planner_type: Optional[str]) -> bool:
        return planner_type in ('RHCR', 'RHCR_CBS', 'RHCR_ECBS', 'RHCR_PBS')

    @staticmethod
    def _planner_supports_partial_replan(planner_type: Optional[str]) -> bool:
        return planner_type in ('AStar', 'EnhancedAStar')

    def _planner_replan_interval(self, planner_type: Optional[str], planner: Optional[PathPlannerBase]) -> int:
        if planner is None or not self._planner_uses_periodic_replan(planner_type):
            return 0
        return int(max(1, getattr(planner, "horizon", 1)))

    def _predict_goal_count(self, goals: List[Position], budget: int) -> int:
        """Predict how many goals the planner needs to fill the path window.

        Uses a budget-based greedy prediction:
        - The first goal is always included.
        - Subsequent goals are included if the remaining budget can cover
          the Manhattan-distance edge from the previous goal.

        This is a *prediction* (estimating how many goals the planner
        should see), not a *truncation* (discarding excess goals).

        Args:
            goals: Ordered list of target positions.
            budget: Available distance budget (typically ``path_window``).

        Returns:
            Number of goals to pass to the planner (1 <= result <= len(goals)).
        """
        if len(goals) <= 1:
            return len(goals)
        count = 1
        remaining = budget
        prev = goals[0]
        for i in range(1, len(goals)):
            edge = abs(goals[i].x - prev.x) + abs(goals[i].y - prev.y)
            if remaining >= edge:
                count += 1
                remaining -= edge
                prev = goals[i]
            else:
                break
        return count

    def _collect_planner_inputs(self) -> Tuple[Dict[int, AGV], Dict[int, Position], Dict[int, List[Position]]]:
        agv_dict = {}
        target_positions = {}
        goal_sequences: Dict[int, List[Position]] = {}

        pw = self.path_window
        for agent_name, agv in self.agvs.items():
            if agent_name in self.agents:
                current = self.task_manager.current_task(agv.id)
                if current and current.status == TaskStatus.ACTIVE:
                    agv_dict[agv.id] = agv
                    target_positions[agv.id] = current.target_pos
                    # Collect full task goal sequence for multi-goal planning
                    # (goal_id as state dimension in low-level A*/SIPP)
                    goals = self.task_manager.goal_sequence(agv.id)

                    # Predict how many goals the planner needs.
                    # Budget = pw * 4: generous enough for multi-goal
                    # chain planning.  The actual number of goals is
                    # already capped by num_visible_tasks (default 2),
                    # so this budget won't overload the planner.
                    goal_count = self._predict_goal_count(goals, pw * 4)
                    goals = goals[:goal_count]

                    goal_sequences[agv.id] = goals
        return agv_dict, target_positions, goal_sequences

    @staticmethod
    def _prune_planner_paths(planner: Optional[PathPlannerBase], planned_ids: set[int]) -> None:
        if planner is None:
            return
        for agv_id in planner.get_all_path_ids():
            if agv_id not in planned_ids:
                planner.clear_path(agv_id)

    def _plan_observation_paths(self) -> None:
        # Only plan when a dedicated observation planner exists.
        # When obs_path_planner is None, the main path_planner's paths
        # are already available via _get_cached_planner_paths_info().
        # Calling plan_multi_agent on the main planner would overwrite
        # its planned_paths and reset _path_heads, causing path desync.
        if self.obs_path_planner is None:
            # Record zero timing for consistency
            for agent in self.agents:
                self._info_cache[agent]["obs_planner_time_ms"] = 0.0
                self._info_cache[agent]["obs_planner_timing_detail"] = {}
            return
        planner = self.obs_path_planner
        agv_dict, target_positions, goal_sequences = self._collect_planner_inputs()

        obs_plan_time_ms = 0.0
        if target_positions:
            t0 = time.perf_counter()
            if goal_sequences and hasattr(planner, 'set_goal_sequences'):
                planner.set_goal_sequences(goal_sequences)
            planner.plan_multi_agent(agv_dict, target_positions)
            t1 = time.perf_counter()
            obs_plan_time_ms = float((t1 - t0) * 1000.0)

        obs_planner_timing_detail = dict(planner.last_timing) if planner.last_timing else {}

        for agent in self.agents:
            self._info_cache[agent]["obs_planner_time_ms"] = float(obs_plan_time_ms)
            self._info_cache[agent]["obs_planner_timing_detail"] = obs_planner_timing_detail

        self._prune_planner_paths(planner, set(agv_dict.keys()))

    def _calculate_map_size(self):
        if self.shelf_cols > 0:
            total_shelf_width = self.shelf_cols * self.shelf_width + (self.shelf_cols - 1) * self.corridor_width
        else:
            total_shelf_width = 0

        if self.shelf_rows > 0:
            total_shelf_height = self.shelf_rows * self.shelf_height + (self.shelf_rows - 1) * self.corridor_width
        else:
            total_shelf_height = 0

        width = 2 + 2 * self.corridor_out_width + total_shelf_width
        height = 2 + 2 * self.corridor_out_width + total_shelf_height

        return width, height

    def _create_initial_map(self):
        grid_map = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    grid = Wall(x, y)
                else:
                    if self._is_shelf_area(x, y):
                        grid = Shelf(x, y)
                    else:
                        grid = Corridor(x, y)
                row.append(grid)
            grid_map.append(row)
        return grid_map

    def _is_shelf_area(self, x, y):
        if self.shelf_cols <= 0 or self.shelf_rows <= 0:
            return False

        outer_margin = self.corridor_out_width

        total_shelf_width = self.shelf_cols * self.shelf_width + (self.shelf_cols - 1) * self.corridor_width
        total_shelf_height = self.shelf_rows * self.shelf_height + (self.shelf_rows - 1) * self.corridor_width

        if total_shelf_width > self.width - 2 - 2 * outer_margin or \
                total_shelf_height > self.height - 2 - 2 * outer_margin:
            return False

        shelf_area_width = self.width - 2 - 2 * outer_margin
        shelf_area_height = self.height - 2 - 2 * outer_margin

        start_x = outer_margin + 1 + (shelf_area_width - total_shelf_width) // 2
        start_y = outer_margin + 1 + (shelf_area_height - total_shelf_height) // 2

        for shelf_row in range(self.shelf_rows):
            for shelf_col in range(self.shelf_cols):
                shelf_start_x = start_x + shelf_col * (self.shelf_width + self.corridor_width)
                shelf_start_y = start_y + shelf_row * (self.shelf_height + self.corridor_width)

                if shelf_start_x <= x < shelf_start_x + self.shelf_width and \
                        shelf_start_y <= y < shelf_start_y + self.shelf_height:
                    return True

        return False

    def _build_tile_codes(self):
        """Precompute integer tile codes for vectorized FOV lookup.
        0=corridor, 1=wall, 2=shelf."""
        codes = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid_map[y][x]
                if isinstance(tile, Wall):
                    codes[y, x] = 1
                elif isinstance(tile, Shelf):
                    codes[y, x] = 2
        self._tile_codes = codes
        self._passable_mask = (codes != 1)  # corridor(0) or shelf(2)
        self._shelf_mask = (codes == 2)

    def _seed(self, seed=None):
        self.np_random, self.seed_val = seeding.np_random(seed)
        return [self.seed_val]


    def _initialize_agvs(self):
        for agent_name in sorted(self.agvs.keys()):
            agv = self.agvs[agent_name]
            if 0 <= agv._x < self.width and 0 <= agv._y < self.height:
                self.grid_map[agv._y][agv._x].remove_agv(agv)

        self._agv_positions.clear()
        self._position_to_agv.clear()
        self._agent_at.fill(-1)

        corridor_positions = []
        fallback_positions = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid_map[y][x].occupiable:
                    fallback_positions.append((x, y))
                    if not self._shelf_mask[y, x]:
                        corridor_positions.append((x, y))

        passable_positions = (
            corridor_positions if len(corridor_positions) >= self.num_agvs else fallback_positions
        )
        if len(passable_positions) < self.num_agvs:
            raise ValueError(
                f"num_agvs={self.num_agvs} exceeds occupiable start cells={len(passable_positions)}"
            )

        selected_positions = self.np_random.choice(
            len(passable_positions),
            size=self.num_agvs,
            replace=False
        )

        sorted_selection = sorted(enumerate(selected_positions), key=lambda x: x[1])
        for i, pos_idx in sorted_selection:
            x, y = passable_positions[pos_idx]

            agv = self.agvs[f"agv_{i}"]
            agv.reset(x, y)

            self.grid_map[y][x].add_agv(agv)

            pos = Position(x, y)
            self._agv_positions[agv.id] = pos
            self._position_to_agv[(x, y)] = agv
            self._agent_at[y * self.width + x] = agv.id

    def action_mask(self, agent):
        if agent not in self.agvs or agent not in self.agents:
            return np.zeros(5, dtype=np.int8)

        agv = self.agvs[agent]
        x, y = agv.x, agv.y

        # B2: Reuse pre-allocated buffer
        mask = self._info_cache[agent]["action_mask"]
        mask[:] = 1
        p = self._passable_mask
        s = self._shelf_mask

        if y - 1 < 0:
            mask[Action.UP] = 0
        elif not p[y - 1, x]:
            mask[Action.UP] = 0
        elif s[y, x] and s[y - 1, x]:
            mask[Action.UP] = 0

        if y + 1 >= self.height:
            mask[Action.DOWN] = 0
        elif not p[y + 1, x]:
            mask[Action.DOWN] = 0
        elif s[y, x] and s[y + 1, x]:
            mask[Action.DOWN] = 0

        if x - 1 < 0:
            mask[Action.LEFT] = 0
        elif not p[y, x - 1]:
            mask[Action.LEFT] = 0
        elif s[y, x] and s[y, x - 1]:
            mask[Action.LEFT] = 0

        if x + 1 >= self.width:
            mask[Action.RIGHT] = 0
        elif not p[y, x + 1]:
            mask[Action.RIGHT] = 0
        elif s[y, x] and s[y, x + 1]:
            mask[Action.RIGHT] = 0

        return mask

    def teleport_agv(self, agent_name, x, y):
        if agent_name not in self.agvs:
            return

        agv = self.agvs[agent_name]
        old_x, old_y = agv._x, agv._y
        r = self.fov_radius
        w = self.width

        self.grid_map[old_y][old_x].remove_agv(agv)

        old_key = (old_x, old_y)
        if old_key in self._position_to_agv:
            del self._position_to_agv[old_key]

        agv.set_position(x, y)

        self.grid_map[y][x].add_agv(agv)

        new_pos = Position(x, y)
        self._agv_positions[agv.id] = new_pos
        self._position_to_agv[(x, y)] = agv
        self._agv_grid[old_y, old_x] = -1
        self._padded_agv_grid[old_y + r, old_x + r] = -1
        self._agv_grid[y, x] = agv.id
        self._padded_agv_grid[y + r, x + r] = agv.id
        self._agent_at[old_y * w + old_x] = -1
        self._agent_at[y * w + x] = agv.id
        self._render_prev_positions[agent_name] = new_pos
        self._render_current_positions[agent_name] = new_pos

    def reset(self, seed=None, options=None):
        self._seed(seed=seed)

        self._episode_count += 1
        self._current_step = 0
        self._step_distance_cache.clear()
        self._planner_disabled = False
        self._planner_disabled_reason = ""
        self._planner_last_plan_time_ms = 0.0
        self._planner_last_timed_out = False
        self._planner_timeout_cooldown = 0
        self._planner_goal_sequences_snapshot.clear()
        self._held_endpoints.clear()
        self._initialize_agvs()
        self._render_prev_positions = self._snapshot_positions() if self.render_mode else {}
        self._render_current_positions = self._snapshot_positions() if self.render_mode else {}
        self.agents = self.possible_agents[:]
        self._agent_terminations = {agent: False for agent in self.possible_agents}
        self._agent_truncations = {agent: False for agent in self.possible_agents}
        self.task_manager.reset(self.np_random)
        
        self._plan_paths()
        self._plan_observation_paths()
        if self.path_planner is not None or self.obs_path_planner is not None:
            self._refresh_planner_paths_info_cache()

        for agent in self.agents:
            self._info_cache[agent]["conflicted"] = False
            self._info_cache[agent]["invalid_action"] = False
            self._info_cache[agent]["task_completed"] = False

        self._update_target_grid()
        self._update_agv_grid()

        observations = self.get_obs_batch()
        infos = {agent: self.get_info(agent) for agent in self.agents}

        return observations, infos

    def _update_target_grid(self):
        """Rebuild target position grid from current AGV targets (+ padded sync)."""
        self._target_grid.fill(0.0)
        r = self.fov_radius
        for agv in self.agvs.values():
            if agv.target_pos is not None:
                self._target_grid[agv.target_pos.y, agv.target_pos.x] = 1.0
        # Sync padded target grid
        self._padded_target_grid.fill(0.0)
        self._padded_target_grid[r:r + self.height, r:r + self.width] = self._target_grid

    def _update_agv_grid(self):
        """Rebuild AGV presence grid from current AGV positions (+ padded sync)."""
        self._agv_grid.fill(-1)
        r = self.fov_radius
        w = self.width
        self._agent_at.fill(-1)
        for agv in self.agvs.values():
            self._agv_grid[agv._y, agv._x] = agv.id
            self._agent_at[agv._y * w + agv._x] = agv.id
        # Sync padded agv grid
        self._padded_agv_grid[r:r + self.height, r:r + self.width] = self._agv_grid

    def _compute_fov(self, agent):
        """Compute FOV tensor for agent."""
        agv = self.agvs[agent]
        r = self.fov_radius
        fov = self._fov_buffers[agent]
        fov_size = self.fov_size

        # Direct slice from padded arrays — no clamp, no np.pad needed
        # Padded arrays have fov_radius border, so index (agv.y, agv.x) in
        # original space maps to (agv.y, agv.x) in padded space and the
        # slice [agv.y : agv.y+2r+1, agv.x : agv.x+2r+1] is always in bounds.
        tile = self._padded_tile_codes[agv.y:agv.y + fov_size, agv.x:agv.x + fov_size]
        agv_presence = self._padded_agv_grid[agv.y:agv.y + fov_size, agv.x:agv.x + fov_size]
        # Fill FOV channels in-place using pre-allocated buffer
        fov[0] = (tile == 0)       # corridor
        fov[1] = (tile == 1)       # wall / oob
        fov[2] = (tile == 2)       # shelf
        # other AGVs: present AND not self
        fov[3] = (agv_presence >= 0) & (agv_presence != agv.id)
        # Only expose this agent's current goal.  Other agents' goals and
        # future entries in the ego task queue are not part of its FOV.
        fov[4].fill(0.0)
        if agv.target_pos is not None:
            dx = int(agv.target_pos.x - agv.x)
            dy = int(agv.target_pos.y - agv.y)
            if abs(dx) <= r and abs(dy) <= r:
                fov[4, r + dy, r + dx] = 1.0

        return fov

    def _build_planner_paths_info(self) -> Dict[str, Dict[str, Any]]:
        """Build planner path info for all agents.

        Ensures path_abs[0] always equals the agent's current position,
        even when the agent is delayed (behind the path head) due to
        conflict resolution.  When the agent's position doesn't match
        path[head], the agent's actual position is prepended so that
        PlannerPolicy can compute the correct next action.
        """
        planner = self._get_observation_planner()
        out: Dict[str, Dict[str, Any]] = {}
        alive_agents = set(self.agents)
        pw = self.path_window
        target_len = pw + 1
        for agent_name, agv in self.agvs.items():
            if planner is None:
                out[agent_name] = {
                    "path_abs": np.zeros((0, 2), dtype=np.float32),
                    "alive": bool(agent_name in alive_agents),
                    "has_path": False,
                }
                continue
            else:
                path = planner.get_path(int(agv.id)) or []
                if not path:
                    out[agent_name] = {
                        "path_abs": np.zeros((0, 2), dtype=np.float32),
                        "alive": bool(agent_name in alive_agents),
                        "has_path": False,
                    }
                    continue
                head = int(planner.get_path_head(int(agv.id)))
                head = min(max(0, head), len(path) - 1)
                remaining = list(path[head:])
                seq = remaining if remaining else [agv.position]
                # CRITICAL: Ensure path_abs[0] == current agent position.
                # When the runtime conflict resolver delays an agent, keep the
                # public path grounded at the actual current position so
                # PlannerPolicy can compute the next action from info alone.
                if seq[0] != agv.position:
                    seq = [agv.position] + seq
            # Truncate to at most target_len (path_window + 1).
            if len(seq) > target_len:
                seq = seq[:target_len]
            # Pad short paths to exactly target_len by repeating the last
            # position, unless padding is disabled (expose raw path as-is).
            elif self.padding_path_enable and len(seq) < target_len and seq:
                seq = seq + [seq[-1]] * (target_len - len(seq))
            buf = self._path_info_bufs[agent_name]
            for idx, pos in enumerate(seq):
                buf[idx, 0] = float(pos.x)
                buf[idx, 1] = float(pos.y)
            # When not padded, expose only the filled portion of the buffer.
            path_abs = buf if len(seq) == target_len else buf[:len(seq)]
            out[agent_name] = {
                "path_abs": path_abs,
                "alive": bool(agent_name in alive_agents),
                "has_path": True,
            }
        return out

    def _refresh_planner_paths_info_cache(self) -> None:
        self._planner_paths_info_cache = self._build_planner_paths_info()

    def _get_cached_planner_paths_info(self) -> Dict[str, Dict[str, Any]]:
        if not isinstance(self._planner_paths_info_cache, dict):
            self._refresh_planner_paths_info_cache()
        return self._planner_paths_info_cache or {}

    def get_obs(self, agent):
        agv = self.agvs[agent]

        self_states = self._self_states_cache[agent]
        self_states["position"][0] = agv.x / (self.width - 1) if self.width > 1 else 0.0
        self_states["position"][1] = agv.y / (self.height - 1) if self.height > 1 else 0.0

        if agv.target_pos is not None:
            dx_raw = agv.target_pos.x - agv.x
            dy_raw = agv.target_pos.y - agv.y
            self_states["target_rel"][0] = dx_raw / (self.width - 1) if self.width > 1 else 0.0
            self_states["target_rel"][1] = dy_raw / (self.height - 1) if self.height > 1 else 0.0
        else:
            self_states["target_rel"][0] = 0.0
            self_states["target_rel"][1] = 0.0
            dx_raw = 0
            dy_raw = 0

        fov = self._compute_fov(agent)
        self._fov_tensors_cache[agent] = fov
        area = float(self.fov_size * self.fov_size)
        self_states["fov_density"][0] = float(fov[3].sum() / area) if area > 0 else 0.0
        self_states["target_visible"][0] = 1.0 if (agv.target_pos is not None and abs(dx_raw) <= self.fov_radius and abs(dy_raw) <= self.fov_radius) else 0.0
        denom = float(np.hypot(max(0, self.width - 1), max(0, self.height - 1)))
        self_states["target_dist_norm"][0] = float(np.hypot(dx_raw, dy_raw) / denom) if denom > 0 else 0.0

        return {
            "self_states": {
                "position": self_states["position"].copy(),
                "fov_density": self_states["fov_density"].copy(),
                "target_rel": self_states["target_rel"].copy(),
                "target_visible": self_states["target_visible"].copy(),
                "target_dist_norm": self_states["target_dist_norm"].copy(),
            },
            "fov": self._fov_tensors_cache[agent].copy(),
        }

    def get_obs_batch(self) -> Dict:
        """B1: Build observations for all agents at once using pre-allocated buffers.
        
        Returns dict mapping agent name to observation dict.
        Uses in-place updates to cached buffers and np.copyto for efficiency.
        """
        observations = {}
        w_minus_1 = float(self.width - 1) if self.width > 1 else 1.0
        h_minus_1 = float(self.height - 1) if self.height > 1 else 1.0
        area = float(self.fov_size * self.fov_size)
        fov_radius = self.fov_radius
        denom = float(np.hypot(max(0, self.width - 1), max(0, self.height - 1)))
        
        for agent in self.agents:
            agv = self.agvs[agent]
            self_states = self._self_states_cache[agent]
            
            # Position (in-place update)
            self_states["position"][0] = agv._x / w_minus_1
            self_states["position"][1] = agv._y / h_minus_1
            
            # Target relative position
            if agv.target_pos is not None:
                dx_raw = agv.target_pos.x - agv._x
                dy_raw = agv.target_pos.y - agv._y
                self_states["target_rel"][0] = dx_raw / w_minus_1
                self_states["target_rel"][1] = dy_raw / h_minus_1
            else:
                dx_raw = 0
                dy_raw = 0
                self_states["target_rel"][0] = 0.0
                self_states["target_rel"][1] = 0.0
            
            # FOV (compute using optimized padded array method)
            fov = self._compute_fov(agent)
            self._fov_tensors_cache[agent] = fov
            
            # Derived states (in-place update)
            self_states["fov_density"][0] = float(fov[3].sum() / area) if area > 0 else 0.0
            self_states["target_visible"][0] = 1.0 if (agv.target_pos is not None and abs(dx_raw) <= fov_radius and abs(dy_raw) <= fov_radius) else 0.0
            self_states["target_dist_norm"][0] = float(np.hypot(dx_raw, dy_raw) / denom) if denom > 0 else 0.0
            
            # Return observation with np.copyto for buffer reuse
            obs = {
                "self_states": {
                    "position": np.empty_like(self_states["position"]),
                    "fov_density": np.empty_like(self_states["fov_density"]),
                    "target_rel": np.empty_like(self_states["target_rel"]),
                    "target_visible": np.empty_like(self_states["target_visible"]),
                    "target_dist_norm": np.empty_like(self_states["target_dist_norm"]),
                },
                "fov": np.empty_like(fov),
            }
            np.copyto(obs["self_states"]["position"], self_states["position"])
            np.copyto(obs["self_states"]["fov_density"], self_states["fov_density"])
            np.copyto(obs["self_states"]["target_rel"], self_states["target_rel"])
            np.copyto(obs["self_states"]["target_visible"], self_states["target_visible"])
            np.copyto(obs["self_states"]["target_dist_norm"], self_states["target_dist_norm"])
            np.copyto(obs["fov"], fov)
            
            observations[agent] = obs
        
        return observations

    def get_info(self, agent):
        self._info_cache[agent]["action_mask"] = self.action_mask(agent)
        info = self._info_cache[agent]
        # Return cached planner_meta if it exists, otherwise build it
        if "_planner_meta_cached" not in info:
            info["_planner_meta_cached"] = {
                "skipped": bool(info.get("planner_skipped", False)),
                "timed_out": bool(info.get("planner_timed_out", False)),
                "time_ms": float(info.get("planner_time_ms", 0.0)),
                "timing_detail": info.get("planner_timing_detail", {}),
                "obs_time_ms": float(info.get("obs_planner_time_ms", 0.0)),
                "obs_timing_detail": info.get("obs_planner_timing_detail", {}),
                "disabled": bool(info.get("planner_disabled", False)),
                "disable_reason": str(info.get("planner_disable_reason", "")),
                "replanned": bool(info.get("planner_replanned", False)),
                "partial_replan": bool(info.get("planner_partial_replan", False)),
                "replanned_agents": list(info.get("planner_replanned_agents", [])),
            }
        # Issue 5: copy path_abs to avoid exposing the pre-allocated buffer
        paths_info = self._get_cached_planner_paths_info().get(agent, {})
        if paths_info and 'path_abs' in paths_info:
            paths_info = dict(paths_info)  # shallow copy
            paths_info['path_abs'] = paths_info['path_abs'].copy()
        return {
            "action_mask": info["action_mask"].copy(),
            "conflicted": bool(info.get("conflicted", False)),
            "invalid_action": bool(info.get("invalid_action", False)),
            "task_completed": bool(info.get("task_completed", False)),
            "progress_target_pos": info.get("progress_target_pos"),
            "progress_distance_prev": info.get("progress_distance_prev"),
            "progress_distance_now": info.get("progress_distance_now"),
            "act_val_time_ms": float(info.get("act_val_time_ms", 0.0)),
            "planner_meta": info["_planner_meta_cached"],
            "planner_paths": paths_info,
        }

    def step(self, actions):
        _p = self._profile_data if self.render_mode else None
        if _p is not None:
            _p[7] = time.perf_counter_ns()  # step_start
        prev_render_positions = self._snapshot_positions() if self.render_mode else {}

        self._step_distance_cache.clear()
        self._begin_step_planner_meta()

        # ── Phase 1: Validate and resolve actions ──────────────────────
        feasible_actions, invalid_actions, old_positions = self._validate_actions(actions)

        # ── Phase 2: Conflict resolution (position-based digraph) ─────
        _t_act_val = time.perf_counter()
        committed_agents = self._resolve_movement_conflicts(feasible_actions)
        if _p is not None:
            _p[0] = time.perf_counter_ns()  # end digraph

        # ── Phase 3: Execute movements ────────────────────────────────
        self._execute_movements(committed_agents, feasible_actions)
        if _p is not None:
            _p[1] = time.perf_counter_ns()  # end act_exe
        _act_val_time_ms = (time.perf_counter() - _t_act_val) * 1000.0

        # ── Phase 4: Task completion & replan decision ────────────────
        self._current_step += 1
        if self._current_step >= self.max_episode_steps:
            for agent in self.agents:
                self._agent_truncations[agent] = True

        reward_targets, task_completed_flags, completed_agents = self._process_tasks()
        if _p is not None:
            _p[2] = time.perf_counter_ns()  # end task_ck

        # ── Phase 5: Planner path management ──────────────────────────
        replan_flag, obs_replan_flag, replan_agents = self._evaluate_replan_triggers(completed_agents)
        if _p is not None:
            _p[3] = time.perf_counter_ns()  # end deviance

        initial_constraints = self._prepare_replan_data(replan_flag)

        if replan_flag:
            self._plan_paths(initial_constraints=initial_constraints, agent_names=replan_agents)
            self._clear_planner_meta_cache()
        if obs_replan_flag:
            self._plan_observation_paths()
            self._clear_planner_meta_cache()
        # Always refresh planner paths info so that info['planner_paths']
        # reflects the latest paths and current agent positions.
        if self.path_planner is not None or self.obs_path_planner is not None:
            self._refresh_planner_paths_info_cache()
        if _p is not None:
            _p[4] = time.perf_counter_ns()  # end planning

        # ── Phase 6: Rewards, info, observations ──────────────────────
        rewards = self._compute_rewards(
            old_positions, invalid_actions, reward_targets, task_completed_flags
        )
        self._update_info_cache(
            reward_targets, old_positions, invalid_actions,
            completed_agents, _act_val_time_ms,
        )

        if self.render_mode:
            self._render_prev_positions = prev_render_positions
            self._render_current_positions = self._snapshot_positions()

        if _p is not None:
            _p[5] = time.perf_counter_ns()  # end reward_info

        observations = self.get_obs_batch()
        infos = {agent: self.get_info(agent) for agent in self.agents}
        if _p is not None:
            _p[6] = time.perf_counter_ns()  # end obs_info
            _p[8] = time.perf_counter_ns()  # step_end

        terminations = {agent: self._agent_terminations[agent] for agent in self.agents}
        truncations = {agent: self._agent_truncations[agent] for agent in self.agents}
        self.agents = [
            agent for agent in self.agents
            if not (self._agent_terminations[agent] or self._agent_truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos

    # ── Step sub-methods ─────────────────────────────────────────────────

    def _validate_actions(self, actions) -> Tuple[Dict, Dict, Dict]:
        """Phase 1: Validate requested actions and determine feasible moves (C1 optimized)."""
        feasible_actions = {}
        invalid_actions = {}
        old_positions = {}
        
        # Pack agent data into numpy arrays for C++ validation
        agent_list = list(self.agents)
        n = len(agent_list)
        if n == 0:
            return feasible_actions, invalid_actions, old_positions
        
        agv_x = np.empty(n, dtype=np.int32)
        agv_y = np.empty(n, dtype=np.int32)
        agv_actions = np.empty(n, dtype=np.int32)
        
        # Fill arrays and store old positions
        for i, agent in enumerate(agent_list):
            agv = self.agvs[agent]
            agv_x[i] = agv._x
            agv_y[i] = agv._y
            old_positions[agent] = agv.position
            
            try:
                req_action = Action(actions[agent])
            except Exception:
                req_action = Action.STAY
            agv.req_action = req_action
            agv_actions[i] = int(req_action)
        
        # Call C++ batch validation
        feasible_flags = _cpp_validate_actions(
            self.width, self.height,
            agv_x, agv_y, agv_actions,
            self._passable_mask, self._shelf_mask
        )
        
        # Process results
        for i, agent in enumerate(agent_list):
            agv = self.agvs[agent]
            is_feasible = feasible_flags[i]
            
            if is_feasible:
                feasible_actions[agent] = agv.req_action
                invalid_actions[agent] = False
            else:
                feasible_actions[agent] = Action.STAY
                if agv.req_action != Action.STAY:
                    invalid_actions[agent] = True
                else:
                    invalid_actions[agent] = False
                agv.req_action = Action.STAY
        
        return feasible_actions, invalid_actions, old_positions

    def _resolve_movement_conflicts(self, feasible_actions: Dict) -> set:
        """Phase 2: Build conflict graph and determine which agents can move.

        Uses a directed graph where nodes are grid positions.
        Resolution: 2-cycle=blocked, 4-cycle+=rotation, DAG=longest chain.
        """
        committed_agents = set()
        ge = self._graph_engine

        pos_to_idx: Dict[Tuple[int, int], int] = {}
        idx_to_pos: List[Tuple[int, int]] = []
        pos_agents: Dict[Tuple[int, int], List[str]] = {}

        for agent in self.agents:
            agv = self.agvs[agent]
            pos = (agv.x, agv.y)
            if pos not in pos_to_idx:
                pos_to_idx[pos] = len(idx_to_pos)
                idx_to_pos.append(pos)
            pos_agents.setdefault(pos, []).append(agent)

        target_of: Dict[str, Tuple[int, int]] = {}
        for agent in self.agents:
            action = feasible_actions[agent]
            if action != Action.STAY:
                agv = self.agvs[agent]
                tx, ty = self._get_target_position(agv, action)
                if 0 <= tx < self.width and 0 <= ty < self.height:
                    tpos = (tx, ty)
                    target_of[agent] = tpos
                    if tpos not in pos_to_idx:
                        pos_to_idx[tpos] = len(idx_to_pos)
                        idx_to_pos.append(tpos)
                else:
                    target_of[agent] = (agv.x, agv.y)

        n_pos = len(idx_to_pos)
        if n_pos > self._graph_capacity:
            self._graph_capacity = max(n_pos, 2 * len(self.agents) + 8)
            self._graph_engine = _CppFastGraph(self._graph_capacity)
            ge = self._graph_engine
        ge.reset(n_pos)

        for agent in self.agents:
            agv = self.agvs[agent]
            current = (agv.x, agv.y)
            if feasible_actions[agent] != Action.STAY:
                target = target_of.get(agent, current)
                if current != target:
                    ge.add_edge(pos_to_idx[current], pos_to_idx[target])
            else:
                ge.add_edge(pos_to_idx[current], pos_to_idx[current])

        cpp_components = ge.components()
        for comp in cpp_components:
            cycle = ge.find_cycle(comp)
            if cycle:
                if len(cycle) == 2:
                    continue  # 2-cycle = edge conflict: block all
                for u, v in cycle:
                    pos = idx_to_pos[u]
                    if pos in pos_agents:
                        for agent in pos_agents[pos]:
                            committed_agents.add(self.agvs[agent])
            else:
                path = ge.dag_longest_path(comp)
                for idx in path:
                    pos = idx_to_pos[idx]
                    if pos in pos_agents:
                        for agent in pos_agents[pos]:
                            if self.agvs[agent] not in committed_agents:
                                committed_agents.add(self.agvs[agent])

        self._conflicted_agents = set()
        return committed_agents

    def _execute_movements(self, committed_agents: set, feasible_actions: Dict):
        """Phase 3: Execute resolved movements, mark conflicted agents."""
        for agent in self.agents:
            agv = self.agvs[agent]
            if agv in committed_agents:
                self._execute_action(agent, feasible_actions[agent])
            else:
                self._execute_action(agent, Action.STAY)
                self._conflicted_agents.add(agent)

    def _process_tasks(self) -> Tuple[Dict, Dict, set]:
        """Phase 4: Check task completion and prepare reward data."""
        reward_targets = {}
        task_completed_flags = {}
        for agent in self.agents:
            agv = self.agvs[agent]
            task = self.task_manager.current_task(agv.id)
            reward_targets[agent] = task.target_pos if task is not None else agv.target_pos

        completed_agents = self.task_manager.process_completions(self.np_random)
        if completed_agents:
            self._update_target_grid()
        for agent in self.agents:
            task_completed_flags[agent] = agent in completed_agents
        return reward_targets, task_completed_flags, completed_agents

    def _evaluate_replan_triggers(self, completed_agents: set) -> Tuple[bool, bool, Optional[set[str]]]:
        """Phase 5a: Determine if replanning is needed for main/obs planners.

        Replanning triggers:
        1. Agent deviated from its planned path (true deviation, not delay).
        2. Periodic horizon expiry (for windowed/RHCR planners).
        3. Path length insufficient for path_window with pending goals.

        Task completion alone does NOT trigger replan because all planners
        support multi-goal planning — existing paths through remaining
        goals stay valid.

        Path head tracking:
        After movement execution, advance a path head only when the agent
        actually reaches the next planned timestep.  A runtime conflict delay
        leaves the head unchanged so PlannerPolicy retries the same segment.
        If the user takes an action that leaves the agent off the next planned
        timestep without a runtime conflict, that is a true deviation.
        """
        deviated_agents = self._sync_planner_path_heads(self.path_planner)
        obs_deviated_agents = self._sync_planner_path_heads(self.obs_path_planner)

        replan_flag = bool(deviated_agents)
        obs_replan_flag = bool(obs_deviated_agents)
        partial_replan_agents: Optional[set[str]] = None
        if replan_flag and self._planner_supports_partial_replan(self.path_planner_type):
            partial_replan_agents = set(deviated_agents)

        main_interval = self._planner_replan_interval(self.path_planner_type, self.path_planner)
        periodic_due = bool(main_interval and self._current_step % main_interval == 0)
        if periodic_due:
            replan_flag = True
            partial_replan_agents = None

        obs_interval = self._planner_replan_interval(self.obs_path_planner_type, self.obs_path_planner)
        if obs_interval:
            obs_replan_flag = obs_replan_flag or (self._current_step % obs_interval == 0)

        # Path sufficiency check
        insufficient_agents = self._agents_needing_replan_for_path_sufficiency()
        if insufficient_agents:
            replan_flag = True
            if self._planner_supports_partial_replan(self.path_planner_type) and not periodic_due:
                if partial_replan_agents is None:
                    partial_replan_agents = set()
                partial_replan_agents.update(insufficient_agents)
            else:
                partial_replan_agents = None

        return replan_flag, obs_replan_flag, partial_replan_agents

    def _sync_planner_path_heads(self, planner: Optional[PathPlannerBase]) -> set[str]:
        """Advance path heads by observed motion and return true deviations."""
        if planner is None:
            return set()

        deviated_agents: set[str] = set()
        conflicted = self._conflicted_agents or set()
        for agent in self.agents:
            if self._agent_terminations.get(agent, False):
                continue
            agv = self.agvs[agent]
            path = planner.get_path(agv.id)
            if not path:
                continue

            head = int(planner.get_path_head(agv.id))
            if head < 0:
                head = 0
                planner.set_path_head(agv.id, 0)
            if head >= len(path):
                planner.set_path_head(agv.id, len(path) - 1)
                continue

            current = agv.position
            if head + 1 < len(path) and current == path[head + 1]:
                planner.advance_step(agv.id)
                continue

            if current == path[head]:
                if head + 1 < len(path) and agent not in conflicted:
                    deviated_agents.add(agent)
                continue

            deviated_agents.add(agent)
        return deviated_agents

    def _agents_needing_replan_for_path_sufficiency(self) -> set[str]:
        """Return alive agents needing replanning due to insufficient path.

        Replan is needed when remaining path length < path_window + 1
        and the current task queue differs from, or is not covered by, the
        sequence used in the last planning round.  If the planner already
        planned through every known queued goal and the complete path is still
        shorter than the window, replanning with the same inputs cannot help.
        """
        if not self.path_planner:
            return set()
        pw = self.path_window
        insufficient_agents: set[str] = set()
        for agent in self.agents:
            if self._agent_terminations.get(agent, False):
                continue
            agv = self.agvs[agent]
            path = self.path_planner.get_path(agv.id)
            head = self.path_planner.get_path_head(agv.id)
            remaining_len = len(path) - head if path else 0
            if remaining_len < pw + 1:
                current_seq = tuple(self.task_manager.goal_sequence(agv.id))
                if not current_seq:
                    continue
                if not path:
                    insufficient_agents.add(agent)
                    continue

                planned_seq = self._planner_goal_sequences_snapshot.get(agv.id, tuple())
                if current_seq != planned_seq:
                    insufficient_agents.add(agent)
                    continue

                final_goal = planned_seq[-1] if planned_seq else current_seq[-1]
                if path[-1] != final_goal:
                    insufficient_agents.add(agent)
        return insufficient_agents

    def _should_replan_for_path_sufficiency(self) -> bool:
        """Return whether any alive agent needs replanning due to insufficient path."""
        return bool(self._agents_needing_replan_for_path_sufficiency())

    def _prepare_replan_data(self, replan_flag: bool) -> Optional[Dict]:
        """Phase 5b: Build constraints and predictive tasks for replanning."""
        initial_constraints = None
        if replan_flag and self._planner_uses_periodic_replan(self.path_planner_type) and self.path_planner is not None:
            k_robust = self.planner_args.get('k_robust', 0)
            if k_robust > 0:
                initial_constraints = self._build_initial_constraints(k_robust)
        return initial_constraints

    def _clear_planner_meta_cache(self):
        """Clear cached planner metadata so it gets rebuilt."""
        for agent in self.agents:
            self._info_cache[agent].pop("_planner_meta_cached", None)

    def _begin_step_planner_meta(self):
        """Reset per-step planner timing while preserving disabled state."""
        for agent in self.agents:
            info = self._info_cache[agent]
            info["planner_skipped"] = self.path_planner is not None
            info["planner_timed_out"] = False
            info["planner_time_ms"] = 0.0
            info["planner_timing_detail"] = {}
            info["obs_planner_time_ms"] = 0.0
            info["obs_planner_timing_detail"] = {}
            info["planner_disabled"] = bool(self._planner_disabled)
            info["planner_disable_reason"] = self._planner_disabled_reason
            info["planner_replanned"] = False
            info["planner_partial_replan"] = False
            info["planner_replanned_agents"] = []
            info.pop("_planner_meta_cached", None)

    def _compute_rewards(self, old_positions, invalid_actions, reward_targets, task_completed_flags) -> Dict:
        """Phase 6a: Compute rewards for all agents."""
        rewards = {}
        for agent in self.agents:
            rewards[agent] = self._calculate_reward(
                agent,
                prev_pos=old_positions.get(agent),
                invalid_action=invalid_actions.get(agent, False),
                target_pos=reward_targets.get(agent),
                task_completed=task_completed_flags.get(agent, False),
            )
        return rewards

    def _update_info_cache(self, reward_targets, old_positions, invalid_actions,
                           completed_agents, act_val_time_ms):
        """Phase 6b: Update per-agent info cache with step results."""
        for agent in self.agents:
            reward_target = reward_targets.get(agent)
            d_prev = self._distance_to_target(
                old_positions.get(agent), reward_target
            ) if old_positions.get(agent) is not None and reward_target is not None else None
            d_now = self._distance_to_target(
                self.agvs[agent].position, reward_target
            ) if reward_target is not None else None
            self._info_cache[agent]["conflicted"] = agent in self._conflicted_agents
            self._info_cache[agent]["invalid_action"] = invalid_actions.get(agent, False)
            self._info_cache[agent]["task_completed"] = agent in completed_agents
            self._info_cache[agent]["progress_target_pos"] = (
                None if reward_target is None else (int(reward_target.x), int(reward_target.y))
            )
            self._info_cache[agent]["progress_distance_prev"] = None if d_prev is None else int(d_prev)
            self._info_cache[agent]["progress_distance_now"] = None if d_now is None else int(d_now)
            self._info_cache[agent]["act_val_time_ms"] = float(act_val_time_ms)

    def _get_target_position(self, agv, action):
        if action == Action.UP:
            return agv.x, agv.y - 1
        elif action == Action.DOWN:
            return agv.x, agv.y + 1
        elif action == Action.LEFT:
            return agv.x - 1, agv.y
        elif action == Action.RIGHT:
            return agv.x + 1, agv.y
        else:
            return agv.x, agv.y

    def _execute_action(self, agent, action):
        agv = self.agvs[agent]
        old_x, old_y = agv._x, agv._y
        old_grid = self.grid_map[old_y][old_x]
        r = self.fov_radius
        w = self.width

        old_grid.remove_agv(agv)
        old_key = (old_x, old_y)
        if old_key in self._position_to_agv:
            del self._position_to_agv[old_key]

        if action != Action.STAY:
            new_x, new_y = self._get_target_position(agv, action)

            if (0 <= new_x < self.width and 0 <= new_y < self.height):
                can_move = True
                if self._shelf_mask[old_y, old_x] and self._shelf_mask[new_y, new_x]:
                    can_move = False

                if self._passable_mask[new_y, new_x] and can_move:
                    agv.set_position(new_x, new_y)
                    self.grid_map[new_y][new_x].add_agv(agv)
                    new_pos = Position(new_x, new_y)
                    self._agv_positions[agv.id] = new_pos
                    self._position_to_agv[(new_x, new_y)] = agv
                    self._agv_grid[old_y, old_x] = -1
                    self._padded_agv_grid[old_y + r, old_x + r] = -1
                    self._agv_grid[new_y, new_x] = agv.id
                    self._padded_agv_grid[new_y + r, new_x + r] = agv.id
                    self._agent_at[old_y * w + old_x] = -1
                    self._agent_at[new_y * w + new_x] = agv.id
                else:
                    self.grid_map[old_y][old_x].add_agv(agv)
                    self._position_to_agv[old_key] = agv
                    self._agent_at[old_y * w + old_x] = agv.id
            else:
                self.grid_map[old_y][old_x].add_agv(agv)
                self._position_to_agv[old_key] = agv
                self._agent_at[old_y * w + old_x] = agv.id
        else:
            self.grid_map[old_y][old_x].add_agv(agv)
            self._position_to_agv[old_key] = agv
            self._agent_at[old_y * w + old_x] = agv.id



    def _get_distance_grid(self, target_pos: Position) -> np.ndarray:
        """BFS distance grid from target_pos."""
        key = (target_pos.x, target_pos.y)
        grid = self._distance_cache.get(key)
        if grid is not None:
            self._distance_cache.move_to_end(key)
            return grid

        grid = _cpp_bfs_distance_grid(
            self.width, self.height,
            self._passable_mask, self._shelf_mask,
            target_pos.x, target_pos.y,
        )

        self._distance_cache[key] = grid
        self._distance_cache.move_to_end(key)
        if len(self._distance_cache) > self._distance_cache_size:
            self._distance_cache.popitem(last=False)

        return grid

    def _distance_to_target(self, pos: Position, target_pos: Position) -> Optional[int]:
        """Distance from pos to target_pos."""
        cache_key = (pos.x, pos.y, target_pos.x, target_pos.y)
        cached = self._step_distance_cache.get(cache_key)
        if cached is not None:
            return cached
        dist_grid = self._get_distance_grid(target_pos)
        d = int(dist_grid[pos.y, pos.x])
        result = None if d < 0 else d
        self._step_distance_cache[cache_key] = result
        return result

    def _calculate_reward_legacy(
        self,
        agent,
        prev_pos: Optional[Position] = None,
        invalid_action: bool = False,
        target_pos: Optional[Position] = None,
        task_completed: bool = False,
    ) -> float:
        agv = self.agvs[agent]
        cfg = self.legacy_reward_config
        r = float(cfg.each_step_reward)

        if invalid_action:
            r += float(cfg.invalid_action_penalty)

        if self._conflicted_agents is not None and agent in self._conflicted_agents:
            r += float(cfg.conflict_penalty)

        if prev_pos is not None and target_pos is not None:
            d_prev = self._distance_to_target(prev_pos, target_pos)
            d_now = self._distance_to_target(agv.position, target_pos)
            if d_prev is not None and d_now is not None:
                progress = float(np.clip(d_prev - d_now, -1, 1))
                r += float(cfg.progress_shaping_weight) * progress

        if task_completed:
            r += float(cfg.task_completion_reward)
        return r

    def _calculate_reward(
        self,
        agent,
        prev_pos: Optional[Position] = None,
        invalid_action: bool = False,
        target_pos: Optional[Position] = None,
        task_completed: bool = False,
    ) -> float:
        return self._calculate_reward_legacy(
            agent,
            prev_pos=prev_pos,
            invalid_action=invalid_action,
            target_pos=target_pos,
            task_completed=task_completed,
        )

    def _snapshot_positions(self) -> Dict[str, Position]:
        return {
            agent_name: Position(self.agvs[agent_name]._x, self.agvs[agent_name]._y)
            for agent_name in self.possible_agents
        }

    def _build_initial_constraints(self, k_robust: int = 1) -> Dict[int, List[Tuple[Position, int]]]:
        """Extract constraints from already-executed path portions.

        Following the official RHCR design (BasicSystem::update_initial_constraints):
        collect the last k_robust positions from the executed portion of each
        agent's path and convert them into vertex constraints for the new
        planning round. This prevents new paths from conflicting with movements
        that have already been committed.

        Each past position ``path[head-dt]`` was occupied at relative time
        ``-dt`` in the old plan.  In the new planning round (time 0 = current
        position), k-robust expansion covers ``[-dt-k, -dt+k]``.  The overlap
        with ``[0, ∞)`` is ``[0, k-dt]``.

        We pass ``t=0`` and let the SIPP reservation table's k-robust
        expansion handle the interval ``[0, k+1)``.  For the non-SIPP path,
        the constraint at time 0 prevents the agent from immediately
        returning to a recently-occupied position.
        """
        constraints: Dict[int, List[Tuple[Position, int]]] = {}
        if self.path_planner is None:
            return constraints

        all_paths = self.path_planner.get_paths()
        all_heads = self.path_planner.get_path_heads()
        for agv_id, path in all_paths.items():
            head = all_heads.get(agv_id, 0)
            agent_constraints = []
            # The agent's current position (last executed position in the path).
            # If head >= len(path), the agent has finished its path and is at
            # the last position.
            current_idx = min(head, len(path) - 1) if path else 0
            current_pos = path[current_idx] if path else None
            # Collect positions from head-1 back to head-k_robust
            for dt in range(1, k_robust + 1):
                t_idx = head - dt
                if 0 <= t_idx < len(path):
                    pos = path[t_idx]
                    # Skip if the agent was at the same position as its
                    # current position (it was waiting).  Adding a
                    # constraint at the agent's own current position would
                    # block its own search start, causing spurious planning
                    # failures in congested scenarios where agents are stuck.
                    if pos == current_pos:
                        continue
                    # Position was occupied dt steps ago (relative time -dt).
                    # Pass t=0; SIPP's k-robust expansion creates interval [0, k+1),
                    # which covers the needed range [0, k-dt] for all dt.
                    agent_constraints.append((pos, 0))
            if agent_constraints:
                constraints[agv_id] = agent_constraints
        return constraints

    
    def _plan_paths(
        self,
        initial_constraints: Optional[Dict[int, List[Tuple[Position, int]]]] = None,
        agent_names: Optional[set[str]] = None,
    ):
        """Plan paths for alive agents."""
        if not self.path_planner:
            return

        agv_dict, target_positions, goal_sequences = self._collect_planner_inputs()
        partial_replan = agent_names is not None
        requested_ids: Optional[set[int]] = None
        if partial_replan:
            requested_ids = {
                self.agvs[name].id for name in agent_names
                if name in self.agvs and name in self.agents
            }
            agv_dict = {aid: agv for aid, agv in agv_dict.items() if aid in requested_ids}
            target_positions = {aid: pos for aid, pos in target_positions.items() if aid in agv_dict}
            goal_sequences = {aid: seq for aid, seq in goal_sequences.items() if aid in agv_dict}

        timed_out = False
        plan_time_ms = 0.0
        budget_s = self.planner_args.get("max_planning_time", None)
        if budget_s is None:
            budget_s = getattr(self.path_planner, "MAX_PLANNING_TIME", None)
        budget_s = None if budget_s is None else float(budget_s)

        previous_paths = self.path_planner.get_paths()
        previous_heads = self.path_planner.get_path_heads()
        previous_goal_snapshot = dict(self._planner_goal_sequences_snapshot)

        # Compute deadline (Issue 2): pass absolute deadline to planner
        deadline = None
        if budget_s is not None:
            deadline = time.time() + budget_s

        planner_failed = False
        no_paths = False
        if target_positions:
            t0 = time.perf_counter()
            # Pass ordered task goals to planners with multi-goal low-level search.
            if goal_sequences and hasattr(self.path_planner, 'set_goal_sequences'):
                self.path_planner.set_goal_sequences(goal_sequences)
            # Pass initial constraints to RHCR-family planners.
            if initial_constraints and hasattr(self.path_planner, 'set_initial_constraints'):
                self.path_planner.set_initial_constraints(initial_constraints)
            self.path_planner.replan(agv_dict, target_positions, deadline=deadline)
            t1 = time.perf_counter()
            plan_time_ms = float((t1 - t0) * 1000.0)

            # Check planning success via the new mechanism (Issue 1)
            if not self.path_planner.is_last_plan_successful():
                planner_failed = True
                # Only label as timeout if actual time is close to budget;
                # otherwise the planner simply could not find a feasible solution.
                if budget_s is not None and plan_time_ms >= budget_s * 900.0:
                    timed_out = True

            if partial_replan:
                new_paths = self.path_planner.get_paths()
                new_heads = self.path_planner.get_path_heads()
                self.path_planner.set_paths(previous_paths)
                self.path_planner.set_path_heads(previous_heads)
                for aid in agv_dict:
                    if aid in new_paths:
                        self.path_planner.set_path(aid, new_paths[aid])
                        self.path_planner.set_path_head(aid, new_heads.get(aid, 0))
                    else:
                        self.path_planner.clear_path(aid)
                for aid, seq in goal_sequences.items():
                    self._planner_goal_sequences_snapshot[aid] = tuple(seq)
            else:
                self._planner_goal_sequences_snapshot = {
                    aid: tuple(seq) for aid, seq in goal_sequences.items()
                }
            if not planner_failed:
                no_paths = bool(agv_dict) and all(
                    not self.path_planner.get_path(aid)
                    for aid in agv_dict.keys()
                )
                planner_failed = no_paths
        else:
            if partial_replan and requested_ids is not None:
                for aid in requested_ids:
                    self.path_planner.clear_path(aid)
                    self._planner_goal_sequences_snapshot.pop(aid, None)
            else:
                self._planner_goal_sequences_snapshot = {}

        planner_timing_detail = dict(self.path_planner.last_timing) if self.path_planner and self.path_planner.last_timing else {}

        using_previous_plan = False
        if planner_failed and self._current_step > 0 and previous_paths:
            # RHCR is a rolling-horizon planner.  A failed replan should not
            # invalidate the already committed executable suffix; keep agents
            # moving and try to replan again on a later trigger.
            self.path_planner.set_paths(previous_paths)
            self.path_planner.set_path_heads(previous_heads)
            self._planner_goal_sequences_snapshot = previous_goal_snapshot
            planner_failed = False
            using_previous_plan = True

        self._planner_last_plan_time_ms = float(plan_time_ms)
        self._planner_last_timed_out = bool(timed_out)
        self._planner_disabled = planner_failed
        if timed_out:
            self._planner_disabled_reason = "timed_out"
        elif no_paths:
            self._planner_disabled_reason = "no_paths"
        elif using_previous_plan:
            self._planner_disabled_reason = "replan_failed_using_previous_plan"
        else:
            self._planner_disabled_reason = ""

        # When planner fails: clear paths and print warning
        if planner_failed or using_previous_plan:
            if timed_out and budget_s is not None:
                detail = f"exceeded {budget_s:.1f}s budget, took {plan_time_ms:.0f}ms"
            elif timed_out:
                detail = f"timed out, took {plan_time_ms:.0f}ms"
            else:
                detail = "planner returned no paths"
            if using_previous_plan:
                print(f"Warning: Planner replan failed; using previous paths ({detail})")
            else:
                print(f"Warning: Planner cannot find paths! ({detail})")
                # Initial planning has no safe previous suffix to fall back to.
                if partial_replan and requested_ids is not None:
                    for aid in requested_ids:
                        self.path_planner.clear_path(aid)
                else:
                    self.path_planner.clear_all_paths()

        for agent in self.agents:
            self._info_cache[agent]["planner_skipped"] = False
            self._info_cache[agent]["planner_timed_out"] = bool(timed_out)
            self._info_cache[agent]["planner_time_ms"] = float(plan_time_ms)
            self._info_cache[agent]["planner_timing_detail"] = planner_timing_detail
            self._info_cache[agent]["planner_disabled"] = planner_failed
            self._info_cache[agent]["planner_disable_reason"] = self._planner_disabled_reason
            self._info_cache[agent].pop("_planner_meta_cached", None)

        if partial_replan:
            if requested_ids is not None:
                for aid in requested_ids - set(agv_dict.keys()):
                    self.path_planner.clear_path(aid)
        else:
            self._prune_planner_paths(self.path_planner, set(agv_dict.keys()))

        replanned_agent_names = (
            sorted(agent_names)
            if partial_replan and agent_names is not None
            else sorted(self.agents)
        )
        planned_any = bool(target_positions)
        for agent in self.agents:
            self._info_cache[agent]["planner_replanned"] = planned_any
            self._info_cache[agent]["planner_partial_replan"] = bool(partial_replan)
            self._info_cache[agent]["planner_replanned_agents"] = replanned_agent_names
            self._info_cache[agent].pop("_planner_meta_cached", None)

    def close(self):
        pass
