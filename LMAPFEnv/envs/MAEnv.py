from typing import Optional, Literal

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo import ParallelEnv

from .MAEnv_base import WarehouseEnvBase, PRESET_MAPS
from .rendering import WarehouseWidget, WarehouseMainWindow
from ..planning_query import PathQueryResult


class WarehouseEnv(ParallelEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "warehouse_v0"}

    def __init__(self, num_agvs=3, fov_size=5, render_mode=None, map_size: str = 'medium',
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
                 path_window: int = 10,
                 planner_args: Optional[dict] = None,
                 if_continuous: bool = True,
                 num_visible_tasks: Optional[int] = None,
                 kstep_conflict_check: Optional[int] = None,
                 targets_only_on_shelf: bool = True,
                 padding_path_enable: bool = True):

        # Backward compat: kstep_conflict_check is deprecated alias for path_window
        if kstep_conflict_check is not None:
            import warnings
            warnings.warn(
                "kstep_conflict_check is deprecated, use path_window instead",
                DeprecationWarning, stacklevel=2)
            path_window = kstep_conflict_check

        EzPickle.__init__(
            self,
            num_agvs,
            fov_size,
            render_mode,
            map_size,
            max_episode_steps,
            path_planner,
            path_window,
            planner_args,
            if_continuous,
            num_visible_tasks,
            None,
            targets_only_on_shelf,
            padding_path_enable,
        )
        ParallelEnv.__init__(self)

        self._env = WarehouseEnvBase(
            num_agvs=num_agvs,
            fov_size=fov_size,
            render_mode=render_mode,
            map_size=map_size,
            max_episode_steps=max_episode_steps,
            path_planner=path_planner,
            path_window=path_window,
            planner_args=planner_args,
            num_visible_tasks=num_visible_tasks,
            targets_only_on_shelf=targets_only_on_shelf,
            padding_path_enable=padding_path_enable,
        )

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.fov_size = fov_size
        self.map_size = map_size
        self.width = self._env.width
        self.height = self._env.height
        self.num_agvs = num_agvs
        self.if_continuous = bool(if_continuous)

        if self.render_mode == "human":
            self._init_render_window()

    @property
    def agents(self):
        return self._env.agents

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def grid_map(self):
        return self._env.grid_map

    @property
    def agvs(self):
        return self._env.agvs

    @property
    def action_spaces(self):
        return self._env.action_spaces

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    @property
    def map_config(self):
        return self._env.map_config

    @property
    def agent_terminations(self):
        return self._env._agent_terminations

    @property
    def episode_count(self):
        return self._env._episode_count

    @property
    def current_step(self):
        return self._env._current_step

    @property
    def path_planner(self):
        return self._env.path_planner

    @property
    def path_window(self):
        return self._env.path_window

    @property
    def num_visible_tasks(self):
        return self._env.num_visible_tasks

    @property
    def targets_only_on_shelf(self):
        return self._env.targets_only_on_shelf

    @property
    def padding_path_enable(self):
        return self._env.padding_path_enable

    def _get_cached_planner_paths_info(self):
        return self._env._get_cached_planner_paths_info()

    @property
    def task_manager(self):
        return self._env.task_manager

    def get_agent_info(self, agent_name):
        if agent_name not in self.agvs:
            return None

        agv = self.agvs[agent_name]
        is_terminated = self._env._agent_terminations.get(agent_name, False)
        is_truncated = self._env._agent_truncations.get(agent_name, False)

        return {
            "id": agv.id,
            "position": (agv.x, agv.y),
            "render_prev_position": self._env._render_prev_positions.get(agent_name, agv.position).to_tuple(),
            "render_current_position": self._env._render_current_positions.get(agent_name, agv.position).to_tuple(),
            "status": agv.status,
            "target_pos": agv.target_pos,
            "is_terminated": is_terminated,
            "is_truncated": is_truncated,
            "is_alive": agent_name in self.agents
        }

    def get_statistics(self):
        total_agents = len(self.possible_agents)
        alive_agents = len(self.agents)
        terminated_count = sum(1 for agent in self.possible_agents if self._env._agent_terminations.get(agent, False))
        truncated_count = sum(1 for agent in self.possible_agents if self._env._agent_truncations.get(agent, False))

        return {
            "total_agents": total_agents,
            "alive_agents": alive_agents,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "working_rate": alive_agents / total_agents if total_agents > 0 else 0,
            "episode": self._env._episode_count,
            "current_step": self._env._current_step,
            "max_episode_steps": self._env.max_episode_steps,
            "map_size": (self._env.width, self._env.height)
        }

    def observation_space(self, agent):
        return self._env.observation_spaces[agent]

    def action_space(self, agent):
        return self._env.action_spaces[agent]

    def action_mask(self, agent):
        return self._env.action_mask(agent)

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        if self.render_mode == "human":
            self.render(if_continuous=self.if_continuous)

        return observations, rewards, terminations, truncations, infos

    def query_paths(
        self,
        planner_type: str,
        planner_args: Optional[dict] = None,
        *,
        timeout: Optional[float] = None,
        use_current_constraints: bool = True,
    ) -> PathQueryResult:
        """Compute paths once without changing environment or planner state."""
        return self._env.query_paths(
            planner_type,
            planner_args,
            timeout=timeout,
            use_current_constraints=use_current_constraints,
        )

    def teleport_agv(self, agent_name, x, y):
        return self._env.teleport_agv(agent_name, x, y)

    def render(self, if_continuous: Optional[bool] = None):
        if if_continuous is None:
            if_continuous = self.if_continuous
        if self.render_mode == "human":
            self._render_human(if_continuous=bool(if_continuous))
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array(if_continuous=bool(if_continuous))
        return None

    def _init_render_window(self):
        self._main_window = WarehouseMainWindow(self)

    def _render_human(self, if_continuous: bool = False):
        self._main_window.update_ui(if_continuous=if_continuous)

    def _init_rgb_widget(self):
        self._rgb_widget = WarehouseWidget(self)

    def _render_rgb_array(self, if_continuous: bool = False):
        if not hasattr(self, '_rgb_widget') or self._rgb_widget is None:
            self._init_rgb_widget()

        self._rgb_widget.render(if_continuous=if_continuous)
        arr = self._rgb_widget.get_rgb_array()
        arr = np.transpose(arr, (1, 0, 2))
        return arr

    def close(self):
        self._env.close()

        try:
            if hasattr(self, '_main_window') and self._main_window is not None:
                self._main_window.close()
                self._main_window = None

            if hasattr(self, '_rgb_widget') and self._rgb_widget is not None:
                self._rgb_widget.close()
                self._rgb_widget = None
        except Exception:
            pass

    def state(self) -> np.ndarray:
        return None
