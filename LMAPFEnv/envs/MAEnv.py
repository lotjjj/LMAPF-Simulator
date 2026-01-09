"""
Warehouse environment main module - WarehouseEnv environment class
"""
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
from PySide6.QtWidgets import QApplication
from gymnasium.spaces import Box, Discrete, Dict
from pettingzoo import ParallelEnv

from .entities import AGV, Wall, Shelf, Corridor, ChargingStation, Action, Direction
from .rendering import WarehouseWidget, WarehouseMainWindow


class TaskStatus(Enum):
    """Task status enumeration"""
    ACTIVE = 0      # Active (not completed)
    COMPLETED = 1    # Completed
    ABANDONED = 2    # Abandoned


class Task:
    """Task class - target position for AGV"""
    def __init__(self, target_pos: Tuple[int, int], status: TaskStatus = TaskStatus.ACTIVE):
        self.target_pos = target_pos  # Target coordinate (x, y)
        self.status = status  # Task status
        self.assigned_agent = None  # Assigned agent_id


class Tasks:
    """Task manager singleton"""
    _instance = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tasks: Dict[int, Task] = {}
            cls._lock = True
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset task manager"""
        if cls._instance is not None:
            cls._instance.tasks.clear()

    def add_task(self, agent_id: int, target_pos: Tuple[int, int]):
        """Add task for AGV"""
        self.tasks[agent_id] = Task(target_pos, TaskStatus.ACTIVE)

    def get_task(self, agent_id: int) -> Optional[Task]:
        """Get AGV task"""
        return self.tasks.get(agent_id)

    def update_task_status(self, agent_id: int, status: TaskStatus):
        """Update task status"""
        if agent_id in self.tasks:
            self.tasks[agent_id].status = status

    def assign_random_target(self, agent_id: int, passable_positions: list, rng):
        """Assign random target position for AGV"""
        target_pos = rng.choice(passable_positions)
        self.add_task(agent_id, target_pos)
        return target_pos

    def is_task_completed(self, agent_id: int, current_pos: Tuple[int, int]) -> bool:
        """Check if task is completed"""
        task = self.tasks.get(agent_id)
        if task is None or task.status != TaskStatus.ACTIVE:
            return False
        return current_pos == task.target_pos


@dataclass
class BatteryConfig:
    energy_decay: float = 0.005  # Battery decay rate
    energy_decay_noise: float = 0.002  # Battery decay noise standard deviation
    min_battery_level: float = 0.0  # Minimum battery level
    max_battery_level: float = 1.0  # Maximum battery level
    charging_rate: float = 0.1  # Charging rate
    low_battery_threshold: float = 0.1  # Low battery threshold


@dataclass
class MapConfig:
    shelf_cols: int = 3  # Number of shelf columns
    shelf_rows: int = 3  # Number of shelf rows
    shelf_width: int = 3  # Single shelf width (grid cells)
    shelf_height: int = 2  # Single shelf height (grid cells)
    corridor_width: int = 1  # Corridor width between shelves (grid cells)
    corridor_out_width: int = 2  # Outer corridor width along walls (grid cells)


class WarehouseEnv(ParallelEnv):
    """Warehouse AGV simulation environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "name": "warehouse_v0"}

    def __init__(self, num_agvs=6,  fov_size=5, render_mode=None, enable_battery=False,
                 battery_config: Optional[BatteryConfig] = BatteryConfig(),
                 map_config: Optional[MapConfig] = MapConfig(),
                 max_episode_steps=500):
        """
        Initialize warehouse environment

        Parameters:
        num_agvs: Number of AGVs
        render_mode: Rendering mode
        enable_battery: Enable battery system (charging stations, battery consumption, battery observation and battery reward)
        fov_size: Local observation field of view size (grid cells), must be odd, default is 5
        battery_config: Battery configuration parameters, use default config if None
        map_config: Map configuration parameters, use default config if None
        max_episode_steps: Maximum number of steps per episode, default is 1000
        """
        super().__init__()
        self.num_agvs = num_agvs
        self.render_mode = render_mode
        self.enable_battery = enable_battery
        self.max_episode_steps = max_episode_steps

        self.battery_config = battery_config
        self.map_config = map_config

        # Map configuration parameters
        self.shelf_cols = self.map_config.shelf_cols
        self.shelf_rows = self.map_config.shelf_rows
        self.shelf_width = self.map_config.shelf_width
        self.shelf_height = self.map_config.shelf_height
        self.corridor_width = self.map_config.corridor_width
        self.corridor_out_width = self.map_config.corridor_out_width

        # Observation configuration parameters
        if fov_size % 2 == 0:
            raise ValueError(f"fov_size must be odd, right now it is {fov_size}")
        self.fov_size = fov_size
        self.fov_radius = fov_size // 2

        # Calculate map size based on shelf parameters
        self.width, self.height = self._calculate_map_size()

        # Create AGVs
        self.agvs = {}
        self.possible_agents = [f"agv_{i}" for i in range(num_agvs)]
        self.agents = self.possible_agents[:]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Create map
        self.grid_map = self._create_initial_map()

        # AGV position reverse index: {agv_id: (x, y)}
        self._agv_positions = {}

        # Passable positions for random target assignment
        self._passable_positions = self._get_passable_positions()

        # Episode counter
        self._episode_count = 0
        self._current_step = 0

        # Seed management
        self._seed = None
        self._rng_state = None
        self.np_random = None

        # Task reward
        self.task_completion_reward = 10.0

        self._action_space = Discrete(4)
        self._observation_space = self._create_observation_space()

        if self.render_mode == "human":
            self._init_render_window()

    def _calculate_map_size(self):
        # Calculate total width and height of shelf area
        # No shelves when shelf_cols or shelf_rows is 0
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
                grid = None

                # Boundary walls
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    grid = Wall(x, y)
                else:
                    # Check if in charging area (adjacent to walls, width=1)
                    if self.enable_battery and self._is_charging_area(x, y):
                        grid = ChargingStation(x, y)
                    # Check if in shelf area (centered layout)
                    elif self._is_shelf_area(x, y):
                        grid = Shelf(x, y)
                    # Others are corridors
                    else:
                        grid = Corridor(x, y)

                row.append(grid)
            grid_map.append(row)
        return grid_map

    def _is_shelf_area(self, x, y):
        """Check if position is in shelf area - centered layout"""
        # Return False if no shelves
        if self.shelf_cols <= 0 or self.shelf_rows <= 0:
            return False
        
        # Outer corridor width
        outer_margin = self.corridor_out_width

        # Calculate total shelf width and height
        total_shelf_width = self.shelf_cols * self.shelf_width + (self.shelf_cols - 1) * self.corridor_width
        total_shelf_height = self.shelf_rows * self.shelf_height + (self.shelf_rows - 1) * self.corridor_width

        # No shelves if space too small
        if total_shelf_width > self.width - 2 - 2 * outer_margin or \
                total_shelf_height > self.height - 2 - 2 * outer_margin:
            return False

        # Calculate shelf area start position (centered)
        shelf_area_width = self.width - 2 - 2 * outer_margin
        shelf_area_height = self.height - 2 - 2 * outer_margin

        start_x = outer_margin + 1 + (shelf_area_width - total_shelf_width) // 2
        start_y = outer_margin + 1 + (shelf_area_height - total_shelf_height) // 2

        # Check if in any shelf area
        for shelf_row in range(self.shelf_rows):
            for shelf_col in range(self.shelf_cols):
                shelf_start_x = start_x + shelf_col * (self.shelf_width + self.corridor_width)
                shelf_start_y = start_y + shelf_row * (self.shelf_height + self.corridor_width)

                if shelf_start_x <= x < shelf_start_x + self.shelf_width and \
                        shelf_start_y <= y < shelf_start_y + self.shelf_height:
                    return True

        return False

    def _is_charging_area(self, x, y):
        """Check if position is in charging area (adjacent to walls, width=1)"""
        # Charging station width is 1, adjacent to outer walls
        # Top edge (y=1)
        if y == 1 and 0 < x < self.width - 1:
            return True
        # Bottom edge (y=height-2)
        if y == self.height - 2 and 1 < x < self.width - 1:
            return True
        # Left edge (x=1)
        if x == 1 and 1 < y < self.height - 1:
            return True
        # Right edge (x=width-2)
        if x == self.width - 2 and 1 < y < self.height - 1:
            return True

        return False

    def _initialize_agvs(self):
        """Initialize AGV positions, battery, and direction"""
        for i in range(self.num_agvs):
            # Randomly select an occupiable position
            while True:
                x = self.np_random.integers(1, self.width - 2)
                y = self.np_random.integers(1, self.height - 2)
                if self.grid_map[y][x].occupiable and len(self.grid_map[y][x].agvs) == 0:
                    # Randomly initialize battery (0.5 to 1.0)
                    battery_level = self.np_random.uniform(0.5, 1.0)
                    # Use Direction enum to initialize direction
                    direction = list(Direction)[self.np_random.integers(0, 4)]
                    
                    agv = AGV(i, (x, y), battery_level=battery_level, direction=direction)
                    self.agvs[f"agv_{i}"] = agv
                    self.grid_map[y][x].add_agv(agv)
                    # Update position index
                    self._agv_positions[i] = (x, y)
                    break

    def _create_observation_space(self):
        """Create observation space (internal method) - local fov_sizexfov_size observation"""
        # Define self_states space
        self_states_dict = {
            "position": Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32),
            "direction": Discrete(4),  # Direction: 0=up, 1=down, 2=left, 3=right
            "target": Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32),  # Target position
        }
        if self.enable_battery:
            self_states_dict["battery"] = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self_states_space = Dict(self_states_dict)

        # Define fov space (multi-channel map)
        # Channels: local_map, other_agv_positions, other_agv_directions (one-hot encoded: 4 directions)
        num_channels = 3 + 4 if self.enable_battery else 3 + 4  # Correction: direction channels exist even without battery
        # Correction: channels fixed at 3 (local_map, other_agv_pos, other_agv_dir_onehot)
        # other_agv_directions is one-hot encoded, takes 4 channels
        num_channels = 3  # local_map, other_agv_positions, one-hot directions (4 channels)
        if self.enable_battery:
            num_channels += 1  # battery level channel
        # Correction: channels should be local_map(1) + other_agv_pos(1) + other_agv_dir_onehot(4) = 6
        fov_num_channels = 1 + 1 + 4  # local_map, other_agv_positions, other_agv_directions (one-hot)
        if self.enable_battery:
            fov_num_channels += 1  # Add channel for battery level in fov if needed, but logically it belongs to self_states
        # Actually, let's keep fov_num_channels as 6 (or 5 if no battery in fov) and handle battery in self_states
        fov_num_channels = 1 + 1 + 4  # local_map, other_agv_pos, other_agv_dir_onehot
        fov_space = Box(low=0, high=1, shape=(fov_num_channels, self.fov_size, self.fov_size), dtype=np.float32)

        obs_space = Dict({
            "self_states": self_states_space,
            "fov": fov_space
        })
        return obs_space

    def observation_space(self, agent):
        """Define observation space - return cached space object"""
        return self._observation_space

    def action_space(self, agent):
        """Define action space - return cached space object"""
        return self._action_space

    def action_mask(self, agent):
        """Return action mask - considering battery, walls, AGV conflicts, and shelf restrictions"""
        if agent not in self.agvs or agent not in self.agents:
            return [0, 0, 0, 0]

        agv = self.agvs[agent]
        # 从 agv 对象获取位置，而不是依赖 _agv_positions
        x, y = agv.x, agv.y

        # 初始化所有动作为可用: [Forward, TurnLeft, TurnRight, Stop]
        mask = [1, 1, 1, 1]

        # Check if Forward action is available
        # Calculate forward position based on current direction, using Direction.get_delta()
        dx, dy = agv.direction.get_delta()
        new_x, new_y = x + dx, y + dy

        # Check boundaries
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            mask[Action.FORWARD.value] = 0
        else:
            target_grid = self.grid_map[new_y][new_x]
            current_grid = self.grid_map[y][x]

            if not target_grid.passable:
                mask[Action.FORWARD.value] = 0
            # Check shelf-to-shelf movement restriction
            elif isinstance(target_grid, Shelf) and isinstance(current_grid, Shelf):
                mask[Action.FORWARD.value] = 0
            # Check AGV conflict
            elif len(target_grid.agvs) > 0:
                mask[Action.FORWARD.value] = 0

        return mask

    def teleport_agv(self, agent_name, x, y):
        """
        Safely teleport AGV to specified position (for testing/debugging)

        Args:
            agent_name: AGV name (e.g., "agv_0")
            x: Target x coordinate
            y: Target y coordinate
        """
        if agent_name not in self.agvs:
            return

        agv = self.agvs[agent_name]

        # Remove AGV from old position
        old_grid = self.grid_map[agv.y][agv.x]
        old_grid.remove_agv(agv)

        # Update AGV position
        agv.x = x
        agv.y = y

        # Add to new position
        new_grid = self.grid_map[y][x]
        new_grid.add_agv(agv)

        # Update position index
        self._agv_positions[agv.id] = (x, y)

    def reset(self, seed=None, options=None):
        if seed is not None:
            # 如果提供了种子，创建一个新的确定性生成器
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            # 如果首次调用且无种子，创建一个非确定性生成器
            self.np_random = np.random.default_rng()

        self._action_space.sample = lambda: self.np_random.integers(0, 4)

        # Recreate map and AGVs
        self.grid_map = self._create_initial_map()
        self._agv_positions.clear()  # Clear position index
        self._initialize_agvs()

        # Reset task manager
        Tasks.reset()

        # Assign initial tasks to all AGVs
        self._assign_tasks()

        # Reset state management variables
        self.agents = self.possible_agents[:]  # Reset active agent list
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._agent_dones = {agent: False for agent in self.agents}
        self._agent_rewards = {agent: 0 for agent in self.agents}
        self._agent_terminations = {agent: False for agent in self.agents}
        self._agent_truncations = {agent: False for agent in self.agents}
        self._episode_count += 1  # Increment episode counter
        self._current_step = 0  # Reset step counter

        # Get initial observations and infos
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def _get_obs(self, agent):
        """Get observation for a single agent"""
        agv = self.agvs[agent]

        # --- Build fov (Field of View) ---
        # Channels: local_map(1), other_agv_positions(1), other_agv_directions_onehot(4)
        fov_num_channels = 1 + 1 + 4
        fov_tensor = np.zeros((fov_num_channels, self.fov_size, self.fov_size), dtype=np.float32)

        for dy in range(-self.fov_radius, self.fov_radius + 1):
            for dx in range(-self.fov_radius, self.fov_radius + 1):
                nx, ny = agv.x + dx, agv.y + dy
                fov_dy = dy + self.fov_radius
                fov_dx = dx + self.fov_radius

                # Check boundary
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Channel 0: local_map (passable)
                    fov_tensor[0, fov_dy, fov_dx] = 1.0 if self.grid_map[ny][nx].passable else 0.0
                    
                    # Channel 1: other_agv_positions
                    other_agv = None
                    for other_agent in self.possible_agents:
                        if other_agent in self.agvs and other_agent != agent:
                            other_agv_obj = self.agvs[other_agent]
                            if other_agv_obj.x == nx and other_agv_obj.y == ny:
                                other_agv = other_agv_obj
                                break
                    
                    fov_tensor[1, fov_dy, fov_dx] = 1.0 if other_agv is not None else 0.0
                    
                    # Channels 2-5: other_agv_directions (one-hot encoded using Direction enum)
                    # One-hot encoding order: [UP, DOWN, LEFT, RIGHT]
                    one_hot_dir = np.zeros(4, dtype=np.float32)
                    if other_agv is not None:
                        # Convert direction to Direction enum and use its value for one-hot encoding
                        one_hot_dir[other_agv.direction.value] = 1.0
                    fov_tensor[2:6, fov_dy, fov_dx] = one_hot_dir
                else:
                    # Out of bounds is impassable
                    fov_tensor[0, fov_dy, fov_dx] = 0.0
                    fov_tensor[1, fov_dy, fov_dx] = 0.0
                    fov_tensor[2:6, fov_dy, fov_dx] = 0.0

        # --- Build self_states ---
        # Get target position from Tasks singleton
        task = Tasks().get_task(agv.id)
        target_pos = task.target_pos if task is not None else np.array([0, 0], dtype=np.int32)

        self_states = {
            "position": np.array([agv.x, agv.y], dtype=np.int32),
            "direction": np.array(agv.direction.value, dtype=np.int32),
            "target": np.array(target_pos, dtype=np.int32),
        }
        if self.enable_battery:
            self_states["battery"] = np.array([agv.battery_level], dtype=np.float32)

        obs = {
            "self_states": self_states,
            "fov": fov_tensor
        }
        return obs

    def _get_info(self, agent):
        """Get info"""
        # Can return additional info here, e.g., action mask
        return {"action_mask": self.action_mask(agent)}

    def step(self, actions):
        # Step 1: 静态动作处理 - 检查动作的可行性
        # 为每个agent确定其是否可以执行请求的动作
        feasible_actions = {}

        for agent, action in actions.items():
            if agent in self.agents and not self._agent_terminations[agent]:
                agv = self.agvs[agent]
                agv.req_action = Action(action)

                # 检查动作是否可行
                feasible = True
                if agv.req_action == Action.FORWARD:
                    # 获取目标位置
                    target_x, target_y = self._get_target_position_static(agv)

                    # 检查边界
                    if not (0 <= target_x < self.width and 0 <= target_y < self.height):
                        feasible = False
                    else:
                        target_grid = self.grid_map[target_y][target_x]
                        current_grid = self.grid_map[agv.y][agv.x]

                        # 检查是否尝试进入Wall
                        if not target_grid.passable:
                            feasible = False
                        # 检查是否尝试从一个货架移动到另一个货架
                        elif isinstance(current_grid, Shelf) and isinstance(target_grid, Shelf):
                            feasible = False
                        # 检查目标位置是否已被其他AGV占用
                        # elif len(target_grid.agvs) > 0:
                        #     feasible = False

                if feasible:
                    feasible_actions[agent] = agv.req_action
                else:
                    # 不可行动作转换为STOP
                    feasible_actions[agent] = Action.STOP
                    agv.req_action = Action.STOP
            else:
                # 对于已终止的agent，设置为STOP
                if agent in self.agvs:
                    self.agvs[agent].req_action = Action.STOP
                feasible_actions[agent] = Action.STOP

        # Step 2: 动态冲突解决 - 使用networkx建图分析
        # 构建有向图，只考虑可行的FORWARD动作
        G = nx.DiGraph()

        # 添加所有节点（当前AGV位置）
        for agent in self.agents:
            if agent in self.agvs and not self._agent_terminations[agent]:
                agv = self.agvs[agent]
                current_pos = (agv.x, agv.y)
                G.add_node(current_pos)

        # 添加边，只对可行的FORWARD动作添加移动边
        for agent in self.agents:
            if agent in self.agvs and not self._agent_terminations[agent]:
                agv = self.agvs[agent]
                current_pos = (agv.x, agv.y)

                if feasible_actions[agent] == Action.FORWARD:
                    # 获取目标位置
                    target_x, target_y = self._get_target_position_static(agv)
                    target_pos = (target_x, target_y)

                    # 只有当目标位置在边界内时才添加边
                    if 0 <= target_x < self.width and 0 <= target_y < self.height:
                        G.add_edge(current_pos, target_pos)
                    else:
                        # 如果目标位置越界，添加自环
                        G.add_edge(current_pos, current_pos)
                else:
                    # 非FORWARD动作添加自环
                    G.add_edge(current_pos, current_pos)

        # Step 3: 分析连通分量并解决冲突
        committed_agents = set()

        for component_nodes in nx.weakly_connected_components(G):
            component_nodes_list = list(component_nodes)

            if len(component_nodes_list) == 1:
                # 单节点分量：所有相关的AGV都保持不动
                node = component_nodes_list[0]
                x, y = node
                grid = self.grid_map[y][x]
                for agv in grid.agvs:
                    agent_id = f"agv_{agv.id}"
                    if agent_id in self.agents and agent_id in feasible_actions:
                        committed_agents.add(agv)
                continue

            # 创建子图
            comp_subgraph = G.subgraph(component_nodes_list).copy()

            try:
                # 尝试找环
                cycle = nx.find_cycle(comp_subgraph, orientation='original')

                # 检查是否是2节点环（交换动作，物理上不可能）
                if len(cycle) == 2:
                    # 对于2节点环，保持所有相关AGV不动
                    for edge in cycle:
                        start_node = edge[0]
                        x, y = start_node
                        grid = self.grid_map[y][x]
                        for agv in grid.agvs:
                            committed_agents.add(agv)
                    continue

                # 对于多节点环，所有节点上的AGV都可以移动
                for edge in cycle:
                    start_node = edge[0]
                    x, y = start_node
                    grid = self.grid_map[y][x]
                    for agv in grid.agvs:
                        committed_agents.add(agv)


            except nx.NetworkXNoCycle:
                # 无环：在DAG中找最长路径
                try:
                    longest_path = nx.dag_longest_path(comp_subgraph)

                    # 路径上的所有AGV都可以移动
                    for node in longest_path:
                        x, y = node
                        grid = self.grid_map[y][x]
                        for agv in grid.agvs:
                            committed_agents.add(agv)
                except:
                    # 如果DAG最长路径失败，跳过此组件
                    pass

        # Step 4: 执行最终确定的动作
        for agent in self.agents:
            if agent in self.agvs and not self._agent_terminations[agent]:
                agv = self.agvs[agent]

                # 检查该AGV是否被承诺移动
                if agv in committed_agents:
                    # 执行前进动作
                    self._execute_action(agent, feasible_actions[agent])
                else:
                    # 执行停止或其他动作
                    self._execute_action(agent, Action.STOP)

        # Step 5: 更新电池（如果启用）
        if self.enable_battery:
            self._update_battery_levels()

        # Step 5.5: 检查最大步数
        self._current_step += 1
        if self._current_step >= self.max_episode_steps:
            for agent in self.agents:
                self._agent_truncations[agent] = True

        # Step 6: 检查电池耗尽
        if self.enable_battery:
            for agent in self.agvs:
                if agent in self.agents:
                    agv = self.agvs[agent]
                    if agv.battery_level <= 0:
                        self._agent_terminations[agent] = True

        self._check_task_completion()

        rewards = {}
        for agent in self.agents:
            if not self._agent_terminations[agent]:
                rewards[agent] = self._calculate_reward(agent)
            else:
                rewards[agent] = 0

        # Step 8: 获取观察和信息
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        # Step 9: 更新累积奖励和移除终止的agent
        for agent in self.agents:
            if not (self._agent_terminations[agent] or self._agent_truncations[agent]):
                self._cumulative_rewards[agent] += rewards[agent]

        self.agents = [agent for agent in self.agents if
                       not (self._agent_terminations[agent] or self._agent_truncations[agent])]

        # Step 10: 渲染
        if self.render_mode == "human":
            self.render()

        return observations, rewards, self._agent_terminations, self._agent_truncations, infos

    def _get_target_position_static(self, agv):
        """Get target position (for static checks) - does not move AGV"""
        dx, dy = agv.direction.get_delta()
        new_x = agv.x + dx
        new_y = agv.y + dy
        return new_x, new_y

    def _copy_grid_map_state(self):
        """Copy current grid map state - using position index optimization"""
        # Return shallow copy of position index directly, avoiding full map traversal
        return self._agv_positions.copy()

    def _execute_action(self, agent, action):
        """Execute a single action - using Action and Direction enums"""
        agv = self.agvs[agent]
        old_x, old_y = agv.x, agv.y
        old_grid = self.grid_map[old_y][old_x]

        # Remove AGV from old grid
        old_grid.remove_agv(agv)

        # Execute based on action type
        action_enum = Action(action)
        new_x, new_y = old_x, old_y

        if action_enum == Action.FORWARD:
            # Forward: move based on current direction using Direction.get_delta()
            dx, dy = agv.direction.get_delta()
            new_x, new_y = old_x + dx, old_y + dy

            # Check boundaries and passability
            if (0 <= new_x < self.width and 0 <= new_y < self.height):
                target_grid = self.grid_map[new_y][new_x]
                can_move = True
                if isinstance(old_grid, Shelf) and isinstance(target_grid, Shelf):
                    can_move = False

                if target_grid.passable and can_move:
                    agv.x, agv.y = new_x, new_y
                    target_grid.add_agv(agv)
                    self._agv_positions[agv.id] = (new_x, new_y)
                else:
                    self.grid_map[old_y][old_x].add_agv(agv)
            else:
                self.grid_map[old_y][old_x].add_agv(agv)

        elif action_enum == Action.TURN_LEFT:
            # Turn left: use Direction.turn_left() method
            agv.direction = agv.direction.turn_left()
            self.grid_map[old_y][old_x].add_agv(agv)

        elif action_enum == Action.TURN_RIGHT:
            # Turn right: use Direction.turn_right() method
            agv.direction = agv.direction.turn_right()
            self.grid_map[old_y][old_x].add_agv(agv)

        elif action_enum == Action.STOP:
            # Stop: stay in place
            self.grid_map[old_y][old_x].add_agv(agv)

    def _update_battery_levels(self):
        """Update battery levels - using position index optimization"""
        for agent, agv in self.agvs.items():
            if not self._agent_terminations[agent]:
                # Calculate battery decay (base decay + random noise)
                decay = self.battery_config.energy_decay
                noise = self.np_random.normal(0, self.battery_config.energy_decay_noise)
                total_decay = max(0, decay + noise)  # Ensure decay is positive
                
                # Apply battery decay
                agv.battery_level = max(
                    self.battery_config.min_battery_level,
                    agv.battery_level - total_decay
                )

                # Use position index to quickly find current grid
                if agv.id in self._agv_positions:
                    x, y = self._agv_positions[agv.id]
                    current_grid = self.grid_map[y][x]
                    if isinstance(current_grid, ChargingStation):
                        # Charge at charging station
                        agv.battery_level = min(
                            self.battery_config.max_battery_level,
                            agv.battery_level + self.battery_config.charging_rate
                        )

    def _calculate_reward(self, agent):
        """Calculate reward"""
        agv = self.agvs[agent]

        hanging_reward = -0.005

        # Task completion reward
        task_completion_reward = 0.0
        task = Tasks().get_task(agv.id)
        if task is not None and task.status == TaskStatus.COMPLETED:
            task_completion_reward = self.task_completion_reward

        # Battery-related rewards when battery is enabled
        if self.enable_battery:
            # Battery reward/penalty
            battery_reward = agv.battery_level * 10

            return battery_reward + hanging_reward + task_completion_reward

        return hanging_reward + task_completion_reward

    def _get_passable_positions(self):
        """Get list of all passable positions"""
        passable_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid_map[y][x].passable:
                    passable_positions.append((x, y))
        return passable_positions

    def _assign_tasks(self):
        """Assign tasks to all AGVs"""
        tasks = Tasks()
        for agent_name, agv in self.agvs.items():
            # Assign a random target for each AGV
            target_pos = tasks.assign_random_target(agv.id, self._passable_positions, self.np_random)
            # Update AGV's target_pos attribute
            agv.target_pos = target_pos

    def _check_task_completion(self):
        """Check task completion and update"""
        tasks = Tasks()
        for agent_name, agv in self.agvs.items():
            task = tasks.get_task(agv.id)
            if task is not None and task.status == TaskStatus.ACTIVE:
                # Check if reached target position
                if agv.x == task.target_pos[0] and agv.y == task.target_pos[1]:
                    # Mark task as completed
                    tasks.update_task_status(agv.id, TaskStatus.COMPLETED)
                    print(f"[Task Completed] Agent {agent_name} reached target at {task.target_pos}")

                    # Assign new task
                    new_target = tasks.assign_random_target(agv.id, self._passable_positions, self.np_random)
                    agv.target_pos = new_target

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode is None:
            # headless
            pass
        return None

    def _init_render_window(self):
        """Initialize render window"""
        from PySide6.QtWidgets import QApplication
        # Initialize Qt app (using singleton pattern)
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        # Create main window
        self._main_window = WarehouseMainWindow(self)
        self._main_window.show()

    def _render_human(self):
        """Render to GUI window - only update display"""
        # Only update map and info panel, avoid full refresh
        self._main_window.update_ui()

        self._app.processEvents()

    def _init_rgb_widget(self):
        """Initialize RGB rendering widget"""
        from PySide6.QtWidgets import QApplication
        # Initialize Qt app (using singleton pattern)
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        # Create widget
        self._rgb_widget = WarehouseWidget(self)
        self._rgb_widget.resize(self.width * self._rgb_widget.cell_size,
                                self.height * self._rgb_widget.cell_size)

    def _render_rgb_array(self):
        """Render as RGB array - optimized version"""
        # Initialize widget (if needed)
        if not hasattr(self, '_rgb_widget') or self._rgb_widget is None:
            self._init_rgb_widget()

        # Render to image
        pixmap = self._rgb_widget.grab()
        image = pixmap.toImage()

        # Convert to numpy array
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())

        arr = np.array(ptr).reshape(height, width, 4)
        arr = arr[:, :, :3]  # Remove alpha channel

        return arr

    def close(self):
        """Close render window and clean up all Qt resources"""
        try:
            # Close main window
            if hasattr(self, '_main_window') and self._main_window is not None:
                self._main_window.close()
                self._main_window = None

            # Clean up RGB rendering widget
            if hasattr(self, '_rgb_widget') and self._rgb_widget is None:
                self._rgb_widget.deleteLater()
                self._rgb_widget = None

            # Quit Qt app (only when not shared instance)
            if hasattr(self, '_app') and self._app is not None:
                # Check if this is a shared QApplication instance
                if QApplication.instance() is self._app:
                    # Only quit when this is the only instance
                    self._app.quit()
                self._app = None
        except Exception:
            # Ignore exceptions during cleanup, ensure close method doesn't throw errors
            pass


    def state(self) -> np.ndarray:
        """Return global state (not implemented for this environment)"""
        # For partially observable environments, this method is optional
        # Return None as this environment uses local observations only
        return None



