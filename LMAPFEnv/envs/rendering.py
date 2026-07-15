import time
from typing import Dict, List, Tuple, Optional

import pygame

from .entities import AGV, Position, TaskStatus


class WarehouseRenderer:
    """Main rendering class for warehouse simulation"""

    def __init__(self, env):
        self.env = env

        self.cell_size = max(10, min(30, 600 // int(max(env.width, env.height))))

        self.window_width = env.width * self.cell_size
        self.window_height = env.height * self.cell_size

        self.show_trajectories = True
        self.trajectory_opacity = 180
        self.continuous_frames = 8
        self.continuous_fps = 60

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Warehouse AGV Simulation")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont('Arial', 12)
        self.running = True

        self._grid_surface = None
        self._trajectory_surface = None
        self._build_grid_surface()

    def _build_grid_surface(self):
        """Create static grid background with coordinate labels on grid"""
        self._grid_surface = pygame.Surface((self.window_width, self.window_height))
        for y in range(self.env.height):
            for x in range(self.env.width):
                grid = self.env.grid_map[y][x]
                color = grid.render_color
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self._grid_surface, color, rect)
                pygame.draw.rect(self._grid_surface, (0, 0, 0), rect, 1)

        label_color = (180, 50, 50)
        axis_font = pygame.font.SysFont('Arial', max(8, self.cell_size // 3), bold=True)

        step_x = 1 if self.cell_size >= 15 else max(1, self.env.width // 20)
        step_y = 1 if self.cell_size >= 15 else max(1, self.env.height // 20)

        for x in range(0, self.env.width, step_x):
            text = axis_font.render(str(x), True, label_color)
            tx = x * self.cell_size + self.cell_size // 2 - text.get_width() // 2
            ty = 1
            self._grid_surface.blit(text, (tx, ty))

        for y in range(0, self.env.height, step_y):
            text = axis_font.render(str(y), True, label_color)
            tx = 1
            ty = y * self.cell_size + self.cell_size // 2 - text.get_height() // 2
            self._grid_surface.blit(text, (tx, ty))

    def _cell_center(self, x, y):
        """Convert grid (x, y) to pixel center."""
        return (x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size

    def render(self, if_continuous: bool = False):
        """Render the current state, optionally animating the latest step."""
        if if_continuous and self._has_motion_to_animate():
            self._render_continuous()
        else:
            self._poll_events()
            self._render_frame(progress=1.0)
            self.clock.tick(60)

    def _poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _render_continuous(self):
        frame_count = max(2, int(self.continuous_frames))
        for frame_idx in range(1, frame_count + 1):
            self._poll_events()
            if not self.running:
                break
            progress = frame_idx / frame_count
            self._render_frame(progress=progress)
            self.clock.tick(max(1, int(self.continuous_fps)))

    def _render_frame(self, progress: float = 1.0):
        self.screen.blit(self._grid_surface, (0, 0))

        terminated_agvs = []
        active_agvs = []
        for agent, agv in self.env.agvs.items():
            agent_info = self.env.get_agent_info(agent)
            if agent_info is not None:
                is_terminated = agent_info["is_terminated"]
                if is_terminated:
                    terminated_agvs.append((agent, agv, True))
                else:
                    active_agvs.append((agent, agv, False))

        for agent_name, agv, is_terminated in terminated_agvs:
            self._draw_agv(
                agv,
                is_terminated,
                position=self._get_agent_render_position(agent_name, progress),
            )

        if self.show_trajectories:
            self._draw_trajectories(progress=progress)

        self._draw_task_targets()

        for agent_name, agv, is_terminated in active_agvs:
            self._draw_agv(
                agv,
                is_terminated,
                position=self._get_agent_render_position(agent_name, progress),
            )

        pygame.display.flip()

    def _has_motion_to_animate(self) -> bool:
        for agent_name in self.env.agvs.keys():
            info = self.env.get_agent_info(agent_name)
            if info is None:
                continue
            if tuple(info["render_prev_position"]) != tuple(info["render_current_position"]):
                return True
        return False

    def _get_agent_render_position(self, agent_name: str, progress: float) -> Tuple[float, float]:
        info = self.env.get_agent_info(agent_name)
        if info is None:
            agv = self.env.agvs[agent_name]
            return float(agv.x), float(agv.y)

        prev_x, prev_y = info["render_prev_position"]
        curr_x, curr_y = info["render_current_position"]
        alpha = max(0.0, min(1.0, float(progress)))
        x = prev_x + (curr_x - prev_x) * alpha
        y = prev_y + (curr_y - prev_y) * alpha
        return float(x), float(y)

    def _draw_agv(self, agv, is_terminated=False, position: Optional[Tuple[float, float]] = None):
        """Draw individual AGV"""
        draw_x, draw_y = position if position is not None else (agv.x, agv.y)
        center_x, center_y = self._cell_center(draw_x, draw_y)
        radius = self.cell_size // 3

        if is_terminated:
            color = (200, 200, 200)
            border_color = (150, 150, 150)
        else:
            color = AGV.AGV_COLORS[agv.id % len(AGV.AGV_COLORS)]
            border_color = (0, 0, 0)

        pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), radius)
        pygame.draw.circle(self.screen, border_color, (int(center_x), int(center_y)), radius, 2)

    def _draw_task_targets(self):
        """Draw task targets as light green diamonds"""
        tasks = self.env._tasks
        for agent, agv in self.env.agvs.items():
            agent_info = self.env.get_agent_info(agent)
            if agent_info is not None and agent_info["is_terminated"]:
                continue
                
            task = tasks.get_task(agv.id)
            if task is not None and task.status == TaskStatus.ACTIVE:
                target_x, target_y = task.target_pos.x, task.target_pos.y
                
                target_color = (144, 238, 144)
                center_x, center_y = self._cell_center(target_x, target_y)
                diamond_size = self.cell_size // 3
                
                points = [
                    (center_x, center_y - diamond_size),
                    (center_x + diamond_size, center_y),
                    (center_x, center_y + diamond_size),
                    (center_x - diamond_size, center_y)
                ]
                
                pygame.draw.polygon(self.screen, target_color, points)
                pygame.draw.polygon(self.screen, (0, 100, 0), points, 2)

    def set_trajectory_visibility(self, visible: bool):
        """Toggle trajectory visibility"""
        self.show_trajectories = visible

    def set_trajectory_opacity(self, opacity: int):
        """Set trajectory opacity (0-255)"""
        self.trajectory_opacity = max(0, min(255, opacity))

    def _draw_trajectories(self, progress: float = 1.0):
        """Draw path planner trajectories.

        Reads directly from the planner_paths info cache so that what is
        rendered on screen is *byte-identical* to what
        ``info[agent]['planner_paths']['path_abs']`` reports.
        """
        # Use the same data source as info output
        planner_info = self.env._get_cached_planner_paths_info()
        if not planner_info:
            return

        if self._trajectory_surface is None:
            self._trajectory_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        else:
            self._trajectory_surface.fill((0, 0, 0, 0))

        for agent_name, agv in self.env.agvs.items():
            agent_info = self.env.get_agent_info(agent_name)
            if agent_info is not None and agent_info["is_terminated"]:
                continue

            path_data = planner_info.get(agent_name, {})
            if not path_data.get("has_path", False):
                continue
            path_abs = path_data.get("path_abs")
            if path_abs is None or len(path_abs) < 2:
                continue

            r, g, b = AGV.AGV_COLORS[agv.id % len(AGV.AGV_COLORS)]
            trajectory_color = (r, g, b, self.trajectory_opacity)

            # Convert path_abs numpy array → list of (x, y) grid positions
            seq = [(int(round(float(path_abs[i, 0]))), int(round(float(path_abs[i, 1]))))
                   for i in range(len(path_abs))]

            if not seq or len(seq) <= 1:
                if seq:
                    self._draw_target_marker(self._trajectory_surface,
                                             type('P', (), {'x': seq[-1][0], 'y': seq[-1][1]})(),
                                             trajectory_color)
                continue

            # Validate adjacency: split into contiguous axis-aligned segments
            segments: List[List[Tuple[float, float]]] = []
            current_seg: List[Tuple[float, float]] = []
            for i, (gx, gy) in enumerate(seq):
                px, py = self._cell_center(gx, gy)
                if current_seg and i > 0:
                    prev_gx, prev_gy = seq[i - 1]
                    mdist = abs(gx - prev_gx) + abs(gy - prev_gy)
                    if mdist > 1:
                        segments.append(current_seg)
                        current_seg = []
                current_seg.append((px, py))
            if current_seg:
                segments.append(current_seg)

            for seg in segments:
                deduped_points: List[Tuple[float, float]] = []
                for point in seg:
                    if not deduped_points or point != deduped_points[-1]:
                        deduped_points.append(point)
                self._draw_single_trajectory(
                    self._trajectory_surface, deduped_points, trajectory_color)

            last = seq[-1]
            self._draw_target_marker(self._trajectory_surface,
                                     type('P', (), {'x': last[0], 'y': last[1]})(),
                                     trajectory_color)

        self.screen.blit(self._trajectory_surface, (0, 0))

    def _draw_single_trajectory(self, surface, path_points: List[Tuple[float, float]], color: Tuple[int, int, int, int]):
        """Draw a single trajectory from the current rendered position to the goal."""
        if not path_points or len(path_points) <= 1:
            return

        for i in range(len(path_points) - 1):
            center_x1, center_y1 = path_points[i]
            center_x2, center_y2 = path_points[i + 1]

            line_width = max(2, self.cell_size // 6)
            pygame.draw.line(surface, color, (int(center_x1), int(center_y1)), (int(center_x2), int(center_y2)), line_width)

            dot_radius = self.cell_size // 8
            pygame.draw.circle(surface, color, (int(center_x2), int(center_y2)), dot_radius)

    def _draw_target_marker(self, surface, target_pos: Position, color: Tuple[int, int, int, int]):
        """Draw target position marker"""
        center_x, center_y = self._cell_center(target_pos.x, target_pos.y)
        radius = self.cell_size // 4

        points = [
            (int(center_x), int(center_y - radius)),
            (int(center_x + radius), int(center_y)),
            (int(center_x), int(center_y + radius)),
            (int(center_x - radius), int(center_y))
        ]
        pygame.draw.polygon(surface, color, points)
        border_width = max(1, self.cell_size // 12)
        pygame.draw.polygon(surface, (0, 0, 0, 255), points, border_width)

    def get_rgb_array(self):
        """Get RGB array for video recording"""
        return pygame.surfarray.array3d(self.screen)

    def close(self):
        """Clean up resources"""
        self.running = False
        pygame.quit()


WarehouseWidget = WarehouseRenderer


class WarehouseMainWindow:
    """Main window wrapper for rendering"""
    
    def __init__(self, env):
        self.env = env
        self.warehouse_widget = WarehouseRenderer(env)
        self.auto_simulation = False

    def update_ui(self, if_continuous: bool = False):
        """Update UI components"""
        self.warehouse_widget.render(if_continuous=if_continuous)

    def close(self):
        """Close window"""
        self.warehouse_widget.close()

    @property
    def running(self):
        """Check if window is running"""
        return self.warehouse_widget.running
