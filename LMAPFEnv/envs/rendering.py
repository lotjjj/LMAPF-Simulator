
import time

from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QPainter, QColor, QBrush, QPen
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel
)

from .entities import AGV


class WarehouseWidget(QWidget):
    """仓库地图显示组件"""
    def __init__(self, env):
        super().__init__()
        self.env = env
        # 根据地图大小动态调整单元格尺寸，支持更大的仓库地图
        max_dimension = max(env.width, env.height)
        if max_dimension <= 20:
            self.cell_size = 30
        elif max_dimension <= 40:
            self.cell_size = 20
        elif max_dimension <= 60:
            self.cell_size = 15
        else:
            self.cell_size = 10
        self.setMinimumSize(self.env.width * self.cell_size, self.env.height * self.cell_size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 缓存颜色对象，避免重复创建
        color_cache = {}
        
        # 绘制网格
        for y in range(self.env.height):
            for x in range(self.env.width):
                grid = self.env.grid_map[y][x]

                # 绘制网格背景（使用缓存的颜色）
                color_key = grid.render_color
                if color_key not in color_cache:
                    color_cache[color_key] = QColor(*color_key)
                color = color_cache[color_key]
                
                painter.fillRect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size, color)

                # 绘制边框
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawRect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)

        # 绘制AGV
        for agent, agv in self.env.agvs.items():
            is_terminated = self.env._agent_terminations.get(agent, False)
            self._draw_agv(painter, agv, is_terminated)

    def _draw_agv(self, painter, agv, is_terminated=False):
        """绘制单个AGV"""
        center_x = (agv.x + 0.5) * self.cell_size
        center_y = (agv.y + 0.5) * self.cell_size
        radius = self.cell_size // 3

        # 绘制AGV主体
        base_color = AGV.STATUS_COLORS[agv.status]
        if is_terminated:
            # 终止的AGV使用浅色（降低饱和度和亮度）
            color = QColor(base_color)
            color.setHsv(color.hue(), color.saturation() // 3, 200)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(150, 150, 150), 1))  # 浅灰色边框
        else:
            painter.setBrush(QBrush(base_color))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
        
        painter.drawEllipse(center_x - radius, center_y - radius,
                            radius * 2, radius * 2)

        # 绘制方向半径
        self._draw_direction_radius(painter, center_x, center_y, radius, agv.direction, is_terminated)

        # 如果启用电池，绘制电量条
        if self.env.enable_battery:
            self._draw_battery_bar(painter, agv, center_x, center_y, radius, is_terminated)

    def _draw_direction_radius(self, painter, center_x, center_y, radius, direction, is_terminated=False):
        """绘制从AGV中心指向direction的半径"""
        # 使用 Direction 枚举的 get_delta 方法获取偏移量
        dx, dy = direction.get_delta()
        dx = dx * radius
        dy = dy * radius

        # 绘制方向半径线
        if is_terminated:
            line_color = QColor(200, 200, 200)  # 浅灰色
        else:
            line_color = QColor(255, 255, 255)  # 白色
        
        painter.setPen(QPen(line_color, max(2, self.cell_size // 8)))
        painter.drawLine(
            int(center_x),
            int(center_y),
            int(center_x + dx),
            int(center_y + dy)
        )

    def _draw_battery_bar(self, painter, agv, center_x, center_y, radius, is_terminated=False):
        """绘制AGV电量条"""
        bar_width = radius * 2
        # 根据单元格大小动态调整电量条高度
        bar_height = max(2, self.cell_size // 8)
        bar_x = center_x - radius
        bar_y = center_y + radius + 1

        # 绘制背景
        if is_terminated:
            bg_color = QColor(220, 220, 220)
            border_color = QColor(180, 180, 180)
        else:
            bg_color = QColor(200, 200, 200)
            border_color = QColor(0, 0, 0)
        
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 1))
        painter.drawRect(int(bar_x), int(bar_y), int(bar_width), int(bar_height))

        # 绘制电量
        battery_width = int(bar_width * agv.battery_level)
        if agv.battery_level > 0.5:
            battery_color = QColor(76, 175, 80)  # 绿色
        elif agv.battery_level > 0.2:
            battery_color = QColor(255, 193, 7)  # 黄色
        else:
            battery_color = QColor(244, 67, 54)  # 红色

        if is_terminated:
            # 终止的AGV电量条使用浅色
            battery_color.setHsv(battery_color.hue(), battery_color.saturation() // 3, 200)

        painter.setBrush(QBrush(battery_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(int(bar_x), int(bar_y), battery_width, int(bar_height))


class InfoPanel(QWidget):
    """Information panel component - Real-time display of alive_agents"""
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.setStyleSheet("""
            QWidget {
                background-color: #2c2c2c;
                border: 2px solid #6c757d;
                border-radius: 8px;
            }
        """)
        
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("📊 Warehouse Status")
        title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
            padding: 8px;
            background-color: #495057;
            border-radius: 4px;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Separator line
        line = QWidget()
        line.setFixedHeight(3)
        line.setStyleSheet("background-color: #6c757d; border-radius: 1px;")
        layout.addWidget(line)
        
        # Alive AGV count display
        self.alive_label = QLabel()
        self.alive_label.setStyleSheet("""
            color: #28a745;
            font-size: 16px;
            font-weight: bold;
            padding: 8px;
            background-color: #1e1e1e;
            border-radius: 4px;
            border: 2px solid #28a745;
        """)
        self.alive_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.alive_label)
        
        # Total AGV count display
        self.total_label = QLabel()
        self.total_label.setStyleSheet("""
            color: #adb5bd;
            font-size: 16px;
            padding: 8px;
            background-color: #343a40;
            border-radius: 4px;
        """)
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.total_label)
        
        # Episode counter
        self.episode_label = QLabel()
        self.episode_label.setStyleSheet("""
            color: #0d6efd;
            font-size: 16px;
            padding: 8px;
            background-color: #343a40;
            border-radius: 4px;
        """)
        self.episode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.episode_label)
        
        # Map information
        self.map_label = QLabel()
        self.map_label.setStyleSheet("""
            color: #ffc107;
            font-size: 16px;
            padding: 8px;
            background-color: #343a40;
            border-radius: 4px;
        """)
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.map_label)
        
        # Battery status
        self.battery_label = QLabel()
        self.battery_label.setStyleSheet("""
            color: #fd7e14;
            font-size: 16px;
            padding: 8px;
            background-color: #343a40;
            border-radius: 4px;
        """)
        self.battery_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.battery_label)
        
        # Add stretch space
        layout.addStretch()
        
        # Initial update
        self.update_info()

    def update_info(self):
        """Update information panel display"""
        # Alive AGV count
        alive_count = len(self.env.agents)
        total_count = len(self.env.possible_agents)
        
        self.alive_label.setText(f"Working Rate: {(alive_count/total_count*100)}%")
        self.total_label.setText(f"Total: {total_count}")
        self.episode_label.setText(f"Episode: {self.env._episode_count}")
        self.map_label.setText(f"Map: {self.env.width}×{self.env.height}")
        
        # Battery status
        if self.env.enable_battery:
            avg_battery = 0
            count = 0
            for agent, agv in self.env.agvs.items():
                if agent in self.env.agents:
                    avg_battery += agv.battery_level
                    count += 1
            if count > 0:
                avg_battery = avg_battery / count
                self.battery_label.setText(f"🔋 Avg Battery: {avg_battery:.1%}")
                self.battery_label.setVisible(True)
            else:
                self.battery_label.setVisible(False)
        else:
            self.battery_label.setVisible(False)


class SimulationWorker(QObject):
    """仿真工作线程"""
    step_completed = Signal()

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.running = False

    def run_simulation(self):
        """运行仿真循环"""
        self.running = True
        while self.running:
            # 随机动作作为示例
            actions = {}
            for agent in self.env.agents: # 修改：只对活动智能体采样动作
                actions[agent] = self.env.action_space(agent).sample()

            observations, rewards, terminations, truncations, infos = self.env.step(actions)

            # 发送信号通知主线程更新UI
            self.step_completed.emit()

            # 如果所有智能体都终止或截断，重置环境
            if not self.env.agents: # 修改：检查env.agents是否为空
                self.env.reset(seed=None) # 重置时不提供seed，使用随机位置

            time.sleep(0.1)  # 100ms


class WarehouseMainWindow(QMainWindow):
    """仓库仿真主窗口"""
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.setWindowTitle("Warehouse AGV Simulation")

        # 设置UI
        self.setup_ui()

        # 设置模拟线程
        self.simulation_worker = SimulationWorker(self.env)
        self.thread = QThread()
        self.simulation_worker.moveToThread(self.thread)
        self.simulation_worker.step_completed.connect(self.update_ui)
        self.thread.started.connect(self.simulation_worker.run_simulation)

        # 定时器更新UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # 100ms更新一次UI

        # 自动调整窗口大小以适应内容
        QTimer.singleShot(0, self.adjust_window_size)

    def setup_ui(self):
        """设置UI布局"""
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局 - 水平分割左右两部分
        main_layout = QHBoxLayout(central_widget)

        # 左侧：地图显示
        left_panel = QVBoxLayout()

        # 地图显示区域
        self.warehouse_widget = WarehouseWidget(self.env)
        left_panel.addWidget(self.warehouse_widget)

        # 右侧：信息看板
        self.info_panel = InfoPanel(self.env)

        # 将左右两部分添加到主布局
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.info_panel, 1)

        # 自动调整窗口大小以适应内容
        self.adjustSize()

    def update_ui(self):
        """更新UI显示"""
        # 更新地图
        self.warehouse_widget.update()

        # 更新信息看板
        self.info_panel.update_info()

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止仿真工作线程
        self.simulation_worker.running = False
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        
        # 调用环境的close方法清理资源
        if hasattr(self.env, 'close'):
            self.env.close()
        
        event.accept()

    def adjust_window_size(self):
        """自动调整窗口大小以适应内容，消除空白区域"""
        # 获取中央部件的理想大小
        central_widget = self.centralWidget()
        if central_widget:
            # 计算所有子部件所需的总大小
            size_hint = central_widget.sizeHint()

            # 设置窗口大小为内容所需大小，并添加一些边距
            self.resize(size_hint.width() + 20, size_hint.height() + 20)
