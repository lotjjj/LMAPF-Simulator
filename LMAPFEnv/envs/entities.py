
from enum import Enum
from typing import Tuple

from PySide6.QtGui import QColor


class Action(Enum):
    """AGV动作枚举"""
    FORWARD = 0      # 前进
    TURN_LEFT = 1    # 左转
    TURN_RIGHT = 2   # 右转
    STOP = 3         # 停止


class Direction(Enum):
    """AGV方向枚举"""
    UP = 0       # 上
    DOWN = 1     # 下
    LEFT = 2     # 左
    RIGHT = 3    # 右


    def turn_left(self):
        """左转，直接修改self"""
        if self == Direction.UP:
            return Direction.LEFT
        elif self == Direction.LEFT:
            return Direction.DOWN
        elif self == Direction.DOWN:
            return Direction.RIGHT
        else:  # RIGHT
            return Direction.UP

    def turn_right(self):
        """右转，直接修改self"""
        if self == Direction.UP:
            return Direction.RIGHT
        elif self == Direction.RIGHT:
            return Direction.DOWN
        elif self == Direction.DOWN:
            return Direction.LEFT
        else:  # LEFT
            return Direction.UP

    def get_delta(self):
         if self == Direction.UP:
             return 0, -1
         elif self == Direction.DOWN:
             return 0, 1
         elif self == Direction.LEFT:
             return -1, 0
         else:
             return 1, 0

    @staticmethod
    def from_value(value: int) -> 'Direction':
        """从整数值创建Direction枚举"""
        for direction in Direction:
            if direction.value == value:
                return direction
        raise ValueError(f"Invalid direction value: {value}")


class AGV:
    """AGV智能体类"""
    # 状态颜色映射
    STATUS_COLORS = {
        "IDLE": QColor(220, 53, 69),      # 红色 - 闲置
        "MOVING": QColor(13, 110, 253),   # 蓝色 - 移动中
        "WORKING": QColor(25, 135, 84),   # 绿色 - 工作中
        "CHARGING": QColor(255, 193, 7)   # 黄色 - 充电中
    }

    def __init__(self, id: int, start_pos: Tuple[int, int], battery_level: float = 1.0, direction: Direction = Direction.UP):
        self.id = id
        self.x, self.y = start_pos
        self.battery_level = battery_level
        self.target_pos = None
        self.carrying_item = False
        self.status = "IDLE"  # IDLE, MOVING, WORKING, CHARGING
        self.path = []
        self.speed = 1  # 每步移动的格子数
        self.capacity = 1  # 负载容量
        self.direction = direction  # 方向: Direction枚举


class Grid:
    """网格基类"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.occupiable = True  # 是否可以被AGV占用
        self.passable = True  # 是否可以通过
        self.render_color = (255, 255, 255)  # 默认白色
        self.agvs = []  # 当前在该网格上的AGV列表

    def add_agv(self, agv):
        """添加AGV到该网格"""
        if agv not in self.agvs:
            self.agvs.append(agv)

    def remove_agv(self, agv):
        """从该网格移除AGV"""
        if agv in self.agvs:
            self.agvs.remove(agv)

    def on_enter(self, agv):
        """当AGV进入时的回调函数"""
        return True  # 默认允许进入

    def on_occupy(self, agv):
        """当AGV占用时的回调函数"""
        return True  # 默认允许占用

    def on_leave(self, agv):
        """当AGV离开时的回调函数"""
        return True  # 默认允许离开


class Wall(Grid):
    """墙壁类"""
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.occupiable = False
        self.passable = False
        self.render_color = (108, 117, 125)  # 深灰色墙壁


class Shelf(Grid):
    """货架类"""
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.occupiable = True  # 货架可以被AGV占用
        self.passable = True   # 货架可以通过
        self.render_color = (184, 115, 51)  # 棕橙色货架


class Corridor(Grid):
    """走廊类"""
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.occupiable = True
        self.passable = True
        self.render_color = (248, 249, 250)  # 极浅灰色走廊


class ChargingStation(Grid):
    """充电站类"""
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.occupiable = True
        self.passable = True
        self.render_color = (255, 215, 0)
        self.charging_rate = 0.1

    def on_occupy(self, agv):
        """当AGV占用时充电"""
        if hasattr(agv, 'battery_level'):
            agv.battery_level = min(1.0, agv.battery_level + self.charging_rate)
        return True
