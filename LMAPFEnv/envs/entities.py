from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple, Dict, List, Optional, Iterator, Union

from gymnasium.spaces import Box, Discrete, Dict
import numpy as np


class Action(IntEnum):
    """AGV movement actions"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    

class Direction(IntEnum):
    """Movement directions with delta vectors"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def get_delta(self):
        """Get movement delta for direction"""
        if self == Direction.UP:
            return 0, -1
        elif self == Direction.DOWN:
            return 0, 1
        elif self == Direction.LEFT:
            return -1, 0
        else:
            return 1, 0


@dataclass(frozen=True)
class Position:
    """Immutable position in 2D grid"""
    x: int
    y: int

    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y))

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Position index out of range")

    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, tuple):
            if len(other) == 2:
                return self.x == other[0] and self.y == other[1]
        return False

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __lt__(self, other: 'Position') -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __le__(self, other: 'Position') -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) <= (other.x, other.y)

    def __gt__(self, other: 'Position') -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) > (other.x, other.y)

    def __ge__(self, other: 'Position') -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) >= (other.x, other.y)

    def __repr__(self) -> str:
        return f"Position({self.x}, {self.y})"

    def to_tuple(self) -> Tuple[int, int]:
        return self.x, self.y

class AGV:
    """Automated Guided Vehicle entity.
    
    Stores _x and _y as plain ints; the `position` property creates a
    Position only when needed.
    """
    
    STATUS_COLORS = {
        "IDLE": (220, 53, 69, 255),
        "MOVING": (13, 110, 253, 255),
        "WORKING": (25, 135, 84, 255)
    }

    AGV_COLORS = [
        (13, 110, 253),
        (220, 53, 69),
        (25, 135, 84),
    ]

    __slots__ = (
        'id', '_x', '_y', 'target_pos', 'req_action', 'status',
        'path', 'fov_size', 'path_window',
    )

    def __init__(self, id: int, start_pos: Tuple[int, int], fov_size: int = 5, path_window: int = 3, ksteps: Optional[int] = None):
        self.id = id
        if isinstance(start_pos, tuple):
            self._x, self._y = start_pos[0], start_pos[1]
        else:
            self._x, self._y = start_pos.x, start_pos.y
        self.target_pos = None
        self.req_action = Action.STAY
        self.status = "IDLE"
        self.path = []
        self.fov_size = fov_size
        # Backward compat: ksteps is deprecated alias for path_window
        if ksteps is not None:
            import warnings
            warnings.warn("AGV(ksteps=...) is deprecated, use path_window=...", DeprecationWarning, stacklevel=2)
            path_window = ksteps
        self.path_window = path_window

    @property
    def action_space(self):
        """Action space for AGV"""
        return Discrete(5)

    def observation_space(self):
        self_states_dict = {
            "position": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "fov_density": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "target_rel": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "target_visible": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "target_dist_norm": Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }

        self_states_space = Dict(self_states_dict)

        fov_num_channels = 5
        fov_space = Box(low=0, high=1, shape=(fov_num_channels, self.fov_size, self.fov_size), dtype=np.float32)

        obs_space = Dict({
            "self_states": self_states_space,
            "fov": fov_space
        })

        return obs_space

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int):
        self._x = value

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, value: int):
        self._y = value

    @property
    def position(self) -> Position:
        """Lazily create Position — hot paths should use .x / .y directly."""
        return Position(self._x, self._y)

    @position.setter
    def position(self, value):
        if isinstance(value, tuple):
            self._x, self._y = value[0], value[1]
        else:
            self._x, self._y = value.x, value.y

    def set_position(self, x: int, y: int):
        """Set new position using integer coordinates (no Position created)."""
        self._x = x
        self._y = y

    def reset(self, x: int, y: int):
        """Reset AGV to initial state"""
        self._x = x
        self._y = y
        self.target_pos = None
        self.req_action = Action.STAY
        self.status = "IDLE"
        self.path = []


class TaskStatus(Enum):
    """Task status enumeration"""
    ACTIVE = 0
    COMPLETED = 1
    ABANDONED = 2


class Task:
    """Individual task for AGV"""
    
    def __init__(self, target_pos: Union[Tuple[int, int], Position], status: TaskStatus = TaskStatus.ACTIVE):
        if isinstance(target_pos, tuple):
            self.target_pos = Position(*target_pos)
        else:
            self.target_pos = target_pos
        self.status = status
        self.assigned_agent = None


class Grid:
    """Base grid cell class"""
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.occupiable = True
        self.passable = True
        self.render_color = (255, 255, 255)
        self.agvs: set = set()  # Use set for O(1) add/remove

    def add_agv(self, agv):
        """Add AGV to this grid cell"""
        self.agvs.add(agv)

    def remove_agv(self, agv):
        """Remove AGV from this grid cell"""
        self.agvs.discard(agv)  # discard doesn't raise if not found

    def on_enter(self, agv):
        """Called when AGV enters this cell"""
        return True

    def on_occupy(self, agv):
        """Called when AGV occupies this cell"""
        return True

    def on_leave(self, agv):
        """Called when AGV leaves this cell"""
        return True


class Wall(Grid):
    """Wall obstacle entity"""
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.occupiable = False
        self.passable = False
        self.render_color = (108, 117, 125)


class Shelf(Grid):
    """Shelf storage entity"""
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.occupiable = True
        self.passable = True
        self.render_color = (184, 115, 51)


class Corridor(Grid):
    """Corridor path entity"""
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.occupiable = True
        self.passable = True
        self.render_color = (248, 249, 250)
