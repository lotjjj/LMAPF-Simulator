"""Path validation utilities for verifying planned agent paths against the grid map."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..envs.entities import Position, AGV, Action


@dataclass
class PathValidationResult:
    is_valid: bool
    invalid_agent: Optional[int] = None
    invalid_step_index: Optional[int] = None
    invalid_move: Optional[Tuple[Position, Position]] = None
    reason: Optional[str] = None


class PathValidator:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.height = len(grid_map)
        self.width = len(grid_map[0]) if self.height > 0 else 0

    def _is_valid_position(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def _is_passable(self, pos: Position) -> bool:
        if not self._is_valid_position(pos):
            return False
        return self.grid_map[pos.y][pos.x].passable

    def _is_shelf_to_shelf(self, from_pos: Position, to_pos: Position) -> bool:
        from ..envs.entities import Shelf
        if not self._is_valid_position(from_pos) or not self._is_valid_position(to_pos):
            return False
        from_grid = self.grid_map[from_pos.y][from_pos.x]
        to_grid = self.grid_map[to_pos.y][to_pos.x]
        return isinstance(from_grid, Shelf) and isinstance(to_grid, Shelf)

    def _is_adjacent_move(self, from_pos: Position, to_pos: Position) -> bool:
        dx = abs(to_pos.x - from_pos.x)
        dy = abs(to_pos.y - from_pos.y)
        return (dx == 0 and dy == 1) or (dx == 1 and dy == 0) or (dx == 0 and dy == 0)

    def validate_move(self, from_pos: Position, to_pos: Position) -> Tuple[bool, Optional[str]]:
        if from_pos == to_pos:
            return True, None
            
        if not self._is_adjacent_move(from_pos, to_pos):
            return False, f"Not adjacent: from {from_pos} to {to_pos}"
            
        if not self._is_passable(to_pos):
            return False, f"Target position {to_pos} is not passable"
            
        if self._is_shelf_to_shelf(from_pos, to_pos):
            return False, f"Shelf to shelf move: {from_pos} to {to_pos}"
            
        return True, None

    def validate_path(self, start_pos: Position, path: List[Position]) -> Tuple[bool, Optional[int], Optional[str]]:
        if not path:
            return True, None, None
            
        if len(path) == 1:
            return path[0] == start_pos, 0 if path[0] != start_pos else None, None

        for i in range(len(path) - 1):
            from_pos = path[i]
            to_pos = path[i + 1]
            is_valid, reason = self.validate_move(from_pos, to_pos)
            if not is_valid:
                return False, i + 1, reason
                
        return True, None, None

    def validate_agent_paths(self, agvs: Dict[int, AGV], paths: Dict[int, List[Position]]) -> Dict[int, PathValidationResult]:
        results = {}
        for agent_id, path in paths.items():
            if agent_id not in agvs:
                results[agent_id] = PathValidationResult(
                    is_valid=False,
                    reason=f"Agent {agent_id} not in agvs dict"
                )
                continue
                
            agv = agvs[agent_id]
            is_valid, invalid_step, reason = self.validate_path(agv.position, path)
            
            if not is_valid:
                invalid_move = None
                if invalid_step is not None and 0 < invalid_step < len(path):
                    invalid_move = (path[invalid_step - 1], path[invalid_step])
                    
                results[agent_id] = PathValidationResult(
                    is_valid=False,
                    invalid_agent=agent_id,
                    invalid_step_index=invalid_step,
                    invalid_move=invalid_move,
                    reason=reason
                )
            else:
                results[agent_id] = PathValidationResult(is_valid=True)
                
        return results

    def check_first_step_invalid(self, agvs: Dict[int, AGV], paths: Dict[int, List[Position]]) -> Dict[int, PathValidationResult]:
        results = {}
        for agent_id, path in paths.items():
            if agent_id not in agvs:
                results[agent_id] = PathValidationResult(
                    is_valid=False,
                    reason=f"Agent {agent_id} not in agvs dict"
                )
                continue
                
            agv = agvs[agent_id]
            
            if len(path) < 2:
                results[agent_id] = PathValidationResult(is_valid=True)
                continue
                
            from_pos = agv.position
            to_pos = path[1]
            
            if from_pos != path[0]:
                results[agent_id] = PathValidationResult(
                    is_valid=False,
                    invalid_agent=agent_id,
                    invalid_step_index=0,
                    reason=f"Path start {path[0]} doesn't match agv current position {from_pos}"
                )
                continue
                
            is_valid, reason = self.validate_move(from_pos, to_pos)
            
            if not is_valid:
                results[agent_id] = PathValidationResult(
                    is_valid=False,
                    invalid_agent=agent_id,
                    invalid_step_index=1,
                    invalid_move=(from_pos, to_pos),
                    reason=reason
                )
            else:
                results[agent_id] = PathValidationResult(is_valid=True)
                
        return results

