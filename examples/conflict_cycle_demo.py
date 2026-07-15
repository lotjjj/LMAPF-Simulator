from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LMAPFEnv.envs import MapConfig, PRESET_MAPS, WarehouseEnv
from LMAPFEnv.envs.entities import Action, Position

DEMO_MAP_NAME = "conflict_cycle_demo"
DEMO_RING_POSITIONS = [
    Position(1, 1),
    Position(2, 1),
    Position(3, 1),
    Position(3, 2),
    Position(3, 3),
    Position(2, 3),
    Position(1, 3),
    Position(1, 2),
]


def register_demo_map() -> None:
    PRESET_MAPS[DEMO_MAP_NAME] = MapConfig(
        shelf_cols=1,
        shelf_rows=1,
        shelf_width=1,
        shelf_height=1,
        corridor_width=1,
        corridor_out_width=1,
    )


def resize_renderer(env: WarehouseEnv, cell_size: int) -> None:
    if env.render_mode != "human" or not hasattr(env, "_main_window"):
        return

    import pygame

    renderer = env._main_window.warehouse_widget
    renderer.cell_size = int(max(20, cell_size))
    renderer.window_width = env.width * renderer.cell_size
    renderer.window_height = env.height * renderer.cell_size
    renderer.screen = pygame.display.set_mode((renderer.window_width, renderer.window_height))
    renderer.font = pygame.font.SysFont("Arial", max(16, renderer.cell_size // 2))
    renderer._trajectory_surface = None
    renderer._build_grid_surface()


def hide_task_markers(env: WarehouseEnv) -> None:
    if env.render_mode == "human" and hasattr(env, "_main_window"):
        env._main_window.warehouse_widget._draw_task_targets = lambda: None
    elif env.render_mode == "rgb_array":
        env.render()
        if hasattr(env, "_rgb_widget") and env._rgb_widget is not None:
            env._rgb_widget._draw_task_targets = lambda: None


def place_ring_agents(env: WarehouseEnv, ring_positions: Iterable[Position]) -> None:
    ordered_agents = sorted(env.agvs.keys(), key=lambda name: int(name.split("_")[1]))
    for agent_name, pos in zip(ordered_agents, ring_positions, strict=True):
        env.teleport_agv(agent_name, pos.x, pos.y)
        env.agvs[agent_name].target_pos = pos


def build_cycle_successors() -> dict[Position, Position]:
    successors: dict[Position, Position] = {}
    for index, pos in enumerate(DEMO_RING_POSITIONS):
        successors[pos] = DEMO_RING_POSITIONS[(index + 1) % len(DEMO_RING_POSITIONS)]
    return successors


def action_from_positions(current: Position, target: Position) -> int:
    dx = target.x - current.x
    dy = target.y - current.y
    if dx == 1 and dy == 0:
        return Action.RIGHT
    if dx == -1 and dy == 0:
        return Action.LEFT
    if dx == 0 and dy == 1:
        return Action.DOWN
    if dx == 0 and dy == -1:
        return Action.UP
    return Action.STAY


def select_ring_actions(env: WarehouseEnv, successors: dict[Position, Position]) -> dict[str, int]:
    actions: dict[str, int] = {}
    for agent_name in env.agents:
        current = env.agvs[agent_name].position
        target = successors[current]
        actions[agent_name] = action_from_positions(current, target)
    return actions


def draw_overlay(env: WarehouseEnv, step_idx: int, conflicted_agents: list[str]) -> None:
    if env.render_mode != "human" or not hasattr(env, "_main_window"):
        return

    import pygame

    renderer = env._main_window.warehouse_widget
    screen = renderer.screen
    font = pygame.font.SysFont("Arial", max(18, renderer.cell_size // 2), bold=True)
    lines = [
        f"Step {step_idx}",
        "Conflict Check Demo: long cycle is committed",
        f"Conflicted agents: {conflicted_agents if conflicted_agents else 'None'}",
    ]

    y = 8
    for text in lines:
        surface = font.render(text, True, (20, 20, 20))
        padding = 4
        background = pygame.Surface(
            (surface.get_width() + padding * 2, surface.get_height() + padding * 2),
            pygame.SRCALPHA,
        )
        background.fill((255, 255, 255, 220))
        screen.blit(background, (8, y))
        screen.blit(surface, (8 + padding, y + padding))
        y += surface.get_height() + padding * 2 + 4

    pygame.display.flip()


def run_demo(seed: int, laps: int, sleep_s: float, headless: bool, cell_size: int) -> None:
    register_demo_map()
    render_mode = "rgb_array" if headless else "human"
    env = WarehouseEnv(
        num_agvs=len(DEMO_RING_POSITIONS),
        map_size=DEMO_MAP_NAME,
        path_planner=None,
        render_mode=render_mode,
        max_episode_steps=max(64, laps * len(DEMO_RING_POSITIONS) + 8),
        if_continuous=not headless,
    )

    try:
        env.reset(seed=seed)
        place_ring_agents(env, DEMO_RING_POSITIONS)
        hide_task_markers(env)
        resize_renderer(env, cell_size)

        successors = build_cycle_successors()
        total_steps = max(1, laps) * len(DEMO_RING_POSITIONS)

        if not headless:
            env.render()
            draw_overlay(env, step_idx=0, conflicted_agents=[])
            time.sleep(max(0.0, sleep_s))

        for step_idx in range(1, total_steps + 1):
            if not headless and hasattr(env, "_main_window") and not env._main_window.running:
                break

            actions = select_ring_actions(env, successors)
            _, _, _, _, infos = env.step(actions)
            conflicted_agents = [agent for agent, info in infos.items() if info["conflicted"]]

            positions = {agent: env.agvs[agent].position.to_tuple() for agent in sorted(env.agents)}
            print(
                f"step={step_idx:02d} "
                f"conflicted={conflicted_agents if conflicted_agents else 'None'} "
                f"positions={positions}"
            )

            if not headless:
                draw_overlay(env, step_idx=step_idx, conflicted_agents=conflicted_agents)
                time.sleep(max(0.0, sleep_s))
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize multi-AGV cycle movement under the environment conflict check."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--laps", type=int, default=4, help="Number of full clockwise ring laps.")
    parser.add_argument("--sleep", type=float, default=0.4, help="Seconds to pause between steps.")
    parser.add_argument("--cell-size", type=int, default=80, help="Human render cell size in pixels.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a pygame window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    run_demo(
        seed=int(args.seed),
        laps=max(1, int(args.laps)),
        sleep_s=max(0.0, float(args.sleep)),
        headless=bool(args.headless),
        cell_size=int(args.cell_size),
    )


if __name__ == "__main__":
    main()
