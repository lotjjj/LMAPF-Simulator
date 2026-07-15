from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LMAPFEnv.utils import normalize_map_sizes, parse_planner_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="renders/maps")
    parser.add_argument("--map-sizes", nargs="*", default=None)
    parser.add_argument("--num-agvs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path-planner", type=str, default=None)
    parser.add_argument("--planner-args", type=str, default=None)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--hide-trajectories", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from LMAPFEnv.envs import WarehouseEnv
    import pygame

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    planner_args = parse_planner_args(args.planner_args)
    map_sizes = normalize_map_sizes(args.map_sizes)

    for map_size in map_sizes:
        env = WarehouseEnv(
            num_agvs=int(args.num_agvs),
            map_size=map_size,
            render_mode="rgb_array",
            path_planner=args.path_planner,
            planner_args=planner_args or None,
        )
        env.reset(seed=int(args.seed))

        for _ in range(max(0, int(args.steps))):
            actions = {agent: 4 for agent in env.agents}
            env.step(actions)

        env.render()
        if args.hide_trajectories and hasattr(env, "_rgb_widget") and env._rgb_widget is not None:
            env._rgb_widget.set_trajectory_visibility(False)
            env.render()

        filename = f"warehouse_{map_size}_{env.width}x{env.height}.png"
        path = outdir / filename
        pygame.image.save(env._rgb_widget.screen, str(path))
        env.close()


if __name__ == "__main__":
    main()
