from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LMAPFEnv.algorithms.path_planners import PlannerPolicy
from LMAPFEnv.envs import WarehouseEnv
from LMAPFEnv.utils import parse_planner_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple planner-driven warehouse simulation demo.")
    parser.add_argument("--num-agvs", type=int, default=20)
    parser.add_argument("--map-size", type=str, default="long")
    parser.add_argument("--path-planner", type=str, default="RHCR")
    parser.add_argument("--planner-args", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--continuous", type=bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = WarehouseEnv(
        num_agvs=int(args.num_agvs),
        map_size=args.map_size,
        path_planner=args.path_planner,
        planner_args=parse_planner_args(args.planner_args) or None,
        render_mode=args.render_mode,
        if_continuous=bool(args.continuous),
    )

    total_tasks_completed = 0
    total_conflicts = 0
    episode_count = 0

    try:
        _, infos = env.reset(seed=int(args.seed))
        policy = PlannerPolicy(env.path_planner)
        policy.update_info(infos)

        for step_idx in range(max(0, int(args.steps))):
            actions = policy.select_actions(env.agvs, env.agents, infos)
            _, _, _, _, infos = env.step(actions)
            policy.update_info(infos)

            # Count task completions and conflicts from info
            for agent_name, agent_info in infos.items():
                if agent_info.get('task_completed', False):
                    total_tasks_completed += 1
                if agent_info.get('conflicted', False):
                    total_conflicts += 1

            time.sleep(max(0.0, float(args.sleep)))
            if not env.agents:
                episode_count += 1
                _, infos = env.reset(seed=int(args.seed) + step_idx + 1)
                policy.update_info(infos)

        print(f"\n=== Simulation Summary ===")
        print(f"Steps executed: {step_idx + 1}")
        print(f"Episodes: {episode_count + 1}")
        print(f"Tasks completed: {total_tasks_completed}")
        print(f"Total conflicts: {total_conflicts}")
        print(f"==========================")
    finally:
        env.close()


if __name__ == "__main__":
    main()
