"""Benchmark all 6 planners with optimized parameters.

Usage: python run_benchmark.py [--planner PLANNER]
"""

import argparse
import sys
import time as time_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LMAPFEnv.envs import WarehouseEnv
from LMAPFEnv.algorithms.path_planners import PlannerPolicy

PLANNERS = ["AStar", "PBS", "RHCR", "RHCR_PBS"]
DEFAULT_STEPS = 200
DEFAULT_PLOT = PROJECT_ROOT / "docs" / "runtime_benchmark.png"

# Optimized per-planner config
PLANNER_CONFIGS = {
    "CBS":        {"max_cbs_nodes": 1000, "max_low_level_steps": 200, "max_planning_time": 20.0},
    "ECBS":       {"max_cbs_nodes": 1000, "max_low_level_steps": 200, "max_planning_time": 20.0, "w": 1.5},
    "AStar":      {"max_low_level_steps": 200},
    "PBS":        {"max_pbs_nodes": 1000, "max_low_level_steps": 200, "max_planning_time": 20.0},
    "RHCR":       {"planning_window": 10, "horizon": 5, "max_low_level_steps": 200},
    "RHCR_CBS":   {"planning_window": 10, "horizon": 5, "max_cbs_nodes": 1000, "max_low_level_steps": 200, "max_planning_time": 20.0},
    "RHCR_PBS":   {"planning_window": 10, "horizon": 5, "max_pbs_nodes": 1000, "max_low_level_steps": 200, "max_planning_time": 20.0},
    "RHCR_ECBS":  {"planning_window": 10, "horizon": 5, "max_cbs_nodes": 1000, "max_low_level_steps": 200, "max_planning_time": 20.0, "w": 1.5},
}

def _pm(infos: dict, agent: str, key: str):
    pm = infos.get(agent, {}).get("planner_meta", {})
    return pm.get(key, None)


def _planner_time_ms(infos: dict, agent: str) -> float:
    pm = infos.get(agent, {}).get("planner_meta", {})
    if pm.get("skipped", False):
        return 0.0
    return float(pm.get("time_ms", 0.0) or 0.0)


def _has_public_path(infos: dict, agent: str) -> bool:
    planner_paths = infos.get(agent, {}).get("planner_paths", {})
    if not planner_paths.get("has_path", False):
        return False
    path_abs = planner_paths.get("path_abs")
    return path_abs is not None and len(path_abs) >= 2


def _count_public_paths(infos: dict, agents) -> int:
    return sum(1 for agent in agents if _has_public_path(infos, agent))


def run_single_planner(
    planner_type: str,
    seed: int = 42,
    max_steps: int = DEFAULT_STEPS,
    no_path_patience: int = 2,
) -> dict:
    planner_args = PLANNER_CONFIGS[planner_type]
    print(f"\n{'='*60}")
    print(f"  Planner: {planner_type}")
    print(f"  args: {planner_args}")
    print(f"{'='*60}")

    t0 = time_module.perf_counter()
    env = WarehouseEnv(
        num_agvs=10, map_size="s",
        path_planner=planner_type, planner_args=planner_args,
        render_mode=None,
        max_episode_steps=max_steps,
    )
    obs, info = env.reset(seed=seed)
    policy = PlannerPolicy(env.path_planner)
    policy.update_info(info)

    total_conflicts = 0
    total_completions = 0
    max_plan = 0.0
    total_plan = 0.0
    plan_calls = 0
    disabled = False
    disabled_step = -1
    exit_reason = "max_steps"
    steps_run = 0
    no_path_streak = 0
    conflict_detail: list = []

    last_info = info
    initial_pm = max((_planner_time_ms(info, a) for a in info), default=0.0)
    if initial_pm > 0:
        total_plan += initial_pm
        plan_calls += 1
        max_plan = max(max_plan, initial_pm)

    for step_idx in range(max_steps):
        if not env.agents:
            exit_reason = "max_steps" if steps_run >= max_steps else "no_active_agents"
            break

        path_count = _count_public_paths(last_info, env.agents)
        if path_count == 0:
            no_path_streak += 1
            if no_path_streak >= max(1, no_path_patience):
                exit_reason = "no_paths"
                print(f"    stop at step {step_idx}: planner exposed no paths")
                break
        else:
            no_path_streak = 0

        actions = policy.select_actions(env.agvs, env.agents, last_info)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        policy.update_info(infos)
        last_info = infos
        steps_run = step_idx + 1

        if not infos:
            exit_reason = "max_steps" if steps_run >= max_steps else "no_active_agents"
            break

        info_agents = list(infos.keys())
        step_conflicts = sum(1 for a in info_agents if infos.get(a, {}).get("conflicted", False))
        total_conflicts += step_conflicts
        total_completions += sum(1 for a in info_agents if infos.get(a, {}).get("task_completed", False))

        if step_conflicts > 0 and len(conflict_detail) < 10:
            conf_agents = [a for a in info_agents if infos.get(a, {}).get("conflicted", False)]
            conf_pos = {a: (env.agvs[a].x, env.agvs[a].y) for a in conf_agents}
            # What action was each conflicted agent trying to take?
            conf_actions = {}
            for a in conf_agents:
                if a in actions:
                    act_names = ["UP","DOWN","LEFT","RIGHT","STAY"]
                    conf_actions[a] = act_names[actions[a]]
            conflict_detail.append((step_idx, step_conflicts, conf_agents[:5], conf_pos, conf_actions))

        pm = max((_planner_time_ms(infos, a) for a in info_agents), default=0.0)
        if pm > 0:
            total_plan += pm
            plan_calls += 1
            max_plan = max(max_plan, pm)

        for a in info_agents:
            d = _pm(infos, a, "disabled")
            if d and not disabled:
                disabled = True
                disabled_step = step_idx

        if (step_idx + 1) % 50 == 0:
            print(f"    step {step_idx+1:>3d}: completions={total_completions:>3d} "
                  f"conflicts={sum(1 for a in info_agents if infos.get(a,{}).get('conflicted',False)):>2d} "
                  f"plan={pm:>8.1f}ms")

        if not env.agents:
            exit_reason = "max_steps" if steps_run >= max_steps else "no_active_agents"
            break

    elapsed = time_module.perf_counter() - t0
    env.close()

    return {
        "planner": planner_type,
        "steps_run": steps_run,
        "total_completions": total_completions,
        "total_conflicts": total_conflicts,
        "avg_plan_ms": total_plan / max(1, plan_calls),
        "max_plan_ms": max_plan,
        "elapsed_s": elapsed,
        "disabled": disabled,
        "disabled_step": disabled_step,
        "exit_reason": exit_reason,
        "conflict_detail": conflict_detail,
    }


def generate_runtime_plot(results: list[dict], output_path: Path) -> None:
    if not results:
        return

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r["planner"] for r in results]
    elapsed = [float(r["elapsed_s"]) for r in results]
    avg_plan = [float(r["avg_plan_ms"]) for r in results]
    max_plan_vals = [float(r["max_plan_ms"]) for r in results]
    colors = ["#2a9d8f" if r["exit_reason"] == "max_steps" and not r["disabled"] else "#e76f51" for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, elapsed, color=colors)
    axes[0].set_title("Wall-clock runtime")
    axes[0].set_ylabel("seconds")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, avg_plan, color=colors, label="avg")
    axes[1].plot(names, max_plan_vals, color="#264653", marker="o", linewidth=1.5, label="max")
    axes[1].set_title("Planner call time")
    axes[1].set_ylabel("milliseconds")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(frameon=False)

    fig.suptitle("LMAPF runtime benchmark: 10 AGVs, small map, 200 steps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", type=str, default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-path-patience", type=int, default=2)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    planners = [args.planner] if args.planner else PLANNERS
    results = []
    for pt in planners:
        if pt not in PLANNERS:
            print(f"Unknown: {pt}")
            continue
        r = run_single_planner(
            pt,
            seed=args.seed,
            max_steps=args.steps,
            no_path_patience=args.no_path_patience,
        )
        results.append(r)

    print("\n")
    print("=" * 100)
    hdrs = ["Planner", "Steps", "Completions", "Conflicts", "AvgPlan(ms)", "MaxPlan(ms)", "Elapsed(s)", "Disabled", "Exit"]
    print(f"{hdrs[0]:<14} {hdrs[1]:>6} {hdrs[2]:>12} {hdrs[3]:>10} {hdrs[4]:>12} {hdrs[5]:>12} {hdrs[6]:>10} {hdrs[7]:>9} {hdrs[8]:>14}")
    print("-" * 100)
    for r in results:
        ds = f"Y@{r['disabled_step']}" if r['disabled'] else "N"
        print(f"{r['planner']:<14} {r['steps_run']:>6} {r['total_completions']:>12} {r['total_conflicts']:>10} "
              f"{r['avg_plan_ms']:>12.1f} {r['max_plan_ms']:>12.1f} {r['elapsed_s']:>10.1f} {ds:>9} {r['exit_reason']:>14}")
    print("=" * 100)

    print("\n--- PASS/FAIL ---")
    for r in results:
        ok = (
            r["exit_reason"] == "max_steps"
            and r["total_conflicts"] == 0
            and r["total_completions"] >= 80
            and not r["disabled"]
        )
        detail = f"  conflicts at: {r['conflict_detail']}" if r["conflict_detail"] else ""
        print(f"  {r['planner']:<14}: {'PASS' if ok else 'FAIL'}  "
              f"(comp={r['total_completions']}, conf={r['total_conflicts']}, "
              f"disabled={'Y' if r['disabled'] else 'N'}, exit={r['exit_reason']}){detail}")

    if not args.no_plot:
        generate_runtime_plot(results, args.plot)
        print(f"\nRuntime plot saved to: {args.plot}")


if __name__ == "__main__":
    main()
