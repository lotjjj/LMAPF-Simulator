# LMAPF-Simulator

[中文文档](./README.zh.md) | [Italiano](./README.it.md)

A warehouse multi-AGV simulation environment for **Lifelong Multi-Agent Path Finding (LMAPF/MAPF)** research. Built on PettingZoo `ParallelEnv` and Gymnasium, featuring pluggable planners and rolling-horizon variants.

<video src="./renders/episode_demo.mp4" controls muted loop playsinline></video>

[Watch the MP4 demo](./renders/episode_demo.mp4)

Generate this compact MP4 demo locally:

```bash
python tools/save_episode_video.py
```

Use `--sleep` and `--fps` to tune movement smoothness and playback duration.

Forked from [RWARE](https://github.com/semitable/robotic-warehouse). Some planner implementations in this repo are experimental.

## Features

- **PettingZoo `ParallelEnv`** interface
- **Built-in warehouse maps**: `small`, `medium`, `large`, `long` presets
- **Continuous task queue**: new tasks auto-assigned on arrival
- **Conflict resolution**: the environment's directed-graph step resolves runtime conflicts without planner intervention, powered by a **C++ FastGraph engine**
- **Pluggable planners**: A*, CBS, ECBS, PBS, RHCR and their variants
- **Visualization**: map/FOV rendering, conflict cycle demos

## Install

### 1. Install a C++ compiler

The C++ FastGraph engine and A\* engine are **auto-compiled** on first import. You'll need a C++ compiler:

| Platform | Compiler | How to install |
|----------|----------|---------------|
| **Windows** | MSVC (Visual Studio) | Install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022), select "Desktop development with C++" in the installer |
| **Linux** | GCC ≥ 10 | `sudo apt install build-essential cmake` (Ubuntu/Debian) |
| **macOS** | Clang (Xcode) | `xcode-select --install` |

MinGW (Windows) or a system package manager's GCC also works.

### 2. Create a conda environment (recommended)

```bash
conda create -n lmapf python=3.12 -y
conda activate lmapf
```

### 3. Clone & install

```bash
git clone https://github.com/lotjjj/LMAPF-Simulator.git
cd LMAPF-Simulator
pip install -e .
```

Key dependencies: `numpy`, `pettingzoo`, `gymnasium`, `pygame`, `matplotlib`, `imageio[ffmpeg]`.

### 4. Verify installation

```python
from LMAPFEnv import WarehouseEnv
from LMAPFEnv.algorithms.path_planners import _HAS_CXX_ASTAR
print("C++ A* engine:", "enabled" if _HAS_CXX_ASTAR else "not available")
```

The first `import LMAPFEnv` will auto-detect your C++ compiler and build the FastGraph engine (~30-60 s). To compile manually:

```bash
python build_cpp_graph.py
```

## Quick Start

### 1. Create environment

```python
from LMAPFEnv import WarehouseEnv

env = WarehouseEnv(
    num_agvs=6,
    map_size="medium",
    render_mode=None,
    max_episode_steps=500,
)
observations, infos = env.reset(seed=42)
```

### 2. Run with random actions

```python
done = False
while not done:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    done = all(terminations[a] or truncations[a] for a in env.agents)
env.close()
```

### 3. Run with a planner

```python
from LMAPFEnv import WarehouseEnv
from LMAPFEnv.algorithms.path_planners import PlannerPolicy

env = WarehouseEnv(
    num_agvs=10,
    map_size="long",
    path_planner="RHCR",
    planner_args={"planning_window": 10, "horizon": 3},
    render_mode="human",
)
obs, infos = env.reset(seed=42)
policy = PlannerPolicy(env.path_planner)

for _ in range(200):
    actions = policy.select_actions(env.agvs, env.agents)
    obs, rewards, terminations, truncations, infos = env.step(actions)
    if not env.agents:
        break
env.close()
```

### 4. Run benchmark

```bash
# All 6 planners
python run_benchmark.py

# Single planner
python run_benchmark.py --planner RHCR_PBS
```

## Performance

![runtime-benchmark](docs/runtime_benchmark.png)

Runtime measured with `python run_benchmark.py --steps 200` on the `long` map (10 AGVs, no rendering, seed 42, random task assignment). Elapsed time includes environment setup, reset, initial planning, and the 200-step rollout. Average/max planner time counts actual planner calls only; skipped steps are excluded.

| Planner | Steps | Completions | Conflicts | Avg plan (ms) | Max plan (ms) | Elapsed (s) |
|---------|------:|------------:|----------:|--------------:|--------------:|------------:|
| CBS | 200 | 137 | 0 | 4421.2 | 6886.0 | 79.8 |
| ECBS | 200 | 138 | 0 | 13432.3 | 21280.2 | 255.4 |
| PBS | 200 | 137 | 0 | 22338.7 | 27701.2 | 402.4 |
| RHCR_CBS | 200 | 137 | 0 | 605.5 | 2082.8 | 122.3 |
| RHCR_PBS | 200 | 126 | 0 | 87.9 | 500.2 | 18.1 |
| RHCR_ECBS | 200 | 133 | 0 | 211.0 | 2628.3 | 43.0 |

_All runs completed 200 steps with no disabled/no-path exit. Measured with the C++ FastGraph engine and C++ A* planner engine enabled on Python 3.12.13._

## Examples & Tools

| Command | Description |
|---------|-------------|
| `python examples/run_planner_demo.py --path-planner RHCR --continuous` | Planner-driven interactive demo |
| `python examples/conflict_cycle_demo.py` | Conflict cycle demonstration |
| `python tools/save_map_renders.py --map-sizes small medium large long` | Export map renders |
| `python tools/save_agent_fov_renders.py --path-planner RHCR --steps 5` | Export FOV renders |
| `python tools/save_episode_video.py` | Record a compact MP4 episode for README demos |

## Planners

| Type | Planners |
|------|----------|
| Single-agent | `AStar`, `EnhancedAStar` |
| MAPF | `CBS`, `ECBS`, `PBS` |
| Rolling-horizon | `RHCR`, `RHCR_CBS`, `RHCR_ECBS`, `RHCR_PBS` |

Common `planner_args`: `shelf_penalty`, `max_cbs_nodes`, `max_pbs_nodes`, `max_low_level_steps`, `max_planning_time`, `w`, `visible_agv_penalty`, `planning_window`, `horizon`.

## Environment API

### Core parameters

`WarehouseEnv(num_agvs, map_size, fov_size, max_episode_steps, render_mode, path_planner, planner_args)`

### Task generation

- Tasks are generated only on `shelf` cells.
- Each alive AGV keeps `num_visible_tasks` active queued targets, defaulting to `2`.
- A map is valid only when `num_agvs * num_visible_tasks <= shelf_count`; otherwise environment creation raises `ValueError`.
- Initial AGV placement prefers corridor cells so shelf targets remain available for task generation.

### Action space

Each agent uses `Discrete(5)` with fixed order:

```python
[UP, DOWN, LEFT, RIGHT, STAY]
```

### Observation

`obs` is a `gymnasium.spaces.Dict`:

```python
obs = Dict({
    "self_states": Dict({...}),
    "fov": Box(...),
})
```

`self_states` always contains:

| Field | Shape | Meaning |
|-------|-------|---------|
| `position` | `(2,)` | Normalized global position |
| `fov_density` | `(1,)` | Density of other AGVs in FOV |
| `target_rel` | `(2,)` | Normalized offset to current target |
| `target_visible` | `(1,)` | Whether target is inside FOV |
| `target_dist_norm` | `(1,)` | Normalized Euclidean distance to target |

`fov` has shape `(5, fov_size, fov_size)` with channels:
`corridor`, `wall/OOB`, `shelf`, `other AGVs`, `visible goals`.

### Info

Both `reset()` and `step()` return `infos[agent_name]`. The most useful fields are:

| Key | Meaning |
|-----|---------|
| `action_mask` | Local feasibility mask for `[UP, DOWN, LEFT, RIGHT, STAY]` |
| `conflicted` | Agent was force-stayed by conflict resolution |
| `invalid_action` | Requested action was illegal and replaced by `STAY` |
| `task_completed` | Agent reached its current task target in this step |
| `progress_target_pos` | Reward-reference target used for this step |
| `progress_distance_prev` | Distance to that target before execution |
| `progress_distance_now` | Distance to that target after execution |
| `act_val_time_ms` | Action validation + conflict-resolution time |
| `planner_meta` | Planner timing / timeout / disable diagnostics |
| `planner_paths` | Shared cached path snapshot for all agents |

Notes:

- `action_mask` only captures local validity; an action can still end up not executing if `conflicted=True`.
- `planner_paths` is a global snapshot, so it is usually identical across all `infos[agent]`.

### Reward-target timing

When an agent finishes a task in the current step:

1. The environment freezes `progress_target_pos` for reward accounting.
2. It computes `progress_distance_prev/now` against that frozen target.
3. It marks `task_completed=True`.
4. It advances the task queue so `next task` becomes the new `current task`.

This means:

- `progress_target_pos` still points to the just-completed old target.
- `progress_distance_prev/now` are also measured against that old target.
- The next observation already reflects the new `current task` through `agv.target_pos`.

### Termination and truncation

- `terminations[agent]` is always `False`.
- `truncations[agent]` becomes `True` for all alive agents when `max_episode_steps` is reached.
- The environment also truncates all alive agents if more than half of them remain at the same position over the congestion window.
- Agents marked terminated or truncated in a step are removed from `env.agents` before the next step.

### Default reward

```python
reward = each_step_reward
       + invalid_action_penalty
       + conflict_penalty
       + progress_shaping_weight * clip(d_prev - d_now, -1, 1)
       + task_completion_reward
```

Default values:

- `each_step_reward = -0.002`
- `invalid_action_penalty = -0.05`
- `conflict_penalty = -0.6`
- `progress_shaping_weight = 0.01`
- `task_completion_reward = +2.0`

## License

MIT License
