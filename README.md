# LMAPF-Simulator

A Multi-Agent Path Finding Simulator for warehouse AGV environments, built with PettingZoo and Gymnasium.

## Environment Requirements

### System Requirements
- **Python**: >= 3.10
- **Operating System**: Windows, macOS, or Linux

### Dependencies
- numpy >= 1.21.0
- networkx >= 2.6.0
- pettingzoo >= 1.22.0
- gymnasium >= 0.26.0
- PySide6 >= 6.4.0 (for GUI rendering, optional)

## Installation

### Install from Source (Development Mode)

```bash
# Clone the repository
git clone https://github.com/yourusername/LMAPF-Simulator.git
cd LMAPF-Simulator

# Install in development mode
pip install -e .
```

### Install Dependencies Only

If you want to install dependencies manually:

```bash
pip install numpy networkx pettingzoo gymnasium PySide6
```


## Quick Start

### Basic Usage

```python
from LMAPFEnv import WarehouseEnv

# Create environment
env = WarehouseEnv(
    num_agvs=6,
    fov_size=5,
    render_mode=None,  # Set to "human" for GUI, "rgb_array" for image output
    enable_battery=False,
    max_episode_steps=500
)

# Reset environment
observations, infos = env.reset(seed=42)

# Run one episode
done = False
while not done:
    # Sample random actions
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # Step environment
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Check if episode is done
    done = all(terminations[agent] or truncations[agent] for agent in env.agents)

# Clean up
env.close()
```

### Custom Map Configuration

```python
from LMAPFEnv.envs.MAEnv import WarehouseEnv, MapConfig

# Custom map configuration
map_config = MapConfig(
    shelf_cols=3,
    shelf_rows=3,
    shelf_width=3,
    shelf_height=2,
    corridor_width=1,
    corridor_out_width=2
)

env = WarehouseEnv(num_agvs=6, map_config=map_config)
obs, infos = env.reset()
```

## Environment Details

### Observation Space
Each agent receives a local observation with:
- **self_states**: Dictionary containing:
  - `position`: Current (x, y) coordinates
  - `direction`: Facing direction (0=up, 1=down, 2=left, 3=right)
  - `target`: Target position (x, y)
  - `battery`: Battery level (if battery enabled)
- **fov**: Field of view tensor of shape (6, fov_size, fov_size) containing:
  - Local passable map
  - Other AGV positions
  - Other AGV directions (one-hot encoded)

### Action Space
Each agent can take 4 discrete actions:
- 0: Move Forward
- 1: Turn Left
- 2: Turn Right
- 3: Stop

### Reward Structure
- **Task completion**: +10.0 for reaching target
- **Hanging penalty**: -0.005 per step
- **Battery reward**: Battery level × 10 (if battery enabled)

### Task System
The environment automatically assigns random target positions to each AGV. When an AGV reaches its target, a new task is automatically assigned.

## Features

- **Multi-Agent Path Finding**: Support for multiple AGVs in warehouse environments
- **Task Assignment**: Automatic random task generation and assignment
- **Conflict Resolution**: Built-in collision avoidance and conflict resolution
- **Battery System**: Optional battery consumption and charging mechanics
- **Flexible Rendering**: Support for GUI, RGB array, and headless modes
- **Customizable Maps**: Configurable shelf layouts and corridor widths

## Supported Algorithms

- **MAPPO**: Multi-Agent PPO
- **LMAPF**: Learning-based Multi-Agent Path Finding
- **PettingZoo**: Compatible with all PettingZoo algorithms

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
