# LMAPF-Simulator

[English](./README.md) | [中文](./README.zh.md)

Ambiente di simulazione multi-AGV per magazzini, pensato per la ricerca su **Lifelong Multi-Agent Path Finding (LMAPF/MAPF)**. Basato su PettingZoo `ParallelEnv` e Gymnasium, con planner intercambiabili e varianti rolling-horizon.

<video src="./renders/episode_demo.mp4" controls muted loop playsinline></video>

[Guardare la demo MP4](./renders/episode_demo.mp4)

Generare localmente questa demo MP4 compatta:

```bash
python tools/save_episode_video.py
```

Usare `--sleep` e `--fps` per regolare fluidita del movimento e durata del video.

Il progetto deriva da [RWARE](https://github.com/semitable/robotic-warehouse). Alcune implementazioni dei planner in questo repository sono sperimentali.

## Funzionalità

- Interfaccia **PettingZoo `ParallelEnv`**
- Mappe di magazzino integrate: preset `small`, `medium`, `large`, `long`
- Coda continua di task: nuovi task assegnati automaticamente al raggiungimento del target
- Risoluzione dei conflitti: lo step dell'ambiente usa un grafo diretto per risolvere conflitti runtime senza intervento del planner, con motore **C++ FastGraph**
- Planner intercambiabili: A*, CBS, ECBS, PBS, RHCR e relative varianti
- Visualizzazione: rendering di mappe/FOV e demo di cicli di conflitto

## Installazione

### 1. Installare un compilatore C++

Il motore C++ FastGraph e il motore A\* vengono **compilati automaticamente** al primo import. Serve un compilatore C++:

| Piattaforma | Compilatore | Installazione |
|-------------|-------------|---------------|
| **Windows** | MSVC (Visual Studio) | Installare [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) e selezionare "Desktop development with C++" |
| **Linux** | GCC >= 10 | `sudo apt install build-essential cmake` (Ubuntu/Debian) |
| **macOS** | Clang (Xcode) | `xcode-select --install` |

Anche MinGW su Windows o un GCC installato tramite package manager di sistema possono funzionare.

### 2. Creare un ambiente conda (consigliato)

```bash
conda create -n lmapf python=3.12 -y
conda activate lmapf
```

### 3. Clonare e installare

```bash
git clone https://github.com/lotjjj/LMAPF-Simulator.git
cd LMAPF-Simulator
pip install -e .
```

Dipendenze principali: `numpy`, `pettingzoo`, `gymnasium`, `pygame`, `matplotlib`, `imageio[ffmpeg]`.

### 4. Verificare l'installazione

```python
from LMAPFEnv import WarehouseEnv
from LMAPFEnv.algorithms.path_planners import _HAS_CXX_ASTAR
print("C++ A* engine:", "enabled" if _HAS_CXX_ASTAR else "not available")
```

Il primo `import LMAPFEnv` rileva automaticamente il compilatore C++ e compila il motore FastGraph (~30-60 s). Per compilare manualmente:

```bash
python build_cpp_graph.py
```

## Quick Start

### 1. Creare l'ambiente

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

### 2. Eseguire con azioni casuali

```python
done = False
while not done:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    done = all(terminations[a] or truncations[a] for a in env.agents)
env.close()
```

### 3. Eseguire con un planner

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

### 4. Eseguire benchmark

```bash
# Tutti i 6 planner
python run_benchmark.py

# Un singolo planner
python run_benchmark.py --planner RHCR_PBS
```

## Performance

![runtime-benchmark](docs/runtime_benchmark.png)

Runtime misurato con `python run_benchmark.py --steps 200` sulla mappa `long` (10 AGV, senza rendering, seed 42, assegnazione task casuale). Il tempo elapsed include creazione ambiente, reset, pianificazione iniziale e rollout di 200 step. Il tempo medio/massimo del planner conta solo le chiamate effettive al planner; gli step saltati sono esclusi.

| Planner | Steps | Completions | Conflicts | Avg plan (ms) | Max plan (ms) | Elapsed (s) |
|---------|------:|------------:|----------:|--------------:|--------------:|------------:|
| CBS | 200 | 137 | 0 | 4421.2 | 6886.0 | 79.8 |
| ECBS | 200 | 138 | 0 | 13432.3 | 21280.2 | 255.4 |
| PBS | 200 | 137 | 0 | 22338.7 | 27701.2 | 402.4 |
| RHCR_CBS | 200 | 137 | 0 | 605.5 | 2082.8 | 122.3 |
| RHCR_PBS | 200 | 126 | 0 | 87.9 | 500.2 | 18.1 |
| RHCR_ECBS | 200 | 133 | 0 | 211.0 | 2628.3 | 43.0 |

_Tutte le esecuzioni hanno completato 200 step senza uscita disabled/no-path. Misurato con motore C++ FastGraph e motore planner C++ A* abilitati su Python 3.12.13._

## Esempi e strumenti

| Comando | Descrizione |
|---------|-------------|
| `python examples/run_planner_demo.py --path-planner RHCR --continuous` | Demo interattiva guidata da planner |
| `python examples/conflict_cycle_demo.py` | Dimostrazione di ciclo di conflitto |
| `python tools/save_map_renders.py --map-sizes small medium large long` | Esporta rendering delle mappe |
| `python tools/save_agent_fov_renders.py --path-planner RHCR --steps 5` | Esporta rendering FOV |
| `python tools/save_episode_video.py` | Registra un episode MP4 compatto per demo nel README |

## Planner

| Tipo | Planner |
|------|---------|
| Single-agent | `AStar`, `EnhancedAStar` |
| MAPF | `CBS`, `ECBS`, `PBS` |
| Rolling-horizon | `RHCR`, `RHCR_CBS`, `RHCR_ECBS`, `RHCR_PBS` |

`planner_args` comuni: `shelf_penalty`, `max_cbs_nodes`, `max_pbs_nodes`, `max_low_level_steps`, `max_planning_time`, `w`, `visible_agv_penalty`, `planning_window`, `horizon`.

## API dell'ambiente

### Parametri principali

`WarehouseEnv(num_agvs, map_size, fov_size, max_episode_steps, render_mode, path_planner, planner_args)`

### Generazione dei task

- I task vengono generati solo su celle `shelf`.
- Ogni AGV vivo mantiene `num_visible_tasks` target attivi in coda, con default `2`.
- Una mappa è valida solo se `num_agvs * num_visible_tasks <= shelf_count`; altrimenti la creazione dell'ambiente solleva `ValueError`.
- Il posizionamento iniziale degli AGV preferisce celle corridoio, così i target shelf restano disponibili per la generazione dei task.

### Spazio delle azioni

Ogni agent usa `Discrete(5)` con ordine fisso:

```python
[UP, DOWN, LEFT, RIGHT, STAY]
```

### Osservazione

`obs` è un `gymnasium.spaces.Dict`:

```python
obs = Dict({
    "self_states": Dict({...}),
    "fov": Box(...),
})
```

`self_states` contiene sempre:

| Campo | Shape | Significato |
|-------|-------|-------------|
| `position` | `(2,)` | Posizione globale normalizzata |
| `fov_density` | `(1,)` | Densità degli altri AGV nel FOV |
| `target_rel` | `(2,)` | Offset normalizzato verso il target corrente |
| `target_visible` | `(1,)` | Se il target è dentro il FOV |
| `target_dist_norm` | `(1,)` | Distanza euclidea normalizzata dal target |

`fov` ha shape `(5, fov_size, fov_size)` con canali:
`corridor`, `wall/OOB`, `shelf`, `other AGVs`, `visible goals`.

### Info

Sia `reset()` sia `step()` restituiscono `infos[agent_name]`. I campi più utili sono:

| Chiave | Significato |
|--------|-------------|
| `action_mask` | Maschera di validità locale per `[UP, DOWN, LEFT, RIGHT, STAY]` |
| `conflicted` | Agent forzato a restare fermo dalla risoluzione dei conflitti |
| `invalid_action` | Azione richiesta illegale, sostituita con `STAY` |
| `task_completed` | L'agent ha raggiunto il task target corrente in questo step |
| `progress_target_pos` | Target di riferimento usato per il reward nello step |
| `progress_distance_prev` | Distanza da quel target prima dell'esecuzione |
| `progress_distance_now` | Distanza da quel target dopo l'esecuzione |
| `act_val_time_ms` | Tempo per validazione azioni + risoluzione conflitti |
| `planner_meta` | Diagnostica su timing, timeout e disable del planner |
| `planner_paths` | Snapshot condiviso dei path per tutti gli agent |

Note:

- `action_mask` rappresenta solo la validità locale; un'azione può comunque non essere eseguita se `conflicted=True`.
- `planner_paths` è uno snapshot globale, quindi di solito è identico per tutti gli `infos[agent]`.

### Timing del reward target

Quando un agent completa un task nello step corrente:

1. L'ambiente congela `progress_target_pos` per il calcolo del reward.
2. Calcola `progress_distance_prev/now` rispetto a quel target congelato.
3. Imposta `task_completed=True`.
4. Avanza la coda dei task, così `next task` diventa il nuovo `current task`.

Questo significa che:

- `progress_target_pos` punta ancora al vecchio target appena completato.
- `progress_distance_prev/now` sono misurate rispetto allo stesso vecchio target.
- L'osservazione successiva riflette già il nuovo `current task` tramite `agv.target_pos`.

### Termination e truncation

- `terminations[agent]` è sempre `False`.
- `truncations[agent]` diventa `True` per tutti gli agent vivi quando viene raggiunto `max_episode_steps`.
- L'ambiente tronca anche tutti gli agent vivi se più della metà resta nella stessa posizione durante la finestra di congestione.
- Gli agent marcati come terminated o truncated in uno step vengono rimossi da `env.agents` prima dello step successivo.

### Reward di default

```python
reward = each_step_reward
       + invalid_action_penalty
       + conflict_penalty
       + progress_shaping_weight * clip(d_prev - d_now, -1, 1)
       + task_completion_reward
```

Valori di default:

- `each_step_reward = -0.002`
- `invalid_action_penalty = -0.05`
- `conflict_penalty = -0.6`
- `progress_shaping_weight = 0.01`
- `task_completion_reward = +2.0`

## Licenza

MIT License
