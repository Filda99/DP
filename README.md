# Heterogeneous MAPPO — Drone Wildfire Suppression

Multi-agent reinforcement learning system for cooperative wildfire suppression using a heterogeneous team of drones. A quadrotor **scout** explores the fire area and broadcasts messages to a fixed-wing **commander**, which navigates to the fire and extinguishes it. Training uses the **Heterogeneous MAPPO** algorithm with parallel rollout collection.

---

## Project structure

```
DP/
├── src/                        # Environment, models, training scripts
│   ├── train_scout.py          #   Scout solo training
│   ├── train_multi.py          #   Heterogeneous MAPPO (scout + commander)
│   ├── models.py               #   ScoutActor, CommanderActor, PrivilegedCritic
│   ├── env_core.py             #   DroneFireEnv (PettingZoo ParallelEnv)
│   ├── reward_config.py        #   Reward parameters
│   ├── simulation.py           #   PyBullet simulation backend
│   ├── fire_grid.py            #   Fire spread model
│   ├── grid_mapper.py          #   World↔grid coordinate mapping
│   ├── map_importer.py         #   OSM terrain loading
│   ├── drones/                 #   Per-drone physics controllers
│   └── README.md               #   ← Training details & file descriptions
│
├── demos/                      # Visualisation & evaluation scripts
│   ├── scout/                  #   Scout-only demo (N scouts, no commander)
│   ├── full_team/              #   Full team demo (scouts + commander)
│   ├── evaluation/             #   Quantitative evaluation harness
│   └── README.md               #   ← Demo overview & quick start
│
├── saved_models/               # Trained checkpoints
│   ├── multi/                  #   scout_best.pt, cmdr_best.pt, scout_b{N}.pt, ...
│   └── scout_solo/             #   scout_b{N}.pt from solo training
│
├── data/                       # OSM terrain cache (.gpkg files)
├── results/                    # Training plots and analysis outputs
├── requirements.txt
└── README.md                   # ← This file
```

---

## Architecture

### Agents

| Agent | Network | Observation | Action |
|---|---|---|---|
| Quadrotor scout | `ScoutActor` (CNN + Self-Attention + GRU, hidden=128) | Local fire map 32×32 + self state (15D) + neighbour states | Roll, Pitch, Yaw, Throttle (4D, per-step) |
| Fixed-wing commander | `CommanderActor` (Encoder + Cross-Attention + GRU, hidden=64) | Self state (17D) + scout messages (5D × N scouts) | dx, dy, target_alt, water_trigger (4D waypoint, every 50 steps) |
| Privileged critics | `PrivilegedCritic` (MLP + GRU) × 2 | Agent obs + fire pos + fire intensity + other agent pos | Value estimate (CTDE) |

### Commander waypoint pipeline

```
CommanderActor NN → waypoint [dx, dy, alt, water]  (every 50 physics steps)
        ↓
heading controller → [heading_cmd, alt, water]     (every step)
        ↓
env_core.py flight controller → [roll, pitch, throttle, water] → physics
```

### Scout message

Each scout broadcasts a 5D message: `[pos_x, pos_y, fire_intensity, alt, heading]`. The commander aggregates messages from all scouts via cross-attention to decide its next waypoint.

---

## Reward summary

| Reward | Value | Agent |
|---|---|---|
| Survival bonus | +0.3/step | Both |
| Episode completion | +20.0 | Both |
| Boundary penalty | −5.0 × (1 − dist/threshold)² | Scout (<150m), Cmdr (<300m) |
| Altitude penalty | Continuous gradient outside ideal band | Scout (40–200m), Cmdr (40–150m) |
| Crash | −200 | Both |
| Fire detection | +1.0 flat + intensity × 10.0 | Scout |
| Approach shaping | ±0.3 per 100m (potential-based) | Scout (no fire visible) |
| Fire approach gradient | 0.0–1.0 (distance-based) | Commander |
| Water extinguish | min(eff × 50, 3.0)/step | Commander |
| Scout share of extinguish | 20% of commander bonus | Scout |
| Fire spread penalty | −Δcells × 0.02 | Both |
| Reward clipping | [−15, +15] | Both |

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyBullet, PyTorch, GeoPandas, osmnx.

---

## Where to go next

| Goal | See |
|---|---|
| Train a model | [src/README.md](src/README.md) |
| Run a visual demo | [demos/README.md](demos/README.md) |
| Evaluate a checkpoint | [demos/evaluation/README.md](demos/evaluation/README.md) |

---

## Saved models

| File | Description |
|---|---|
| `saved_models/multi/scout_best.pt` | Best ScoutActor (rolling avg reward) |
| `saved_models/multi/cmdr_best.pt` | Best CommanderActor |
| `saved_models/multi/scout_b{N}.pt` | Periodic snapshots every 10 batches |
| `saved_models/multi/cmdr_b{N}.pt` | Commander periodic snapshots |
| `saved_models/scout_solo/scout_b{N}.pt` | Scout from solo pre-training |

---

## Useful tmux commands

```bash
tmux new -s train          # new session
tmux attach -t train       # reconnect
tmux ls                    # list sessions
tmux kill-session -t train # kill session
```

---

*DP project — Heterogeneous multi-agent drone coordination for wildfire suppression using MAPPO*

|---|---|---|---|
| Quadrotor scout | `ScoutActor` (CNN + Self-Attention + GRU, hidden=128) | Local fire map 32×32 + self state (15D) + neighbour positions | Roll, Pitch, Yaw, Throttle (4D, per-step) |
| Fixed-wing commander | `CommanderActor` (Encoder + Cross-Attention + GRU, hidden=64) | Self state (17D) + scout messages (5D × 2) | dx, dy, target_alt, water_trigger (4D waypoint, every 50 steps) |
| Privileged critics | `PrivilegedCritic` (MLP + GRU) × 2 | Agent obs + fire pos + fire intensity + other agent pos | Value estimate (CTDE) |

### Commander waypoint pipeline

```
CommanderActor NN → waypoint [dx, dy, alt, water] (every 50 physics steps)
        ↓
train_multi.py heading controller → [heading_cmd, alt, water] (every step)
        ↓
env_core.py flight controller → [roll, pitch, throttle, water] → physics
```

### Training loop

```
Main process (CPU)
  for each batch (30 episodes):
    1. Snapshot network weights
    2. Run 15 workers × 2 episodes (inline, sequential)
    3. Collect buffers: scout [eps × max_steps], commander [eps × num_decisions]
    4. PPO update (4 epochs, minibatch, separate optimizers per agent)
    5. Save checkpoint if rolling avg reward improved
```

## Key hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `num_workers` | 15 | Parallel rollout workers |
| `eps_per_worker` | 2 | Episodes per worker per batch |
| `max_steps` | 4000 | Buffer size (upper bound) |
| `steps_range` | (1500, 4000) | Actual episode length (randomized) |
| `map_size_range` | (600, 2000) | Map size randomization [m] |
| `waypoint_steps` | 50 | Commander decision interval |
| `waypoint_range` | 100 m | Max waypoint displacement |
| `gamma` / `gamma_cmdr` | 0.99 / 0.95 | Discount factors |
| `clip_coef` | 0.2 | PPO clipping range |
| `update_epochs` | 4 | PPO gradient passes per batch |
| `lr_scout` | 1e-4 | Scout actor learning rate |
| `lr_cmdr` | 3e-4 | Commander actor learning rate |
| `lr_critic` | 3e-4 | Both critics learning rate |
| `hidden_dim_scout` | 128 | Scout GRU hidden size |
| `hidden_dim_cmdr` | 64 | Commander GRU hidden size |
| `entropy_scout` | 0.003 | Scout entropy coefficient |
| `entropy_cmdr` | 0.01 | Commander entropy coefficient |

## Reward summary

| Reward | Value | Agent |
|---|---|---|
| Survival bonus | +0.3/step | Both |
| Episode completion | +20.0 | Both |
| Boundary penalty | −5.0 × (1 − dist/threshold)² | Scout (<150m), Cmdr (<300m) |
| Altitude penalty | Continuous gradient outside ideal band | Scout (40–200m), Cmdr (40–150m) |
| Crash (ground/boundary/ceiling) | −200 | Both |
| Fire detection | +1.0 flat + intensity × 10.0 | Scout (in altitude band) |
| Approach shaping | ±0.3 per 100m (potential-based) | Scout (when no fire visible) |
| Fire approach gradient | 0.0–1.0 (distance-based) | Commander |
| Water extinguish | min(eff × 50, 3.0)/step | Commander |
| Scout share of extinguish | 20% of commander bonus | Scout |
| Fire spread penalty | −Δcells × 0.02 | Both |
| Reward clipping | [−15, +15] | Both |

## Quick start

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Training (`train_multi.py`)

Modely se ukládají do `saved_models/multi/`. Checkpointy: `scout_best.pt`, `cmdr_best.pt` + periodické `scout_b{BATCH}.pt`, `cmdr_b{BATCH}.pt` každých 10 batchů.

### Trénink od nuly
```bash
cd ~/tmp/DP
python src/train_multi.py
```

### Pokračování z checkpointu (resume)
```bash
# Resume z batch 70:
python src/train_multi.py \
  --resume-scout saved_models/multi/scout_b0070.pt \
  --resume-cmdr  saved_models/multi/cmdr_b0070.pt

# Resume z nejlepšího modelu:
python src/train_multi.py \
  --resume-scout saved_models/multi/scout_best.pt \
  --resume-cmdr  saved_models/multi/cmdr_best.pt
```

### Trénink s logováním trajektorií
```bash
python src/train_multi.py --log-episodes --log-dir /tmp/ep_logs
```

### Trénink přes tmux (dlouhé běhy, odpojitelné)
```bash
# 1. SSH na cluster
ssh xjahnf00@aeroworks

# 2. Nová tmux session
tmux new -s train

# 3. Spusť trénink
cd ~/tmp/DP
python src/train_multi.py

# 4. Odpoj se: Ctrl+B, pak D  (trénink běží dál)

# 5. Připoj se zpátky:
tmux attach -t train
```

---

## Demo (`demo_both_training.py`)

Vizualizace natrénovaných modelů. Defaultně bere `saved_models/multi/scout_best.pt` + `cmdr_best.pt`.

### Spuštění s nejlepšími modely
```bash
cd ~/tmp/DP
python demos/MAPPO_easiestScenario/demo_both_training.py
```

### Demo s konkrétním checkpointem
Uprav cesty v `demo_both_training.py` (řádky 35–36):
```python
MODEL_SCOUT     = os.path.join(project_root, "saved_models", "multi", "scout_b0070.pt")
MODEL_COMMANDER = os.path.join(project_root, "saved_models", "multi", "cmdr_b0070.pt")
```

---

## Demo — pouze scouti (`demo_scout.py`)

Vizualizace N-scout MAPPO týmu bez commandera. Výstup: `.gif` + `_analysis.png` per seed, `scout_summary.csv`.

### Základní spuštění (2 scouti, 5 epizod)
```bash
cd ~/tmp/DP
python demos/scout/demo_scout.py
```

### Více scoutů a ohňů
```bash
python demos/scout/demo_scout.py --scouts 5 --fires 6 --seeds 42
```

### Konkrétní seed(y) a větší mapa
```bash
python demos/scout/demo_scout.py --scouts 3 --seeds 42 43 44 --grid-size 1200
```

### Všechny parametry
| Parametr | Default | Popis |
|---|---|---|
| `--model` | auto (newest in `saved_models/`) | Cesta k `.pt` checkpointu |
| `--scouts` | 2 | Počet scoutů (1–5) |
| `--fires` | 3 | Počet zdrojů ohně |
| `--seeds` | 100…104 | Seznam seedů (jeden seed = jedna epizoda) |
| `--seed-start` | 100 | Počáteční seed (pokud `--seeds` není zadán) |
| `--episodes` | 5 | Počet epizod od `--seed-start` |
| `--max-steps` | 1000 | Délka epizody |
| `--grid-size` | 1000 | Velikost mapy v metrech |
| `--gif-every` | 2 | Každý N-tý krok uložit snímek do GIF |
| `--gif-fps` | 15 | FPS výstupního GIF |
| `--no-gif` | — | Nevytvářet GIF (jen analýzu) |
| `--no-osm` | — | Nepoužívat OSM terén |
| `--fixed-fire` | — | 1 oheň ve středu, scouti 200–250 m od něj; auto-uloží 3 PNG snímky |

### Příklady
```bash
# 5 scoutů, 6 ohňů, větší mapa
python demos/scout/demo_scout.py --scouts 5 --fires 6 --seeds 42 --grid-size 1200 --max-steps 1500

# Fixní oheň uprostřed, 3 scouti, 3 PNG snímky pro prezentaci
python demos/scout/demo_scout.py --scouts 3 --fixed-fire --seeds 42 --max-steps 800

# Rychlé testování bez GIF
python demos/scout/demo_scout.py --scouts 2 --no-gif --seeds 42

# Konkrétní model
python demos/scout/demo_scout.py --model saved_models/scout_solo/scout_b0710.pt --scouts 1
```

---

## Užitečné tmux příkazy
```bash
tmux ls                       # seznam sessions
tmux attach -t train          # připojit se
tmux kill-session -t train    # zabít session
```

## Saved models

| Soubor | Popis |
|---|---|
| `saved_models/multi/scout_best.pt` | Nejlepší ScoutActor (podle rolling avg reward) |
| `saved_models/multi/cmdr_best.pt` | Nejlepší CommanderActor |
| `saved_models/multi/scout_b{N}.pt` | Periodické snapshoty každých 10 batchů |
| `saved_models/multi/cmdr_b{N}.pt` | Periodické snapshoty commandera |
| `saved_models/multi/training_b{N}.png` | Grafy tréninku |


## Moje ulozene commandy:

Spusteni evaluation celeho reseni:
```bash 
python tools/eval_full_scenario.py     --scout-model saved_models/multi/scout_best.pt     --cmdr-model  saved_models/multi/cmdr_b0780.pt     --runs 50     --max-steps 1000
```

Scout - ruzne mapy
```bash
python demos/scout/demo_scout.py     --seeds 203 --max-steps 400 --scouts 5 --fires 3     --grid-sizes 800 1200 2000
```


---

*DP project — Heterogeneous multi-agent drone coordination for wildfire suppression using MAPPO*