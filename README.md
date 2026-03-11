# Heterogeneous MAPPO — Drone Wildfire Suppression

Multi-agent reinforcement learning system for cooperative wildfire suppression using a heterogeneous team of drones. A quadrotor scout (ScoutActor) explores the fire area and broadcasts messages to a fixed-wing commander (CommanderActor), which coordinates the suppression. Training uses the **Heterogeneous MAPPO** algorithm with parallel rollout collection across 20 CPU workers.

## Project structure

```
src/
├── train.py        # Main training entry point (Heterogeneous MAPPO)
├── worker.py       # Parallel rollout worker (runs in ProcessPoolExecutor)
├── models.py       # ScoutActor, CommanderActor, MAPPOCritic
├── env_core.py     # DroneFireEnv (PettingZoo-style multi-agent env)
├── simulation.py   # PyBullet simulation backend
├── fire_grid.py    # Fire spread model
├── grid_mapper.py  # Local 32x32 fire map for scout observations
├── drones/         # Per-drone physics controllers
└── visualizer.py   # Training dashboard plots

saved_models/       # Best checkpoints (scout_best.pt, commander_best.pt)
logs/               # Per-episode JSON logs
results/            # Plots and analysis outputs
run.sh              # SGE cluster job script (20-core SMP)
requirements.txt
```

## Architecture

### Agents

| Agent | Network | Observation | Action |
|---|---|---|---|
| Quadrotor scout | `ScoutActor` (CNN + GRU + Self-Attention) | Local fire map 32×32, self state, neighbour positions | Roll, Pitch, Yaw, Throttle |
| Fixed-wing commander | `CommanderActor` (GRU + Cross-Attention) | Self state, scout messages (5-dim vectors) | Roll, Pitch, Yaw, Throttle |
| Shared critic | `MAPPOCritic` (MLP + GRU) | Global state (fire map 16×16 + all agent states) | Value estimate |

### Training loop

```
Main process (GPU/CPU)
  for each batch:
    1. Snapshot network weights → CPU tensors
    2. Dispatch 20 workers in parallel (ProcessPoolExecutor)
       each worker: 3 episodes × full rollout → returns tensors
    3. Collect & cat all worker buffers
    4. PPO update (update_epochs passes, Adam optimizer)
    5. Save checkpoint if rolling avg reward improved
```

### Key hyperparameters (`src/train.py`)

| Parameter | Value | Description |
|---|---|---|
| `num_episodes` | 2500 | Total training episodes |
| `max_steps` | 2500 | Max timesteps per episode |
| `num_workers` | 20 | Parallel rollout workers |
| `eps_per_worker` | 3 | Episodes per worker per batch |
| `episodes_per_batch` | 60 | Total episodes per gradient update |
| `update_epochs` | 4 | PPO gradient passes per batch |
| `clip_coef` | 0.2 | PPO clipping range |
| `lr_commander` / `lr_critic` | 3e-4 | Learning rates |
| `lr_scout_fine_tune` | 5e-5 | Scout fine-tune LR (pre-trained) |
| `gamma` | 0.99 | Discount factor |

## Quick start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run training locally
```bash
cd /path/to/DP
python src/train.py
```

Output: `saved_models/scout_best.pt`, `saved_models/commander_best.pt`, `final_training_plot.png`

### Run on the cluster (SGE)
```bash
qsub run.sh
```
`run.sh` requests 20 SMP slots and 2h wall time. Output goes to `out.$JOB_ID.txt`.

## Running on the cluster with tmux (interactive, long jobs)

Use this when you want to SSH in, start a long run, and safely disconnect.

**Start the job:**
```bash
# 1. SSH into the cluster
ssh xjahnf00@aeroworks

# 2. Start a named tmux session
tmux new -s mujjob

# 3. Inside tmux — run the training
cd ~/tmp/DP
python src/train.py

# 4. Detach from tmux (training keeps running after you log out)
#    Press:  Ctrl + B,  then  D
```

**Come back to a running job:**
```bash
# SSH back in, then:
tmux attach -t mujjob
```

**Other useful tmux commands:**
```bash
tmux ls               # list all sessions
tmux kill-session -t mujjob   # kill the session
```

## Profiling the PPO update

`train.py` has a built-in cProfile hook. Set the flag at the top of `train()`:

```python
profiling = True   # runs one batch, prints cProfile table, saves ppo_update.prof, then exits
```

View the saved profile:
```bash
pip install snakeviz
snakeviz ppo_update.prof
```

**Profiling results (CPU, 1 batch, 8 epochs, 150k samples):**

| Op | Self time | Notes |
|---|---|---|
| `loss.backward()` | 222s (71%) | Unavoidable on CPU — GPU gives ~20× speedup |
| `GRU` (all networks) | 31s (10%) | Sequential on CPU |
| `conv2d` (ScoutActor CNN) | 29s (9%) | Freeze CNN if not fine-tuning |
| `linear` layers | 17s (5%) | |

The code itself has no Python-level overhead — the bottleneck is pure math on CPU. To speed up: use a GPU, reduce `update_epochs`, or freeze `scout_actor.cnn` with `scout_actor.cnn.requires_grad_(False)`.

## Saved models

| File | Description |
|---|---|
| `saved_models/scout_best.pt` | Best ScoutActor checkpoint (rolling avg reward) |
| `saved_models/commander_best.pt` | Best CommanderActor checkpoint |
| `saved_models/scout_ep{N}.pt` | Periodic snapshots every 10 batches |

---

*DP project — Heterogeneous multi-agent drone coordination for wildfire suppression using MAPPO*