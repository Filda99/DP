# src/ — Environment, models, and training

## File descriptions

| File | Description |
|---|---|
| `train_scout.py` | Scout solo pre-training (quadcopters only, no commander) |
| `train_multi.py` | Heterogeneous MAPPO training (scouts + commander, CTDE) |
| `models.py` | `ScoutActor`, `CommanderActor`, `PrivilegedCritic` network definitions |
| `env_core.py` | `DroneFireEnv` — PettingZoo `ParallelEnv` wrapping PyBullet physics |
| `reward_config.py` | Centralised reward parameter configuration |
| `simulation.py` | PyBullet simulation backend (physics, collision, sensors) |
| `fire_grid.py` | Cellular automaton fire spread model |
| `grid_mapper.py` | World ↔ grid coordinate mapping utilities |
| `map_importer.py` | OSM terrain loading (`load_environment_from_osm_cache`) |
| `drones/` | Per-drone physics controllers (quadrotor, fixed-wing) |

---

## Scout pre-training (`train_scout.py`)

Train N scouts without a commander. Scouts learn to find fire and stay near it.
Use this to obtain a frozen scout checkpoint for multi-agent training.

```bash
cd ~/tmp/DP

# Train from scratch
python src/train_scout.py

# Resume from checkpoint
python src/train_scout.py --resume saved_models/scout_solo/scout_b0710.pt
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--resume` | `""` | Path to a ScoutActor checkpoint to resume from |

Checkpoints are saved to `saved_models/scout_solo/` as `scout_b{N}.pt` every 10 batches.

---

## Multi-agent training (`train_multi.py`)

Heterogeneous MAPPO: scouts (frozen or co-trained) + commander.
Uses centralised training with decentralised execution (CTDE).

### Training from scratch

```bash
cd ~/tmp/DP
python src/train_multi.py
```

### Resume from checkpoint

```bash
# From batch 70:
python src/train_multi.py \
  --resume-scout saved_models/multi/scout_b0070.pt \
  --resume-cmdr  saved_models/multi/cmdr_b0070.pt

# From best checkpoint:
python src/train_multi.py \
  --resume-scout saved_models/multi/scout_best.pt \
  --resume-cmdr  saved_models/multi/cmdr_best.pt
```

### With trajectory logging

```bash
python src/train_multi.py --log-episodes --log-dir /tmp/ep_logs
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--resume-scout` | `""` | Path to ScoutActor checkpoint |
| `--resume-cmdr` | `""` | Path to CommanderActor checkpoint |
| `--log-episodes` | off | Save trajectory logs per episode |
| `--log-dir` | `/tmp/ep_logs` | Directory for trajectory logs |
| `--start-ep` | 0 | Episode counter offset for logging |

Checkpoints are saved to `saved_models/multi/`:
- `scout_best.pt`, `cmdr_best.pt` — best by rolling average reward
- `scout_b{N}.pt`, `cmdr_b{N}.pt` — periodic snapshots every 10 batches

---

## Key hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `num_workers` | 15 | Parallel rollout workers |
| `eps_per_worker` | 2 | Episodes per worker per batch |
| `max_steps` | 4000 | Step buffer size (upper bound) |
| `steps_range` | (1500, 4000) | Actual episode length (randomised) |
| `map_size_range` | (600, 2000) | Map size randomisation [m] |
| `waypoint_steps` | 50 | Commander decision interval (steps) |
| `waypoint_range` | 100 m | Max waypoint displacement per decision |
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


