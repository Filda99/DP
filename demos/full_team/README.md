# full_team/ — Heterogeneous team demos: Scout + Commander

Visualisation demos for the **full heterogeneous team**: N quadcopter scouts +
N fixed-wing commanders. The commander is controlled by a trained `CommanderActor`
network and navigates toward fire exclusively through scout messages (cross-attention).

## Scripts

| Script | Purpose |
|---|---|
| `demo_both.py` | Interactive visual demo — GIF + analysis PNG per episode |
| `demo_final_collage.py` | Thesis figure generator — saves frame snapshots at checkpoint steps + collage |

---

## demo_both.py

### Outputs

| File | Description |
|---|---|
| `demo_seed{N}.gif` | Episode animation |
| `demo_training_analysis_seed{N}.png` | Reward, altitude, water level and fire plots |
| Stdout table | Summary metrics across all episodes |

### Printed metrics
| Metric | Description |
|---|---|
| `supp%` | Fire suppression percentage (end_cells / peak_cells) |
| `FW` | Commander survival (✓ / ✗) |
| `S` | Each scout's survival (✓ / ✗) |
| `R_cmdr` | Commander total reward for the episode |

### Usage

```bash
cd ~/tmp/DP

# Default — auto-detect latest checkpoints
python demos/full_team/demo_both.py

# 30 episodes from seed=200 with specific checkpoints
python demos/full_team/demo_both.py \
    --scout     saved_models/multi/scout_b0030.pt \
    --commander saved_models/multi/cmdr_b0780.pt \
    --episodes  30 \
    --seed-start 200

# GIF for every episode
python demos/full_team/demo_both.py \
    --episodes 10 --seed-start 200 --gif-all

# Fast run without GIF
python demos/full_team/demo_both.py \
    --episodes 30 --seed-start 200 --no-gif
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--scout` | auto | Path to ScoutActor checkpoint |
| `--commander` | auto | Path to CommanderActor checkpoint |
| `--episodes` | 5 | Number of episodes |
| `--seed-start` | 100 | Start seed |
| `--max-steps` | 1000 | Episode length |
| `--gif-all` | — | Generate a GIF for every episode (default: first only) |
| `--no-gif` | — | No GIF output |

## Fire compass architecture

The commander never receives the ground-truth fire position from the environment —
it navigates exclusively through scout messages. Before each new waypoint, a fire
compass is injected into `self_state[19:23]`:

```
self_state[19] = compass_x      # normalised direction to fire (X)
self_state[20] = compass_y      # normalised direction to fire (Y)
self_state[21] = dist_norm      # distance to fire / 1000 m
self_state[22] = max_intensity  # max fire intensity visible to scouts
```

The compass is computed from `msg[2] > 0.01` (scout sees fire).

## Notes

- The scout checkpoint is **frozen** — only the commander trains
- The FW operates in **waypoint mode**: every 50 steps the NN outputs a new waypoint;
  between waypoints a heading controller tracks the target
- The refill zone is on the opposite side of the map from the fire (~1 000 m away)

---

## demo_final_collage.py

Runs the full-team simulation and saves individual PNG frames at configurable
checkpoint steps plus a final collage image (PNG + PDF). Intended for thesis figures.

### Usage

```bash
python demos/full_team/demo_final_collage.py
```

Configuration is done by editing constants at the top of the script
(`N_QUADS`, `N_FIXED`, `GRID_SIZE`, `MAX_STEPS`, `EPISODE_SEEDS`, etc.).

### Outputs

| File | Description |
|---|---|
| `step_{NNNN}.png` | Individual frame snapshot at each checkpoint step |
| `collage_seed{N}.png` | Combined collage of all checkpoint frames |
| `collage_seed{N}.pdf` | PDF version of the collage |
