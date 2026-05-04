# demo_scout — Scout fire reconnaissance

Visualisation demo for a team of N quadcopter scouts (no commander).
Each scout is controlled by a trained `ScoutActor` network — it flies
autonomously, searches for fire, and sends messages to neighbours via a
cross-attention mechanism.

## Why it exists

The scout is one of two roles in the heterogeneous MAPPO team. This demo
isolates its behaviour without the commander, allowing verification that
the scout:
- finds fire (fire discovery)
- stays near it (coverage)
- survives 1 000 steps (boundary and altitude management)
- communicates meaningfully (`msg[2]` = fire intensity)

## Outputs

| File | Description |
|---|---|
| `scout_s{seed}_n{N}_{size}m.gif` | Episode animation — OSM map + trajectories + camera panels |
| `scout_s{seed}_n{N}_{size}m_analysis.png` | Cumulative reward, altitude and fire-under-camera plots |
| `scout_summary.csv` | Summary metrics table across all episodes |

### Printed metrics
| Metric | Meaning |
|---|---|
| `fire_disc` | Whether any scout found fire (YES / no) |
| `coverage%` | % of steps where at least one scout saw fire in its camera |
| `avg_seen` | Mean fire intensity under the camera (0–1 scale) |
| `survived` | Number of scouts that survived / total |
| `R/scout` | Mean cumulative reward per scout |

## Usage

```bash
# From the project root:
cd /homes/eva/xj/xjahnf00/tmp/DP

# Default — 2 scouts, 5 episodes from seed=100, 3 fires, 1000m map
python demos/scout/demo_scout.py

# Custom configuration
python demos/scout/demo_scout.py \
    --scouts 3 \
    --fires  3 \
    --seeds  200 201 202 203 204

# Multiple map sizes — one run per size, same seed
python demos/scout/demo_scout.py \
    --seeds 202 --max-steps 400 --scouts 3 --fires 3 \
    --grid-sizes 800 1200 2000 --no-osm

# 4 scouts, 4 fires — each scout starts near a different fire
python demos/scout/demo_scout.py \
    --scouts 4 --fires 4 --seeds 200 201 202

# Fast run without GIF (metrics + PNG only)
python demos/scout/demo_scout.py \
    --scouts 3 --fires 3 --seeds 200 201 202 203 204 --no-gif

# Custom checkpoint
python demos/scout/demo_scout.py \
    --model saved_models/multi/scout_b0780.pt \
    --scouts 2 --fires 3 --episodes 10 --seed-start 300
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | auto (latest `scout_b*.pt`) | Path to ScoutActor checkpoint |
| `--scouts` | 2 | Number of scouts (2–5) |
| `--fires` | 3 | Number of fire sources per episode (≥150 m apart) |
| `--seeds` | — | Explicit seed list |
| `--seed-start` | 100 | Start seed (when `--seeds` is not given) |
| `--episodes` | 5 | Number of episodes (with `--seed-start`) |
| `--max-steps` | 1000 | Episode length in steps |
| `--grid-size` | — | Single map size in metres (overrides `--grid-sizes`) |
| `--grid-sizes` | `1000` | One or more map sizes — runs a full set of episodes per size |
| `--gif-every` | 2 | Write every N-th step as a GIF frame |
| `--no-gif` | — | Skip GIF generation (faster) |
| `--no-osm` | — | No OSM terrain (faster, empty map) |

## GIF layout

```
┌──────────────────────────┬──────────────────┐
│                          │  Scout 0 camera  │
│   Global map + fire      │  (32×32 px FOV)  │
│   OSM terrain + trails   ├──────────────────┤
│   Scout FOV rectangles   │  Scout 1 camera  │
│                          ├──────────────────┤
│                          │  Scout N camera  │
│                          ├──────────────────┤
│                          │  Mission stats   │
└──────────────────────────┴──────────────────┘
```

## Notes

- Scouts **cannot suppress fire** (no water tank) — fire keeps spreading throughout the episode
- Checkpoint is auto-detected as the latest `saved_models/multi/scout_b*.pt`
- With `--scouts N --fires N` each scout starts near a different fire source → natural diversification
- Scouts share network weights (parameter sharing) — with identical observations they may behave
  identically; this is a training property, not a bug
