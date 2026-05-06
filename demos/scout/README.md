# demo_scout — Scout fire reconnaissance

Visualisation demo for a team of N quadcopter scouts with no commander.
Each scout runs a trained `ScoutActor` network, searches for fire autonomously,
and communicates with neighbours via cross-attention messages.

## Outputs

| File | Description |
|---|---|
| `scout_s{seed}_n{N}_{size}m.gif` | Animated episode — OSM map, trajectories, camera panels |
| `scout_s{seed}_n{N}_{size}m_analysis.png` | Cumulative reward, altitude, fire-under-camera plots |
| `scout_summary.csv` | Summary metrics table across all episodes |

## Usage

```bash
cd ~/tmp/DP

# Default — 2 scouts, 5 episodes, 3 fires, 1000 m map
python demos/scout/demo_scout.py

# 5 scouts, 6 fires, larger map
python demos/scout/demo_scout.py --scouts 5 --fires 6 --seeds 42 --grid-size 1200

# Fixed fire at centre, 3 scouts, auto-saves 3 PNG frames for presentation
python demos/scout/demo_scout.py --scouts 3 --fixed-fire --seeds 42 --max-steps 800

# Multiple map sizes in one run
python demos/scout/demo_scout.py --scouts 3 --fires 3 --seeds 42 \
    --grid-sizes 800 1200 2000

# Fast run without GIF (metrics + PNG only)
python demos/scout/demo_scout.py --scouts 2 --no-gif --seeds 42

# Custom checkpoint
python demos/scout/demo_scout.py \
    --model saved_models/scout_solo/scout_b0710.pt --scouts 1
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | auto (latest `scout_b*.pt`) | Path to ScoutActor checkpoint |
| `--scouts` | 2 | Number of scouts (1–5) |
| `--fires` | 3 | Number of fire sources per episode (≥150 m apart) |
| `--seeds` | — | Explicit seed list (one seed = one episode) |
| `--seed-start` | 100 | Start seed when `--seeds` is not given |
| `--episodes` | 5 | Number of episodes from `--seed-start` |
| `--max-steps` | 1000 | Episode length in steps |
| `--grid-size` | — | Single map size in metres (overrides `--grid-sizes`) |
| `--grid-sizes` | `1000` | One or more map sizes — runs all episodes for each size |
| `--gif-every` | 2 | Write every N-th step as a GIF frame |
| `--gif-fps` | 15 | Output GIF framerate |
| `--no-gif` | — | Skip GIF generation (faster) |
| `--no-osm` | — | No OSM terrain background |
| `--fixed-fire` | — | 1 fire at centre, scouts spawn 200–250 m away; auto-saves 3 PNG frames |

## Printed metrics

| Metric | Meaning |
|---|---|
| `fire_disc` | Whether any scout found fire (YES / no) |
| `coverage%` | % of steps with at least one scout seeing fire |
| `avg_seen` | Mean fire intensity under camera (0–1 scale) |
| `survived` | Scouts survived / total |
| `R/scout` | Mean cumulative reward per scout |

## GIF layout

```
┌──────────────────────────┬──────────────────┐
│                          │  Scout 0 camera  │
│   Global map + fire      │  (32×32 px FoV)  │
│   OSM terrain + trails   ├──────────────────┤
│   Scout FoV rectangles   │  Scout 1 camera  │
│                          ├──────────────────┤
│                          │  ...             │
│                          ├──────────────────┤
│                          │  Mission stats   │
└──────────────────────────┴──────────────────┘
```

## Notes

- Scouts **cannot suppress fire** — fire keeps spreading throughout the episode
- Checkpoint is auto-detected as the latest `saved_models/multi/scout_b*.pt`
- With `--scouts N --fires N` each scout starts near a different fire source
- Scouts share network weights (parameter sharing)


