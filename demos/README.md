# demos/

Collection of demo and evaluation scripts for the heterogeneous MAPPO team (Scout + Commander).

## Structure

```
demos/
├── scout/              ← Quadcopter scouts only (no commander)
│   ├── README.md
│   └── demo_scout.py → (symlink to MAPPO_easiestScenario/)
│
├── full_team/          ← Full heterogeneous team: scouts + commander
│   ├── README.md
│   └── demo_both_training.py → (symlink)
│
├── evaluation/         ← Quantitative evaluation over N episodes
│   ├── README.md
│   └── demo_eval_2scouts.py → (symlink)
│
└── MAPPO_easiestScenario/   ← Source files (run directly from here)
    ├── demo_scout.py
    ├── demo_both_training.py
    └── demo_eval_2scouts.py
```

## Quick Start

```bash
cd /homes/eva/xj/xjahnf00/tmp/DP

# 1. Scout fire search — scouts look for 3 fires on the map
python demos/MAPPO_easiestScenario/demo_scout.py \
    --scouts 3 --fires 3 --seeds 200 201 202 203 204

# 2. Visual demo of full team — scouts + commander
python demos/MAPPO_easiestScenario/demo_both_training.py \
    --episodes 5 --seed-start 200

# 3. Quantitative evaluation — 30 episodes, results table
python demos/MAPPO_easiestScenario/demo_eval_2scouts.py \
    --episodes 30 --seed-start 0
```

## Checkpoints

| Model | Path | Description |
|---|---|---|
| Scout (frozen) | `saved_models/multi/scout_b0030.pt` | Frozen scout trained in solo phase |
| Scout (best) | `saved_models/multi/scout_best.pt` | Best multi-agent scout checkpoint |
| Commander | `saved_models/multi/cmdr_b{N}.pt` | Commander at batch N |

All scripts auto-detect the latest checkpoint if `--model` / `--commander` is not specified.
