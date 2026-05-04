# demo_eval_2scouts — Quantitative evaluation of the Scout + Commander team

Evaluation harness for **objective performance measurement** of a trained
heterogeneous team (2 scouts + 1 commander) over a large number of episodes.

Unlike `demo_both_training`, this script **does not generate GIFs** — it focuses
on statistics across 30 episodes, making it much faster and suitable for
checkpoint comparison during training.

## Why it exists

- Objective A/B comparison between two checkpoints
- Detection of systematic failure modes (FW misses fire, FW dies early, scouts do not report)
- Diagnostic input — pinpointing exactly where training fails

## Outputs

| File | Description |
|---|---|
| Stdout table | Per-episode results + aggregate statistics |
| `eval_results_{timestamp}.png` | Box/scatter plots of key metrics |

### Metrics
| Metric | Description |
|---|---|
| `OK` | FW successfully dropped water near fire |
| `FW_MISSES_FIRE` | FW fired but missed the fire |
| `WATER_DEPLETED_EARLY` | FW ran out of water without a hit |
| `SCOUTS_NOT_REPORTING` | Scouts did not find fire → FW has no navigation |
| Avg refill events | Mean number of tank refills per episode |
| Trigger precision | % of water drops that hit fire |

## Usage

```bash
# From the project root:
cd /homes/eva/xj/xjahnf00/tmp/DP

# Default — 30 episodes with auto-detected checkpoints
python demos/MAPPO_easiestScenario/demo_eval_2scouts.py

# Specific checkpoints
python demos/MAPPO_easiestScenario/demo_eval_2scouts.py \
    --scout     saved_models/multi/scout_b0030.pt \
    --commander saved_models/multi/cmdr_b0780.pt \
    --episodes  30 \
    --seed-start 0

# Compare two checkpoints (run twice with different --commander)
python demos/MAPPO_easiestScenario/demo_eval_2scouts.py \
    --commander saved_models/multi/cmdr_b0500.pt --episodes 30
python demos/MAPPO_easiestScenario/demo_eval_2scouts.py \
    --commander saved_models/multi/cmdr_b0780.pt --episodes 30
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--scout` | auto | Path to ScoutActor checkpoint |
| `--commander` | auto | Path to CommanderActor checkpoint |
| `--episodes` | 30 | Number of episodes |
| `--seed-start` | 0 | Start seed |

## Reference results at batch 730

```
OK: 20/30 (67%)   FW_MISSES_FIRE: 3/30   WATER_DEPLETED: 2/30
Avg refill events:   1.07 / episode
Avg empty steps:   356 / 1000
Trigger precision:  39.1%   (was 3.5% before fire compass = 10× improvement)
Min dist FW→fire:    7.3 m  (navigation working)
```
