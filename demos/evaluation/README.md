# Evaluation scripts

Headless evaluation scripts — no GIFs, pure metrics + CSV.

| Script | Purpose |
|--------|--------|
| `eval_full_team.py` | Thesis result table: multi-config sweep (map size × scout count) |
| `eval_scout.py` | Scout-only headless evaluation: discovery, coverage, separation metrics |
| `eval_3scouts_multifire.py` | 3 scouts + 1 FW on 3–4 fires — GIF + analysis per seed |

---

## eval_full_team.py

Systematic evaluation of the full heterogeneous team across multiple
configurations. Produces the thesis result table with **mean ± std** per metric.

### Metrics

| Metric | Description |
|--------|-------------|
| `Disc%` | % of episodes where ≥1 scout discovered fire |
| `Dwell%±σ` | % of steps FW was actively suppressing (water hit burning cells) |
| `Supp%±σ` | Fire suppression: `1 − end_cells / peak_cells` |
| `Refills±σ` | Tank refills per episode |
| `R_cmdr±σ` | Total commander episode reward |
| `FW_surv%` | % of episodes FW survived to the end |
| `Failures` | Episodes where FW died early OR scouts never found fire |

### Output

- Stdout summary table (one row per config)
- `results/eval_full_team.csv` — per-episode raw data for further analysis

### Usage

```bash
# From project root:
python demos/evaluation/eval_full_team.py \
    --scout  saved_models/finetune_07/scout_best.pt \
    --cmdr   saved_models/finetune_07/cmdr_best.pt \
    --episodes 50

# Custom configs:
python demos/evaluation/eval_full_team.py \
    --scout  saved_models/finetune_07/scout_best.pt \
    --cmdr   saved_models/finetune_07/cmdr_best.pt \
    --episodes 50 \
    --scouts 1 2 \
    --map-sizes 800 1200 2000 \
    --seed-start 1000
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scout` | `finetune_07/scout_best.pt` | ScoutActor checkpoint |
| `--cmdr` | `finetune_07/cmdr_best.pt` | CommanderActor checkpoint |
| `--episodes` | 50 | Episodes per configuration |
| `--scouts` | `1 2` | Scout counts to sweep |
| `--map-sizes` | `800 1200 2000` | Map sizes in metres |
| `--seed-start` | 1000 | First seed |
| `--csv` | `results/eval_full_team.csv` | Output CSV path |

### Example output

```
──────────────────────────────────────────────────────────────────────────────────────────────────────
    Config    Map  N  Disc%    Dwell%     Supp%   Refills     R_cmdr  FW_surv%  Failures
──────────────────────────────────────────────────────────────────────────────────────────────────────
  S1C1_800m   800m  1  82.0%   8.3±3.1%  34.2±18.4%  1.2±0.8   +180±95   94.0%    9/50
 S1C1_1200m  1200m  1  76.0%   6.1±2.8%  22.1±15.2%  1.0±0.7   +130±88   91.0%   12/50
  S2C1_800m   800m  2  96.0%  11.2±3.5%  48.7±17.1%  1.4±0.9   +220±87   95.0%    2/50
```

---

## demo_scout.py

Scout-only evaluation (no commander). See [`demos/scout/README.md`](../scout/README.md) for full details.

Quick reference:

```bash
python demos/scout/demo_scout.py \
    --model saved_models/finetune_07/scout_best.pt \
    --scouts 2 --seeds 50 --map-sizes 800 1200 2000
```

Metrics: `coverage_pct`, `avg_fire_seen`, `fire_discovered`, `R/scout` — exported to `scout_summary.csv`.
