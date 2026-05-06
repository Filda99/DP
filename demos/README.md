# demos/

Visualisation and evaluation scripts for the heterogeneous MAPPO team.  
Each subdirectory contains its own `README.md` with full usage details.

## Contents

| Directory | What it contains |
|---|---|
| [`scout/`](scout/README.md) | Scout-only demo — N quadcopters searching for fire, no commander |
| [`full_team/`](full_team/README.md) | Full team demo — scouts + fixed-wing commander |
| [`evaluation/`](evaluation/README.md) | Quantitative evaluation harness over many episodes |

## Quick start

```bash
cd ~/tmp/DP

# Scout demo — 2 scouts, 5 episodes
python demos/scout/demo_scout.py

# Full team demo — scouts + commander
python demos/full_team/demo_both_training.py

# Quantitative evaluation — 30 episodes
python demos/evaluation/demo_eval_2scouts.py --episodes 30
```

All scripts auto-detect the latest checkpoint from `saved_models/`. Use `--model` / `--scout` / `--commander` to specify a path.
