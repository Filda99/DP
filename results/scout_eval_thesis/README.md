# Scout Evaluation Results

## Setup

- **Model**: `results/TrainingQuad/050626_TrainingMultipleScoutsOnMultipleFires/scout_b0120.pt`
- **Architecture**: ScoutActor — CNN(1→16→32) + self-attention + GRU(128)
- **Configurations**: 5 scout counts (1–5) × 8 map sizes (500–4000 m) = 40 configs
- **Episodes per config**: 6 (seeds 0–5), total 240 episodes
- **Max steps**: 500 per episode
- **Environment**: OSM terrain (49.35°N, 16.42°E), single fire source

## Files

| File | Description |
|------|-------------|
| `eval_scout.csv` | Raw per-episode data (240 rows) |
| `scout_eval_summary.txt` | Aggregated table + LaTeX code |
| `scout_eval_lines.pdf` | Line plots: 6 metrics vs map size, per scout count |
| `scout_eval_heatmaps.pdf` | Heatmaps: scouts × map size for 4 key metrics |
| `scout_eval_discovery.pdf` | Discovery time vs map size (error bars) |
| `scout_eval_dwell_bars.pdf` | Grouped bar chart: fire visibility by config |

## Key Findings

### 1. Survival — Near Perfect

Survival rate is 100% across nearly all configurations. The only exception is
4 scouts on a 4000 m map (95.8%), where one drone crashed in 24 total. The
scout policy is highly robust — it learned altitude control and boundary
avoidance reliably.

### 2. Fire Discovery — Scales with Scout Count

Discovery rate (% of episodes where at least one scout finds the fire) degrades
with map size but improves with more scouts:

- **500–800 m**: 100% discovery across all scout counts — trivial maps.
- **1200–1500 m**: 83–100% with 1–3 scouts, 100% with 4–5.
- **2000–2500 m**: Single-scout discovery drops to 33–83%. With 4+ scouts,
  83–100% is achievable.
- **3000–4000 m**: Even 5 scouts only reach 67–83%. Maps are near the limit
  of what the current policy can handle.

### 3. Fire Visibility (Dwell) — The Core Metric

Team dwell (% of episode steps where *any* scout sees fire) shows the clearest
scaling pattern:

- **500 m**: 87–100% dwell regardless of scout count.
- **1200 m**: 66% (1 scout) → 89% (3 scouts) → 94% (5 scouts).
- **2000 m**: 47% (2 scouts) → 77% (4 scouts) — substantial improvement
  with each additional scout.
- **4000 m**: 19–63%, high variance. Only 3+ scouts achieve >50%.

**Optimal operating point**: 3–4 scouts on 1000–1500 m maps yield >80% dwell
with near-zero crash rate.

### 4. Discovery Time — Logarithmic Growth

Discovery time (steps until first scout sees fire) grows roughly logarithmically
with map area. On 500 m maps, discovery is nearly instant (0–10 steps). On
2000 m maps, 2–3 scouts need ~125–150 steps. Adding more scouts reduces
discovery time but with diminishing returns.

### 5. Per-Scout Efficiency — Diminishing Returns

Per-scout mean dwell (how much each individual scout contributes) peaks at
2–3 scouts and declines with 4–5 scouts. This indicates redundancy: scouts
begin overlapping coverage areas. The separation metric confirms this — mean
inter-scout distance is only 22–37 m on 500 m maps but 300–875 m on 4000 m maps,
showing scouts spread out but cannot cover the full area.

### 6. Reward Correlation

Reward per scout correlates well with dwell/discovery performance on small-medium
maps (+200–330 on 500–800 m). On large maps (3000–4000 m), reward becomes
negative for some configurations, reflecting penalties from exploration without
fire contact and the difficulty of the task.

## Limitations

- **6 episodes per config** provides limited statistical power. Confidence
  intervals are wide, especially for large maps where outcomes are binary
  (fire found/not found).
- **Single fire source** — multi-fire scenarios may show different scaling.
- **Fixed model** — the scout was trained primarily on 800–1500 m maps;
  performance on 3000–4000 m maps reflects generalization, not optimized
  behavior.
