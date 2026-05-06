"""
eval_scout.py
=============
Headless quantitative evaluation of trained ScoutActor across multiple
map sizes, seed counts and swarm sizes.  No GIFs, no rendering — runs
fast (parallel workers optional).

Metrics per episode
-------------------
  surv%         Scout survival rate (%)
  disc          Fire discovered (bool)
  t_disc        Steps until first fire seen  (−1 if never)
  dwell%        % steps where ≥1 scout sees fire (coverage over time)
  dwell_each%   Per-scout dwell% (min / mean / max across scouts)
  sep_avg       Mean inter-scout separation [m] (multi-scout only)
  avg_seen      Mean fire-camera intensity when fire visible
  R/scout       Average reward per scout

Usage
-----
  python tools/eval_scout.py
  python tools/eval_scout.py --model saved_models/finetune_07/scout_best.pt
  python tools/eval_scout.py --scouts 2 --runs 30 --grid-sizes 800 1200 2000
  python tools/eval_scout.py --no-osm --runs 50 --scouts 1 2 3
"""

import argparse
import os
import sys
import csv
import glob
import random
import time

import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(os.path.join(PROJECT, "src"))

from env_core import DroneFireEnv
from models import ScoutActor

HIDDEN_DIM   = 128
NORM_DIST    = 1000.0
FIRE_VIS_THR = 0.1   # min local-map sum to count as "sees fire"


# =============================================================================
# Single episode
# =============================================================================

def run_episode(env, actor, seed, ep_num, n_quads, device,
                max_steps=500, n_fires=2, far_spawn=False):
    """Run one headless episode. Returns metrics dict.

    far_spawn=True: pass a high episode number (>25 000) so the curriculum
    places scouts far from the fire (full-difficulty regime).  This makes
    t_disc a meaningful navigation metric instead of always 0.
    """
    import pybullet as _p
    import random as _rnd

    quad_agents = [f"quad_{i}" for i in range(n_quads)]

    # Use a high episode number to trigger full-difficulty curriculum spawn
    curriculum_ep = 30_000 + seed if far_spawn else ep_num
    obs, _ = env.reset(seed=seed, epizode_number=curriculum_ep)

    # ── Spawn extra fires (env already placed 1 via curriculum) ─────────
    b = env.map_bounds * 0.65
    extra_positions = []
    _rnd.seed(seed * 7 + 13)
    attempts = 0
    existing = list(getattr(env, 'fire_positions', [(env.fire_x, env.fire_y)]))
    all_fire_pos = list(existing)
    while len(extra_positions) < (n_fires - len(existing)) and attempts < 200:
        attempts += 1
        fx = _rnd.uniform(-b, b)
        fy = _rnd.uniform(-b, b)
        if any(np.hypot(fx - px, fy - py) < 150.0 for px, py in all_fire_pos):
            continue
        extra_positions.append((fx, fy))
        all_fire_pos.append((fx, fy))
        env.sim.environment.start_fire_at_position(
            [fx, fy], intensity=_rnd.uniform(0.6, 0.9),
            radius_m=_rnd.uniform(6.0, 12.0))

    # Warm up fire for a few steps
    for _ in range(10):
        env.sim.environment.update_fire_simulation(real_dt=0.1)

    obs = {ag: env._get_obs(ag) for ag in env.agents}

    hidden = {ag: torch.zeros(1, 1, HIDDEN_DIM).to(device) for ag in quad_agents}

    # Per-step tracking
    total_rewards   = {ag: 0.0 for ag in quad_agents}
    steps_sees_fire = {ag: 0   for ag in quad_agents}   # dwell per scout
    steps_covered   = 0     # steps ≥1 scout sees fire
    t_discovery     = -1    # first step any scout sees fire
    sep_sum         = 0.0   # for mean separation
    sep_count       = 0
    total_steps     = 0
    fire_seen_vals  = []

    for step in range(max_steps):
        if not env.agents:
            break

        actions = {}
        for ag in quad_agents:
            if ag not in env.agents or ag not in env.sim.drones:
                continue
            l_map = torch.FloatTensor(obs[ag]["local_map"]).to(device).unsqueeze(0)
            s_st  = torch.FloatTensor(obs[ag]["self_state"]).to(device).unsqueeze(0)
            n_st  = torch.FloatTensor(obs[ag]["neighbor_states"]).to(device).unsqueeze(0)
            n_m   = torch.BoolTensor(obs[ag]["neighbor_mask"]).to(device).unsqueeze(0)
            with torch.no_grad():
                dist, _msg, h_out = actor(l_map, s_st, n_st, n_m, hidden[ag])
            hidden[ag] = h_out
            actions[ag] = dist.mean.squeeze(0).cpu().numpy()

        if not actions:
            break

        obs, rewards, _, _, _ = env.step(actions)
        total_steps += 1

        # ── Per-step metrics ──────────────────────────────────────────
        any_sees = False
        alive_positions = []
        for ag in quad_agents:
            total_rewards[ag] += rewards.get(ag, 0.0)
            if ag not in env.sim.drones:
                continue
            # fire visibility from local map
            lm = obs[ag]["local_map"][0] if ag in obs else None
            fire_val = float(np.sum(lm)) if lm is not None else 0.0
            if fire_val > FIRE_VIS_THR:
                steps_sees_fire[ag] += 1
                any_sees = True
                fire_seen_vals.append(fire_val)

            alive_positions.append(env.sim.drones[ag].get_position()[:2])

        if any_sees:
            steps_covered += 1
            if t_discovery < 0:
                t_discovery = step

        # Mean pairwise separation
        if len(alive_positions) >= 2:
            for i in range(len(alive_positions)):
                for j in range(i + 1, len(alive_positions)):
                    sep_sum += np.hypot(
                        alive_positions[i][0] - alive_positions[j][0],
                        alive_positions[i][1] - alive_positions[j][1])
                    sep_count += 1

    # ── Aggregate ─────────────────────────────────────────────────────
    surv = sum(1 for ag in quad_agents if ag in env.sim.drones)
    total_r = sum(total_rewards.values())
    ts = total_steps or 1

    dwell_each = [steps_sees_fire[ag] / ts * 100.0 for ag in quad_agents]

    return {
        "seed":         seed,
        "n_scouts":     n_quads,
        "map_m":        int(env.grid_size_m),
        "max_steps":    max_steps,
        "steps_done":   total_steps,
        "surv":         surv,
        "surv_pct":     round(surv / n_quads * 100.0, 1),
        "disc":         int(t_discovery >= 0),
        "t_disc":       t_discovery,
        "dwell_pct":    round(steps_covered / ts * 100.0, 1),
        "dwell_min":    round(min(dwell_each), 1),
        "dwell_mean":   round(float(np.mean(dwell_each)), 1),
        "dwell_max":    round(max(dwell_each), 1),
        "sep_avg_m":    round(sep_sum / sep_count, 1) if sep_count > 0 else -1.0,
        "avg_seen":     round(float(np.mean(fire_seen_vals)), 3) if fire_seen_vals else 0.0,
        "total_R":      round(total_r, 1),
        "R_per_scout":  round(total_r / n_quads, 1),
    }


# =============================================================================
# Pretty table
# =============================================================================

COLS = [
    ("seed",      "seed"),
    ("map[m]",    "map_m"),
    ("sc",        "n_scouts"),
    ("surv%",     "surv_pct"),
    ("disc",      "disc"),
    ("t_disc",    "t_disc"),
    ("dwell%",    "dwell_pct"),
    ("dw_min%",   "dwell_min"),
    ("dw_mean%",  "dwell_mean"),
    ("sep[m]",    "sep_avg_m"),
    ("avg_seen",  "avg_seen"),
    ("R/scout",   "R_per_scout"),
]


def print_table(rows):
    header = [h for h, _ in COLS]
    widths = [max(len(h), 6) for h in header]
    for row in rows:
        for i, (_, k) in enumerate(COLS):
            widths[i] = max(widths[i], len(str(row.get(k, ""))))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(row.get(k, "")) for _, k in COLS]))
    print(sep)


def print_aggregates(rows, grid_sizes, scout_counts):
    print("\nAggregated averages")
    print("─" * 70)
    for gs in grid_sizes:
        for ns in scout_counts:
            sub = [r for r in rows if r["map_m"] == int(gs) and r["n_scouts"] == ns]
            if not sub:
                continue
            disc_n  = sum(r["disc"] for r in sub)
            print(
                f"  map={int(gs):>5}m  scouts={ns}  n={len(sub):>3}  "
                f"surv={np.mean([r['surv_pct'] for r in sub]):5.1f}%  "
                f"disc={disc_n}/{len(sub)}  "
                f"dwell={np.mean([r['dwell_pct'] for r in sub]):5.1f}%  "
                f"dw_mean={np.mean([r['dwell_mean'] for r in sub]):5.1f}%  "
                f"sep={np.mean([r['sep_avg_m'] for r in sub if r['sep_avg_m'] > 0]):6.0f}m  "
                f"R/sc={np.mean([r['R_per_scout'] for r in sub]):+7.1f}"
            )
    print("─" * 70)


# =============================================================================
# Main
# =============================================================================

def _find_model():
    for pattern in [
        "saved_models/finetune_07/scout_best.pt",
        "saved_models/multi/scout_best.pt",
        "results/TrainingTogether/**/scout_best.pt",
        "saved_models/**/scout*.pt",
    ]:
        matches = sorted(glob.glob(os.path.join(PROJECT, pattern), recursive=True))
        if matches:
            return matches[-1]
    return None


def main():
    ap = argparse.ArgumentParser(description="Headless scout evaluation")
    ap.add_argument("--model",       default=None)
    ap.add_argument("--scouts",      type=int, nargs="+", default=[2],
                    help="Scout counts to evaluate (e.g. 1 2 3)")
    ap.add_argument("--runs",        type=int, default=20,
                    help="Episodes per (map_size × scout_count) combination")
    ap.add_argument("--seed-start",  type=int, default=0)
    ap.add_argument("--max-steps",   type=int, default=500)
    ap.add_argument("--grid-sizes",  type=float, nargs="+",
                    default=[800.0, 1200.0, 2000.0])
    ap.add_argument("--fires",       type=int, default=2,
                    help="Fire sources per episode")
    ap.add_argument("--far-spawn",   action="store_true",
                    help="Spawn scouts far from fire (full-difficulty curriculum). "
                         "Makes t_disc a real navigation metric instead of always 0.")
    ap.add_argument("--out",         default=os.path.join(PROJECT, "results",
                                                           "eval_scout.csv"))
    args = ap.parse_args()

    model_path = args.model or _find_model()
    if model_path is None:
        raise FileNotFoundError("No scout checkpoint found — use --model")
    print(f"Model : {model_path}")
    print(f"Scouts: {args.scouts}  |  Maps: {args.grid_sizes}  |  "
          f"Runs/combo: {args.runs}  |  Fires/ep: {args.fires}  |  "
          f"Far-spawn: {args.far_spawn}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load actor once (obs shape is map-size independent)
    _tmp = DroneFireEnv(num_quads=max(args.scouts), num_fixed=0,
                        grid_size_m=1000.0, max_steps=10)
    obs_sp = _tmp.observation_space("quad_0")

    actor = ScoutActor(self_state_dim=obs_sp["self_state"].shape[0],
                       msg_dim=5, hidden_dim=HIDDEN_DIM).to(device)
    sd = torch.load(model_path, map_location=device)
    actor.load_state_dict(sd, strict=False)
    actor.eval()
    print(f"Loaded {sum(p.numel() for p in actor.parameters()):,} parameters\n")

    all_rows = []
    global_seed = args.seed_start

    for n_scouts in args.scouts:
        for grid_size in args.grid_sizes:
            tag = f"scouts={n_scouts}  map={int(grid_size)}m"
            print(f"{'─'*50}\n{tag}\n{'─'*50}")

            env = DroneFireEnv(num_quads=n_scouts, num_fixed=0,
                               grid_size_m=grid_size, max_steps=args.max_steps)

            for run_i in range(args.runs):
                seed = global_seed
                global_seed += 1
                t0 = time.time()
                result = run_episode(
                    env, actor, seed, run_i, n_scouts, device,
                    max_steps=args.max_steps, n_fires=args.fires,
                    far_spawn=args.far_spawn)
                elapsed = time.time() - t0
                all_rows.append(result)
                print(
                    f"  [{run_i+1:>3}/{args.runs}] seed={seed}  "
                    f"surv={result['surv_pct']:5.1f}%  "
                    f"disc={result['disc']}  "
                    f"t_disc={result['t_disc']:>4}  "
                    f"dwell={result['dwell_pct']:5.1f}%  "
                    f"dw_mean={result['dwell_mean']:5.1f}%  "
                    f"R/sc={result['R_per_scout']:+7.1f}  "
                    f"({elapsed:.1f}s)")

            env.sim.stop_simulation()

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print_table(all_rows)
    print_aggregates(all_rows, args.grid_sizes, args.scouts)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV → {args.out}")


if __name__ == "__main__":
    main()
