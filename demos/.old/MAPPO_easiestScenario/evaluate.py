"""
evaluate.py
───────────
Multi-scenario evaluation harness for the heterogeneous MAPPO team.

Runs N episodes with different seeds and reports aggregated statistics,
so you can objectively compare checkpoints and spot recurring failure modes.

Usage:
    cd /homes/eva/xj/xjahnf00/tmp/DP
    python demos/evaluate.py                          # uses defaults
    python demos/evaluate.py --scout saved_models/scout_ep19400.pt \\
                              --commander saved_models/commander_ep19400.pt \\
                              --episodes 30 --seeds 0-29

Output:
    - Printed summary table (per-episode + aggregate)
    - demos/eval_results.png  — grid of box/scatter plots
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULTS  (override with CLI args)
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_SCOUT_PATH     = os.path.join(project_root, "saved_models", "scout_best.pt")
DEFAULT_COMMANDER_PATH = os.path.join(project_root, "saved_models", "commander_best.pt")
DEFAULT_N_EPISODES     = 20
DEFAULT_MAX_STEPS      = 2000
DEFAULT_GRID_SIZE      = 2000.0
DEFAULT_N_QUADS        = 1
DEFAULT_N_FIXED        = 1
OUTPUT_PATH            = os.path.join(project_root, "demos", "eval_results.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(scout_path, commander_path, n_quads, n_fixed, grid_size, max_steps, device):
    env_tmp = DroneFireEnv(num_quads=n_quads, num_fixed=n_fixed,
                           grid_size_m=grid_size, max_steps=max_steps)
    scout_self_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0] if n_quads > 0 else 15
    fixed_self_dim = env_tmp.observation_space(env_tmp.fixed_agents[0])["self_state"].shape[0] if n_fixed > 0 else 17
    # env_tmp.sim.stop_simulation()

    scout_actor = None
    if n_quads > 0:
        scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=5, hidden_dim=128).to(device)
        if os.path.exists(scout_path):
            scout_actor.load_state_dict(torch.load(scout_path, map_location=device))
            print(f"  ✅ Scout:     {scout_path}")
        else:
            print(f"  ⚠️  Scout:     NOT FOUND ({scout_path}) — random weights")
        scout_actor.eval()

    commander_actor = None
    if n_fixed > 0:
        commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=5).to(device)
        if os.path.exists(commander_path):
            commander_actor.load_state_dict(torch.load(commander_path, map_location=device))
            print(f"  ✅ Commander: {commander_path}")
        else:
            print(f"  ⚠️  Commander: NOT FOUND ({commander_path}) — random weights")
        commander_actor.eval()

    return scout_actor, commander_actor


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(env, scout_actor, commander_actor, seed, episode_number, device):
    """
    Run one full episode and return a flat dict of scalar metrics.

    Metrics collected
    -----------------
    total_reward_scout     : sum of per-step scout rewards
    total_reward_commander : sum of per-step commander rewards
    scout_lifespan         : steps scout was alive
    commander_lifespan     : steps commander was alive
    scout_death_cause      : 'survived' | 'ground' | 'boundary' | 'ceiling' | 'physics'
    commander_death_cause  : same
    fire_seen_steps        : steps where scout local fire intensity > 0.01
    max_fire_intensity     : peak fire intensity seen by scout
    water_dropped_pct      : fraction of water tank used (0→1)
    fire_extinguished      : total extinguish effectiveness accumulated
    scout_max_alt          : max altitude reached by scout
    commander_max_alt      : max altitude reached by commander
    commander_min_alt      : minimum altitude (how close to ground)
    """
    obs, _ = env.reset(seed=seed, epizode_number=episode_number)

    h_scout = torch.zeros(1, 1, 128).to(device)
    h_cmdr  = torch.zeros(1, 1, 128).to(device)

    metrics = {
        "total_reward_scout":     0.0,
        "total_reward_commander": 0.0,
        "scout_lifespan":         0,
        "commander_lifespan":     0,
        "scout_death_cause":      "survived",
        "commander_death_cause":  "survived",
        "fire_seen_steps":        0,
        "max_fire_intensity":     0.0,
        "water_dropped_pct":      0.0,
        "fire_extinguished":      0.0,
        "scout_max_alt":          0.0,
        "commander_max_alt":      0.0,
        "commander_min_alt":      9999.0,
    }

    scout_alive_last = True
    commander_alive_last = True
    water_start = 1.0

    for step in range(env.max_steps):
        if not env.agents:
            break

        actions = {}
        scout_msg = torch.zeros(1, 1, 5).to(device)

        # ── Scout forward pass ─────────────────────────────────────────────
        if scout_actor is not None and "quad_0" in env.agents:
            l_map = torch.FloatTensor(obs["quad_0"]["local_map"]).unsqueeze(0).to(device)
            s_st  = torch.FloatTensor(obs["quad_0"]["self_state"]).unsqueeze(0).to(device)
            n_st  = torch.FloatTensor(obs["quad_0"]["neighbor_states"]).unsqueeze(0).to(device)
            n_m   = torch.BoolTensor(obs["quad_0"]["neighbor_mask"]).unsqueeze(0).to(device)
            with torch.no_grad():
                dist, msg, h_scout = scout_actor(l_map, s_st, n_st, n_m, h_scout)
            scout_msg = msg.unsqueeze(1)           # [1, msg_dim] → [1, 1, msg_dim]
            actions["quad_0"] = dist.mean.squeeze(0).cpu().numpy()

            # Metrics
            intensity = float(obs["quad_0"]["self_state"][14])
            if intensity > 0.01:
                metrics["fire_seen_steps"] += 1
            metrics["max_fire_intensity"] = max(metrics["max_fire_intensity"], intensity)
            alt = float(obs["quad_0"]["self_state"][2]) * env.map_bounds
            metrics["scout_max_alt"] = max(metrics["scout_max_alt"], alt)

        # ── Commander forward pass ─────────────────────────────────────────
        if commander_actor is not None and "fixed_0" in env.agents:
            s_st_f = torch.FloatTensor(obs["fixed_0"]["self_state"]).unsqueeze(0).to(device)
            scout_present = "quad_0" in env.agents
            m_m = torch.BoolTensor([[not scout_present]]).to(device)
            with torch.no_grad():
                dist, _, h_cmdr = commander_actor(s_st_f, scout_msg, m_m, h_cmdr)
            actions["fixed_0"] = dist.mean.squeeze(0).cpu().numpy()

            # Metrics
            water_lvl = float(obs["fixed_0"]["self_state"][10])
            fw_z = float(obs["fixed_0"]["self_state"][2]) * env.map_bounds
            metrics["commander_max_alt"] = max(metrics["commander_max_alt"], fw_z)
            metrics["commander_min_alt"] = min(metrics["commander_min_alt"], fw_z)

        # ── Step ───────────────────────────────────────────────────────────
        obs, rewards, terminations, truncations, _ = env.step(actions)

        metrics["total_reward_scout"]     += rewards.get("quad_0", 0.0)
        metrics["total_reward_commander"] += rewards.get("fixed_0", 0.0)

        # extinguish effectiveness
        for f_agent in env.fixed_agents:
            metrics["fire_extinguished"] += env.sim.drone_extinguish_stats.get(f_agent, 0.0)

        # Lifespan tracking
        if "quad_0" in env.possible_agents:
            if "quad_0" in env.agents or (scout_alive_last and terminations.get("quad_0", False)):
                metrics["scout_lifespan"] = step + 1
            if terminations.get("quad_0", False) and scout_alive_last:
                scout_alive_last = False
                # Diagnose death cause from position
                if "quad_0" in env.sim.drones:
                    pass  # died by truncation — shouldn't happen but guard
                else:
                    pos_x = float(obs.get("quad_0", {}).get("self_state", [0]*3)[0]) * env.map_bounds
                    pos_z = float(obs.get("quad_0", {}).get("self_state", [0]*3)[2]) * env.map_bounds
                    if abs(pos_x) > env.map_bounds * 0.98:
                        metrics["scout_death_cause"] = "boundary"
                    elif pos_z < 2.0:
                        metrics["scout_death_cause"] = "ground"
                    elif pos_z > 200.0:
                        metrics["scout_death_cause"] = "ceiling"
                    else:
                        metrics["scout_death_cause"] = "physics"

        if "fixed_0" in env.possible_agents:
            if "fixed_0" in env.agents or (commander_alive_last and terminations.get("fixed_0", False)):
                metrics["commander_lifespan"] = step + 1
            if terminations.get("fixed_0", False) and commander_alive_last:
                commander_alive_last = False
                pos_x = float(obs.get("fixed_0", {}).get("self_state", [0]*3)[0]) * env.map_bounds
                pos_z = float(obs.get("fixed_0", {}).get("self_state", [0]*3)[2]) * env.map_bounds
                if abs(pos_x) > env.map_bounds * 0.98:
                    metrics["commander_death_cause"] = "boundary"
                elif pos_z < 2.0:
                    metrics["commander_death_cause"] = "ground"
                elif pos_z > 400.0:
                    metrics["commander_death_cause"] = "ceiling"
                else:
                    metrics["commander_death_cause"] = "physics"

    # Water usage: compare final level to start
    if commander_actor is not None and "fixed_0" in env.possible_agents:
        final_water = obs.get("fixed_0", {}).get("self_state", None)
        if final_water is not None:
            metrics["water_dropped_pct"] = max(0.0, 1.0 - float(final_water[10]))

    # Clamp min alt if commander never existed
    if metrics["commander_min_alt"] == 9999.0:
        metrics["commander_min_alt"] = 0.0

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(all_metrics, seeds):
    """Print a per-episode table and aggregated statistics."""
    cols = [
        ("Seed", "seed", 5),
        ("Ep#",  "episode_number", 5),
        ("R_scout", "total_reward_scout", 9),
        ("R_cmdr",  "total_reward_commander", 9),
        ("S_life", "scout_lifespan", 7),
        ("C_life", "commander_lifespan", 7),
        ("FireSeen", "fire_seen_steps", 9),
        ("MaxIntens", "max_fire_intensity", 10),
        ("Water%",  "water_dropped_pct", 7),
        ("Extg",    "fire_extinguished", 8),
        ("S_death", "scout_death_cause", 9),
        ("C_death", "commander_death_cause", 9),
    ]

    header = "  ".join(f"{label:>{w}}" for label, _, w in cols)
    print("\n" + "═" * len(header))
    print("  EVALUATION RESULTS")
    print("═" * len(header))
    print(header)
    print("─" * len(header))

    for m, seed in zip(all_metrics, seeds):
        row = []
        for label, key, w in cols:
            val = m.get(key, "—")
            if isinstance(val, float):
                row.append(f"{val:>{w}.2f}")
            else:
                row.append(f"{str(val):>{w}}")
        print("  ".join(row))

    print("─" * len(header))
    # Aggregate
    numeric_keys = ["total_reward_scout", "total_reward_commander",
                    "scout_lifespan", "commander_lifespan",
                    "fire_seen_steps", "max_fire_intensity",
                    "water_dropped_pct", "fire_extinguished"]
    print("\n  AGGREGATE (mean ± std):")
    for key in numeric_keys:
        vals = [m[key] for m in all_metrics]
        print(f"    {key:<30s}  {np.mean(vals):8.2f}  ±  {np.std(vals):.2f}")

    death_keys = ["scout_death_cause", "commander_death_cause"]
    for key in death_keys:
        from collections import Counter
        counts = Counter(m[key] for m in all_metrics)
        print(f"    {key:<30s}  {dict(counts)}")

    pct_saw_fire = 100 * np.mean([m["fire_seen_steps"] > 0 for m in all_metrics])
    pct_fw_survived = 100 * np.mean([m["commander_lifespan"] == DEFAULT_MAX_STEPS for m in all_metrics])
    print(f"\n    Scout saw fire in      {pct_saw_fire:.0f}% of episodes")
    print(f"    Commander survived     {pct_fw_survived:.0f}% of episodes")
    print("═" * len(header) + "\n")


def save_plots(all_metrics, output_path):
    """Save a 3×3 grid of plots summarising evaluation results."""
    n = len(all_metrics)
    xs = list(range(n))

    def vals(key):
        return [m[key] for m in all_metrics]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(f"Evaluation across {n} episodes", fontsize=13, fontweight='bold')

    kw_scatter = dict(s=40, zorder=3)
    kw_line    = dict(linewidth=1.2, alpha=0.7)

    # Row 0 — rewards
    ax = axes[0, 0]
    ax.plot(xs, vals("total_reward_scout"),     color="cyan",   label="Scout",     **kw_line)
    ax.plot(xs, vals("total_reward_commander"), color="tomato", label="Commander", **kw_line)
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax.set_title("Total reward per episode"); ax.legend(fontsize=7); ax.set_xlabel("episode")

    ax = axes[0, 1]
    ax.scatter(xs, vals("scout_lifespan"),     color="cyan",   **kw_scatter, label="Scout")
    ax.scatter(xs, vals("commander_lifespan"), color="tomato", **kw_scatter, label="Commander")
    ax.axhline(DEFAULT_MAX_STEPS, color="grey", linewidth=0.8, linestyle="--", label="max")
    ax.set_title("Lifespan (steps)"); ax.legend(fontsize=7); ax.set_xlabel("episode")

    ax = axes[0, 2]
    ax.scatter(xs, vals("fire_seen_steps"), color="orange", **kw_scatter)
    ax.set_title("Steps scout saw fire"); ax.set_xlabel("episode")
    ax.axhline(0, color="grey", linewidth=0.5)

    # Row 1 — mission quality
    ax = axes[1, 0]
    ax.scatter(xs, vals("max_fire_intensity"), color="darkorange", **kw_scatter)
    ax.set_title("Peak fire intensity seen by scout"); ax.set_xlabel("episode")

    ax = axes[1, 1]
    ax.bar(xs, vals("water_dropped_pct"), color="deepskyblue", alpha=0.7)
    ax.set_ylim(0, 1.05); ax.set_title("Water used (fraction)"); ax.set_xlabel("episode")

    ax = axes[1, 2]
    ax.scatter(xs, vals("fire_extinguished"), color="red", **kw_scatter)
    ax.set_title("Fire extinguished (cumulative eff.)"); ax.set_xlabel("episode")

    # Row 2 — altitude diagnostics
    ax = axes[2, 0]
    ax.scatter(xs, vals("scout_max_alt"), color="cyan", **kw_scatter)
    ax.axhline(120, color="grey", linewidth=0.8, linestyle="--", label="ideal max 120m")
    ax.axhline(40,  color="grey", linewidth=0.8, linestyle=":",  label="ideal min 40m")
    ax.set_title("Scout max altitude"); ax.legend(fontsize=7); ax.set_xlabel("episode")

    ax = axes[2, 1]
    ax.scatter(xs, vals("commander_max_alt"), color="tomato", **kw_scatter, label="max alt")
    ax.scatter(xs, vals("commander_min_alt"), color="salmon",  **kw_scatter, marker="v", label="min alt")
    ax.axhline(5, color="red", linewidth=0.8, linestyle="--", label="ground crash zone")
    ax.set_title("Commander altitude range"); ax.legend(fontsize=7); ax.set_xlabel("episode")

    # Death cause distribution (pie charts)
    ax = axes[2, 2]
    from collections import Counter
    scout_deaths   = Counter(m["scout_death_cause"]   for m in all_metrics)
    cmdr_deaths    = Counter(m["commander_death_cause"] for m in all_metrics)
    labels = sorted(set(list(scout_deaths.keys()) + list(cmdr_deaths.keys())))
    x_pos = np.arange(len(labels))
    w = 0.35
    ax.bar(x_pos - w/2, [scout_deaths.get(l, 0) for l in labels], w, label="Scout",     color="cyan",   alpha=0.8)
    ax.bar(x_pos + w/2, [cmdr_deaths.get(l, 0)  for l in labels], w, label="Commander", color="tomato", alpha=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_title("Death causes"); ax.legend(fontsize=7)

    for ax_row in axes:
        for ax in ax_row:
            ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"✅ Plot saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Multi-scenario MAPPO evaluation")
    p.add_argument("--scout",     default=DEFAULT_SCOUT_PATH,     help="Path to scout .pt checkpoint")
    p.add_argument("--commander", default=DEFAULT_COMMANDER_PATH, help="Path to commander .pt checkpoint")
    p.add_argument("--episodes",  type=int, default=DEFAULT_N_EPISODES, help="Number of evaluation episodes")
    p.add_argument("--seed-start", type=int, default=100, help="First episode seed (also used as episode_number for curriculum)")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--no-fixed",  action="store_true", help="Disable commander (scout-only evaluation)")
    p.add_argument("--output",    default=OUTPUT_PATH, help="Output plot path")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_quads = DEFAULT_N_QUADS
    n_fixed = 0 if args.no_fixed else DEFAULT_N_FIXED

    print(f"\n🔍 Evaluation config:")
    print(f"   Episodes:  {args.episodes}")
    print(f"   Seeds:     {args.seed_start} → {args.seed_start + args.episodes - 1}")
    print(f"   N_QUADS:   {n_quads}   N_FIXED: {n_fixed}")
    print(f"   Device:    {device}")
    print(f"   Max steps: {args.max_steps}")

    scout_actor, commander_actor = load_models(
        args.scout, args.commander,
        n_quads, n_fixed, DEFAULT_GRID_SIZE, args.max_steps, device
    )

    env = DroneFireEnv(num_quads=n_quads, num_fixed=n_fixed,
                       grid_size_m=DEFAULT_GRID_SIZE, max_steps=args.max_steps)

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    all_metrics = []

    print(f"\n▶ Running {args.episodes} episodes...\n")
    for i, seed in enumerate(seeds):
        episode_number = seed  # seeds > 1500 → random fire position (curriculum)
        m = run_episode(env, scout_actor, commander_actor, seed, episode_number, device)
        m["seed"] = seed
        m["episode_number"] = episode_number
        all_metrics.append(m)

        # One-line progress
        print(f"  [{i+1:3d}/{args.episodes}] seed={seed:4d} | "
              f"R_scout={m['total_reward_scout']:7.1f}  R_cmdr={m['total_reward_commander']:7.1f} | "
              f"S_life={m['scout_lifespan']:4d}  C_life={m['commander_lifespan']:4d} | "
              f"fire_steps={m['fire_seen_steps']:4d}  extg={m['fire_extinguished']:.3f} | "
              f"C_death={m['commander_death_cause']}")

    print_table(all_metrics, seeds)
    save_plots(all_metrics, args.output)


if __name__ == "__main__":
    main()
