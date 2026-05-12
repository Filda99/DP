#!/usr/bin/env python3
"""
eval_comprehensive.py — Full multi-agent system evaluation
==========================================================

Runs 500+ episodes across diverse configurations:
  - Map sizes:    800m, 1200m, 2000m
  - Scout counts: 2, 3, 4
  - FW counts:    2, 3
  - Fire counts:  1, 2, 3
  - Episode len:  1000, 1500 steps

Outputs:
  results/eval_comprehensive/data.csv       — raw per-episode data
  results/eval_comprehensive/summary.txt    — aggregated table + LaTeX
  results/eval_comprehensive/*.pdf          — publication-quality graphs

Usage:
  python tools/eval_comprehensive.py \\
      --scout-model saved_models/finetune/scout_best.pt \\
      --cmdr-model  saved_models/finetune/cmdr_best.pt \\
      --runs-per-config 20 \\
      --out-dir results/eval_comprehensive
"""

import sys, os, argparse, csv, random, time, glob, json
from itertools import product
from collections import defaultdict

import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(os.path.join(PROJECT, "src"))

from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor
from commander_control import CommanderController

# ── Constants ────────────────────────────────────────────────────────────────
HIDDEN_DIM_SCOUT = 128
HIDDEN_DIM_CMDR  = 64
WAYPOINT_STEPS   = 30
WAYPOINT_RANGE   = 200.0
WP_REACHED_DIST  = 30.0
NORM_DIST        = 1000.0
WATER_CAPACITY   = 200.0
SCOUT_MSG_DIM    = 5


# ── Fire stats utility ──────────────────────────────────────────────────────

def _fire_stats(env):
    fg = env.sim.environment.fire_grid
    if fg is None:
        return {"burning": 0, "intensity": 0.0, "burned_frac": 0.0,
                "peak_burning": 0}
    B, I = fg.B, fg.I
    burning = int(B.sum())
    i_sum  = float(I.sum())
    burned = int((I > 0.01).sum())
    return {
        "burning":      burning,
        "intensity":    round(i_sum, 3),
        "burned_frac":  round(burned / B.size * 100.0, 2),
        "peak_burning": burning,
    }


# ── Single episode ──────────────────────────────────────────────────────────

def run_episode(scout_actor, cmdr_actor, device,
                n_quads, n_fw, seed, grid_size_m, max_steps,
                n_fires) -> dict:
    """Run one full episode and return a dict of metrics."""

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DroneFireEnv(
        num_quads=n_quads, num_fixed=n_fw,
        grid_size_m=grid_size_m, max_steps=max_steps,
        n_fires_range=(n_fires, n_fires),
    )
    obs, _ = env.reset(epizode_number=seed)

    quad_agents  = list(env.quad_agents)
    fixed_agents = list(env.fixed_agents)

    # ── GRU hidden states ──
    h_scout = {q: torch.zeros(1, 1, HIDDEN_DIM_SCOUT).to(device) for q in quad_agents}
    h_cmdr  = {f: torch.zeros(1, 1, HIDDEN_DIM_CMDR).to(device)  for f in fixed_agents}

    # ── Tracking variables ──
    scout_alive       = {q: True  for q in quad_agents}
    fw_alive          = {f: True  for f in fixed_agents}
    scout_rewards     = {q: 0.0   for q in quad_agents}
    fw_rewards        = {f: 0.0   for f in fixed_agents}
    scout_fire_steps  = {q: 0     for q in quad_agents}
    scout_total_steps = {q: 0     for q in quad_agents}
    scout_death_cause = {q: "survived" for q in quad_agents}
    fw_death_cause    = {f: "survived" for f in fixed_agents}
    fw_refill_count   = {f: 0     for f in fixed_agents}
    fw_prev_water     = {}
    fw_water_start    = {}
    fw_water_end      = {}
    fw_valve_steps    = {f: 0     for f in fixed_agents}
    fw_total_steps    = {f: 0     for f in fixed_agents}
    fw_mode_steps     = {f: {"nn": 0, "scripted": 0, "emergency": 0}
                         for f in fixed_agents}
    fw_death_step     = {f: -1    for f in fixed_agents}

    # Water drop tracking
    wd_total = 0
    wd_hit   = 0
    wd_alts  = []
    wd_dists = []

    last_scout_msgs = {q: torch.zeros(1, SCOUT_MSG_DIM) for q in quad_agents}
    last_scout_valid = {q: False for q in quad_agents}
    fire_discovered = False
    t_discovery     = -1
    steps_done      = 0

    # Fire tracking
    peak_burning = 0
    fire_history = []

    # Per-FW CommanderController
    cmdr_ctrl = {}
    for f in fixed_agents:
        ctrl = CommanderController(WAYPOINT_RANGE, WAYPOINT_STEPS, WP_REACHED_DIST)
        ctrl.reset(env.map_bounds)
        cmdr_ctrl[f] = ctrl

    # Init water
    for f in fixed_agents:
        d = env.sim.drones.get(f)
        if d:
            fw_water_start[f] = d.current_water
            fw_prev_water[f]  = d.current_water
        else:
            fw_water_start[f] = WATER_CAPACITY
            fw_prev_water[f]  = WATER_CAPACITY

    fs_init = _fire_stats(env)

    # ── Main loop ──
    for step in range(max_steps):
        if not env.agents:
            break

        actions = {}

        # --- Scouts ---
        for q in quad_agents:
            if not scout_alive[q] or q not in env.agents:
                scout_alive[q] = False
                continue
            scout_total_steps[q] += 1
            with torch.no_grad():
                lm = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0).to(device)
                ss = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0).to(device)
                ns = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0).to(device)
                nm = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0).to(device)
                dist, msg, h_out = scout_actor(lm, ss, ns, nm, h_scout[q])
                act = dist.mean
            h_scout[q] = h_out
            actions[q] = act.squeeze(0).cpu().numpy()
            last_scout_msgs[q] = msg.detach().cpu()
            last_scout_valid[q] = True

            fire_in_cam = float(np.sum(obs[q]["local_map"]))
            if fire_in_cam > 0.1:
                scout_fire_steps[q] += 1
                if not fire_discovered:
                    fire_discovered = True
                    t_discovery = step

        # --- Commander (FW) ---
        for f in fixed_agents:
            if not fw_alive[f] or f not in env.agents:
                fw_alive[f] = False
                continue
            fw_total_steps[f] += 1

            drone = env.sim.drones.get(f)
            if drone is None:
                fw_alive[f] = False
                continue

            # Build scout message tensor
            msg_list = []
            mask_list = []
            for q in quad_agents:
                msg_list.append(last_scout_msgs[q])
                mask_list.append(not (scout_alive.get(q, False) and last_scout_valid.get(q, False)))
            msgs_t = torch.stack(msg_list, dim=1).to(device)  # (1, n_scouts, msg_dim)
            mask_t = torch.BoolTensor([mask_list]).to(device)

            # Build FW neighbor tensor
            pos_f = drone.get_position()
            map_half = env.map_bounds
            fw_neigh_list = []
            fw_mask_list = []
            for f2 in fixed_agents:
                if f2 == f:
                    continue
                d2 = env.sim.drones.get(f2)
                if d2 is not None and fw_alive.get(f2, False):
                    p2 = d2.get_position()
                    fw_neigh_list.append([
                        (p2[0] - pos_f[0]) / max(map_half, 1.0),
                        (p2[1] - pos_f[1]) / max(map_half, 1.0),
                        (p2[2] - pos_f[2]) / 100.0
                    ])
                    fw_mask_list.append(False)
                else:
                    fw_neigh_list.append([0.0, 0.0, 0.0])
                    fw_mask_list.append(True)
            if not fw_neigh_list:
                fw_neigh_list = [[0.0, 0.0, 0.0]]
                fw_mask_list = [True]
            fw_neigh_t = torch.FloatTensor([fw_neigh_list]).to(device)
            fw_mask_t_n = torch.BoolTensor([fw_mask_list]).to(device)

            # Also update cached scout messages for valve logic
            cmdr_ctrl[f].last_scout_msgs = msgs_t
            cmdr_ctrl[f].last_scout_mask = mask_t

            action, h_cmdr[f], ctrl_info = cmdr_ctrl[f].step(
                drone, obs[f]["self_state"], env,
                cmdr_actor, h_cmdr[f], msgs_t, mask_t,
                deterministic=True,
                fw_neighbor_states=fw_neigh_t, fw_neighbor_mask=fw_mask_t_n)
            actions[f] = action

            # Track mode
            if ctrl_info.get('in_emergency'):
                fw_mode_steps[f]["emergency"] += 1
            elif ctrl_info.get('scripted'):
                fw_mode_steps[f]["scripted"] += 1
            else:
                fw_mode_steps[f]["nn"] += 1

            # Valve tracking
            if len(action) > 2 and action[2] > 0.5:
                fw_valve_steps[f] += 1

        # Environment step
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps_done += 1

        # Rewards
        for q in quad_agents:
            scout_rewards[q] += rewards.get(q, 0.0)
        for f in fixed_agents:
            fw_rewards[f] += rewards.get(f, 0.0)

        # Water drop tracking from infos
        for f in fixed_agents:
            inf = infos.get(f, {})
            if "wd_alt" in inf:
                wd_total += 1
                wd_alts.append(inf["wd_alt"])
                wd_dists.append(inf.get("wd_dist", 0))
                if inf.get("wd_eff", 0) > 0:
                    wd_hit += 1

        # Refill detection
        for f in fixed_agents:
            d = env.sim.drones.get(f)
            if d and d.water_capacity > 0:
                cur_w = d.current_water
                if cur_w > fw_prev_water.get(f, 0) + 1.0:
                    fw_refill_count[f] += 1
                fw_prev_water[f] = cur_w
                fw_water_end[f] = cur_w

        # Fire tracking
        fs_now = _fire_stats(env)
        if fs_now["burning"] > peak_burning:
            peak_burning = fs_now["burning"]
        if step % 50 == 0:
            fire_history.append((step, fs_now["burning"], fs_now["intensity"]))

        # Deaths
        for q in quad_agents:
            if terms.get(q, False) and scout_alive[q]:
                scout_alive[q] = False
                dc = infos.get(q, {}).get("death_cause", "unknown")
                scout_death_cause[q] = dc
        for f in fixed_agents:
            if terms.get(f, False) and fw_alive[f]:
                fw_alive[f] = False
                dc = infos.get(f, {}).get("death_cause", "unknown")
                fw_death_cause[f] = dc
                fw_death_step[f] = step

    # ── Final stats ──
    fs_final = _fire_stats(env)
    for f in fixed_agents:
        if f not in fw_water_end:
            d = env.sim.drones.get(f)
            fw_water_end[f] = d.current_water if d else 0.0

    # Aggregation
    scouts_survived = sum(1 for q in quad_agents if scout_alive[q])
    fw_survived     = sum(1 for f in fixed_agents if fw_alive[f])

    dwell_vals = []
    for q in quad_agents:
        tot = scout_total_steps[q]
        dwell_vals.append(scout_fire_steps[q] / tot * 100.0 if tot > 0 else 0.0)

    total_water = 0.0
    total_refills = 0
    for f in fixed_agents:
        dropped = (fw_water_start.get(f, WATER_CAPACITY)
                   + fw_refill_count[f] * WATER_CAPACITY
                   - fw_water_end.get(f, 0.0))
        total_water += max(0.0, dropped)
        total_refills += fw_refill_count[f]

    valve_pct = 0.0
    tot_fw_st = sum(fw_total_steps[f] for f in fixed_agents)
    tot_valve = sum(fw_valve_steps[f] for f in fixed_agents)
    if tot_fw_st > 0:
        valve_pct = tot_valve / tot_fw_st * 100.0

    # Mode breakdown (averaged)
    mode_nn_pct  = 0.0
    mode_sc_pct  = 0.0
    mode_em_pct  = 0.0
    if tot_fw_st > 0:
        mode_nn_pct = sum(fw_mode_steps[f]["nn"] for f in fixed_agents) / tot_fw_st * 100
        mode_sc_pct = sum(fw_mode_steps[f]["scripted"] for f in fixed_agents) / tot_fw_st * 100
        mode_em_pct = sum(fw_mode_steps[f]["emergency"] for f in fixed_agents) / tot_fw_st * 100

    # Scout death breakdown
    sc_death_boundary = sum(1 for q in quad_agents if scout_death_cause[q] == "boundary")
    sc_death_ground   = sum(1 for q in quad_agents if scout_death_cause[q] == "ground_crash")
    sc_death_ceiling  = sum(1 for q in quad_agents if scout_death_cause[q] == "ceiling")
    sc_death_other    = n_quads - scouts_survived - sc_death_boundary - sc_death_ground - sc_death_ceiling

    # FW death breakdown
    fw_death_boundary = sum(1 for f in fixed_agents if fw_death_cause[f] == "boundary")
    fw_death_ground   = sum(1 for f in fixed_agents if fw_death_cause[f] == "ground_crash")
    fw_death_other    = n_fw - fw_survived - fw_death_boundary - fw_death_ground

    # Fire outcome
    fire_reduction = fs_init["intensity"] - fs_final["intensity"]
    suppressed = peak_burning - fs_final["burning"]
    fire_extinguished = fs_final["burning"] == 0

    # Water accuracy
    accuracy = wd_hit / wd_total * 100 if wd_total > 0 else 0.0
    avg_wd_alt  = np.mean(wd_alts) if wd_alts else 0.0
    avg_wd_dist = np.mean(wd_dists) if wd_dists else 0.0

    avg_scout_r = sum(scout_rewards.values()) / max(n_quads, 1)
    avg_fw_r    = sum(fw_rewards.values()) / max(n_fw, 1)

    env.sim.stop_simulation()

    return {
        "seed":              seed,
        "map_size_m":        int(grid_size_m),
        "max_steps":         max_steps,
        "steps_done":        steps_done,
        "n_scouts":          n_quads,
        "n_fw":              n_fw,
        "n_fires":           n_fires,
        # Scout
        "scouts_survived":   scouts_survived,
        "scouts_surv_pct":   round(scouts_survived / n_quads * 100, 1),
        "sc_death_boundary": sc_death_boundary,
        "sc_death_ground":   sc_death_ground,
        "sc_death_ceiling":  sc_death_ceiling,
        "fire_discovered":   int(fire_discovered),
        "t_discovery":       t_discovery,
        "scout_dwell_pct":   round(float(np.mean(dwell_vals)), 1),
        "scout_avg_reward":  round(avg_scout_r, 2),
        # FW
        "fw_survived":       fw_survived,
        "fw_surv_pct":       round(fw_survived / max(n_fw, 1) * 100, 1),
        "fw_death_boundary": fw_death_boundary,
        "fw_death_ground":   fw_death_ground,
        "fw_refills":        total_refills,
        "fw_water_dropped_L": round(total_water, 1),
        "fw_valve_pct":      round(valve_pct, 1),
        "fw_avg_reward":     round(avg_fw_r, 2),
        "fw_mode_nn_pct":    round(mode_nn_pct, 1),
        "fw_mode_scripted_pct": round(mode_sc_pct, 1),
        "fw_mode_emergency_pct": round(mode_em_pct, 1),
        # Water drops
        "wd_total":          wd_total,
        "wd_hit":            wd_hit,
        "wd_accuracy_pct":   round(accuracy, 1),
        "wd_avg_alt":        round(avg_wd_alt, 1),
        "wd_avg_dist":       round(avg_wd_dist, 1),
        # Fire
        "fire_peak_cells":   peak_burning,
        "fire_final_cells":  fs_final["burning"],
        "fire_suppressed":   suppressed,
        "fire_intensity_init":  fs_init["intensity"],
        "fire_intensity_final": fs_final["intensity"],
        "fire_reduction":    round(fire_reduction, 2),
        "burned_frac_pct":   fs_final["burned_frac"],
        "fire_extinguished": int(fire_extinguished),
    }


# ── Model loading ───────────────────────────────────────────────────────────

def load_models(scout_path, cmdr_path, device, n_quads_max=4, n_fw_max=3):
    tmp_env = DroneFireEnv(num_quads=n_quads_max, num_fixed=n_fw_max,
                           grid_size_m=1000.0, max_steps=5)
    scout_self_dim = tmp_env.observation_space("quad_0")["self_state"].shape[0]
    fixed_self_dim = tmp_env.observation_space("fixed_0")["self_state"].shape[0]

    scout_actor = ScoutActor(self_state_dim=scout_self_dim,
                             msg_dim=SCOUT_MSG_DIM,
                             hidden_dim=HIDDEN_DIM_SCOUT).to(device)
    cmdr_actor  = CommanderActor(self_state_dim=fixed_self_dim,
                                msg_input_dim=SCOUT_MSG_DIM,
                                hidden_dim=HIDDEN_DIM_CMDR).to(device)

    if os.path.exists(scout_path):
        scout_actor.load_state_dict(torch.load(scout_path, map_location=device,
                                               weights_only=True))
        print(f"[OK] Scout: {scout_path}")
    else:
        print(f"[WARN] Scout not found: {scout_path}")

    if os.path.exists(cmdr_path):
        cmdr_actor.load_state_dict(torch.load(cmdr_path, map_location=device,
                                              weights_only=True))
        print(f"[OK] Cmdr:  {cmdr_path}")
    else:
        print(f"[WARN] Cmdr not found: {cmdr_path}")

    scout_actor.eval()
    cmdr_actor.eval()
    return scout_actor, cmdr_actor


# ── Aggregation ─────────────────────────────────────────────────────────────

def aggregate_rows(rows, group_keys):
    """Group rows by group_keys and compute mean±std for numeric cols."""
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups[key].append(r)

    agg = []
    numeric_keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))
                    and k not in group_keys and k != "seed"]

    for key_vals, group in sorted(groups.items()):
        row = {k: v for k, v in zip(group_keys, key_vals)}
        row["n_episodes"] = len(group)
        for nk in numeric_keys:
            vals = [r[nk] for r in group if r.get(nk) is not None]
            if vals:
                row[f"{nk}_mean"] = round(np.mean(vals), 2)
                row[f"{nk}_std"]  = round(np.std(vals), 2)
                row[f"{nk}_min"]  = round(np.min(vals), 2)
                row[f"{nk}_max"]  = round(np.max(vals), 2)
            else:
                row[f"{nk}_mean"] = 0
                row[f"{nk}_std"]  = 0
        agg.append(row)
    return agg


# ── PDF Plots ───────────────────────────────────────────────────────────────

def save_plots(all_rows, out_dir):
    """Generate publication-quality PDF plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(out_dir, exist_ok=True)

    # Config labels
    def cfg_label(r):
        return f"{r['n_scouts']}S/{r['n_fw']}F/{r['n_fires']}f\n{r['map_size_m']}m"

    # ── Figure 1: Fire suppression overview (6 panels) ──────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Multi-Agent System — Fire Suppression Performance", fontsize=14, y=1.02)

    # Group by (n_scouts, n_fw)
    team_configs = sorted(set((r["n_scouts"], r["n_fw"]) for r in all_rows))

    # 1a: Fire final cells by team config
    ax = axes[0, 0]
    data, labels = [], []
    for ns, nf in team_configs:
        vals = [r["fire_final_cells"] for r in all_rows
                if r["n_scouts"] == ns and r["n_fw"] == nf]
        data.append(vals)
        labels.append(f"{ns}S/{nf}F")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Fire Final [cells]")
    ax.set_title("Final Burning Cells by Team")
    ax.grid(True, alpha=0.3, axis='y')

    # 1b: Fire extinguished rate by team
    ax = axes[0, 1]
    ext_rates = []
    for ns, nf in team_configs:
        sub = [r for r in all_rows if r["n_scouts"] == ns and r["n_fw"] == nf]
        ext_rates.append(sum(r["fire_extinguished"] for r in sub) / len(sub) * 100)
    colors = plt.cm.Set2(np.linspace(0, 1, len(team_configs)))
    ax.bar(range(len(team_configs)), ext_rates, color=colors, alpha=0.8)
    ax.set_xticks(range(len(team_configs)))
    ax.set_xticklabels([f"{ns}S/{nf}F" for ns, nf in team_configs])
    ax.set_ylabel("Extinguished [%]")
    ax.set_title("Fire Extinguishment Rate")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # 1c: Suppressed cells by n_fires
    ax = axes[0, 2]
    fire_counts = sorted(set(r["n_fires"] for r in all_rows))
    data, labels = [], []
    for nf in fire_counts:
        vals = [r["fire_suppressed"] for r in all_rows if r["n_fires"] == nf]
        data.append(vals)
        labels.append(f"{nf} fire{'s' if nf > 1 else ''}")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], ['#2ecc71', '#e67e22', '#e74c3c'][:len(data)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Suppressed [cells]")
    ax.set_title("Fire Suppression by Fire Count")
    ax.grid(True, alpha=0.3, axis='y')

    # 1d: Water accuracy by team
    ax = axes[1, 0]
    data, labels = [], []
    for ns, nf in team_configs:
        vals = [r["wd_accuracy_pct"] for r in all_rows
                if r["n_scouts"] == ns and r["n_fw"] == nf and r["wd_total"] > 0]
        data.append(vals if vals else [0])
        labels.append(f"{ns}S/{nf}F")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("Water Drop Accuracy")
    ax.grid(True, alpha=0.3, axis='y')

    # 1e: FW altitude distribution
    ax = axes[1, 1]
    data, labels = [], []
    for ns, nf in team_configs:
        vals = [r["wd_avg_alt"] for r in all_rows
                if r["n_scouts"] == ns and r["n_fw"] == nf and r["wd_avg_alt"] > 0]
        data.append(vals if vals else [0])
        labels.append(f"{ns}S/{nf}F")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Water Drop Altitude")
    ax.grid(True, alpha=0.3, axis='y')

    # 1f: Burned fraction by map size
    ax = axes[1, 2]
    map_sizes = sorted(set(r["map_size_m"] for r in all_rows))
    data, labels = [], []
    for ms in map_sizes:
        vals = [r["burned_frac_pct"] for r in all_rows if r["map_size_m"] == ms]
        data.append(vals)
        labels.append(f"{ms}m")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], ['#3498db', '#9b59b6', '#1abc9c'][:len(data)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Burned [%]")
    ax.set_title("Burned Area by Map Size")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, "fire_suppression.pdf")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  PDF → {path}")

    # ── Figure 2: Survival & coordination (6 panels) ────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Agent Survival & Coordination", fontsize=14, y=1.02)

    # 2a: Scout survival by config
    ax = axes[0, 0]
    data, labels = [], []
    for ns, nf in team_configs:
        vals = [r["scouts_surv_pct"] for r in all_rows
                if r["n_scouts"] == ns and r["n_fw"] == nf]
        data.append(vals)
        labels.append(f"{ns}S/{nf}F")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Survival [%]")
    ax.set_title("Scout Survival")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # 2b: FW survival by config
    ax = axes[0, 1]
    data, labels = [], []
    for ns, nf in team_configs:
        vals = [r["fw_surv_pct"] for r in all_rows
                if r["n_scouts"] == ns and r["n_fw"] == nf]
        data.append(vals)
        labels.append(f"{ns}S/{nf}F")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Survival [%]")
    ax.set_title("FW Survival")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # 2c: Scout dwell % by map size
    ax = axes[0, 2]
    data, labels = [], []
    for ms in map_sizes:
        vals = [r["scout_dwell_pct"] for r in all_rows if r["map_size_m"] == ms]
        data.append(vals)
        labels.append(f"{ms}m")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], ['#3498db', '#9b59b6', '#1abc9c'][:len(data)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Dwell [%]")
    ax.set_title("Scout Fire Visibility by Map")
    ax.grid(True, alpha=0.3, axis='y')

    # 2d: Discovery time by (n_scouts, map_size)
    ax = axes[1, 0]
    scout_counts = sorted(set(r["n_scouts"] for r in all_rows))
    for ns in scout_counts:
        ms_vals, means, stds = [], [], []
        for ms in map_sizes:
            sub = [r["t_discovery"] for r in all_rows
                   if r["n_scouts"] == ns and r["map_size_m"] == ms and r["t_discovery"] >= 0]
            if sub:
                ms_vals.append(ms)
                means.append(np.mean(sub))
                stds.append(np.std(sub))
        if ms_vals:
            ax.errorbar(ms_vals, means, yerr=stds, marker='o', capsize=4,
                       linewidth=2, label=f"{ns} scouts")
    ax.set_xlabel("Map Size [m]")
    ax.set_ylabel("Discovery Time [steps]")
    ax.set_title("Fire Discovery Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2e: FW mode breakdown (stacked bar)
    ax = axes[1, 1]
    x = np.arange(len(team_configs))
    nn_vals, sc_vals, em_vals = [], [], []
    for ns, nf in team_configs:
        sub = [r for r in all_rows if r["n_scouts"] == ns and r["n_fw"] == nf]
        nn_vals.append(np.mean([r["fw_mode_nn_pct"] for r in sub]))
        sc_vals.append(np.mean([r["fw_mode_scripted_pct"] for r in sub]))
        em_vals.append(np.mean([r["fw_mode_emergency_pct"] for r in sub]))
    ax.bar(x, nn_vals, label="NN", color='#2ecc71', alpha=0.8)
    ax.bar(x, sc_vals, bottom=nn_vals, label="Scripted", color='#3498db', alpha=0.8)
    bot = [n + s for n, s in zip(nn_vals, sc_vals)]
    ax.bar(x, em_vals, bottom=bot, label="Emergency", color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ns}S/{nf}F" for ns, nf in team_configs])
    ax.set_ylabel("Mode [%]")
    ax.set_title("FW Decision Mode Breakdown")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    # 2f: Refills per episode by n_fw
    ax = axes[1, 2]
    fw_counts = sorted(set(r["n_fw"] for r in all_rows))
    data, labels = [], []
    for nf in fw_counts:
        vals = [r["fw_refills"] for r in all_rows if r["n_fw"] == nf]
        data.append(vals)
        labels.append(f"{nf} FW")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], ['#e67e22', '#e74c3c', '#9b59b6'][:len(data)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Refills / episode")
    ax.set_title("FW Refill Count")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, "survival_coordination.pdf")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  PDF → {path}")

    # ── Figure 3: Reward distributions ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Reward Distributions", fontsize=14, y=1.02)

    # 3a: Scout reward by config
    ax = axes[0]
    data, labels = [], []
    for ns, nf in team_configs:
        vals = [r["scout_avg_reward"] for r in all_rows
                if r["n_scouts"] == ns and r["n_fw"] == nf]
        data.append(vals)
        labels.append(f"{ns}S/{nf}F")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Reward")
    ax.set_title("Scout Avg Reward")
    ax.grid(True, alpha=0.3, axis='y')

    # 3b: FW reward by config
    ax = axes[1]
    data, labels = [], []
    for ns, nf in team_configs:
        vals = [r["fw_avg_reward"] for r in all_rows
                if r["n_scouts"] == ns and r["n_fw"] == nf]
        data.append(vals)
        labels.append(f"{ns}S/{nf}F")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Reward")
    ax.set_title("FW Avg Reward")
    ax.grid(True, alpha=0.3, axis='y')

    # 3c: Total reward (scout + fw) by n_fires
    ax = axes[2]
    data, labels = [], []
    for nf in fire_counts:
        vals = [r["scout_avg_reward"] + r["fw_avg_reward"]
                for r in all_rows if r["n_fires"] == nf]
        data.append(vals)
        labels.append(f"{nf} fire{'s' if nf > 1 else ''}")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], ['#2ecc71', '#e67e22', '#e74c3c'][:len(data)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Total Reward")
    ax.set_title("Combined Reward by Fire Count")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, "rewards.pdf")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  PDF → {path}")

    # ── Figure 4: Scalability — n_fw vs fire outcome ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Scalability Analysis — FW Count Impact", fontsize=14, y=1.02)

    for ax_i, (metric, ylabel, title) in enumerate([
        ("fire_final_cells", "Final Burning [cells]", "Fire Outcome"),
        ("wd_accuracy_pct",  "Accuracy [%]",          "Water Drop Accuracy"),
        ("fw_water_dropped_L", "Water Dropped [L]",   "Total Water Used"),
    ]):
        ax = axes[ax_i]
        for nfires in fire_counts:
            means, stds, xs = [], [], []
            for nfw in fw_counts:
                sub = [r[metric] for r in all_rows
                       if r["n_fw"] == nfw and r["n_fires"] == nfires]
                if sub:
                    xs.append(nfw)
                    means.append(np.mean(sub))
                    stds.append(np.std(sub))
            if xs:
                ax.errorbar(xs, means, yerr=stds, marker='s', capsize=4,
                           linewidth=2, label=f"{nfires} fire{'s' if nfires > 1 else ''}")
        ax.set_xlabel("Number of FW")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "scalability.pdf")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  PDF → {path}")


# ── Summary table ───────────────────────────────────────────────────────────

def save_summary(agg_rows, all_rows, out_dir):
    """Save summary table as .txt and LaTeX."""
    lines = []
    lines.append("=" * 150)
    lines.append(f"{'Config':<18} {'N':>4}  {'ScSurv%':>10}  {'FWSurv%':>10}  "
                 f"{'Dwell%':>10}  {'tDisc':>10}  "
                 f"{'Accuracy%':>10}  {'Alt[m]':>10}  "
                 f"{'FinalBurn':>10}  {'Suppressed':>10}  "
                 f"{'Exting%':>8}  {'Burned%':>10}  "
                 f"{'Refills':>8}  {'ScR':>10}  {'FWR':>10}")
    lines.append("=" * 150)

    for a in agg_rows:
        tag = f"{a['n_scouts']}S/{a['n_fw']}F/{a['n_fires']}f {a['map_size_m']}m"
        n = a['n_episodes']

        # Extinguishment rate (raw count)
        sub = [r for r in all_rows
               if r["n_scouts"] == a["n_scouts"] and r["n_fw"] == a["n_fw"]
               and r["n_fires"] == a["n_fires"] and r["map_size_m"] == a["map_size_m"]]
        ext_pct = sum(r["fire_extinguished"] for r in sub) / max(len(sub), 1) * 100

        lines.append(
            f"{tag:<18} {n:>4}  "
            f"{a.get('scouts_surv_pct_mean', 0):5.1f}±{a.get('scouts_surv_pct_std', 0):4.1f}  "
            f"{a.get('fw_surv_pct_mean', 0):5.1f}±{a.get('fw_surv_pct_std', 0):4.1f}  "
            f"{a.get('scout_dwell_pct_mean', 0):5.1f}±{a.get('scout_dwell_pct_std', 0):4.1f}  "
            f"{a.get('t_discovery_mean', 0):5.0f}±{a.get('t_discovery_std', 0):4.0f}  "
            f"{a.get('wd_accuracy_pct_mean', 0):5.1f}±{a.get('wd_accuracy_pct_std', 0):4.1f}  "
            f"{a.get('wd_avg_alt_mean', 0):5.1f}±{a.get('wd_avg_alt_std', 0):4.1f}  "
            f"{a.get('fire_final_cells_mean', 0):5.1f}±{a.get('fire_final_cells_std', 0):4.1f}  "
            f"{a.get('fire_suppressed_mean', 0):5.1f}±{a.get('fire_suppressed_std', 0):4.1f}  "
            f"{ext_pct:5.1f}%  "
            f"{a.get('burned_frac_pct_mean', 0):5.2f}±{a.get('burned_frac_pct_std', 0):4.2f}  "
            f"{a.get('fw_refills_mean', 0):5.1f}  "
            f"{a.get('scout_avg_reward_mean', 0):+6.1f}±{a.get('scout_avg_reward_std', 0):4.1f}  "
            f"{a.get('fw_avg_reward_mean', 0):+6.1f}±{a.get('fw_avg_reward_std', 0):4.1f}"
        )
    lines.append("=" * 150)

    # Overall summary
    lines.append("\n── Overall Summary ──")
    n_total = len(all_rows)
    lines.append(f"Total episodes: {n_total}")
    lines.append(f"Scout survival:     {np.mean([r['scouts_surv_pct'] for r in all_rows]):.1f}%")
    lines.append(f"FW survival:        {np.mean([r['fw_surv_pct'] for r in all_rows]):.1f}%")
    lines.append(f"Fire discovered:    {sum(r['fire_discovered'] for r in all_rows) / n_total * 100:.1f}%")
    lines.append(f"Fire extinguished:  {sum(r['fire_extinguished'] for r in all_rows) / n_total * 100:.1f}%")
    lines.append(f"Water accuracy:     {np.mean([r['wd_accuracy_pct'] for r in all_rows if r['wd_total'] > 0]):.1f}%")
    lines.append(f"Avg drop altitude:  {np.mean([r['wd_avg_alt'] for r in all_rows if r['wd_avg_alt'] > 0]):.1f}m")
    lines.append(f"Avg drop distance:  {np.mean([r['wd_avg_dist'] for r in all_rows if r['wd_avg_dist'] > 0]):.1f}m")
    lines.append(f"Avg scout reward:   {np.mean([r['scout_avg_reward'] for r in all_rows]):+.1f}")
    lines.append(f"Avg FW reward:      {np.mean([r['fw_avg_reward'] for r in all_rows]):+.1f}")

    text = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nSummary → {out_path}")
    print(text)

    # LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{tabular}{l r r r r r r r r r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Config & N & Sc.Surv [\%] & FW.Surv [\%] & Dwell [\%] & "
                       r"Acc [\%] & FinalBurn & Exting [\%] & Sc.R & FW.R \\")
    latex_lines.append(r"\midrule")
    for a in agg_rows:
        tag = f"{a['n_scouts']}S/{a['n_fw']}F/{a['n_fires']}f {a['map_size_m']}m"
        sub = [r for r in all_rows
               if r["n_scouts"] == a["n_scouts"] and r["n_fw"] == a["n_fw"]
               and r["n_fires"] == a["n_fires"] and r["map_size_m"] == a["map_size_m"]]
        ext_pct = sum(r["fire_extinguished"] for r in sub) / max(len(sub), 1) * 100
        latex_lines.append(
            f"{tag} & {a['n_episodes']} & "
            f"${a.get('scouts_surv_pct_mean',0):.1f} \\pm {a.get('scouts_surv_pct_std',0):.1f}$ & "
            f"${a.get('fw_surv_pct_mean',0):.1f} \\pm {a.get('fw_surv_pct_std',0):.1f}$ & "
            f"${a.get('scout_dwell_pct_mean',0):.1f} \\pm {a.get('scout_dwell_pct_std',0):.1f}$ & "
            f"${a.get('wd_accuracy_pct_mean',0):.1f} \\pm {a.get('wd_accuracy_pct_std',0):.1f}$ & "
            f"${a.get('fire_final_cells_mean',0):.1f} \\pm {a.get('fire_final_cells_std',0):.1f}$ & "
            f"${ext_pct:.0f}$ & "
            f"${a.get('scout_avg_reward_mean',0):+.1f}$ & "
            f"${a.get('fw_avg_reward_mean',0):+.1f}$ \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    latex_path = os.path.join(out_dir, "summary_latex.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"LaTeX → {latex_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Comprehensive multi-agent evaluation")
    ap.add_argument("--scout-model", default="saved_models/finetune/scout_best.pt")
    ap.add_argument("--cmdr-model",  default="saved_models/finetune/cmdr_best.pt")
    ap.add_argument("--runs-per-config", type=int, default=20,
                    help="Episodes per configuration combination")
    ap.add_argument("--max-steps",   type=int, nargs="+", default=[1000, 1500])
    ap.add_argument("--scouts",      type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--fw",          type=int, nargs="+", default=[2, 3])
    ap.add_argument("--fires",       type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--map-sizes",   type=float, nargs="+",
                    default=[800.0, 1200.0, 2000.0, 3000.0])
    ap.add_argument("--seed-start",  type=int, default=0)
    ap.add_argument("--out-dir",     default=os.path.join(PROJECT, "results",
                                                           "eval_comprehensive"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scout_path = os.path.join(PROJECT, args.scout_model)
    cmdr_path  = os.path.join(PROJECT, args.cmdr_model)
    n_quads_max = max(args.scouts)
    n_fw_max    = max(args.fw)
    scout_actor, cmdr_actor = load_models(scout_path, cmdr_path, device,
                                          n_quads_max, n_fw_max)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "data.csv")

    # Build config grid
    configs = list(product(args.scouts, args.fw, args.fires, args.map_sizes,
                           args.max_steps))
    total_eps = len(configs) * args.runs_per_config
    print(f"\nConfigurations: {len(configs)}")
    print(f"Runs/config:    {args.runs_per_config}")
    print(f"Total episodes: {total_eps}")
    print(f"Scouts:  {args.scouts}")
    print(f"FW:      {args.fw}")
    print(f"Fires:   {args.fires}")
    print(f"Maps:    {args.map_sizes}")
    print(f"Steps:   {args.max_steps}")
    print(f"Output:  {out_dir}\n")

    CSV_FIELDS = None
    all_rows = []
    run_id = 0

    with open(csv_path, "w", newline="") as csvf:
        writer = None

        for cfg_i, (n_sc, n_fw, n_fires, map_m, msteps) in enumerate(configs):
            tag = f"{n_sc}S/{n_fw}F/{n_fires}f/{int(map_m)}m/{msteps}st"
            print(f"\n{'─'*60}")
            print(f"Config {cfg_i+1}/{len(configs)}: {tag}")
            print(f"{'─'*60}")

            for run in range(args.runs_per_config):
                run_id += 1
                seed = args.seed_start + run_id
                t0 = time.time()

                try:
                    row = run_episode(
                        scout_actor, cmdr_actor, device,
                        n_quads=n_sc, n_fw=n_fw, seed=seed,
                        grid_size_m=map_m, max_steps=msteps,
                        n_fires=n_fires,
                    )
                    row["run_id"] = run_id
                    row["config"] = tag
                    elapsed = time.time() - t0

                    print(f"  [{run_id:>4}/{total_eps}] seed={seed:>5}  "
                          f"ScSurv={row['scouts_surv_pct']:>5.1f}%  "
                          f"FWSurv={row['fw_surv_pct']:>5.1f}%  "
                          f"dwell={row['scout_dwell_pct']:>4.1f}%  "
                          f"acc={row['wd_accuracy_pct']:>4.1f}%  "
                          f"final={row['fire_final_cells']:>3}  "
                          f"ext={row['fire_extinguished']}  "
                          f"{elapsed:.1f}s", flush=True)

                except Exception as exc:
                    print(f"  [{run_id:>4}/{total_eps}] seed={seed} ERROR: {exc}")
                    row = {"run_id": run_id, "seed": seed, "config": tag,
                           "n_scouts": n_sc, "n_fw": n_fw, "n_fires": n_fires,
                           "map_size_m": int(map_m), "max_steps": msteps,
                           "error": str(exc)}

                all_rows.append(row)

                # Write CSV
                if writer is None:
                    CSV_FIELDS = [k for k in row.keys() if k != "error"]
                    writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS,
                                           extrasaction='ignore')
                    writer.writeheader()
                writer.writerow(row)
                csvf.flush()

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE — {len(all_rows)} episodes")
    print(f"{'='*60}")

    # Filter out errors
    valid = [r for r in all_rows if "error" not in r]
    print(f"Valid episodes: {len(valid)}")

    if not valid:
        print("No valid episodes — skipping analysis")
        return

    # Save CSV path
    print(f"\nCSV → {csv_path}")

    # Aggregation
    group_keys = ["n_scouts", "n_fw", "n_fires", "map_size_m"]
    agg = aggregate_rows(valid, group_keys)

    # Summary table
    save_summary(agg, valid, out_dir)

    # Plots
    print("\nGenerating plots...")
    save_plots(valid, out_dir)

    # Save config as JSON
    config_info = {
        "scout_model": args.scout_model,
        "cmdr_model":  args.cmdr_model,
        "total_episodes": len(all_rows),
        "valid_episodes": len(valid),
        "runs_per_config": args.runs_per_config,
        "scouts": args.scouts,
        "fw": args.fw,
        "fires": args.fires,
        "map_sizes": args.map_sizes,
        "max_steps": args.max_steps,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
