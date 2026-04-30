"""
demo_eval_2scouts.py
────────────────────
Evaluation harness: 2 Scout (quad) + 1 Commander (fixed-wing), N episodes.

Zodpovídané otázky:
  • Kolik epizod přežily oba scouti / aspoň jeden / nikdo?
  • Kolik epizod přežil Commander (FW)?
  • Kolik dronů doletělo k ohni (fire_seen_steps > 0)?
  • Kolik epizod FW shodil vodu (fire_extinguished > 0)?
  • Průměrná efektivita hašení, reward, lifespan.

Usage (z root projektu):
    python demos/MAPPO_easiestScenario/demo_eval_2scouts.py
    python demos/MAPPO_easiestScenario/demo_eval_2scouts.py \\
        --scout   results/.../scout_b0700.pt \\
        --commander saved_models/multi/cmdr_b0300.pt \\
        --episodes 20 --seed-start 200
"""

import argparse
import os
import sys
import numpy as np
import torch
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT KONFIGURACE  (musí odpovídat demo_both_training.py)
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_SCOUT_PATH = os.path.join(
    project_root,
    "results/TrainingTogether/09_finalTraining/2on1/02_scoutsFineTuned/scout_b0700.pt"
)
DEFAULT_COMMANDER_PATH = os.path.join(
    project_root, "saved_models/multi/cmdr_b0300.pt"
)

N_QUADS    = 2
N_FIXED    = 1
GRID_SIZE  = 1200.0
MAX_STEPS  = 1000
MSG_DIM    = 5
HIDDEN_SCOUT = 128
HIDDEN_CMDR  = 64

# Commander waypoint params — musí odpovídat train_multi.py
WAYPOINT_STEPS  = 30
WAYPOINT_RANGE  = 200.0
WP_REACHED_DIST = 30.0

# "Reached fire" threshold — kolik kroků musí scout vidět oheň v local_map
FIRE_REACHED_STEPS_THRESHOLD = 3

OUTPUT_DIR = os.path.join(project_root, "output")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(scout_path, cmdr_path, device):
    env_tmp = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                           grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    scout_self_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0]
    fixed_self_dim = env_tmp.observation_space("fixed_0")["self_state"].shape[0]

    scout = ScoutActor(self_state_dim=scout_self_dim, msg_dim=MSG_DIM,
                       hidden_dim=HIDDEN_SCOUT).to(device)
    cmdr  = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=MSG_DIM,
                           action_dim=4, hidden_dim=HIDDEN_CMDR).to(device)

    for path, name, model in [(scout_path, "Scout", scout),
                               (cmdr_path,  "Commander", cmdr)]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name}: NENALEZEN ({path}) — náhodné váhy")
    scout.eval(); cmdr.eval()
    return scout, cmdr


# ═══════════════════════════════════════════════════════════════════════════════
# JEDEN RUN EPIZODY
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(env, scout_actor, cmdr_actor, seed, ep_num, device):
    """
    Vrací dict metrik pro jednu epizodu.

    Klíčové metriky:
        scout_N_lifespan        : int   — počet kroků živý
        scout_N_survived        : bool  — dožil max_steps
        scout_N_reached_fire    : bool  — viděl oheň > threshold kroků
        scout_N_fire_seen_steps : int   — celkem kroků s ohněm v local_map
        scout_N_death_cause     : str
        fw_lifespan             : int
        fw_survived             : bool
        fw_water_used_pct       : float — kolik % tanku spotřeboval
        fire_extinguished       : float — kumulativní efektivita hašení
        fw_death_cause          : str
        any_scout_reached_fire  : bool
        both_scouts_reached_fire: bool
        fire_was_extinguished   : bool  — fire_extinguished > 0.01
        total_reward_scouts     : float — součet odměn všech scoutů
        total_reward_fw         : float
    """
    obs, _ = env.reset(seed=seed, epizode_number=ep_num)

    quad_names = env.quad_agents    # ["quad_0", "quad_1"]
    f_agent    = env.fixed_agents[0] if env.fixed_agents else None

    map_half = env.map_bounds
    safe_limit        = max(50.0, map_half * 0.7)
    boundary_emergency = max(50.0, map_half * 0.6)

    # GRU hidden states
    h_scout = {q: torch.zeros(1, 1, HIDDEN_SCOUT).to(device) for q in quad_names}
    h_cmdr  = torch.zeros(1, 1, HIDDEN_CMDR).to(device)

    # Per-scout tracking
    scout_alive        = {q: True       for q in quad_names}
    scout_lifespan     = {q: 0          for q in quad_names}
    scout_death_cause  = {q: "survived" for q in quad_names}
    scout_fire_seen    = {q: 0          for q in quad_names}
    scout_reward       = {q: 0.0        for q in quad_names}
    scout_terminated   = {q: False      for q in quad_names}

    # Latest scout messages for commander
    scout_msgs = {q: {"msg": torch.zeros(1, MSG_DIM).to(device), "valid": False}
                  for q in quad_names}

    # Commander tracking
    fw_alive        = True
    fw_lifespan     = 0
    fw_death_cause  = "survived"
    fw_terminated   = False
    fw_reward       = 0.0
    fire_extinguished_total = 0.0

    # Commander waypoint state
    need_new_waypoint = True
    target_x, target_y = 0.0, 0.0
    target_alt_raw, water_raw = 0.0, -0.5   # sensible defaults until first NN call
    steps_in_segment = 0

    # Fire cell tracking
    _fg0 = env.sim.environment.fire_grid
    fire_cells_start = int(np.sum(_fg0.B)) if _fg0 is not None else 0
    fire_cells_peak  = fire_cells_start

    for step in range(env.max_steps):
        if not env.agents:
            break

        # Aktualizuj peak burning cells
        _fg = env.sim.environment.fire_grid
        if _fg is not None:
            fire_cells_peak = max(fire_cells_peak, int(np.sum(_fg.B)))

        actions = {}

        # ── Scout forward passes ──────────────────────────────────────────
        for q in quad_names:
            if scout_alive[q] and q in env.agents:
                l_map = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0).to(device)
                s_st  = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0).to(device)
                n_st  = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0).to(device)
                n_m   = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    dist, msg, h_scout[q] = scout_actor(l_map, s_st, n_st, n_m, h_scout[q])
                actions[q] = dist.mean.squeeze(0).cpu().numpy()
                scout_msgs[q]["msg"]   = msg          # [1, MSG_DIM]
                scout_msgs[q]["valid"] = True

                # Fire seen: intensity v local_map > malý práh
                fire_intensity = float(obs[q]["local_map"].max())
                if fire_intensity > 0.01:
                    scout_fire_seen[q] += 1

                scout_lifespan[q] = step + 1
            else:
                scout_alive[q] = False

        # ── Commander forward pass (waypoint mode) ────────────────────────
        if f_agent and fw_alive and f_agent in env.agents:
            drone = env.sim.drones.get(f_agent)

            # Boundary emergency → steer ke středu
            in_boundary_emergency = False
            if drone is not None:
                pos = drone.get_position()
                if abs(pos[0]) > boundary_emergency or abs(pos[1]) > boundary_emergency:
                    target_x, target_y = 0.0, 0.0
                    in_boundary_emergency = True

            if not in_boundary_emergency:
                if drone is not None:
                    pos = drone.get_position()
                    dist_to_wp = np.hypot(target_x - pos[0], target_y - pos[1])
                    if dist_to_wp < WP_REACHED_DIST:
                        need_new_waypoint = True
                if steps_in_segment >= WAYPOINT_STEPS:
                    need_new_waypoint = True

                if need_new_waypoint:
                    s_st_f = torch.FloatTensor(obs[f_agent]["self_state"]).unsqueeze(0).to(device)
                    msgs_t = torch.stack(
                        [scout_msgs[q]["msg"] for q in quad_names], dim=1)      # [1, N, MSG_DIM]
                    msgs_m = torch.tensor(
                        [[not scout_msgs[q]["valid"] for q in quad_names]],
                        dtype=torch.bool).to(device)                             # [1, N]
                    with torch.no_grad():
                        dist_c, _, h_cmdr = cmdr_actor(s_st_f, msgs_t, msgs_m, h_cmdr)
                    act_np = dist_c.mean.squeeze(0).cpu().numpy()

                    cur_pos = drone.get_position() if drone else np.zeros(3)
                    target_x = float(np.clip(
                        cur_pos[0] + act_np[0] * WAYPOINT_RANGE, -safe_limit, safe_limit))
                    target_y = float(np.clip(
                        cur_pos[1] + act_np[1] * WAYPOINT_RANGE, -safe_limit, safe_limit))
                    target_alt_raw = float(act_np[2])
                    water_raw      = float(act_np[3])

                    steps_in_segment = 0
                    need_new_waypoint = False

            # Heading controller → 3-element physical action (stejný jako train_multi.py)
            fw_drone = env.sim.drones.get(f_agent)
            if fw_drone is not None:
                pos = fw_drone.get_position()
                dx = target_x - pos[0]
                dy = target_y - pos[1]
                dist_to = np.sqrt(dx**2 + dy**2)
                if dist_to > 1.0:
                    desired_heading = np.arctan2(dy, dx)
                    cur_yaw = fw_drone.get_orientation_rpy()[2]
                    heading_error = _wrap_angle(desired_heading - cur_yaw)
                    heading_cmd = float(np.clip(heading_error / np.pi, -1.0, 1.0))
                else:
                    heading_cmd = 0.0
                actions[f_agent] = np.array(
                    [heading_cmd, target_alt_raw, water_raw], dtype=np.float32)

            steps_in_segment += 1
            fw_lifespan = step + 1
        else:
            fw_alive = False

        # ── Krok prostředí ────────────────────────────────────────────────
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Odměny
        for q in quad_names:
            scout_reward[q] += rewards.get(q, 0.0)
        if f_agent:
            fw_reward += rewards.get(f_agent, 0.0)

        # Hašení
        if f_agent:
            fire_extinguished_total += env.sim.drone_extinguish_stats.get(f_agent, 0.0)

        # Smrt / přežití scoutů
        for q in quad_names:
            if not scout_alive[q]:
                continue
            if terminations.get(q, False):
                scout_alive[q]      = False
                scout_terminated[q] = True
                scout_lifespan[q]   = step + 1
                scout_death_cause[q] = infos.get(q, {}).get("death_cause", "physics")
            elif truncations.get(q, False):
                scout_alive[q]     = False
                scout_lifespan[q]  = step + 1
                scout_death_cause[q] = "survived"  # dožil max_steps
            else:
                scout_lifespan[q] = step + 1       # stále naživu

        # Smrt / přežití commandera
        if f_agent and fw_alive:
            if terminations.get(f_agent, False):
                fw_alive      = False
                fw_terminated = True
                fw_lifespan   = step + 1
                fw_death_cause = infos.get(f_agent, {}).get("death_cause", "physics")
            elif truncations.get(f_agent, False):
                fw_alive      = False
                fw_lifespan   = step + 1
                fw_death_cause = "survived"
            else:
                fw_lifespan = step + 1

    # Water usage
    fw_water_used_pct = 0.0
    if f_agent:
        final_obs_fw = obs.get(f_agent, {}).get("self_state", None)
        if final_obs_fw is not None and len(final_obs_fw) > 10:
            fw_water_used_pct = max(0.0, 1.0 - float(final_obs_fw[10]))

    # Fire cell stats
    _fgend = env.sim.environment.fire_grid
    fire_cells_end = int(np.sum(_fgend.B)) if _fgend is not None else 0
    # % potlačení ohně: jak moc se FW podařilo snížit peak
    # Pokud oheň nevzrostl (fire_cells_peak == 0), není co hasit
    fire_suppression_pct = max(0.0,
        (fire_cells_peak - fire_cells_end) / max(1, fire_cells_peak) * 100.0)

    # Sestavení výsledků
    m = {}
    for qi, q in enumerate(quad_names):
        idx = qi
        m[f"scout_{idx}_lifespan"]      = scout_lifespan[q]
        m[f"scout_{idx}_survived"]      = not scout_terminated[q]
        m[f"scout_{idx}_reached_fire"]  = (scout_fire_seen[q] >= FIRE_REACHED_STEPS_THRESHOLD)
        m[f"scout_{idx}_fire_seen_steps"] = scout_fire_seen[q]
        m[f"scout_{idx}_death_cause"]   = scout_death_cause[q]
        m[f"scout_{idx}_reward"]        = scout_reward[q]

    m["fw_lifespan"]      = fw_lifespan
    m["fw_survived"]      = not fw_terminated
    m["fw_water_used_pct"] = fw_water_used_pct
    m["fire_extinguished"] = fire_extinguished_total
    m["fire_cells_peak"]   = fire_cells_peak
    m["fire_cells_end"]    = fire_cells_end
    m["fire_suppression_pct"] = fire_suppression_pct
    m["fw_death_cause"]    = fw_death_cause
    m["fw_reward"]         = fw_reward

    m["any_scout_reached_fire"]   = any(m[f"scout_{i}_reached_fire"]  for i in range(N_QUADS))
    m["both_scouts_reached_fire"] = all(m[f"scout_{i}_reached_fire"]  for i in range(N_QUADS))
    m["both_scouts_survived"]     = all(m[f"scout_{i}_survived"]      for i in range(N_QUADS))
    m["any_scout_survived"]       = any(m[f"scout_{i}_survived"]      for i in range(N_QUADS))
    m["fire_was_extinguished"]    = fire_extinguished_total > 0.01
    m["total_reward_scouts"]      = sum(scout_reward[q] for q in quad_names)
    m["fw_terminated"]            = fw_terminated

    return m


# ═══════════════════════════════════════════════════════════════════════════════
# VÝPIS TABULKY
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(all_m, seeds):
    sep = "─" * 120
    print("\n" + "═" * 120)
    print("  VÝSLEDKY EVALUACE  —  2 Scout + 1 Commander")
    print("═" * 120)
    hdr = (f"{'Seed':>5}  {'S0_life':>7}  {'S0_fire':>7}  {'S0_surv':>7}  "
           f"{'S1_life':>7}  {'S1_fire':>7}  {'S1_surv':>7}  "
           f"{'FW_life':>7}  {'FW_surv':>7}  {'Water%':>6}  "
           f"{'Peak':>5}  {'End':>5}  {'Supp%':>6}  "
           f"{'S0_death':>8}  {'S1_death':>8}  {'FW_death':>8}")
    print(hdr)
    print(sep)
    for m, seed in zip(all_m, seeds):
        print(
            f"{seed:>5}  "
            f"{m['scout_0_lifespan']:>7}  {m['scout_0_fire_seen_steps']:>7}  {str(m['scout_0_survived']):>7}  "
            f"{m['scout_1_lifespan']:>7}  {m['scout_1_fire_seen_steps']:>7}  {str(m['scout_1_survived']):>7}  "
            f"{m['fw_lifespan']:>7}  {str(m['fw_survived']):>7}  "
            f"{m['fw_water_used_pct']*100:>5.1f}%  "
            f"{m['fire_cells_peak']:>5}  {m['fire_cells_end']:>5}  {m['fire_suppression_pct']:>5.1f}%  "
            f"{m['scout_0_death_cause']:>8}  {m['scout_1_death_cause']:>8}  {m['fw_death_cause']:>8}"
        )

    print(sep)
    n = len(all_m)

    def pct(key):
        return 100 * np.mean([m[key] for m in all_m])

    def avg(key):
        return np.mean([m[key] for m in all_m])

    def std(key):
        return np.std([m[key] for m in all_m])

    print(f"\n  PŘEHLED KLÍČOVÝCH METRIK  (n={n} epizod)")
    print(f"  {'Metrika':<40}  {'Hodnota':>10}")
    print(f"  {'─'*52}")
    print(f"  {'Přežil scout_0 (%)':<40}  {pct('scout_0_survived'):>9.1f}%")
    print(f"  {'Přežil scout_1 (%)':<40}  {pct('scout_1_survived'):>9.1f}%")
    print(f"  {'Přežili OBA scouti (%)':<40}  {pct('both_scouts_survived'):>9.1f}%")
    print(f"  {'Přežil Commander/FW (%)':<40}  {pct('fw_survived'):>9.1f}%")
    print(f"  {'Scout_0 dosletěl k ohni (%)':<40}  {pct('scout_0_reached_fire'):>9.1f}%")
    print(f"  {'Scout_1 dosletěl k ohni (%)':<40}  {pct('scout_1_reached_fire'):>9.1f}%")
    print(f"  {'Aspoň 1 scout u ohně (%)':<40}  {pct('any_scout_reached_fire'):>9.1f}%")
    print(f"  {'Oba scouti u ohně (%)':<40}  {pct('both_scouts_reached_fire'):>9.1f}%")
    print(f"  {'FW shodil vodu (extg>0) (%)':<40}  {pct('fire_was_extinguished'):>9.1f}%")
    print(f"  {'─'*52}")
    print(f"  {'Průměrný peak ohně (buňky)':<40}  {avg('fire_cells_peak'):>9.1f} ± {std('fire_cells_peak'):.1f}")
    print(f"  {'Průměrný zbytek ohně na konci (buňky)':<40}  {avg('fire_cells_end'):>9.1f} ± {std('fire_cells_end'):.1f}")
    print(f"  {'Průměrné % potlačení ohně':<40}  {avg('fire_suppression_pct'):>9.1f}% ± {std('fire_suppression_pct'):.1f}")
    print(f"  {'Průměrné % vody spotřebováno':<40}  {avg('fw_water_used_pct')*100:>9.1f}%")
    print(f"  {'─'*52}")
    print(f"  {'Průměrný lifespan Scout_0':<40}  {avg('scout_0_lifespan'):>9.1f} ± {std('scout_0_lifespan'):.1f}")
    print(f"  {'Průměrný lifespan Scout_1':<40}  {avg('scout_1_lifespan'):>9.1f} ± {std('scout_1_lifespan'):.1f}")
    print(f"  {'Průměrný lifespan Commander':<40}  {avg('fw_lifespan'):>9.1f} ± {std('fw_lifespan'):.1f}")
    print(f"  {'Průměrná odměna Scout_0':<40}  {avg('scout_0_reward'):>9.1f} ± {std('scout_0_reward'):.1f}")
    print(f"  {'Průměrná odměna Scout_1':<40}  {avg('scout_1_reward'):>9.1f} ± {std('scout_1_reward'):.1f}")
    print(f"  {'Průměrná odměna Commander':<40}  {avg('fw_reward'):>9.1f} ± {std('fw_reward'):.1f}")

    print(f"\n  PŘÍČINY SMRTI")
    for agent_key, label in [("scout_0_death_cause", "Scout_0"),
                               ("scout_1_death_cause", "Scout_1"),
                               ("fw_death_cause",      "Commander")]:
        cnt = Counter(m[agent_key] for m in all_m)
        print(f"  {label:<12}: {dict(cnt)}")

    print("═" * 120 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFY
# ═══════════════════════════════════════════════════════════════════════════════

def save_plots(all_m, output_path, n_episodes):
    n = len(all_m)
    xs = list(range(n))

    def vals(key):
        return [m[key] for m in all_m]

    # ── Global font sizes for thesis readability ──────────────────────
    TITLE_FS  = 11
    LABEL_FS  = 10
    TICK_FS   = 9
    LEGEND_FS = 9
    ANNOT_FS  = 8

    plt.rcParams.update({
        "font.size":        LABEL_FS,
        "axes.titlesize":   TITLE_FS,
        "axes.labelsize":   LABEL_FS,
        "xtick.labelsize":  TICK_FS,
        "ytick.labelsize":  TICK_FS,
        "legend.fontsize":  LEGEND_FS,
    })

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    # No suptitle — described by \caption / \label in the thesis
    kw = dict(linewidth=1.4, alpha=0.85)
    ks = dict(s=50, zorder=3)

    # ── Row 0: Survival & fire contact ───────────────────────────────────
    # Survival: grouped bars per episode so each agent is clearly visible
    ax = axes[0, 0]
    width = 0.28
    s0_surv = [int(m["scout_0_survived"]) for m in all_m]
    s1_surv = [int(m["scout_1_survived"]) for m in all_m]
    fw_surv = [int(m["fw_survived"])      for m in all_m]
    ax.bar([x - width for x in xs], s0_surv, width=width,
           label="Scout 0",    color="deepskyblue", alpha=0.85)
    ax.bar(xs,                   s1_surv, width=width,
           label="Scout 1",    color="royalblue",   alpha=0.85)
    ax.bar([x + width for x in xs], fw_surv, width=width,
           label="Commander", color="tomato",       alpha=0.85)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["died", "survived"])
    ax.set_title("Survival per episode")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[0, 1]
    ax.bar(xs, [int(m["scout_0_reached_fire"]) for m in all_m],
           color="deepskyblue", alpha=0.8, label="Scout 0")
    ax.bar(xs, [int(m["scout_1_reached_fire"]) for m in all_m],
           color="royalblue",   alpha=0.8, label="Scout 1",
           bottom=[int(m["scout_0_reached_fire"]) for m in all_m])
    ax.set_ylim(0, 2.5)
    ax.set_title("Scouts that reached fire (stacked)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[0, 2]
    ax.scatter(xs, vals("scout_0_fire_seen_steps"), color="deepskyblue",
               **ks, label="Scout 0")
    ax.scatter(xs, vals("scout_1_fire_seen_steps"), color="royalblue",
               **ks, label="Scout 1", marker="s")
    ax.axhline(FIRE_REACHED_STEPS_THRESHOLD, color="orange", linewidth=1.2,
               linestyle="--", label=f"threshold={FIRE_REACHED_STEPS_THRESHOLD}")
    ax.set_title("Steps with fire in local map")
    ax.set_ylabel("Steps")
    ax.set_xlabel("Episode")
    ax.legend()

    # ── Row 1: Fire suppression ───────────────────────────────────────────
    ax = axes[1, 0]
    ends      = vals("fire_cells_end")
    peaks     = vals("fire_cells_peak")
    suppressed = [max(0, p - e) for p, e in zip(peaks, ends)]
    ax.bar(xs, suppressed, label="suppressed", color="steelblue",  alpha=0.85)
    ax.bar(xs, ends,       label="remaining",  color="tomato",     alpha=0.85,
           bottom=suppressed)
    ax.set_title("Fire cells: suppressed vs. remaining")
    ax.set_ylabel("Grid cells")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[1, 1]
    supp_pcts  = vals("fire_suppression_pct")
    bar_colors = ["steelblue" if v >= 50 else ("orange" if v > 0 else "tomato")
                  for v in supp_pcts]
    ax.bar(xs, supp_pcts, color=bar_colors, alpha=0.85)
    ax.axhline(50, color="grey", linewidth=0.9, linestyle="--", label="50 %")
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Fire suppression: (peak − end) / peak")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(xs, vals("scout_0_reward"), color="deepskyblue", **kw, label="Scout 0")
    ax.plot(xs, vals("scout_1_reward"), color="royalblue",   **kw, label="Scout 1")
    ax.plot(xs, vals("fw_reward"),      color="tomato",      **kw, label="Commander")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_title("Cumulative reward per episode")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Episode")
    ax.legend()

    # ── Row 2: Lifespans & summary ────────────────────────────────────────
    ax = axes[2, 0]
    ax.scatter(xs, vals("scout_0_lifespan"), color="deepskyblue",
               **ks, label="Scout 0")
    ax.scatter(xs, vals("scout_1_lifespan"), color="royalblue",
               **ks, label="Scout 1", marker="s")
    ax.axhline(MAX_STEPS, color="grey", linewidth=0.9, linestyle="--",
               label=f"max={MAX_STEPS}")
    ax.set_title("Scout lifespan (steps)")
    ax.set_ylabel("Steps")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[2, 1]
    ax.scatter(xs, vals("fw_lifespan"), color="tomato", **ks, label="Commander")
    ax.axhline(MAX_STEPS, color="grey", linewidth=0.9, linestyle="--",
               label=f"max={MAX_STEPS}")
    ax.set_title("Commander lifespan (steps)")
    ax.set_ylabel("Steps")
    ax.set_xlabel("Episode")
    ax.legend()

    # Summary bar chart
    ax = axes[2, 2]
    summary = {
        "S0\nsurvived":   np.mean(vals("scout_0_survived"))   * 100,
        "S1\nsurvived":   np.mean(vals("scout_1_survived"))   * 100,
        "FW\nsurvived":   np.mean(vals("fw_survived"))        * 100,
        "S0\nat fire":    np.mean(vals("scout_0_reached_fire")) * 100,
        "S1\nat fire":    np.mean(vals("scout_1_reached_fire")) * 100,
        "FW\nextinguished": np.mean(vals("fire_was_extinguished")) * 100,
        "Avg\nsupp %":    np.mean(vals("fire_suppression_pct")),
    }
    labels = list(summary.keys())
    values = list(summary.values())
    bar_c  = ["deepskyblue", "royalblue", "tomato",
              "deepskyblue", "royalblue", "red", "steelblue"]
    bars = ax.bar(labels, values, color=bar_c, alpha=0.85)
    ax.set_ylim(0, 118)
    ax.set_ylabel("%")
    ax.set_title(f"Summary (n={n} episodes)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=ANNOT_FS)

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as PDF (vector) for thesis — fall back to PNG if path ends with .png
    if output_path.endswith(".png"):
        pdf_path = output_path.replace(".png", ".pdf")
    else:
        pdf_path = output_path if output_path.endswith(".pdf") else output_path + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"\u2705 Plot saved \u2192 {pdf_path}")

    # Reset rcParams to avoid side effects
    plt.rcParams.update(plt.rcParamsDefault)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluace: 2 Scout + 1 Commander")
    p.add_argument("--scout",      default=DEFAULT_SCOUT_PATH)
    p.add_argument("--commander",  default=DEFAULT_COMMANDER_PATH)
    p.add_argument("--episodes",   type=int, default=20)
    p.add_argument("--seed-start", type=int, default=200,
                   help="První seed pro reset() — určuje náhodnost mapy")
    p.add_argument("--ep-num",     type=int, default=30000,
                   help="Číslo epizody předané curriculum plánovači (default 30000 = plná obtížnost: "
                        "spawn 50–960 m od ohně). Warmup < 3000, ramp 3000–25000, plná ≥ 25000.")
    p.add_argument("--max-steps",  type=int, default=MAX_STEPS)
    p.add_argument("--output",     default=os.path.join(OUTPUT_DIR, "eval_2scouts.pdf"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vysvětlit curriculum fázi
    ep = args.ep_num
    if ep < 3000:
        diff_label = f"WARMUP (spawn 30–80 m od ohně, oheň v {GRID_SIZE/2*0.1:.0f} m od středu)"
    elif ep < 25000:
        t = (ep - 3000) / (25000 - 3000)
        sr_min = 20 + 80 * t
        sr_max = 100 + (GRID_SIZE / 2 * 0.8 - 100) * t
        diff_label = f"RAMP t={t:.2f} (spawn {sr_min:.0f}–{sr_max:.0f} m od ohně)"
    else:
        diff_label = f"PLNÁ OBTÍŽNOST (spawn 50–{GRID_SIZE/2*0.8:.0f} m od ohně)"

    print(f"\n🔍 Konfigurace evaluace:")
    print(f"   N_QUADS={N_QUADS}  N_FIXED={N_FIXED}  "
          f"GRID_SIZE={GRID_SIZE}m  MAX_STEPS={args.max_steps}")
    print(f"   Epizody:  {args.episodes}")
    print(f"   Seeds:    {args.seed_start} → {args.seed_start + args.episodes - 1}")
    print(f"   ep_num:   {args.ep_num}  →  {diff_label}")
    print(f"   Device:   {device}")
    print()

    scout_actor, cmdr_actor = load_models(args.scout, args.commander, device)

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                       grid_size_m=GRID_SIZE, max_steps=args.max_steps)

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    all_m = []

    print(f"\n▶ Spouštím {args.episodes} epizod...\n")
    for i, seed in enumerate(seeds):
        ep_num = args.ep_num          # curriculum fáze — fixní pro celou evaluaci
        m = run_episode(env, scout_actor, cmdr_actor, seed, ep_num, device)
        m["seed"] = seed
        all_m.append(m)

        fire_icon = "🔥" if m["any_scout_reached_fire"] else "  "
        extg_icon = "💧" if m["fire_was_extinguished"] else "  "
        surv_s = f"S0={'✓' if m['scout_0_survived'] else '✗'} S1={'✓' if m['scout_1_survived'] else '✗'}"
        surv_f = f"FW={'✓' if m['fw_survived'] else '✗'}"
        print(f"  [{i+1:2d}/{args.episodes}] seed={seed:4d} | "
              f"{surv_s} {surv_f} | "
              f"fire_seen=({m['scout_0_fire_seen_steps']:3d},{m['scout_1_fire_seen_steps']:3d}) "
              f"{fire_icon} {extg_icon} | "
              f"peak={m['fire_cells_peak']:4d}cells  supp={m['fire_suppression_pct']:5.1f}%  water={m['fw_water_used_pct']*100:4.1f}%")

    print_table(all_m, seeds)
    save_plots(all_m, args.output, args.episodes)


if __name__ == "__main__":
    main()
