"""
diagnose_fw_water.py
────────────────────
Proč FW někdy nepotlačí oheň i když hoří 100+ buněk?

Sleduje per-step:
  • vzdálenost FW od ohně
  • vzdálenost FW od refill zóny
  • hladina vody FW
  • kdy FW triggeruje vodu (+ jestli s efektem)
  • vzdálenost scoutů od ohně (jestli vůbec reportují polohu)

Výstup: tabulka na stdout + PNG grafy pro každou epizodu kde supp=0%.

Usage:
    python tools/diagnose_fw_water.py
    python tools/diagnose_fw_water.py \\
        --scout   saved_models/multi/scout_b0030.pt \\
        --commander saved_models/multi/cmdr_b0730.pt \\
        --episodes 30 --seed-start 200
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor

# ── stejné konstanty jako demo_eval_2scouts.py ─────────────────────────────
N_QUADS       = 2
N_FIXED       = 1
GRID_SIZE     = 1200.0
MAX_STEPS     = 1000
MSG_DIM       = 5
HIDDEN_SCOUT  = 128
HIDDEN_CMDR   = 64
WAYPOINT_STEPS  = 30
WAYPOINT_RANGE  = 200.0
WP_REACHED_DIST = 30.0

DEFAULT_SCOUT = os.path.join(project_root, "saved_models/multi/scout_b0030.pt")
DEFAULT_CMDR  = os.path.join(project_root, "saved_models/multi/cmdr_b0730.pt")


def _inject_fire_compass(ss, scout_msg_dict, quad_names, fw_pos, norm_dist=1000.0):
    fire_x_list, fire_y_list, max_intensity = [], [], 0.0
    for q in quad_names:
        msg = scout_msg_dict[q].squeeze(0)
        intensity = float(msg[2])
        if intensity > 0.01:
            fire_x_list.append(float(msg[0]) * norm_dist)
            fire_y_list.append(float(msg[1]) * norm_dist)
            max_intensity = max(max_intensity, intensity)
    if fire_x_list:
        cx, cy = float(np.mean(fire_x_list)), float(np.mean(fire_y_list))
        dx, dy = cx - fw_pos[0], cy - fw_pos[1]
        dist = float(np.hypot(dx, dy))
        dist_norm = min(dist / norm_dist, 2.0)
        compass_x, compass_y = (dx / dist, dy / dist) if dist > 1.0 else (0.0, 0.0)
    else:
        compass_x, compass_y, dist_norm, max_intensity = 0.0, 0.0, 2.0, 0.0
    ss[19] = compass_x; ss[20] = compass_y
    ss[21] = dist_norm; ss[22] = max_intensity


def load_models(scout_path, cmdr_path, device):
    # observation_space() funguje bez reset() — nevyžaduje sim
    env_tmp = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                           grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    scout_self_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0]
    fixed_self_dim = env_tmp.observation_space("fixed_0")["self_state"].shape[0]
    # sim neexistuje před reset(), prostě env zahodíme (GC se postará)

    scout = ScoutActor(self_state_dim=scout_self_dim, msg_dim=MSG_DIM,
                       hidden_dim=HIDDEN_SCOUT).to(device)
    cmdr  = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=MSG_DIM,
                           action_dim=4, hidden_dim=HIDDEN_CMDR).to(device)
    for path, name, model in [(scout_path, "Scout", scout), (cmdr_path, "Cmdr", cmdr)]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name}: nenalezen ({path})")
    scout.eval(); cmdr.eval()
    return scout, cmdr


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_episode(env, scout_actor, cmdr_actor, seed, ep_num, device):
    """Vrací dict metrik + per-step trace."""
    obs, _ = env.reset(seed=seed, epizode_number=ep_num)
    quad_names  = env.quad_agents[:]
    f_agent     = env.fixed_agents[0] if env.fixed_agents else None

    map_half = env.map_bounds
    safe_limit         = max(50.0, map_half * 0.7)
    boundary_emergency = max(50.0, map_half * 0.6)

    # GRU hidden states (stejný formát jako demo_eval_2scouts.py)
    h_scout = {q: torch.zeros(1, 1, HIDDEN_SCOUT).to(device) for q in quad_names}
    h_cmdr  = torch.zeros(1, 1, HIDDEN_CMDR).to(device)
    scout_msgs = {q: {"msg": torch.zeros(1, MSG_DIM).to(device), "valid": False}
                  for q in quad_names}

    scout_alive = {q: True for q in quad_names}
    fw_alive    = True

    # waypoint state
    target_x, target_y   = 0.0, 0.0
    target_alt_raw        = 0.0
    water_raw             = -0.5    # výchozí = netriggeruje
    steps_in_segment      = WAYPOINT_STEPS   # force new WP on step 0
    need_new_waypoint     = True

    # per-step trace lists
    trace = {
        "fw_dist_fire":   [],
        "fw_dist_refill": [],
        "fw_water_level": [],
        "fw_triggered":   [],
        "fw_effective":   [],
        "fire_cells":     [],
        "scout_compass":  [],
        "fw_x": [], "fw_y": [],
    }

    total_triggers  = 0
    total_effective = 0
    refill_events   = 0
    prev_water_abs  = None   # absolute litres před krokem

    fire_x = env.fire_x
    fire_y = env.fire_y
    refill_pos = None
    if env.sim.environment.refill_zone is not None:
        refill_pos = env.sim.environment.refill_zone['position']

    for step in range(MAX_STEPS):
        if not env.agents:
            break

        actions = {}

        # ── Scout forward passes ─────────────────────────────────────────
        for q in quad_names:
            if scout_alive[q] and q in env.agents:
                l_map = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0).to(device)
                s_st  = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0).to(device)
                n_st  = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0).to(device)
                n_m   = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    dist_s, msg, h_scout[q] = scout_actor(l_map, s_st, n_st, n_m, h_scout[q])
                actions[q] = dist_s.mean.squeeze(0).cpu().numpy()
                scout_msgs[q]["msg"]   = msg
                scout_msgs[q]["valid"] = True
            else:
                scout_alive[q] = False

        # ── Commander forward pass ───────────────────────────────────────
        triggered_this_step = False

        if f_agent and fw_alive and f_agent in env.agents:
            drone = env.sim.drones.get(f_agent)

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

                if need_new_waypoint and drone is not None:
                    _inject_fire_compass(obs[f_agent]["self_state"],
                                         {q: scout_msgs[q]["msg"] for q in quad_names},
                                         quad_names, drone.get_position())
                    s_st_f = torch.FloatTensor(obs[f_agent]["self_state"]).unsqueeze(0).to(device)
                    msgs_t = torch.stack([scout_msgs[q]["msg"] for q in quad_names], dim=1)
                    msgs_m = torch.tensor([[not scout_msgs[q]["valid"] for q in quad_names]],
                                          dtype=torch.bool).to(device)
                    with torch.no_grad():
                        dist_c, _, h_cmdr = cmdr_actor(s_st_f, msgs_t, msgs_m, h_cmdr)
                    act_np = dist_c.mean.squeeze(0).cpu().numpy()
                    cur_pos = drone.get_position()
                    target_x       = float(np.clip(cur_pos[0] + act_np[0] * WAYPOINT_RANGE, -safe_limit, safe_limit))
                    target_y       = float(np.clip(cur_pos[1] + act_np[1] * WAYPOINT_RANGE, -safe_limit, safe_limit))
                    target_alt_raw = float(act_np[2])
                    water_raw      = float(act_np[3])
                    steps_in_segment  = 0
                    need_new_waypoint = False

            # Heading controller — identický s demo_eval_2scouts.py
            fw_drone = env.sim.drones.get(f_agent)
            if fw_drone is not None:
                pos = fw_drone.get_position()
                dx, dy = target_x - pos[0], target_y - pos[1]
                if np.hypot(dx, dy) > 1.0:
                    desired_heading = np.arctan2(dy, dx)
                    cur_yaw = fw_drone.get_orientation_rpy()[2]
                    heading_error = _wrap_angle(desired_heading - cur_yaw)
                    heading_cmd = float(np.clip(heading_error / np.pi, -1.0, 1.0))
                else:
                    heading_cmd = 0.0
                actions[f_agent] = np.array([heading_cmd, target_alt_raw, water_raw], dtype=np.float32)

                # Water tracking — sleduj absolutní hladinu před krokem
                cur_water_abs = fw_drone.current_water
                if prev_water_abs is not None and cur_water_abs > prev_water_abs + 1.0:
                    refill_events += 1
                prev_water_abs = cur_water_abs

            triggered_this_step = (water_raw > 0.0)
            if triggered_this_step:
                total_triggers += 1

            steps_in_segment += 1

        # ── Step ────────────────────────────────────────────────────────
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Efektivní drop: hladina vody klesla (oheň spotřeboval vodu)
        effective_this_step = False
        if f_agent and triggered_this_step:
            fw_drone_post = env.sim.drones.get(f_agent)
            if fw_drone_post is not None and prev_water_abs is not None:
                water_after = fw_drone_post.current_water
                if water_after < prev_water_abs - 0.5:   # spotřeboval alespoň 0.5 L
                    effective_this_step = True
                    total_effective += 1

        # Update alive flags
        for q in quad_names:
            if terminations.get(q, False) or truncations.get(q, False):
                scout_alive[q] = False
        if f_agent and (terminations.get(f_agent, False) or truncations.get(f_agent, False)):
            fw_alive = False

        # Record trace
        fw_drone = env.sim.drones.get(f_agent) if f_agent else None
        fw_pos   = fw_drone.get_position() if fw_drone else np.zeros(3)
        dist_fire   = float(np.hypot(fw_pos[0] - fire_x, fw_pos[1] - fire_y))
        dist_refill = (float(np.hypot(fw_pos[0] - refill_pos[0], fw_pos[1] - refill_pos[1]))
                       if refill_pos is not None else float('nan'))
        water_lvl = (fw_drone.current_water / fw_drone.water_capacity
                     if (fw_drone and fw_drone.water_capacity > 0) else 0.0)
        fire_cells = (int(np.sum(env.sim.environment.fire_grid.B))
                      if env.sim.environment.fire_grid is not None else 0)
        any_scout_reports = any(
            float(scout_msgs[q]["msg"].squeeze(0)[2]) > 0.01 for q in quad_names)

        trace["fw_dist_fire"].append(dist_fire)
        trace["fw_dist_refill"].append(dist_refill)
        trace["fw_water_level"].append(water_lvl)
        trace["fw_triggered"].append(triggered_this_step)
        trace["fw_effective"].append(effective_this_step)
        trace["fire_cells"].append(fire_cells)
        trace["scout_compass"].append(any_scout_reports)
        trace["fw_x"].append(fw_pos[0])
        trace["fw_y"].append(fw_pos[1])

    peak_cells  = max(trace["fire_cells"]) if trace["fire_cells"] else 0
    end_cells   = trace["fire_cells"][-1]  if trace["fire_cells"] else 0
    supp_pct    = (1.0 - end_cells / peak_cells) * 100.0 if peak_cells > 0 else 100.0
    water_empty_steps = sum(1 for w in trace["fw_water_level"] if w < 0.05)
    trigger_when_empty = sum(1 for t, w in zip(trace["fw_triggered"], trace["fw_water_level"])
                             if t and w < 0.05)
    scout_silent_steps = sum(1 for s in trace["scout_compass"] if not s)
    min_dist_fire = min(trace["fw_dist_fire"]) if trace["fw_dist_fire"] else float('nan')

    # Diagnose root cause
    if supp_pct == 0.0 and peak_cells >= 30:
        if water_empty_steps > MAX_STEPS * 0.5:
            root_cause = "VODA_VYČERPÁNA_BRZY"
        elif min_dist_fire > 300:
            root_cause = "FW_NEDOLÉTL_K_OHNI"
        elif scout_silent_steps > MAX_STEPS * 0.6:
            root_cause = "SCOUTI_NEREPORTUJÍ"
        elif total_triggers == 0:
            root_cause = "FW_NETRIGGERUJE"
        else:
            root_cause = "FW_MÍJÍ_OHEŇ"
    else:
        root_cause = "OK" if supp_pct > 50 else "ČÁSTEČNÉ"

    return {
        "seed": seed,
        "peak_cells": peak_cells,
        "end_cells": end_cells,
        "supp_pct": supp_pct,
        "total_triggers": total_triggers,
        "total_effective": total_effective,
        "water_empty_steps": water_empty_steps,
        "trigger_when_empty": trigger_when_empty,
        "refill_events": refill_events,
        "scout_silent_steps": scout_silent_steps,
        "min_dist_fire": min_dist_fire,
        "root_cause": root_cause,
        "trace": trace,
        "fire_x": fire_x, "fire_y": fire_y,
        "refill_pos": refill_pos,
    }


def plot_episode(ep, out_dir, seed):
    """Uloží diagnostický PNG pro jednu epizodu."""
    trace = ep["trace"]
    steps = range(len(trace["fw_dist_fire"]))
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Seed {seed} | peak={ep['peak_cells']} cells | supp={ep['supp_pct']:.1f}% | {ep['root_cause']}",
                 fontsize=12, fontweight='bold')

    # 1. Vzdálenost FW od ohně
    ax = axes[0, 0]
    ax.plot(steps, trace["fw_dist_fire"], color='red', lw=1)
    ax.axhline(200, color='orange', ls='--', lw=0.8, label='200 m fire zone')
    ax.axhline(50,  color='red',    ls='--', lw=0.8, label='50 m close')
    ax.set_ylabel("Dist FW→oheň [m]")
    ax.set_xlabel("Krok")
    ax.legend(fontsize=7)
    ax.set_title("Vzdálenost FW od ohně")

    # 2. Hladina vody + refill events
    ax = axes[0, 1]
    ax.plot(steps, trace["fw_water_level"], color='blue', lw=1, label='water level')
    # mark refill events
    for i in range(1, len(trace["fw_water_level"])):
        if trace["fw_water_level"][i] > trace["fw_water_level"][i-1] + 0.05:
            ax.axvline(i, color='cyan', lw=1, alpha=0.7)
    ax.set_ylabel("Hladina vody [0–1]")
    ax.set_xlabel("Krok")
    ax.set_title("Hladina vody FW (cyan = refill)")
    ax.set_ylim(0, 1.1)

    # 3. Vzdálenost FW od refill zóny
    ax = axes[1, 0]
    ax.plot(steps, trace["fw_dist_refill"], color='green', lw=1)
    refill_r = 30.0  # typický radius refill zóny
    ax.axhline(refill_r, color='lime', ls='--', lw=0.8, label=f'{refill_r}m refill radius')
    ax.set_ylabel("Dist FW→refill [m]")
    ax.set_xlabel("Krok")
    ax.legend(fontsize=7)
    ax.set_title("Vzdálenost od refill zóny")

    # 4. Oheň + triggery
    ax = axes[1, 1]
    ax.fill_between(steps, trace["fire_cells"], alpha=0.3, color='red', label='fire cells')
    trigger_steps = [i for i, t in enumerate(trace["fw_triggered"]) if t]
    eff_steps     = [i for i, e in enumerate(trace["fw_effective"]) if e]
    if trigger_steps:
        ax.scatter(trigger_steps,
                   [trace["fire_cells"][i] for i in trigger_steps],
                   marker='|', color='blue', s=20, label='trigger', zorder=5)
    if eff_steps:
        ax.scatter(eff_steps,
                   [trace["fire_cells"][i] for i in eff_steps],
                   marker='*', color='gold', s=60, label='efektivní', zorder=6)
    ax.set_ylabel("Hořící buňky")
    ax.set_xlabel("Krok")
    ax.legend(fontsize=7)
    ax.set_title(f"Oheň + dropy (triggers={len(trigger_steps)}, efekt={len(eff_steps)})")

    # 5. Scout reporting
    ax = axes[2, 0]
    ax.fill_between(steps, [1 if s else 0 for s in trace["scout_compass"]],
                    alpha=0.5, color='purple', label='scout reports fire')
    ax.set_ylim(-0.1, 1.5)
    ax.set_ylabel("Scout vidí oheň")
    ax.set_xlabel("Krok")
    ax.set_title("Scout reportuje polohu ohně")

    # 6. Trajektorie FW (2D)
    ax = axes[2, 1]
    ax.plot(trace["fw_x"], trace["fw_y"], color='navy', lw=0.8, alpha=0.7, label='FW trajektorie')
    ax.scatter([ep["fire_x"]], [ep["fire_y"]], marker='*', color='red', s=200, label='oheň', zorder=5)
    if ep["refill_pos"] is not None:
        ax.scatter([ep["refill_pos"][0]], [ep["refill_pos"][1]],
                   marker='s', color='cyan', s=100, label='refill', zorder=5)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Trajektorie FW")
    ax.legend(fontsize=7)
    ax.set_aspect('equal')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"diag_seed{seed}.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scout",      default=DEFAULT_SCOUT)
    parser.add_argument("--commander",  default=DEFAULT_CMDR)
    parser.add_argument("--episodes",   type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--plot-all",   action="store_true",
                        help="Uložit PNG i pro epizody kde supp>0")
    parser.add_argument("--out-dir",    default=os.path.join(project_root, "output", "diag_water"))
    args = parser.parse_args()

    device = torch.device("cpu")
    scout_actor, cmdr_actor = load_models(args.scout, args.commander, device)

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                       grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)

    print(f"\nDiagnostika {args.episodes} epizod (seeds {args.seed_start}–{args.seed_start+args.episodes-1})\n")

    results = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        ep   = run_episode(env, scout_actor, cmdr_actor, seed,
                           ep_num=seed * 1000, device=device)
        results.append(ep)

        # Print one-liner
        water_pct = ep["total_effective"] / max(ep["total_triggers"], 1) * 100
        print(f"  seed={seed:3d} | peak={ep['peak_cells']:4d} | supp={ep['supp_pct']:5.1f}% "
              f"| trig={ep['total_triggers']:4d} eff={ep['total_effective']:3d} ({water_pct:4.1f}%) "
              f"| min_dist={ep['min_dist_fire']:5.0f}m "
              f"| refill={ep['refill_events']:2d}x water_empty={ep['water_empty_steps']:4d}steps "
              f"| silent_scout={ep['scout_silent_steps']:4d}steps "
              f"| → {ep['root_cause']}")

        # Plot failed or all
        if ep["supp_pct"] == 0.0 and ep["peak_cells"] >= 30 or args.plot_all:
            path = plot_episode(ep, args.out_dir, seed)
            print(f"           PNG → {path}")

    env.sim.stop_simulation()

    # ── Souhrnná tabulka ────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  SHRNUTÍ PŘÍČIN NEÚSPĚCHU")
    print("═" * 80)
    from collections import Counter
    causes = Counter(ep["root_cause"] for ep in results)
    total  = len(results)
    for cause, cnt in causes.most_common():
        print(f"  {cause:<30s}: {cnt:3d}/{total} epizod  ({cnt/total*100:.0f}%)")

    failed = [ep for ep in results if ep["supp_pct"] == 0.0 and ep["peak_cells"] >= 30]
    if failed:
        print(f"\n  Detaily {len(failed)} epizod se supp=0% a peak≥30 buněk:")
        print(f"  {'seed':>5} {'peak':>6} {'triggers':>8} {'eff%':>6} {'min_dist':>9} "
              f"{'refills':>7} {'empty_steps':>11} {'silent_scout':>12} {'příčina'}")
        print("  " + "-" * 80)
        for ep in sorted(failed, key=lambda e: e["peak_cells"], reverse=True):
            water_pct = ep["total_effective"] / max(ep["total_triggers"], 1) * 100
            print(f"  {ep['seed']:5d} {ep['peak_cells']:6d} {ep['total_triggers']:8d} "
                  f"{water_pct:5.1f}% {ep['min_dist_fire']:9.0f}m "
                  f"{ep['refill_events']:7d} {ep['water_empty_steps']:11d} "
                  f"{ep['scout_silent_steps']:12d}  {ep['root_cause']}")

    # ── Agregované statistiky ────────────────────────────────────────────
    print("\n  AGREGÁT (všechny epizody):")
    avg_refill   = np.mean([ep["refill_events"]    for ep in results])
    avg_empty    = np.mean([ep["water_empty_steps"] for ep in results])
    avg_min_dist = np.mean([ep["min_dist_fire"]     for ep in results])
    avg_triggers = np.mean([ep["total_triggers"]    for ep in results])
    avg_eff_pct  = np.mean([ep["total_effective"] / max(ep["total_triggers"], 1) * 100
                            for ep in results])
    print(f"  Průměrné refill events za epizodu : {avg_refill:.2f}")
    print(f"  Průměrné kroky s prázdnou vodou   : {avg_empty:.1f} / {MAX_STEPS}")
    print(f"  Průměrná min. vzdálenost FW→oheň  : {avg_min_dist:.1f} m")
    print(f"  Průměrný počet triggerů            : {avg_triggers:.1f}")
    print(f"  Průměrná přesnost dropů            : {avg_eff_pct:.1f}%")


if __name__ == "__main__":
    main()
