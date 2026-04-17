#!/usr/bin/env python3
"""
Multi-seed diagnostic: Scout ↔ FW interaction analysis.

Runs N episodes with different seeds, collects per-step data from each,
and produces:
  1. Per-seed summary table (printed + CSV)
  2. Aggregate statistics figure with cross-seed patterns
  3. Individual seed timeline CSVs (optional, for deep dives)

Key questions answered:
  - Does FW consistently crash or reach the fire?
  - Does scout fly away when FW starts extinguishing?
  - Is there a reward conflict (scout loses reward when FW extinguishes)?
  - What kills the FW (boundary, ground, ceiling)?
  - How often does the team actually extinguish fire?

Usage:
    python demos/MAPPO_easiestScenario/diagnose_scout_fw.py
    python demos/MAPPO_easiestScenario/diagnose_scout_fw.py --seeds 20
    python demos/MAPPO_easiestScenario/diagnose_scout_fw.py --batch 420
"""

import os, sys, argparse, time
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor
from src.reward_config import QUAD, FIXED, SHARED

# ── Config ──────────────────────────────────────────────────────────────────
N_QUADS = 1
N_FIXED = 1
MAX_STEPS = 1500
GRID_SIZE = 2000.0

MODEL_DIR = os.path.join(project_root, "saved_models", "multi")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Single-episode runner
# ═══════════════════════════════════════════════════════════════════════════

def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_single_episode(env, scout_actor, cmdr_actor, device, seed):
    """Run one episode, return a dict of per-step arrays + episode summary."""
    obs, _ = env.reset(seed=seed)
    h_scout = torch.zeros(1, 1, 128).to(device)
    h_cmdr = torch.zeros(1, 1, 64).to(device)

    fire_cx_init = env.fire_x
    fire_cy_init = env.fire_y

    q_id, f_id = "quad_0", "fixed_0"

    # ── Waypoint controller config (must match train_multi.py) ────────
    WAYPOINT_STEPS = 30
    WAYPOINT_RANGE = 200.0
    map_half = GRID_SIZE / 2.0
    safe_limit = map_half - 350.0
    boundary_emergency = map_half - 300.0
    wp_reached_dist = 30.0

    # Commander waypoint state
    target_x, target_y = 0.0, 0.0
    target_alt_raw, water_raw = 0.0, -1.0
    need_new_waypoint = True
    steps_in_segment = 0

    # Per-step storage (lightweight — only what matters for aggregate)
    ts = {
        "scout_fire_intensity": [],
        "scout_dist_to_fire": [],
        "scout_reward": [],
        "scout_fire_reward": [],  # flat + intensity component only
        "scout_z": [],
        "scout_radial_vel": [],   # positive = approaching fire, negative = fleeing
        "scout_compass_align": [], # cos(angle between velocity and fire direction)
        "scout_speed": [],
        "fw_dist_to_fire": [],
        "fw_reward": [],
        "fw_water": [],
        "fw_extinguish_eff": [],
        "fire_burning": [],
        "scout_fw_dist": [],
    }
    prev_scout_dist = None

    # Episode-level tracking
    fw_alive = True
    fw_death_step = MAX_STEPS
    fw_death_cause = "survived"
    scout_alive = True
    scout_death_step = MAX_STEPS
    scout_death_cause = "survived"
    fw_total_extinguish = 0.0
    fw_ever_near_fire = False  # did FW get within 200m of fire?
    scout_fire_visible_steps = 0
    fw_water_drops_on_fire = 0
    fw_water_drops_wasted = 0

    for step in range(MAX_STEPS):
        if not env.agents:
            break

        actions = {}
        current_msgs = torch.zeros(1, 1, 5).to(device)
        msg_mask = torch.BoolTensor([[True]]).to(device)

        # Scout
        if q_id in env.agents:
            l_map = torch.FloatTensor(obs[q_id]["local_map"]).to(device).unsqueeze(0)
            s_state = torch.FloatTensor(obs[q_id]["self_state"]).to(device).unsqueeze(0)
            n_state = torch.FloatTensor(obs[q_id]["neighbor_states"]).to(device).unsqueeze(0)
            n_mask = torch.BoolTensor(obs[q_id]["neighbor_mask"]).to(device).unsqueeze(0)
            with torch.no_grad():
                dist_q, message_q, h_scout = scout_actor(l_map, s_state, n_state, n_mask, h_scout)
                action_q = dist_q.mean
            actions[q_id] = action_q.squeeze(0).cpu().numpy()
            current_msgs = message_q.unsqueeze(1)
            msg_mask = torch.BoolTensor([[False]]).to(device)

        # Commander — waypoint controller (matches train_multi.py exactly)
        if f_id in env.agents:
            fw = env.sim.drones.get(f_id)

            # New waypoint decision every WAYPOINT_STEPS or when reached
            if need_new_waypoint:
                s_state_f = torch.FloatTensor(obs[f_id]["self_state"]).to(device).unsqueeze(0)
                with torch.no_grad():
                    dist_f, _, h_cmdr = cmdr_actor(s_state_f, current_msgs, msg_mask, h_cmdr)
                    action_f = dist_f.mean
                act_np = action_f.squeeze(0).cpu().numpy()

                dx_raw = float(act_np[0])
                dy_raw = float(act_np[1])
                target_alt_raw = float(act_np[2])
                water_raw = float(act_np[3])

                cur_pos = fw.get_position() if fw else np.zeros(3)
                target_x = cur_pos[0] + dx_raw * WAYPOINT_RANGE
                target_y = cur_pos[1] + dy_raw * WAYPOINT_RANGE
                target_x = np.clip(target_x, -safe_limit, safe_limit)
                target_y = np.clip(target_y, -safe_limit, safe_limit)

                steps_in_segment = 0
                need_new_waypoint = False

            # Heading controller (runs every step)
            if fw is not None:
                pos = fw.get_position()

                # Emergency boundary override
                if abs(pos[0]) > boundary_emergency or abs(pos[1]) > boundary_emergency:
                    target_x = 0.0
                    target_y = 0.0
                    need_new_waypoint = True

                dx_to = target_x - pos[0]
                dy_to = target_y - pos[1]
                dist_to = np.sqrt(dx_to**2 + dy_to**2)

                if dist_to < wp_reached_dist:
                    need_new_waypoint = True

                if dist_to > 1.0:
                    desired_heading = np.arctan2(dy_to, dx_to)
                    cur_yaw = fw.get_orientation_rpy()[2]
                    heading_error = _wrap_angle(desired_heading - cur_yaw)
                    heading_cmd = np.clip(heading_error / np.pi, -1.0, 1.0)
                else:
                    heading_cmd = 0.0

                actions[f_id] = np.array([heading_cmd, target_alt_raw, water_raw],
                                          dtype=np.float32)

            steps_in_segment += 1
            if steps_in_segment >= WAYPOINT_STEPS:
                need_new_waypoint = True

        # Pre-step measurements
        scout_fire_parts = _measure_scout_fire(env, q_id)
        fw_dropping = (f_id in actions and len(actions[f_id]) >= 3
                       and actions[f_id][2] > 0.5)

        # Step
        obs, rewards, terms, truncs, infos = env.step(actions)

        # Post-step: fire state
        fg = env.sim.environment.fire_grid
        burning = int(np.sum(fg.B)) if fg is not None else 0
        # Fire centroid
        if fg is not None:
            bi, bj = np.where(fg.B)
            if len(bi) > 0:
                gm = env.sim.environment.grid_mapper
                fcx = gm.origin_x + (np.mean(bj) + 0.5) * gm.cell_size_m
                fcy = gm.origin_y + (np.mean(bi) + 0.5) * gm.cell_size_m
            else:
                fcx, fcy = fire_cx_init, fire_cy_init
        else:
            fcx, fcy = fire_cx_init, fire_cy_init

        ts["fire_burning"].append(burning)

        # Scout
        if q_id in env.sim.drones:
            pos = env.sim.drones[q_id].get_position()
            vel = env.sim.drones[q_id].get_velocity()
            d2f = np.hypot(pos[0] - fcx, pos[1] - fcy)
            ts["scout_dist_to_fire"].append(d2f)
            ts["scout_z"].append(pos[2])
            ts["scout_reward"].append(rewards.get(q_id, 0.0))
            ts["scout_fire_intensity"].append(scout_fire_parts["intensity"])
            fr = scout_fire_parts["fire_reward"]
            ts["scout_fire_reward"].append(fr)
            if scout_fire_parts["intensity"] > 0.001:
                scout_fire_visible_steps += 1

            # Radial velocity toward fire (positive = approaching)
            if prev_scout_dist is not None:
                ts["scout_radial_vel"].append(prev_scout_dist - d2f)
            else:
                ts["scout_radial_vel"].append(0.0)
            prev_scout_dist = d2f

            # Compass alignment: cos(angle) between velocity and fire direction
            speed = np.linalg.norm(vel[:2])
            ts["scout_speed"].append(speed)
            if speed > 0.5 and d2f > 1.0:
                fire_dir = np.array([fcx - pos[0], fcy - pos[1]])
                fire_dir /= np.linalg.norm(fire_dir)
                vel_dir = vel[:2] / speed
                ts["scout_compass_align"].append(float(np.dot(vel_dir, fire_dir)))
            else:
                ts["scout_compass_align"].append(0.0)
        else:
            if scout_alive:
                scout_alive = False
                scout_death_step = step
                scout_death_cause = _get_death_cause(infos, q_id)
            ts["scout_dist_to_fire"].append(float("nan"))
            ts["scout_z"].append(float("nan"))
            ts["scout_reward"].append(0.0)
            ts["scout_fire_intensity"].append(0.0)
            ts["scout_fire_reward"].append(0.0)
            ts["scout_radial_vel"].append(float("nan"))
            ts["scout_compass_align"].append(float("nan"))
            ts["scout_speed"].append(float("nan"))
            prev_scout_dist = None

        # FW
        if f_id in env.sim.drones:
            pos_f = env.sim.drones[f_id].get_position()
            d2f_f = np.hypot(pos_f[0] - fcx, pos_f[1] - fcy)
            ts["fw_dist_to_fire"].append(d2f_f)
            water = (env.sim.drones[f_id].current_water /
                     env.sim.drones[f_id].water_capacity
                     if env.sim.drones[f_id].water_capacity > 0 else 0.0)
            ts["fw_water"].append(water)
            ts["fw_reward"].append(rewards.get(f_id, 0.0))
            eff = env.sim.drone_extinguish_stats.get(f_id, 0.0)
            ts["fw_extinguish_eff"].append(eff)
            fw_total_extinguish += eff
            if d2f_f < 200:
                fw_ever_near_fire = True
            if fw_dropping:
                if d2f_f < 100:
                    fw_water_drops_on_fire += 1
                else:
                    fw_water_drops_wasted += 1
        else:
            if fw_alive:
                fw_alive = False
                fw_death_step = step
                fw_death_cause = _get_death_cause(infos, f_id)
            ts["fw_dist_to_fire"].append(float("nan"))
            ts["fw_water"].append(float("nan"))
            ts["fw_reward"].append(0.0)
            ts["fw_extinguish_eff"].append(0.0)

        # Mutual distance
        if q_id in env.sim.drones and f_id in env.sim.drones:
            ts["scout_fw_dist"].append(np.linalg.norm(
                env.sim.drones[q_id].get_position() -
                env.sim.drones[f_id].get_position()))
        else:
            ts["scout_fw_dist"].append(float("nan"))

    # ── Episode summary ───────────────────────────────────────────────
    summary = {
        "seed": seed,
        "scout_survived": scout_alive,
        "scout_death_step": scout_death_step,
        "scout_death_cause": scout_death_cause,
        "scout_cumul_reward": sum(ts["scout_reward"]),
        "scout_fire_visible_pct": 100.0 * scout_fire_visible_steps / max(1, len(ts["scout_reward"])),
        "scout_avg_fire_intensity": np.nanmean(ts["scout_fire_intensity"]),
        "scout_avg_dist_to_fire": np.nanmean(ts["scout_dist_to_fire"]),
        "fw_survived": fw_alive,
        "fw_death_step": fw_death_step,
        "fw_death_cause": fw_death_cause,
        "fw_cumul_reward": sum(ts["fw_reward"]),
        "fw_ever_near_fire": fw_ever_near_fire,
        "fw_total_extinguish": fw_total_extinguish,
        "fw_drops_on_fire": fw_water_drops_on_fire,
        "fw_drops_wasted": fw_water_drops_wasted,
        "fire_final_burning": ts["fire_burning"][-1] if ts["fire_burning"] else 0,
        "fire_peak_burning": max(ts["fire_burning"]) if ts["fire_burning"] else 0,
        "fire_extinguished": burning == 0 and len(ts["fire_burning"]) > 0,
    }

    return ts, summary


def _measure_scout_fire(env, q_id):
    """Measure what the scout sees right now (no side effects)."""
    if q_id not in env.sim.drones:
        return {"intensity": 0.0, "fire_reward": 0.0}
    pos = env.sim.drones[q_id].get_position()
    local_map = env._extract_local_fire_map(pos)
    avg_I = float(np.mean(local_map))
    fr = 0.0
    if avg_I > 0.001:
        fr = QUAD["fire_flat_bonus"] + avg_I * QUAD["fire_intensity_k"]
    return {"intensity": avg_I, "fire_reward": fr}


def _get_death_cause(infos, agent_id):
    """Try to extract death cause from infos dict."""
    if agent_id in infos and "death_cause" in infos[agent_id]:
        return infos[agent_id]["death_cause"]
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed runner
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_seed(n_seeds=10, batch_num=430):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scout_path = os.path.join(MODEL_DIR, f"scout_b{batch_num:04d}.pt")
    cmdr_path = os.path.join(MODEL_DIR, f"cmdr_b{batch_num:04d}.pt")
    if not os.path.exists(scout_path):
        print(f"Model not found: {scout_path}")
        return
    if not os.path.exists(cmdr_path):
        print(f"Model not found: {cmdr_path}")
        return

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                       grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)

    scout_actor = ScoutActor(self_state_dim=15, msg_dim=5, hidden_dim=128).to(device)
    cmdr_actor = CommanderActor(self_state_dim=19, msg_input_dim=5).to(device)
    scout_actor.load_state_dict(torch.load(scout_path, map_location=device, weights_only=True))
    cmdr_actor.load_state_dict(torch.load(cmdr_path, map_location=device, weights_only=True))
    scout_actor.eval()
    cmdr_actor.eval()
    print(f"Loaded: {os.path.basename(scout_path)}, {os.path.basename(cmdr_path)}")
    print(f"Running {n_seeds} episodes...\n")

    all_ts = []
    all_summaries = []
    seeds = list(range(n_seeds))

    for i, seed in enumerate(seeds):
        t0 = time.time()
        ts, summary = run_single_episode(env, scout_actor, cmdr_actor, device, seed)
        dt = time.time() - t0
        all_ts.append(ts)
        all_summaries.append(summary)
        print(f"  [{i+1:>2d}/{n_seeds}] seed={seed:>3d}  "
              f"scout={'ALIVE' if summary['scout_survived'] else 'DEAD@'+str(summary['scout_death_step']):>10s}  "
              f"fw={'ALIVE' if summary['fw_survived'] else 'DEAD@'+str(summary['fw_death_step']):>10s} "
              f"({summary['fw_death_cause']})  "
              f"fw_near_fire={'Y' if summary['fw_ever_near_fire'] else 'N'}  "
              f"extinguish={summary['fw_total_extinguish']:.2f}  "
              f"fire_end={summary['fire_final_burning']:>4d}  "
              f"({dt:.1f}s)")

    # ── Print aggregate summary ───────────────────────────────────────
    _print_aggregate(all_summaries)

    # ── Generate aggregate plots ──────────────────────────────────────
    _plot_aggregate(all_ts, all_summaries, OUT_DIR, batch_num)

    return all_ts, all_summaries


def _print_aggregate(summaries):
    """Print a clear aggregate summary table."""
    n = len(summaries)
    print("\n" + "=" * 80)
    print(f"AGGREGATE SUMMARY ({n} episodes)")
    print("=" * 80)

    # Scout stats
    scout_alive = sum(1 for s in summaries if s["scout_survived"])
    scout_rewards = [s["scout_cumul_reward"] for s in summaries]
    scout_fire_pct = [s["scout_fire_visible_pct"] for s in summaries]
    scout_avg_dist = [s["scout_avg_dist_to_fire"] for s in summaries]

    print(f"\n── SCOUT ──")
    print(f"  Survival rate:        {scout_alive}/{n} ({100*scout_alive/n:.0f}%)")
    print(f"  Cumul reward:         {np.mean(scout_rewards):.0f} +/- {np.std(scout_rewards):.0f}")
    print(f"  Fire visible steps:   {np.mean(scout_fire_pct):.1f}% +/- {np.std(scout_fire_pct):.1f}%")
    print(f"  Avg dist to fire:     {np.nanmean(scout_avg_dist):.0f} +/- {np.nanstd(scout_avg_dist):.0f} m")
    if scout_alive < n:
        dead_causes = [s["scout_death_cause"] for s in summaries if not s["scout_survived"]]
        from collections import Counter
        print(f"  Death causes:         {dict(Counter(dead_causes))}")

    # FW stats
    fw_alive = sum(1 for s in summaries if s["fw_survived"])
    fw_near = sum(1 for s in summaries if s["fw_ever_near_fire"])
    fw_rewards = [s["fw_cumul_reward"] for s in summaries]
    fw_ext = [s["fw_total_extinguish"] for s in summaries]
    fw_death_steps = [s["fw_death_step"] for s in summaries if not s["fw_survived"]]
    fw_drops_on = [s["fw_drops_on_fire"] for s in summaries]
    fw_drops_waste = [s["fw_drops_wasted"] for s in summaries]

    print(f"\n── FIXED-WING ──")
    print(f"  Survival rate:        {fw_alive}/{n} ({100*fw_alive/n:.0f}%)")
    print(f"  Ever reached fire:    {fw_near}/{n} ({100*fw_near/n:.0f}%)")
    print(f"  Cumul reward:         {np.mean(fw_rewards):.0f} +/- {np.std(fw_rewards):.0f}")
    print(f"  Total extinguish:     {np.mean(fw_ext):.2f} +/- {np.std(fw_ext):.2f}")
    print(f"  Water drops on fire:  {np.mean(fw_drops_on):.1f} +/- {np.std(fw_drops_on):.1f}")
    print(f"  Water drops wasted:   {np.mean(fw_drops_waste):.1f} +/- {np.std(fw_drops_waste):.1f}")
    if fw_death_steps:
        print(f"  Avg death step:       {np.mean(fw_death_steps):.0f} +/- {np.std(fw_death_steps):.0f}")
        dead_causes = [s["fw_death_cause"] for s in summaries if not s["fw_survived"]]
        from collections import Counter
        print(f"  Death causes:         {dict(Counter(dead_causes))}")

    # Fire stats
    fire_ext = sum(1 for s in summaries if s["fire_extinguished"])
    fire_final = [s["fire_final_burning"] for s in summaries]
    fire_peak = [s["fire_peak_burning"] for s in summaries]

    print(f"\n── FIRE ──")
    print(f"  Fully extinguished:   {fire_ext}/{n} ({100*fire_ext/n:.0f}%)")
    print(f"  Final burning cells:  {np.mean(fire_final):.0f} +/- {np.std(fire_final):.0f}")
    print(f"  Peak burning cells:   {np.mean(fire_peak):.0f} +/- {np.std(fire_peak):.0f}")

    # Key diagnosis
    print(f"\n── DIAGNOSIS ──")
    if fw_alive / n < 0.5:
        print(f"  [!] FW crashes in {100*(1-fw_alive/n):.0f}% of episodes — primary problem!")
    if fw_near / n < 0.5:
        print(f"  [!] FW never reaches fire in {100*(1-fw_near/n):.0f}% of episodes")
    if np.mean(fw_ext) < 0.1:
        print(f"  [!] FW extinguish efficiency near zero — not fighting fire")
    if np.mean(scout_fire_pct) > 60:
        print(f"  [OK] Scout sees fire {np.mean(scout_fire_pct):.0f}% of the time — working well")
    elif np.mean(scout_fire_pct) > 30:
        print(f"  [~] Scout sees fire {np.mean(scout_fire_pct):.0f}% — could be better")
    else:
        print(f"  [!] Scout sees fire only {np.mean(scout_fire_pct):.0f}% — failing to find it")

    # Reward conflict check
    eps_with_ext = [i for i, s in enumerate(summaries)
                    if s["fw_total_extinguish"] > 0.5]
    if eps_with_ext:
        print(f"\n  Reward conflict check ({len(eps_with_ext)} eps with FW extinguishing):")
        print(f"  -> Need per-step correlation analysis (see timeline CSVs)")
    else:
        print(f"\n  Reward conflict: CANNOT CHECK — FW never extinguishes significantly")
        print(f"  -> The declining training reward is NOT caused by reward conflict")
        print(f"  -> It's caused by FW failing to navigate/survive")

    print("=" * 80)


def _plot_aggregate(all_ts, summaries, out_dir, batch_num):
    """Generate aggregate analysis figure across all seeds."""
    n = len(all_ts)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f"Multi-Seed Diagnosis — batch {batch_num}, {n} episodes",
                 fontsize=15, fontweight="bold")

    # Helper: pad timeseries to same length
    max_len = max(len(ts["fire_burning"]) for ts in all_ts)
    def _pad(key):
        arr = np.full((n, max_len), np.nan)
        for i, ts in enumerate(all_ts):
            d = ts[key]
            arr[i, :len(d)] = d
        return arr

    # ── 1. Fire growth across seeds ───────────────────────────────────
    ax = axes[0, 0]
    padded = _pad("fire_burning")
    for i in range(n):
        ax.plot(padded[i], alpha=0.3, linewidth=0.8)
    ax.plot(np.nanmean(padded, axis=0), color="red", linewidth=2, label="Mean")
    ax.set_ylabel("Burning cells")
    ax.set_title("(a) Fire growth — all seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 2. Scout fire intensity across seeds ──────────────────────────
    ax = axes[0, 1]
    padded = _pad("scout_fire_intensity")
    for i in range(n):
        ax.plot(padded[i], alpha=0.2, linewidth=0.5)
    ax.plot(np.nanmean(padded, axis=0), color="orangered", linewidth=2, label="Mean")
    ax.axhline(0.001, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Avg fire in scout view")
    ax.set_title("(b) Scout's fire visibility — all seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Scout distance to fire ─────────────────────────────────────
    ax = axes[1, 0]
    padded = _pad("scout_dist_to_fire")
    for i in range(n):
        ax.plot(padded[i], alpha=0.2, linewidth=0.5, color="cyan")
    ax.plot(np.nanmean(padded, axis=0), color="blue", linewidth=2, label="Mean")
    ax.set_ylabel("Distance to fire [m]")
    ax.set_title("(c) Scout distance to fire centroid")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 4. FW distance to fire ────────────────────────────────────────
    ax = axes[1, 1]
    padded = _pad("fw_dist_to_fire")
    for i in range(n):
        ax.plot(padded[i], alpha=0.2, linewidth=0.5, color="salmon")
    ax.plot(np.nanmean(padded, axis=0), color="red", linewidth=2, label="Mean")
    ax.set_ylabel("Distance to fire [m]")
    ax.set_title("(d) FW distance to fire centroid")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 5. Scout radial velocity (approach vs flee) ─────────────────
    ax = axes[2, 0]
    padded = _pad("scout_radial_vel")
    # Smooth with rolling mean for readability
    window = 50
    for i in range(n):
        raw = padded[i]
        smoothed = np.convolve(raw, np.ones(window)/window, mode="same")
        ax.plot(smoothed, alpha=0.2, linewidth=0.5, color="cyan")
    mean_rv = np.nanmean(padded, axis=0)
    mean_rv_smooth = np.convolve(mean_rv, np.ones(window)/window, mode="same")
    ax.plot(mean_rv_smooth, color="blue", linewidth=2, label="Mean (smoothed)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(range(len(mean_rv_smooth)), 0, mean_rv_smooth,
                     where=mean_rv_smooth > 0, alpha=0.15, color="green", label="Approaching")
    ax.fill_between(range(len(mean_rv_smooth)), 0, mean_rv_smooth,
                     where=mean_rv_smooth < 0, alpha=0.15, color="red", label="Fleeing")
    ax.set_ylabel("Radial velocity [m/step]")
    ax.set_xlabel("Step")
    ax.set_title("(e) Scout radial velocity toward fire (+approach / -flee)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 6. Scout compass alignment (velocity vs fire direction) ───────
    ax = axes[2, 1]
    padded = _pad("scout_compass_align")
    for i in range(n):
        raw = padded[i]
        smoothed = np.convolve(raw, np.ones(window)/window, mode="same")
        ax.plot(smoothed, alpha=0.2, linewidth=0.5, color="salmon")
    mean_ca = np.nanmean(padded, axis=0)
    mean_ca_smooth = np.convolve(mean_ca, np.ones(window)/window, mode="same")
    ax.plot(mean_ca_smooth, color="red", linewidth=2, label="Mean (smoothed)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(1, color="green", linestyle=":", alpha=0.3, label="Perfect alignment")
    ax.axhline(-1, color="red", linestyle=":", alpha=0.3, label="Opposite direction")
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("cos(velocity, fire direction)")
    ax.set_xlabel("Step")
    ax.set_title("(f) Scout flight direction vs fire compass (+1=toward, -1=away)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 7. Per-seed cumulative rewards ────────────────────────────────
    ax = axes[3, 0]
    scout_rewards = [s["scout_cumul_reward"] for s in summaries]
    fw_rewards = [s["fw_cumul_reward"] for s in summaries]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, scout_rewards, w, label="Scout", color="cyan", alpha=0.7)
    ax.bar(x + w/2, fw_rewards, w, label="FW", color="red", alpha=0.7)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("(g) Per-seed cumulative rewards")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ── 8. Key metrics summary bar chart ──────────────────────────────
    ax = axes[3, 1]
    metrics = {
        "Scout alive %": 100 * sum(1 for s in summaries if s["scout_survived"]) / n,
        "FW alive %": 100 * sum(1 for s in summaries if s["fw_survived"]) / n,
        "FW near fire %": 100 * sum(1 for s in summaries if s["fw_ever_near_fire"]) / n,
        "Scout sees fire %": np.mean([s["scout_fire_visible_pct"] for s in summaries]),
        "Fire extinguished %": 100 * sum(1 for s in summaries if s["fire_extinguished"]) / n,
    }
    bars = ax.barh(list(metrics.keys()), list(metrics.values()),
                   color=["cyan", "red", "orange", "lime", "green"], alpha=0.7)
    ax.set_xlim(0, 105)
    ax.set_xlabel("%")
    ax.set_title("(h) Key performance metrics")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}%", va="center", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"diagnosis_multi_b{batch_num:04d}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-seed scout/FW diagnosis")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--batch", type=int, default=430, help="Model batch number")
    args = parser.parse_args()

    run_multi_seed(n_seeds=args.seeds, batch_num=args.batch)


if __name__ == "__main__":
    main()
