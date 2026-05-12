#!/usr/bin/env python3
"""
diagnose_episode.py — Visual + text diagnostic of a single episode.

Outputs:
  output/diag_trajectories.png  — top-down map with drone paths, fire, valve events
  output/diag_timeseries.png    — altitude, distance-to-scout, valve, water over time
  output/diag_log.txt           — step-by-step text log
"""
import sys, os, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(os.path.join(PROJECT, "src"))

from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor
from commander_control import CommanderController

# ── Config ──────────────────────────────────────────────────
SCOUT_PATH = os.path.join(PROJECT, "results/TrainingQuad/050626_TrainingMultipleScoutsOnMultipleFires/scout_b0120.pt")
CMDR_PATH  = os.path.join(PROJECT, "saved_models/finetune/cmdr_best.pt")
N_QUADS, N_FIXED = 4, 3
GRID = 1200.0
MAX_STEPS = 1000
SEED = 42
OUT_DIR = os.path.join(PROJECT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

dev = torch.device("cpu")

# ── Load models ─────────────────────────────────────────────
env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                   grid_size_m=GRID, max_steps=MAX_STEPS,
                   n_fires_range=(1, 1))
s_dim = env.observation_space("quad_0")["self_state"].shape[0]
f_dim = env.observation_space(env.fixed_agents[0])["self_state"].shape[0]

scout_actor = ScoutActor(self_state_dim=s_dim, msg_dim=5, hidden_dim=128).to(dev)
cmdr_actor  = CommanderActor(self_state_dim=f_dim, msg_input_dim=5,
                             action_dim=4, hidden_dim=64).to(dev)
scout_actor.load_state_dict(torch.load(SCOUT_PATH, map_location=dev), strict=False)
cmdr_actor.load_state_dict(torch.load(CMDR_PATH, map_location=dev), strict=False)
scout_actor.eval(); cmdr_actor.eval()

obs, _ = env.reset(seed=SEED)
map_half = env.map_bounds

# ── Per-agent state ─────────────────────────────────────────
ctrls = {}; h_cmdrs = {}
for fa in env.fixed_agents:
    c = CommanderController(waypoint_range=200.0, waypoint_steps=30, wp_reached_dist=30.0)
    c.reset(map_half)
    ctrls[fa] = c
    h_cmdrs[fa] = torch.zeros(1, 1, 64)

h_scouts = {q: torch.zeros(1, 1, 128) for q in env.quad_agents}
scout_msgs = {q: np.zeros(5) for q in env.quad_agents}

# ── Recording buffers ──────────────────────────────────────
rec = {
    "scout_xy": {q: [] for q in env.quad_agents},
    "fw_xy":    {f: [] for f in env.fixed_agents},
    "fw_alt":   {f: [] for f in env.fixed_agents},
    "fw_water": {f: [] for f in env.fixed_agents},
    "fw_valve": {f: [] for f in env.fixed_agents},
    "fw_dist_to_nearest_scout": {f: [] for f in env.fixed_agents},
    "fw_speed": {f: [] for f in env.fixed_agents},
    "fw_mode":  {f: [] for f in env.fixed_agents},  # "nn", "scripted", "emergency"
    "fire_cells": [],
    "valve_events_xy": [],  # (x, y) where valve opened
    "scout_msg2": {q: [] for q in env.quad_agents},  # fire_intensity in msg
    "scout_dist_to_fire": {q: [] for q in env.quad_agents},
    "valve_debug": [],  # detailed valve logic debug
}

log_lines = []
def log(msg):
    log_lines.append(msg)
    print(msg)

log(f"Map: {GRID}m  map_half={map_half}  seed={SEED}")
log(f"Scout model: {os.path.basename(SCOUT_PATH)}")
log(f"Cmdr model:  {os.path.basename(CMDR_PATH)}")
log(f"safe_limit={ctrls[env.fixed_agents[0]].safe_limit:.0f}m  "
    f"boundary_emergency={ctrls[env.fixed_agents[0]].boundary_emergency:.0f}m")
log("")

# ── Main loop ──────────────────────────────────────────────
for step in range(MAX_STEPS):
    actions = {}

    # Scouts
    for q in env.quad_agents:
        if q not in obs:
            continue
        o = obs[q]
        lm = torch.FloatTensor(o["local_map"]).unsqueeze(0)
        ss = torch.FloatTensor(o["self_state"]).unsqueeze(0)
        ns = torch.FloatTensor(o["neighbor_states"]).unsqueeze(0)
        nm = torch.BoolTensor(o["neighbor_mask"]).unsqueeze(0)
        with torch.no_grad():
            dist, msg, h_scouts[q] = scout_actor(lm, ss, ns, nm, h_scouts[q])
        actions[q] = dist.mean.squeeze(0).numpy()
        scout_msgs[q] = msg.squeeze(0).detach().numpy()
        if q in env.sim.drones:
            p = env.sim.drones[q].get_position()
            rec["scout_xy"][q].append((p[0], p[1]))
            rec["scout_msg2"][q].append(float(scout_msgs[q][2]))
            # distance to fire
            d_fire = np.hypot(p[0] - env.fire_x, p[1] - env.fire_y)
            rec["scout_dist_to_fire"][q].append(d_fire)

    # Commanders
    for fa in env.fixed_agents:
        if fa not in obs:
            continue
        o = obs[fa]
        drone = env.sim.drones.get(fa)
        if drone is None:
            continue
        pos = drone.get_position()
        rec["fw_xy"][fa].append((pos[0], pos[1]))
        rec["fw_alt"][fa].append(pos[2])
        rec["fw_water"][fa].append(drone.current_water if hasattr(drone, 'current_water') else 0)
        vel = drone.get_velocity()
        rec["fw_speed"][fa].append(np.linalg.norm(vel[:2]))

        ctrl_i = ctrls[fa]
        s_st_t = torch.FloatTensor(o["self_state"]).unsqueeze(0)
        ml, mk = [], []
        for q in env.quad_agents:
            if q in env.sim.drones:
                ml.append(torch.FloatTensor(scout_msgs[q]))
                mk.append(False)
            else:
                ml.append(torch.zeros(5))
                mk.append(True)
        msgs_t = torch.stack(ml).unsqueeze(0)
        mask_t = torch.BoolTensor(mk).unsqueeze(0)
        ctrl_i.last_scout_msgs = msgs_t
        ctrl_i.last_scout_mask = mask_t

        # FW neighbor states
        fw_nl, fw_ml = [], []
        for f2 in env.fixed_agents:
            if f2 == fa: continue
            d2 = env.sim.drones.get(f2)
            if d2 is not None:
                p2 = d2.get_position()
                fw_nl.append([(p2[0]-pos[0])/max(map_half,1), (p2[1]-pos[1])/max(map_half,1), (p2[2]-pos[2])/100])
                fw_ml.append(False)
            else:
                fw_nl.append([0,0,0]); fw_ml.append(True)
        fw_nt = torch.FloatTensor([fw_nl]); fw_mt = torch.BoolTensor([fw_ml])

        action, h_cmdrs[fa], info = ctrl_i.step(
            drone, o["self_state"], env, cmdr_actor, h_cmdrs[fa],
            msgs_t, mask_t, deterministic=True,
            fw_neighbor_states=fw_nt, fw_neighbor_mask=fw_mt)
        actions[fa] = action

        # Record valve
        valve_open = len(action) > 2 and action[2] > 0.5
        rec["fw_valve"][fa].append(1 if valve_open else 0)
        if valve_open:
            rec["valve_events_xy"].append((pos[0], pos[1]))

        # Mode
        if info.get('in_emergency'):
            rec["fw_mode"][fa].append("E")
        elif info.get('scripted'):
            rec["fw_mode"][fa].append("S")
        else:
            rec["fw_mode"][fa].append("N")

        # Distance to nearest alive scout
        min_dist = 9999.0
        nearest_scout_msg2 = 0.0
        nearest_scout_id = ""
        for q in env.quad_agents:
            if q in env.sim.drones:
                sq = env.sim.drones[q].get_position()
                d = np.hypot(pos[0]-sq[0], pos[1]-sq[1])
                if d < min_dist:
                    min_dist = d
                    nearest_scout_msg2 = float(scout_msgs[q][2])
                    nearest_scout_id = q
        rec["fw_dist_to_nearest_scout"][fa].append(min_dist)

        # Valve debug: log when FW is close to scout
        if min_dist < 50.0:
            rec["valve_debug"].append(
                f"  step={step} {fa}: dist={min_dist:.1f}m alt={pos[2]:.1f} "
                f"water={drone.current_water:.0f} "
                f"nearest={nearest_scout_id} msg2={nearest_scout_msg2:.4f} "
                f"valve={'OPEN' if (len(action)>2 and action[2]>0.5) else 'CLOSED'}"
            )

    # Fire stats
    fg = env.sim.environment.fire_grid
    if fg is not None:
        fs = fg.get_stats()
        rec["fire_cells"].append(fs.get("burning_cells", 0))
    else:
        rec["fire_cells"].append(0)

    obs, rewards, terms, truncs, infos = env.step(actions)

    # Log key events every 100 steps
    if step % 100 == 0 or step == MAX_STEPS - 1:
        alive_sc = sum(1 for q in env.quad_agents if q in env.sim.drones)
        alive_fw = sum(1 for f in env.fixed_agents if f in env.sim.drones)
        fire_n = rec["fire_cells"][-1]
        fw_info = []
        for fa in env.fixed_agents:
            if rec["fw_alt"][fa]:
                alt = rec["fw_alt"][fa][-1]
                w = rec["fw_water"][fa][-1]
                d = rec["fw_dist_to_nearest_scout"][fa][-1]
                spd = rec["fw_speed"][fa][-1]
                mode = rec["fw_mode"][fa][-1]
                v_count = sum(rec["fw_valve"][fa])
                fw_info.append(f"{fa}[{mode}] alt={alt:.0f} w={w:.0f} d_sc={d:.0f} spd={spd:.1f} valve_tot={v_count}")
        log(f"step {step:4d}  scouts={alive_sc}/{N_QUADS}  fw={alive_fw}/{N_FIXED}  fire={fire_n}")
        for fi in fw_info:
            log(f"    {fi}")

env.close()

# ── Summary stats ──────────────────────────────────────────
log(f"\n{'='*60}")
log("SUMMARY")
log(f"{'='*60}")
total_valve = sum(sum(rec["fw_valve"][f]) for f in env.fixed_agents)
log(f"Total valve-open steps: {total_valve}")
log(f"Valve events (positions): {len(rec['valve_events_xy'])}")
log(f"Final fire cells: {rec['fire_cells'][-1] if rec['fire_cells'] else '?'}")
log(f"Peak fire cells:  {max(rec['fire_cells']) if rec['fire_cells'] else '?'}")
for fa in env.fixed_agents:
    if rec["fw_dist_to_nearest_scout"][fa]:
        dists = rec["fw_dist_to_nearest_scout"][fa]
        modes = rec["fw_mode"][fa]
        n_mode = {"N": modes.count("N"), "S": modes.count("S"), "E": modes.count("E")}
        log(f"  {fa}: min_dist_scout={min(dists):.0f}m  avg={np.mean(dists):.0f}m  "
            f"modes: NN={n_mode['N']} scripted={n_mode['S']} emergency={n_mode['E']}  "
            f"valve_opens={sum(rec['fw_valve'][fa])}")

# Scout message stats
log(f"\n{'─'*60}")
log("SCOUT MESSAGES (msg[2] = fire intensity)")
for q in env.quad_agents:
    vals = rec["scout_msg2"].get(q, [])
    dists = rec["scout_dist_to_fire"].get(q, [])
    if vals:
        log(f"  {q}: msg2 max={max(vals):.4f}  mean={np.mean(vals):.4f}  "
            f"min_dist_fire={min(dists):.0f}m  avg_dist_fire={np.mean(dists):.0f}m")

# Valve debug log
if rec["valve_debug"]:
    log(f"\n{'─'*60}")
    log(f"VALVE DEBUG ({len(rec['valve_debug'])} close approaches):")
    for line in rec["valve_debug"][:50]:  # limit output
        log(line)
else:
    log(f"\n{'─'*60}")
    log("VALVE DEBUG: No FW came within 50m of any scout!")

# ── Save log ───────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "diag_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

# ── Plot 1: Trajectories ──────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xlim(-map_half*1.1, map_half*1.1)
ax.set_ylim(-map_half*1.1, map_half*1.1)
ax.set_aspect("equal")
ax.set_facecolor("#f5f5dc")

# Map boundary
rect = plt.Rectangle((-map_half, -map_half), 2*map_half, 2*map_half,
                      fill=False, edgecolor="black", linewidth=2, linestyle="--", label="Map boundary")
ax.add_patch(rect)

# Fire start
if hasattr(env, 'fire_positions'):
    for fx, fy in env.fire_positions:
        ax.plot(fx, fy, 'r*', markersize=20, zorder=10, label="Fire start")

# Scout trajectories
colors_sc = ["#00bfff", "#00ff7f", "#ff69b4", "#ffa500"]
for i, q in enumerate(env.quad_agents):
    xy = rec["scout_xy"][q]
    if xy:
        xs, ys = zip(*xy)
        ax.plot(xs, ys, color=colors_sc[i % len(colors_sc)], linewidth=0.5, alpha=0.6)
        ax.plot(xs[0], ys[0], 'o', color=colors_sc[i % len(colors_sc)], markersize=6)
        ax.plot(xs[-1], ys[-1], 's', color=colors_sc[i % len(colors_sc)], markersize=6)

# FW trajectories
colors_fw = ["#ff0000", "#cc0066", "#990099"]
for i, f in enumerate(env.fixed_agents):
    xy = rec["fw_xy"][f]
    if xy:
        xs, ys = zip(*xy)
        ax.plot(xs, ys, color=colors_fw[i % len(colors_fw)], linewidth=1.0, alpha=0.8)
        ax.plot(xs[0], ys[0], '^', color=colors_fw[i % len(colors_fw)], markersize=10, label=f"{f} start")
        ax.plot(xs[-1], ys[-1], 'v', color=colors_fw[i % len(colors_fw)], markersize=10)

# Valve events
if rec["valve_events_xy"]:
    vx, vy = zip(*rec["valve_events_xy"])
    ax.scatter(vx, vy, c="blue", marker="x", s=15, alpha=0.5, zorder=9, label="Valve open")

# Safe limit / emergency circles
sl = ctrls[env.fixed_agents[0]].safe_limit
be = ctrls[env.fixed_agents[0]].boundary_emergency
for r, c, lbl in [(sl, "orange", "safe_limit"), (be, "red", "boundary_emerg")]:
    rect2 = plt.Rectangle((-r, -r), 2*r, 2*r, fill=False, edgecolor=c, linewidth=1, linestyle=":", label=lbl)
    ax.add_patch(rect2)

ax.legend(loc="upper left", fontsize=7)
ax.set_title(f"Episode Trajectories — {GRID:.0f}m map, seed={SEED}")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "diag_trajectories.png"), dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/diag_trajectories.png")

# ── Plot 2: Time series ───────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

# 2a: FW altitude
ax = axes[0]
for i, f in enumerate(env.fixed_agents):
    if rec["fw_alt"][f]:
        ax.plot(rec["fw_alt"][f], color=colors_fw[i], label=f, linewidth=0.8)
ax.axhline(80, color="gray", linestyle="--", alpha=0.5, label="valve alt limit (80m)")
ax.axhline(30, color="green", linestyle="--", alpha=0.5, label="min alt (30m)")
ax.set_ylabel("Altitude [m]"); ax.legend(fontsize=7); ax.set_title("FW Altitude")

# 2b: Distance to nearest scout
ax = axes[1]
for i, f in enumerate(env.fixed_agents):
    if rec["fw_dist_to_nearest_scout"][f]:
        ax.plot(rec["fw_dist_to_nearest_scout"][f], color=colors_fw[i], label=f, linewidth=0.8)
ax.axhline(30, color="green", linestyle="--", alpha=0.5, label="valve dist limit (30m)")
ax.set_ylabel("Dist to scout [m]"); ax.legend(fontsize=7); ax.set_title("FW Distance to Nearest Scout")
ax.set_ylim(0, min(1000, max(200, max(max(rec["fw_dist_to_nearest_scout"][f]) for f in env.fixed_agents if rec["fw_dist_to_nearest_scout"][f]))))

# 2c: Valve state
ax = axes[2]
for i, f in enumerate(env.fixed_agents):
    if rec["fw_valve"][f]:
        ax.plot(rec["fw_valve"][f], color=colors_fw[i], label=f, linewidth=0.8, alpha=0.7)
ax.set_ylabel("Valve (0/1)"); ax.legend(fontsize=7); ax.set_title("FW Water Valve State")

# 2d: Water remaining
ax = axes[3]
for i, f in enumerate(env.fixed_agents):
    if rec["fw_water"][f]:
        ax.plot(rec["fw_water"][f], color=colors_fw[i], label=f, linewidth=0.8)
ax.set_ylabel("Water [L]"); ax.legend(fontsize=7); ax.set_title("FW Water Remaining")

# 2e: Fire cells
ax = axes[4]
if rec["fire_cells"]:
    ax.plot(rec["fire_cells"], color="red", linewidth=1.0)
ax.set_ylabel("Burning cells"); ax.set_xlabel("Step"); ax.set_title("Fire Spread")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "diag_timeseries.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/diag_timeseries.png")

# ── Plot 3: FW speed ──────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
for i, f in enumerate(env.fixed_agents):
    if rec["fw_speed"][f]:
        ax.plot(rec["fw_speed"][f], color=colors_fw[i], label=f, linewidth=0.8)
ax.axhline(30, color="gray", linestyle="--", alpha=0.5, label="max (30 m/s)")
ax.axhline(15, color="green", linestyle="--", alpha=0.5, label="slow (15 m/s)")
ax.set_ylabel("Speed [m/s]"); ax.set_xlabel("Step"); ax.legend(fontsize=7)
ax.set_title("FW Horizontal Speed")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "diag_speed.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/diag_speed.png")

print("\nDone.")
