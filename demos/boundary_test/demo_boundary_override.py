"""
demo_boundary_override.py
─────────────────────────
Test: navedeme fixed-wing přímo k hranici mapy a ve chvíli kdy překročí
boundary_emergency threshold, přepíšeme waypoint na [0, 0].

Otázka: zvládne se letadlo otočit a přežít, nebo ztratí stabilitu?

Výstup:
  - demo_boundary_test.png  (trajektorie + výška + heading + dist_to_edge)
  - Konzolový výpis: přežil / zemřel, min dist od okraje, otáčecí čas
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv

# ── Config ───────────────────────────────────────────────────────────────
GRID_SIZE = 1000.0          # 1km map, map_bounds = 500
MAX_STEPS = 1500
MAP_BOUNDS = GRID_SIZE / 2.0
BOUNDARY_EMERGENCY = MAP_BOUNDS - 200.0   # 300m — override triggers here
KILL_ZONE = MAP_BOUNDS                    # 500m — death

# ── Helpers ──────────────────────────────────────────────────────────────
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ── Setup env ────────────────────────────────────────────────────────────
env = DroneFireEnv(num_quads=0, num_fixed=1, grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
obs, infos = env.reset(seed=42, epizode_number=0)
f_agent = env.fixed_agents[0]

# ── State tracking ───────────────────────────────────────────────────────
xs, ys, zs = [], [], []
headings = []
dist_to_edges = []
override_step = None
phase = "OUTBOUND"  # OUTBOUND → OVERRIDE → RETURNING
target_x, target_y = MAP_BOUNDS + 50.0, 0.0   # waypoint outside map!
target_alt_raw = 0.0   # mid altitude
survived = True

print(f"Map: {GRID_SIZE}m (bounds ±{MAP_BOUNDS})")
print(f"Boundary emergency: ±{BOUNDARY_EMERGENCY}m")
print(f"Phase 1: fly straight to x={target_x} (past boundary)")
print(f"Phase 2: when |x| > {BOUNDARY_EMERGENCY}, override to [0, 0]")
print(f"{'='*60}")

for step in range(MAX_STEPS):
    drone = env.sim.drones.get(f_agent)
    if drone is None:
        print(f"  Step {step}: DRONE DESTROYED")
        survived = False
        break

    pos = drone.get_position()
    rpy = drone.get_orientation_rpy()
    dist_edge = min(
        MAP_BOUNDS - abs(pos[0]),
        MAP_BOUNDS - abs(pos[1])
    )

    xs.append(pos[0])
    ys.append(pos[1])
    zs.append(pos[2])
    headings.append(np.degrees(rpy[2]))
    dist_to_edges.append(dist_edge)

    # ── Phase transition ─────────────────────────────────────────────
    if phase == "OUTBOUND" and (abs(pos[0]) > BOUNDARY_EMERGENCY or abs(pos[1]) > BOUNDARY_EMERGENCY):
        phase = "OVERRIDE"
        override_step = step
        target_x = 0.0
        target_y = 0.0
        print(f"  Step {step}: OVERRIDE triggered at pos=({pos[0]:.0f}, {pos[1]:.0f}), dist_edge={dist_edge:.0f}m")
        print(f"            Heading: {np.degrees(rpy[2]):.1f}°, new target: [0, 0]")

    if phase == "OVERRIDE" and abs(pos[0]) < BOUNDARY_EMERGENCY and abs(pos[1]) < BOUNDARY_EMERGENCY:
        phase = "RETURNING"
        print(f"  Step {step}: Safe zone reached at pos=({pos[0]:.0f}, {pos[1]:.0f})")

    # ── Heading controller (same as train_multi.py) ──────────────────
    dx_to = target_x - pos[0]
    dy_to = target_y - pos[1]
    dist_to = np.sqrt(dx_to**2 + dy_to**2)

    if dist_to > 1.0:
        desired_heading = np.arctan2(dy_to, dx_to)
        heading_error = wrap_angle(desired_heading - rpy[2])
        heading_cmd = np.clip(heading_error / np.pi, -1.0, 1.0)
    else:
        heading_cmd = 0.0

    action = np.array([heading_cmd, target_alt_raw, 0.0], dtype=np.float32)
    actions = {f_agent: action}

    obs, rewards, terms, truncs, infos = env.step(actions)

    if terms.get(f_agent, False):
        cause = infos.get(f_agent, {}).get("death_cause", "unknown")
        print(f"  Step {step}: DEAD — {cause} at pos=({pos[0]:.0f}, {pos[1]:.0f})")
        survived = False
        break

    if truncs.get(f_agent, False):
        print(f"  Step {step}: Episode truncated (time limit)")
        break

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Result: {'SURVIVED' if survived else 'CRASHED'}")
print(f"Total steps: {len(xs)}")
if override_step is not None:
    min_dist = min(dist_to_edges[override_step:])
    min_dist_step = override_step + dist_to_edges[override_step:].index(min_dist)
    print(f"Override at step {override_step}")
    print(f"Min distance to edge after override: {min_dist:.1f}m (step {min_dist_step})")
    print(f"Steps from override to min dist: {min_dist_step - override_step}")
    if survived and phase == "RETURNING":
        # Find when it got back inside safe zone
        for s in range(override_step, len(dist_to_edges)):
            if dist_to_edges[s] > (MAP_BOUNDS - BOUNDARY_EMERGENCY) + 10:
                print(f"Time to turn around: {s - override_step} steps ({(s - override_step)*5/240:.1f}s physics)")
                break
else:
    print("Override never triggered!")

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Boundary Override Test — {'SURVIVED' if survived else 'CRASHED'}", fontsize=14, fontweight='bold')

# 1. Top-down trajectory
ax = axes[0, 0]
ax.set_title("Trajectory (top-down)")
ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.7)
ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
ax.plot(xs[-1], ys[-1], 'r*' if not survived else 'g*', markersize=12, label='End')
if override_step is not None:
    ax.plot(xs[override_step], ys[override_step], 'r^', markersize=12, label=f'Override (step {override_step})')
# Draw boundaries
bnd = MAP_BOUNDS
ax.plot([-bnd, bnd, bnd, -bnd, -bnd], [-bnd, -bnd, bnd, bnd, -bnd], 'r--', linewidth=2, label=f'Kill zone (±{bnd:.0f}m)')
emg = BOUNDARY_EMERGENCY
ax.plot([-emg, emg, emg, -emg, -emg], [-emg, -emg, emg, emg, -emg], 'orange', linewidth=1.5, linestyle='--', label=f'Emergency (±{emg:.0f}m)')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 2. Altitude
ax = axes[0, 1]
ax.set_title("Altitude (m)")
ax.plot(zs, 'b-', linewidth=0.8)
if override_step:
    ax.axvline(override_step, color='r', linestyle='--', alpha=0.7, label='Override')
ax.set_xlabel("Step")
ax.set_ylabel("Altitude (m)")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Distance to nearest edge
ax = axes[1, 0]
ax.set_title("Distance to nearest edge (m)")
ax.plot(dist_to_edges, 'g-', linewidth=0.8)
ax.axhline(0, color='r', linestyle='-', linewidth=2, label='Kill zone')
ax.axhline(MAP_BOUNDS - BOUNDARY_EMERGENCY, color='orange', linestyle='--', label=f'Emergency threshold ({MAP_BOUNDS - BOUNDARY_EMERGENCY:.0f}m)')
if override_step:
    ax.axvline(override_step, color='r', linestyle='--', alpha=0.7, label='Override step')
ax.set_xlabel("Step")
ax.set_ylabel("Distance (m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Heading
ax = axes[1, 1]
ax.set_title("Heading (degrees)")
ax.plot(headings, 'purple', linewidth=0.8)
if override_step:
    ax.axvline(override_step, color='r', linestyle='--', alpha=0.7, label='Override')
ax.set_xlabel("Step")
ax.set_ylabel("Heading (°)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_boundary_test.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")
plt.close()

if env.sim is not None:
    env.sim.stop_simulation()
