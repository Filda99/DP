"""
diagnose_valve.py — Diagnostic: valve open/close accuracy
==========================================================

Simulates 10 straight FW flyovers through fire.
For each pass, manually moves FW step by step and checks
whether the valve logic opens at the right position.

Usage:
    cd src && python3 ../tools/diagnose_valve.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from commander_control import CommanderController

# ── Parameters ────────────────────────────────────────────────────
FIRE_X, FIRE_Y = 0.0, 0.0     # fire position
QUAD_ALT       = 80.0          # scout altitude
FW_ALT         = 55.0          # FW altitude
FW_SPEED       = 20.0          # m/s
DT             = 1.0 / 240.0   # simulation timestep
N_PASSES       = 10
MAP_BOUNDS     = 600.0

# ── Fake objects to satisfy the valve logic ───────────────────────
class FakeDrone:
    def __init__(self, pos, water=200.0):
        self.pos = np.array(pos, dtype=float)
        self.current_water = water
        self.water_capacity = 200.0
    def get_position(self):
        return self.pos.copy()
    def get_orientation_rpy(self):
        return [0, 0, self.heading]
    def get_velocity(self):
        return np.array([0, 0, 0])

class FakeEnv:
    def __init__(self, quad_drone):
        self.quad_agents = ['quad_0']
        self.sim = type('S', (), {'drones': {'quad_0': quad_drone}})()

# ── Setup ─────────────────────────────────────────────────────────
# Scout hovering directly over fire, seeing fire centered in camera
scout_drone = FakeDrone([FIRE_X, FIRE_Y, QUAD_ALT])
fake_env = FakeEnv(scout_drone)

NORM_DIST = 1000.0
scout_msg = torch.FloatTensor([[
    FIRE_X / NORM_DIST,   # msg[0]: scout norm_x
    FIRE_Y / NORM_DIST,   # msg[1]: scout norm_y
    0.5,                  # msg[2]: fire intensity
    0.0,                  # msg[3]: dyn_x (fire centered)
    0.0,                  # msg[4]: dyn_y
]])
scout_mask = torch.BoolTensor([False])

# Estimated fire from message
fov_half = max(10.0, QUAD_ALT * 1.5) / 2.0
est_x = FIRE_X + 0.0 * fov_half
est_y = FIRE_Y + 0.0 * fov_half
print(f"Fire at ({FIRE_X}, {FIRE_Y})")
print(f"Scout at ({FIRE_X}, {FIRE_Y}, {QUAD_ALT}m), FOV half={fov_half:.0f}m")
print(f"Estimated fire: ({est_x}, {est_y})")
print()

# ── Controller ────────────────────────────────────────────────────
ctrl = CommanderController()
ctrl.reset(map_half=MAP_BOUNDS)
ctrl.last_scout_msgs = scout_msg
ctrl.last_scout_mask = scout_mask

# ── Run passes ────────────────────────────────────────────────────
print(f"{'Pass':>4}  {'Angle':>5}  {'Valve Steps':>11}  {'Water Used':>10}  "
      f"{'Min Dist':>8}  {'Valve Range':>11}  {'First Open':>10}")
print("-" * 78)

all_results = []

for pass_idx in range(N_PASSES):
    angle = pass_idx * (2 * np.pi / N_PASSES)
    start_dist = 200.0

    # FW starts at distance, heading towards fire
    start_x = FIRE_X + start_dist * np.cos(angle)
    start_y = FIRE_Y + start_dist * np.sin(angle)
    heading = np.arctan2(FIRE_Y - start_y, FIRE_X - start_x)

    # Create FW drone
    fw = FakeDrone([start_x, start_y, FW_ALT], water=200.0)
    fw.heading = heading

    # Set waypoint to fire
    ctrl.target_x = FIRE_X
    ctrl.target_y = FIRE_Y
    ctrl.target_alt_raw = 0.0
    ctrl.steps_in_segment = 0
    ctrl.wp_reached = False

    valve_steps = 0
    min_dist = float('inf')
    valve_dists = []
    first_open_dist = None
    water_start = fw.current_water

    # Fly straight through (use larger time steps for speed)
    step_dt = 0.05  # 50ms per step, FW moves 1m per step at 20m/s
    total_time = 25.0  # enough to fly 500m
    n_steps = int(total_time / step_dt)

    for step in range(n_steps):
        pos = fw.get_position()
        dist = np.hypot(pos[0] - FIRE_X, pos[1] - FIRE_Y)
        min_dist = min(min_dist, dist)

        # Reset segment counter so heading_action doesn't expire
        ctrl.steps_in_segment = 0
        ctrl.wp_reached = False

        # Call heading_action to get valve decision
        action = ctrl.heading_action(fw, env=fake_env)
        valve_cmd = action[2]

        if valve_cmd > 0:
            valve_steps += 1
            valve_dists.append(dist)
            if first_open_dist is None:
                first_open_dist = dist
            fw.current_water = max(0, fw.current_water - 100.0 * step_dt)

        # Move FW forward
        fw.pos[0] += FW_SPEED * np.cos(heading) * step_dt
        fw.pos[1] += FW_SPEED * np.sin(heading) * step_dt

        # Stop after passing through
        if dist > 100.0 and step > 100:
            break

    water_used = water_start - fw.current_water
    valve_range = f"{min(valve_dists):.0f}-{max(valve_dists):.0f}m" if valve_dists else "n/a"
    first_str = f"{first_open_dist:.0f}m" if first_open_dist else "n/a"

    print(f"{pass_idx+1:4d}  {np.degrees(angle):5.0f}°  {valve_steps:11d}  "
          f"{water_used:8.1f} L  {min_dist:6.1f} m  {valve_range:>11}  {first_str:>10}")

    all_results.append({
        'valve_steps': valve_steps,
        'water_used': water_used,
        'min_dist': min_dist,
        'valve_dists': valve_dists,
    })

# ── Summary ───────────────────────────────────────────────────────
print()
total_valve = sum(r['valve_steps'] for r in all_results)
all_valve_dists = [d for r in all_results for d in r['valve_dists']]
passes_with_valve = sum(1 for r in all_results if r['valve_steps'] > 0)

print(f"=== SUMMARY ({N_PASSES} passes) ===")
print(f"  Passes with valve open: {passes_with_valve}/{N_PASSES}")
print(f"  Total valve steps:      {total_valve}")
if all_valve_dists:
    print(f"  Valve open dist range:  {min(all_valve_dists):.1f} - {max(all_valve_dists):.1f} m")
    print(f"  Avg valve dist:         {np.mean(all_valve_dists):.1f} m")
    print(f"  Median valve dist:      {np.median(all_valve_dists):.1f} m")
else:
    print(f"  VALVE NEVER OPENED — check logic!")

# ── Test 2: Scout offset from fire ────────────────────────────────
print("\n\n=== TEST 2: Scout 30m from fire, fire in camera corner ===\n")
SCOUT_OFF_X, SCOUT_OFF_Y = 30.0, 20.0
scout_drone2 = FakeDrone([FIRE_X + SCOUT_OFF_X, FIRE_Y + SCOUT_OFF_Y, QUAD_ALT])
fake_env2 = FakeEnv(scout_drone2)

fov_half2 = max(10.0, QUAD_ALT * 1.5) / 2.0
dyn_x_msg = -SCOUT_OFF_X / fov_half2
dyn_y_msg = -SCOUT_OFF_Y / fov_half2
est_x2 = (FIRE_X + SCOUT_OFF_X) + dyn_x_msg * fov_half2
est_y2 = (FIRE_Y + SCOUT_OFF_Y) + dyn_y_msg * fov_half2
print(f"Scout at ({FIRE_X+SCOUT_OFF_X}, {FIRE_Y+SCOUT_OFF_Y}, {QUAD_ALT}m)")
print(f"dyn_x={dyn_x_msg:.3f}, dyn_y={dyn_y_msg:.3f}")
print(f"Estimated fire: ({est_x2:.1f}, {est_y2:.1f}) vs actual ({FIRE_X}, {FIRE_Y})")
print(f"Estimation error: {np.hypot(est_x2-FIRE_X, est_y2-FIRE_Y):.1f}m\n")

scout_msg2 = torch.FloatTensor([[
    (FIRE_X + SCOUT_OFF_X) / NORM_DIST,
    (FIRE_Y + SCOUT_OFF_Y) / NORM_DIST,
    0.3, dyn_x_msg, dyn_y_msg,
]])
ctrl2 = CommanderController()
ctrl2.reset(map_half=MAP_BOUNDS)
ctrl2.last_scout_msgs = scout_msg2
ctrl2.last_scout_mask = scout_mask

print(f"{'Pass':>4}  {'Angle':>5}  {'Valve Steps':>11}  "
      f"{'Min Dist':>8}  {'Valve Range':>11}  {'First Open':>10}")
print("-" * 66)

for pass_idx in range(N_PASSES):
    angle = pass_idx * (2 * np.pi / N_PASSES)
    start_x = FIRE_X + 200.0 * np.cos(angle)
    start_y = FIRE_Y + 200.0 * np.sin(angle)
    heading = np.arctan2(FIRE_Y - start_y, FIRE_X - start_x)

    fw = FakeDrone([start_x, start_y, FW_ALT], water=200.0)
    fw.heading = heading
    ctrl2.target_x = FIRE_X
    ctrl2.target_y = FIRE_Y
    ctrl2.steps_in_segment = 0
    ctrl2.wp_reached = False

    valve_steps = 0
    min_dist = float('inf')
    valve_dists = []
    first_open_dist = None
    step_dt = 0.05

    for step in range(500):
        pos = fw.get_position()
        dist = np.hypot(pos[0] - FIRE_X, pos[1] - FIRE_Y)
        min_dist = min(min_dist, dist)
        ctrl2.steps_in_segment = 0
        ctrl2.wp_reached = False

        action = ctrl2.heading_action(fw, env=fake_env2)
        if action[2] > 0:
            valve_steps += 1
            valve_dists.append(dist)
            if first_open_dist is None:
                first_open_dist = dist
            fw.current_water = max(0, fw.current_water - 100.0 * step_dt)

        fw.pos[0] += FW_SPEED * np.cos(heading) * step_dt
        fw.pos[1] += FW_SPEED * np.sin(heading) * step_dt
        if dist > 100.0 and step > 100:
            break

    vr = f"{min(valve_dists):.0f}-{max(valve_dists):.0f}m" if valve_dists else "n/a"
    fo = f"{first_open_dist:.0f}m" if first_open_dist else "n/a"
    print(f"{pass_idx+1:4d}  {np.degrees(angle):5.0f}°  {valve_steps:11d}  "
          f"{min_dist:6.1f} m  {vr:>11}  {fo:>10}")
