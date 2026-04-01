"""
diagnose_flight.py — Run fixed-wing with constant actions and log what happens.
No neural network, no training. Pure physics + env analysis.

Usage:
    cd src/
    python diagnose_flight.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from env_core import DroneFireEnv

def run_episode(env, action, label, max_steps=500):
    """Run one episode with a constant action vector."""
    obs, _ = env.reset(epizode_number=42)

    # Kill fire
    if env.sim.environment.fire_grid is not None:
        env.sim.environment.fire_grid.B[:] = False
        env.sim.environment.fire_grid.I[:] = 0.0

    f = env.fixed_agents[0]
    drone = env.sim.drones[f]
    
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Action: {action}")
    start_pos = drone.get_position().copy()
    print(f"Spawn: pos=({start_pos[0]:.0f}, {start_pos[1]:.0f}, {start_pos[2]:.0f})")
    print(f"       heading={np.degrees(drone.state_chi):.0f}°, Va={drone.state_va:.1f} m/s")
    print(f"{'='*60}")

    dt_per_step = env.sim.timestep * 5  # frame_skip=5
    print(f"dt per RL step: {dt_per_step:.4f}s")
    print(f"Distance per step at 15 m/s: {15*dt_per_step:.2f}m")
    print(f"Map bounds: ±{env.map_bounds:.0f}m")
    print()
    print(f"{'Step':>5} {'Time':>6} {'X':>7} {'Y':>7} {'Z':>6} {'Va':>5} {'Chi°':>6} {'Gamma°':>7} {'Phi°':>6} {'Reward':>7} {'Status'}")
    print("-" * 85)

    total_reward = 0
    for step in range(max_steps):
        if f not in env.agents:
            print(f"  DEAD at step {step}")
            break

        pos = env.sim.drones[f].get_position()
        d = env.sim.drones[f]
        t = step * dt_per_step

        actions = {f: np.array(action, dtype=np.float32)}
        obs, rewards, terms, truncs, infos = env.step(actions)
        r = rewards.get(f, 0.0)
        total_reward += r

        status = ""
        if terms.get(f, False):
            status = f"DEAD: {infos.get(f, {}).get('death_cause', '?')}"

        if step % 50 == 0 or terms.get(f, False) or step < 5:
            print(f"{step:5d} {t:6.1f}s {pos[0]:7.0f} {pos[1]:7.0f} {pos[2]:6.1f} "
                  f"{d.state_va:5.1f} {np.degrees(d.state_chi):6.1f} "
                  f"{np.degrees(d.state_gamma):7.2f} {np.degrees(d.state_phi):6.2f} "
                  f"{r:7.2f} {status}")

    print(f"\nTotal reward: {total_reward:.1f}, survived {step} / {max_steps} steps")
    dist_from_start = np.linalg.norm(drone.get_position()[:2] - start_pos[:2]) if f in env.sim.drones else float('nan')
    print(f"Distance traveled: {dist_from_start:.0f}m (from spawn)")

env = DroneFireEnv(num_quads=0, num_fixed=1, grid_size_m=2000.0, max_steps=500)

# Test 1: All zeros — level flight
run_episode(env, [0.0, 0.0, 0.0, 0.0], "All zeros (level flight)")

# Test 2: Slight left turn
run_episode(env, [0.3, 0.0, 0.0, 0.0], "Gentle left turn (roll=0.3)")

# Test 3: Orbiting — continuous turn
run_episode(env, [0.5, 0.0, 0.0, 0.0], "Continuous turn (roll=0.5)")

# Test 4: What a trained network might try — slight pitch down
run_episode(env, [0.0, -0.1, 0.0, 0.0], "Slight nose down (pitch=-0.1)")

# Test 5: Full throttle straight
run_episode(env, [0.0, 0.0, 1.0, 0.0], "Full throttle straight")

if hasattr(env, 'sim') and env.sim is not None:
    env.sim.stop_simulation()

print("\n\n=== SUMMARY ===")
print(f"fps={env.sim.fps if hasattr(env, 'sim') else '?'}, dt={1/30:.4f}s, frame_skip=5")
print(f"dt per RL step = {5/30:.4f}s")
print(f"Map: ±{env.map_bounds:.0f}m")
print(f"At 15 m/s, straight flight reaches boundary in:")
print(f"  From center:   {env.map_bounds / 15 / (5/30):.0f} steps = {env.map_bounds / 15:.0f}s")
print(f"  From 400m out: {(env.map_bounds - 400) / 15 / (5/30):.0f} steps = {(env.map_bounds-400) / 15:.0f}s")
print(f"\nThe ONLY way to survive 500 steps is to TURN.")
print(f"Orbiting at roll=0.5 (22.5°), radius ≈ Va²/(g·tan(φ)) = {15**2/(9.81*np.tan(np.radians(22.5))):.0f}m")
