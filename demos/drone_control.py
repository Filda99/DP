#!/usr/bin/env python3
"""
Demo 5: Drone Control Verification (Final Models)
Verifies the stability of Joystick Control (Quad) and Autopilot (Fixed-Wing).
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

def plot_flight_analysis(name, times, positions, inputs, input_labels, output_file):
    fig = plt.figure(figsize=(12, 15))
    gs = GridSpec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.3)
    
    # 1. 3D Trajectory
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax3d.plot(positions[:, 0], positions[:, 1], np.zeros_like(positions[:, 2]), 'k--', alpha=0.2)
    ax3d.set_title(f'{name} - 3D Trajectory', fontweight='bold')
    ax3d.legend()
    
    # 2. Position
    ax_pos = fig.add_subplot(gs[1])
    ax_pos.plot(times, positions[:, 0], 'r-', label='Pos X')
    ax_pos.plot(times, positions[:, 1], 'g-', label='Pos Y')
    ax_pos.plot(times, positions[:, 2], 'b-', label='Pos Z (Altitude)')
    ax_pos.set_title('Position vs Time', fontweight='bold')
    ax_pos.grid(True, alpha=0.4)
    ax_pos.legend()

    # 3. Inputs
    ax_inp = fig.add_subplot(gs[2], sharex=ax_pos)
    colors = ['orange', 'purple', 'cyan', 'brown']
    for i, label in enumerate(input_labels):
        if i < inputs.shape[1]:
            ax_inp.plot(times, inputs[:, i], label=label, color=colors[i % len(colors)], linewidth=2)
    ax_inp.set_title('Control Inputs', fontweight='bold')
    ax_inp.grid(True, alpha=0.4)
    ax_inp.legend()

    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved graph: {output_file}")

def run_drone_control_test():
    print("=" * 70)
    print("🎮 DEMO 5: FINAL PHYSICS VERIFICATION")
    print("=" * 70)
    
    sim = Simulation()
    sim.start_simulation()
    
    sim.add_quadcopter("Quad_Test", position=[0, 0, 5])
    sim.add_fixedwing("Plane_Test", position=[0, 0, 30], max_thrust=30.0)
    
    log_times = []
    log_quad = {'pos': [], 'input': []}
    log_plane = {'pos': [], 'input': []}
    
    total_time = 20.0 
    steps = int(total_time / sim.timestep)
    
    for step in range(steps):
        t = sim.simulation_time
        
        # --- QUADCOPTER (Precise Maneuvers) ---
        q_in = [0.0, 0.0, 0.0, 0.0]
        if 2.0 <= t < 3.0:   q_in = [-0.5, 0.0, 0.0, 0.0] # Left
        # elif 5.0 <= t < 6.0: q_in = [0.0, 0.5, 0.0, 0.0]  # Forward
        # elif 8.0 <= t < 10.0:q_in = [0.0, 0.0, 0.0, 0.5]  # Up
        # # Else: Hover (Input 0)
            
        # --- FIXED WING (Autopilot) ---
        p_in = [0.0, 0.6, 0.0] # Cruise
        if 2.0 <= t < 5.0:   p_in = [-0.5, 0.6, 0.0] # Bank Right
        elif 8.0 <= t < 12.0:p_in = [0.5, 0.6, 0.1]  # Bank Left + Pitch Up
        elif 15.0 <= t:      p_in = [0.0, 0.0, 0.0]  # Engines Off (Glide)

        controls = { "Quad_Test": q_in, "Plane_Test": p_in }
        sim.step_simulation(controls)
        
        if step % 5 == 0:
            log_times.append(t)
            if "Quad_Test" in sim.drones:
                log_quad['pos'].append(sim.drones["Quad_Test"].get_position())
            else: log_quad['pos'].append([np.nan]*3)
            log_quad['input'].append(q_in)
            
            if "Plane_Test" in sim.drones:
                log_plane['pos'].append(sim.drones["Plane_Test"].get_position())
            else: log_plane['pos'].append([np.nan]*3)
            log_plane['input'].append(p_in)
            
        if step % 60 == 0:
            print(f"  Time: {t:.1f}s")

    # Plotting
    output_dir = 'output/demo_05_graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_flight_analysis("Quadcopter", np.array(log_times), np.array(log_quad['pos']), np.array(log_quad['input']),
        ['Roll', 'Pitch', 'Yaw', 'Vertical'], f"{output_dir}/quadcopter_final.png")
    
    plot_flight_analysis("Fixed-Wing", np.array(log_times), np.array(log_plane['pos']), np.array(log_plane['input']),
        ['Bank', 'Throttle', 'Pitch'], f"{output_dir}/fixedwing_final.png")
    
    sim.stop_simulation()

if __name__ == "__main__":
    run_drone_control_test()