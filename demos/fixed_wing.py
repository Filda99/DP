#!/usr/bin/env python3
"""
Demo 7: Fixed-Wing Physics Verification
Tests stalling and wind drift behavior.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

def run_fixedwing_test():
    print("=" * 70)
    print("✈️ DEMO 7: FIXED-WING PHYSICS")
    print("=" * 70)
    
    sim = Simulation()
    sim.start_simulation()
    
    # Start with NO WIND
    sim.set_wind([0.0, 0.0, 0.0])
    
    # Add plane high enough
    sim.add_fixedwing("Plane_Test", position=[0, 0, 100], max_thrust=40.0)
    
    log_times = []
    log_pos = []
    log_speed = []
    
    print("Phase 1: Cruise (No Wind)")
    for _ in range(180): # 3s
        sim.step_simulation({"Plane_Test": [0.0, 0.7, 0.0]}) # 70% Throttle, Level
        log_times.append(sim.simulation_time)
        log_pos.append(sim.drones["Plane_Test"].get_position())
        log_speed.append(sim.drones["Plane_Test"].get_speed())

    print("Phase 2: Crosswind 10 m/s (Plane should drift Y)")
    sim.set_wind([0.0, 10.0, 0.0]) # Wind from side
    for _ in range(300): # 5s
        sim.step_simulation({"Plane_Test": [0.0, 0.7, 0.0]}) # Still flying straight
        log_times.append(sim.simulation_time)
        log_pos.append(sim.drones["Plane_Test"].get_position())
        log_speed.append(sim.drones["Plane_Test"].get_speed())
        
    print("Phase 3: Engine Cut (Gliding/Stall)")
    sim.set_wind([0.0, 0.0, 0.0])
    for _ in range(300): # 5s
        sim.step_simulation({"Plane_Test": [0.0, 0.0, 0.1]}) # 0% Throttle, Try to pull up
        log_times.append(sim.simulation_time)
        log_pos.append(sim.drones["Plane_Test"].get_position())
        log_speed.append(sim.drones["Plane_Test"].get_speed())

    # Plot
    positions = np.array(log_pos)
    output_dir = 'output/demo_07_graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top Down View (XY)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-')
    ax1.set_title('Top-Down Trajectory (XY)')
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Side)')
    ax1.grid(True)
    ax1.text(positions[180,0], positions[180,1], 'Wind Start', color='red')
    
    # Altitude Profile
    ax2.plot(log_times, positions[:, 2], 'g-')
    ax2.set_title('Altitude Profile (Z)')
    ax2.set_ylabel('Height (m)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    ax2.axvline(x=8.0, color='r', linestyle='--', label='Engine Cut')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fixedwing_physics.png")
    print(f"✅ Saved analysis to {output_dir}/fixedwing_physics.png")
    
    sim.stop_simulation()

if __name__ == "__main__":
    run_fixedwing_test()