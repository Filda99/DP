#!/usr/bin/env python3
"""
Demo 5: Drone Control Verification (Wind Bypass Test)
Tests if the drone stays still in 10 m/s wind (should pass) 
and drifts in 20 m/s wind (should fail).
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path (three levels up from this file)
# demos/quadcopter/wind_test.py -> project root at ../../..
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.simulation import Simulation

def plot_flight_analysis(name, times, positions, output_file):
    fig = plt.figure(figsize=DemoConfig.QUAD_WIND_TEST_FIGURE_SIZE)
    
    # Position vs Time
    plt.plot(times, positions[:, 0], 'r-', label='Pos X', linewidth=2)
    plt.plot(times, positions[:, 1], 'g-', label='Pos Y', linewidth=2)
    plt.plot(times, positions[:, 2], 'b-', label='Pos Z')
    
    # Mark wind change
    plt.axvline(x=5.0, color='k', linestyle='--', label='Wind Increase (10 -> 20 m/s)')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.grid(True, alpha=0.4)
    plt.legend()

    plt.savefig(output_file, dpi=100)
    plt.close()
    print(f"✅ Saved graph: {output_file}")

def run_wind_test():
    print("=" * 70)
    print("🌪️ DEMO 5: WIND RESISTANCE TEST")
    print("=" * 70)
    
    sim = Simulation()
    sim.start_simulation()
    
    # --- PHASE 1: SAFE WIND (10 m/s) ---
    print("\nTesting Safe Wind (10 m/s)...")
    sim.set_wind(DemoConfig.QUAD_WIND_INITIAL) 
    sim.add_quadcopter("Quad_Test", position=[0, 0, DemoConfig.QUAD_SPAWN_HEIGHT])
    
    log_times = []
    log_pos = []
    
    # Run for 5 seconds with 10 m/s wind (Expect STABILITY)
    for step in range(300): # 5s at 60Hz
        sim.step_simulation({"Quad_Test": [0,0,0,0]}) # Hover command
        if step % 10 == 0:
            log_times.append(sim.simulation_time)
            log_pos.append(sim.drones["Quad_Test"].get_position())

    # --- PHASE 2: EXTREME WIND (20 m/s) ---
    print("\nIncreasing Wind to 20 m/s (Extreme)...")
    sim.set_wind([20.0, 0.0, 0.0])
    
    # Run for 5 more seconds (Expect DRIFT)
    for step in range(300):
        sim.step_simulation({"Quad_Test": [0,0,0,0]})
        if step % 10 == 0:
            log_times.append(sim.simulation_time)
            log_pos.append(sim.drones["Quad_Test"].get_position())

    # Plot
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_flight_analysis("Quad_Wind_Test", np.array(log_times), np.array(log_pos), f"{output_dir}/quad_wind_test.pdf")
    
    sim.stop_simulation()

if __name__ == "__main__":
    run_wind_test()