#!/usr/bin/env python3
"""
Demo 6: Simple Hover Test (Unit Test)
Verifies if the drone can fight gravity and maintain altitude.
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

def run_hover_test():
    print("=" * 70)
    print("🚁 DEMO 6: SIMPLE HOVER TEST (10s)")
    print("=" * 70)
    
    sim = Simulation()
    sim.start_simulation()
    
    # 1. Add Quadcopter at 5 meters
    # Note: We use standard mass 0.5kg. Gravity force should be approx 4.905 N
    sim.add_quadcopter("Hover_Test", position=[0, 0, 5], mass=0.5)
    
    # Data logging
    times = []
    z_positions = []
    z_velocities = []
    z_inputs = []
    
    duration = 10.0
    steps = int(duration / sim.timestep)
    
    print(f"Starting hover test... (Target Height: 5.0m)")
    
    for step in range(steps):
        # Input [Roll, Pitch, Yaw, Vertical]
        # 0.0 means "Hover" / "Maintain current state"
        controls = {
            "Hover_Test": [0.0, 0.0, 0.0, 0.0]
        }
        
        sim.step_simulation(controls)
        
        # Log data
        t = sim.simulation_time
        if "Hover_Test" in sim.drones:
            drone = sim.drones["Hover_Test"]
            pos = drone.get_position()
            vel = drone.get_velocity()
            
            times.append(t)
            z_positions.append(pos[2])
            z_velocities.append(vel[2])
            z_inputs.append(0.0) # We are sending 0 input
        else:
            print(f"❌ CRASH detected at t={t:.2f}s!")
            break
            
        if step % 60 == 0:
            print(f"  t={t:.1f}s | Alt: {pos[2]:.2f}m | Vel Z: {vel[2]:.3f} m/s")

    # --- PLOTTING ---
    print("\n📊 Generating graph...")
    os.makedirs('output', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Altitude
    ax1.plot(times, z_positions, 'b-', linewidth=2, label='Altitude')
    ax1.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='Target (5m)')
    ax1.set_title('Vertical Performance - Altitude')
    ax1.set_ylabel('Height (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Velocity
    ax2.plot(times, z_velocities, 'g-', linewidth=2, label='Vertical Velocity')
    ax2.axhline(y=0.0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Vertical Performance - Velocity')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/demo_06_hover.png')
    print(f"✅ Saved graph to output/demo_06_hover.png")
    
    sim.stop_simulation()

if __name__ == "__main__":
    run_hover_test()