"""
Fixed-Wing Demo Script

Tests the kinematic guidance model by performing a sequence of maneuvers:
1. Climb
2. Coordinated Turn
3. Level Flight
4. Water Drop
"""

import numpy as np
import sys
import os
import glob
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache

def run_demo():
    # 1. Setup Simulation
    sim = Simulation()
    sim.start_simulation()
    
    # Add Fixed Wing Drone
    # Start at 50m height to ensure safe maneuvering
    drone_name = "FirePlane"
    sim.add_fixedwing(drone_name, position=[0, 0, 50], mass=2.0, water_capacity=2000.0)
    
    # 2. Dynamic Time Calculation
    # GET THE ACTUAL TIMESTEP FROM SIMULATION
    dt = sim.timestep 
    
    duration_sec = 60
    total_steps = int(duration_sec / dt) # Calculate steps based on actual dt

    sim.set_wind([0.0, 0.0, 0.0]) # No wind for this demo
    
    print(f"🚀 Starting Fixed-Wing Demo ({duration_sec}s, dt={dt:.4f}s)...")

    # 3. Run Control Loop
    for step in range(total_steps):
        time = step * dt
        
        # --- Flight Plan (Scenario) ---
        
        # Default controls: [Roll, Throttle, Pitch, Water]
        roll_cmd = 0.0
        throttle_cmd = 0.5     # Cruising speed
        pitch_cmd = 0.0        # Maintain altitude
        water_cmd = 0.0
        
        # Fly up
        if time < 5.0:
            pitch_cmd = 0.9 
            throttle_cmd = 1.0 # Full power

        # Stall
        elif time < 20.0:
            throttle_cmd = 0     # Cruising speed
            pitch_cmd = 0.0        # Maintain altitude

        elif time < 35.0:
            pitch_cmd = 0.45
            throttle_cmd = 0.3

        elif time < 45.0:
            # Phase 2: Right Turn (5-15s)
            # Roll +0.5 (approx 22 degrees bank)
            roll_cmd = 0.8
        
        # Step Simulation
        sim.step_simulation({drone_name: [roll_cmd, throttle_cmd, pitch_cmd, water_cmd]})

        if step % 60 == 0:
            if drone_name in sim.drones:
                print(f"Time: {time:.1f}s | Pos: {sim.drones[drone_name].get_position()}")

    sim.stop_simulation()
    
    # 3. Extract Data for Plotting
    log_data = sim.simulation_log['drones'][drone_name]
    positions = np.array(log_data['positions'])
    velocities = np.array(log_data['velocities'])
    inputs_log = np.array(log_data['control_inputs'])
    water_log = np.array(log_data['water_levels'])
    times = np.array(sim.simulation_log['times'])
    
    # Ensure arrays match length (simulation logging might capture one extra/less depending on step order)
    min_len = min(len(positions), len(times), len(inputs_log))
    positions = positions[:min_len]
    inputs_log = inputs_log[:min_len]
    times = times[:min_len]

    # 4. Generate Plots
    plot_results(positions, inputs_log, water_log, times)

def plot_results(positions, inputs, water_levels, times):
    """Generates publication-quality plots using SciencePlots."""
    
    # Attempt to use scienceplots style
    try:
        import scienceplots
        # 'no-latex' prevents errors if LaTeX is not installed on your system
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        print("⚠️ 'scienceplots' library not found. Using default style.")
        print("   Run: pip install scienceplots")
        plt.style.use('default')

    # creating a dictionary
    font = {'size': 10}

    # using rc function
    plt.rc('font', **font)
    # Create Figure
    # Note: SciencePlots usually prefers smaller figures (3-4 inches). 
    # Since we have a complex 3-row layout, we keep it large but may need to adjust font sizes.
    fig = plt.figure(figsize=(10, 12))
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # --- ROW 1, COL 1: Side View (X vs Z) ---
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(x, z, label='Path')
    ax1.scatter(x[0], z[0], c='g', label='Start', s=20, zorder=5)
    ax1.scatter(x[-1], z[-1], c='r', label='End', s=20, zorder=5)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Altitude Z (m)')
    ax1.set_title('Side View (Elevation)')
    ax1.legend()
    ax1.grid(True, linestyle=':')

    # --- ROW 1, COL 2: Top-Down View (X vs Y) ---
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(x, y, label='Track')
    ax2.scatter(x[0], y[0], c='g', label='Start', s=20, zorder=5)
    ax2.scatter(x[-1], y[-1], c='r', label='End', s=20, zorder=5)
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Top-Down View')
    ax2.axis('equal') 
    ax2.grid(True, linestyle=':')
    ax2.legend()

    # --- ROW 2: Position vs Time (UPDATED) ---
    ax3 = fig.add_subplot(3, 1, 2)
    
    # Left Axis: Horizontal (X, Y)
    ln1 = ax3.plot(times, positions[:, 0], label='X (North)', color='C0', linestyle='-')
    ln2 = ax3.plot(times, positions[:, 1], label='Y (East)', color='C1', linestyle='-')
    ax3.set_ylabel('Horizontal Pos (m)')
    ax3.grid(True, linestyle=':')
    
    # Right Axis: Vertical (Z)
    ax3_right = ax3.twinx()
    ln3 = ax3_right.plot(times, positions[:, 2], label='Z (Altitude)', color='C2', linestyle='-')
    ax3_right.set_ylabel('Altitude Z (m)')
    ax3_right.tick_params(axis='y') # Color the tick labels to match the line
    
    # Combine Legends
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='upper left') # or 'best'
    
    ax3.set_title('Position History')

    # --- ROW 3: Control Inputs ---
    ax4 = fig.add_subplot(3, 1, 3)
    ln1 = ax4.plot(times, inputs[:, 0], label='Roll', color='C0', linestyle='-')
    ln2 = ax4.plot(times, inputs[:, 1], label='Throttle', color='C1', linestyle='-')
    ln3 = ax4.plot(times, inputs[:, 2], label='Pitch', color='C2', linestyle='-')
    ax4.set_ylabel('Input'); ax4.set_ylim(0.1, 1.1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylim([-1.1, 1.1])
    ax4.set_title('Commands History')
    
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', ncol=4)
    ax4.grid(True, linestyle=':')

    plt.tight_layout()
    plt.savefig('output/demo_fixedwing.pdf') # PDF is better for papers
    print("✅ Scientific plot saved to output/demo_fixedwing.pdf")


if __name__ == "__main__":
    run_demo()