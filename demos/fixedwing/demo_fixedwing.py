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
import matplotlib
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pybullet as p

# Cesta k projektu
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

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
            pitch_cmd = 1.0 
            throttle_cmd = 1.0 # Full power

        # Stall
        elif time < 20.0:
            throttle_cmd = 0     # Cruising speed
            pitch_cmd = 0.0        # Maintain altitude

        elif time < 35.0:
            pitch_cmd = 0.5
            throttle_cmd = 0.5

        elif time < 45.0:
            # Phase 2: Right Turn (5-15s)
            # Roll +0.5 (approx 22 degrees bank)
            roll_cmd = 0.5
        
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
    ax2.scatter(x[0], y[0], c='g', s=20, zorder=5)
    ax2.scatter(x[-1], y[-1], c='r', s=20, zorder=5)
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Top-Down View')
    ax2.axis('equal') 
    ax2.grid(True, linestyle=':')
    ax2.legend()

    # --- ROW 2: Position vs Time ---
    ax3 = fig.add_subplot(3, 1, 2)
    ax3.plot(times, positions[:, 0], label='X (North)', linestyle='--')
    ax3.plot(times, positions[:, 1], label='Y (East)', linestyle='-.')
    ax3.plot(times, positions[:, 2], label='Z (Altitude)')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position History')
    ax3.legend()
    ax3.grid(True, linestyle=':')

    # --- ROW 3: Control Inputs ---
    ax4 = fig.add_subplot(3, 1, 3)
    ln1 = ax4.plot(times, inputs[:, 0], label='Roll', color='C0')
    ln2 = ax4.plot(times, inputs[:, 1], label='Throttle', color='C1', linestyle='--')
    ln3 = ax4.plot(times, inputs[:, 2], label='Pitch', color='C2', linestyle='-.')
    ax4.set_ylabel('Input'); ax4.set_ylim(0.1, 1.1)
    
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', ncol=4)

    plt.tight_layout()
    plt.savefig('demo_science_paper.pdf') # PDF is better for papers
    print("✅ Scientific plot saved to demo_science_paper.png and .pdf")

# def plot_results(positions, inputs, times):
#     """Generates 2D Projections, Position-Time, and Control Input plots."""
    
#     # Increase height to accommodate 3 rows
#     fig = plt.figure(figsize=(14, 15)) 
#     # fig.suptitle('Fixed-Wing Kinematic Guidance Test', fontsize=16)

#     x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

#     # --- ROW 1, COL 1: Side View (X vs Z) ---
#     ax1 = fig.add_subplot(3, 2, 1)
#     ax1.plot(x, z, label='Flight Path', linewidth=2, color='blue')
#     ax1.scatter(x[0], z[0], color='green', label='Start', s=50, zorder=5)
#     ax1.scatter(x[-1], z[-1], color='red', label='End', s=50, zorder=5)
    
#     ax1.set_xlabel('X Position (m)')
#     ax1.set_ylabel('Z Altitude (m)')
#     ax1.set_title('Side View (X-Z Projection)')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()

#     # --- ROW 1, COL 2: Top-Down View (X vs Y) ---
#     ax2 = fig.add_subplot(3, 2, 2)
#     ax2.plot(x, y, label='Ground Track', linewidth=2, color='blue')
#     ax2.scatter(x[0], y[0], color='green', label='Start', s=50, zorder=5)
#     ax2.scatter(x[-1], y[-1], color='red', label='End', s=50, zorder=5)
    
#     ax2.set_xlabel('X Position (m)')
#     ax2.set_ylabel('Y Position (m)')
#     ax2.set_title('Top-Down View (X-Y Projection)')
#     ax2.grid(True, alpha=0.3)
#     ax2.axis('equal') # Important for correct turn geometry visualization
#     ax2.legend()

#     # --- ROW 2: Position vs Time (X, Y, Z) ---
#     ax3 = fig.add_subplot(3, 1, 2)
#     ax3.plot(times, positions[:, 0], label='X (North)', linestyle='--')
#     ax3.plot(times, positions[:, 1], label='Y (East)', linestyle='-.')
#     ax3.plot(times, positions[:, 2], label='Z (Altitude)', linewidth=2)
    
#     ax3.set_xlabel('Time (s)')
#     ax3.set_ylabel('Position (m)')
#     ax3.set_title('Position vs Time')
#     ax3.grid(True, alpha=0.3)
#     ax3.legend()

#     # --- ROW 3: Control Inputs vs Time ---
#     ax4 = fig.add_subplot(3, 1, 3)
    
#     # Inputs: [Roll, Throttle, Pitch, Water]
#     ax4.plot(times, inputs[:, 0], label='Roll Cmd (-1..1)', color='purple')
#     ax4.plot(times, inputs[:, 1], label='Throttle Cmd (0..1)', color='orange')
#     ax4.plot(times, inputs[:, 2], label='Pitch/Climb Cmd (-1..1)', color='green')
    
#     # Plot Water Trigger as a filled area
#     ax4.fill_between(times, 0, inputs[:, 3], color='blue', alpha=0.2, label='Water Trigger')

#     ax4.set_xlabel('Time (s)')
#     ax4.set_ylabel('Input Value')
#     ax4.set_title('Control Inputs')
#     ax4.grid(True, alpha=0.3)
#     ax4.legend(loc='upper right')

#     plt.tight_layout()
    
#     # Save the file
#     plt.savefig('Fixed-Wing_Kinematic_Guidance_Test.pdf', dpi=100, bbox_inches='tight')
#     print("✅ Plot saved to demo_results.png")

if __name__ == "__main__":
    run_demo()