"""
Quadcopter Demo Script

Tests the holonomic control model by performing a sequence of maneuvers:
1. Hover
2. Forward Flight
3. Yaw Turn (Heading Change)
4. Lateral Strafe
5. Vertical Climb/Descent
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pybullet as p

# Path setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

def run_demo():
    # 1. Setup Simulation
    sim = Simulation()
    sim.start_simulation()
    
    # Add Quadcopter
    drone_name = "Quad1"
    # Start slightly above ground
    sim.add_quadcopter(drone_name, position=[0, 0, 2])

    sim.set_wind([0, 0, 0])  # No wind for this demo
    
    # 2. Simulation Settings
    dt = sim.timestep 
    duration_sec = 36.0
    total_steps = int(duration_sec / dt)
    
    print(f"🚀 Starting Quadcopter Demo ({duration_sec}s, dt={dt:.4f}s)...")

    # Manually tracking data to ensure exact alignment for plotting
    history_pos = []
    history_cmd = []
    history_time = []

    # 3. Run Control Loop
    for step in range(total_steps):
        time = step * dt
        
        # --- Flight Plan (Scenario) ---
        # Quad Inputs: [Roll, Pitch, YawRate, VerticalVelocity]
        # Assumptions: +Pitch = Forward, +Roll = Right, +Vel = Up
        
        roll_cmd = 0.0
        pitch_cmd = 0.0
        yaw_rate_cmd = 0.0
        vert_vel_cmd = 0.0
        phase = ""

        if 0.0 <= time < 3.0:
            phase = "Hover"
            
        elif 3.0 <= time < 7.0:
            phase = "Forward"
            pitch_cmd = 0.2  # Pitch forward (rad)
            
        elif 7.0 <= time < 10.0:
            phase = "Stop"

        elif 10.0 <= time < 13.0:
            phase = "Yaw Turn"
            yaw_rate_cmd = -1  # rad/s
            
        elif 13.0 <= time < 15.0:
             phase = "Stabilize"

        elif 15.0 <= time < 19.0:
            phase = "Forward (New Heading)"
            pitch_cmd = 0.2
            
        elif 19.0 <= time < 22.0:
            phase = "Stop"
            
        elif 22.0 <= time < 25.0:
            phase = "Strafe Right"
            roll_cmd = 0.2
            
        elif 25.0 <= time < 28.0:
            phase = "Strafe Left"
            roll_cmd = -0.6
            
        elif 28.0 <= time < 31.0:
            phase = "Stop"
            
        elif 31.0 <= time < 33.5:
            phase = "Climb"
            vert_vel_cmd = 1  # m/s
            
        elif 33.5 <= time < 36.0:
            phase = "Descend"
            vert_vel_cmd = -0.5 # m/s

        # Apply Controls
        cmd = [roll_cmd, pitch_cmd, yaw_rate_cmd, vert_vel_cmd]
        sim.step_simulation({drone_name: cmd})

        # Log Data
        if drone_name in sim.drones:
            pos = sim.drones[drone_name].get_position()
            history_pos.append(pos)
            history_cmd.append(cmd)
            history_time.append(time)

        # Print status
        if step % int(1.0/dt) == 0:
            pos = sim.drones[drone_name].get_position()
            print(f"T: {time:4.1f}s | Phase: {phase:<20} | Pos: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

    sim.stop_simulation()
    
    # 4. Prepare Data for Plotting
    log_data = sim.simulation_log['drones'][drone_name]
    # positions = np.array(history_pos)
    positions = np.array(log_data['positions'])
    inputs_log = np.array(history_cmd)
    times = np.array(history_time)
    
    # 5. Generate Plots
    plot_results(positions, inputs_log, times)

def plot_results(positions, inputs, times):
    """Generates publication-quality plots matching the Fixed-Wing style."""
    
    # Attempt to use scienceplots style
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except ImportError:
        print("⚠️ 'scienceplots' library not found. Using default style.")
        plt.style.use('default')

    # Create Figure
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
    # Quad Inputs: [Roll, Pitch, YawRate, VertVel]
    ax4 = fig.add_subplot(3, 1, 3)
    
    ln1 = ax4.plot(times, inputs[:, 0], label='Roll (rad)', color='C0')
    ln2 = ax4.plot(times, inputs[:, 1], label='Pitch (rad)', color='C1', linestyle='--')
    ln3 = ax4.plot(times, inputs[:, 2], label='YawRate (rad/s)', color='C2', linestyle='-.')
    
    # Vertical Velocity might have different scale, allow it to plot normally but label correctly
    ln4 = ax4.plot(times, inputs[:, 3], label='VertVel (m/s)', color='C3', linestyle=':')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Command Input')
    ax4.set_title('Control Inputs')
    
    lines = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', ncol=4, fontsize='small')
    ax4.grid(True, linestyle=':')

    plt.tight_layout()
    
    # Save files
    output_filename = 'demo_quadcopter_science'
    plt.savefig(f'{output_filename}.png', dpi=300)
    plt.savefig(f'{output_filename}.pdf') 
    print(f"✅ Scientific plot saved to {output_filename}.png and .pdf")

if __name__ == "__main__":
    run_demo()