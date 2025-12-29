"""
Quadcopter Forward Flight Test (Velocity Step Response)

Objective: Verify that 'Pitch' input correctly translates to 'Forward Velocity'.
Scenario:
1. Hover (0-2s)
2. Command Forward 50% (Target 7.5 m/s) (2-6s)
3. Command Stop (0%) (6-9s)
4. Command Backward 50% (Target -7.5 m/s) (9-13s)
5. Stop (13-15s)
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

def run_forward_test():
    # 1. Setup
    sim = Simulation()
    sim.start_simulation()
    sim.set_wind([0, 0, 0]) # No wind to isolate dynamics
    
    drone_name = "QuadForward"
    sim.add_quadcopter(drone_name, position=[0, 0, 2])
    
    dt = sim.timestep
    duration = 15.0
    steps = int(duration / dt)
    
    print(f"🚀 Running Forward Velocity Test ({duration}s)...")
    
    # Data Logs
    times = []
    cmd_vel_x = []
    actual_vel_x = []
    positions_x = []
    pitch_angles = []

    # 2. Control Loop
    for step in range(steps):
        t = step * dt
        
        # --- INPUT SEQUENCE ---
        pitch_input = 0.0
        phase = "Hover"
        
        if 2.0 <= t < 6.0:
            phase = "Forward 50%"
            pitch_input = 0.5  # Should result in 0.5 * 15.0 = 7.5 m/s
            
        elif 6.0 <= t < 9.0:
            phase = "Brake / Stop"
            pitch_input = 0.0
            
        elif 9.0 <= t < 13.0:
            phase = "Backward 50%"
            pitch_input = -0.5 # Should result in -7.5 m/s
            
        # Apply Control
        sim.step_simulation({drone_name: [0.0, pitch_input, 0.0, 0.0]})
        
        # --- LOGGING ---
        if drone_name in sim.drones:
            drone = sim.drones[drone_name]
            
            # 1. Actual State
            vel = drone.get_velocity() # [vx, vy, vz]
            pos = drone.get_position()
            rpy = drone.get_orientation_rpy()
            
            # 2. Target Velocity (for comparison graph)
            # Based on Quadcopter config: Max speed 15.0 m/s
            target_v = pitch_input * 15.0 
            
            times.append(t)
            cmd_vel_x.append(target_v)
            actual_vel_x.append(vel[0]) # X-velocity (Forward)
            positions_x.append(pos[0])
            pitch_angles.append(np.degrees(rpy[1])) # Pitch in degrees

        if step % 30 == 0:
             # Print live debug info
             print(f"T={t:4.1f} | Cmd={pitch_input*15.0:4.1f} m/s | Act={vel[0]:4.1f} m/s | PosX={pos[0]:5.1f}")

    sim.stop_simulation()
    
    # 3. Plotting
    plot_forward_dynamics(times, cmd_vel_x, actual_vel_x, positions_x, pitch_angles)

def plot_forward_dynamics(time, cmd_vel, act_vel, pos_x, pitch):
    """Generates detailed response graphs."""
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(10, 10))
    
    # Graph 1: Velocity Response (The most important one)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time, cmd_vel, 'r--', label='Target Velocity (Cmd)', linewidth=1.5)
    ax1.plot(time, act_vel, 'b-', label='Actual Velocity', linewidth=2)
    ax1.set_title('Velocity Controller Step Response')
    ax1.set_ylabel('Velocity X (m/s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':')
    
    # Graph 2: Position (Did it return?)
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(time, pos_x, 'g-', linewidth=2)
    ax2.set_title('Position Displacement (X-Axis)')
    ax2.set_ylabel('Position X (m)')
    ax2.grid(True, linestyle=':')
    
    # Graph 3: Pitch Angle (Physics Check)
    # Since we use "Magic Force" (Holonomic Model), pitch might remain near 0.
    # If we simulated rotor thrust, this would dip negative.
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(time, pitch, 'k-', linewidth=1.5)
    ax3.set_title('Drone Pitch Angle')
    ax3.set_ylabel('Pitch (Degrees)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylim(-10, 10) # Zoom in around 0
    ax3.grid(True, linestyle=':')
    
    plt.tight_layout()
    plt.savefig('output_forward_flight.png', dpi=300)
    print("\n✅ Saved analysis to output_forward_flight.png")

if __name__ == "__main__":
    run_forward_test()