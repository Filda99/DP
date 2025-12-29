"""
Quadcopter Separated Maneuver Tests

Runs isolated tests for each flight dynamic to clearly visualize behavior:
1. Hover Stability
2. Forward Flight (Pitch)
3. Lateral Flight (Strafe/Roll)
4. Vertical Flight (Climb)
5. Heading Change (Yaw)
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

def run_all_tests():
    # 1. Hover Test
    run_maneuver_test(
        test_name="1_Hover_Stability",
        duration=5.0,
        cmd_func=lambda t: [0.0, 0.0, 0.0, 0.0], # [Roll, Pitch, Yaw, Vert]
        description="Drone should maintain steady position [0,0,2]."
    )

    # 2. Forward Flight (Pitch)
    run_maneuver_test(
        test_name="2_Forward_Flight",
        duration=5.0,
        cmd_func=lambda t: [0.0, 0.5, 0.0, 0.0], # Pitch forward +0.5
        description="Drone should accelerate in +X direction."
    )

    # 3. Lateral Strafe (Roll)
    run_maneuver_test(
        test_name="3_Lateral_Strafe",
        duration=5.0,
        cmd_func=lambda t: [0.5, 0.0, 0.0, 0.0], # Roll right +0.5
        description="Drone should move in -Y direction (Right)."
    )

    # 4. Vertical Climb
    run_maneuver_test(
        test_name="4_Vertical_Climb",
        duration=5.0,
        cmd_func=lambda t: [0.0, 0.0, 0.0, 1.0], # Vertical Vel +1.0 m/s
        description="Drone should climb at constant speed."
    )

    # 5. Yaw Turn (Special plotting needed)
    run_maneuver_test(
        test_name="5_Yaw_Turn",
        duration=5.0,
        cmd_func=lambda t: [0.0, 0.0, 0.1, 0.0], # Yaw Rate +1.0 rad/s
        description="Drone should rotate 90+ degrees.",
        plot_type="yaw" 
    )

def run_maneuver_test(test_name, duration, cmd_func, description, plot_type="standard"):
    print(f"\n🧪 STARTING TEST: {test_name}")
    print(f"   ℹ️  {description}")
    
    # 1. Setup
    sim = Simulation()
    sim.start_simulation()
    sim.set_wind([0, 0, 0])
    
    drone_name = "QuadTest"
    sim.add_quadcopter(drone_name, position=[0, 0, 2]) # Start at 2m
    
    dt = sim.timestep
    steps = int(duration / dt)
    
    # Data Storage
    history = {
        'time': [],
        'pos': [],
        'rpy': [], # Roll Pitch Yaw
        'cmd': []
    }

    # 2. Loop
    for step in range(steps):
        t = step * dt
        
        # Get Command from the lambda function passed in
        cmd = cmd_func(t)
        
        # Apply
        sim.step_simulation({drone_name: cmd})
        
        # Log
        if drone_name in sim.drones:
            drone = sim.drones[drone_name]
            history['time'].append(t)
            history['pos'].append(drone.get_position())
            history['rpy'].append(drone.get_orientation_rpy())
            history['cmd'].append(cmd)

    sim.stop_simulation()
    
    # 3. Plot
    print(f"   📊 Generating graph: output_{test_name}.png")
    
    data_arrays = {
        'time': np.array(history['time']),
        'pos': np.array(history['pos']),
        'rpy': np.array(history['rpy']),
        'cmd': np.array(history['cmd'])
    }
    
    if plot_type == "yaw":
        plot_yaw_results(test_name, data_arrays)
    else:
        plot_standard_results(test_name, data_arrays)

def plot_standard_results(name, data):
    """Plots Position and Inputs for movement tests."""
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(10, 10))
    times = data['time']
    pos = data['pos']
    cmd = data['cmd']

    # 1. Top Down View (Trajectory)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(pos[:, 0], pos[:, 1], label='Trajectory', linewidth=2)
    ax1.scatter(pos[0, 0], pos[0, 1], c='g', label='Start')
    ax1.scatter(pos[-1, 0], pos[-1, 1], c='r', label='End')
    ax1.set_title(f'{name}: Top Down (XY)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # 2. Altitude vs Time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, pos[:, 2], color='purple', label='Altitude')
    ax2.set_title('Altitude (Z) vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z (m)')
    ax2.grid(True, linestyle=':')

    # 3. Position Components vs Time
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(times, pos[:, 0], label='X Pos', linestyle='--')
    ax3.plot(times, pos[:, 1], label='Y Pos', linestyle='-.')
    ax3.plot(times, pos[:, 2], label='Z Pos')
    ax3.set_title('Position History')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.grid(True, linestyle=':')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f"output_{name}.png", dpi=300)
    plt.close()

def plot_yaw_results(name, data):
    """Special plotter for Heading/Yaw tests (Orientation is more important than Position)."""
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(10, 8))
    times = data['time']
    rpy = data['rpy'] # [Roll, Pitch, Yaw]
    cmd = data['cmd'] # [Roll, Pitch, YawRate, Vert]

    # Convert Yaw to degrees for easier reading
    yaw_deg = np.degrees(rpy[:, 2])

    # 1. Heading vs Time
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(times, yaw_deg, color='blue', linewidth=2, label='Heading (Yaw)')
    ax1.set_title(f'{name}: Heading Change')
    ax1.set_ylabel('Yaw Angle (Degrees)')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # 2. Input vs Response
    ax2 = fig.add_subplot(2, 1, 2)
    # Scale command for visibility (cmd is rate, plot is angle, so just show pattern)
    ax2.plot(times, cmd[:, 2], color='red', linestyle='--', label='Input: Yaw Rate (rad/s)')
    ax2.set_title('Control Input')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Input Value')
    ax2.grid(True, linestyle=':')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"output_{name}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_all_tests()