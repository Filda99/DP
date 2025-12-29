"""
Quadcopter "Staircase" Test (Frame Transformation Verification)

Scenario:
1. Fly Forward (Heading 0 deg) -> Should move along X-axis
2. Turn Left 45 deg
3. Fly Forward (Heading 45 deg) -> Should move diagonal (X=Y)
4. Turn Left 45 deg (Total 90)
5. Fly Forward (Heading 90 deg) -> Should move along Y-axis

This proves that 'Forward' input is correctly rotated by the current Yaw.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pybullet as p

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

def run_staircase_test():
    # 1. Setup
    sim = Simulation()
    sim.start_simulation()
    sim.set_wind([0, 0, 0])
    
    drone_name = "QuadStair"
    sim.add_quadcopter(drone_name, position=[0, 0, 2])
    
    dt = sim.timestep
    # Longer duration for the sequence
    duration = 25.0 
    steps = int(duration / dt)
    
    history_pos = []
    history_yaw = []
    
    print(f"🚀 Running Staircase Test ({duration}s)...")

    # 2. Control Loop
    for step in range(steps):
        t = step * dt
        
        # Default Inputs
        pitch_cmd = 0.0 # Forward Velocity
        yaw_cmd = 0.0   # Yaw Rate
        phase = ""
        
        # --- SEQUENCE ---
        
        # A. Fly North (0 to 5s)
        if 1.0 < t < 5.0:
            phase = "1. Fly Forward (0 deg)"
            pitch_cmd = 0.5 # Half speed
            
        # B. Turn 45 deg (5s to 7s)
        # We want to turn 45 deg (approx 0.78 rad). 
        # If we yaw at 0.4 rad/s for 2s -> 0.8 rad.
        elif 5.0 <= t < 7.0:
            phase = "2. Turn 45 deg"
            yaw_cmd = 0.4 
            
        # C. Fly Diagonal (7s to 11s)
        elif 7.0 <= t < 11.0:
            phase = "3. Fly Forward (45 deg)"
            pitch_cmd = 0.5
            
        # D. Turn 45 deg again (11s to 13s)
        elif 11.0 <= t < 13.0:
            phase = "4. Turn 45 deg"
            yaw_cmd = -0.4
            
        # E. Fly West/East (13s to 17s)
        elif 13.0 <= t < 17.0:
            phase = "5. Fly Forward (90 deg)"
            pitch_cmd = 0.5
            
        # F. Stop
        else:
            phase = "Hover"

        # Apply
        # Inputs: [Roll, Pitch, YawRate, VertVel]
        sim.step_simulation({drone_name: [0.0, pitch_cmd, yaw_cmd, 0.0]})
        
        # Log
        if drone_name in sim.drones:
            drone = sim.drones[drone_name]
            pos = drone.get_position()
            rpy = drone.get_orientation_rpy()
            
            history_pos.append(pos)
            history_yaw.append(rpy[2]) # Yaw
            
        if step % 60 == 0:
            print(f"T={t:4.1f} | Yaw={np.degrees(history_yaw[-1]):4.0f}° | Phase: {phase}")

    sim.stop_simulation()
    plot_staircase(history_pos, history_yaw)

def plot_staircase(pos_data, yaw_data):
    """Plots the path top-down to verify geometry."""
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except:
        plt.style.use('default')

    pos = np.array(pos_data)
    yaw = np.degrees(np.array(yaw_data))
    
    fig = plt.figure(figsize=(10, 8))
    
    # 1. Top Down Track
    ax1 = fig.add_subplot(1, 1, 1)
    
    # Color line by Time or Yaw to see progression
    sc = ax1.scatter(pos[:, 0], pos[:, 1], c=yaw, cmap='hsv', s=5, label='Path Color=Yaw')
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('Heading (Degrees)')
    
    ax1.set_title('Staircase Test: Fly -> Turn -> Fly')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.axis('equal')
    ax1.grid(True)
    
    # Annotate Start/End
    ax1.text(pos[0,0], pos[0,1], "START", fontweight='bold')
    ax1.text(pos[-1,0], pos[-1,1], "END", fontweight='bold')

    plt.tight_layout()
    plt.savefig('output_staircase.png', dpi=300)
    print("✅ Saved output_staircase.png")

if __name__ == "__main__":
    run_staircase_test()