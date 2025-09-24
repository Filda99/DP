#!/usr/bin/env python3
"""
Úplně jednoduchá PyBullet simulace - přímé ovládání silami.
Bez komplikovaných wrapper tříd.
"""

import sys
import os
import numpy as np
import time
import pybullet as p

# Matplotlib pro vizualizaci
try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available - no visualization will be created")

def create_flight_visualization(trajectory, forces_log, velocities_log, commands_log, joystick_commands):
    """Vytvoří komprehensivní vizualizaci letu."""
    
    if not HAS_MATPLOTLIB:
        print("Cannot create visualization - matplotlib not available")
        return
    
    print("\n📊 Creating flight visualization...")
    
    # Převod na numpy arrays
    traj = np.array(trajectory)
    forces = np.array(forces_log)
    vels = np.array(velocities_log)
    
    # Vytvoř figure s subploty
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D Trajectory
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=3, alpha=0.8, label='Flight path')
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=100, marker='o', label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=100, marker='s', label='End')
    
    # Mark command boundaries
    step_idx = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(joystick_commands)))
    for i, (cmd, duration, desc) in enumerate(joystick_commands):
        if step_idx < len(traj):
            end_idx = min(step_idx + duration, len(traj) - 1)
            ax1.plot(traj[step_idx:end_idx, 0], traj[step_idx:end_idx, 1], traj[step_idx:end_idx, 2], 
                    color=colors[i], linewidth=2, alpha=0.6)
            step_idx = end_idx
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Flight Trajectory')
    ax1.legend()
    
    # 2. Top view (XY)
    ax2 = fig.add_subplot(232)
    ax2.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, alpha=0.8)
    ax2.scatter(traj[0, 0], traj[0, 1], color='green', s=100, marker='o', label='Start')
    ax2.scatter(traj[-1, 0], traj[-1, 1], color='red', s=100, marker='s', label='End')
    
    # Šipky pro směr
    for i in range(0, len(traj), len(traj)//20):
        if i + 1 < len(traj):
            dx = traj[i+1, 0] - traj[i, 0]
            dy = traj[i+1, 1] - traj[i, 1]
            ax2.arrow(traj[i, 0], traj[i, 1], dx*5, dy*5, head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Square Flight Pattern')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. Altitude profile
    ax3 = fig.add_subplot(233)
    time_steps = range(len(traj))
    ax3.plot(time_steps, traj[:, 2], 'b-', linewidth=2, label='Altitude')
    ax3.axhline(y=5.0, color='green', linestyle='--', alpha=0.7, label='Target altitude')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Profile')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Forces over time
    ax4 = fig.add_subplot(234)
    ax4.plot(time_steps, forces[:, 0], 'r-', label='Force X', linewidth=2)
    ax4.plot(time_steps, forces[:, 1], 'g-', label='Force Y', linewidth=2)
    ax4.plot(time_steps, forces[:, 2], 'b-', label='Force Z', linewidth=2)
    ax4.axhline(y=4.9, color='gray', linestyle='--', alpha=0.7, label='Hover force (4.9N)')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Force (N)')
    ax4.set_title('Applied Forces')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Speed profile
    ax5 = fig.add_subplot(235)
    speeds = [np.linalg.norm(vel) for vel in vels]
    ax5.plot(time_steps, speeds, 'purple', linewidth=2, label='Total speed')
    ax5.plot(time_steps, [abs(vel[0]) for vel in vels], 'r--', alpha=0.7, label='Speed X')
    ax5.plot(time_steps, [abs(vel[1]) for vel in vels], 'g--', alpha=0.7, label='Speed Y')
    ax5.plot(time_steps, [abs(vel[2]) for vel in vels], 'b--', alpha=0.7, label='Speed Z')
    ax5.set_xlabel('Simulation Step')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Velocity Profile')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Command timeline
    ax6 = fig.add_subplot(236)
    step_idx = 0
    for i, (cmd, duration, desc) in enumerate(joystick_commands):
        color = colors[i]
        ax6.barh(i, duration, left=step_idx, color=color, alpha=0.7, 
                label=f'{desc[:20]}{"..." if len(desc) > 20 else ""}')
        
        # Joystick values text
        joystick_text = f"[{cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f}]"
        ax6.text(step_idx + duration/2, i, joystick_text, 
                ha='center', va='center', fontsize=8, weight='bold')
        
        step_idx += duration
    
    ax6.set_xlabel('Simulation Step')
    ax6.set_ylabel('Command Index')
    ax6.set_title('Joystick Command Timeline')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quadcopter_flight_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'quadcopter_flight_analysis.png'")
    plt.close()
    
    # Dodatečný graf - force vs position
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Force vs X position
    ax1.scatter(traj[:, 0], forces[:, 0], c=time_steps, cmap='plasma', alpha=0.6)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('X Force (N)')
    ax1.set_title('X Force vs X Position')
    ax1.grid(True)
    
    # Force vs Y position  
    ax2.scatter(traj[:, 1], forces[:, 1], c=time_steps, cmap='plasma', alpha=0.6)
    ax2.set_xlabel('Y Position (m)')
    ax2.set_ylabel('Y Force (N)')
    ax2.set_title('Y Force vs Y Position')
    ax2.grid(True)
    
    # Z force analysis
    ax3.scatter(traj[:, 2], forces[:, 2], c=time_steps, cmap='plasma', alpha=0.6)
    ax3.axhline(y=4.9, color='red', linestyle='--', label='Hover force')
    ax3.set_xlabel('Z Position (m)')
    ax3.set_ylabel('Z Force (N)')
    ax3.set_title('Z Force vs Altitude')
    ax3.legend()
    ax3.grid(True)
    
    # Phase space (position vs velocity)
    speed_total = [np.linalg.norm(vel) for vel in vels]
    ax4.scatter(traj[:, 2], speed_total, c=time_steps, cmap='plasma', alpha=0.6)
    ax4.set_xlabel('Altitude (m)')
    ax4.set_ylabel('Total Speed (m/s)')
    ax4.set_title('Phase Space: Altitude vs Speed')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('quadcopter_force_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Force analysis saved as 'quadcopter_force_analysis.png'")
    plt.close()

def simple_quadcopter_demo():
    """Jednoduchá PyBullet simulace přímo přes PyBullet API."""
    
    print("🚁 Simple PyBullet Quadcopter Demo")
    print("=" * 50)
    
    # Spustí PyBullet
    client = p.connect(p.DIRECT)  # Headless mode
    p.setGravity(0, 0, -9.81)
    
    # Vytvoř zem (simple plane)
    plane_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1], rgbaColor=[0.7, 0.7, 0.7, 1])
    plane_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1])
    plane_id = p.createMultiBody(0, plane_collision, plane_visual, [0, 0, -0.1])
    
    # Vytvoř kvadrokoptéru - jednoduchý box
    start_pos = [0, 0, 5]
    visual_id = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.3, 0.3, 0.1],
        rgbaColor=[0.2, 0.8, 0.2, 1.0]
    )
    collision_id = p.createCollisionShape(
        p.GEOM_BOX, 
        halfExtents=[0.3, 0.3, 0.1]
    )
    
    drone_id = p.createMultiBody(
        baseMass=0.5,  # 0.5 kg
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=start_pos
    )
    
    print(f"✓ Drone created with ID {drone_id} at {start_pos}")
    
    # Joystick sekvence pro čtverec
    joystick_commands = [
        # Formát: [left_right, forward_back, up_down, duration, description]
        ([0.0, 0.0, 0.0], 50, "Hover at start"),
        ([-1.0, 0.0, 0.0], 80, "Fly LEFT"),
        ([0.0, 0.0, 0.0], 30, "Hover corner 1"),
        ([0.0, 1.0, 0.0], 80, "Fly FORWARD"), 
        ([0.0, 0.0, 0.0], 30, "Hover corner 2"),
        ([1.0, 0.0, 0.0], 80, "Fly RIGHT"),
        ([0.0, 0.0, 0.0], 30, "Hover corner 3"),
        ([0.0, -1.0, 0.0], 80, "Fly BACK"),
        ([0.0, 0.0, 0.0], 30, "Square complete"),
        ([0.0, 0.0, 1.0], 60, "Fly UP"),
        ([0.0, 0.0, 0.0], 30, "Hover at top"),
        ([0.0, 0.0, -0.5], 60, "Fly DOWN"),
        ([0.0, 0.0, 0.0], 40, "Final hover"),
    ]
    
    print("Starting joystick simulation...")
    print("=" * 50)
    
    total_steps = 0
    
    # Logging pro vizualizaci
    trajectory = []
    forces_log = []
    velocities_log = []
    commands_log = []
    
    for cmd_idx, (joystick, duration, description) in enumerate(joystick_commands):
        print(f"\n🎮 {description}")
        print(f"  Joystick: [L/R:{joystick[0]:+4.1f}, F/B:{joystick[1]:+4.1f}, U/D:{joystick[2]:+4.1f}]")
        
        for step in range(duration):
            # Převod joystick inputu na síly
            # Horizontální síly (X, Y)
            force_x = joystick[0] * 10.0  # Max 10N horizontálně
            force_y = joystick[1] * 10.0
            
            # Vertikální síla (Z) - hover + input
            hover_force = 0.5 * 9.81  # Kompenzace gravitace (mass * g)
            vertical_input = joystick[2] * 15.0  # Max 15N extra nahoru/dolů
            force_z = hover_force + vertical_input
            
            force = [force_x, force_y, force_z]
            
            # Aplikuj sílu na střed hmotnosti
            p.applyExternalForce(
                drone_id, 
                -1,  # Base link
                force, 
                [0, 0, 0],  # Pozice síly (střed)
                p.WORLD_FRAME
            )
            
            # Krok simulace
            p.stepSimulation()
            
            # Logování pro graf
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            vel, _ = p.getBaseVelocity(drone_id)
            trajectory.append(list(pos))
            velocities_log.append(list(vel))
            forces_log.append(force)
            commands_log.append([cmd_idx, step, description])
            
            # Status každých 20 kroků
            if step % 20 == 0:
                speed = np.linalg.norm(vel)
                print(f"  Step {step:3d}: Pos=[{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}] "
                      f"Speed={speed:4.1f}m/s Force=[{force[0]:5.1f}, {force[1]:5.1f}, {force[2]:5.1f}]N")
            
            total_steps += 1
            time.sleep(0.01)  # Real-time feel
        
        # Final position po příkazu
        pos, _ = p.getBasePositionAndOrientation(drone_id)
        print(f"  ✓ Command completed at: [{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}]")
    
    # Final stats
    final_pos, _ = p.getBasePositionAndOrientation(drone_id)
    start_pos = np.array([0, 0, 5])
    final_distance = np.linalg.norm(np.array(final_pos) - start_pos)
    
    print("\n" + "=" * 50)
    print("🎉 SIMPLE SIMULATION COMPLETED!")
    print(f"📊 Total steps: {total_steps}")
    print(f"   Start position: [0.0, 0.0, 5.0]")
    print(f"   Final position: [{final_pos[0]:6.1f}, {final_pos[1]:6.1f}, {final_pos[2]:6.1f}]")
    print(f"   Distance from start: {final_distance:.1f}m")
    
    if final_distance < 3.0:
        print("   ✅ Excellent precision!")
    elif final_distance < 8.0:
        print("   ✓ Good precision")
    else:
        print("   ⚠ Could be more precise")
    
    p.disconnect()
    print("PyBullet disconnected.")
    
    # Vytvoř vizualizaci
    create_flight_visualization(trajectory, forces_log, velocities_log, commands_log, joystick_commands)

if __name__ == "__main__":
    simple_quadcopter_demo()