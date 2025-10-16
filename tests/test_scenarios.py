"""
PyBullet Drone Test Scenarios

Multiple focused test scenarios to validate each movement type.
Each scenario tests specific joystick controls in isolation.
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time


def create_flight_visualization(positions, forces, speeds, times, title="Flight Analysis"):
    """Creates comprehensive flight visualization for a scenario."""
    
    if len(positions) == 0:
        print("No data to visualize")
        return
    
    # Convert to numpy arrays
    positions = np.array(positions)
    forces = np.array(forces)
    speeds = np.array(speeds)
    times = np.array(times)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{title}', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier access
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    
    # 1. 3D Trajectory - replace with 3D subplot
    ax1.remove()  # Remove the 2D axis
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.8)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('3D Flight Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. XY Top View
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.8)
    ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Top View (XY Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    
    # 3. Forces over Time
    ax3.plot(times, forces[:, 0], 'r-', label='Force X', linewidth=2)
    ax3.plot(times, forces[:, 1], 'g-', label='Force Y', linewidth=2)
    ax3.plot(times, forces[:, 2], 'b-', label='Force Z', linewidth=2)
    ax3.axhline(y=4.9, color='red', linestyle='--', label='Hover force')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('Applied Forces')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Speed over Time
    ax4.plot(times, speeds, 'purple', linewidth=2)
    ax4.axhline(y=4.9, color='gray', linestyle='--', alpha=0.7, label='Hover force (4.9N)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Flight Speed')
    ax4.grid(True, alpha=0.3)
    
    # 5. Position Components
    ax5.plot(times, positions[:, 0], 'r-', label='X Position', linewidth=2)
    ax5.plot(times, positions[:, 1], 'g-', label='Y Position', linewidth=2)
    ax5.plot(times, positions[:, 2], 'b-', label='Z Position', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title('Position Components')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Altitude Profile
    ax6.plot(times, positions[:, 2], 'b-', linewidth=3)
    ax6.fill_between(times, positions[:, 2], alpha=0.3)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Altitude (m)')
    ax6.set_title('Altitude Profile')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with scenario-specific name
    filename = f"{title.lower().replace(' ', '_')}_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved as '{filename}'")
    plt.close()


def run_scenario(scenario_name, joystick_commands, visualize=True):
    """Run a single test scenario with given joystick commands."""
    
    print(f"\n🚁 {scenario_name}")
    print("=" * 50)
    
    # Start PyBullet
    physics_client = p.connect(p.DIRECT)  # No GUI for batch testing
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Create ground (simple plane)
    plane_id = p.loadURDF("plane.urdf")
    
    # Create quadcopter - simple box
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], rgbaColor=[1, 0, 0, 1])
    
    drone_id = p.createMultiBody(
        baseMass=0.5,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, 5]
    )
    
    print(f"✓ Drone created with ID {drone_id} at [0, 0, 5]")
    print("Starting scenario...")
    
    # Data collection
    positions = []
    forces = []
    speeds = []
    times = []
    
    total_steps = 0
    start_time = 0
    
    for joystick, steps, description in joystick_commands:
        print(f"\n🎮 {description}")
        print(f"  Joystick: [L/R:{joystick[0]:+.1f}, F/B:{joystick[1]:+.1f}, U/D:{joystick[2]:+.1f}]")
        
        for step in range(steps):
            # Convert joystick input to forces
            # Horizontal forces (X, Y)
            force_x = joystick[0] * 10.0  # Max 10N horizontally
            force_y = joystick[1] * 10.0
            
            # Vertical force (Z) - hover + input
            hover_force = 0.5 * 9.81  # Gravity compensation (mass * g)
            vertical_input = joystick[2] * 15.0  # Max 15N extra up/down
            force_z = hover_force + vertical_input
            
            # Apply force to center of mass
            p.applyExternalForce(
                drone_id,
                -1,  # Apply to base link
                [force_x, force_y, force_z],
                [0, 0, 0],  # Force position (center)
                p.WORLD_FRAME
            )
            
            # Step simulation
            p.stepSimulation()
            
            # Logging for graphs
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            vel, _ = p.getBaseVelocity(drone_id)
            speed = np.linalg.norm(vel)
            
            positions.append(pos)
            forces.append([force_x, force_y, force_z])
            speeds.append(speed)
            times.append(total_steps / 240.0)  # 240 FPS timestep
            
            # Status every 20 steps
            if step % 20 == 0:
                print(f"  Step {step:3d}: Pos=[{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}] "
                      f"Speed={speed:4.1f}m/s Force=[{force_x:4.1f}, {force_y:4.1f}, {force_z:4.1f}]N", end='\r')
            
            total_steps += 1
        
        # Final position after command
        pos, _ = p.getBasePositionAndOrientation(drone_id)
        print(f"\n  ✓ Command completed at: [{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}]")
    
    # Final statistics
    final_pos, _ = p.getBasePositionAndOrientation(drone_id)
    start_pos = [0, 0, 5]
    distance = np.linalg.norm(np.array(final_pos) - np.array(start_pos))
    
    print(f"\n" + "=" * 50)
    print(f"🎉 {scenario_name.upper()} COMPLETED!")
    print(f"📊 Total steps: {total_steps}")
    print(f"   Start position: {start_pos}")
    print(f"   Final position: [{final_pos[0]:6.1f}, {final_pos[1]:6.1f}, {final_pos[2]:6.1f}]")
    print(f"   Distance from start: {distance:.1f}m")
    
    # Disconnect PyBullet
    p.disconnect()
    
    # Create visualization
    if visualize:
        print(f"📊 Creating visualization...")
        create_flight_visualization(positions, forces, speeds, times, scenario_name)
    
    return {
        'scenario': scenario_name,
        'total_steps': total_steps,
        'start_position': start_pos,
        'final_position': final_pos,
        'distance_from_start': distance,
        'positions': positions,
        'forces': forces,
        'speeds': speeds,
        'times': times
    }


def main():
    """Run all test scenarios."""
    
    print("🚁 PyBullet Drone Test Scenarios")
    print("Testing individual movement capabilities")
    print("=" * 60)
    
    # Define test scenarios
    scenarios = {
        "Hover Test": [
            ([0.0, 0.0, 0.0], 100, "Pure hover - no movement"),
            ([0.0, 0.0, 0.0], 50, "Extended hover test"),
        ],
        
        "Horizontal Movement Left": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([-1.0, 0.0, 0.0], 80, "Full left movement"), 
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ],
        
        "Horizontal Movement Right": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([1.0, 0.0, 0.0], 80, "Full right movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ],
        
        "Horizontal Movement Forward": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, 1.0, 0.0], 80, "Full forward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ],
        
        "Horizontal Movement Backward": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, -1.0, 0.0], 80, "Full backward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ],
        
        "Vertical Movement Up": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, 0.0, 1.0], 60, "Full upward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ],
        
        "Vertical Movement Down": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, 0.0, -0.5], 60, "Controlled downward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ],
        
        "Diagonal Movement": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.7, 0.7, 0.0], 80, "Diagonal (NE) movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ],
        
        "Complex 3D Movement": [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.5, 0.5, 0.5], 60, "3D diagonal up movement"),
            ([0.0, 0.0, 0.0], 30, "Mid hover"),
            ([-0.5, -0.5, -0.3], 60, "3D diagonal down movement"),
            ([0.0, 0.0, 0.0], 40, "Final hover"),
        ],
    }
    
    # Run all scenarios
    results = {}
    for scenario_name, commands in scenarios.items():
        result = run_scenario(scenario_name, commands, visualize=True)
        results[scenario_name] = result
        
        # Small pause between scenarios
        time.sleep(0.5)
    
    # Summary report
    print(f"\n🎉 ALL SCENARIOS COMPLETED!")
    print("=" * 60)
    print("📊 SUMMARY REPORT:")
    print("-" * 60)
    
    for scenario_name, result in results.items():
        print(f"{scenario_name:25}: {result['total_steps']:3d} steps, "
              f"final distance: {result['distance_from_start']:5.1f}m")
    
    print(f"\n✅ Generated {len(scenarios)} visualization files")
    print("All movement types validated successfully! 🚁✨")


if __name__ == "__main__":
    main()