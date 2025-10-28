"""
Test script showing drone behavior near GROWING fire.
Demonstrates how fire spread increases convection forces over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


class Arrow3D(FancyArrowPatch):
    """Helper class for drawing 3D arrows"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def run_growing_fire_test(altitude, horizontal_distance=5.0, duration=30.0, dt=0.1):
    """
    Run test with GROWING fire to show increasing convection effects.
    
    Args:
        altitude: Hovering altitude
        horizontal_distance: Distance from fire center
        duration: Simulation time (longer to see fire growth)
        dt: Time step
    """
    print(f"\n{'='*70}")
    print(f"GROWING FIRE TEST: Hovering at {altitude}m, {horizontal_distance}m from fire")
    print(f"{'='*70}")
    
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Enable fire with aggressive spread parameters
    sim.environment.enable_fire_simulation(
        grid_width_m=100,
        grid_height_m=100,
        cell_size_m=2.0,
        dt=0.1,
        alpha=0.5,  # Less distance decay = spreads further
        k_wind=2.0  # Stronger wind influence
    )
    
    # Make fire spread MORE aggressive
    sim.environment.fire_grid.l_base *= 2.0  # 2x faster base spread
    
    sim.fire_enabled = True
    sim.grid_mapper = sim.environment.grid_mapper
    sim.fire_grid = sim.environment.fire_grid
    
    # Set moderate wind to help fire spread
    sim.weather = {'wind_velocity': np.array([2.0, 0.0, 0.0])}
    
    # Start MULTIPLE fires to create larger initial fire
    fire_positions = [
        (0, 0),    # Center
        (2, 0),    # Right
        (-2, 0),   # Left
        (0, 2),    # Up
        (0, -2),   # Down
    ]
    
    print(f"🔥 Starting initial fire cluster...")
    fires_started = 0
    for pos in fire_positions:
        if sim.environment.start_fire_at_position(pos, intensity=0.8):
            fires_started += 1
    print(f"   ✓ {fires_started} initial fire cells ignited")
    
    # Add drone
    start_pos = [horizontal_distance, 0.0, altitude]
    drone = sim.add_quadcopter("test_drone", position=start_pos, mass=0.5)
    
    print(f"✅ Drone at: ({horizontal_distance}, 0, {altitude}) m")
    print(f"✅ Wind: {sim.weather['wind_velocity']} m/s")
    print(f"\n🔄 Running {duration}s simulation with GROWING fire...")
    
    # Storage
    trajectory = {
        'times': [],
        'positions': [],
        'velocities': [],
        'airflows': [],
        'forces_applied': [],
        'fire_stats': [],  # Track fire growth
        'burning_cells': [],
        'target_altitude': altitude,
        'horizontal_distance': horizontal_distance
    }
    
    # Run simulation
    steps = int(duration / dt)
    progress_marks = [0, 10, 20, 30]
    
    for step in range(steps):
        t = step * dt
        
        # Get current state
        pos = drone.get_position()
        vel = drone.get_velocity()
        airflow = sim.get_local_airflow(pos)
        
        # Get fire statistics
        fire_state = sim.environment.get_fire_state()
        if fire_state:
            fire_stats = fire_state['fire_stats']
            burning_cells = fire_stats['burning_cells']
        else:
            burning_cells = 0
        
        # Store data
        trajectory['times'].append(t)
        trajectory['positions'].append(pos.copy())
        trajectory['velocities'].append(vel.copy())
        trajectory['airflows'].append(airflow.copy())
        trajectory['burning_cells'].append(burning_cells)
        
        # Weak hovering controller
        target_pos = np.array(start_pos)
        pos_error = target_pos - pos
        vel_error = -vel
        
        Kp = 0.5
        Kd = 0.3
        control_correction = Kp * pos_error + Kd * vel_error
        
        max_correction = 5.0
        joystick_input = np.clip(control_correction / max_correction, -1.0, 1.0)
        
        trajectory['forces_applied'].append(joystick_input.copy())
        
        # Step simulation with fire update
        controls = {"test_drone": joystick_input}
        sim.step_simulation(controls)
        
        # Progress reporting
        current_sec = int(t)
        if current_sec in progress_marks:
            print(f"   [{current_sec:2d}s] Fire: {burning_cells:3d} cells | "
                  f"Airflow Z: {airflow[2]:5.2f} m/s | "
                  f"Drone drift: {np.linalg.norm(pos - start_pos):.2f}m")
            progress_marks.remove(current_sec)
    
    # Results
    final_pos = trajectory['positions'][-1]
    displacement = final_pos - np.array(start_pos)
    initial_burning = trajectory['burning_cells'][0]
    final_burning = trajectory['burning_cells'][-1]
    
    print(f"\n📊 RESULTS:")
    print(f"   Start position:  [{start_pos[0]:6.2f}, {start_pos[1]:6.2f}, {start_pos[2]:6.2f}] m")
    print(f"   Final position:  [{final_pos[0]:6.2f}, {final_pos[1]:6.2f}, {final_pos[2]:6.2f}] m")
    print(f"   Total drift:     {np.linalg.norm(displacement):.2f} m")
    print(f"\n🔥 FIRE GROWTH:")
    print(f"   Initial: {initial_burning} cells")
    print(f"   Final:   {final_burning} cells")
    print(f"   Growth:  {final_burning - initial_burning} cells (+{(final_burning/max(1,initial_burning)-1)*100:.0f}%)")
    
    avg_airflow = np.mean(trajectory['airflows'], axis=0)
    print(f"\n🌬️  Average airflow:")
    print(f"   Vertical (Z): {avg_airflow[2]:6.2f} m/s")
    
    sim.stop_simulation()
    
    return trajectory


def plot_growing_fire_comparison(trajectories, filename='output/growing_fire_comparison.png'):
    """
    Create visualization showing fire growth effect on drones at different altitudes.
    """
    fig = plt.figure(figsize=(20, 14))
    
    scenarios = [
        ('Low Altitude (10m)', trajectories['low'], 0, 'red'),
        ('Mid Altitude (25m)', trajectories['mid'], 1, 'orange'),
        ('High Altitude (40m)', trajectories['high'], 2, 'green')
    ]
    
    for idx, (title, traj, row, color) in enumerate(scenarios):
        positions = np.array(traj['positions'])
        airflows = np.array(traj['airflows'])
        times = np.array(traj['times'])
        burning_cells = np.array(traj['burning_cells'])
        
        # Plot 1: 3D Trajectory with time coloring
        ax1 = fig.add_subplot(3, 4, row*4 + 1, projection='3d')
        
        # Fire location
        ax1.scatter([0], [0], [0], c='red', s=300, marker='*', 
                   label='Fire', edgecolors='darkred', linewidths=2, alpha=0.8)
        
        # Trajectory with gradient
        for i in range(len(positions) - 1):
            # Color based on fire size at this time
            fire_fraction = burning_cells[i] / max(burning_cells)
            trajectory_color = plt.cm.hot(fire_fraction)
            ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                    color=trajectory_color, linewidth=2, alpha=0.7)
        
        # Start and end
        ax1.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                   c='blue', s=150, marker='o', label='Start')
        ax1.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                   c='purple', s=150, marker='X', label='End')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{title}\n3D Path (color = fire size)', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.set_xlim([-2, 10])
        ax1.set_ylim([-5, 5])
        ax1.set_zlim([0, 50])
        ax1.view_init(elev=20, azim=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fire Growth Over Time
        ax2 = fig.add_subplot(3, 4, row*4 + 2)
        ax2.plot(times, burning_cells, 'r-', linewidth=3, label='Burning cells')
        ax2.fill_between(times, 0, burning_cells, alpha=0.3, color='red')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Burning Cells')
        ax2.set_title(f'{title}\nFire Growth', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add growth rate annotation
        growth = burning_cells[-1] - burning_cells[0]
        ax2.text(0.95, 0.95, f'Growth: +{growth} cells\n({(burning_cells[-1]/max(1,burning_cells[0])-1)*100:.0f}%)',
                transform=ax2.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Plot 3: Airflow Magnitude Over Time
        ax3 = fig.add_subplot(3, 4, row*4 + 3)
        
        # Vertical airflow
        ax3.plot(times, airflows[:, 2], 'b-', linewidth=2, label='Updraft (Z)', alpha=0.8)
        ax3.fill_between(times, 0, airflows[:, 2], alpha=0.2, color='blue')
        
        # Horizontal airflow magnitude
        horizontal_mag = np.sqrt(airflows[:, 0]**2 + airflows[:, 1]**2)
        ax3.plot(times, horizontal_mag, 'g-', linewidth=2, label='Horizontal', alpha=0.8)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Airflow (m/s)')
        ax3.set_title(f'{title}\nAirflow vs Fire Size', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add correlation annotation
        if len(airflows) > 10:
            updraft_growth = airflows[-10:, 2].mean() - airflows[:10, 2].mean()
            ax3.text(0.05, 0.95, f'Updraft change:\n{updraft_growth:+.2f} m/s',
                    transform=ax3.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Plot 4: Drift Distance Over Time
        ax4 = fig.add_subplot(3, 4, row*4 + 4)
        
        start_pos = positions[0]
        drift_distances = [np.linalg.norm(pos - start_pos) for pos in positions]
        
        ax4.plot(times, drift_distances, color=color, linewidth=3, label='Total drift')
        ax4.fill_between(times, 0, drift_distances, alpha=0.3, color=color)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Drift Distance (m)')
        ax4.set_title(f'{title}\nDrone Displacement', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add final drift annotation
        ax4.text(0.95, 0.05, f'Final drift:\n{drift_distances[-1]:.2f} m',
                transform=ax4.transAxes, fontsize=10, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('GROWING FIRE TEST: Drone Behavior vs Fire Spread\nFire grows over 30 seconds, creating increasing convection forces',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {filename}")
    plt.close()


def plot_fire_evolution_snapshots(traj, filename='output/fire_evolution_snapshots.png'):
    """
    Show fire growth and drone position at different time points.
    """
    fig = plt.figure(figsize=(20, 10))
    
    positions = np.array(traj['positions'])
    times = np.array(traj['times'])
    burning_cells = np.array(traj['burning_cells'])
    
    # Select 6 time points
    snapshot_times = [0, 6, 12, 18, 24, 30]
    
    for plot_idx, snap_time in enumerate(snapshot_times):
        idx = np.argmin(np.abs(times - snap_time))
        
        ax = fig.add_subplot(2, 3, plot_idx + 1, projection='3d')
        
        # Fire (size represents burning cells)
        fire_size = 50 + burning_cells[idx] * 10  # Scale with fire size
        ax.scatter([0], [0], [0], c='red', s=fire_size, marker='*',
                  edgecolors='darkred', linewidths=2, alpha=0.8,
                  label=f'{int(burning_cells[idx])} cells burning')
        
        # Path so far
        ax.plot(positions[:idx+1, 0], positions[:idx+1, 1], positions[:idx+1, 2],
               'b-', linewidth=2, alpha=0.5)
        
        # Current drone
        curr_pos = positions[idx]
        ax.scatter([curr_pos[0]], [curr_pos[1]], [curr_pos[2]],
                  c='blue', s=200, marker='o', edgecolors='darkblue', linewidths=2)
        
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(f'T = {snap_time:.0f}s\nFire: {int(burning_cells[idx])} cells',
                    fontsize=12, fontweight='bold')
        ax.set_xlim([-5, 10])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 50])
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle(f'Fire Evolution: {traj["target_altitude"]:.0f}m Altitude\nWatch fire grow (red star size) and drone drift',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def main():
    """Run growing fire test at three altitudes."""
    print("="*70)
    print("GROWING FIRE TEST - 30 Second Simulation")
    print("="*70)
    print("\nFire will GROW and SPREAD during simulation")
    print("Convection forces should INCREASE as fire gets larger")
    print("Drone controller will struggle more as fire intensifies")
    print("="*70)
    
    trajectories = {}
    
    print("\n🔥 SCENARIO 1: LOW ALTITUDE (10m)")
    trajectories['low'] = run_growing_fire_test(altitude=10.0, horizontal_distance=5.0, duration=30.0)
    
    print("\n🔥 SCENARIO 2: MID ALTITUDE (25m)")
    trajectories['mid'] = run_growing_fire_test(altitude=25.0, horizontal_distance=5.0, duration=30.0)
    
    print("\n🔥 SCENARIO 3: HIGH ALTITUDE (40m)")
    trajectories['high'] = run_growing_fire_test(altitude=40.0, horizontal_distance=5.0, duration=30.0)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    plot_growing_fire_comparison(trajectories)
    plot_fire_evolution_snapshots(trajectories['mid'], 'output/fire_evolution_mid_altitude.png')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Fire Growth Impact on Drones")
    print("="*70)
    
    for name, traj in trajectories.items():
        positions = np.array(traj['positions'])
        burning_cells = np.array(traj['burning_cells'])
        airflows = np.array(traj['airflows'])
        
        initial_fire = burning_cells[0]
        final_fire = burning_cells[-1]
        fire_growth = final_fire - initial_fire
        
        initial_updraft = airflows[:50, 2].mean()  # First 5 seconds
        final_updraft = airflows[-50:, 2].mean()   # Last 5 seconds
        updraft_increase = final_updraft - initial_updraft
        
        start_pos = positions[0]
        final_drift = np.linalg.norm(positions[-1] - start_pos)
        
        print(f"\n{name.upper()} ({traj['target_altitude']:.0f}m):")
        print(f"  Fire: {initial_fire} → {final_fire} cells (+{fire_growth}, +{fire_growth/max(1,initial_fire)*100:.0f}%)")
        print(f"  Updraft: {initial_updraft:.2f} → {final_updraft:.2f} m/s ({updraft_increase:+.2f} m/s)")
        print(f"  Final drift: {final_drift:.2f} m")
        print(f"  📊 Fire grew {fire_growth/max(1,initial_fire)*100:.0f}%, updraft increased {updraft_increase/max(0.1,initial_updraft)*100:.0f}%")
    
    print("\n" + "="*70)
    print("✅ GROWING FIRE TEST COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • output/growing_fire_comparison.png - Full 3x4 comparison")
    print("  • output/fire_evolution_mid_altitude.png - Time-lapse snapshots")
    print("\nKey insight: As fire grows, convection forces increase,")
    print("causing drones to drift more despite controller efforts!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
