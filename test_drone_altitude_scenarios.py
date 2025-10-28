"""
Test script to visualize drone behavior at different altitudes near fire.
Shows 3 scenarios: Low (10m), Mid (25m), High (40m) altitude hovering.
Generates 3D visualizations showing step-by-step what happens to a hovering drone.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d
import sys
import os

# Add project root to path
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


def run_hovering_test(altitude, duration=10.0, dt=0.1):
    """
    Run a test where TWO drones try to hover at given altitude near fire.
    One at 10m distance (FAR) and one at 5m distance (NEAR).
    
    Args:
        altitude: Target hovering altitude in meters
        duration: Simulation duration in seconds
        dt: Time step in seconds
        
    Returns:
        Dictionary with trajectory data for both drones
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: Hovering at {altitude}m altitude")
    print(f"{'='*70}")
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Enable fire simulation with single fire
    sim.environment.enable_fire_simulation(
        grid_width_m=100,
        grid_height_m=100,
        cell_size_m=2.0
    )
    sim.fire_enabled = True
    sim.grid_mapper = sim.environment.grid_mapper
    sim.fire_grid = sim.environment.fire_grid
    
    # Set minimal wind to isolate fire effects
    sim.weather = {'wind_velocity': np.array([0.0, 0.0, 0.0])}  # ZERO wind to see pure fire effects
    
    # Start fire at origin with high intensity
    sim.environment.start_fire_at_position([0, 0], intensity=1.0)
    
    # Add TWO quadcopters at different distances from fire
    start_pos_far = [10.0, 0.0, altitude]   # FAR drone (10m away)
    start_pos_near = [5.0, 0.0, altitude]   # NEAR drone (5m away - 2x closer!)
    
    drone_far = sim.add_quadcopter("drone_far", position=start_pos_far, mass=0.5)
    drone_near = sim.add_quadcopter("drone_near", position=start_pos_near, mass=0.5)
    
    drone_far = sim.add_quadcopter("drone_far", position=start_pos_far, mass=0.5)
    drone_near = sim.add_quadcopter("drone_near", position=start_pos_near, mass=0.5)
    
    print(f"✅ Drone FAR starting position:  {start_pos_far} (10m from fire)")
    print(f"✅ Drone NEAR starting position: {start_pos_near} (5m from fire)")
    print(f"✅ Fire at origin (0, 0) with intensity 1.0")
    print(f"✅ Global wind: {sim.weather['wind_velocity']} m/s (ZERO for clear comparison)")
    print(f"\n🔄 Running simulation for {duration}s...")
    
    # Storage for trajectories
    trajectory = {
        'far': {
            'times': [],
            'positions': [],
            'velocities': [],
            'airflows': [],
            'forces_applied': [],
            'target_altitude': altitude,
            'start_pos': start_pos_far
        },
        'near': {
            'times': [],
            'positions': [],
            'velocities': [],
            'airflows': [],
            'forces_applied': [],
            'target_altitude': altitude,
            'start_pos': start_pos_near
        }
    }
    
    # Run simulation
    steps = int(duration / dt)
    for step in range(steps):
        t = step * dt
        
        # Process BOTH drones
        for drone_name, drone, start_pos, traj_key in [
            ("drone_far", drone_far, start_pos_far, 'far'),
            ("drone_near", drone_near, start_pos_near, 'near')
        ]:
            # Get current state
            pos = drone.get_position()
            vel = drone.get_velocity()
            airflow = sim.get_local_airflow(pos)
            
            # Store data
            trajectory[traj_key]['times'].append(t)
            trajectory[traj_key]['positions'].append(pos.copy())
            trajectory[traj_key]['velocities'].append(vel.copy())
            trajectory[traj_key]['airflows'].append(airflow.copy())
            
            # Simple hovering controller - try to maintain position and altitude
            # This is intentionally weak to show the fire effects
            target_pos = np.array(start_pos)
            pos_error = target_pos - pos
            vel_error = -vel  # Want zero velocity
            
            # PID-like control (weak gains to show disturbance)
            Kp = 0.5  # Weak proportional gain
            Kd = 0.3  # Weak derivative gain
            control_correction = Kp * pos_error + Kd * vel_error
            
            # Convert to joystick input (normalized -1 to 1)
            # Scale down to show fire effects more clearly
            max_correction = 5.0  # Maximum force in Newtons
            joystick_input = np.clip(control_correction / max_correction, -1.0, 1.0)
            
            # Store the joystick input applied
            trajectory[traj_key]['forces_applied'].append(joystick_input.copy())
        
        # Step simulation with both drone controls
        controls = {
            "drone_far": trajectory['far']['forces_applied'][-1],
            "drone_near": trajectory['near']['forces_applied'][-1]
        }
        sim.step_simulation(controls)
    
    # Calculate statistics for BOTH drones
    print(f"\n📊 RESULTS:")
    
    for label, traj_key, start_pos in [
        ("FAR (10m)", 'far', start_pos_far),
        ("NEAR (5m)", 'near', start_pos_near)
    ]:
        final_pos = trajectory[traj_key]['positions'][-1]
        displacement = final_pos - np.array(start_pos)
        avg_airflow = np.mean(trajectory[traj_key]['airflows'], axis=0)
        
        print(f"\n   {label}:")
        print(f"      Start:        [{start_pos[0]:6.2f}, {start_pos[1]:6.2f}, {start_pos[2]:6.2f}] m")
        print(f"      Final:        [{final_pos[0]:6.2f}, {final_pos[1]:6.2f}, {final_pos[2]:6.2f}] m")
        print(f"      Displacement: [{displacement[0]:+6.2f}, {displacement[1]:+6.2f}, {displacement[2]:+6.2f}] m")
        print(f"      Total drift:  {np.linalg.norm(displacement):.2f} m")
        print(f"      Avg airflow:  [{avg_airflow[0]:+6.2f}, {avg_airflow[1]:+6.2f}, {avg_airflow[2]:+6.2f}] m/s")
    
    sim.stop_simulation()
    
    return trajectory


def plot_3d_trajectory(trajectories, filename='drone_altitude_comparison.png'):
    """
    Create comprehensive 3D visualization of all three altitude scenarios.
    Now shows BOTH drones (FAR and NEAR) for each altitude.
    """
    fig = plt.figure(figsize=(20, 14))
    
    scenarios = [
        ('Low Altitude (10m)', trajectories['low'], 0, 'red', 'orange'),
        ('Mid Altitude (25m)', trajectories['mid'], 1, 'blue', 'cyan'),
        ('High Altitude (40m)', trajectories['high'], 2, 'green', 'lime')
    ]
    
    # Create 3x3 grid of plots
    for idx, (title, traj, row, color_far, color_near) in enumerate(scenarios):
        # Extract data for BOTH drones
        positions_far = np.array(traj['far']['positions'])
        airflows_far = np.array(traj['far']['airflows'])
        positions_near = np.array(traj['near']['positions'])
        airflows_near = np.array(traj['near']['airflows'])
        times = np.array(traj['far']['times'])
        
        # Plot 1: 3D Trajectory
        ax1 = fig.add_subplot(3, 3, row*3 + 1, projection='3d')
        
        # Fire location
        fire_x, fire_y = 0, 0
        fire_size = 300
        ax1.scatter([fire_x], [fire_y], [0], c='red', s=fire_size, marker='*', 
                   label='Fire', edgecolors='darkred', linewidths=2, alpha=0.8, zorder=10)
        ax1.scatter([fire_x], [fire_y], [0], c='red', s=fire_size, marker='*', 
                   label='Fire', edgecolors='darkred', linewidths=2, alpha=0.8, zorder=10)
        
        # FAR Drone (10m away) - Trajectory
        ax1.plot(positions_far[:, 0], positions_far[:, 1], positions_far[:, 2],
                color=color_far, linewidth=3, label='FAR (10m)', alpha=0.8, linestyle='-')
        ax1.scatter([positions_far[0, 0]], [positions_far[0, 1]], [positions_far[0, 2]], 
                   c=color_far, s=150, marker='o', edgecolors='black', linewidths=2, zorder=5)
        ax1.scatter([positions_far[-1, 0]], [positions_far[-1, 1]], [positions_far[-1, 2]], 
                   c=color_far, s=150, marker='X', edgecolors='black', linewidths=2, zorder=5)
        
        # NEAR Drone (5m away) - Trajectory
        ax1.plot(positions_near[:, 0], positions_near[:, 1], positions_near[:, 2],
                color=color_near, linewidth=3, label='NEAR (5m)', alpha=0.8, linestyle='--')
        ax1.scatter([positions_near[0, 0]], [positions_near[0, 1]], [positions_near[0, 2]], 
                   c=color_near, s=150, marker='o', edgecolors='black', linewidths=2, zorder=5)
        ax1.scatter([positions_near[-1, 0]], [positions_near[-1, 1]], [positions_near[-1, 2]], 
                   c=color_near, s=150, marker='X', edgecolors='black', linewidths=2, zorder=5)
        
        # Add airflow vectors for NEAR drone (shows stronger effects)
        sample_indices = np.linspace(0, len(positions_near)-1, 4, dtype=int)
        for i in sample_indices:
            pos = positions_near[i]
            airflow = airflows_near[i] * 0.5  # Scale for visibility
            
            # Draw arrow showing airflow
            arrow = Arrow3D([pos[0], pos[0] + airflow[0]],
                          [pos[1], pos[1] + airflow[1]],
                          [pos[2], pos[2] + airflow[2]],
                          mutation_scale=15, lw=2, arrowstyle='->', color='magenta', alpha=0.7)
            ax1.add_artist(arrow)
        
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_title(f'{title}\n3D Trajectories (FAR vs NEAR)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_xlim([-5, 15])
        ax1.set_ylim([-10, 10])
        ax1.set_zlim([0, 50])
        ax1.view_init(elev=20, azim=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Position vs Time (NEAR drone - shows stronger effects)
        ax2 = fig.add_subplot(3, 3, row*3 + 2)
        ax2.plot(times, positions_near[:, 0], '-', color=color_near, label='X (NEAR)', linewidth=2.5)
        ax2.plot(times, positions_far[:, 0], '--', color=color_far, label='X (FAR)', linewidth=1.5, alpha=0.7)
        ax2.plot(times, positions_near[:, 2], '-', color='blue', label='Z (NEAR)', linewidth=2.5)
        ax2.plot(times, positions_far[:, 2], '--', color='navy', label='Z (FAR)', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=traj['far']['target_altitude'], color='k', linestyle=':', alpha=0.5, label='Target alt')
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Position (m)', fontsize=10)
        ax2.set_title(f'{title}\nPosition: NEAR (solid) vs FAR (dashed)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Airflow Components - Comparison
        ax3 = fig.add_subplot(3, 3, row*3 + 3)
        # NEAR drone (thicker lines)
        ax3.plot(times, airflows_near[:, 0], '-', color='red', label='X NEAR', linewidth=2.5, alpha=0.8)
        ax3.plot(times, airflows_near[:, 2], '-', color='blue', label='Z NEAR (updraft)', linewidth=2.5, alpha=0.8)
        # FAR drone (thinner, dashed lines)
        ax3.plot(times, airflows_far[:, 0], '--', color='pink', label='X FAR', linewidth=1.5, alpha=0.6)
        ax3.plot(times, airflows_far[:, 2], '--', color='cyan', label='Z FAR (updraft)', linewidth=1.5, alpha=0.6)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Airflow (m/s)', fontsize=10)
        ax3.set_title(f'{title}\nAirflow: NEAR vs FAR', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Add annotation comparing the two
        avg_updraft_near = np.mean(airflows_near[:, 2])
        avg_updraft_far = np.mean(airflows_far[:, 2])
        ratio = avg_updraft_near / avg_updraft_far if avg_updraft_far > 0.01 else 0
        
        comparison_text = f"NEAR: {avg_updraft_near:.2f} m/s\nFAR: {avg_updraft_far:.2f} m/s\n{ratio:.1f}x stronger!"
        ax3.text(0.98, 0.98, comparison_text, transform=ax3.transAxes,
                fontsize=9, fontweight='bold', color='darkblue',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Drone Hovering Test: Distance Comparison (FAR=10m vs NEAR=5m from fire)\nFire at (0,0), Zero global wind',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive visualization: {filename}")
    plt.close()


def plot_side_view_comparison(trajectories, filename='drone_altitude_sideview.png'):
    """
    Create side-view comparison showing all three scenarios together.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scenarios = [
        ('Low Altitude (10m)', trajectories['low'], 'red'),
        ('Mid Altitude (25m)', trajectories['mid'], 'orange'),
        ('High Altitude (40m)', trajectories['high'], 'green')
    ]
    
    for ax_idx, (title, traj, color) in enumerate(scenarios):
        ax = axes[ax_idx]
        positions = np.array(traj['positions'])
        airflows = np.array(traj['airflows'])
        
        # Draw fire
        fire_rect = Rectangle((-1, 0), 2, 2, color='red', alpha=0.7, label='Fire')
        ax.add_patch(fire_rect)
        
        # Draw convection column
        conv_height = 50
        plume_width_ground = 2
        plume_width_top = 8
        x_plume = [-plume_width_ground, -plume_width_top, plume_width_top, plume_width_ground, -plume_width_ground]
        y_plume = [0, conv_height, conv_height, 0, 0]
        ax.fill(x_plume, y_plume, color='orange', alpha=0.15, label='Convection column')
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 2], color=color, linewidth=3, 
               label='Drone path', marker='o', markersize=3, markevery=10)
        
        # Start and end markers
        ax.scatter([positions[0, 0]], [positions[0, 2]], c='blue', s=200, marker='o',
                  edgecolors='darkblue', linewidths=2, label='Start', zorder=5)
        ax.scatter([positions[-1, 0]], [positions[-1, 2]], c='purple', s=200, marker='X',
                  edgecolors='darkviolet', linewidths=2, label='End', zorder=5)
        
        # Draw airflow vectors at key points
        sample_indices = np.linspace(0, len(positions)-1, 8, dtype=int)
        for i in sample_indices:
            pos = positions[i]
            airflow = airflows[i]
            
            # Scale airflow for visibility
            scale = 1.5
            dx = airflow[0] * scale
            dz = airflow[2] * scale
            
            # Determine arrow color based on direction
            if dz > 0.5:  # Strong updraft
                arrow_color = 'red'
                alpha = 0.8
            elif dx < -0.1:  # Inward
                arrow_color = 'blue'
                alpha = 0.6
            elif dx > 0.1:  # Outward
                arrow_color = 'green'
                alpha = 0.6
            else:
                arrow_color = 'gray'
                alpha = 0.4
            
            ax.arrow(pos[0], pos[2], dx, dz,
                    head_width=0.8, head_length=0.5, fc=arrow_color, ec=arrow_color,
                    alpha=alpha, linewidth=1.5, zorder=3)
        
        # Add altitude reference lines
        ax.axhline(y=25, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(14, 25, '50% height', fontsize=8, alpha=0.7)
        
        # Labels and formatting
        ax.set_xlabel('Horizontal Distance from Fire (m)', fontsize=11)
        ax.set_ylabel('Altitude (m)', fontsize=11)
        ax.set_title(f'{title}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-10, 15])
        ax.set_ylim([0, 50])
        
        # Add text annotation
        displacement = positions[-1] - positions[0]
        text = f"Displacement:\n"
        text += f"Horiz: {displacement[0]:+.1f}m\n"
        text += f"Vert: {displacement[2]:+.1f}m"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Side View: Drone Hovering at Different Altitudes\nArrows show airflow direction and magnitude',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved side-view comparison: {filename}")
    plt.close()


def plot_step_by_step_snapshots(trajectory, altitude, filename='drone_snapshots.png'):
    """
    Create step-by-step snapshots showing drone position at different time points.
    """
    positions = np.array(trajectory['positions'])
    airflows = np.array(trajectory['airflows'])
    times = np.array(trajectory['times'])
    
    # Select 6 time points
    snapshot_times = [0, 2, 4, 6, 8, 10]
    snapshot_indices = [np.argmin(np.abs(times - t)) for t in snapshot_times]
    
    fig = plt.figure(figsize=(18, 10))
    
    for plot_idx, (snap_time, snap_idx) in enumerate(zip(snapshot_times, snapshot_indices)):
        # 3D view
        ax = fig.add_subplot(2, 3, plot_idx + 1, projection='3d')
        
        # Fire
        ax.scatter([0], [0], [0], c='red', s=300, marker='*', 
                  label='Fire', edgecolors='darkred', linewidths=2, alpha=0.8)
        
        # Trajectory up to this point
        ax.plot(positions[:snap_idx+1, 0], positions[:snap_idx+1, 1], positions[:snap_idx+1, 2],
               'b-', linewidth=2, alpha=0.5, label='Path so far')
        
        # Current drone position
        curr_pos = positions[snap_idx]
        ax.scatter([curr_pos[0]], [curr_pos[1]], [curr_pos[2]], 
                  c='blue', s=200, marker='o', label='Drone', 
                  edgecolors='darkblue', linewidths=2)
        
        # Current airflow
        curr_airflow = airflows[snap_idx] * 1.0  # Scale for visibility
        arrow = Arrow3D([curr_pos[0], curr_pos[0] + curr_airflow[0]],
                       [curr_pos[1], curr_pos[1] + curr_airflow[1]],
                       [curr_pos[2], curr_pos[2] + curr_airflow[2]],
                       mutation_scale=20, lw=3, arrowstyle='->', color='red', alpha=0.8)
        ax.add_artist(arrow)
        
        # Add text showing airflow values
        airflow_text = f"Airflow:\n"
        airflow_text += f"X: {curr_airflow[0]:+.2f} m/s\n"
        airflow_text += f"Y: {curr_airflow[1]:+.2f} m/s\n"
        airflow_text += f"Z: {curr_airflow[2]:+.2f} m/s"
        
        ax.text2D(0.05, 0.95, airflow_text, transform=ax.transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(f'T = {snap_time:.1f}s\nPos: ({curr_pos[0]:.1f}, {curr_pos[1]:.1f}, {curr_pos[2]:.1f})',
                    fontsize=11, fontweight='bold')
        ax.set_xlim([-5, 15])
        ax.set_ylim([-10, 10])
        ax.set_zlim([0, 50])
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle(f'Step-by-Step Snapshots: Hovering at {altitude}m Altitude\nRed arrow shows airflow affecting the drone',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved step-by-step snapshots: {filename}")
    plt.close()


def main():
    """Run all three scenarios and create visualizations."""
    print("="*70)
    print("DRONE ALTITUDE HOVERING TEST")
    print("="*70)
    print("\nThis test shows how fire convection affects a hovering drone")
    print("at three different altitudes:")
    print("  • LOW (10m):  Below transition height - inward pull expected")
    print("  • MID (25m):  At transition height - maximum updraft expected")
    print("  • HIGH (40m): Above transition height - outward push expected")
    print("="*70)
    
    # Run three scenarios
    trajectories = {}
    
    print("\n" + "="*70)
    print("RUNNING SCENARIO 1: LOW ALTITUDE")
    print("="*70)
    trajectories['low'] = run_hovering_test(altitude=10.0, duration=10.0)
    
    print("\n" + "="*70)
    print("RUNNING SCENARIO 2: MID ALTITUDE")
    print("="*70)
    trajectories['mid'] = run_hovering_test(altitude=25.0, duration=10.0)
    
    print("\n" + "="*70)
    print("RUNNING SCENARIO 3: HIGH ALTITUDE")
    print("="*70)
    trajectories['high'] = run_hovering_test(altitude=40.0, duration=10.0)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    plot_3d_trajectory(trajectories, 'output/drone_altitude_comparison.png')
    # TODO: Update these functions to handle new data structure
    # plot_side_view_comparison(trajectories, 'output/drone_altitude_sideview.png')
    # plot_step_by_step_snapshots(trajectories['mid'], 25.0, 'output/drone_mid_altitude_snapshots.png')
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON: FAR (10m) vs NEAR (5m)")
    print("="*70)
    
    for name, traj in trajectories.items():
        positions_far = np.array(traj['far']['positions'])
        airflows_far = np.array(traj['far']['airflows'])
        positions_near = np.array(traj['near']['positions'])
        airflows_near = np.array(traj['near']['airflows'])
        
        displacement_far = positions_far[-1] - positions_far[0]
        displacement_near = positions_near[-1] - positions_near[0]
        avg_airflow_far = np.mean(airflows_far, axis=0)
        avg_airflow_near = np.mean(airflows_near, axis=0)
        
        print(f"\n{name.upper()} ALTITUDE ({traj['far']['target_altitude']:.0f}m):")
        print(f"  FAR DRONE (10m from fire):")
        print(f"    Displacement: [{displacement_far[0]:+6.2f}, {displacement_far[1]:+6.2f}, {displacement_far[2]:+6.2f}] m")
        print(f"    Total drift:  {np.linalg.norm(displacement_far):.2f} m")
        print(f"    Avg airflow:  [{avg_airflow_far[0]:+6.2f}, {avg_airflow_far[1]:+6.2f}, {avg_airflow_far[2]:+6.2f}] m/s")
        
        print(f"  NEAR DRONE (5m from fire):")
        print(f"    Displacement: [{displacement_near[0]:+6.2f}, {displacement_near[1]:+6.2f}, {displacement_near[2]:+6.2f}] m")
        print(f"    Total drift:  {np.linalg.norm(displacement_near):.2f} m")
        print(f"    Avg airflow:  [{avg_airflow_near[0]:+6.2f}, {avg_airflow_near[1]:+6.2f}, {avg_airflow_near[2]:+6.2f}] m/s")
        
        # Determine behavior
        if avg_airflow_near[0] < -0.1:
            radial_behavior = "PULLED TOWARD FIRE ←"
        elif avg_airflow_near[0] > 0.1:
            radial_behavior = "PUSHED AWAY FROM FIRE →"
        else:
            radial_behavior = "NEUTRAL ≈"
        
        print(f"  NEAR Radial behavior: {radial_behavior}")
        print(f"  NEAR Vertical effect: {'STRONG UPDRAFT ↑↑' if avg_airflow_near[2] > 1.5 else 'MODERATE UPDRAFT ↑' if avg_airflow_near[2] > 0.5 else 'WEAK UPDRAFT'}")
        
        # Calculate ratios
        updraft_ratio = avg_airflow_near[2] / avg_airflow_far[2] if avg_airflow_far[2] > 0.01 else 0
        drift_ratio = np.linalg.norm(displacement_near) / np.linalg.norm(displacement_far) if np.linalg.norm(displacement_far) > 0.01 else 0
        
        print(f"  🔥 NEAR experiences {updraft_ratio:.2f}x stronger updraft than FAR")
        print(f"  🔥 NEAR drifts {drift_ratio:.2f}x more than FAR")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • output/drone_altitude_comparison.png - 3x3 FAR vs NEAR comparison")
    print("\nKey Finding:")
    print("  🔥 Drones CLOSER to fire experience MUCH STRONGER convection effects!")
    print("  🔥 Distance from fire matters as much as altitude!")
    print("\n")


if __name__ == "__main__":
    main()
