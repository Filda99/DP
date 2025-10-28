"""
Dual Drone Comparison: Near vs Far from Growing Fire
Shows vertical profile comparison with side-by-side drones.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import proj3d
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


class Arrow3D:
    """Helper for 3D arrows"""
    pass


def run_dual_drone_test(altitude, duration=30.0):
    """
    Run test with TWO drones at same altitude:
    - Drone 1: Directly above fire (1m horizontal distance)
    - Drone 2: Farther away (10m horizontal distance)
    """
    print(f"\n{'='*70}")
    print(f"DUAL DRONE TEST: Altitude {altitude}m - Near vs Far")
    print(f"{'='*70}")
    
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Enable aggressive fire
    sim.environment.enable_fire_simulation(
        grid_width_m=100,
        grid_height_m=100,
        cell_size_m=2.0,
        dt=0.1,
        alpha=0.5,
        k_wind=2.0
    )
    sim.environment.fire_grid.l_base *= 2.0  # Aggressive spread
    
    sim.fire_enabled = True
    sim.grid_mapper = sim.environment.grid_mapper
    sim.fire_grid = sim.environment.fire_grid
    
    # Minimal wind
    sim.weather = {'wind_velocity': np.array([1.0, 0.0, 0.0])}
    
    # Start fire cluster
    fire_positions = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    print(f"🔥 Starting fire cluster...")
    for pos in fire_positions:
        sim.environment.start_fire_at_position(pos, intensity=0.8)
    
    # Add TWO drones
    near_pos = [1.0, 0.0, altitude]   # Almost directly above fire
    far_pos = [10.0, 0.0, altitude]   # 10m away
    
    drone_near = sim.add_quadcopter("near_fire", position=near_pos, mass=0.5)
    drone_far = sim.add_quadcopter("far_from_fire", position=far_pos, mass=0.5)
    
    print(f"✅ NEAR drone at: (1m, 0, {altitude}m) - almost above fire")
    print(f"✅ FAR drone at:  (10m, 0, {altitude}m) - distant")
    print(f"\n🔄 Running {duration}s with growing fire...\n")
    
    # Storage for both drones
    trajectory = {
        'times': [],
        'fire_cells': [],
        'near': {
            'positions': [],
            'velocities': [],
            'airflows': [],
            'start_pos': np.array(near_pos)
        },
        'far': {
            'positions': [],
            'velocities': [],
            'airflows': [],
            'start_pos': np.array(far_pos)
        },
        'altitude': altitude
    }
    
    # Run simulation
    steps = int(duration / 0.1)
    progress_marks = [0, 10, 20, 30]
    
    for step in range(steps):
        t = step * 0.1
        
        # Get states for both drones
        pos_near = drone_near.get_position()
        pos_far = drone_far.get_position()
        vel_near = drone_near.get_velocity()
        vel_far = drone_far.get_velocity()
        airflow_near = sim.get_local_airflow(pos_near)
        airflow_far = sim.get_local_airflow(pos_far)
        
        # Get fire size
        fire_state = sim.environment.get_fire_state()
        burning_cells = fire_state['fire_stats']['burning_cells'] if fire_state else 0
        
        # Store
        trajectory['times'].append(t)
        trajectory['fire_cells'].append(burning_cells)
        trajectory['near']['positions'].append(pos_near.copy())
        trajectory['near']['velocities'].append(vel_near.copy())
        trajectory['near']['airflows'].append(airflow_near.copy())
        trajectory['far']['positions'].append(pos_far.copy())
        trajectory['far']['velocities'].append(vel_far.copy())
        trajectory['far']['airflows'].append(airflow_far.copy())
        
        # Weak hovering controllers for both
        def get_control(pos, vel, target_pos):
            pos_error = target_pos - pos
            vel_error = -vel
            control = 0.5 * pos_error + 0.3 * vel_error
            return np.clip(control / 5.0, -1.0, 1.0)
        
        control_near = get_control(pos_near, vel_near, near_pos)
        control_far = get_control(pos_far, vel_far, far_pos)
        
        # Step simulation
        controls = {
            "near_fire": control_near,
            "far_from_fire": control_far
        }
        sim.step_simulation(controls)
        
        # Progress
        if int(t) in progress_marks:
            drift_near = np.linalg.norm(pos_near - near_pos)
            drift_far = np.linalg.norm(pos_far - far_pos)
            print(f"[{int(t):2d}s] Fire: {burning_cells:3d} cells | "
                  f"NEAR: {airflow_near[2]:5.2f} m/s, drift {drift_near:.2f}m | "
                  f"FAR: {airflow_far[2]:5.2f} m/s, drift {drift_far:.2f}m")
            progress_marks.remove(int(t))
    
    # Summary
    final_pos_near = trajectory['near']['positions'][-1]
    final_pos_far = trajectory['far']['positions'][-1]
    drift_near = np.linalg.norm(final_pos_near - near_pos)
    drift_far = np.linalg.norm(final_pos_far - far_pos)
    
    avg_airflow_near = np.mean(trajectory['near']['airflows'], axis=0)
    avg_airflow_far = np.mean(trajectory['far']['airflows'], axis=0)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   NEAR drone (1m from fire):")
    print(f"     Total drift: {drift_near:.2f}m")
    print(f"     Avg updraft: {avg_airflow_near[2]:.2f} m/s")
    print(f"   FAR drone (10m from fire):")
    print(f"     Total drift: {drift_far:.2f}m")
    print(f"     Avg updraft: {avg_airflow_far[2]:.2f} m/s")
    print(f"   🔥 NEAR experiences {avg_airflow_near[2]/max(0.01, avg_airflow_far[2]):.1f}x stronger updraft!")
    
    sim.stop_simulation()
    return trajectory


def plot_vertical_comparison(trajectories, filename='output/vertical_dual_drone_comparison.png'):
    """
    Create vertical comparison plots for near vs far drones.
    """
    fig = plt.figure(figsize=(20, 12))
    
    altitudes = ['Low (10m)', 'Mid (25m)', 'High (40m)']
    colors_near = ['darkred', 'darkorange', 'darkgreen']
    colors_far = ['lightcoral', 'gold', 'lightgreen']
    
    for row, (alt_name, traj, color_near, color_far) in enumerate(zip(altitudes, 
                                                                        [trajectories['low'], trajectories['mid'], trajectories['high']],
                                                                        colors_near, colors_far)):
        times = np.array(traj['times'])
        fire_cells = np.array(traj['fire_cells'])
        
        near_airflows = np.array(traj['near']['airflows'])
        far_airflows = np.array(traj['far']['airflows'])
        near_positions = np.array(traj['near']['positions'])
        far_positions = np.array(traj['far']['positions'])
        
        # Plot 1: Vertical Airflow Comparison
        ax1 = fig.add_subplot(3, 4, row*4 + 1)
        ax1.plot(times, near_airflows[:, 2], color=color_near, linewidth=3, 
                label='NEAR (1m from fire)', alpha=0.9)
        ax1.plot(times, far_airflows[:, 2], color=color_far, linewidth=3, 
                label='FAR (10m from fire)', alpha=0.9, linestyle='--')
        ax1.fill_between(times, near_airflows[:, 2], alpha=0.2, color=color_near)
        ax1.fill_between(times, far_airflows[:, 2], alpha=0.2, color=color_far)
        
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Vertical Airflow (m/s)', fontsize=11)
        ax1.set_title(f'{alt_name}\nUpdraft: Near (solid) vs Far (dashed)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add difference annotation
        avg_near = near_airflows[:, 2].mean()
        avg_far = far_airflows[:, 2].mean()
        ratio = avg_near / max(0.01, avg_far)
        ax1.text(0.98, 0.98, f'NEAR: {avg_near:.2f} m/s\nFAR: {avg_far:.2f} m/s\nRatio: {ratio:.1f}x',
                transform=ax1.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Plot 2: Altitude Deviation
        ax2 = fig.add_subplot(3, 4, row*4 + 2)
        target_alt = traj['altitude']
        near_alt_dev = near_positions[:, 2] - target_alt
        far_alt_dev = far_positions[:, 2] - target_alt
        
        ax2.plot(times, near_alt_dev, color=color_near, linewidth=2, label='NEAR drone')
        ax2.plot(times, far_alt_dev, color=color_far, linewidth=2, label='FAR drone', linestyle='--')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        ax2.fill_between(times, near_alt_dev, alpha=0.2, color=color_near)
        ax2.fill_between(times, far_alt_dev, alpha=0.2, color=color_far)
        
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Altitude Deviation (m)', fontsize=11)
        ax2.set_title(f'{alt_name}\nAltitude Control (0 = perfect)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Horizontal Drift
        ax3 = fig.add_subplot(3, 4, row*4 + 3)
        near_drift = [np.linalg.norm(p - traj['near']['start_pos']) for p in near_positions]
        far_drift = [np.linalg.norm(p - traj['far']['start_pos']) for p in far_positions]
        
        ax3.plot(times, near_drift, color=color_near, linewidth=3, label='NEAR drone')
        ax3.plot(times, far_drift, color=color_far, linewidth=3, label='FAR drone', linestyle='--')
        ax3.fill_between(times, near_drift, alpha=0.2, color=color_near)
        ax3.fill_between(times, far_drift, alpha=0.2, color=color_far)
        
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Total Drift (m)', fontsize=11)
        ax3.set_title(f'{alt_name}\nPosition Drift from Start', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Add final drift annotation
        ax3.text(0.02, 0.98, f'Final:\nNEAR: {near_drift[-1]:.2f}m\nFAR: {far_drift[-1]:.2f}m',
                transform=ax3.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Plot 4: Fire Growth (same for all, but shown per row)
        ax4 = fig.add_subplot(3, 4, row*4 + 4)
        ax4.plot(times, fire_cells, 'r-', linewidth=3)
        ax4.fill_between(times, fire_cells, alpha=0.3, color='red')
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Burning Cells', fontsize=11)
        ax4.set_title(f'{alt_name}\nFire Growth', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        growth = fire_cells[-1] - fire_cells[0]
        ax4.text(0.95, 0.95, f'Growth: +{growth}\n({fire_cells[-1]} cells)',
                transform=ax4.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    plt.suptitle('Dual Drone Comparison: NEAR (1m) vs FAR (10m) from Growing Fire\nSolid lines = NEAR drone (stronger effects) | Dashed lines = FAR drone (weaker effects)',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {filename}")
    plt.close()


def plot_side_view_dual_drones(trajectories, filename='output/side_view_dual_drones.png'):
    """
    Side-view showing both drones' paths through convection column.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    scenarios = [
        ('Low Altitude (10m)', trajectories['low'], 'red'),
        ('Mid Altitude (25m)', trajectories['mid'], 'orange'),
        ('High Altitude (40m)', trajectories['high'], 'green')
    ]
    
    for ax, (title, traj, color) in zip(axes, scenarios):
        near_pos = np.array(traj['near']['positions'])
        far_pos = np.array(traj['far']['positions'])
        
        # Fire and convection column
        fire_rect = Rectangle((-2, 0), 4, 2, color='red', alpha=0.7, label='Fire')
        ax.add_patch(fire_rect)
        
        conv_height = 50
        x_plume = [-2, -8, 8, 2, -2]
        y_plume = [0, conv_height, conv_height, 0, 0]
        ax.fill(x_plume, y_plume, color='orange', alpha=0.15, label='Convection')
        
        # Transition line
        ax.axhline(y=25, color='k', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(12, 25, '50% height', fontsize=9, alpha=0.6)
        
        # Drone paths
        ax.plot(near_pos[:, 0], near_pos[:, 2], color='darkred', linewidth=4, 
               label='NEAR drone (1m)', marker='o', markersize=4, markevery=30)
        ax.plot(far_pos[:, 0], far_pos[:, 2], color='darkblue', linewidth=4, 
               label='FAR drone (10m)', linestyle='--', marker='s', markersize=4, markevery=30)
        
        # Start/end markers
        ax.scatter([near_pos[0, 0]], [near_pos[0, 2]], c='green', s=250, marker='o',
                  edgecolors='darkgreen', linewidths=3, zorder=5, label='Start')
        ax.scatter([near_pos[-1, 0]], [near_pos[-1, 2]], c='purple', s=250, marker='X',
                  edgecolors='darkviolet', linewidths=3, zorder=5, label='End')
        
        ax.set_xlabel('Horizontal Distance from Fire (m)', fontsize=11)
        ax.set_ylabel('Altitude (m)', fontsize=11)
        ax.set_title(f'{title}\nRed=NEAR (strong effects) | Blue=FAR (weak effects)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-5, 15])
        ax.set_ylim([0, 50])
        
        # Drift statistics
        near_drift = np.linalg.norm(near_pos[-1] - near_pos[0])
        far_drift = np.linalg.norm(far_pos[-1] - far_pos[0])
        stats_text = f'Drift:\nNEAR: {near_drift:.2f}m\nFAR: {far_drift:.2f}m\nRatio: {near_drift/max(0.01,far_drift):.1f}x'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Side View: NEAR (1m) vs FAR (10m) Drone Comparison\nNEAR drone experiences much stronger fire effects',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def main():
    """Run dual drone tests at three altitudes."""
    print("="*70)
    print("DUAL DRONE TEST: Near vs Far from Fire")
    print("="*70)
    print("\nComparing two drones at same altitude:")
    print("  🔴 NEAR: 1m from fire (almost directly above)")
    print("  🔵 FAR:  10m from fire (distant)")
    print("\nBoth trying to hover, exposed to GROWING fire")
    print("="*70)
    
    trajectories = {}
    
    print("\n🔥 TEST 1: LOW ALTITUDE (10m)")
    trajectories['low'] = run_dual_drone_test(altitude=10.0, duration=30.0)
    
    print("\n🔥 TEST 2: MID ALTITUDE (25m)")
    trajectories['mid'] = run_dual_drone_test(altitude=25.0, duration=30.0)
    
    print("\n🔥 TEST 3: HIGH ALTITUDE (40m)")
    trajectories['high'] = run_dual_drone_test(altitude=40.0, duration=30.0)
    
    # Visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    plot_vertical_comparison(trajectories)
    plot_side_view_dual_drones(trajectories)
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY: Near vs Far Comparison")
    print("="*70)
    
    for name, traj in [('LOW (10m)', trajectories['low']), 
                       ('MID (25m)', trajectories['mid']), 
                       ('HIGH (40m)', trajectories['high'])]:
        near_airflow = np.array(traj['near']['airflows'])
        far_airflow = np.array(traj['far']['airflows'])
        near_pos = np.array(traj['near']['positions'])
        far_pos = np.array(traj['far']['positions'])
        
        near_updraft = near_airflow[:, 2].mean()
        far_updraft = far_airflow[:, 2].mean()
        
        near_drift = np.linalg.norm(near_pos[-1] - traj['near']['start_pos'])
        far_drift = np.linalg.norm(far_pos[-1] - traj['far']['start_pos'])
        
        print(f"\n{name}:")
        print(f"  NEAR (1m): Updraft {near_updraft:.2f} m/s, Drift {near_drift:.2f}m")
        print(f"  FAR (10m): Updraft {far_updraft:.2f} m/s, Drift {far_drift:.2f}m")
        print(f"  📊 NEAR has {near_updraft/max(0.01,far_updraft):.1f}x stronger updraft")
        print(f"  📊 NEAR drifts {near_drift/max(0.01,far_drift):.1f}x more")
    
    print("\n" + "="*70)
    print("✅ DUAL DRONE TEST COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • output/vertical_dual_drone_comparison.png - 3x4 grid comparison")
    print("  • output/side_view_dual_drones.png - Side-view paths")
    print("\n💡 Key Insight: Distance from fire MATTERS!")
    print("   Drones near fire experience MUCH stronger effects!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
