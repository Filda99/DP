"""
Dual Drone Comparison with Fire Spread Tracking
Shows WHERE the fire spreads and how it affects each drone.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def run_dual_drone_with_fire_tracking(altitude, duration=30.0):
    """
    Run test tracking fire spread relative to drone positions.
    """
    print(f"\n{'='*70}")
    print(f"DUAL DRONE + FIRE TRACKING: Altitude {altitude}m")
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
    sim.environment.fire_grid.l_base *= 2.0
    
    sim.fire_enabled = True
    sim.grid_mapper = sim.environment.grid_mapper
    sim.fire_grid = sim.environment.fire_grid
    
    # Minimal wind
    sim.weather = {'wind_velocity': np.array([1.0, 0.0, 0.0])}
    
    # Start fire at origin
    fire_positions = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    print(f"🔥 Starting fire cluster at origin (0, 0)...")
    for pos in fire_positions:
        sim.environment.start_fire_at_position(pos, intensity=0.8)
    
    # Add TWO drones
    near_pos = [1.0, 0.0, altitude]   # Almost at fire
    far_pos = [10.0, 0.0, altitude]   # 10m away
    
    drone_near = sim.add_quadcopter("near_fire", position=near_pos, mass=0.5)
    drone_far = sim.add_quadcopter("far_from_fire", position=far_pos, mass=0.5)
    
    print(f"✅ NEAR drone at: ({near_pos[0]:.1f}m, {near_pos[1]:.1f}m, {altitude}m)")
    print(f"✅ FAR drone at:  ({far_pos[0]:.1f}m, {far_pos[1]:.1f}m, {altitude}m)")
    print(f"\n🔄 Running {duration}s simulation...\n")
    
    # Storage
    trajectory = {
        'times': [],
        'near': {
            'positions': [],
            'airflows': [],
            'distances_to_nearest_fire': [],
            'start_pos': np.array(near_pos)
        },
        'far': {
            'positions': [],
            'airflows': [],
            'distances_to_nearest_fire': [],
            'start_pos': np.array(far_pos)
        },
        'fire_snapshots': [],  # Store fire grids at intervals
        'altitude': altitude
    }
    
    # Run simulation
    steps = int(duration / 0.1)
    snapshot_interval = int(5.0 / 0.1)  # Every 5 seconds
    
    for step in range(steps):
        t = step * 0.1
        
        # Get drone states
        pos_near = drone_near.get_position()
        pos_far = drone_far.get_position()
        vel_near = drone_near.get_velocity()
        vel_far = drone_far.get_velocity()
        airflow_near = sim.get_local_airflow(pos_near)
        airflow_far = sim.get_local_airflow(pos_far)
        
        # Calculate distance to nearest fire for each drone
        fire_grid = sim.fire_grid.I  # Intensity grid
        burning_cells = np.argwhere(fire_grid > 0.1)
        
        if len(burning_cells) > 0:
            # Convert cell indices to world coordinates
            fire_world_positions = []
            for cell_idx in burning_cells:
                row, col = cell_idx
                world_pos = sim.grid_mapper.cell_to_world(row, col)
                fire_world_positions.append(world_pos)
            fire_world_positions = np.array(fire_world_positions)
            
            # Distance from each drone to nearest fire cell
            dist_near = np.min([np.linalg.norm(pos_near[:2] - fp) for fp in fire_world_positions])
            dist_far = np.min([np.linalg.norm(pos_far[:2] - fp) for fp in fire_world_positions])
        else:
            dist_near = 999
            dist_far = 999
        
        # Store
        trajectory['times'].append(t)
        trajectory['near']['positions'].append(pos_near.copy())
        trajectory['near']['airflows'].append(airflow_near.copy())
        trajectory['near']['distances_to_nearest_fire'].append(dist_near)
        trajectory['far']['positions'].append(pos_far.copy())
        trajectory['far']['airflows'].append(airflow_far.copy())
        trajectory['far']['distances_to_nearest_fire'].append(dist_far)
        
        # Save fire grid snapshot
        if step % snapshot_interval == 0:
            trajectory['fire_snapshots'].append({
                'time': t,
                'grid': fire_grid.copy(),
                'burning_cells': len(burning_cells)
            })
        
        # Weak hovering controllers
        def get_control(pos, vel, target_pos):
            pos_error = target_pos - pos
            vel_error = -vel
            control = 0.5 * pos_error + 0.3 * vel_error
            return np.clip(control / 5.0, -1.0, 1.0)
        
        control_near = get_control(pos_near, vel_near, near_pos)
        control_far = get_control(pos_far, vel_far, far_pos)
        
        # Step
        controls = {
            "near_fire": control_near,
            "far_from_fire": control_far
        }
        sim.step_simulation(controls)
        
        # Progress every 10s
        if int(t) % 10 == 0 and step > 0:
            print(f"[{int(t):2d}s] Fire: {len(burning_cells):3d} cells | "
                  f"NEAR: dist={dist_near:.1f}m, updraft={airflow_near[2]:.2f} m/s | "
                  f"FAR: dist={dist_far:.1f}m, updraft={airflow_far[2]:.2f} m/s")
    
    # Summary
    avg_dist_near = np.mean(trajectory['near']['distances_to_nearest_fire'])
    avg_dist_far = np.mean(trajectory['far']['distances_to_nearest_fire'])
    min_dist_near = np.min(trajectory['near']['distances_to_nearest_fire'])
    min_dist_far = np.min(trajectory['far']['distances_to_nearest_fire'])
    
    print(f"\n📊 DISTANCE TO FIRE SUMMARY:")
    print(f"   NEAR drone:")
    print(f"     Average distance to nearest fire: {avg_dist_near:.1f}m")
    print(f"     Minimum distance to fire: {min_dist_near:.1f}m")
    print(f"   FAR drone:")
    print(f"     Average distance to nearest fire: {avg_dist_far:.1f}m")
    print(f"     Minimum distance to fire: {min_dist_far:.1f}m")
    
    if min_dist_far < 5.0:
        print(f"\n   ⚠️  WARNING: Fire spread VERY CLOSE to FAR drone (within 5m)!")
        print(f"       This explains why both drones experienced similar effects!")
    
    sim.stop_simulation()
    return trajectory


def plot_fire_spread_analysis(trajectories, filename='output/fire_spread_analysis.png'):
    """
    Visualize where fire spread relative to drones.
    """
    fig = plt.figure(figsize=(20, 15))
    
    scenarios = [
        ('Low Altitude (10m)', trajectories['low'], 'red'),
        ('Mid Altitude (25m)', trajectories['mid'], 'orange'),
        ('High Altitude (40m)', trajectories['high'], 'green')
    ]
    
    for row, (title, traj, color) in enumerate(scenarios):
        # Plot 1: Fire spread map (top view)
        ax1 = fig.add_subplot(3, 4, row*4 + 1)
        
        # Show fire at different times
        snapshots = traj['fire_snapshots']
        for i, snapshot in enumerate([snapshots[0], snapshots[len(snapshots)//2], snapshots[-1]]):
            grid = snapshot['grid']
            time = snapshot['time']
            burning_cells = np.argwhere(grid > 0.1)
            
            if len(burning_cells) > 0:
                fire_world = []
                for cell_idx in burning_cells:
                    row_idx, col_idx = cell_idx
                    # Approximate world position
                    x = (col_idx - 25) * 2.0
                    y = (row_idx - 25) * 2.0
                    fire_world.append([x, y])
                fire_world = np.array(fire_world)
                
                alpha = 0.3 + i * 0.3
                size = 20 + i * 30
                label = f't={int(time)}s ({len(burning_cells)} cells)'
                ax1.scatter(fire_world[:, 0], fire_world[:, 1], 
                           c='red', s=size, alpha=alpha, label=label)
        
        # Drone positions
        near_pos = traj['near']['start_pos']
        far_pos = traj['far']['start_pos']
        ax1.scatter([near_pos[0]], [near_pos[1]], c='blue', s=500, marker='D', 
                   edgecolors='darkblue', linewidths=3, label='NEAR drone', zorder=10)
        ax1.scatter([far_pos[0]], [far_pos[1]], c='cyan', s=500, marker='D', 
                   edgecolors='darkcyan', linewidths=3, label='FAR drone', zorder=10)
        
        # Circles showing initial distance
        circle_near = Circle((0, 0), 1.0, fill=False, edgecolor='blue', 
                            linestyle='--', linewidth=2, alpha=0.5)
        circle_far = Circle((0, 0), 10.0, fill=False, edgecolor='cyan', 
                           linestyle='--', linewidth=2, alpha=0.5)
        ax1.add_patch(circle_near)
        ax1.add_patch(circle_far)
        
        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_title(f'{title}\nFire Spread Pattern (Top View)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-30, 30])
        ax1.set_ylim([-30, 30])
        ax1.set_aspect('equal')
        
        # Plot 2: Distance to fire over time
        ax2 = fig.add_subplot(3, 4, row*4 + 2)
        times = np.array(traj['times'])
        dist_near = np.array(traj['near']['distances_to_nearest_fire'])
        dist_far = np.array(traj['far']['distances_to_nearest_fire'])
        
        ax2.plot(times, dist_near, 'b-', linewidth=3, label='NEAR drone')
        ax2.plot(times, dist_far, 'c--', linewidth=3, label='FAR drone')
        ax2.fill_between(times, dist_near, alpha=0.2, color='blue')
        ax2.fill_between(times, dist_far, alpha=0.2, color='cyan')
        
        # Mark when fire gets close
        close_threshold = 5.0
        ax2.axhline(y=close_threshold, color='red', linestyle=':', linewidth=2, 
                   alpha=0.7, label=f'Close fire ({close_threshold}m)')
        
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Distance to Nearest Fire (m)', fontsize=11)
        ax2.set_title(f'{title}\nHow Close Did Fire Get?', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Add stats
        min_near = dist_near.min()
        min_far = dist_far.min()
        avg_near = dist_near.mean()
        avg_far = dist_far.mean()
        
        stats_text = f'Min distance:\nNEAR: {min_near:.1f}m\nFAR: {min_far:.1f}m\n\nAvg:\nNEAR: {avg_near:.1f}m\nFAR: {avg_far:.1f}m'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Plot 3: Updraft vs Distance correlation
        ax3 = fig.add_subplot(3, 4, row*4 + 3)
        airflows_near = np.array(traj['near']['airflows'])
        airflows_far = np.array(traj['far']['airflows'])
        
        # Scatter plot showing correlation
        ax3.scatter(dist_near, airflows_near[:, 2], c='blue', s=10, alpha=0.5, label='NEAR')
        ax3.scatter(dist_far, airflows_far[:, 2], c='cyan', s=10, alpha=0.5, label='FAR')
        
        ax3.set_xlabel('Distance to Fire (m)', fontsize=11)
        ax3.set_ylabel('Vertical Airflow (m/s)', fontsize=11)
        ax3.set_title(f'{title}\nUpdraft vs Distance Correlation', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Add trend annotation
        if dist_far.max() < 8:
            ax3.text(0.5, 0.95, '⚠️ Fire spread close to FAR drone!\nBoth drones in same plume',
                    transform=ax3.transAxes, fontsize=10, va='top', ha='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # Plot 4: Fire cells growth
        ax4 = fig.add_subplot(3, 4, row*4 + 4)
        snapshot_times = [s['time'] for s in traj['fire_snapshots']]
        snapshot_cells = [s['burning_cells'] for s in traj['fire_snapshots']]
        
        ax4.plot(snapshot_times, snapshot_cells, 'r-', linewidth=3, marker='o', markersize=8)
        ax4.fill_between(snapshot_times, snapshot_cells, alpha=0.3, color='red')
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Burning Cells', fontsize=11)
        ax4.set_title(f'{title}\nFire Growth', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        growth = snapshot_cells[-1] - snapshot_cells[0]
        ax4.text(0.95, 0.95, f'Growth:\n+{growth} cells',
                transform=ax4.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    plt.suptitle('Fire Spread Analysis: Did Fire Reach the FAR Drone?\nShows fire pattern, distance tracking, and correlation with airflow effects',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {filename}")
    plt.close()


def main():
    """Run analysis with fire spread tracking."""
    print("="*70)
    print("DUAL DRONE TEST WITH FIRE SPREAD TRACKING")
    print("="*70)
    print("\nThis test will show WHERE the fire spreads and how it affects")
    print("the distance between each drone and the nearest fire!")
    print("="*70)
    
    trajectories = {}
    
    print("\n🔥 TEST 1: LOW ALTITUDE (10m)")
    trajectories['low'] = run_dual_drone_with_fire_tracking(altitude=10.0, duration=30.0)
    
    print("\n🔥 TEST 2: MID ALTITUDE (25m)")
    trajectories['mid'] = run_dual_drone_with_fire_tracking(altitude=25.0, duration=30.0)
    
    print("\n🔥 TEST 3: HIGH ALTITUDE (40m)")
    trajectories['high'] = run_dual_drone_with_fire_tracking(altitude=40.0, duration=30.0)
    
    # Create visualization
    print("\n" + "="*70)
    print("CREATING FIRE SPREAD VISUALIZATION")
    print("="*70)
    
    plot_fire_spread_analysis(trajectories)
    
    # Final analysis
    print("\n" + "="*70)
    print("FIRE SPREAD IMPACT ANALYSIS")
    print("="*70)
    
    for name, traj in [('LOW (10m)', trajectories['low']), 
                       ('MID (25m)', trajectories['mid']), 
                       ('HIGH (40m)', trajectories['high'])]:
        
        dist_near = np.array(traj['near']['distances_to_nearest_fire'])
        dist_far = np.array(traj['far']['distances_to_nearest_fire'])
        
        print(f"\n{name}:")
        print(f"  NEAR drone:")
        print(f"    Min distance to fire: {dist_near.min():.1f}m")
        print(f"    Avg distance to fire: {dist_near.mean():.1f}m")
        print(f"  FAR drone:")
        print(f"    Min distance to fire: {dist_far.min():.1f}m")
        print(f"    Avg distance to fire: {dist_far.mean():.1f}m")
        
        if dist_far.min() < 5.0:
            print(f"  🔥 Fire spread VERY CLOSE to FAR drone!")
            print(f"  ⚠️  This explains similar effects - both drones in same plume!")
        elif dist_far.min() < 10.0:
            print(f"  ⚠️  Fire spread moderately close to FAR drone")
        else:
            print(f"  ✅ Fire stayed distant from FAR drone")
    
    print("\n" + "="*70)
    print("✅ FIRE SPREAD ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated: output/fire_spread_analysis.png")
    print("This shows the exact fire spread pattern and drone-fire distances!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
