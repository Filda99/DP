#!/usr/bin/env python3
"""
Demo 2: Water-Based Fire Suppression

Demonstrates the new moisture-based suppression model:
- Drones drop water on fires
- Water increases terrain moisture
- Moisture prevents re-ignition
- Moisture evaporates over time
- Immediate intensity reduction
- Probability of instant extinguishment
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def run_water_suppression_demo():
    """Run water suppression demonstration."""
    print("=" * 70)
    print("💧 DEMO 2: WATER-BASED FIRE SUPPRESSION")
    print("=" * 70)
    print()
    print("Demonstrating:")
    print("  • Water drops increase terrain moisture")
    print("  • Moisture prevents fire re-ignition")
    print("  • Moisture evaporates over time")
    print("  • Immediate fire intensity reduction")
    print()
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Setup simpler environment
    sim.environment.add_forest_zone(center=[0, 0], radius=40, density=1.0)
    
    # Enable fire
    sim.enable_fire_simulation(
        grid_width_m=80,
        grid_height_m=80,
        cell_size_m=2.0
    )
    
    # Light wind
    sim.set_wind([3.0, 0.0, 0.0])
    
    # Start fire in center
    sim.start_fire((0, 0), intensity=0.4)
    print("  🔥 Fire started at (0, 0)")
    
    # Add firefighting drone
    sim.add_quadcopter("Firefighter", position=[0, 0, 10])
    print("  🚁 Drone deployed at low altitude for water drops")
    
    print()
    print("Running simulation...")
    print("  Phase 1 (0-20s): Fire spreads freely")
    print("  Phase 2 (20-60s): Drone fights fire with water")
    print()
    
    # Track statistics
    stats_time = []
    stats_burning = []
    stats_moisture = []
    
    snapshots = []
    snapshot_times = [0, 20, 35, 50, 65]  # Before and during firefighting
    
    for step in range(int(70 / sim.timestep)):
        current_time = sim.simulation_time
        
        # Control strategy
        if current_time < 20:
            # Phase 1: Let fire spread, drone stays away
            controls = {
                "Firefighter": np.array([0, 0, 0])  # Hover
            }
        else:
            # Phase 2: Aggressive firefighting
            # Drone patrols over fire area
            phase = (current_time - 20) / 10
            x = 10 * np.cos(phase * np.pi)
            y = 10 * np.sin(phase * np.pi)
            
            drone_pos = sim.drones["Firefighter"].get_position()
            target = np.array([x, y, 8])  # Low altitude for effective drops
            direction = target - drone_pos
            
            # Simple proportional control
            controls = {
                "Firefighter": direction * 0.5
            }
        
        sim.step_simulation(controls)
        
        # Collect statistics
        if step % 24 == 0:  # Every 0.1 seconds
            fire_state = sim.environment.get_fire_state()
            if fire_state:
                state = fire_state['fire_grid_state']
                stats_time.append(current_time)
                stats_burning.append(np.sum(state['B']))
                stats_moisture.append(np.mean(state['M']))
        
        # Snapshots
        if any(abs(current_time - t) < sim.timestep for t in snapshot_times):
            fire_state = sim.environment.get_fire_state()
            if fire_state:
                snapshots.append({
                    'time': current_time,
                    'state': fire_state['fire_grid_state'].copy(),
                    'drone_pos': sim.drones["Firefighter"].get_position().copy()
                })
                print(f"  📸 Snapshot at t={current_time:.1f}s - " +
                      f"Burning: {np.sum(fire_state['fire_grid_state']['B'])} cells")
    
    # Create visualizations
    create_suppression_visualization(sim, snapshots)
    create_stats_plot(stats_time, stats_burning, stats_moisture)
    
    print()
    print("✅ Demo complete! Check output/demo_02_*.png")
    print("=" * 70)


def create_suppression_visualization(sim, snapshots):
    """Create visualization showing suppression process."""
    os.makedirs('output', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    
    for idx, snapshot in enumerate(snapshots[:6]):
        ax = axes[idx]
        state = snapshot['state']
        
        H, W = state['B'].shape
        
        # Create visualization with moisture overlay
        img = np.zeros((H, W, 3))
        
        # Base: green forest
        img[:, :] = [0.1, 0.5, 0.1]
        
        # Blue tint for moisture
        moisture = state['M']
        img[:, :, 2] += moisture * 0.6  # Add blue
        img[:, :, 0] -= moisture * 0.05  # Reduce red
        img = np.clip(img, 0, 1)
        
        # Burned areas (darker)
        fuel = state['F']
        burn_factor = np.clip(fuel, 0, 1)[:, :, np.newaxis]
        img = img * (0.3 + 0.7 * burn_factor)
        
        # Active fires (bright)
        burning = state['B']
        intensity = state['I']
        img[burning] = np.stack([
            np.ones_like(intensity[burning]),
            intensity[burning] * 0.4,
            np.zeros_like(intensity[burning])
        ], axis=-1)
        
        ax.imshow(img, origin='lower', extent=[x_min, x_max, y_min, y_max])
        
        # Show drone position
        drone_pos = snapshot['drone_pos']
        ax.plot(drone_pos[0], drone_pos[1], 'b^', markersize=15, 
               markeredgecolor='white', markeredgewidth=2, label='Drone')
        
        # Draw water drop radius
        drop_radius = 5.0  # meters (from simulation)
        circle = plt.Circle((drone_pos[0], drone_pos[1]), drop_radius,
                           fill=False, edgecolor='cyan', linewidth=2,
                           linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        
        ax.set_title(f't = {snapshot["time"]:.1f}s | ' +
                    f'Burning: {np.sum(burning)} | ' +
                    f'Avg Moisture: {np.mean(moisture):.2f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right')
    
    plt.suptitle('Water-Based Fire Suppression (Blue tint = Moisture)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/demo_02_suppression.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_stats_plot(time, burning, moisture):
    """Create statistics plot showing effectiveness."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Burning cells over time
    ax1.plot(time, burning, 'r-', linewidth=2, label='Burning cells')
    ax1.axvline(x=20, color='blue', linestyle='--', linewidth=2, 
               label='Firefighting starts', alpha=0.7)
    ax1.set_ylabel('Number of Burning Cells', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_title('Fire Suppression Effectiveness', fontsize=14, fontweight='bold')
    
    # Average moisture over time
    ax2.plot(time, moisture, 'b-', linewidth=2, label='Average moisture')
    ax2.axvline(x=20, color='blue', linestyle='--', linewidth=2, 
               label='Firefighting starts', alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Terrain Moisture', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('output/demo_02_stats.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_water_suppression_demo()
