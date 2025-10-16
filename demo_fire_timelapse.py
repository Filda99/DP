#!/usr/bin/env python3
"""
Fire Spread Time-lapse Demonstration

Creates a series of images showing fire spread over time.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def save_fire_snapshot(sim, step_num, output_dir='output/timelapse'):
    """Save a snapshot of current fire state."""
    os.makedirs(output_dir, exist_ok=True)
    
    fire_state = sim.environment.get_fire_state()
    if not fire_state:
        return
    
    final_state = fire_state['fire_grid_state']
    H, W = final_state['B'].shape
    
    # Get grid bounds
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    
    # Create composite image
    img = np.zeros((H, W, 3))
    
    # Unburned fuel as green
    unburned_mask = (final_state['F'] > 0.5)
    img[unburned_mask] = [0.0, 0.5, 0.0]  # Dark green
    
    # Partially burned as yellow/orange
    partial_mask = (final_state['F'] > 0.1) & (final_state['F'] <= 0.5)
    fuel_ratio = final_state['F'][partial_mask]
    img[partial_mask, 0] = 1.0
    img[partial_mask, 1] = fuel_ratio
    img[partial_mask, 2] = 0.0
    
    # Completely burned as dark brown/black
    burned_mask = (final_state['F'] <= 0.1) & (~final_state['B'])
    img[burned_mask] = [0.2, 0.1, 0.0]
    
    # Currently burning as bright red
    burning_mask = final_state['B']
    img[burning_mask] = [1.0, 0.0, 0.0]
    
    # No fuel (water, buildings) as light blue/gray
    no_fuel_mask = (final_state['F'] == 0) & (~final_state['B'])
    for i in range(H):
        for j in range(W):
            if no_fuel_mask[i, j]:
                world_pos = sim.environment.grid_mapper.cell_to_world(i, j)
                is_lake = False
                for zone in sim.environment.terrain_zones:
                    if zone['type'] == 'lake':
                        center = zone['center']
                        radius = zone['radius']
                        dist = np.sqrt((world_pos[0] - center[0])**2 + (world_pos[1] - center[1])**2)
                        if dist <= radius:
                            img[i, j] = [0.3, 0.5, 0.9]
                            is_lake = True
                            break
                if not is_lake:
                    img[i, j] = [0.6, 0.6, 0.6]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    im = ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower', interpolation='nearest')
    
    # Add fire start markers - get from environment
    fire_starts = [(-15, -10), (10, -15)]
    for x, y in fire_starts:
        ax.plot(x, y, 'w*', markersize=20, markeredgecolor='red', markeredgewidth=3)
    
    # Add terrain features
    for zone in sim.environment.terrain_zones:
        if zone['type'] == 'forest':
            circle = plt.Circle(zone['center'], zone['radius'], 
                              fill=False, edgecolor='green', linewidth=3, linestyle='--')
            ax.add_patch(circle)
        elif zone['type'] == 'lake':
            circle = plt.Circle(zone['center'], zone['radius'],
                              fill=False, edgecolor='blue', linewidth=3, linestyle='--')
            ax.add_patch(circle)
    
    # Stats
    stats = fire_state['fire_stats']
    time_sec = step_num * 0.1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'Fire Spread - Step {step_num} ({time_sec:.1f}s)\n'
                 f'Burning: {stats["burning_cells"]} cells | '
                 f'Fuel: {stats["total_fuel"]:.1f} | '
                 f'Burned: {stats["burn_percentage"]:.1f}%',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/step_{step_num:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_fire_timelapse():
    """Run fire simulation and save snapshots."""
    
    print("=" * 70)
    print("🔥 FIRE SPREAD TIME-LAPSE")
    print("=" * 70)
    print("\nCreating snapshots of fire spread every 6 seconds...")
    print()
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    try:
        # Setup environment with CUSTOM larger forest
        print("📍 Setting up environment...")
        
        # Create a BIG forest in the center instead of multiple small ones
        sim.environment.add_forest_area([0, 0], radius=25, tree_count=40)
        
        # Add a lake on one side to demonstrate fire blocking
        sim.environment.add_lake([-30, -30], radius=12)
        
        # Add a few buildings
        sim.environment.add_city_block([-20, 20], [8, 8, 15])
        sim.environment.add_city_block([20, 20], [8, 8, 15])
        
        print(f"✅ Custom environment: 1 large forest (r=25m), 1 lake, 2 buildings")
        
        # Enable fire simulation
        print("🔥 Enabling fire simulation...")
        # Larger cells (5m instead of 2.5m) = fewer cells = slower overall burn time
        sim.enable_fire_simulation(grid_width_m=100, grid_height_m=100, cell_size_m=5.0)
        
        # Adjust fire spread parameters - SLOWER spread for better visualization
        if sim.environment.fire_enabled:
            # Slower base spread rate (1.5x instead of 2x)
            sim.environment.fire_grid.l_base *= 1.5
            # Moderate distance decay (fire spreads moderately far)
            sim.environment.fire_grid.alpha = 0.3
            # Moderate wind influence
            sim.environment.fire_grid.k_wind = 1.0
            print("   🔥 Fire parameters: SLOWER spread (1.5x base rate, alpha=0.3)")
            print(f"   Grid: {sim.environment.grid_mapper.grid_height_cells}x{sim.environment.grid_mapper.grid_width_cells} cells")
        
        # Set wind
        print("💨 Setting wind conditions...")
        sim.set_wind([8.0, 5.0, 0.0], turbulence=0.3)
        
        # Start fires
        print("\n🔥 Starting fires...")
        fires = [
            (-15, -10, "Southwest part of forest (will spread with wind)"),
            (10, -15, "South part of forest (near center)"),
        ]
        
        for x, y, desc in fires:
            if sim.start_fire((x, y), intensity=0.5):
                i, j = sim.environment.grid_mapper.world_to_cell((x, y))
                print(f"   ✓ Fire at ({x:3d}, {y:3d}) → cell [{i:2d},{j:2d}] - {desc}")
        
        # Check initial fire state
        fire_state = sim.environment.get_fire_state()
        if fire_state:
            B_initial = fire_state['fire_grid_state']['B']
            num_burning = np.sum(B_initial)
            print(f"\n🔍 DEBUG: Immediately after starting fires: {num_burning} cells burning")
            burning_cells = np.argwhere(B_initial)
            print(f"   Burning cell indices (i, j):")
            for i, j in burning_cells[:10]:  # Show first 10
                x, y = sim.environment.grid_mapper.cell_to_world(i, j)
                print(f"      [{i:2d},{j:2d}] → world ({x:6.1f}, {y:6.1f})")
        
        # Save initial state BEFORE any simulation steps
        print("\n📸 Saving snapshots...")
        save_fire_snapshot(sim, 0)
        print(f"   Saved: step_0000.png (t=0.0s) - BEFORE any simulation steps")
        
        # Run simulation and save snapshots every 5 steps (0.5 seconds) for first 30 seconds
        total_steps = 300  # 30 seconds
        snapshot_interval = 5  # Every 0.5 seconds
        
        for step in range(1, total_steps + 1):
            # Step simulation (no drones)
            sim.step_simulation({})
            
            # Save snapshot at intervals
            if step % snapshot_interval == 0:
                save_fire_snapshot(sim, step)
                t = step * 0.1
                stats = sim.get_simulation_summary()['fire']
                print(f"   Saved: step_{step:04d}.png (t={t:.1f}s) - "
                      f"Burning: {stats['burning_cells']:3d} cells, "
                      f"Fuel: {stats['total_fuel']:6.1f}")
        
        print("\n✅ Time-lapse complete!")
        print(f"📁 {(total_steps // snapshot_interval) + 1} images saved to output/timelapse/")
        print("\n💡 View images in sequence to see fire spread animation")
        
    finally:
        sim.stop_simulation()


if __name__ == '__main__':
    run_fire_timelapse()
