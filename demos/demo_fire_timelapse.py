#!/usr/bin/env python3
"""
Fire Spread Time-lapse Demonstr    # No fuel (water, buildings) as blue/black
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
                            img[i, j] = [0.3, 0.5, 0.9]  # Blue for water
                            is_lake = True
                            break
                if not is_lake:
                    img[i, j] = [0.0, 0.0, 0.0]  # Black for buildingsseries of images showing fire spread over time.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def save_fire_snapshot(sim, step_num, fire_starts=None, output_dir='output/timelapse'):
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
    
    # No fuel (water, buildings) as blue/black
    no_fuel_mask = (final_state['F'] == 0) & (~final_state['B'])
    for i in range(H):
        for j in range(W):
            if no_fuel_mask[i, j]:
                world_pos = sim.environment.grid_mapper.cell_to_world(i, j)
                is_lake = False
                is_building = False
                
                # Check if it's a lake
                for zone in sim.environment.terrain_zones:
                    if zone['type'] == 'lake':
                        center = zone['center']
                        radius = zone['radius']
                        dist = np.sqrt((world_pos[0] - center[0])**2 + (world_pos[1] - center[1])**2)
                        if dist <= radius:
                            img[i, j] = [0.3, 0.5, 0.9]  # Blue for water
                            is_lake = True
                            break
                
                # Check if it's a building (stored in obstacles)
                if not is_lake:
                    for obstacle in sim.environment.obstacles:
                        if obstacle['type'] == 'city_block':
                            # Check if point is inside building bounding box
                            pos = obstacle['position']
                            size = obstacle['size']
                            half_x = size[0] / 2
                            half_y = size[1] / 2
                            if (pos[0] - half_x <= world_pos[0] <= pos[0] + half_x and
                                pos[1] - half_y <= world_pos[1] <= pos[1] + half_y):
                                img[i, j] = [0.0, 0.0, 0.0]  # Black for buildings
                                is_building = True
                                break
                
                # Default gray for open terrain with no fuel
                if not is_lake and not is_building:
                    img[i, j] = [0.6, 0.6, 0.6]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    im = ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower', interpolation='nearest')
    
    # Add fire start markers if provided
    if fire_starts:
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
    
    # Add building outlines (from obstacles)
    for obstacle in sim.environment.obstacles:
        if obstacle['type'] == 'city_block':
            pos = obstacle['position']
            size = obstacle['size']
            # Create rectangle patch (bottom-left corner, width, height)
            from matplotlib.patches import Rectangle
            rect = Rectangle((pos[0] - size[0]/2, pos[1] - size[1]/2), 
                           size[0], size[1],
                           fill=False, edgecolor='black', linewidth=3, linestyle='-')
            ax.add_patch(rect)
    
    # Add wind direction arrow
    wind_velocity = sim.environment.weather['wind_velocity']
    wind_speed = np.linalg.norm(wind_velocity[:2])  # Only x, y components
    if wind_speed > 0.1:
        # Position arrow in top-right corner of plot
        arrow_base_x = x_max - 0.15 * (x_max - x_min)
        arrow_base_y = y_max - 0.1 * (y_max - y_min)
        
        # Scale arrow length to wind speed (but keep it visible)
        arrow_scale = 0.05 * (x_max - x_min)  # Arrow length proportional to map size
        wind_dx = wind_velocity[0] / wind_speed * arrow_scale
        wind_dy = wind_velocity[1] / wind_speed * arrow_scale
        
        # Draw the arrow
        ax.arrow(arrow_base_x, arrow_base_y, wind_dx, wind_dy,
                head_width=arrow_scale*0.3, head_length=arrow_scale*0.2,
                fc='white', ec='black', linewidth=2, zorder=1000)
        
        # Add wind speed label
        wind_angle_deg = np.degrees(np.arctan2(wind_velocity[1], wind_velocity[0]))
        ax.text(arrow_base_x, arrow_base_y - 0.03 * (y_max - y_min),
               f'Wind: {wind_speed:.1f} m/s\n{wind_angle_deg:.0f}°',
               fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               ha='center', va='top', zorder=1001)
    
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
        # Setup environment with LARGE terrain features
        print("📍 Setting up environment...")
        
        # Create multiple large forests spread across the map
        sim.environment.add_forest_area([0, 0], radius=40, tree_count=60)
        sim.environment.add_forest_area([80, -60], radius=35, tree_count=50)
        sim.environment.add_forest_area([-70, 70], radius=30, tree_count=45)
        
        # Add several lakes to demonstrate fire blocking
        sim.environment.add_lake([-60, -60], radius=20)
        sim.environment.add_lake([90, 90], radius=18)
        
        # Add buildings in different areas
        sim.environment.add_city_block([-40, 40], [12, 12, 15])
        sim.environment.add_city_block([50, 50], [10, 10, 15])
        sim.environment.add_city_block([0, -80], [15, 15, 20])
        
        print(f"✅ Large environment: 3 forests, 2 lakes, 3 buildings")
        
        # Enable fire simulation with MUCH LARGER grid
        print("🔥 Enabling fire simulation...")
        # 300x300 meter world with 1x1m cells = 300x300 grid (90,000 cells)
        sim.enable_fire_simulation(grid_width_m=300, grid_height_m=300, cell_size_m=1.0)
        
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
        
        # Wind is now handled by environment (random initial + dynamic changes)
        print("💨 Wind initialized by environment (random direction, changes over time)")
        
        # Start fires - RANDOM positions within one of the forest areas
        print("\n🔥 Starting fires...")
        import random
        random.seed()  # Use current time as seed for randomness
        
        # Randomly choose 1-2 fires (80% chance of 1 fire, 20% chance of 2 fires)
        num_fires = 1 if random.random() < 0.8 else 2
        
        # Randomly pick a forest to start fires in
        forest_zones = [zone for zone in sim.environment.terrain_zones if zone['type'] == 'forest']
        
        # Generate random fire positions within a randomly selected forest
        fires = []
        if forest_zones:
            # Pick a random forest
            chosen_forest = random.choice(forest_zones)
            center = chosen_forest['center']
            radius = chosen_forest['radius']
            
            for _ in range(num_fires):
                # Random angle and radius within forest (stay 5m from edge for safety)
                angle = random.uniform(0, 2 * np.pi)
                fire_radius = random.uniform(5, radius - 5)
                x = center[0] + fire_radius * np.cos(angle)
                y = center[1] + fire_radius * np.sin(angle)
                fires.append((x, y))
        else:
            # Fallback: center of map
            fires.append((0, 0))
        
        fire_starts = []  # Track for visualization
        for x, y in fires:
            if sim.start_fire((x, y), intensity=0.5):
                i, j = sim.environment.grid_mapper.world_to_cell((x, y))
                print(f"   ✓ Random fire at ({x:6.1f}, {y:6.1f}) → cell [{i:3d},{j:3d}]")
                fire_starts.append((x, y))
        
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
        save_fire_snapshot(sim, 0, fire_starts=fire_starts)
        print(f"   Saved: step_0000.png (t=0.0s) - BEFORE any simulation steps")
        
        # Run simulation and save snapshots every 5 steps (0.5 seconds) for first 30 seconds
        total_steps = 300  # 30 seconds
        snapshot_interval = 5  # Every 0.5 seconds
        
        for step in range(1, total_steps + 1):
            # Step simulation (no drones)
            sim.step_simulation({})
            
            # Save snapshot at intervals
            if step % snapshot_interval == 0:
                save_fire_snapshot(sim, step, fire_starts=fire_starts)
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
