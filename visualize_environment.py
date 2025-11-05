#!/usr/bin/env python3
"""
Environment Visualization Tool

Loads OSM data and creates detailed visualization showing:
1. Top-down environment map with buildings, forests, water
2. Fire grid fuel level distribution
3. Statistics about the loaded environment

This helps you:
- Verify OSM data was loaded correctly
- Find good locations to start fires (forests/open areas)
- Compare the simulation environment to the real map
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def visualize_environment(location, grid_size=6000, cell_size=20.0, distance_m=3000):
    """
    Load and visualize an OSM environment.
    
    Args:
        location: Location query (e.g., "Tišnov, Czech Republic")
        grid_size: Fire grid size in meters
        cell_size: Fire grid cell size in meters
        distance_m: Download radius in meters around location center
    """
    print("=" * 70)
    print(f"🌍 ENVIRONMENT VISUALIZATION: {location}")
    print("=" * 70)
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Load OSM data
    print(f"\n📍 Loading environment from OSM...")
    sim.setup_osm_environment(location, default_building_height=8.0, distance_m=distance_m)
    
    # Enable fire grid
    print(f"\n🔥 Creating fire grid ({grid_size}×{grid_size}m, {cell_size}m cells)...")
    sim.enable_fire_simulation(grid_width_m=grid_size, grid_height_m=grid_size, cell_size_m=cell_size)
    
    print("\n" + "=" * 70)
    print("📊 ENVIRONMENT STATISTICS")
    print("=" * 70)
    
    # Count objects
    buildings = len(sim.environment.obstacles)
    forests = sum(1 for z in sim.environment.terrain_zones if z['type'] == 'forest')
    lakes = sum(1 for z in sim.environment.terrain_zones if z['type'] == 'lake')
    
    print(f"\n🏗️  Physical Objects:")
    print(f"   Buildings: {buildings:,}")
    print(f"   Forest areas: {forests}")
    print(f"   Water bodies: {lakes}")
    
    # Fire grid statistics
    if sim.environment.fire_grid:
        H, W = sim.environment.fire_grid.H, sim.environment.fire_grid.W
        total_cells = H * W
        
        fuel = sim.environment.fire_grid.fuel_burn_rate
        zero_fuel = np.sum(fuel == 0.0)
        low_fuel = np.sum((fuel > 0.0) & (fuel < 0.05))
        high_fuel = np.sum(fuel >= 0.05)
        
        print(f"\n🔥 Fire Grid ({H}×{W} = {total_cells:,} cells):")
        print(f"   No fuel (buildings/water): {zero_fuel:,} cells ({100*zero_fuel/total_cells:.1f}%)")
        print(f"   Low fuel (grass/open):     {low_fuel:,} cells ({100*low_fuel/total_cells:.1f}%)")
        print(f"   High fuel (forest):        {high_fuel:,} cells ({100*high_fuel/total_cells:.1f}%)")
        
        # Find good fire starting locations
        print(f"\n🎯 Suggested Fire Starting Locations:")
        
        x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
        
        # Find cells with fuel
        burnable_locations = []
        for i in range(H):
            for j in range(W):
                if fuel[i, j] > 0.0:
                    world_pos = sim.environment.grid_mapper.cell_to_world(i, j)
                    burnable_locations.append((world_pos[0], world_pos[1], fuel[i, j]))
        
        if burnable_locations:
            # Sample a few locations
            sample_size = min(5, len(burnable_locations))
            step = len(burnable_locations) // sample_size
            samples = burnable_locations[::step][:sample_size]
            
            for idx, (x, y, f) in enumerate(samples, 1):
                fuel_type = "forest" if f >= 0.05 else "grass"
                print(f"   {idx}. ({x:6.1f}, {y:6.1f}) - {fuel_type} (fuel={f:.3f})")
        else:
            print(f"   ⚠️  No burnable locations found! Entire area is buildings/water.")
        
        # Calculate coverage area
        grid_area = grid_size * grid_size / 1_000_000  # km²
        print(f"\n📏 Coverage:")
        print(f"   Grid area: {grid_area:.2f} km²")
        print(f"   Resolution: {cell_size}m × {cell_size}m cells")
    
    print("\n" + "=" * 70)
    
    # Save visualization
    output_file = f"output/env_{location.replace(', ', '_').replace(' ', '_')}.png"
    os.makedirs("output", exist_ok=True)
    
    print(f"\n📸 Saving visualization to: {output_file}")
    # Use detailed=True for visualization tool (we want to see everything)
    # Set to False for faster rendering with large areas
    use_detailed = (len(sim.environment.obstacles) < 1000)  # Auto-decide based on size
    print(f"   Using detailed mode: {use_detailed}")
    sim.environment.save_environment_map(output_file, show_fire_grid=True, detailed=use_detailed)
    
    # Create additional detailed view
    create_detailed_view(sim, location)
    
    sim.stop_simulation()
    
    print("\n" + "=" * 70)
    print("✅ Visualization complete!")
    print("=" * 70)
    print(f"\n📂 Files created:")
    print(f"   • {output_file}")
    print(f"   • output/env_{location.replace(', ', '_').replace(' ', '_')}_detailed.png")
    print()


def create_detailed_view(sim, location):
    """Create a detailed multi-panel view of the environment."""
    
    if sim.environment.fire_grid is None:
        return
    
    fig = plt.figure(figsize=(20, 10))
    
    # Panel 1: Fuel map
    ax1 = plt.subplot(1, 3, 1)
    fuel = sim.environment.fire_grid.fuel_burn_rate
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    
    im1 = ax1.imshow(fuel, extent=[x_min, x_max, y_min, y_max], 
                     origin='lower', cmap='YlOrRd', interpolation='nearest')
    ax1.set_title('Fuel Burn Rate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    plt.colorbar(im1, ax=ax1, label='Burn Rate (1/s)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Binary fuel map (burnable vs non-burnable)
    ax2 = plt.subplot(1, 3, 2)
    binary_fuel = (fuel > 0.0).astype(float)
    im2 = ax2.imshow(binary_fuel, extent=[x_min, x_max, y_min, y_max],
                     origin='lower', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title('Burnable Areas', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar2.ax.set_yticklabels(['Non-burnable', 'Burnable'])
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Fuel level histogram
    ax3 = plt.subplot(1, 3, 3)
    fuel_flat = fuel.flatten()
    fuel_nonzero = fuel_flat[fuel_flat > 0]
    
    ax3.hist(fuel_flat, bins=50, alpha=0.7, color='blue', label='All cells', edgecolor='black')
    if len(fuel_nonzero) > 0:
        ax3.hist(fuel_nonzero, bins=30, alpha=0.7, color='red', label='Burnable only', edgecolor='black')
    
    ax3.set_xlabel('Fuel Burn Rate (1/s)', fontsize=12)
    ax3.set_ylabel('Number of Cells', fontsize=12)
    ax3.set_title('Fuel Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    # Add statistics text
    H, W = fuel.shape
    total = H * W
    zero = np.sum(fuel == 0.0)
    low = np.sum((fuel > 0.0) & (fuel < 0.05))
    high = np.sum(fuel >= 0.05)
    
    stats_text = f"Total cells: {total:,}\n"
    stats_text += f"Non-burnable: {zero:,} ({100*zero/total:.1f}%)\n"
    stats_text += f"Low fuel: {low:,} ({100*low/total:.1f}%)\n"
    stats_text += f"High fuel: {high:,} ({100*high/total:.1f}%)"
    
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Environment Analysis: {location}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = f"output/env_{location.replace(', ', '_').replace(' ', '_')}_detailed.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # You can change these parameters
    LOCATION = "Tišnov, Czech Republic"
    GRID_SIZE = 6000  # 6km x 6km grid
    CELL_SIZE = 20.0  # 20m cells
    DISTANCE_M = 3000  # Download 3km radius around center
    
    visualize_environment(LOCATION, GRID_SIZE, CELL_SIZE, DISTANCE_M)
    
    print("\n💡 TIP: To visualize a different location, edit this file and change:")
    print(f"   LOCATION = '{LOCATION}'")
    print("\n   Try: 'Brno, Czech Republic', 'Golden Gate Park, San Francisco', etc.")
