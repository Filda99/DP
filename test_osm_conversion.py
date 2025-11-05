#!/usr/bin/env python3
"""
Test OSM to Environment Conversion

This script demonstrates how real OSM data is converted into the simulation's
internal representation:
- Buildings → Rectangular boxes (obstacles, non-burnable)
- Forests → Multiple overlapping circles (high fuel, burnable)
- Water → Multiple overlapping circles (fire breaks, non-burnable)
- Grass/Meadow → Default terrain (medium fuel, burnable)
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def test_osm_conversion():
    """Test loading OSM data and show what gets converted."""
    print("=" * 70)
    print("🧪 TESTING OSM DATA CONVERSION")
    print("=" * 70)
    print()
    print("This test shows how real-world map features are converted:")
    print("  • Buildings → Rectangular blocks (PyBullet collision boxes)")
    print("  • Forests → Multiple circles with trees (better shape approximation)")
    print("  • Water → Multiple circles (better shape approximation)")
    print("  • Grass/Meadow → Default burnable terrain (fire grid handles)")
    print()
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Load a small area for testing
    location = "Tišnov, Czech Republic"
    print(f"📍 Loading: {location}")
    print()
    
    # This will download and convert OSM data
    sim.setup_osm_environment(location, default_building_height=8.0)
    
    print()
    print("=" * 70)
    print("📊 CONVERSION SUMMARY")
    print("=" * 70)
    
    # Show what's in the environment
    print(f"Obstacles (buildings): {len(sim.environment.obstacles)}")
    print(f"Terrain zones (forests + water): {len(sim.environment.terrain_zones)}")
    
    # Count by type
    forests = sum(1 for zone in sim.environment.terrain_zones if zone['type'] == 'forest')
    lakes = sum(1 for zone in sim.environment.terrain_zones if zone['type'] == 'lake')
    
    print(f"  - Forest circles: {forests}")
    print(f"  - Water circles: {lakes}")
    print()
    
    print("🔥 Now you can enable fire simulation and it will use this real terrain!")
    print("   - Fire spreads faster in forests (high fuel)")
    print("   - Fire stops at water bodies")
    print("   - Fire stops at buildings")
    print("   - Fire spreads slower in grass/open areas")
    print()
    
    # Optionally enable fire to show it works
    print("Enabling fire grid...")
    sim.enable_fire_simulation(grid_width_m=500, grid_height_m=500, cell_size_m=5.0)
    
    if sim.environment.fire_grid:
        H, W = sim.environment.fire_grid.H, sim.environment.fire_grid.W
        print(f"✅ Fire grid created: {H}×{W} cells")
        
        # Show fuel distribution
        import numpy as np
        fuel = sim.environment.fire_grid.fuel_burn_rate
        zero_fuel = np.sum(fuel == 0.0)
        low_fuel = np.sum((fuel > 0.0) & (fuel < 0.05))
        high_fuel = np.sum(fuel >= 0.05)
        
        print(f"   Fuel distribution:")
        print(f"   - No fuel (water/buildings): {zero_fuel} cells ({100*zero_fuel/(H*W):.1f}%)")
        print(f"   - Low fuel (grass/open): {low_fuel} cells ({100*low_fuel/(H*W):.1f}%)")
        print(f"   - High fuel (forest): {high_fuel} cells ({100*high_fuel/(H*W):.1f}%)")
    
    print()
    print("=" * 70)
    print("✅ Test complete!")
    print("=" * 70)
    
    # Save environment visualization
    print()
    print("📸 Saving environment visualization...")
    output_file = "output/environment_initial.png"
    os.makedirs("output", exist_ok=True)
    sim.environment.save_environment_map(output_file, show_fire_grid=True)
    
    print()
    print(f"🎨 View the environment map at: {output_file}")
    print("   This shows:")
    print("   • Buildings (gray rectangles)")
    print("   • Forests (green circles)")
    print("   • Water (blue circles)")
    print("   • Fire grid fuel levels (color overlay)")
    print()
    
    sim.stop_simulation()


if __name__ == "__main__":
    test_osm_conversion()
