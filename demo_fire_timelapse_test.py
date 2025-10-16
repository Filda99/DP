#!/usr/bin/env python3
"""Quick test of fire time-lapse with wind arrows - smaller grid for testing."""

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

# Import the save function from the main demo
import importlib.util
spec = importlib.util.spec_from_file_location("demo_fire", "demo_fire_timelapse.py")
demo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_module)
save_fire_snapshot = demo_module.save_fire_snapshot

def run_quick_test():
    """Quick test with smaller grid."""
    
    print("=" * 70)
    print("🔥 FIRE SPREAD TEST - Wind Arrow Visualization")
    print("=" * 70)
    
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    try:
        print("📍 Setting up environment...")
        sim.environment.add_forest_area([0, 0], radius=30, tree_count=50)
        sim.environment.add_lake([-40, -40], radius=15)
        sim.environment.add_city_block([40, 40], [10, 10, 15])
        
        print("🔥 Enabling fire simulation...")
        # Smaller 150x150m grid with 1.5m cells = 100x100 grid (10,000 cells)
        sim.enable_fire_simulation(grid_width_m=150, grid_height_m=150, cell_size_m=1.5)
        
        if sim.environment.fire_enabled:
            sim.environment.fire_grid.l_base *= 1.5
            sim.environment.fire_grid.alpha = 0.3
            sim.environment.fire_grid.k_wind = 1.0
            print(f"   Grid: {sim.environment.grid_mapper.grid_height_cells}x{sim.environment.grid_mapper.grid_width_cells} cells")
        
        print("💨 Wind initialized by environment")
        
        # Start one fire in the center of the forest
        print("\n🔥 Starting fire...")
        import random
        random.seed()
        
        fire_x = random.uniform(-15, 15)
        fire_y = random.uniform(-15, 15)
        sim.start_fire((fire_x, fire_y), intensity=0.5)
        fire_starts = [(fire_x, fire_y)]
        
        print(f"   ✓ Fire at ({fire_x:.1f}, {fire_y:.1f})")
        
        # Save snapshots - just first 10 seconds
        print("\n📸 Saving snapshots...")
        save_fire_snapshot(sim, 0, fire_starts=fire_starts)
        print(f"   Saved: step_0000.png (t=0.0s)")
        
        for step in range(1, 101):  # 10 seconds
            sim.step_simulation({})
            
            if step % 10 == 0:  # Every 1 second
                save_fire_snapshot(sim, step, fire_starts=fire_starts)
                stats = sim.get_simulation_summary()['fire']
                wind = sim.environment.weather['wind_velocity']
                wind_speed = np.linalg.norm(wind[:2])
                print(f"   Step {step:3d} (t={step*0.1:.1f}s) - "
                      f"Burning: {stats['burning_cells']:3d}, "
                      f"Wind: {wind_speed:.1f} m/s")
        
        print("\n✅ Test complete!")
        
    finally:
        sim.stop_simulation()


if __name__ == '__main__':
    run_quick_test()
