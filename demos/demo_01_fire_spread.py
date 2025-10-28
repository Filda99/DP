#!/usr/bin/env python3
"""
Demo 1: Pure Fire Spread with New Physics

Demonstrates the improved fire physics model:
- Terrain-dependent fuel burn rates
- Forest: slow burn (0.03)
- Open areas: fast burn (0.08)
- Buildings/lakes: no burn (0.0)
- Realistic fire spread patterns

No suppression - just natural fire behavior.
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


def run_fire_spread_demo():
    """Run pure fire spread demonstration."""
    print("=" * 70)
    print("🔥 DEMO 1: FIRE SPREAD WITH NEW PHYSICS")
    print("=" * 70)
    print()
    print("Demonstrating:")
    print("  • Terrain-dependent burn rates")
    print("  • Fire blocked by water and buildings")
    print("  • Wind effects on spread direction")
    print()
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Setup mixed environment
    sim.setup_mixed_environment()
    
    # Enable fire with larger grid
    sim.enable_fire_simulation(
        grid_width_m=100,
        grid_height_m=100,
        cell_size_m=2.0
    )
    
    # Set wind blowing east
    sim.set_wind([8.0, 0.0, 0.0])
    
    # Start fires in forest areas
    fire_starts = [
        (-30, -30),  # Southwest forest
        (20, 25),    # Northeast area
    ]
    
    for pos in fire_starts:
        sim.start_fire(pos, intensity=0.3)
        print(f"  🔥 Started fire at {pos}")
    
    print()
    print("Running simulation for 60 seconds...")
    print()
    
    # Run simulation and collect snapshots
    snapshots = []
    snapshot_times = [0, 15, 30, 45, 60]  # seconds
    
    for step in range(int(60 / sim.timestep)):
        sim.step_simulation({})
        
        current_time = sim.simulation_time
        if any(abs(current_time - t) < sim.timestep for t in snapshot_times):
            fire_state = sim.environment.get_fire_state()
            if fire_state:
                snapshots.append({
                    'time': current_time,
                    'state': fire_state['fire_grid_state'].copy()
                })
                print(f"  📸 Snapshot at t={current_time:.1f}s")
    
    # Create visualization
    create_fire_spread_visualization(sim, snapshots, fire_starts)
    
    # Final statistics
    final_state = sim.environment.get_fire_state()
    if final_state:
        stats = final_state['fire_grid_state']
        total_burned = np.sum(stats['F'] < 0.5)  # Cells with significant fuel loss
        total_cells = stats['F'].size
        print()
        print(f"📊 Final Statistics:")
        print(f"   Total cells burned: {total_burned}/{total_cells} ({100*total_burned/total_cells:.1f}%)")
        print(f"   Currently burning: {np.sum(stats['B'])}")
    
    print()
    print("✅ Demo complete! Check output/demo_01_fire_spread.png")
    print("=" * 70)


def create_fire_spread_visualization(sim, snapshots, fire_starts):
    """Create visualization showing fire spread over time."""
    os.makedirs('output', exist_ok=True)
    
    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Get grid bounds
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    
    for idx, snapshot in enumerate(snapshots):
        ax = axes[idx]
        state = snapshot['state']
        
        # Create RGB image
        H, W = state['B'].shape
        img = np.zeros((H, W, 3))
        
        # Terrain base colors
        for i in range(H):
            for j in range(W):
                burn_rate = sim.environment.fire_grid.fuel_burn_rate[i, j]
                
                if burn_rate == 0.0:
                    # Water or buildings
                    world_pos = sim.environment.grid_mapper.cell_to_world(i, j)
                    is_water = False
                    
                    for zone in sim.environment.terrain_zones:
                        if zone['type'] == 'lake':
                            center = zone['center']
                            radius = zone['radius']
                            dist = np.sqrt((world_pos[0] - center[0])**2 + 
                                         (world_pos[1] - center[1])**2)
                            if dist <= radius:
                                img[i, j] = [0.2, 0.4, 0.8]  # Blue water
                                is_water = True
                                break
                    
                    if not is_water:
                        img[i, j] = [0.3, 0.3, 0.3]  # Gray buildings
                        
                elif burn_rate < 0.05:
                    # Forest (slow burn)
                    img[i, j] = [0.1, 0.4, 0.1]  # Dark green
                else:
                    # Open area (fast burn)
                    img[i, j] = [0.6, 0.5, 0.2]  # Tan/brown
        
        # Overlay fuel consumption (darker = more burned)
        fuel_remaining = state['F']
        burn_mask = fuel_remaining < 0.9
        img[burn_mask] *= (fuel_remaining[burn_mask, np.newaxis] * 0.5 + 0.5)
        
        # Overlay burning cells (bright orange/red)
        burning = state['B']
        intensity = state['I']
        img[burning] = np.stack([
            np.ones_like(intensity[burning]),
            intensity[burning] * 0.5,
            np.zeros_like(intensity[burning])
        ], axis=-1)
        
        # Display
        ax.imshow(img, origin='lower', extent=[x_min, x_max, y_min, y_max])
        ax.set_title(f't = {snapshot["time"]:.1f}s', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Mark initial fire locations
        if idx == 0:
            for pos in fire_starts:
                ax.plot(pos[0], pos[1], 'r*', markersize=20, 
                       markeredgecolor='white', markeredgewidth=2)
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_snapshots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Fire Spread with Terrain-Dependent Burn Rates', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/demo_01_fire_spread.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_fire_spread_demo()
