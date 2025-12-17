#!/usr/bin/env python3
"""
Demo 1 DEBUG: Save every step for detailed analysis
"""

import numpy as np
import sys
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache


def save_frame(sim, state, frame_num, time, output_dir):
    """Save a single frame visualization."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    H, W = state['B'].shape
    
    # === PANEL 1: Fire state (burning cells) ===
    img = np.zeros((H, W, 3))
    burn_rate = sim.environment.fire_grid.fuel_burn_rate
    
    # Default: open area (tan)
    img[:, :] = [0.6, 0.5, 0.2]
    
    # Forest (high fuel)
    forest_mask = (burn_rate >= 0.05)
    img[forest_mask] = [0.1, 0.4, 0.1]
    
    # Buildings/water (zero fuel) - gray
    zero_mask = (burn_rate == 0.0)
    img[zero_mask] = [0.3, 0.3, 0.3]
    
    # Overlay burning cells (RED)
    burning = state['B']
    intensity = state['I']
    img[burning] = np.stack([
        np.ones_like(intensity[burning]),
        intensity[burning] * 0.3,
        np.zeros_like(intensity[burning])
    ], axis=-1)
    
    ax1.imshow(img, origin='lower', extent=[x_min, x_max, y_min, y_max])
    
    # Add wind arrow
    wind_vel = sim.environment.weather['wind_velocity']
    wind_x, wind_y = wind_vel[0], wind_vel[1]
    wind_speed = np.sqrt(wind_x**2 + wind_y**2)
    
    # Draw wind arrow in top-right corner
    arrow_start_x = x_max - 200
    arrow_start_y = y_max - 150
    arrow_scale = 5.0
    
    ax1.arrow(arrow_start_x, arrow_start_y, 
             wind_x * arrow_scale, wind_y * arrow_scale,
             head_width=10, head_length=20, fc='yellow', ec='black', 
             linewidth=2, alpha=0.95, zorder=10)
    
    ax1.text(arrow_start_x, arrow_start_y + 100, 
            f'Wind: {wind_speed:.1f} m/s',
            color='yellow', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='yellow', linewidth=2),
            ha='center', zorder=10)
    
    ax1.set_title(f'Fire State (Burning Cells)', fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)
    
    # === PANEL 2: Fuel remaining ===
    fuel = state['F']
    im2 = ax2.imshow(fuel, origin='lower', extent=[x_min, x_max, y_min, y_max],
                     cmap='YlOrRd_r', vmin=0, vmax=1)
    ax2.set_title(f'Fuel Remaining', fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    plt.colorbar(im2, ax=ax2, label='Fuel (0-1)')
    ax2.grid(True, alpha=0.3)
    
    # === PANEL 3: Intensity ===
    im3 = ax3.imshow(intensity, origin='lower', extent=[x_min, x_max, y_min, y_max],
                     cmap='hot', vmin=0, vmax=1)
    ax3.set_title(f'Fire Intensity', fontweight='bold')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    plt.colorbar(im3, ax=ax3, label='Intensity (0-1)')
    ax3.grid(True, alpha=0.3)
    
    # Overall title
    burning_count = np.sum(burning)
    avg_fuel = np.mean(fuel[fuel > 0]) if np.any(fuel > 0) else 0
    plt.suptitle(f't = {time:.1f}s | Burning: {burning_count} cells | Avg Fuel: {avg_fuel:.2f}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frame_{frame_num:03d}.png', dpi=100, bbox_inches='tight')
    plt.close()

def run_debug_demo():
    """Run fire spread with frame-by-frame output using REAL map data."""
    print("=" * 70)
    print("🔬 DEBUG: FIRE SPREAD FRAME-BY-FRAME (Real OSM Data)")
    print("=" * 70)
    
    # Configuration
    LOCATION = "Pec pod Sněžkou, Czech Republic"
    CACHE_PREFIX = "Pec_pod_Sněžkou_Czechia"
    CENTER_LAT = 50.6868
    CENTER_LON = 15.7361
    RADIUS_M = 1000
    
    # Step 2: Create simulation
    sim = Simulation()
    sim.start_simulation()
    
    # Step 3: Load environment
    cache_dir = "data"
    cache_pattern = f"{cache_dir}/{CACHE_PREFIX}_building_*.gpkg"
    use_cache = len(glob.glob(cache_pattern)) > 0
    
    if use_cache:
        print(f"  Loading from cache: {CACHE_PREFIX}")
        load_environment_from_osm_cache(
            environment=sim.environment,
            cache_dir=cache_dir,
            region_prefix=CACHE_PREFIX,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
            radius_m=RADIUS_M,
            default_height_m=8.0
        )
    else:
        print(f"  No cache found, downloading from OSM...")
        # FIX: Updated argument names (default_height_m, radius_m)
        sim.setup_osm_environment(LOCATION, 
                                  default_height_m=8.0,
                                  radius_m=RADIUS_M)
        
    # Enable fire
    sim.enable_fire_simulation(
        grid_width_m=2*RADIUS_M,
        grid_height_m=2*RADIUS_M,
        cell_size_m=2.0
    )
    
    # Save initial environment visualization
    print("\n📸 Saving initial environment map...")
    sim.environment.save_environment_map('output/demo_01_environment.png', 
                                        show_fire_grid=True, 
                                        detailed=False)
    
    # Find a location with fuel to start the fire
    print("\n🔍 Finding forested area to start fire...")
    fire_started = False
    fire_state = sim.environment.get_fire_state()
    
    if fire_state:
        state = fire_state['fire_grid_state']
        fuel_grid = state['F']
        H, W = fuel_grid.shape
        
        # Try to find high fuel cells in bottom middle
        bottom_middle_cells = []
        for cell_i in range(H // 4, H // 2):
            for cell_j in range(W // 3, 2 * W // 3):
                if fuel_grid[cell_i, cell_j] > 0.7:
                    bottom_middle_cells.append((cell_i, cell_j))
        
        if len(bottom_middle_cells) > 0:
            idx = np.random.randint(len(bottom_middle_cells))
            cell_i, cell_j = bottom_middle_cells[idx]
            
            grid_mapper = sim.environment.grid_mapper
            world_x = (cell_j - grid_mapper.grid_width_cells // 2) * grid_mapper.cell_size_m
            world_y = (cell_i - grid_mapper.grid_height_cells // 2) * grid_mapper.cell_size_m
            
            sim.start_fire((world_x, world_y), intensity=0.3)
            fire_started = True
            print(f"✅ Started fire at ({world_x:.0f}, {world_y:.0f})")
    
    if not fire_started:
        print("Using default fire location...")
        sim.start_fire((-100, -100), intensity=0.3)
    
    # Create output directory
    output_dir = 'output/debug_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation
    frame = 0
    save_interval = 30
    
    for step in range(int(30 / sim.timestep)):
        sim.step_simulation({})
        
        if step % save_interval == 0:
            current_time = sim.simulation_time
            fire_state = sim.environment.get_fire_state()
            
            if fire_state:
                state = fire_state['fire_grid_state']
                burning = np.sum(state['B'])
                save_frame(sim, state, frame, current_time, output_dir)
                print(f"  Frame {frame:03d} | t={current_time:.1f}s | Burning: {burning:3d} cells")
                frame += 1
    
    print(f"\n✅ Saved frames to {output_dir}/")

if __name__ == '__main__':
    run_debug_demo()