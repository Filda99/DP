"""
🔥 FIRE SPREAD DEMO - LOADING FROM CACHE
===========================================
This demo is identical to demo_01, but uses pre-downloaded cache files
instead of downloading data from OpenStreetMap API.

Advantages:
- MUCH faster loading (no network download)
- Works offline
- Consistent data (no changes between runs)

Prerequisites:
- Run tools/prefetch_osm_region.py first to create cache files
- Or use existing cache files in data/ directory
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache
import time

# Configuration
LOCATION_NAME = "Křivoklát, Czech Republic"
CENTER_LAT = 50.03785
CENTER_LON = 13.8703645
RADIUS_M = 1500
CACHE_DIR = "data"
REGION_PREFIX = "Křivoklát_Czechia"

OUTPUT_DIR = "output/demo_01b_frames"


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
    arrow_scale = 20.0
    
    ax1.arrow(arrow_start_x, arrow_start_y, 
             wind_x * arrow_scale, wind_y * arrow_scale,
             head_width=50, head_length=40, fc='yellow', ec='black', 
             linewidth=2, alpha=0.95, zorder=10)
    
    # Add wind speed label above arrow
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


print("=" * 70)
print("🔥 FIRE SPREAD DEMO - FROM CACHE")
print("=" * 70)
print(f"📂 Loading from: {CACHE_DIR}/{REGION_PREFIX}_*.gpkg")
print(f"📍 Location: {LOCATION_NAME}")
print(f"   Center: ({CENTER_LAT}°N, {CENTER_LON}°E)")
print(f"   Radius: {RADIUS_M}m")
print()

# Create simulation
sim = Simulation()
sim.start_simulation()

# Load environment from CACHE (instead of downloading)
print(f"📂 Loading environment from cache...")
start_time = time.time()

load_environment_from_osm_cache(
    environment=sim.environment,
    cache_dir=CACHE_DIR,
    region_prefix=REGION_PREFIX,
    center_lat=CENTER_LAT,
    center_lon=CENTER_LON,
    radius_m=RADIUS_M,
    default_height_m=8.0,
    use_city_boundaries=True
)

load_time = time.time() - start_time
print(f"⏱️  Cache loading time: {load_time:.2f}s")
print()

# Enable fire simulation (matching demo_01 settings)
sim.enable_fire_simulation(
    grid_width_m=3000,  # 3km grid to match 1.5km radius (±1500m)
    grid_height_m=3000,
    cell_size_m=15.0  # 200×200 cells
)

# Set wind
sim.set_wind([8.0, 0.0, 0.0])

# Save environment map
os.makedirs(OUTPUT_DIR, exist_ok=True)
map_path = f"{OUTPUT_DIR}/environment.png"
print(f"📸 Saving environment map...")
sim.environment.save_environment_map(map_path, show_fire_grid=True, detailed=False)
print(f"✅ Environment map saved to: {map_path}")
print()

# Find a forested area to start fire (matching demo_01 logic)
print("🔍 Finding forested area to start fire...")
fire_started = False
fire_state = sim.environment.get_fire_state()

if fire_state:
    state = fire_state['fire_grid_state']
    fuel_grid = state['F']
    
    # Find cells with high fuel (forests have fuel > 0.7)
    high_fuel_cells = np.argwhere(fuel_grid > 0.7)
    
    if len(high_fuel_cells) > 0:
        # Choose a random high-fuel cell
        idx = np.random.randint(len(high_fuel_cells))
        cell_i, cell_j = high_fuel_cells[idx]
        
        # Convert cell coordinates to world coordinates
        grid_mapper = sim.environment.grid_mapper
        world_x = (cell_j - grid_mapper.grid_width_cells // 2) * grid_mapper.cell_size_m
        world_y = (cell_i - grid_mapper.grid_height_cells // 2) * grid_mapper.cell_size_m
        
        sim.start_fire((world_x, world_y), intensity=0.3)
        fire_started = True
        print(f"✅ Started fire at ({world_x:.0f}, {world_y:.0f}) in forested area")
        print(f"     Cell: ({cell_i}, {cell_j}), Fuel: {fuel_grid[cell_i, cell_j]:.2f}")
    else:
        print("⚠️  No high-fuel areas found!")
        fire_started = False

if not fire_started:
    print("❌ Could not start fire - no suitable location found")
    sim.stop_simulation()
    exit(1)

print()

# Run simulation
DURATION = 10.0  # seconds
FRAME_INTERVAL = 0.5  # save every 0.5s

print(f"Running simulation for {DURATION} seconds, saving every {FRAME_INTERVAL}s...")
print()

frame_idx = 0
save_interval = int(FRAME_INTERVAL / sim.timestep)  # Number of steps per frame

for step in range(int(DURATION / sim.timestep)):
    sim.step_simulation({})
    
    # Save frame every FRAME_INTERVAL seconds
    if step % save_interval == 0:
        current_time = sim.simulation_time
        fire_state = sim.environment.get_fire_state()
        
        if fire_state:
            state = fire_state['fire_grid_state']
            burning_count = np.sum(state['B'])
            
            # Save visualization using same function as demo_01
            save_frame(sim, state, frame_idx, current_time, OUTPUT_DIR)
            
            print(f"  Frame {frame_idx:03d} | t={current_time:.1f}s | Burning: {burning_count:4d} cells")
        
        frame_idx += 1

print()
print(f"✅ Saved {frame_idx} frames to {OUTPUT_DIR}/")
print(f"   Check frame_000.png, frame_001.png, etc.")
print("=" * 70)

sim.stop_simulation()


