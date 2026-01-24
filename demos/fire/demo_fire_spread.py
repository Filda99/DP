#!/usr/bin/env python3
"""
Demo 1: Fast Fire Simulation
Optimized for speed: Larger grid cells, less frequent saving.
"""

import numpy as np
import sys
import os
import glob
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache

# Graph styling
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except ImportError:
    pass

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (18, 6),
    'figure.dpi': 100
})

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache

def save_frame(sim, state, frame_num, time_val, output_dir):
    """Renders and saves a single frame."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 12), constrained_layout=False)
    ax1, ax2 = axes
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    # fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    # ax1, ax2, ax3 = axes
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    H, W = state['B'].shape
    extent = [x_min, x_max, y_min, y_max]
    
    # === PANEL 1: Environment + Fire ===
    burn_rate = sim.environment.fire_grid.fuel_burn_rate
    env_img = np.zeros((H, W, 3))
    
    # Terrain colors
    mask_water = (burn_rate == 0.0) 
    mask_building = (burn_rate > 0.0) & (burn_rate <= 0.0002)
    mask_forest = (burn_rate > 0.0002) & (burn_rate <= 0.0005)
    mask_grass = (burn_rate >= 0.0005)

    env_img[mask_grass] = [0.6, 0.7, 0.4]    # Grass
    env_img[mask_forest] = [0.1, 0.4, 0.1]   # Forest
    env_img[mask_water] = [0.2, 0.5, 0.9]    # Water
    env_img[mask_building] = [0.5, 0.5, 0.5] # Buildings
    
    # Fire layer
    burning = state['B']
    fire_overlay = np.zeros((H, W, 4))
    fire_overlay[burning] = [1.0, 0.2, 0.0, 0.8] # Red
    
    ax1.imshow(env_img, origin='lower', extent=extent)
    ax1.imshow(fire_overlay, origin='lower', extent=extent)
    
    # Wind Arrow (Saturated/Clamped Size)
    wind_vel = sim.environment.weather['wind_velocity']
    wind_speed = np.linalg.norm(wind_vel[:2])
    
    arrow_x = x_max - (x_max - x_min) * 0.1
    arrow_y = y_max - (y_max - y_min) * 0.1
    
    # FIX: Normalize arrow length so it doesn't break graph limits
    if wind_speed > 0.1:
        # Normalized direction vector
        direction = wind_vel[:2] / wind_speed
        # Fixed visual length (e.g., 150 meters on map)
        visual_length = 30
        dx = direction[0] * visual_length
        dy = direction[1] * visual_length
    else:
        dx, dy = 0, 0

    ax1.arrow(arrow_x - 20, arrow_y - 20, dx, dy, 
              head_width=20, head_length=20, fc='yellow', ec='black', width=8, zorder=10)
    
    txt = ax1.text(arrow_x + 5, arrow_y + 5, f"{wind_speed:.1f} m/s", color='yellow', 
             fontsize=8, ha='center', zorder=11)
    txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground="black")])
    
    ax1.set_title('Map & Fire')
    ax1.set_xlabel('X [m]', fontsize=8)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cax1.axis('off') # Hide it!

    # === PANEL 2: Fuel ===
    fuel = state['F']
    im2 = ax2.imshow(fuel, origin='lower', extent=extent, cmap='YlGn_r', vmin=0, vmax=1)
    ax2.set_title('Remaining Fuel')
    ax2.set_yticklabels([]) 
    ax1.set_xlabel('X [m]', fontsize=8)
    ax1.set_ylabel('Y [m]', fontsize=8)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('Fuel (0-1)', fontsize=8)
    cbar2.ax.tick_params(labelsize=8)

    # === PANEL 3: Intensity ===
    # intensity = state['I']
    # im3 = ax3.imshow(intensity, origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=1)
    # ax3.set_title('Fire Intensity', fontweight='bold')
    # ax3.set_yticklabels([])
    # divider3 = make_axes_locatable(ax3)
    # cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im3, cax=cax3).set_label('Intensity (0-1)')

    # Global Title
    # burning_count = np.sum(burning)
    # burn_area_ha = (burning_count * sim.environment.grid_mapper.cell_size_m**2) / 10000
    # plt.suptitle(f'Time: {time_val:.1f}s | Burned Area: {burn_area_ha:.2f} ha', fontsize=16, fontweight='bold')
    
    plt.savefig(f'{output_dir}/frame_{frame_num:03d}.pdf')
    plt.close()


def save_summary_figure(snapshots, output_dir):
    """
    Saves a 3-row x 2-column summary figure (Start, Middle, End).
    Columns: Map & Fire | Fuel
    """
    # 3 Rows (Time), 2 Columns (Map, Fuel)
    # Adjusted figsize to fit 3 vertical rows clearly
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), constrained_layout=False)
    
    # Adjust margins
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.05, top=0.95, hspace=0.25, wspace=0.15)
    
    # Get bounds from the simulation reference in the first snapshot
    sim_ref = snapshots[0][1]
    x_min, x_max, y_min, y_max = sim_ref.environment.grid_mapper.get_grid_bounds()
    extent = [x_min, x_max, y_min, y_max]

    for row_idx, (time_val, sim, state, wind) in enumerate(snapshots):
        ax1, ax2 = axes[row_idx]
        
        # Apply your tick params
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)

        # ===========================
        # COL 1: Map & Fire
        # ===========================
        H, W = state['B'].shape
        burn_rate = sim.environment.fire_grid.fuel_burn_rate
        env_img = np.zeros((H, W, 3))
        
        # Your specific mask logic
        mask_water = (burn_rate == 0.0) 
        mask_building = (burn_rate > 0.0) & (burn_rate <= 0.0002)
        mask_forest = (burn_rate > 0.0002) & (burn_rate <= 0.0005)
        mask_grass = (burn_rate >= 0.0005)

        env_img[mask_grass] = [0.6, 0.7, 0.4]    # Grass
        env_img[mask_forest] = [0.1, 0.4, 0.1]   # Forest
        env_img[mask_water] = [0.2, 0.5, 0.9]    # Water
        env_img[mask_building] = [0.5, 0.5, 0.5] # Buildings
        
        # Fire layer
        burning = state['B']
        fire_overlay = np.zeros((H, W, 4))
        fire_overlay[burning] = [1.0, 0.2, 0.0, 0.8] # Red
        
        ax1.imshow(env_img, origin='lower', extent=extent)
        ax1.imshow(fire_overlay, origin='lower', extent=extent)
        
        # Wind Arrow logic
        # Note: This uses the CURRENT wind from sim. If wind changes over time, 
        # you might want to store wind in the snapshot tuple too. 
        wind_vel = wind
        wind_speed = np.linalg.norm(wind_vel[:2])
        
        arrow_x = x_max - (x_max - x_min) * 0.1
        arrow_y = y_max - (y_max - y_min) * 0.1
        
        if wind_speed > 0.1:
            direction = wind_vel[:2] / wind_speed
            visual_length = 30 # Your fixed length
            dx = direction[0] * visual_length
            dy = direction[1] * visual_length
        else:
            dx, dy = 0, 0

        ax1.arrow(arrow_x - 20, arrow_y - 20, dx, dy, 
                  head_width=20, head_length=20, fc='yellow', ec='black', width=8, zorder=10)
        
        txt = ax1.text(arrow_x + 5, arrow_y + 5, f"{wind_speed:.1f} m/s", color='yellow', 
                 fontsize=8, ha='center', zorder=11)
        txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground="black")])
        
        # Labels and Titles
        # Only set Main Title on the top row
        if row_idx == 0:
            ax1.set_title('Map & Fire', fontweight='bold')
        
        # Add Time Label to Y-Axis
        ax1.set_ylabel(f'T = {time_val:.0f} s', fontsize=10, fontweight='bold')
        
        # Only show X label on bottom row
        if row_idx == 2:
            ax1.set_xlabel('X [m]', fontsize=8)

        # Invisible Colorbar for alignment
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cax1.axis('off')

        # ===========================
        # COL 2: Fuel
        # ===========================
        fuel = state['F']
        im2 = ax2.imshow(fuel, origin='lower', extent=extent, cmap='YlGn_r', vmin=0, vmax=1)
        
        if row_idx == 0:
            ax2.set_title('Remaining Fuel', fontweight='bold')
            
        ax2.set_yticklabels([]) 
        
        if row_idx == 2:
            ax2.set_xlabel('X [m]', fontsize=8)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('Fuel (0-1)', fontsize=8)
        cbar2.ax.tick_params(labelsize=8)

    # Save
    filename = f"{output_dir}/summary_progression.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Summary Figure saved: {filename}")
    plt.close()

def save_real_satellite_imagery(location, radius_m):
    """Save real satellite imagery from OpenStreetMap before running simulation."""
    import osmnx as ox
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point, box
    
    print("SAVING REAL SATELLITE IMAGERY")
    
    # Get location coordinates
    print(f"📍 Geocoding: {location}")
    try:
        coords = ox.geocode(location)
        print(f"   Coordinates: (lat={coords[0]}, lon={coords[1]})")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Create bounding box
    point = Point(coords[1], coords[0])  # lon, lat
    gdf_point = gpd.GeoDataFrame([{'geometry': point}], crs='EPSG:4326') # WGS84
    gdf_point_utm = gdf_point.to_crs(gdf_point.estimate_utm_crs()) # Meter based coordinate system
    point_utm = gdf_point_utm.geometry.iloc[0] # Center point in UTM to add meters offset for bbox
    
    bbox_utm = box(point_utm.x - radius_m, point_utm.y - radius_m,
                   point_utm.x + radius_m, point_utm.y + radius_m)  # Create bbox in UTM
    gdf_bbox = gpd.GeoDataFrame([{'geometry': bbox_utm}], crs=gdf_point_utm.crs)
    
    # Convert to Web Mercator
    gdf_bbox_web = gdf_bbox.to_crs('EPSG:3857')
    gdf_point_web = gdf_point.to_crs('EPSG:3857')
    
    # Create figure
    print("    Creating satellite map...")
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_title(f"Real Satellite Imagery\n{location}\n(±{radius_m}m radius)", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Plot boundaries
    gdf_bbox_web.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=4)
    gdf_point_web.plot(ax=ax, color='red', markersize=400, marker='x', 
                       linewidths=5, zorder=10)
    
    # Add satellite basemap
    print("    Downloading satellite tiles...")
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom='auto')
        print("    Satellite imagery loaded")
    except Exception as e:
        print(f"   ⚠️  Failed to load satellite imagery: {e}")
    
    ax.set_xlabel("Easting (Web Mercator)", fontsize=12)
    ax.set_ylabel("Northing (Web Mercator)", fontsize=12)
    
    # Save
    output_path = 'output/demo_01_satellite.pdf'
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}\n")

def run_fast_demo():
    print("-" * 50)
    print("FAST FIRE SIMULATION (Optimized)")
    print("-" * 50)
    
    LOCATION = "Pec pod Sněžkou, Czech Republic"
    CACHE_PREFIX = "Pec_pod_Sněžkou_Czech_Republic"
    CENTER_LAT = 50.6868
    CENTER_LON = 15.7361
    RADIUS_M = 300
    
    # 1. Start Simulation
    sim = Simulation()
    sim.start_simulation()
    
    # 2. Load Map
    cache_dir = "data"
    cache_pattern = f"{cache_dir}/{CACHE_PREFIX}*.gpkg"
    
    if len(glob.glob(cache_pattern)) > 0:
        print(f"Loading map from cache: {CACHE_PREFIX}")
        load_environment_from_osm_cache(
            environment=sim.environment,
            cache_dir=cache_dir,
            region_prefix=CACHE_PREFIX,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
            radius_m=RADIUS_M,
            default_height_m=10.0
        )
    else:
        print("Downloading map from OSM...")
        sim.setup_osm_environment(LOCATION, default_building_height=10.0, distance_m=RADIUS_M)
        
    # 3. Grid Setup (Optimized)
    # Larger cell size = fewer cells = faster calculation
    sim.enable_fire_simulation(
        grid_width_m=2*RADIUS_M,
        grid_height_m=2*RADIUS_M,
        cell_size_m=5.0,  
        dt=0.5            
    )

    save_real_satellite_imagery(LOCATION, RADIUS_M)
    
    exit(1)
    
    # Start Fire
    print("Igniting fire...")
    sim.start_fire((100, -100), intensity=0.5)

    # 4. Run Loop
    output_dir = 'output/fast_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    total_time = 3600*4
    save_interval = 1200*4
    
    sim_dt = sim.timestep
    steps_per_save = int(save_interval / sim_dt)
    total_steps = int(total_time / sim_dt)

    # 1. Define storage for snapshots
    snapshots = []
    
    print(f"Running simulation for {total_time}s...")
    
    frame = 0
    start_real_time = time.time()
    
    for step in range(total_steps):
        sim.step_simulation({})

        if step % steps_per_save == 0:
            current_sim_time = sim.simulation_time
            fire_state = sim.environment.get_fire_state()

            if fire_state:
                state = fire_state['fire_grid_state']
                burning_cells = np.sum(state['B'])

                print(f"Fuel of burning cells: {np.sum(state['F'][state['B']]) :.1f}")
                print(f"Frame {frame:03d} | Time: {current_sim_time:.0f}s | Burning Cells: {burning_cells}")
                windState = sim.environment.weather['wind_velocity']
                snapshots.append((sim.simulation_time, sim, state, windState))
                save_frame(sim, state, frame, current_sim_time, output_dir)
                frame += 1
    
    if len(snapshots) == 3:
        save_summary_figure(snapshots, output_dir)
    print(f"Done. Frames saved to {output_dir}/")

if __name__ == '__main__':
    run_fast_demo()