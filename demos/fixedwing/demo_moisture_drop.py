import numpy as np
import sys
import os
import glob
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache

def run_suppression_demo():
    """Run firefighting demonstration with frame-by-frame output using REAL map data."""
    print("=" * 70)
    print("💧 DEMO 2: WATER SUPPRESSION (Real OSM Data)")
    print("=" * 70)
    
    # Configuration
    LOCATION = "Pec pod Sněžkou, Czech Republic"
    CACHE_PREFIX = "Pec_pod_Sněžkou_Czechia"
    CENTER_LAT = 50.6868
    CENTER_LON = 15.7361
    RADIUS_M = 1500
    
    # 1. Initialize Simulation
    sim = Simulation()
    sim.start_simulation()
    
    # 2. Load Environment
    cache_dir = "data"
    cache_pattern = f"{cache_dir}/{CACHE_PREFIX}_building_*.gpkg"
    
    if len(glob.glob(cache_pattern)) > 0:
        print(f"📂 Loading from cache: {CACHE_PREFIX}")
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
        print(f"🌍 No cache found, downloading from OSM...")
        sim.setup_osm_environment(LOCATION, default_building_height=10.0, distance_m=RADIUS_M)
    
    # 3. Enable Fire Simulation (Slower time scale for visualization)
    print("\n⏱️  Initializing fire simulation grid...")
    fire_dt = 0.5  # Run fire physics at 2Hz equivalent per step for faster visual spread
    
    sim.enable_fire_simulation(
        grid_width_m=3000,
        grid_height_m=3000,
        cell_size_m=15.0,
        dt=fire_dt
    )
    
    # 4. Set Wind
    sim.set_wind([8.0, 5.0, 0.0]) # Wind pushing North-East
    
    # 5. Start Fire (Bottom Middle Forest)
    print("\n🔍 Finding forest area to ignite...")
    
    sim.start_fire((67.5, -517.5), intensity=0.5)
    fire_center = np.array([67.5, -517.5])

    # 6. Deploy Firefighter Drone
    # Start offset from fire to begin patrol
    start_pos = [fire_center[0] - 50, fire_center[1] - 50, 80] 
    
    # FixedWing with 5000L tank
    sim.add_fixedwing("Firefighter", position=start_pos, mass=50.0, water_capacity=5000.0)
    print(f"✈️  Deployed 'Firefighter' at {start_pos} with 5000L water.")

    # 7. Simulation Loop
    output_dir = 'output/demo_02_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    total_time = 120.0 # seconds
    steps = int(total_time / sim.timestep)
    save_interval = 300 # Save every 5 seconds (60Hz * 5)
    
    print(f"\n🚀 Running simulation for {total_time}s...")
    
    frame_count = 0
    
    for step in range(steps):
        # --- Control Logic (Circular Patrol) ---
        drone = sim.drones["Firefighter"]
        pos = drone.get_position()
        vel = drone.get_velocity()
        speed = np.linalg.norm(vel)
        heading = np.arctan2(vel[1], vel[0])
        
        # Calculate target point on circle around fire
        # Period = 40s
        phase = (sim.simulation_time / 40.0) * 2 * np.pi
        radius = 400.0
        target_x = fire_center[0] + radius * np.cos(phase)
        target_y = fire_center[1] + radius * np.sin(phase)
        
        # Guidance
        dx = target_x - pos[0]
        dy = target_y - pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        target_heading = np.arctan2(dy, dx)
        
        # Heading Error (shortest path)
        heading_err = target_heading - heading
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        
        # Inputs: [Roll, Throttle, Pitch, Water]
        # Roll: Proportional to heading error
        roll_cmd = np.clip(heading_err * 2.0, -1.0, 1.0)
        
        # Pitch: Maintain 60m altitude
        alt_err = 60.0 - pos[2]
        pitch_cmd = np.clip(alt_err * 0.1, -0.5, 0.5)
        
        # Throttle: Cruise
        throttle_cmd = 0.7 
        
        # Water: ALWAYS OPEN for this demo
        water_cmd = 1.0
        
        sim.step_simulation({"Firefighter": [roll_cmd, throttle_cmd, pitch_cmd, water_cmd]})
        
        # --- Visualization ---
        if step % save_interval == 0:
            fire_state = sim.environment.get_fire_state()
            save_suppression_frame(sim, fire_state['fire_grid_state'], drone, frame_count, sim.simulation_time, output_dir)
            
            # Print status
            wet_cells = np.sum(fire_state['fire_grid_state']['M'] > 0.01)
            burned = np.sum(fire_state['fire_grid_state']['B'])
            print(f"Frame {frame_count:03d} | T={sim.simulation_time:5.1f}s | Water: {drone.current_water:4.0f}L | Wet Cells: {wet_cells} | Burning: {burned}")
            frame_count += 1
            
    print(f"\n✅ Simulation complete. Frames saved to {output_dir}")

def save_suppression_frame(sim, state, drone, frame_num, time, output_dir):
    """Save a 3-panel dashboard frame."""
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.6])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Get Bounds
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    extent = [x_min, x_max, y_min, y_max]
    
    # === PANEL 1: FIRE STATE ===
    # RGB Map construction
    H, W = state['F'].shape
    rgb = np.zeros((H, W, 3))
    
    # 1. Background (Fuel Map)
    fuel = state['F']
    # Forest (Green) vs Open (Light Green)
    rgb[:, :, 0] = 0.2 + (0.6 - fuel) * 0.2  # R
    rgb[:, :, 1] = 0.5 + fuel * 0.2         # G
    rgb[:, :, 2] = 0.2                      # B
    
    # 2. Burned (Grey)
    burned_out = (fuel < 0.2) & (~state['B'])
    rgb[burned_out] = [0.3, 0.3, 0.3]
    
    # 3. Fire (Red/Orange)
    burning = state['B']
    if np.any(burning):
        intensity = state['I'][burning]
        # Overlay red based on intensity
        rgb[burning, 0] = 1.0
        rgb[burning, 1] = 1.0 - intensity # White center for hot fire
        rgb[burning, 2] = 0.0

    ax1.imshow(rgb, origin='lower', extent=extent)
    
    # Drone & Wind
    dpos = drone.get_position()
    ax1.plot(dpos[0], dpos[1], 'cyan', marker='P', markersize=10, markeredgecolor='black', label='Drone')
    
    # Wind Arrow
    w = sim.environment.weather['wind_velocity']
    ax1.arrow(x_min+200, y_max-200, w[0]*20, w[1]*20, head_width=50, color='yellow', width=10)
    ax1.text(x_min+200, y_max-300, f"Wind {np.linalg.norm(w):.1f}m/s", color='yellow', fontweight='bold')
    
    ax1.set_title(f"Fire Status (T={time:.1f}s)")
    ax1.set_xlabel("Meters")
    ax1.set_ylabel("Meters")

    # === PANEL 2: MOISTURE FIELD ===
    moisture = state['M']
    # Use a custom colormap where 0 is transparent/terrain color
    cmap = plt.cm.Blues
    cmap.set_under('white', alpha=0)
    
    # Plot base terrain first for context
    ax2.imshow(state['F'], origin='lower', extent=extent, cmap='Greens', alpha=0.3)
    # Overlay moisture
    im2 = ax2.imshow(moisture, origin='lower', extent=extent, cmap='Blues', vmin=0.01, vmax=1.0)
    
    ax2.plot(dpos[0], dpos[1], 'cyan', marker='P', markersize=10, markeredgecolor='black')
    plt.colorbar(im2, ax=ax2, label='Soil Saturation')
    ax2.set_title("Moisture Dispersion")
    
    # === PANEL 3: TANK GAUGE ===
    ax3.axis('off')
    
    # Draw Tank
    cap = drone.water_capacity
    curr = drone.current_water
    pct = curr / cap if cap > 0 else 0
    
    # Outline
    rect_border = plt.Rectangle((0.3, 0.1), 0.4, 0.8, fill=False, linewidth=3)
    ax3.add_patch(rect_border)
    
    # Fill
    rect_fill = plt.Rectangle((0.3, 0.1), 0.4, 0.8 * pct, facecolor='blue', alpha=0.7)
    ax3.add_patch(rect_fill)
    
    # Text
    ax3.text(0.5, 0.95, "WATER TANK", ha='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.05, f"{curr:.0f} / {cap:.0f} L", ha='center', fontsize=12)
    
    status_color = 'green' if drone.water_valve_open else 'red'
    status_text = "VALVE OPEN" if drone.water_valve_open else "VALVE CLOSED"
    ax3.text(0.5, -0.05, status_text, ha='center', color=status_color, fontweight='bold', fontsize=12)

    plt.savefig(f"{output_dir}/frame_{frame_num:03d}.png", dpi=80)
    plt.close()

if __name__ == "__main__":
    run_suppression_demo()