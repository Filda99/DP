#!/usr/bin/env python3
"""
Demo 2: Water-based Fire Suppression
Shows how drone water drops increase moisture and prevent fire spread.
"""

import numpy as np
import sys
import os
import glob
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache


def run_suppression_demo():
    """Run firefighting demonstration with frame-by-frame output using REAL map data."""
    print("=" * 70)
    print("💧 DEMO 2: WATER SUPPRESSION (Real OSM Data)")
    print("=" * 70)
    
    # Configuration - Pec pod Sněžkou in Krkonoše (SAME AS DEMO 01)
    LOCATION = "Pec pod Sněžkou, Czech Republic"
    CACHE_PREFIX = "Pec_pod_Sněžkou_Czechia"
    CENTER_LAT = 50.6868
    CENTER_LON = 15.7361
    RADIUS_M = 1500  # Same as demo 01 to get identical environment
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Load environment - use cache if available
    cache_dir = "data"
    cache_pattern = f"{cache_dir}/{CACHE_PREFIX}_building_*.gpkg"
    use_cache = len(glob.glob(cache_pattern)) > 0
    
    if use_cache:
        print(f"📂 Loading from cache: {CACHE_PREFIX}")
        load_environment_from_osm_cache(
            environment=sim.environment,
            cache_dir=cache_dir,
            region_prefix=CACHE_PREFIX,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
            radius_m=RADIUS_M,
            default_height_m=10.0,
            use_city_boundaries=True
        )
    else:
        print(f"🌍 No cache found, downloading from OSM...")
        sim.setup_osm_environment(LOCATION, default_building_height=10.0, distance_m=RADIUS_M)
    
    print(f"  Using real map data from {LOCATION} - forests will be actual forest locations!")
    
    # Enable fire - SAME AS DEMO 01
    print("\n⏱️  Initializing fire simulation grid...")
    t_start = time.time()
    
    # SLOW FIRE SPREAD: Use larger dt and lower spread rate
    # Instead of fire spreading every 0.016s (60 FPS), make it spread every 1s
    fire_dt = 1.0  # Fire simulation runs at 1 Hz instead of 60 Hz
    
    sim.enable_fire_simulation(
        grid_width_m=3000,  # Same as demo 01
        grid_height_m=3000,
        cell_size_m=15.0,
        dt=fire_dt  # Fire spreads much slower!
    )
    t_fire_init = time.time() - t_start
    print(f"✅ Fire grid initialized in {t_fire_init:.2f}s (fire_dt={fire_dt:.4f}s)")
    
    print("\n⏱️  Setting wind and mapping fuel from environment...")
    t_start = time.time()
    sim.set_wind([8.0, 0.0, 0.0])  # Same wind as demo 01
    t_wind = time.time() - t_start
    print(f"✅ Wind and fuel mapping done in {t_wind:.2f}s")
    
    # Save initial environment visualization
    print("\n📸 Saving initial environment map...")
    t_start = time.time()
    sim.environment.save_environment_map('output/demo_02_environment.png', 
                                        show_fire_grid=True,
                                        detailed=False)  # Fast mode
    t_viz = time.time() - t_start
    print(f"✅ Environment map saved in {t_viz:.2f}s")
    
    # Find a location with fuel to start the fire - BOTTOM MIDDLE area (SAME AS DEMO 01)
    print("\n🔍 Finding forested area in bottom middle to start fire...")
    fire_started = False
    fire_state = sim.environment.get_fire_state()
    
    if fire_state:
        state = fire_state['fire_grid_state']
        fuel_grid = state['F']
        
        # Find cells with high fuel in BOTTOM MIDDLE area
        # Bottom half: cell_i < H/2, Middle: cell_j near W/2
        H, W = fuel_grid.shape
        bottom_middle_cells = []
        
        for cell_i in range(H // 4, H // 2):  # Bottom quarter to middle
            for cell_j in range(W // 3, 2 * W // 3):  # Middle third horizontally
                if fuel_grid[cell_i, cell_j] > 0.7:  # High fuel
                    bottom_middle_cells.append((cell_i, cell_j))
        
        if len(bottom_middle_cells) > 0:
            # Choose a random high-fuel cell in bottom middle
            idx = np.random.randint(len(bottom_middle_cells))
            cell_i, cell_j = bottom_middle_cells[idx]
            
            # Convert cell coordinates to world coordinates
            grid_mapper = sim.environment.grid_mapper
            world_x = (cell_j - grid_mapper.grid_width_cells // 2) * grid_mapper.cell_size_m
            world_y = (cell_i - grid_mapper.grid_height_cells // 2) * grid_mapper.cell_size_m
            
            sim.start_fire((world_x, world_y), intensity=0.3)
            fire_started = True
            print(f"✅ Started fire at world pos ({world_x:.0f}, {world_y:.0f}) -> cell ({cell_i}, {cell_j})")
            print(f"     Cell: ({cell_i}, {cell_j}), Fuel: {fuel_grid[cell_i, cell_j]:.2f}")
        else:
            print("  ⚠️  No high-fuel areas found in bottom middle! Using fallback...")
            sim.start_fire((-1470, -150), intensity=0.3)
            fire_started = True
    
    if not fire_started:
        print("  ⚠️  Could not access fire state. Starting fire at default location...")
        sim.start_fire((-1470, -150), intensity=0.3)
    
    print("  🔥 Fire started in bottom middle - same location as demo 01")
    
    # Add FIXED-WING firefighting drone with 5000L water tank
    # Start ABOVE the fire area in bottom middle - LOWER altitude for better water coverage
    sim.add_fixedwing("Firefighter", position=[0, -800, 20], water_capacity=5000.0, max_thrust=300.0)
    sim.drones["Firefighter"].open_water_valve()  # OPEN VALVE IMMEDIATELY
    print("  ✈️  Fixed-wing firefighter deployed with 5000L water tank (5 m³)")
    print("      Water valve: OPEN from start (full firefighting mode)")
    print("      Starting position: Above fire (0, -800, 20m)")
    print("      Water flow rate: 200 L/s (realistic aerial firefighting)")
    print("      Max thrust: 300N (high-speed firefighting aircraft)")
    
    print()
    print("Running simulation for 120 seconds...")
    print("  Circular patrol over fire area in bottom middle (LOW ALTITUDE)")
    print()
    print("=" * 140)
    print(f"{'Frame':>6} | {'Time':>6} | {'Phase':^12} | {'Burn':>5} | {'Moisture':^20} | {'Water':>8} | {'Position (X,Y,Z)':^24} | {'Valve':^5} | {'Save':>6}")
    print("=" * 140)
    
    # Create output directory
    output_dir = 'output/demo_02_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation
    frame = 0
    save_interval = 300  # Save every 5 seconds (300 steps at 60 FPS) for slower, more realistic progression
    
    total_steps = int(120 / sim.timestep)  # 120 seconds for more realistic fire spread
    
    # Timing variables
    t_simulation_start = time.time()
    t_step_total = 0.0
    t_save_total = 0.0
    
    for step in range(total_steps):
        t_step_start = time.time()
        current_time = sim.simulation_time
        
        # SIMPLIFIED circular patrol - just send the drone in a circle
        # Period: 30 seconds for one complete circle (adjusted for 120s total simulation)
        phase = (current_time / 30.0) * 2 * np.pi
        
        # Circle centered at (0, 0) with radius 2000m
        center_x, center_y = 0.0, 0.0
        radius = 2000.0

        target_x = center_x + radius * np.cos(phase)
        target_y = center_y + radius * np.sin(phase)
        
        drone_pos = sim.drones["Firefighter"].get_position()
        
        # Calculate direction to target
        dx = target_x - drone_pos[0]
        dy = target_y - drone_pos[1]
        
        # Simple turn control - just turn towards the target
        # Positive turn = turn left (counterclockwise)
        # Negative turn = turn right (clockwise)
        cross_product = dx * np.sin(sim.drones["Firefighter"].current_heading) - dy * np.cos(sim.drones["Firefighter"].current_heading)
        turn_command = np.clip(cross_product * 0.1, -1.0, 1.0)  # Proportional control
        
        # Keep constant throttle and altitude
        throttle_command = 1.0  # Full throttle
        
        # Altitude control - maintain 20m
        target_altitude = 20.0
        altitude_error = target_altitude - drone_pos[2]
        elevator_command = np.clip(altitude_error * 0.5, -1.0, 1.0)
        
        controls = {
            "Firefighter": [turn_command, throttle_command, elevator_command]
        }
        
        sim.step_simulation(controls)
        t_step_total += time.time() - t_step_start
        
        # Save frame
        if step % save_interval == 0:
            t_save_start = time.time()
            fire_state = sim.environment.get_fire_state()
            if fire_state:
                state = fire_state['fire_grid_state']
                burning = np.sum(state['B'])
                # Calculate moisture stats: mean of wet cells (M > 0) and count
                wet_cells = np.sum(state['M'] > 0)
                if wet_cells > 0:
                    moisture_mean = np.mean(state['M'][state['M'] > 0])
                    moisture_str = f"{moisture_mean:.3f} ({wet_cells} cells)"
                else:
                    moisture_str = "0.000 (0 cells)"
                
                drone_pos = sim.drones["Firefighter"].get_position()
                water_remaining = sim.drones["Firefighter"].current_water
                
                save_suppression_frame(sim, state, drone_pos, frame, current_time, water_remaining, output_dir)
                
                t_save_elapsed = time.time() - t_save_start
                t_save_total += t_save_elapsed
                
                phase_label = "FIREFIGHTING"
                print(f"  {frame:>4d} | {current_time:6.1f}s | {phase_label:^12s} | "
                      f"{burning:5d} | {moisture_str:^20s} | "
                      f"{water_remaining:7.1f}L | "
                      f"({drone_pos[0]:6.1f}, {drone_pos[1]:6.1f}, {drone_pos[2]:4.1f}) | "
                      f"{'OPEN' if sim.drones['Firefighter'].water_valve_open else 'SHUT':^5s} | "
                      f"{t_save_elapsed:5.1f}s")
                frame += 1
    
    t_simulation_total = time.time() - t_simulation_start
    avg_step_time = t_step_total / total_steps if total_steps > 0 else 0
    
    print("=" * 140)
    print(f"✅ Saved {frame} frames to {output_dir}/")
    print()
    print(f"⏱️  Timing Summary:")
    print(f"   Total simulation time: {t_simulation_total:.2f}s")
    print(f"   Physics steps: {t_step_total:.2f}s (avg: {avg_step_time*1000:.2f}ms/step)")
    print(f"   Frame saving: {t_save_total:.2f}s (avg: {t_save_total/frame if frame > 0 else 0:.2f}s/frame)")
    print(f"   Overhead: {t_simulation_total - t_step_total - t_save_total:.2f}s")
    print("=" * 70)


def save_suppression_frame(sim, state, drone_pos, frame_num, time, water_remaining, output_dir):
    """Save a single frame with 3 panels: fire state, moisture, water level."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    H, W = state['B'].shape
    
    # === PANEL 1: Fire state with drone ===
    img = np.zeros((H, W, 3))
    
    # Base terrain (forest green)
    img[:, :] = [0.1, 0.4, 0.1]
    
    # Show burned-out areas (grey) - where fuel was consumed
    fuel = state['F']
    burned_out = (fuel < 0.3) & (~state['B'])  # Low fuel and not burning
    img[burned_out] = [0.4, 0.4, 0.4]  # Grey for burned-out areas
    
    # Overlay burning cells (RED)
    burning = state['B']
    intensity = state['I']
    img[burning] = np.stack([
        np.ones_like(intensity[burning]),
        intensity[burning] * 0.3,
        np.zeros_like(intensity[burning])
    ], axis=-1)
    
    ax1.imshow(img, origin='lower', extent=[x_min, x_max, y_min, y_max])
    
    # Add drone position marker
    ax1.plot(drone_pos[0], drone_pos[1], 'o', markersize=3, 
            markeredgecolor='white', markeredgewidth=1, label='Drone', zorder=10)
    
    # Add wind arrow (with visible colors)
    wind_vel = sim.environment.weather['wind_velocity']
    wind_x, wind_y = wind_vel[0], wind_vel[1]
    wind_speed = np.sqrt(wind_x**2 + wind_y**2)
    
    arrow_start_x = x_max - 200
    arrow_start_y = y_max - 150
    arrow_scale = 20.0
    
    ax1.arrow(arrow_start_x, arrow_start_y, 
             wind_x * arrow_scale, wind_y * arrow_scale,
             head_width=50, head_length=40, fc='yellow', ec='black', 
             linewidth=2, alpha=0.95, zorder=10)
    
    ax1.text(arrow_start_x, arrow_start_y - 100, 
            f'Wind: {wind_speed:.1f} m/s',
            color='yellow', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='yellow', linewidth=2),
            ha='center', zorder=11)
    
    ax1.set_title(f'Fire State (t={time:.1f}s)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # === PANEL 2: Moisture field ===
    moisture = state['M']
    im2 = ax2.imshow(moisture, origin='lower', cmap='Blues', 
                     extent=[x_min, x_max, y_min, y_max], 
                     vmin=0, vmax=1.0)
    
    # Add drone position
    ax2.plot(drone_pos[0], drone_pos[1], 'o', markersize=3, 
            markeredgecolor='white', markeredgewidth=1, zorder=10)
    
    ax2.set_title(f'Moisture Field (mean={np.mean(moisture):.3f})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Moisture (0-1)')
    
    # === PANEL 3: Water tank status ===
    # Draw water tank gauge
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Water tank capacity and current level
    water_capacity = sim.drones["Firefighter"].water_capacity
    water_fraction = water_remaining / water_capacity if water_capacity > 0 else 0
    valve_open = sim.drones["Firefighter"].water_valve_open
    
    # Tank outline
    tank_x = 0.3
    tank_y = 0.2
    tank_width = 0.4
    tank_height = 0.6
    
    # Draw tank outline
    tank_rect = plt.Rectangle((tank_x, tank_y), tank_width, tank_height,
                              linewidth=3, edgecolor='black', facecolor='lightgray')
    ax3.add_patch(tank_rect)
    
    # Draw water level
    water_height = tank_height * water_fraction
    water_rect = plt.Rectangle((tank_x, tank_y), tank_width, water_height,
                               facecolor='blue', alpha=0.6)
    ax3.add_patch(water_rect)
    
    # Labels
    ax3.text(0.5, 0.9, 'WATER TANK', ha='center', fontsize=16, fontweight='bold')
    ax3.text(0.5, 0.12, f'{water_remaining:.1f}L / {water_capacity:.1f}L', 
            ha='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.05, f'Valve: {"OPEN 💧" if valve_open else "CLOSED 🔒"}',
            ha='center', fontsize=12, fontweight='bold',
            color='green' if valve_open else 'red')
    
    ax3.set_title(f'Firefighting System (t={time:.1f}s)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frame_{frame_num:03d}.png', dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_suppression_demo()
