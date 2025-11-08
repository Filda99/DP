#!/usr/bin/env python3
"""
Demo 4: Small Fire Suppression Test
Tests if drone can extinguish a small fire started right next to it.
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


def run_small_fire_test():
    """Test drone's ability to extinguish a small fire."""
    print("=" * 70)
    print("🔥 DEMO 4: SMALL FIRE SUPPRESSION TEST")
    print("=" * 70)
    
    # Configuration - Pec pod Sněžkou in Krkonoše
    LOCATION = "Pec pod Sněžkou, Czech Republic"
    CACHE_PREFIX = "Pec_pod_Sněžkou_Czechia"
    CENTER_LAT = 50.6868
    CENTER_LON = 15.7361
    RADIUS_M = 1500
    
    # Create simulation
    sim = Simulation()
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
    
    print(f"  Using real map data from {LOCATION}")
    
    # Enable fire
    print("\n⏱️  Initializing fire simulation grid...")
    t_start = time.time()
    
    fire_dt = 1.0  # Fire simulation runs at 1 Hz
    
    # REDUCED GRID SIZE - only bottom half (1500m instead of 3000m)
    sim.enable_fire_simulation(
        grid_width_m=200,  # Full width
        grid_height_m=200,  # HALF height - only bottom portion
        cell_size_m=15.0,
        dt=fire_dt
    )
    t_fire_init = time.time() - t_start
    print(f"✅ Fire grid initialized in {t_fire_init:.2f}s (fire_dt={fire_dt:.4f}s)")
    print(f"   Grid covers: X=[-1500, 1500], Y=[-750, 750] (bottom half only)")
    
    print("\n⏱️  Setting wind and mapping fuel from environment...")
    t_start = time.time()
    sim.set_wind([0, 0.0, 0.0])  # Light wind
    t_wind = time.time() - t_start
    print(f"✅ Wind and fuel mapping done in {t_wind:.2f}s")
    
    # Save initial environment visualization
    print("\n📸 Saving initial environment map...")
    t_start = time.time()
    sim.environment.save_environment_map('output/demo_04_environment.png', 
                                        show_fire_grid=True,
                                        detailed=False)
    t_viz = time.time() - t_start
    print(f"✅ Environment map saved in {t_viz:.2f}s")
    
    # Start SMALL FIRE in center-bottom area
    # Position at (0, -200) - bottom middle of the reduced grid
    fire_position = (0, -200)
    
    print(f"\n🔥 Starting SMALL fire in bottom center area...")
    print(f"   Target fire position: {fire_position}")
    
    # Find nearby forest cell to start fire
    fire_state = sim.environment.get_fire_state()
    fire_started = False
    actual_fire_pos = (0, -200)  # Default to (0, -200) if no fuel found
    
    if fire_state:
        state = fire_state['fire_grid_state']
        fuel_grid = state['F']
        H, W = fuel_grid.shape
        
        # Convert desired fire position to cell coordinates
        grid_mapper = sim.environment.grid_mapper
        fire_cell_i, fire_cell_j = grid_mapper.world_to_cell(fire_position)
        
        # Search for nearby high-fuel cell (within 10 cell radius)
        best_cell = None
        best_fuel = 0
        
        for di in range(-10, 11):
            for dj in range(-10, 11):
                test_i = fire_cell_i + di
                test_j = fire_cell_j + dj
                
                if 0 <= test_i < H and 0 <= test_j < W:
                    if fuel_grid[test_i, test_j] > best_fuel:
                        best_fuel = fuel_grid[test_i, test_j]
                        best_cell = (test_i, test_j)
        
        if best_cell and best_fuel > 0.3:
            cell_i, cell_j = best_cell
            world_x = (cell_j - grid_mapper.grid_width_cells // 2) * grid_mapper.cell_size_m
            world_y = (cell_i - grid_mapper.grid_height_cells // 2) * grid_mapper.cell_size_m
            
            sim.start_fire((world_x, world_y), intensity=0.3)
            fire_started = True
            actual_fire_pos = (world_x, world_y)
            
            print(f"✅ Fire started at ({world_x:.0f}, {world_y:.0f}) -> cell ({cell_i}, {cell_j})")
            print(f"   Fuel: {fuel_grid[cell_i, cell_j]:.2f}")
        else:
            print("  ⚠️  No fuel found near target! Starting at default (0, -200)...")
            sim.start_fire((0, -200), intensity=0.3)
            fire_started = True
            actual_fire_pos = (0, -200)
    
    if not fire_started:
        print("  ⚠️  Could not access fire state. Starting fire at default...")
        sim.start_fire((0, -200), intensity=0.3)
        actual_fire_pos = (0, -200)
    
    # Add TWO FIXED-WING firefighting drones
    # Drone 1: Circles around the fire (direct suppression)
    drone1_start_pos = [actual_fire_pos[0] + 10, actual_fire_pos[1], 20]
    sim.add_fixedwing("Circler", position=drone1_start_pos, water_capacity=5000.0, max_thrust=300.0)
    sim.drones["Circler"].open_water_valve()
    print("\n  ✈️  Drone 1 'Circler' deployed with 5000L water tank")
    print(f"      Starting position: ({drone1_start_pos[0]:.1f}, {drone1_start_pos[1]:.1f}, {drone1_start_pos[2]:.1f})")
    print(f"      Mission: Circle around fire (radius=100m)")
    print("      Water valve: OPEN from start")
    
    # Drone 2: Flies straight toward fire from the right
    firebreak_distance = 50  # meters to the right of fire
    drone2_start_pos = [actual_fire_pos[0] + firebreak_distance, actual_fire_pos[1], 20]
    
    sim.add_fixedwing("Firebreak", position=drone2_start_pos, water_capacity=5000.0, max_thrust=300.0)
    sim.drones["Firebreak"].open_water_valve()
    print("\n  ✈️  Drone 2 'Firebreak' deployed with 5000L water tank")
    print(f"      Starting position: ({drone2_start_pos[0]:.1f}, {actual_fire_pos[1]:.1f}, 20.0)")
    print(f"      Mission: Fly straight toward fire from the right")
    print("      Water valve: OPEN from start")
    
    print()
    print("Running simulation for 60 seconds...")
    print("  Drone 1 (Circler): Circles around fire")
    print("  Drone 2 (Firebreak): Flies straight toward fire")
    print()
    print("=" * 140)
    print(f"{'Frame':>6} | {'Time':>6} | {'Phase':^12} | {'Burn':>5} | {'Moisture':^20} | {'Water':>8} | {'Valve':^15} | {'Save':>6}")
    print("=" * 140)
    
    # Create output directory
    output_dir = 'output/demo_04_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation
    frame = 0
    save_interval = 180  # Save every 3 seconds (180 steps at 60 FPS)
    
    total_steps = int(60 / sim.timestep)  # 60 seconds
    
    # Timing variables
    t_simulation_start = time.time()
    t_step_total = 0.0
    t_save_total = 0.0
    
    for step in range(total_steps):
        t_step_start = time.time()
        current_time = sim.simulation_time
        
        # === DRONE 1 (Circler): Circular patrol around fire ===
        phase_circle = (current_time / 15.0) * 2 * np.pi
        
        center_x, center_y = actual_fire_pos[0], actual_fire_pos[1]
        radius_circle = 200.0

        target1_x = center_x + radius_circle * np.cos(phase_circle)
        target1_y = center_y + radius_circle * np.sin(phase_circle)
        
        drone1_pos = sim.drones["Circler"].get_position()
        
        # Calculate direction to target
        dx1 = target1_x - drone1_pos[0]
        dy1 = target1_y - drone1_pos[1]
        distance1 = np.sqrt(dx1**2 + dy1**2)
        
        # Desired heading towards target
        desired_heading1 = np.arctan2(dy1, dx1)
        current_heading1 = sim.drones["Circler"].current_heading
        
        # Calculate heading error (handle wrap-around)
        heading_error1 = desired_heading1 - current_heading1
        while heading_error1 > np.pi:
            heading_error1 -= 2 * np.pi
        while heading_error1 < -np.pi:
            heading_error1 += 2 * np.pi
        
        # Turn command
        turn_command1 = np.clip(heading_error1 * 2.0, -1.0, 1.0)
        throttle_command1 = 1.0
        
        # Altitude control - maintain 20m
        altitude_error1 = 20.0 - drone1_pos[2]
        elevator_command1 = np.clip(altitude_error1 * 0.5, -1.0, 1.0)
        
        # === DRONE 2 (Firebreak): Fly straight toward fire ===
        # Target is the fire position
        target2_x = actual_fire_pos[0]
        target2_y = actual_fire_pos[1]
        
        drone2_pos = sim.drones["Firebreak"].get_position()
        
        dx2 = target2_x - drone2_pos[0]
        dy2 = target2_y - drone2_pos[1]
        distance2 = np.sqrt(dx2**2 + dy2**2)
        
        desired_heading2 = np.arctan2(dy2, dx2)
        current_heading2 = sim.drones["Firebreak"].current_heading
        
        heading_error2 = desired_heading2 - current_heading2
        while heading_error2 > np.pi:
            heading_error2 -= 2 * np.pi
        while heading_error2 < -np.pi:
            heading_error2 += 2 * np.pi
        
        turn_command2 = np.clip(heading_error2 * 2.0, -1.0, 1.0)
        throttle_command2 = 1.0
        
        altitude_error2 = 20.0 - drone2_pos[2]
        elevator_command2 = np.clip(altitude_error2 * 0.5, -1.0, 1.0)
        
        controls = {
            "Circler": [turn_command1, throttle_command1, elevator_command1],
            "Firebreak": [turn_command2, throttle_command2, elevator_command2]
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
                # Calculate moisture stats
                wet_cells = np.sum(state['M'] > 0)
                if wet_cells > 0:
                    moisture_mean = np.mean(state['M'][state['M'] > 0])
                    moisture_str = f"{moisture_mean:.3f} ({wet_cells} cells)"
                else:
                    moisture_str = "0.000 (0 cells)"
                
                # Get both drone positions
                drone1_pos_save = sim.drones["Circler"].get_position()
                drone2_pos_save = sim.drones["Firebreak"].get_position()
                water1 = sim.drones["Circler"].current_water
                water2 = sim.drones["Firebreak"].current_water
                
                # Pass list of drone positions to save function
                drone_positions = [drone1_pos_save, drone2_pos_save]
                save_suppression_frame(sim, state, drone_positions, frame, current_time, water1 + water2, output_dir)
                
                t_save_elapsed = time.time() - t_save_start
                t_save_total += t_save_elapsed
                
                phase_label = "SUPPRESSION"
                valve1_status = 'OPEN' if sim.drones['Circler'].water_valve_open else 'SHUT'
                valve2_status = 'OPEN' if sim.drones['Firebreak'].water_valve_open else 'SHUT'
                print(f"  {frame:>4d} | {current_time:6.1f}s | {phase_label:^12s} | "
                      f"{burning:5d} | {moisture_str:^20s} | "
                      f"W1:{water1:5.0f}L W2:{water2:5.0f}L | "
                      f"{valve1_status}/{valve2_status:^13s} | "
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


def save_suppression_frame(sim, state, drone_positions, frame_num, time, water_remaining, output_dir):
    """Save a single frame with 3 panels: fire state, moisture, water level.
    
    Args:
        drone_positions: List of (x, y, z) tuples for each drone
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    H, W = state['B'].shape
    
    # === PANEL 1: Fire state with drones ===
    img = np.zeros((H, W, 3))
    
    # Base terrain (forest green)
    img[:, :] = [0.1, 0.4, 0.1]
    
    # Show burned-out areas (grey)
    fuel = state['F']
    burned_out = (fuel < 0.3) & (~state['B'])
    img[burned_out] = [0.4, 0.4, 0.4]
    
    # Overlay burning cells (RED)
    burning = state['B']
    intensity = state['I']
    img[burning] = np.stack([
        np.ones_like(intensity[burning]),
        intensity[burning] * 0.3,
        np.zeros_like(intensity[burning])
    ], axis=-1)
    
    ax1.imshow(img, origin='lower', extent=[x_min, x_max, y_min, y_max])
    
    # Add drone position markers
    colors = ['cyan', 'magenta']  # Different colors for each drone
    labels = ['Circler', 'Firebreak']
    for i, drone_pos in enumerate(drone_positions):
        color = colors[i] if i < len(colors) else 'white'
        label = labels[i] if i < len(labels) else f'Drone {i+1}'
        ax1.plot(drone_pos[0], drone_pos[1], 'o', markersize=3, color=color,
                markeredgecolor='white', markeredgewidth=1, label=label, zorder=10)
    
    # Add wind arrow
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
    
    # === PANEL 3: Water tank status (combined for both drones) ===
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Use total capacity from both drones
    water_capacity = sim.drones["Circler"].water_capacity + sim.drones["Firebreak"].water_capacity
    water_fraction = water_remaining / water_capacity if water_capacity > 0 else 0
    valve1_open = sim.drones["Circler"].water_valve_open
    valve2_open = sim.drones["Firebreak"].water_valve_open
    
    # Tank outline
    tank_x = 0.3
    tank_y = 0.2
    tank_width = 0.4
    tank_height = 0.6
    
    tank_rect = plt.Rectangle((tank_x, tank_y), tank_width, tank_height,
                              linewidth=3, edgecolor='black', facecolor='lightgray')
    ax3.add_patch(tank_rect)
    
    # Water level
    water_height = tank_height * water_fraction
    water_rect = plt.Rectangle((tank_x, tank_y), tank_width, water_height,
                               facecolor='blue', alpha=0.6)
    ax3.add_patch(water_rect)
    
    # Labels
    ax3.text(0.5, 0.9, 'WATER TANK (Combined)', ha='center', fontsize=16, fontweight='bold')
    ax3.text(0.5, 0.12, f'{water_remaining:.1f}L / {water_capacity:.1f}L', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Show both valve statuses
    valve_status = f'D1: {"OPEN" if valve1_open else "CLOSED"} | D2: {"OPEN" if valve2_open else "CLOSED"}'
    valve_color = 'green' if (valve1_open or valve2_open) else 'red'
    ax3.text(0.5, 0.05, f'Valves: {valve_status}',
            ha='center', fontsize=12, fontweight='bold',
            color=valve_color)
    
    ax3.set_title(f'Firefighting System (t={time:.1f}s)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frame_{frame_num:03d}.png', dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_small_fire_test()
