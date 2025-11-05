#!/usr/bin/env python3
"""
Demo 2: Water-based Fire Suppression
Shows how drone water drops increase moisture and prevent fire spread.
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


def run_suppression_demo():
    """Run firefighting demonstration with frame-by-frame output using REAL map data."""
    print("=" * 70)
    print("💧 DEMO 2: WATER SUPPRESSION (Real OSM Data)")
    print("=" * 70)
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Load REAL environment from OpenStreetMap
    # Using a forested area for realistic wildfire scenario
    sim.setup_osm_environment("Brno, Czech Republic", default_building_height=10.0)
    print("  Using real map data from Brno - forests will be actual forest locations!")
    
    # Enable fire
    sim.enable_fire_simulation(
        grid_width_m=800,  # Larger area to cover real terrain
        grid_height_m=800,
        cell_size_m=5.0
    )
    
    sim.set_wind([5.0, 0.0, 0.0])
    
    # Save initial environment visualization
    print("\n📸 Saving initial environment map...")
    sim.environment.save_environment_map('output/demo_02_environment.png', 
                                        show_fire_grid=True,
                                        detailed=False)  # Fast mode
    
    # Start fires in forested areas (adjust coordinates based on real map)
    # You may need to run once to see where forests are, then adjust
    sim.start_fire((100, 100), intensity=0.5)
    sim.start_fire((120, 80), intensity=0.5)
    sim.start_fire((80, 120), intensity=0.5)
    print("  🔥 Started 3 fires in different locations")
    print("      Note: Fire will only spread in forested areas from the real map!")
    
    # Add FIXED-WING firefighting drone with 5000L water tank (5 cubic meters - realistic)
    # Start on the LEFT side of the map
    # Higher max_thrust for faster flight
    sim.add_fixedwing("Firefighter", position=[-40, 0, 12], water_capacity=5000.0, max_thrust=80.0)
    sim.drones["Firefighter"].open_water_valve()  # OPEN VALVE IMMEDIATELY
    print("  ✈️  Fixed-wing firefighter deployed with 5000L water tank (5 m³)")
    print("      Water valve: OPEN from start (full firefighting mode)")
    print("      Starting position: LEFT side (-40, 0, 12)")
    print("      Water flow rate: 200 L/s (realistic aerial firefighting)")
    print("      Max thrust: 80N (fast response aircraft)")
    
    print()
    print("Running simulation for 20 seconds...")
    print("  Large circular patrol covering entire map")
    print("  Radius: 40m (covers most of the fire area)")
    print()
    
    # Create output directory
    output_dir = 'output/demo_02_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation
    frame = 0
    save_interval = 30  # Save every 0.5 seconds (30 steps at 60 FPS)
    
    total_steps = int(20 / sim.timestep)
    for step in range(total_steps):
        current_time = sim.simulation_time
        
        # Large circular patrol pattern - covers entire map
        # Period: 6 seconds for one complete circle (VERY FAST)
        phase = (current_time / 6.0) * 2 * np.pi
        
        # Start at LEFT (-40, 0) when phase=0, then go in a circle
        patrol_radius = 40.0  # Larger radius
        x = patrol_radius * np.cos(phase)
        y = patrol_radius * np.sin(phase)
        
        drone_pos = sim.drones["Firefighter"].get_position()
        target = np.array([x, y, 10])  # Target altitude: 10m for good coverage
        direction = target - drone_pos
        direction_norm = np.linalg.norm(direction[:2])
        
        # Calculate altitude error for better control
        altitude_error = target[2] - drone_pos[2]
        
        # Fixed-wing joystick: [turn, throttle, elevator]
        # Turn towards target, MAXIMUM SPEED for fast coverage
        if direction_norm > 0.1:
            turn_command = direction[1] / max(direction_norm, 1.0)  # Y direction for turning
            turn_command = np.clip(turn_command, -1.0, 1.0)
        else:
            turn_command = 0.0
        
        # STRONG elevator control to maintain altitude
        # Positive elevator = climb, Negative = dive
        # When drone is too high (altitude_error < 0), we need NEGATIVE elevator to dive down
        elevator_command = np.clip(altitude_error * 2.0, -1.0, 1.0)  # Strong correction
        
        controls = {
            "Firefighter": [turn_command, 1.0, elevator_command]  # FULL THROTTLE, altitude-controlled elevator
        }
        
        sim.step_simulation(controls)
        
        # Save frame
        if step % save_interval == 0:
            fire_state = sim.environment.get_fire_state()
            if fire_state:
                state = fire_state['fire_grid_state']
                burning = np.sum(state['B'])
                moisture = np.mean(state['M'])
                drone_pos = sim.drones["Firefighter"].get_position()
                water_remaining = sim.drones["Firefighter"].current_water
                
                save_suppression_frame(sim, state, drone_pos, frame, current_time, water_remaining, output_dir)
                
                phase_label = "FIREFIGHTING"
                print(f"  Frame {frame:03d} | t={current_time:4.1f}s | {phase_label:12s} | "
                      f"Burning: {burning:4d} | Moisture: {moisture:.3f} | Water: {water_remaining:.1f}L | "
                      f"Pos: ({drone_pos[0]:5.1f}, {drone_pos[1]:5.1f}, {drone_pos[2]:5.1f}) | "
                      f"Valve: {'OPEN' if sim.drones['Firefighter'].water_valve_open else 'CLOSED'}")
                frame += 1
    
    print()
    print(f"✅ Saved {frame} frames to {output_dir}/")
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
    ax1.plot(drone_pos[0], drone_pos[1], 'c^', markersize=15, 
            markeredgecolor='white', markeredgewidth=2, label='Drone', zorder=10)
    
    # Add wind arrow
    wind_vel = sim.environment.weather['wind_velocity']
    wind_x, wind_y = wind_vel[0], wind_vel[1]
    wind_speed = np.sqrt(wind_x**2 + wind_y**2)
    
    arrow_start_x = x_max - 15
    arrow_start_y = y_max - 10
    arrow_scale = 3.0
    
    ax1.arrow(arrow_start_x, arrow_start_y, 
             wind_x * arrow_scale, wind_y * arrow_scale,
             head_width=3, head_length=2, fc='cyan', ec='white', 
             linewidth=3, alpha=0.9, zorder=10)
    
    ax1.text(arrow_start_x, arrow_start_y - 5, 
            f'Wind: {wind_speed:.1f} m/s',
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
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
    ax2.plot(drone_pos[0], drone_pos[1], 'c^', markersize=15, 
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    
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
