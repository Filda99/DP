#!/usr/bin/env python3
"""
Demo 3: Advanced Physics Model
Demonstrates temperature-dependent density, terrain-dependent fuel, and moisture suppression.
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


def run_physics_demo():
    """Run physics demonstration with frame-by-frame output."""
    print("=" * 70)
    print("🌡️  DEMO 3: ADVANCED PHYSICS")
    print("=" * 70)
    print()
    print("Physics features:")
    print("  • Terrain-dependent fuel burn rates")
    print("  • Temperature grid (fire heats air)")
    print("  • Air density: ρ = ρ₀ × (T₀ / T)")
    print("  • Moisture prevents ignition")
    print()
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # NO special terrain - all open (fast burning)
    # sim.environment.add_forest_area(center=[0, 0], radius=25)
    
    # Enable fire - SMALLER grid
    sim.enable_fire_simulation(
        grid_width_m=60,
        grid_height_m=60,
        cell_size_m=2.0
    )
    
    sim.set_wind([4.0, 0.0, 0.0])
    
    # Start MULTIPLE fires to ensure it spreads - HIGHER INTENSITY
    sim.start_fire((0, 0), intensity=0.8)
    sim.start_fire((4, 0), intensity=0.8)
    sim.start_fire((-4, 0), intensity=0.8)
    sim.start_fire((0, 4), intensity=0.8)
    sim.start_fire((0, -4), intensity=0.8)
    print("  🔥 5 intense fires started in center area")
    
    # Add drones SIDE BY SIDE closer to fire (offset so both visible from above)
    sim.add_quadcopter("Low_Altitude", position=[-2, 0, 8])
    sim.add_quadcopter("High_Altitude", position=[2, 0, 25])
    print("  🚁 Low drone at (-2, 0, 8m) - in hot air")
    print("  🚁 High drone at (2, 0, 25m) - in cooler air")
    print("  🚁 Quadcopter at 8m (above fire - hot air)")
    print("  🚁 Quadcopter at 25m (high - cooler air)")
    
    print()
    print("Running simulation for 20 seconds...")
    print("  Phase 1 (0-5s): Fire spreads rapidly in open terrain")
    print("  Phase 2 (5-20s): Low drone drops water - watch temperature drop!")
    print()
    
    # Create output directory
    output_dir = 'output/demo_03_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data collection
    physics_data = {
        'time': [],
        'temp_low': [],
        'temp_high': [],
        'density_low': [],
        'density_high': [],
        'altitude_low': [],
        'altitude_high': []
    }
    
    # Run simulation
    frame = 0
    save_interval = 30  # Save every 0.5 seconds (30 steps at 60 FPS)
    
    total_steps = int(20 / sim.timestep)
    for step in range(total_steps):
        current_time = sim.simulation_time
        
        # PID control to hold drones in position
        low_pos = sim.drones["Low_Altitude"].get_position()
        high_pos = sim.drones["High_Altitude"].get_position()
        
        low_target = np.array([-2.0, 0.0, 8.0])
        high_target = np.array([2.0, 0.0, 25.0])
        
        low_error = low_target - low_pos
        high_error = high_target - high_pos
        
        # Phase 1 (0-5s): Just observe fire
        # Phase 2 (5-20s): Low drone fights fire with water
        if current_time < 5:
            controls = {
                "Low_Altitude": low_error * 0.5,
                "High_Altitude": high_error * 0.5
            }
        else:
            # Low drone drops to 5m and drops water
            low_target[2] = 5.0  # Lower for effective water drops
            low_error = low_target - low_pos
            controls = {
                "Low_Altitude": low_error * 0.5,  # Active firefighting
                "High_Altitude": high_error * 0.5  # Keep observing
            }
        
        sim.step_simulation(controls)
        
        # Collect physics data every step
        if step % 6 == 0:  # Every 0.1s
            low_pos = sim.drones["Low_Altitude"].get_position()
            high_pos = sim.drones["High_Altitude"].get_position()
            
            atm_low = sim.get_local_atmospheric_conditions(low_pos)
            atm_high = sim.get_local_atmospheric_conditions(high_pos)
            
            physics_data['time'].append(current_time)
            physics_data['temp_low'].append(atm_low['temperature'])
            physics_data['temp_high'].append(atm_high['temperature'])
            physics_data['density_low'].append(atm_low['density'])
            physics_data['density_high'].append(atm_high['density'])
            physics_data['altitude_low'].append(low_pos[2])
            physics_data['altitude_high'].append(high_pos[2])
        
        # Save frame
        if step % save_interval == 0:
            fire_state = sim.environment.get_fire_state()
            if fire_state:
                state = fire_state['fire_grid_state']
                burning = np.sum(state['B'])
                
                # Get drone positions
                low_pos = sim.drones["Low_Altitude"].get_position()
                high_pos = sim.drones["High_Altitude"].get_position()
                
                # Get atmospheric data
                atm_low = sim.get_local_atmospheric_conditions(low_pos)
                atm_high = sim.get_local_atmospheric_conditions(high_pos)
                
                save_physics_frame(sim, state, low_pos, high_pos, atm_low, atm_high,
                                 frame, current_time, output_dir)
                
                phase_label = "OBSERVING" if current_time < 5 else "FIREFIGHTING"
                print(f"  Frame {frame:03d} | t={current_time:4.1f}s | {phase_label:12s} | Burning: {burning:3d} | "
                      f"T_low: {atm_low['temperature']:.1f}K | T_high: {atm_high['temperature']:.1f}K")
                frame += 1
    
    # Save physics plots
    save_physics_plots(physics_data, 'output/demo_03_physics_data.png')
    
    print()
    print(f"✅ Saved {frame} frames to {output_dir}/")
    print(f"✅ Saved physics plots to output/demo_03_physics_data.png")
    print()
    print("📊 Final physics observations:")
    if len(physics_data['time']) > 0:
        max_temp_diff = max(np.array(physics_data['temp_low']) - np.array(physics_data['temp_high']))
        print(f"   Max temperature difference: {max_temp_diff:.1f}K")
        print(f"   Final low altitude temp: {physics_data['temp_low'][-1]:.1f}K ({physics_data['temp_low'][-1]-273.15:.1f}°C)")
        print(f"   Final high altitude temp: {physics_data['temp_high'][-1]:.1f}K ({physics_data['temp_high'][-1]-273.15:.1f}°C)")
        print(f"   Final density ratio: {physics_data['density_low'][-1]/physics_data['density_high'][-1]:.3f}")
    print("=" * 70)


def save_physics_frame(sim, state, low_pos, high_pos, atm_low, atm_high, frame_num, time, output_dir):
    """Save a single frame with 3 panels: fire state, terrain, temperature."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    H, W = state['B'].shape
    
    # === PANEL 1: Fire state ===
    img = np.zeros((H, W, 3))
    
    # Base terrain - ALL FOREST (dark green)
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
    
    # Add drone positions
    ax1.plot(low_pos[0], low_pos[1], 'ro', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, label='Low (8m)', zorder=10)
    ax1.plot(high_pos[0], high_pos[1], 'co', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, label='High (25m)', zorder=10)
    
    # Wind arrow
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
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === PANEL 2: Burn rate (terrain type) ===
    burn_rate_grid = sim.environment.fire_grid.fuel_burn_rate
    im2 = ax2.imshow(burn_rate_grid, origin='lower', cmap='YlOrRd', 
                     extent=[x_min, x_max, y_min, y_max],
                     vmin=0, vmax=0.1)
    
    ax2.plot(low_pos[0], low_pos[1], 'ro', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax2.plot(high_pos[0], high_pos[1], 'co', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    
    ax2.set_title('Fuel Burn Rate (terrain-dependent)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Burn rate')
    
    # === PANEL 3: Temperature info ===
    ax3.axis('off')
    
    # Display temperature and density data
    info_text = f"""PHYSICS DATA at t={time:.1f}s

LOW ALTITUDE DRONE (8m):
  Position: ({low_pos[0]:.1f}, {low_pos[1]:.1f}, {low_pos[2]:.1f})
  Temperature: {atm_low['temperature']:.1f} K ({atm_low['temperature']-273.15:.1f}°C)
  Air Density: {atm_low['density']:.3f} kg/m³

HIGH ALTITUDE DRONE (25m):
  Position: ({high_pos[0]:.1f}, {high_pos[1]:.1f}, {high_pos[2]:.1f})
  Temperature: {atm_high['temperature']:.1f} K ({atm_high['temperature']-273.15:.1f}°C)
  Air Density: {atm_high['density']:.3f} kg/m³

PHYSICS EFFECTS:
  ΔT = {atm_low['temperature'] - atm_high['temperature']:.1f} K
  Density ratio: {atm_low['density'] / atm_high['density']:.3f}
  
  Fire heats air → lower density
  Lower density → reduced lift
  Drones must work harder in hot air

TERRAIN EFFECTS:
  Open terrain: Fast burn (0.08/s)
  Forest: Slow burn (0.03/s)
  Water/Buildings: No burn (0.0)
"""
    
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frame_{frame_num:03d}.png', dpi=100, bbox_inches='tight')
    plt.close()


def save_physics_plots(data, filename):
    """Save physics time-series plots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Temperature
    ax1 = axes[0]
    ax1.plot(data['time'], np.array(data['temp_low']) - 273.15, 'r-', 
            linewidth=2, label='Low altitude (8m)')
    ax1.plot(data['time'], np.array(data['temp_high']) - 273.15, 'b-', 
            linewidth=2, label='High altitude (25m)')
    ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('Temperature vs Time (Fire Heating Effect)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Density
    ax2 = axes[1]
    ax2.plot(data['time'], data['density_low'], 'r-', 
            linewidth=2, label='Low altitude (8m)')
    ax2.plot(data['time'], data['density_high'], 'b-', 
            linewidth=2, label='High altitude (25m)')
    ax2.set_ylabel('Air Density (kg/m³)', fontsize=12, fontweight='bold')
    ax2.set_title('Air Density vs Time (ρ = ρ₀ × T₀/T)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Altitude drift
    ax3 = axes[2]
    ax3.plot(data['time'], data['altitude_low'], 'r-', 
            linewidth=2, label='Low altitude drone')
    ax3.plot(data['time'], data['altitude_high'], 'b-', 
            linewidth=2, label='High altitude drone')
    ax3.axhline(y=8, color='r', linestyle=':', alpha=0.5, label='Target 8m')
    ax3.axhline(y=25, color='b', linestyle=':', alpha=0.5, label='Target 25m')
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Drone Altitude (hover in hot air)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_physics_demo()
