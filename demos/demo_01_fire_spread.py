#!/usr/bin/env python3
"""
Demo 1 DEBUG: Save every step for detailed analysis
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


def run_debug_demo():
    """Run fire spread with frame-by-frame output."""
    print("=" * 70)
    print("🔬 DEBUG: FIRE SPREAD FRAME-BY-FRAME")
    print("=" * 70)
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    sim.setup_mixed_environment()
    
    # Enable fire
    sim.enable_fire_simulation(
        grid_width_m=100,
        grid_height_m=100,
        cell_size_m=2.0
    )
    
    # Set wind
    sim.set_wind([8.0, 0.0, 0.0])
    
    # Start ONE fire in southwest
    sim.start_fire((-30, -30), intensity=0.3)
    print("  🔥 Started fire at (-30, -30)")
    
    # Create output directory
    output_dir = 'output/debug_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nRunning simulation for 10 seconds, saving every 0.5s...")
    print()
    
    # Run simulation
    frame = 0
    save_interval = 30  # Save every 0.5 seconds (30 steps at 60 FPS)
    
    for step in range(int(10 / sim.timestep)):
        sim.step_simulation({})
        
        # Save frame every 0.5s
        if step % save_interval == 0:
            current_time = sim.simulation_time
            fire_state = sim.environment.get_fire_state()
            
            if fire_state:
                state = fire_state['fire_grid_state']
                burning = np.sum(state['B'])
                
                # Save visualization
                save_frame(sim, state, frame, current_time, output_dir)
                
                print(f"  Frame {frame:03d} | t={current_time:.1f}s | Burning: {burning:3d} cells")
                frame += 1
    
    print()
    print(f"✅ Saved {frame} frames to {output_dir}/")
    print(f"   Check frame_000.png, frame_001.png, etc.")
    print("=" * 70)


def save_frame(sim, state, frame_num, time, output_dir):
    """Save a single frame visualization."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    H, W = state['B'].shape
    
    # === PANEL 1: Fire state (burning cells) ===
    img = np.zeros((H, W, 3))
    
    # Base terrain colors
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
                # Forest
                img[i, j] = [0.1, 0.4, 0.1]  # Dark green
            else:
                # Open area
                img[i, j] = [0.6, 0.5, 0.2]  # Tan
    
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
    arrow_start_x = x_max - 15
    arrow_start_y = y_max - 10
    arrow_scale = 3.0  # Scale factor for visibility
    
    ax1.arrow(arrow_start_x, arrow_start_y, 
             wind_x * arrow_scale, wind_y * arrow_scale,
             head_width=3, head_length=2, fc='cyan', ec='white', 
             linewidth=3, alpha=0.9, zorder=10)
    
    # Add wind speed label
    ax1.text(arrow_start_x, arrow_start_y - 5, 
            f'Wind: {wind_speed:.1f} m/s',
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
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


if __name__ == '__main__':
    run_debug_demo()
