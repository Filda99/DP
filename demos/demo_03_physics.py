#!/usr/bin/env python3
"""
Demo 3: Advanced Physics - Temperature, Density, and Aerodynamics

Demonstrates the complete physics model:
- Temperature grid affected by fire
- Air density changes with temperature
- Fixed-wing lift calculation using airspeed
- Atmospheric conditions affecting both drone types
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def run_physics_demo():
    """Run advanced physics demonstration."""
    print("=" * 70)
    print("🌡️  DEMO 3: ADVANCED PHYSICS MODEL")
    print("=" * 70)
    print()
    print("Demonstrating:")
    print("  • Temperature grid (20 height levels)")
    print("  • Fire heats air: T = T_base + intensity × 500K")
    print("  • Air density: ρ = ρ₀ × (T₀ / T)")
    print("  • Fixed-wing lift from airspeed, not groundspeed")
    print("  • Quadcopter affected by local density")
    print()
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Simple environment with one fire zone
    sim.environment.add_forest_zone(center=[0, 0], radius=30, density=1.0)
    
    # Enable fire
    sim.enable_fire_simulation(
        grid_width_m=60,
        grid_height_m=60,
        cell_size_m=2.0
    )
    
    # Add wind for airspeed demonstration
    sim.set_wind([5.0, 0.0, 0.0])  # 5 m/s eastward wind
    
    # Start intense fire in center
    sim.start_fire((0, 0), intensity=0.8)
    print("  🔥 Intense fire started at center")
    
    # Add drones at different altitudes
    sim.add_quadcopter("Quad_Low", position=[-5, 0, 5])
    sim.add_quadcopter("Quad_High", position=[5, 0, 25])
    sim.add_fixedwing("FixedWing", position=[0, -10, 15])
    
    print("  🚁 Quadcopter at 5m altitude")
    print("  🚁 Quadcopter at 25m altitude")
    print("  ✈️  Fixed-wing at 15m altitude")
    
    print()
    print("Running simulation for 30 seconds...")
    print()
    
    # Data collection
    data = {
        'time': [],
        'temp_low': [],  # Temperature at 5m
        'temp_high': [],  # Temperature at 25m
        'density_low': [],
        'density_high': [],
        'quad_low_z': [],
        'quad_high_z': [],
        'fixedwing_speed': [],
        'fixedwing_airspeed': [],
    }
    
    snapshots = []
    snapshot_times = [0, 10, 20, 30]
    
    for step in range(int(30 / sim.timestep)):
        current_time = sim.simulation_time
        
        # Simple control - drones try to hover/maintain
        controls = {
            "Quad_Low": np.array([0, 0, 0]),  # Hover
            "Quad_High": np.array([0, 0, 0]),  # Hover
            "FixedWing": np.array([0, 0.6, 0])  # Maintain altitude, moderate throttle
        }
        
        sim.step_simulation(controls)
        
        # Collect data every 0.1s
        if step % 24 == 0:
            # Get atmospheric conditions at drone positions
            quad_low_pos = sim.drones["Quad_Low"].get_position()
            quad_high_pos = sim.drones["Quad_High"].get_position()
            fw_pos = sim.drones["FixedWing"].get_position()
            
            atm_low = sim.get_local_atmospheric_conditions(quad_low_pos)
            atm_high = sim.get_local_atmospheric_conditions(quad_high_pos)
            
            # Fixed-wing speeds
            fw_velocity = sim.drones["FixedWing"].get_velocity()
            fw_groundspeed = np.linalg.norm(fw_velocity)
            
            # Airspeed calculation
            atm_fw = sim.get_local_atmospheric_conditions(fw_pos)
            air_rel_vel = fw_velocity - atm_fw['velocity']
            fw_airspeed = np.linalg.norm(air_rel_vel)
            
            data['time'].append(current_time)
            data['temp_low'].append(atm_low['temperature'])
            data['temp_high'].append(atm_high['temperature'])
            data['density_low'].append(atm_low['density'])
            data['density_high'].append(atm_high['density'])
            data['quad_low_z'].append(quad_low_pos[2])
            data['quad_high_z'].append(quad_high_pos[2])
            data['fixedwing_speed'].append(fw_groundspeed)
            data['fixedwing_airspeed'].append(fw_airspeed)
        
        # Snapshots for temperature visualization
        if any(abs(current_time - t) < sim.timestep for t in snapshot_times):
            fire_state = sim.environment.get_fire_state()
            if fire_state:
                snapshots.append({
                    'time': current_time,
                    'fire_state': fire_state['fire_grid_state'].copy(),
                    'temp_grid': sim.temperature_grid.copy(),
                    'drone_positions': {
                        'Quad_Low': sim.drones["Quad_Low"].get_position().copy(),
                        'Quad_High': sim.drones["Quad_High"].get_position().copy(),
                        'FixedWing': sim.drones["FixedWing"].get_position().copy()
                    }
                })
                print(f"  📸 Snapshot at t={current_time:.1f}s")
    
    # Create visualizations
    create_temperature_visualization(sim, snapshots)
    create_physics_plots(data)
    
    print()
    print("📊 Final observations:")
    print(f"   Temperature difference: {data['temp_low'][-1] - data['temp_high'][-1]:.1f}K")
    print(f"   Density ratio (low/high): {data['density_low'][-1] / data['density_high'][-1]:.3f}")
    print(f"   Fixed-wing groundspeed: {data['fixedwing_speed'][-1]:.2f} m/s")
    print(f"   Fixed-wing airspeed: {data['fixedwing_airspeed'][-1]:.2f} m/s")
    
    print()
    print("✅ Demo complete! Check output/demo_03_*.png")
    print("=" * 70)


def create_temperature_visualization(sim, snapshots):
    """Create 3D temperature visualization."""
    os.makedirs('output', exist_ok=True)
    
    fig = plt.figure(figsize=(20, 10))
    
    for idx, snapshot in enumerate(snapshots[:4]):
        ax = fig.add_subplot(2, 4, idx + 1, projection='3d')
        
        temp_grid = snapshot['temp_grid']
        fire_state = snapshot['fire_state']
        
        # Sample temperature grid (show every 3rd point for clarity)
        H, W = fire_state['B'].shape
        x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
        
        # Create meshgrid for ground plane
        x = np.linspace(x_min, x_max, W)[::3]
        y = np.linspace(y_min, y_max, H)[::3]
        X, Y = np.meshgrid(x, y)
        
        # Plot temperature at different heights
        for h_idx in [0, 5, 10, 15]:  # Ground, 5m, 10m, 15m
            Z = np.ones_like(X) * (h_idx * 2.5)  # Assuming 2.5m per level
            T = temp_grid[h_idx, ::3, ::3]
            
            surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.hot((T - 293) / 200),
                                  alpha=0.3, shade=False)
        
        # Plot fire on ground
        burning = fire_state['B']
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                if burning[i, j]:
                    world_pos = sim.environment.grid_mapper.cell_to_world(i, j)
                    ax.scatter([world_pos[0]], [world_pos[1]], [0], 
                             c='red', s=50, marker='^', alpha=0.8)
        
        # Plot drones
        drone_pos = snapshot['drone_positions']
        for name, pos in drone_pos.items():
            color = 'blue' if 'Quad' in name else 'green'
            ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                      c=color, s=100, marker='o', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_zlim(0, 50)
        ax.set_title(f't = {snapshot["time"]:.1f}s', fontweight='bold')
        ax.view_init(elev=25, azim=45)
        
        # 2D temperature at ground level
        ax2 = fig.add_subplot(2, 4, idx + 5)
        T_ground = temp_grid[0, :, :]
        im = ax2.imshow(T_ground, origin='lower', extent=[x_min, x_max, y_min, y_max],
                       cmap='hot', vmin=293, vmax=500)
        ax2.set_title(f'Ground Temperature t={snapshot["time"]:.1f}s', fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        plt.colorbar(im, ax=ax2, label='Temperature (K)')
    
    plt.suptitle('3D Temperature Distribution & Fire Heating', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/demo_03_temperature.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_physics_plots(data):
    """Create plots showing physics effects."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Temperature and density
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(data['time'], np.array(data['temp_low']) - 273.15, 'r-', 
            linewidth=2, label='Temperature 5m (°C)')
    ax1.plot(data['time'], np.array(data['temp_high']) - 273.15, 'orange', 
            linewidth=2, label='Temperature 25m (°C)')
    ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax1_twin.plot(data['time'], data['density_low'], 'b--', 
                 linewidth=2, label='Density 5m (kg/m³)')
    ax1_twin.plot(data['time'], data['density_high'], 'cyan', linestyle='--',
                 linewidth=2, label='Density 25m (kg/m³)')
    ax1_twin.set_ylabel('Air Density (kg/m³)', fontsize=12, fontweight='bold', color='blue')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    ax1_twin.legend(loc='upper right', fontsize=10)
    
    ax1.set_title('Fire Heating → Lower Air Density', fontsize=14, fontweight='bold')
    
    # Quadcopter altitude drift
    ax2 = axes[1]
    ax2.plot(data['time'], data['quad_low_z'], 'b-', linewidth=2, 
            label='Quad at 5m (hot air)')
    ax2.plot(data['time'], data['quad_high_z'], 'c-', linewidth=2, 
            label='Quad at 25m (cooler air)')
    ax2.axhline(y=5, color='b', linestyle=':', alpha=0.5)
    ax2.axhline(y=25, color='c', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Quadcopter Altitude (Lower density → reduced lift)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Fixed-wing: groundspeed vs airspeed
    ax3 = axes[2]
    ax3.plot(data['time'], data['fixedwing_speed'], 'g-', linewidth=2, 
            label='Groundspeed')
    ax3.plot(data['time'], data['fixedwing_airspeed'], 'purple', linewidth=2,
            linestyle='--', label='Airspeed (used for lift)')
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
    ax3.set_title('Fixed-Wing: Lift uses Airspeed, not Groundspeed', 
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/demo_03_physics.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_physics_demo()
