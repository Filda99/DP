import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil

# Ensure we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

def run_verification():
    # 0. Setup Output Directory
    # ---------------------------------------------------------
    output_dir = "verification_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"📁 Output folder created: {output_dir}/")

    # 1. Initialize Simulation
    # ---------------------------------------------------------
    print("🔥 Initializing Verification Demo...")
    sim = Simulation()
    sim.start_simulation()
    
    # KILL AMBIENT WIND to verify pure fire physics
    sim.set_wind([0, 0, 0])
    
    # Enable Fire (60x60m map, 2m cells = 30x30 grid)
    sim.enable_fire_simulation(grid_width_m=60, grid_height_m=60, cell_size_m=2.0)
    
    # 2. CREATE A BURNING BUILDING (The "Long Fire")
    # ---------------------------------------------------------
    print("🏗️  Constructing central building (High Fuel Source)...")
    
    grid = sim.environment.fire_grid
    mapper = sim.environment.grid_mapper
    
    # Get center cell indices
    cx, cy = mapper.world_to_cell([0, 0])
    
    # Define Building Size (e.g., 6x6 meters)
    # We set fuel very high (50.0) so it burns for the whole simulation
    b_rad = 2
    for r in range(cx - b_rad, cx + b_rad + 1):
        for c in range(cy - b_rad, cy + b_rad + 1):
            grid.F[r, c] = 50.0          # Massive Fuel
            grid.fuel_burn_rate[r, c] = 0.05 # Standard Burn

    # Add Visual Block for PyBullet
    sim.environment.add_city_block(position=[0,0,0], size=[8, 8, 2], color=[0.3, 0.3, 0.3, 1])

    # 3. Setup Drones
    # ---------------------------------------------------------
    # Quadcopters (Stationary Probes)
    sim.add_quadcopter("Quad_Center", position=[0, 0, 15])   # In Plume
    sim.add_quadcopter("Quad_Side", position=[15, 0, 10])    # In Inflow

    # Fixed Wings (Flyovers)
    # They start at Y=-40 and fly North (Y+) through the fire at X=0
    # Wing 1: Low Altitude (15m)
    sim.add_fixedwing("Wing_Low", position=[-40, 0, 15], mass=1.0)
    # Wing 2: High Altitude (40m)
    sim.add_fixedwing("Wing_High", position=[-40, 0, 40], mass=1.0)
    
    # 4. IGNITE
    # ---------------------------------------------------------
    print("🔥 Igniting building...")
    sim.start_fire([0,0], intensity=1.0)

    # 5. Simulation Loop
    # ---------------------------------------------------------
    duration_sec = 20.0 # Short enough for flyover
    steps = int(duration_sec / sim.timestep)
    snapshot_interval = 2.0 # Frequent snapshots
    
    # Data storage
    history = {
        'time': [],
        'drones': {name: {'pos': [], 'updraft': [], 'radial': []} for name in sim.drones}
    }

    print(f"🚀 Running simulation for {duration_sec} seconds...")
    
    for step in range(steps):
        current_time = step * sim.timestep
        
        # A. Controls
        controls = {
            'Quad_Center': [0, 0, 0, 0],   # Hover
            'Quad_Side':   [0, 0, 0, 0],   # Hover
            # Fixed Wing: [Roll=0, Throttle=0.7, Pitch=0, Water=0]
            # Flying straight and level
            'Wing_Low':    [0, 0.7, 0, 0], 
            'Wing_High':   [0, 0.7, 0, 0]
        }
        sim.step_simulation(controls)
        
        # B. Collect Physics Data
        history['time'].append(current_time)
        for name, drone in sim.drones.items():
            pos = drone.get_position()
            history['drones'][name]['pos'].append(pos)
            
            # Atmospheric probe
            atmos = sim.get_local_atmospheric_conditions(pos)
            w = atmos['velocity']
            
            # Vertical (Updraft)
            history['drones'][name]['updraft'].append(w[2])
            
            # Radial (Project onto position vector relative to [0,0])
            dist = np.hypot(pos[0], pos[1])
            if dist > 0.1:
                # Dot product: if wind points towards [0,0], this will be negative
                rad = (w[0]*pos[0] + w[1]*pos[1]) / dist
            else:
                rad = 0
            history['drones'][name]['radial'].append(rad)

        # C. Generate Snapshots
        if step % int(snapshot_interval / sim.timestep) == 0:
            save_snapshot(sim, current_time, output_dir)
            print(f"   📸 Snapshot saved at {current_time:.1f}s")

    sim.stop_simulation()
    
    # 6. Generate Final Plots
    # ---------------------------------------------------------
    print("📊 Generating final verification plots...")
    plot_physics_data(history, output_dir)
    plot_3d_trajectory(history, output_dir)
    print(f"✅ Verification complete. Check folder: {output_dir}/")

def save_snapshot(sim, time, folder):
    """Saves a top-down view of Fire Intensity + Drone Positions."""
    grid = sim.environment.fire_grid
    mapper = sim.environment.grid_mapper
    
    plt.figure(figsize=(8, 8))
    
    # Plot Fire Intensity Map
    plt.imshow(grid.I, origin='lower', cmap='inferno', vmin=0, vmax=1.0,
               extent=[mapper.origin_x, -mapper.origin_x, mapper.origin_y, -mapper.origin_y])
    plt.colorbar(label="Fire Intensity")
    
    # Plot Drone Positions
    colors = {'Quad_Center': 'cyan', 'Quad_Side': 'blue', 'Wing_Low': 'red', 'Wing_High': 'orange'}
    for name, drone in sim.drones.items():
        pos = drone.get_position()
        plt.scatter(pos[0], pos[1], c=colors.get(name, 'white'), s=100, edgecolor='black', label=name, zorder=10)
        plt.text(pos[0]+1, pos[1]+1, name, color='white', fontsize=8, fontweight='bold')

    plt.title(f"T={time:.1f}s")
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.grid(True, alpha=0.3)
    
    filename = f"{folder}/env_step_{int(time*10):04d}.png"
    plt.savefig(filename)
    plt.close()

def plot_physics_data(data, folder):
    """Plots the Wind Influence graphs."""
    time = data['time']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Quadcopters (Stationary)
    ax1.set_title("Quadcopter Probes (Stationary)")
    ax1.plot(time, data['drones']['Quad_Center']['updraft'], 'c-', label='Quad Center (Updraft)')
    ax1.plot(time, data['drones']['Quad_Side']['radial'], 'b--', label='Quad Side (Radial)')
    ax1.axhline(0, color='k', alpha=0.3)
    ax1.set_ylabel("Wind Speed (m/s)")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Fixed Wings (Flyover Profile) - Updraft vs Time
    ax2.set_title("Fixed-Wing Flyover: Updraft Profile")
    ax2.plot(time, data['drones']['Wing_Low']['updraft'], 'r-', label='Wing Low (15m)')
    ax2.plot(time, data['drones']['Wing_High']['updraft'], 'orange', linestyle='--', label='Wing High (40m)')
    ax2.set_ylabel("Updraft (m/s)")
    ax2.legend()
    ax2.grid(True)

    # 3. Fixed Wing - Altitude Hold Check
    ax3.set_title("Fixed-Wing Altitude Check")
    # Extract Z height
    z_low = [p[2] for p in data['drones']['Wing_Low']['pos']]
    z_high = [p[2] for p in data['drones']['Wing_High']['pos']]
    ax3.plot(time, z_low, 'r-', label='Wing Low Z')
    ax3.plot(time, z_high, 'orange', label='Wing High Z')
    ax3.set_ylabel("Altitude (m)")
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{folder}/physics_verification.png")
    plt.close()

def plot_3d_trajectory(data, folder):
    """Plots 3D movement history."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fire base (approximate)
    xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    ax.plot_surface(xx, yy, np.zeros_like(xx), color='red', alpha=0.3)
    
    colors = {'Quad_Center': 'cyan', 'Quad_Side': 'blue', 'Wing_Low': 'red', 'Wing_High': 'orange'}
    
    for name, drone_data in data['drones'].items():
        pos_hist = np.array(drone_data['pos'])
        if len(pos_hist) > 0:
            ax.plot(pos_hist[:,0], pos_hist[:,1], pos_hist[:,2], label=name, color=colors.get(name,'black'), linewidth=2)
            # End point
            ax.scatter(pos_hist[-1,0], pos_hist[-1,1], pos_hist[-1,2], s=50, color=colors.get(name,'black'))

    ax.set_title("3D Drone Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.legend()
    
    plt.savefig(f"{folder}/trajectory_3d.png")
    plt.close()

if __name__ == "__main__":
    run_verification()