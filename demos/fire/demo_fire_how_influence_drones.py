import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil

# Ensure we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    sim.add_quadcopter("Quad_Low", position=[0, 0, 25])
    sim.add_quadcopter("Quad_Side", position=[5, 0, 25])
    sim.add_quadcopter("Quad_High", position=[0, 0, 80])

    # Fixed Wings (Flyovers)
    # They start at Y=-40 and fly North (Y+) through the fire at X=0
    sim.add_fixedwing("Wing_Low", position=[-40, 0, 40], mass=1.0)
    sim.add_fixedwing("Wing_Med", position=[-40, 0, 80], mass=1.0)
    sim.add_fixedwing("Wing_High", position=[-40, 0, 120], mass=1.0)
    
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
            'Quad_Low': [0, 0, 0, 0],   # Hover
            'Quad_Side':   [0, 0, 0, 0],   # Hover
            'Quad_High':   [0, 0, 0, 0],   # Hover
            # Fixed Wing: [Roll=0, Throttle=0.7, Pitch=0, Water=0]
            # Flying straight and level
            'Wing_Low':    [0, 0.7, 0, 0], 
            'Wing_Med':    [0, 0.7, 0, 0],
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
            # save_snapshot(sim, current_time, output_dir)
            print(f"   📸 Snapshot saved at {current_time:.1f}s")

    sim.stop_simulation()
    
    # 6. Generate Final Plots
    # ---------------------------------------------------------
    print("📊 Generating final verification plots...")
    plot_trajectory_analysis(history, output_dir)
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
    colors = {'Quad_Low': 'cyan', 'Quad_Side': 'blue', 'Quad_High': 'yellow', 'Wing_Low': 'red', 'Wing_High': 'orange'}
    for name, drone in sim.drones.items():
        pos = drone.get_position()
        plt.scatter(pos[0], pos[1], c=colors.get(name, 'white'), s=100, edgecolor='black', label=name, zorder=10)
        plt.text(pos[0]+1, pos[1]+1, name, color='white', fontsize=8, fontweight='bold')

    plt.title(f"T={time:.1f}s")
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.grid(True, alpha=0.3)
    
    filename = f"{folder}/env_step_{int(time*10):04d}.pdf"
    plt.savefig(filename)
    plt.close()

def plot_trajectory_analysis(data, folder):
    """Generates X, Y, Z graphs over time for Quads and Wings."""
    time = np.array(data['time'])
    
    # Helper to get XYZ arrays
    def get_xyz(name):
        if name not in data['drones'] or len(data['drones'][name]) == 0:
            print(f"⚠️ Warning: No data found for drone '{name}'")
            return np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time))
        arr = np.array(data['drones'][name]['pos'])
        return arr[:,0], arr[:,1], arr[:,2] # x, y, z
    
    # Helper function for dual-axis plotting
    def plot_dual_axis(ax_left, name, title):
        x, y, z = get_xyz(name)
        
        # Left Axis: X and Y (Lateral)
        l1 = ax_left.plot(time, x, label='X Position', color='red', linestyle='--', alpha=0.7)
        l2 = ax_left.plot(time, y, label='Y Position', color='green', linestyle='--', alpha=0.7)
        ax_left.set_ylabel("Lateral Position (m)", color='black')
        ax_left.set_title(title)
        ax_left.grid(True, alpha=0.3)
        
        # Right Axis: Z (Altitude)
        ax_right = ax_left.twinx()
        l3 = ax_right.plot(time, z, label='Z Altitude', color='blue', linewidth=2)
        ax_right.set_ylabel("Altitude (m)", color='black')
        ax_right.tick_params(axis='y', labelcolor='black')
        
        # Align Grids (Optional, sometimes hard with dual axis)
        # ax_right.grid(False) 

        # Combined Legend
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper left')

   # --- FIGURE 1: QUADCOPTERS ---
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    plot_dual_axis(ax1, 'Quad_Low', "Quad Low (25m above fire) - Hovering")
    plot_dual_axis(ax2, 'Quad_Side', "Quad Side (5m next to fire, 25m above fire) - Hovering")
    plot_dual_axis(ax3, 'Quad_High', "Quad High (80m above fire) - Hovering")
    ax2.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"{folder}/analysis_quadcopters.pdf")
    plt.close()

    # --- FIGURE 2: FIXED WINGS ---
    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(10, 10))
    plot_dual_axis(ax3, 'Wing_Low', "Fixed-Wing Low (40m above fire) - Flighover")
    plot_dual_axis(ax4, 'Wing_Med', "Fixed-Wing Medium (80m above fire) - Flighover")
    plot_dual_axis(ax5, 'Wing_High', "Fixed-Wing High (120m above fire) - Flighover")
    ax4.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"{folder}/analysis_fixedwings.pdf")
    plt.close()

if __name__ == "__main__":
    run_verification()