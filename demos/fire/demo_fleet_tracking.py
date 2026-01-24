import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects 
import os
import math
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import Simulation Logic
from src.simulation import Simulation

def get_control_circle(time, speed, radius, phase_offset=0):
    """Generates velocity commands for a circle."""
    # vx = speed * -sin(t), vy = speed * cos(t)
    # Scaled by time/radius relationship
    omega = speed / radius
    t = time + phase_offset
    vx = -np.sin(omega * t) * speed
    vy =  np.cos(omega * t) * speed
    return [0, vx, vy, 0] # [Roll, Pitch, Yaw, Throttle] -> mapped differently per drone

def run_fleet_demo():
    # 1. Configuration
    LOCATION_NAME = "Královo Pole, Brno, Czechia"
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚁 Initializing Fleet Tracking Demo at {LOCATION_NAME}")

    # 2. Setup Simulation
    sim = Simulation()
    sim.start_simulation()
    distance_m=400
    # Load Environment (Standard OSM)
    sim.setup_osm_environment(LOCATION_NAME, distance_m=distance_m)
    
    # CRITICAL: Enable Fire Sim to get the Terrain Grid (Fuel/Burn Rates)
    # We won't simulate fire, but we need the grid for the background map.
    sim.enable_fire_simulation(grid_width_m=distance_m*2, grid_height_m=distance_m*2, cell_size_m=4.0)

    sim.set_wind([0, 0, 0.0])
    
    # 3. Add Fleet (3 Quads, 2 Fixed-Wings)
    print("   Deploying Fleet (3 Quads, 2 Fixed-Wings)...")
    
    # Quadcopters
    drones = {}
    drones['Q1 (Scout)'] = sim.add_quadcopter("Q1", position=[-200, -150, 30])
    drones['Q2 (Relay)'] = sim.add_quadcopter("Q2", position=[0, -80, 40])
    drones['Q3 (Cam)']   = sim.add_quadcopter("Q3", position=[300, -300, 35])
    
    # Fixed Wings
    drones['FW1 (Eagle)'] = sim.add_fixedwing("FW1", position=[-300, 100, 100])
    drones['FW1 (Eagle)'].state_va = 18.0
    
    # 4. Simulation Loop
    print("▶️  Running Simulation (60s)...")
    duration = 60.0
    steps = int(duration / sim.timestep)
    
    # Trajectory Storage
    trajectories = {name: {'x': [], 'y': []} for name in drones}
    
    for step in range(steps):
        t = step * sim.timestep
        controls = {}
        
        # --- Control Logic ---
        
        # Q1: Large Counter-Clockwise Circle
        # Increased Amplitude (0.8 = Fast) / Decreased Frequency (0.15 = Wide)
        controls["Q1"] = [
            0.8 * np.cos(t * 0.15), # Roll (Right/Left velocity)
            0.8 * np.sin(t * 0.15), # Pitch (Fwd/Back velocity)
            0.0, 0.0
        ]
        
        # Q2: Wide Figure 8 (Lemniscate)
        # Slower frequency (0.1) for x, double frequency (0.2) for y
        controls["Q2"] = [
            0.7 * np.cos(t * 0.1), 
            0.5 * np.sin(t * 0.2), 
            0.0, 0.0
        ]
        
        # Q3: Search Pattern (Scanning)
        # Moves distinctively back and forth
        controls["Q3"] = [
            0.6 * np.sin(t * 0.15), 
            0.3, # Constant forward drift
            0.05, 0.0 # Slow yaw
        ]

        # FW1 (Starts Left): Sharp Right Bank
        controls["FW1"] = [0.7, 0.6, 0, 0] 
        
        # Step
        sim.step_simulation(controls)
        
        # Log Positions
        for name, drone in sim.drones.items():
            pos = drone.get_position()
            # Map internal IDs back to our display names
            for display_name, d_obj in drones.items():
                if d_obj == drone:
                    trajectories[display_name]['x'].append(pos[0])
                    trajectories[display_name]['y'].append(pos[1])
        
        if step % 100 == 0:
            print(f"   Step {step}/{steps}")
            print(f"F1 Position: {sim.drones['FW1'].get_position()}")

    # 5. Generate Graph
    print("🎨 Generating Terrain Map with Flight Paths...")
    save_plot(sim, trajectories, OUTPUT_DIR)

def save_plot(sim, trajectories, output_dir):
    """
    Generates the map using the exact 'Left Side' style.
    Forces the view to stay within the map bounds.
    """
    
    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams.update({'font.size': 14})
    
    # 2. Extract Terrain Data
    grid = sim.environment.fire_grid
    burn_rate = grid.fuel_burn_rate
    H, W = grid.H, grid.W
    
    # Create RGB Image
    env_img = np.zeros((H, W, 3))
    
    mask_water    = (burn_rate == 0.0)
    mask_building = (burn_rate > 0.0) & (burn_rate <= 0.0002)
    mask_forest   = (burn_rate > 0.0002) & (burn_rate <= 0.0005)
    mask_grass    = (burn_rate >= 0.0005)

    env_img[mask_grass]    = [0.6, 0.7, 0.4]    # Sage Green
    env_img[mask_forest]   = [0.1, 0.4, 0.1]    # Dark Green
    env_img[mask_water]    = [0.2, 0.5, 0.9]    # Blue
    env_img[mask_building] = [0.5, 0.5, 0.5]    # Grey
    
    # 3. Plot Map Background
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    extent = [x_min, x_max, y_min, y_max]
    
    ax.imshow(env_img, origin='lower', extent=extent)
    
    # 4. Plot Trajectories
    colors = ['#FFD700', '#FF00FF', '#00FFFF', '#FF4500', '#FFFFFF']
    color_idx = 0
    
    for name, data in trajectories.items():
        xs = np.array(data['x'])
        ys = np.array(data['y'])
        
        if len(xs) == 0: continue
        
        is_fw = "FW" in name
        linestyle = '-' if is_fw else ':' 
        linewidth = 3 if is_fw else 2.5
        color = colors[color_idx % len(colors)]
        
        # Plot Path
        ax.plot(xs, ys, linestyle=linestyle, linewidth=linewidth, color=color, label=name, alpha=0.9)
        
        # Start Marker (Green Circle)
        # Matplotlib will automatically hide this if it's out of bounds
        ax.scatter(xs[0], ys[0], color='lime', edgecolors='black', s=60, zorder=5) 

        # End Marker (Red X) & Label
        # Only draw if the FINAL position is currently inside the map bounds
        if (x_min <= xs[-1] <= x_max) and (y_min <= ys[-1] <= y_max):
            ax.scatter(xs[-1], ys[-1], color='red', edgecolors='white', marker='X', s=100, zorder=6)
            # Fixed the patheffects usage here
            ax.text(xs[-1]+10, ys[-1]+10, name, color='white', fontsize=12, fontweight='bold',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
        
        color_idx += 1

    # 5. Formatting & CLIPPING
    ax.set_xlabel("East [m]", fontsize=14)
    ax.set_ylabel("North [m]", fontsize=14)

    # --- CRITICAL: Force the graph limits to the map size ---
    # This effectively "crops" any drone paths that went outside
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Legend
    patch_grass = mpatches.Patch(color=[0.6, 0.7, 0.4], label='Grass')
    patch_forest = mpatches.Patch(color=[0.1, 0.4, 0.1], label='Forest')
    patch_water = mpatches.Patch(color=[0.2, 0.5, 0.9], label='Water')
    patch_bld = mpatches.Patch(color=[0.5, 0.5, 0.5], label='Buildings')
    
    legend_elements = [patch_grass, patch_forest, patch_water, patch_bld]
    legend_elements.append(Line2D([0], [0], color='none', label=' ')) 
    legend_elements.append(Line2D([0], [0], color='none', label='-- DRONES --'))
    
    c_idx = 0
    for name in trajectories:
        is_fw = "FW" in name
        ls = '-' if is_fw else ':'
        c = colors[c_idx % len(colors)]
        legend_elements.append(Line2D([0], [0], color=c, lw=2, linestyle=ls, label=name))
        c_idx += 1

    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    save_path = os.path.join(output_dir, "fleet_tracking_graph.pdf")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Success! Graph saved to: {save_path}")


if __name__ == "__main__":
    run_fleet_demo()