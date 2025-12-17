"""
Visualization Module

Handles visualization and plotting for simulation results.
Optimized for performance using Poly3DCollection batching.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

class SimulationVisualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_building_faces(self, pos, size):
        """Generates vertices for a single building (helper for batching)."""
        x, y, z = pos[0], pos[1], 0
        dx, dy, dz = size[0], size[1], size[2]
        
        # 8 Corners
        v = [
            [x-dx/2, y-dy/2, z], [x+dx/2, y-dy/2, z], [x+dx/2, y+dy/2, z], [x-dx/2, y+dy/2, z], # Bottom
            [x-dx/2, y-dy/2, z+dz], [x+dx/2, y-dy/2, z+dz], [x+dx/2, y+dy/2, z+dz], [x-dx/2, y+dy/2, z+dz] # Top
        ]
        
        # 6 Faces (as list of vertices)
        return [
            [v[0], v[1], v[2], v[3]], # Bottom
            [v[4], v[5], v[6], v[7]], # Top
            [v[0], v[1], v[5], v[4]], # Front
            [v[2], v[3], v[7], v[6]], # Back
            [v[1], v[2], v[6], v[5]], # Right
            [v[4], v[7], v[3], v[0]]  # Left
        ]

    def create_single_drone_visualization(self, simulation_log, drones, environment, 
                                         drone_name=None, title="Simulation Analysis"):
        """Comprehensive visualization for a single drone."""
        
        drone_name = drone_name or (list(drones.keys())[0] if drones else None)
        if not drone_name or drone_name not in simulation_log['drones']:
            print("❌ No drone data to visualize.")
            return

        data = simulation_log['drones'][drone_name]
        pos = np.array(data['positions'])
        if len(pos) == 0: return

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'{title} - {drone_name}', fontsize=16)

        # --- 1. 3D Trajectory (Batched Rendering) ---
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax1.scatter(pos[0,0], pos[0,1], pos[0,2], c='g', s=50, label='Start')
        ax1.scatter(pos[-1,0], pos[-1,1], pos[-1,2], c='r', s=50, label='End')

        # BATCH DRAWING: Collect all building faces first
        all_faces = []
        for obs in environment.obstacles:
            if obs['type'] == 'city_block':
                all_faces.extend(self._create_building_faces(obs['position'], obs['size']))
        
        # Add all buildings as ONE collection (Huge speedup)
        if all_faces:
            mesh = Poly3DCollection(all_faces, alpha=0.4, facecolor='#DDDDDD', edgecolor='#999999', linewidth=0.5)
            ax1.add_collection3d(mesh)

        # Draw Forests/Lakes (Simple scatter for performance)
        for zone in environment.terrain_zones:
            c, r = zone['center'], zone['radius']
            z = 0
            if zone['type'] == 'forest':
                # Draw a circle on ground
                p = plt.Circle((c[0], c[1]), r, color='green', alpha=0.3)
                ax1.add_patch(p)
                p.set_zorder(1)
            elif zone['type'] == 'lake':
                p = plt.Circle((c[0], c[1]), r, color='blue', alpha=0.3)
                ax1.add_patch(p)

        ax1.set_title('3D Trajectory')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        
        # --- 2. Top Down View ---
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_facecolor('#E8F5E9') # Grass color
        ax2.plot(pos[:, 0], pos[:, 1], 'b-')
        
        # Environment 2D
        for obs in environment.obstacles:
            p, s = obs['position'], obs['size']
            rect = plt.Rectangle((p[0]-s[0]/2, p[1]-s[1]/2), s[0], s[1], facecolor='gray', alpha=0.5)
            ax2.add_patch(rect)
            
        ax2.set_title('Top Down View')
        ax2.axis('equal')

        # --- 3. Telemetry Plots ---
        times = np.array(simulation_log['times'])
        vel = np.array(data['velocities'])
        forces = np.array(data['forces'])
        
        # Speed
        ax3 = fig.add_subplot(2, 3, 3)
        speed = np.linalg.norm(vel, axis=1) if len(vel) > 0 else []
        ax3.plot(times, speed, 'purple')
        ax3.set_title('Speed (m/s)')
        ax3.grid(True, alpha=0.3)

        # Altitude
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(times, pos[:, 2], 'b')
        ax4.set_title('Altitude (m)')
        ax4.grid(True, alpha=0.3)

        # Forces
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(times, forces[:, 2], 'r', label='Z Thrust')
        ax5.set_title('Vertical Thrust (N)')
        ax5.grid(True, alpha=0.3)

        # Info Text
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        info = f"Duration: {times[-1]:.1f}s\nMax Speed: {np.max(speed):.1f} m/s\nEvents: {len(simulation_log['events'])}"
        ax6.text(0.1, 0.5, info, fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"analysis_{drone_name}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"✅ Saved analysis to {save_path}")

    def create_multi_drone_visualization(self, simulation_log, drones, environment, title="Multi-Drone"):
        """Simple pass-through to single visualization for now, can be expanded."""
        # For brevity, implementing a loop
        for name in drones:
            self.create_single_drone_visualization(simulation_log, drones, environment, name, title)
            
    def generate_fire_analysis(self, simulation_log, environment, base_name):
        """Generates simple fire stats plot."""
        if not simulation_log.get('fire_states'): return
        
        states = simulation_log['fire_states']
        times = simulation_log['times'][:len(states)]
        burned = [s['fire_stats']['burning_cells'] for s in states]
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, burned, 'r-')
        plt.title('Fire Spread (Active Cells)')
        plt.xlabel('Time (s)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{base_name}_fire.png"))
        plt.close()