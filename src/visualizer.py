"""
Visualization Module

Handles all visualization and plotting for simulation results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


class SimulationVisualizer:
    """Handles visualization of simulation results."""
    
    def __init__(self, output_dir="output"):
        """Initialize visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _draw_3d_building(self, ax, pos, size):
        """Draw a 3D rectangular building."""
        # Building dimensions
        x, y, z = pos[0], pos[1], 0  # Ground level
        dx, dy, dz = size[0], size[1], size[2]
        
        # Define the 8 vertices of the rectangular prism
        vertices = [
            [x-dx/2, y-dy/2, z],      # Bottom face
            [x+dx/2, y-dy/2, z],
            [x+dx/2, y+dy/2, z],
            [x-dx/2, y+dy/2, z],
            [x-dx/2, y-dy/2, z+dz],   # Top face
            [x+dx/2, y-dy/2, z+dz],
            [x+dx/2, y+dy/2, z+dz],
            [x-dx/2, y+dy/2, z+dz]
        ]
        
        # Define the 6 faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
        ]
        
        # Create and add the 3D building
        building = Poly3DCollection(faces, alpha=0.4, facecolor='lightgray', edgecolor='darkgray')
        ax.add_collection3d(building)
    
    def _draw_3d_forest(self, ax, center, radius):
        """Draw a 3D forest as cylindrical canopy."""
        # Create multiple cylinder levels for tree canopy
        heights = [5, 10, 15]  # Different tree heights
        
        for i, height in enumerate(heights):
            # Create cylinder at each height level
            theta = np.linspace(0, 2*np.pi, 20)
            z_cyl = np.linspace(0, height, 10)
            
            # Simplified: just draw the canopy outline at different heights
            for height in heights:
                x_circle = center[0] + radius * 0.9 * np.cos(theta)
                y_circle = center[1] + radius * 0.9 * np.sin(theta)
                z_circle = np.full_like(x_circle, height)
                ax.plot(x_circle, y_circle, z_circle, color='green', alpha=0.6, linewidth=2)
    
    def _draw_3d_lake(self, ax, center, radius):
        """Draw a 3D lake as a flat disc."""
        # Create circular disc at ground level
        theta = np.linspace(0, 2*np.pi, 30)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = np.zeros_like(x)  # Ground level
        
        # Create vertices for the disc
        vertices = list(zip(x, y, z))
        
        # Create the flat disc surface
        lake_surface = Poly3DCollection([vertices], alpha=0.6, facecolor='lightblue', edgecolor='blue')
        ax.add_collection3d(lake_surface)
        
        # Add a circular outline for clarity
        ax.plot(x, y, z, color='blue', alpha=0.8, linewidth=2)
    
    def create_single_drone_visualization(self, simulation_log, drones, environment, 
                                         drone_name=None, title="Simulation Analysis"):
        """Create comprehensive visualization for a single drone."""
        if drone_name is None:
            # Use first drone if not specified
            drone_name = list(drones.keys())[0] if drones else None
        
        if drone_name is None or drone_name not in simulation_log['drones']:
            print("❌ No drone data available for visualization")
            return
        
        drone_data = simulation_log['drones'][drone_name]
        positions = np.array(drone_data['positions'])
        forces = np.array(drone_data['forces'])
        velocities = np.array(drone_data['velocities'])
        times = np.array(simulation_log['times'])
        
        if len(positions) == 0:
            print("❌ No flight data to visualize")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{title} - {drone_name}', fontsize=16, fontweight='bold')
        
        # Flatten axes
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        
        # 1. 3D Trajectory
        ax1.remove()
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.8)
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
        
        # Add 3D buildings as proper rectangular prisms
        for obstacle in environment.obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            self._draw_3d_building(ax1, pos, size)
        
        # Add 3D forests and lakes
        for zone in environment.terrain_zones:
            if zone['type'] == 'forest':
                center = zone['center']
                radius = zone['radius']
                self._draw_3d_forest(ax1, center, radius)
            elif zone['type'] == 'lake':
                center = zone['center']
                radius = zone['radius']
                self._draw_3d_lake(ax1, center, radius)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('3D Flight Trajectory with Obstacles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top view with environment
        # First, add a light green grass background for the entire area
        ax2.set_facecolor('#E8F5E9')  # Very light green for grass
        
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.8)
        ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
        ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
        
        # Add environment features (buildings in top view with height layers)
        for obstacle in environment.obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            height = size[2]
            
            # Create multiple rectangles to show building height (like stacked blocks)
            layers = min(5, max(1, int(height / 4)))  # More layers for taller buildings
            for i in range(layers):
                # Each layer gets darker/more opaque toward the center
                alpha = 0.2 + (i * 0.6 / layers)  # Gradient from light to dark
                layer_size = max(0.3, 1.0 - (i * 0.1))  # Slightly smaller each layer
                
                rect = plt.Rectangle(
                    (pos[0] - size[0]/2 * layer_size, pos[1] - size[1]/2 * layer_size), 
                    size[0] * layer_size, size[1] * layer_size,
                    facecolor='gray', alpha=alpha, 
                    label='Buildings' if i == 0 else ""
                )
                ax2.add_patch(rect)
        
        for zone in environment.terrain_zones:
            if zone['type'] == 'forest':
                circle = plt.Circle(zone['center'], zone['radius'], 
                                  facecolor='#2E7D32', alpha=0.6, label='Forest')  # Dark green for forest
                ax2.add_patch(circle)
            elif zone['type'] == 'lake':
                circle = plt.Circle(zone['center'], zone['radius'], 
                                  facecolor='blue', alpha=0.3, label='Lake')
                ax2.add_patch(circle)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Top View with Environment')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.legend()
        
        # 3. Forces over time
        ax3.plot(times, forces[:, 0], 'r-', label='Force X', linewidth=2)
        ax3.plot(times, forces[:, 1], 'g-', label='Force Y', linewidth=2)
        ax3.plot(times, forces[:, 2], 'b-', label='Force Z', linewidth=2)
        if drone_name in drones and drones[drone_name].can_hover():
            hover_force = drones[drone_name].get_hover_force()
            ax3.axhline(y=hover_force, color='red', linestyle='--', alpha=0.7, label='Hover force')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force (N)')
        ax3.set_title('Applied Forces')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Speed over time
        speeds = np.linalg.norm(velocities, axis=1)
        ax4.plot(times, speeds, 'purple', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Flight Speed')
        ax4.grid(True, alpha=0.3)
        
        # 5. Position components
        ax5.plot(times, positions[:, 0], 'r-', label='X Position', linewidth=2)
        ax5.plot(times, positions[:, 1], 'g-', label='Y Position', linewidth=2)
        ax5.plot(times, positions[:, 2], 'b-', label='Z Position', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Position (m)')
        ax5.set_title('Position Components')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Environment effects
        ax6.text(0.1, 0.9, f"Environment Info:", fontsize=12, fontweight='bold', transform=ax6.transAxes)
        
        env_info = environment.get_environment_info()
        info_text = f"""
Obstacles: {env_info['obstacles']}
Terrain Zones: {env_info['terrain_zones']}
Wind: {env_info['weather']['wind_velocity']} m/s
Visibility: {env_info['weather']['visibility']} m
Collisions: {len(simulation_log['collisions'])}
        """
        
        ax6.text(0.1, 0.7, info_text, fontsize=10, transform=ax6.transAxes, verticalalignment='top')
        ax6.set_title('Environment & Collision Info')
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save visualization to output directory
        filename = f"{title.lower().replace(' ', '_')}_{drone_name}_analysis.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved as '{filepath}'")
        plt.close()
    
    def create_multi_drone_visualization(self, simulation_log, drones, environment, 
                                        title="Multi-Drone Analysis"):
        """Create visualization showing all drones together."""
        if len(drones) == 0:
            print("❌ No drones to visualize")
            return
            
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{title} - All Drones', fontsize=16, fontweight='bold')
        
        # Flatten axes
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        
        # Colors for different drones
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # 1. 3D Trajectory with all drones
        ax1.remove()
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        for i, (drone_name, drone_data) in enumerate(simulation_log['drones'].items()):
            if len(drone_data['positions']) == 0:
                continue
                
            positions = np.array(drone_data['positions'])
            color = colors[i % len(colors)]
            
            # Plot trajectory
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    color=color, linewidth=2, alpha=0.8, label=drone_name)
            ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                       color=color, s=100, marker='o', alpha=0.8)
            ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                       color=color, s=100, marker='s', alpha=0.8)
        
        # Add environment
        for obstacle in environment.obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            self._draw_3d_building(ax1, pos, size)
            
        for zone in environment.terrain_zones:
            if zone['type'] == 'forest':
                self._draw_3d_forest(ax1, zone['center'], zone['radius'])
            elif zone['type'] == 'lake':
                self._draw_3d_lake(ax1, zone['center'], zone['radius'])
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('3D Flight Trajectories - All Drones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top view with all drones
        # Add light green grass background
        ax2.set_facecolor('#E8F5E9')  # Very light green for grass
        
        for i, (drone_name, drone_data) in enumerate(simulation_log['drones'].items()):
            if len(drone_data['positions']) == 0:
                continue
                
            positions = np.array(drone_data['positions'])
            color = colors[i % len(colors)]
            
            ax2.plot(positions[:, 0], positions[:, 1], color=color, linewidth=2, alpha=0.8, label=drone_name)
            ax2.scatter(positions[0, 0], positions[0, 1], color=color, s=100, marker='o', alpha=0.8)
            ax2.scatter(positions[-1, 0], positions[-1, 1], color=color, s=100, marker='s', alpha=0.8)
        
        # Add environment features (simplified for clarity with multiple drones)
        for obstacle in environment.obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            rect = plt.Rectangle(
                (pos[0] - size[0]/2, pos[1] - size[1]/2), 
                size[0], size[1],
                facecolor='gray', alpha=0.3, edgecolor='black'
            )
            ax2.add_patch(rect)
        
        for zone in environment.terrain_zones:
            if zone['type'] == 'forest':
                circle = plt.Circle(zone['center'], zone['radius'], 
                                  facecolor='#2E7D32', alpha=0.6)  # Dark green for forest
                ax2.add_patch(circle)
            elif zone['type'] == 'lake':
                circle = plt.Circle(zone['center'], zone['radius'], 
                                  facecolor='blue', alpha=0.3)
                ax2.add_patch(circle)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Top View - All Drone Paths')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        # 3-6. Combined plots for all drones
        times = np.array(simulation_log['times'])
        
        for i, (drone_name, drone_data) in enumerate(simulation_log['drones'].items()):
            if len(drone_data['forces']) == 0:
                continue
                
            forces = np.array(drone_data['forces'])
            positions = np.array(drone_data['positions'])
            color = colors[i % len(colors)]
            
            # Force X
            ax3.plot(times, forces[:, 0], color=color, alpha=0.7, label=f'{drone_name}')
            # Force Y  
            ax4.plot(times, forces[:, 1], color=color, alpha=0.7, label=f'{drone_name}')
            # Force Z
            ax5.plot(times, forces[:, 2], color=color, alpha=0.7, label=f'{drone_name}')
            # Altitude
            ax6.plot(times, positions[:, 2], color=color, alpha=0.7, label=f'{drone_name}')
        
        ax3.set_title('X Forces Over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force X (N)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        ax4.set_title('Y Forces Over Time')
        ax4.set_xlabel('Time (s)')  
        ax4.set_ylabel('Force Y (N)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        ax5.set_title('Z Forces Over Time')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Force Z (N)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        ax6.set_title('Altitude Profiles')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Altitude (m)')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        
        # Save plot to output directory
        filename = f"multi_drone_combined_analysis.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Multi-drone visualization saved as '{filepath}'")
    
    def generate_fire_analysis(self, simulation_log, environment, base_name):
        """Generate fire spread analysis visualization."""
        fire_states = simulation_log['fire_states']
        times = simulation_log['times']
        
        if not fire_states:
            return
        
        # Extract fire statistics over time
        burning_cells = []
        total_fuel = []
        avg_intensity = []
        
        for state in fire_states:
            if state and 'fire_stats' in state:
                burning_cells.append(state['fire_stats']['burning_cells'])
                total_fuel.append(state['fire_stats']['total_fuel'])
                avg_intensity.append(state['fire_stats']['avg_intensity'])
        
        # Create fire analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Fire spread over time
        fire_times = times[:len(burning_cells)]
        axes[0, 0].plot(fire_times, burning_cells, 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Burning Cells')
        axes[0, 0].set_title('Fire Spread Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Fuel consumption
        axes[0, 1].plot(fire_times, total_fuel, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Total Fuel Remaining')
        axes[0, 1].set_title('Fuel Consumption Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Fire intensity
        axes[1, 0].plot(fire_times, avg_intensity, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Average Fire Intensity')
        axes[1, 0].set_title('Fire Intensity Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Final fire state visualization
        if fire_states:
            final_state = fire_states[-1]['fire_grid_state']
            grid_bounds = fire_states[-1]['grid_bounds']
            
            # Create composite image
            H, W = final_state['B'].shape
            img = np.zeros((H, W, 3))
            
            # Fuel as green background
            fuel_normalized = final_state['F'] / np.max(final_state['F']) if np.max(final_state['F']) > 0 else final_state['F']
            img[:, :, 1] = fuel_normalized * 0.5
            
            # Burned areas as dark
            burned_mask = (final_state['F'] < 0.1) & (~final_state['B'])
            img[burned_mask] = [0.2, 0.1, 0.0]  # Dark brown for burned areas
            
            # Currently burning as red
            burning_mask = final_state['B']
            img[burning_mask, 0] = 1.0  # Red
            img[burning_mask, 1] = 0.0  # Remove green
            
            # Display with correct spatial extent
            x_min, x_max, y_min, y_max = grid_bounds
            axes[1, 1].imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower')
            axes[1, 1].set_xlabel('X Position (m)')
            axes[1, 1].set_ylabel('Y Position (m)')
            axes[1, 1].set_title('Final Fire State')
            
            # Add drone trajectories to fire map
            for drone_name, drone_data in simulation_log['drones'].items():
                if len(drone_data['positions']) > 0:
                    positions = np.array(drone_data['positions'])
                    axes[1, 1].plot(positions[:, 0], positions[:, 1], 
                                   linewidth=2, alpha=0.7, label=drone_name)
            
            axes[1, 1].legend()
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'{base_name}_fire_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Fire analysis saved to {filepath}")
