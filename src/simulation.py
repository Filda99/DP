"""
Simulation Manager

Manages the complete simulation with multiple drones, environment, and physics.
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .environment import Environment
from .drones import Quadcopter, FixedWing


class Simulation:
    """Complete simulation manager."""
    
    def __init__(self, gui=False):
        """Initialize simulation."""
        self.gui = gui
        self.physics_client = None
        self.drones = {}
        self.environment = Environment()
        self.simulation_time = 0.0
        self.timestep = 1/240.0  # 240 FPS
        
        # Simulation data
        self.simulation_log = {
            'drones': {},
            'environment_effects': [],
            'collisions': [],
            'times': []
        }
        
    def start_simulation(self):
        """Start PyBullet simulation."""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Create ground
        self.environment.create_ground()
        
        print(f"✅ Simulation started ({'GUI' if self.gui else 'headless'} mode)")
        
    def stop_simulation(self):
        """Stop PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
        print("✅ Simulation stopped")
    
    def add_quadcopter(self, name, position=[0, 0, 5], mass=0.5):
        """Add a quadcopter to the simulation."""
        quad = Quadcopter(position, mass)
        self.drones[name] = quad
        
        # Initialize logging for this drone
        self.simulation_log['drones'][name] = {
            'type': 'quadcopter',
            'positions': [],
            'forces': [],
            'velocities': [],
            'control_inputs': []
        }
        
        print(f"✅ Added quadcopter '{name}' at {position}")
        return quad
    
    def add_fixedwing(self, name, position=[0, 0, 5], mass=1.0):
        """Add a fixed-wing aircraft to the simulation."""
        fw = FixedWing(position, mass)
        self.drones[name] = fw
        
        # Initialize logging for this drone
        self.simulation_log['drones'][name] = {
            'type': 'fixedwing', 
            'positions': [],
            'forces': [],
            'velocities': [],
            'control_inputs': []
        }
        
        print(f"✅ Added fixed-wing '{name}' at {position}")
        return fw
    
    def setup_city_environment(self):
        """Setup city environment with buildings."""
        self.environment.create_city_environment()
        
    def setup_natural_environment(self):
        """Setup natural environment with forests and lakes."""
        self.environment.create_natural_environment()
        
    def setup_mixed_environment(self):
        """Setup mixed urban/natural environment."""
        self.environment.create_mixed_environment()
    
    def set_wind(self, wind_velocity, turbulence=0.0):
        """Set wind conditions."""
        self.environment.set_wind(wind_velocity, turbulence)
        print(f"✅ Wind set to {wind_velocity} m/s, turbulence: {turbulence}")
    
    def set_weather(self, visibility=1000.0, precipitation=0.0):
        """Set weather conditions."""
        self.environment.set_weather(visibility, precipitation)
        print(f"✅ Weather set - visibility: {visibility}m, precipitation: {precipitation}")
    
    def step_simulation(self, drone_controls):
        """
        Step the simulation forward.
        
        Args:
            drone_controls: Dict with drone names as keys and joystick inputs as values
                          e.g., {'drone1': [0.5, 0.0, 0.1], 'drone2': [-0.3, 1.0, 0.0]}
        """
        # Apply controls to each drone
        for drone_name, control_input in drone_controls.items():
            if drone_name in self.drones:
                drone = self.drones[drone_name]
                
                # Apply drone control
                forces = drone.apply_control(control_input)
                
                # Apply environmental effects
                position = drone.get_position()
                wind = self.environment.get_wind_at_position(position)
                wind_forces = drone.apply_wind_effect(wind)
                
                # Log data
                self.simulation_log['drones'][drone_name]['positions'].append(position.copy())
                self.simulation_log['drones'][drone_name]['forces'].append(forces.copy())
                self.simulation_log['drones'][drone_name]['velocities'].append(drone.get_velocity().copy())
                self.simulation_log['drones'][drone_name]['control_inputs'].append(control_input.copy())
        
        # Step physics
        p.stepSimulation()
        self.simulation_time += self.timestep
        self.simulation_log['times'].append(self.simulation_time)
        
        # Check for collisions with environment
        self._check_collisions()
    
    def _check_collisions(self):
        """Check for collisions between drones and environment."""
        for drone_name, drone in self.drones.items():
            position = drone.get_position()
            
            # Check obstacle collision
            collision, obstacle = self.environment.is_position_in_obstacle(position)
            if collision:
                collision_data = {
                    'time': self.simulation_time,
                    'drone': drone_name,
                    'obstacle': obstacle['type'],
                    'position': position.copy()
                }
                self.simulation_log['collisions'].append(collision_data)
                print(f"⚠️ Collision detected: {drone_name} hit {obstacle['type']} at {position}")
    
    def run_scenario(self, scenario_function, steps=1000):
        """Run a complete scenario."""
        print(f"🚁 Running scenario for {steps} steps...")
        
        for step in range(steps):
            # Get control inputs from scenario function
            controls = scenario_function(step, self.simulation_time, self.drones)
            
            # Step simulation
            self.step_simulation(controls)
            
            # Progress reporting
            if step % 100 == 0:
                print(f"  Step {step}/{steps} - Time: {self.simulation_time:.2f}s")
        
        print(f"✅ Scenario completed after {steps} steps ({self.simulation_time:.2f}s)")
    
    def get_drone_status(self, drone_name):
        """Get current status of a drone."""
        if drone_name not in self.drones:
            return None
        
        drone = self.drones[drone_name]
        return {
            'name': drone_name,
            'type': drone.get_drone_type(),
            'position': drone.get_position(),
            'velocity': drone.get_velocity(),
            'speed': drone.get_speed(),
            'characteristics': drone.get_flight_characteristics()
        }
    
    def get_all_drone_status(self):
        """Get status of all drones."""
        return {name: self.get_drone_status(name) for name in self.drones.keys()}
    
    def _draw_3d_building(self, ax, pos, size):
        """Draw a 3D rectangular building."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
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
            
            # Cylinder surface
            for j in range(len(theta)-1):
                for k in range(len(z_cyl)-1):
                    # Create quad faces for cylinder
                    x1 = center[0] + radius * np.cos(theta[j]) * (0.8 + 0.2 * k/len(z_cyl))
                    y1 = center[1] + radius * np.sin(theta[j]) * (0.8 + 0.2 * k/len(z_cyl))
                    x2 = center[0] + radius * np.cos(theta[j+1]) * (0.8 + 0.2 * k/len(z_cyl))
                    y2 = center[1] + radius * np.sin(theta[j+1]) * (0.8 + 0.2 * k/len(z_cyl))
                    
            # Simplified: just draw the canopy outline at different heights
            for height in heights:
                x_circle = center[0] + radius * 0.9 * np.cos(theta)
                y_circle = center[1] + radius * 0.9 * np.sin(theta)
                z_circle = np.full_like(x_circle, height)
                ax.plot(x_circle, y_circle, z_circle, color='green', alpha=0.6, linewidth=2)
    
    def _draw_3d_lake(self, ax, center, radius):
        """Draw a 3D lake as a flat disc."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
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
    
    def create_multi_drone_visualization(self, title="Multi-Drone Analysis"):
        """Create visualization showing all drones together."""
        if len(self.drones) == 0:
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
        
        for i, (drone_name, drone_data) in enumerate(self.simulation_log['drones'].items()):
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
        for obstacle in self.environment.obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            self._draw_3d_building(ax1, pos, size)
            
        for zone in self.environment.terrain_zones:
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
        for i, (drone_name, drone_data) in enumerate(self.simulation_log['drones'].items()):
            if len(drone_data['positions']) == 0:
                continue
                
            positions = np.array(drone_data['positions'])
            color = colors[i % len(colors)]
            
            ax2.plot(positions[:, 0], positions[:, 1], color=color, linewidth=2, alpha=0.8, label=drone_name)
            ax2.scatter(positions[0, 0], positions[0, 1], color=color, s=100, marker='o', alpha=0.8)
            ax2.scatter(positions[-1, 0], positions[-1, 1], color=color, s=100, marker='s', alpha=0.8)
        
        # Add environment features (simplified for clarity with multiple drones)
        for obstacle in self.environment.obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            rect = plt.Rectangle(
                (pos[0] - size[0]/2, pos[1] - size[1]/2), 
                size[0], size[1],
                facecolor='gray', alpha=0.3, edgecolor='black'
            )
            ax2.add_patch(rect)
        
        for zone in self.environment.terrain_zones:
            if zone['type'] == 'forest':
                circle = plt.Circle(zone['center'], zone['radius'], 
                                  facecolor='green', alpha=0.3)
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
        times = np.array(self.simulation_log['times'])
        
        for i, (drone_name, drone_data) in enumerate(self.simulation_log['drones'].items()):
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
        
        # Save plot
        filename = f"multi_drone_combined_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Multi-drone visualization saved as '{filename}'")

    def create_visualization(self, drone_name=None, title="Simulation Analysis"):
        """Create comprehensive visualization of simulation results."""
        if drone_name is None:
            # Use first drone if not specified
            drone_name = list(self.drones.keys())[0] if self.drones else None
        
        if drone_name is None or drone_name not in self.simulation_log['drones']:
            print("❌ No drone data available for visualization")
            return
        
        drone_data = self.simulation_log['drones'][drone_name]
        positions = np.array(drone_data['positions'])
        forces = np.array(drone_data['forces'])
        velocities = np.array(drone_data['velocities'])
        times = np.array(self.simulation_log['times'])
        
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
        for obstacle in self.environment.obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            
            # Create a 3D rectangular building
            self._draw_3d_building(ax1, pos, size)
        
        # Add 3D forests and lakes
        for zone in self.environment.terrain_zones:
            if zone['type'] == 'forest':
                center = zone['center']
                radius = zone['radius']
                # Draw forest as 3D cylindrical canopy
                self._draw_3d_forest(ax1, center, radius)
            
            elif zone['type'] == 'lake':
                center = zone['center']
                radius = zone['radius']
                # Draw lake as flat 3D disc
                self._draw_3d_lake(ax1, center, radius)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('3D Flight Trajectory with Obstacles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top view with environment
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.8)
        ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
        ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
        
        # Add environment features (buildings in top view with height layers)
        for obstacle in self.environment.obstacles:
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
                    label='Buildings' if i == 0 else ""  # Only label first layer
                )
                ax2.add_patch(rect)
        
        for zone in self.environment.terrain_zones:
            if zone['type'] == 'forest':
                circle = plt.Circle(zone['center'], zone['radius'], 
                                  facecolor='green', alpha=0.3, label='Forest')
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
        if drone_name in self.drones and self.drones[drone_name].can_hover():
            hover_force = self.drones[drone_name].get_hover_force()
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
        
        env_info = self.environment.get_environment_info()
        info_text = f"""
Obstacles: {env_info['obstacles']}
Terrain Zones: {env_info['terrain_zones']}
Wind: {env_info['weather']['wind_velocity']} m/s
Visibility: {env_info['weather']['visibility']} m
Collisions: {len(self.simulation_log['collisions'])}
        """
        
        ax6.text(0.1, 0.7, info_text, fontsize=10, transform=ax6.transAxes, verticalalignment='top')
        ax6.set_title('Environment & Collision Info')
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        filename = f"{title.lower().replace(' ', '_')}_{drone_name}_analysis.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved as '{filename}'")
        plt.close()
    
    def get_simulation_summary(self):
        """Get complete simulation summary."""
        return {
            'total_time': self.simulation_time,
            'total_steps': len(self.simulation_log['times']),
            'drones': {name: self.get_drone_status(name) for name in self.drones.keys()},
            'environment': self.environment.get_environment_info(),
            'collisions': len(self.simulation_log['collisions'])
        }