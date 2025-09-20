import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
from drone_factory import create_drone
from drones.base_drone import BaseDrone
from environment import Environment, TerrainType


class Simulation:
    """
    A flexible drone simulation class that can handle multiple drones,
    different scenarios, and various collision avoidance strategies.
    """
    
    def __init__(self, 
                 name: str = "Drone Simulation",
                 duration: int = 20,
                 time_step: float = 0.1,
                 plot_bounds: Tuple[float, float, float, float] = (-10, 60, -10, 100),
                 environment: Optional[Environment] = None):
        """
        Initialize the simulation.
        
        Args:
            name: Name of the simulation
            duration: Number of simulation steps
            time_step: Time step between simulation frames
            plot_bounds: (x_min, x_max, y_min, y_max) for the plot area
            environment: Environment object with terrain features (optional)
        """
        self.name = name
        self.duration = duration
        self.time_step = time_step
        self.plot_bounds = plot_bounds
        
        # Environment setup
        if environment is None:
            self.environment = Environment(plot_bounds, f"{name} Environment")
        else:
            self.environment = environment
        
        # Simulation state
        self.drones: List[BaseDrone] = []
        self.goals: Dict[int, np.ndarray] = {}
        
        # Results tracking
        self.positions: Dict[int, List[List[float]]] = {}
        self.collisions: List[int] = []
        self.step_count = 0
        
        # Animation settings
        self.colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        
    def add_drone(self, 
                  drone_type: str, 
                  position: List[float], 
                  heading: float, 
                  goal: List[float],
                  label: Optional[str] = None,
                  altitude: Optional[float] = None) -> int:
        """
        Add a drone to the simulation.
        
        Args:
            drone_type: Type of drone ("quadcopter", "fixedwing")
            position: Initial position - [x, y] (2D) or [x, y, z] (3D)
            heading: Initial heading in degrees
            goal: Target position - [x, y] (2D) or [x, y, z] (3D)
            label: Optional custom label for the drone
            altitude: Optional altitude override (used if position/goal are 2D)
            
        Returns:
            int: Index of the added drone
        """
        # Handle both 2D and 3D inputs
        if len(position) == 2 and altitude is not None:
            start_pos = [position[0], position[1], altitude]
        else:
            start_pos = position  # Let drone factory handle default altitude
        
        if len(goal) == 2 and altitude is not None:
            goal_pos = [goal[0], goal[1], altitude]
            goal_pos = goal
        else:
            goal_pos = goal  # Will be handled by drone's compute_action method
        
        # Check if starting position is safe (3D-aware)
        if len(start_pos) >= 3:
            if not self.environment.is_position_safe(start_pos[0], start_pos[1], start_pos[2]):
                print(f"⚠️ Warning: Starting position {start_pos} may not be safe for flight!")
        else:
            if not self.environment.is_position_safe(start_pos[0], start_pos[1]):
                print(f"⚠️ Warning: Starting position {start_pos} may not be safe for flight!")
        
        # Check if goal position is safe (3D-aware)
        if len(goal_pos) >= 3:
            if not self.environment.is_position_safe(goal_pos[0], goal_pos[1], goal_pos[2]):
                print(f"⚠️ Warning: Goal position {goal_pos} may not be safe for flight!")
        else:
            if not self.environment.is_position_safe(goal_pos[0], goal_pos[1]):
                print(f"⚠️ Warning: Goal position {goal_pos} may not be safe for flight!")
        
        drone = create_drone(drone_type, start_pos, heading, altitude)
        drone_index = len(self.drones)
        
        self.drones.append(drone)
        self.goals[drone_index] = np.array(goal_pos)
        self.positions[drone_index] = []
        
        return drone_index
    
    def remove_drone(self, drone_index):
        self.drones.remove(drone_index)

    def check_collision(self, drone_i: BaseDrone, drone_j: BaseDrone, expanded: int = 0) -> bool:
        """
        Check if two drones are colliding based on their collision zones (3D-aware).
        
        Args:
            drone_i: First drone
            drone_j: Second drone
            expanded: Expanded collision zone 
            
        Returns:
            bool: True if drones are colliding
        """
        zi = drone_i.get_collision_zone()
        zj = drone_j.position
        
        # Ensure both positions are in 3D format for consistent comparison
        pos_i = np.array(drone_i.position)
        pos_j = np.array(drone_j.position)
        
        # Handle different collision zone types
        if len(zi) == 2:
            # Circular/Spherical collision zone (quadcopters)
            center, radius = zi
            center_array = np.array(center)
            
            # Calculate 3D distance if both positions are 3D
            if len(center_array) >= 3 and len(pos_j) >= 3:
                distance = np.linalg.norm(center_array - pos_j)
            else:
                # Fall back to 2D distance for backward compatibility
                distance = np.linalg.norm(center_array[:2] - pos_j[:2])
            
            return distance <= (radius + expanded)
        
        elif len(zi) == 3:
            # Rectangular collision zone (fixed-wing, 2D legacy)
            p1, p2, width = zi
            a, b, p = np.array(p1), np.array(p2), np.array(zj)
            
            # Use only x, y coordinates for 2D rectangular collision
            ab, ap = b[:2] - a[:2], p[:2] - a[:2]
            ab_dot = np.dot(ab, ab)
            if ab_dot == 0:
                return False
            
            proj = np.dot(ap, ab) / ab_dot
            closest = a[:2] + proj * ab
            dist = np.linalg.norm(closest - p[:2])
            
            return 0 <= proj <= 1 and dist <= (width / 2 + expanded)
        
        elif len(zi) == 4:
            # 3D Box collision zone (fixed-wing, 3D)
            p1, p2, width, height = zi
            a, b, p = np.array(p1), np.array(p2), np.array(zj)
            
            # Check if point is within the 3D box defined by the line segment
            ab = b - a
            ap = p - a
            
            # Project point onto the line segment
            ab_dot = np.dot(ab, ab)
            if ab_dot == 0:
                return False
            
            t = np.dot(ap, ab) / ab_dot
            t = max(0, min(1, t))  # Clamp to [0, 1]
            
            # Find closest point on the line segment
            closest_on_line = a + t * ab
            
            # Calculate distance vector from line to point
            dist_vec = p - closest_on_line
            
            # Check if within width and height constraints
            # This is a simplified 3D box check - could be improved with proper orientation
            horizontal_dist = np.linalg.norm(dist_vec[:2])  # x, y distance
            vertical_dist = abs(dist_vec[2]) if len(dist_vec) > 2 else 0  # z distance
            
            return (horizontal_dist <= (width / 2 + expanded) and 
                    vertical_dist <= (height / 2 + expanded))
        
        return False
    
    def detect_collisions(self, expanded: int = 0) -> List[Tuple[int, int]]:
        """
        Detect all current collisions between drones.
        
        Args:
            expanded: Expanded collision zone 
        
        Returns:
            List of tuples (i, j) representing colliding drone pairs
        """
        # Use list comprehension for clarity and efficiency
        # It iterates through all unique pairs of drones and checks for collisions.
        return [
            (i, j)
            for i in range(len(self.drones))
            for j in range(i + 1, len(self.drones))
            if self.check_collision(self.drones[i], self.drones[j], expanded)
        ]
    
    def get_other_drones(self, drone_index: int) -> List[BaseDrone]:
        """
        Get all other drones except the one at the given index.
        
        Args:
            drone_index: Index of the current drone
            
        Returns:
            List of other drones
        """
        return [self.drones[i] for i in range(len(self.drones)) if i != drone_index]
    
    def step(self) -> bool:
        """
        Execute one simulation step.
        
        The method detects actual collisions (expanded=0) for recording and 
        potential collisions (expanded > 0) for avoidance behavior.
        
        Uses a priority-based avoidance system: when two drones are on collision
        course, the drone with lower index continues on its path while the drone
        with higher index performs evasive maneuvers.
        
        Returns:
            bool: True if simulation should continue, False if finished
        """
        if self.step_count >= self.duration:
            return False
        
        # Detect actual collisions (expanded = 0)
        actual_collisions = self.detect_collisions(0)
        if actual_collisions:
            self.collisions.append(self.step_count)
        
        # Detect potential collisions for avoidance
        potential_collisions = self.detect_collisions(10)
        
        # Create avoidance mode mapping with priority system
        # Only one drone per collision pair should avoid, the other continues
        avoidance_mode = {i: False for i in range(len(self.drones))}
        for i, j in potential_collisions:
            # Priority system: lower index has priority and continues straight
            # Higher index drone performs avoidance maneuver
            if i < j:
                avoidance_mode[j] = True  # j avoids, i continues
            else:
                avoidance_mode[i] = True  # i avoids, j continues
        
        # Compute actions for each drone
        actions = []
        for i, drone in enumerate(self.drones):
            # Get other drones for collision avoidance
            other_drones = self.get_other_drones(i)
            
            # Get terrain effects at current position (3D-aware)
            drone_pos = drone.position
            if len(drone_pos) >= 3:
                # 3D position available
                constraints = self.environment.get_flight_constraints_at_position(
                    drone_pos[0], drone_pos[1], drone_pos[2]
                )
            else:
                # Fall back to 2D for backward compatibility
                constraints = self.environment.get_flight_constraints_at_position(
                    drone_pos[0], drone_pos[1]
                )
            weather_effects = self.environment.get_weather_effects()
            
            # Call the drone's own compute_action method
            action = drone.compute_action(
                goal=self.goals[i],
                avoid=avoidance_mode[i],
                other_drones=other_drones
            )
            
            # Apply terrain speed modifiers (3D-aware)
            if hasattr(drone, 'max_speed') and isinstance(action, list):
                # For quadcopters with velocity commands
                speed_mod = constraints['speed_modifier'] * weather_effects['speed_modifier']
                
                if len(action) == 2:
                    # 2D action: [vx, vy]
                    action = [action[0] * speed_mod, action[1] * speed_mod]
                elif len(action) == 3:
                    # 3D action: [vx, vy, vz] - apply speed modifier to horizontal components only
                    action = [action[0] * speed_mod, action[1] * speed_mod, action[2]]
                    
            elif isinstance(action, list) and len(action) == 2:
                # For fixed-wing with 3D actions: [steering, climb_rate]
                steering, climb_rate = action
                if constraints['speed_modifier'] < 1.0:
                    steering *= constraints['speed_modifier']
                action = [steering, climb_rate]
                
            elif isinstance(action, (int, float)):
                # For fixed-wing with steering angles - terrain affects turn rate
                if constraints['speed_modifier'] < 1.0:
                    action *= constraints['speed_modifier']
            
            actions.append(action)
        
        # Move all drones
        for i, drone in enumerate(self.drones):
            drone.move(actions[i])
            self.positions[i].append(drone.position.copy())
        
        self.step_count += 1
        return True
    
    def run(self) -> Dict:
        """
        Run the complete simulation.
        
        Returns:
            Dict containing simulation results
        """
        print(f"🚁 Starting simulation: {self.name}")
        print(f"   Drones: {len(self.drones)}")
        print(f"   Duration: {self.duration} steps")
        
        while self.step():
            pass
        
        results = {
            "name": self.name,
            "steps": self.step_count,
            "collisions": len(self.collisions),
            "collision_steps": self.collisions,
            "positions": self.positions,
            "goals": self.goals,
            "drones": len(self.drones)
        }
        
        print(f"✅ Simulation completed!")
        print(f"   Total collisions: {len(self.collisions)}")
        
        return results
    
    def create_3d_animation(self, output_file: str = "simulation_3d.gif", interval: int = 100):
        """
        Create a 3D animated visualization of the simulation.
        
        Args:
            output_file: Path to save the animation
            interval: Time between frames in milliseconds
        """
        if not self.positions:
            print("❌ No simulation data available. Run simulation first.")
            return
        
        # Check if we actually have 3D data
        has_3d_data = any(len(self.positions[i][0]) >= 3 for i in self.positions if len(self.positions[i]) > 0)
        if not has_3d_data:
            print("⚠️ No 3D position data found. Using create_animation() for 2D visualization.")
            return self.create_animation(output_file, interval)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create line plots for 3D trajectories
        lines = []
        for i in range(len(self.drones)):
            line, = ax.plot([], [], [], color=self.colors[i % len(self.colors)], 
                           label=f"Drone {i}", linewidth=2)
            lines.append(line)
        
        # Add 3D goal markers
        for i, goal in self.goals.items():
            if len(goal) >= 3:
                ax.scatter(*goal, c='green', marker='x', s=100, label=f'Goal {i}')
            else:
                # 2D goal, use default altitude
                ax.scatter(goal[0], goal[1], 50.0, c='green', marker='x', s=100, label=f'Goal {i}')
        
        # Set up 3D plot bounds
        x_min, x_max, y_min, y_max = self.plot_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Calculate altitude bounds from drone positions
        all_altitudes = []
        for drone_positions in self.positions.values():
            for pos in drone_positions:
                if len(pos) >= 3:
                    all_altitudes.append(pos[2])
        
        if all_altitudes:
            z_min = min(all_altitudes) - 10
            z_max = max(all_altitudes) + 10
        else:
            z_min, z_max = 0, 100
        
        ax.set_zlim(z_min, z_max)
        
        # Set labels
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_zlabel('Altitude (meters)')
        ax.legend(loc='upper right')
        
        # Add environment info to title
        env_info = f" | Terrain Zones: {len(self.environment.terrain_zones)}"
        weather = self.environment.weather_conditions
        if weather['wind_speed'] > 0:
            env_info += f" | Wind: {weather['wind_speed']:.1f}m/s"
        
        def update_3d(frame: int):
            # Update title with collision warning and environment info
            collision_warning = '⚠️ COLLISION!' if frame in self.collisions else ''
            ax.set_title(f"{self.name} (3D View) - Step {frame} {collision_warning}{env_info}")
            
            # Clear previous collision visualization
            # Note: 3D collision visualization could be added here
            
            # Update trajectory lines
            for i, line in enumerate(lines):
                if i < len(self.positions) and frame < len(self.positions[i]):
                    traj = np.array(self.positions[i][:frame+1])
                    if len(traj) > 0:
                        if traj.shape[1] >= 3:  # 3D positions
                            line.set_data_3d(traj[:, 0], traj[:, 1], traj[:, 2])
                        else:  # 2D positions, add default altitude
                            default_alt = np.full(len(traj), 50.0)
                            line.set_data_3d(traj[:, 0], traj[:, 1], default_alt)
            
            # Add current drone positions as points
            current_points = []
            for i, drone in enumerate(self.drones):
                if frame < len(self.positions[i]):
                    current_pos = self.positions[i][frame]
                    if len(current_pos) >= 3:
                        ax.scatter(current_pos[0], current_pos[1], current_pos[2], 
                                 c=self.colors[i % len(self.colors)], s=50, alpha=0.8)
                    else:
                        ax.scatter(current_pos[0], current_pos[1], 50.0,
                                 c=self.colors[i % len(self.colors)], s=50, alpha=0.8)
        
        # Create animation
        total_frames = max(len(pos) for pos in self.positions.values()) if self.positions else 1
        ani = animation.FuncAnimation(fig, update_3d, frames=total_frames, 
                                    interval=interval, repeat=True)
        
        # Save animation
        ani.save(output_file, writer="pillow")
        print(f"📹 3D Animation saved as {output_file}")
        plt.close()
    
    def create_smart_animation(self, output_file: str = None, interval: int = 100):
        """
        Create an animated visualization, automatically choosing 2D or 3D based on data.
        
        Args:
            output_file: Path to save the animation (auto-generated if None)
            interval: Time between frames in milliseconds
        """
        if not self.positions:
            print("❌ No simulation data available. Run simulation first.")
            return
        
        # Check if we have meaningful 3D movement (altitude changes > 5m)
        has_meaningful_3d = False
        for drone_positions in self.positions.values():
            if len(drone_positions) > 1:
                altitudes = [pos[2] for pos in drone_positions if len(pos) >= 3]
                if len(altitudes) > 1:
                    altitude_range = max(altitudes) - min(altitudes)
                    if altitude_range > 5.0:  # More than 5m altitude variation
                        has_meaningful_3d = True
                        break
        
        if has_meaningful_3d:
            print("🎬 Creating 3D visualization (detected significant altitude changes)")
            if output_file is None:
                output_file = "simulation_3d.gif"
            self.create_3d_animation(output_file, interval)
        else:
            print("🎬 Creating 2D visualization (minimal altitude changes)")
            if output_file is None:
                output_file = "simulation_2d.gif"
            self.create_animation(output_file, interval)
    
    def create_animation(self, output_file: str = "simulation.gif", interval: int = 100):
        """
        Create an animated visualization of the simulation.
        
        Args:
            output_file: Path to save the animation
            interval: Time between frames in milliseconds
        """
        if not self.positions:
            print("❌ No simulation data available. Run simulation first.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw terrain first (background)
        self.environment.visualize_terrain(ax)
        
        # Create line plots for trajectories
        lines = []
        for i in range(len(self.drones)):
            line, = ax.plot([], [], color=self.colors[i % len(self.colors)], 
                           label=f"Drone {i}", linewidth=2)
            lines.append(line)
        
        # Add goal markers
        for i, goal in self.goals.items():
            ax.scatter(*goal, c='green', marker='x', s=100, zorder=10)
            ax.annotate(f'Goal {i}', goal, xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        # Set up plot
        ax.set_xlim(self.plot_bounds[0], self.plot_bounds[1])
        ax.set_ylim(self.plot_bounds[2], self.plot_bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        
        # Add environment info to title
        env_info = f" | Terrain Zones: {len(self.environment.terrain_zones)}"
        weather = self.environment.weather_conditions
        if weather['wind_speed'] > 0:
            env_info += f" | Wind: {weather['wind_speed']:.1f}m/s"
        
        def update(frame: int):
            # Update title with collision warning and environment info
            collision_warning = '⚠️ COLLISION!' if frame in self.collisions else ''
            ax.set_title(f"{self.name} - Step {frame} {collision_warning}{env_info}")
            
            # Redraw terrain (in case it gets overwritten)
            # Remove old terrain patches first
            terrain_patches = [p for p in ax.patches if hasattr(p, '_terrain_zone')]
            for patch in terrain_patches:
                patch.remove()
            
            # Redraw terrain
            self.environment.visualize_terrain(ax)
            
            # Remove previous collision zone patches (but keep terrain)
            collision_patches = [p for p in reversed(ax.patches) if not hasattr(p, '_terrain_zone')]
            for patch in collision_patches:
                patch.remove()
            
            # Update trajectory lines
            for i, line in enumerate(lines):
                if i < len(self.positions) and frame < len(self.positions[i]):
                    traj = np.array(self.positions[i][:frame+1])
                    if len(traj) > 0:
                        line.set_data(traj[:, 0], traj[:, 1])
            
            # Draw collision zones at current positions
            for i, drone in enumerate(self.drones):
                if frame < len(self.positions[i]):
                    # Update drone position for collision zone calculation
                    current_pos = self.positions[i][frame]
                    drone.position = current_pos
                    
                    zone = drone.get_collision_zone()
                    color = self.colors[i % len(self.colors)]
                    
                    # Handle circular collision zone
                    if len(zone) == 2:
                        center, radius = zone
                        circ = plt.Circle(center, radius, color=color, alpha=0.2)
                        ax.add_patch(circ)
                    
                    # Handle rectangular collision zone
                    elif len(zone) == 3:
                        p1, p2, w = zone
                        dx, dy = np.array(p2) - np.array(p1)
                        length = np.linalg.norm([dx, dy])
                        angle = np.arctan2(dy, dx) * 180 / np.pi
                        rect = plt.Rectangle(p1, length, w, angle=angle, 
                                           color=color, alpha=0.2)
                        ax.add_patch(rect)
        
        # Create animation
        total_frames = max(len(pos) for pos in self.positions.values()) if self.positions else 1
        ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                    interval=interval, repeat=True)
        
        # Save animation
        ani.save(output_file, writer="pillow")
        print(f"📹 Animation saved as {output_file}")
        plt.close()
