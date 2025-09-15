import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
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
                  label: Optional[str] = None) -> int:
        """
        Add a drone to the simulation.
        
        Args:
            drone_type: Type of drone ("quadcopter", "fixedwing")
            position: Initial [x, y] position
            heading: Initial heading in degrees
            goal: Target [x, y] position
            label: Optional custom label for the drone
            
        Returns:
            int: Index of the added drone
        """
        # Check if starting position is safe
        if not self.environment.is_position_safe(position[0], position[1]):
            print(f"⚠️ Warning: Starting position {position} may not be safe for flight!")
        
        # Check if goal position is safe
        if not self.environment.is_position_safe(goal[0], goal[1]):
            print(f"⚠️ Warning: Goal position {goal} may not be safe for flight!")
        
        drone = create_drone(drone_type, position, heading)
        drone_index = len(self.drones)
        
        self.drones.append(drone)
        self.goals[drone_index] = np.array(goal)
        self.positions[drone_index] = []
        
        return drone_index
    
    def remove_drone(self, drone_index):
        self.drones.remove(drone_index)

    def check_collision(self, drone_i: BaseDrone, drone_j: BaseDrone, expanded: int = 0) -> bool:
        """
        Check if two drones are colliding based on their collision zones.
        
        Args:
            drone_i: First drone
            drone_j: Second drone
            expanded: Expanded collision zone 
            
        Returns:
            bool: True if drones are colliding
        """
        zi = drone_i.get_collision_zone()
        zj = drone_j.position
        
        # Check circular collision zone (quadcopters)
        if len(zi) == 2:
            center, radius = zi
            distance = np.linalg.norm(np.array(center) - np.array(zj))
            return distance <= (radius + expanded)
        
        # Check rectangular collision zone (fixed-wing)
        elif len(zi) == 3:
            p1, p2, width = zi
            a, b, p = np.array(p1), np.array(p2), np.array(zj)
            
            ab, ap = b - a, p - a
            proj = np.dot(ap, ab) / np.dot(ab, ab)
            closest = a + proj * ab
            dist = np.linalg.norm(closest - p)
            
            return 0 <= proj <= 1 and dist <= (width / 2 + expanded)
        
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
            
            # Get terrain effects at current position
            constraints = self.environment.get_flight_constraints_at_position(
                drone.position[0], drone.position[1]
            )
            weather_effects = self.environment.get_weather_effects()
            
            # Call the drone's own compute_action method
            action = drone.compute_action(
                goal=self.goals[i],
                avoid=avoidance_mode[i],
                other_drones=other_drones
            )
            
            # Apply terrain speed modifiers
            if hasattr(drone, 'max_speed') and isinstance(action, list):
                # For quadcopters with velocity commands
                speed_mod = constraints['speed_modifier'] * weather_effects['speed_modifier']
                action = [action[0] * speed_mod, action[1] * speed_mod]
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
