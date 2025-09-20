from .base_drone import BaseDrone
import numpy as np

# Data has been taken from datasheet of Parrot ANAFI USA
# which is stored in /datasheets folder of the project.
class Quadcopter(BaseDrone):
    '''Quadcopter drone class with basic movement and collision detection.'''
    
    def __init__(self, position, heading):
        '''Initialize the drone with 3D position and heading.
        The position must be [x, y, z] and heading is in degrees.'''

        def calculate_collision_radius(width, length):
            '''Calculate the collision radius based on the drone's dimensions.
            The collision radius is half the diagonal of the drone's bounding box,
            plus a safety margin to avoid collisions.'''
            diagonal = (width**2 + length**2)**0.5  #  Pythagorean theorem
            radius = diagonal / 2
            safety_margin = 2.0  # Safety margin to avoid collisions in meters
            collision_radius = radius + safety_margin
            return collision_radius
        
        super().__init__(position, heading)
        self.mass = 0.496                  # kg
        self.max_speed = 14.7              # m/s (horizontal)
        self.ascent_speed = 4.0            # m/s (vertical up)
        self.descent_speed = 4.0           # m/s (vertical down)
        self.max_vertical_speed = 4.0      # m/s (maximum vertical velocity)
        self.wind_resistance = 14.7        # m/s
        self.max_flight_time = 1920        # s
        self.collision_radius = calculate_collision_radius(0.373, 0.282)  # m
        
        # 3D flight envelope constraints
        self.min_altitude = 0.5            # Minimum safe altitude (above ground)
        self.max_altitude = 500.0          # Maximum operational altitude for this model

    def move(self, action):
        '''Execute one time step of 3D movement based on velocity commands.
        
        Args:
            action: 3D velocity vector [vx, vy, vz] where each component is in [-1, 1]
                   representing percentage of maximum speed in each direction.
                   
        The drone moves smoothly by applying constant velocity over the time step duration,
        updating its 3D position linearly.'''
        
        if len(action) != 3:
            raise ValueError("Action must be [vx, vy, vz] - 3D velocity vector required")
        
        dx, dy, dz = np.clip(action, -1, 1)
        
        # Calculate velocities based on action and maximum speeds
        vx = dx * self.max_speed
        vy = dy * self.max_speed
        vz = dz * self.max_vertical_speed
        
        # Update position with velocity and time step
        self.position[0] += vx * self.dt 
        self.position[1] += vy * self.dt
        self.position[2] += vz * self.dt
        
        # Enforce altitude constraints
        self.position[2] = max(self.min_altitude, min(self.position[2], self.max_altitude))
        
        self.flight_time += self.dt

    def compute_action(self, goal, avoid=False, other_drones=None):
        """
        Intelligent quadcopter movement strategy with 3D collision avoidance.
        
        Args:
            goal: Target position coordinates as numpy array [x, y, z]
            avoid: If True, performs collision avoidance behavior
            other_drones: List of other drones to avoid
            
        Returns:
            List of [x_velocity, y_velocity, z_velocity]
        """
        # Convert goal to 3D numpy array
        goal_3d = np.array(goal)
        if len(goal_3d) == 2:
            # If 2D goal provided, maintain current altitude
            goal_3d = np.array([goal[0], goal[1], self.position[2]])
        
        # Calculate direction vector from current position to goal
        vec = goal_3d - np.array(self.position)
        norm = np.linalg.norm(vec)
        
        # If very close to goal, stop moving
        if norm < 1.0:
            return [0, 0, 0]
        
        goal_direction = vec / norm
        
        # If avoiding collision, compute intelligent 3D avoidance
        if avoid and other_drones:
            avoidance_vec = np.zeros(3)  # Always compute in 3D
            
            # Sum repulsion forces from all other drones
            for other_drone in other_drones:
                other_pos = np.array(other_drone.position)
                repulsion = np.array(self.position) - other_pos
                dist = np.linalg.norm(repulsion)
                if dist > 0:
                    # Stronger repulsion when closer, with 3D consideration
                    strength = 1.0 / max(dist, 0.1)  # Avoid division by zero
                    
                    # Add altitude separation preference
                    if abs(repulsion[2]) < 5.0:  # If altitude difference is small
                        # Prefer vertical separation over horizontal
                        if self.position[2] > other_pos[2]:
                            repulsion[2] += 5.0  # Go higher
                        else:
                            repulsion[2] -= 5.0  # Go lower
                    
                    avoidance_vec += strength * (repulsion / dist)
            
            avoidance_norm = np.linalg.norm(avoidance_vec)
            if avoidance_norm > 0:
                avoidance_direction = avoidance_vec / avoidance_norm
                
                # Combine goal-seeking with collision avoidance
                avoidance_weight = 2.0
                goal_weight = 0.5
                
                combined_vec = goal_weight * goal_direction + avoidance_weight * avoidance_direction
                combined_norm = np.linalg.norm(combined_vec)
                
                if combined_norm > 0:
                    result = (combined_vec / combined_norm)
                    return result.tolist()      # Return full 3D vector
        
        # Normal goal-seeking behavior
        return goal_direction.tolist()      # Return full 3D vector

    def get_collision_zone(self):
        '''Return the collision zone as a sphere with the drone's position and collision radius.
        
        Returns:
            Tuple of (center_position, radius) where:
            - center_position is [x, y, z] 3D position
            - radius is the collision detection radius
        '''
        return (self.position.copy(), self.collision_radius)

    def info(self):
        x, y, z = self.position
        return f"Quadcopter at [{x:.1f}, {y:.1f}, {z:.1f}m] (altitude: {z:.1f}m)"
