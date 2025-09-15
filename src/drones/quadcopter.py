from .base_drone import BaseDrone
import numpy as np

# Data has been taken from datasheet of Parrot ANAFI USA
# which is stored in /datasheets folder of the project.
class Quadcopter(BaseDrone):
    '''Quadcopter drone class with basic movement and collision detection.'''
    
    def __init__(self, position, heading):
        '''Initialize the drone with position and heading.
        The position is a list [x, y] and heading is in degrees.'''

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
        self.max_speed = 14.7              # m/s
        self.ascent_speed = 4.0            # m/s
        self.descent_speed = 4.0           # m/s
        self.wind_resistance = 14.7        # m/s
        self.max_flight_time = 1920        # s
        self.collision_radius = calculate_collision_radius(0.373, 0.282)  # m

    def move(self, action):
        '''Execute one time step of linear movement based on normalized velocity commands.
        
        Args:
            action: Normalized velocity vector [vx, vy] where each component is in [-1, 1]
                   representing percentage of maximum speed in each direction.
                   
        The drone moves smoothly by applying constant velocity over the time step duration,
        updating its position linearly rather than teleporting instantly.'''
        dx, dy = np.clip(action, -1, 1) # Ensure action is within [-1, 1]
        vx = dx * self.max_speed
        vy = dy * self.max_speed
        self.position[0] += vx * self.dt 
        self.position[1] += vy * self.dt
        self.flight_time += self.dt

    def compute_action(self, goal, avoid=False, other_drones=None):
        """
        Intelligent quadcopter movement strategy with collision avoidance.
        
        Args:
            goal: Target position coordinates as numpy array
            avoid: If True, performs collision avoidance behavior
            other_drones: List of other drones to avoid
            
        Returns:
            List of [x_velocity, y_velocity] for movement
        """
        # Calculate direction vector from current position to goal
        vec = np.array(goal) - np.array(self.position)
        norm = np.linalg.norm(vec)
        
        # If very close to goal, stop moving
        if norm < 1.0:
            return [0, 0]
        
        goal_direction = vec / norm
        
        # If avoiding collision, compute intelligent avoidance
        if avoid and other_drones:
            avoidance_vec = np.zeros(2)
            
            # Sum repulsion forces from all other drones
            for other_drone in other_drones:
                repulsion = np.array(self.position) - np.array(other_drone.position)
                dist = np.linalg.norm(repulsion)
                if dist > 0:
                    # Stronger repulsion when closer
                    strength = 1.0 / max(dist, 0.1)  # Avoid division by zero
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
                    return (combined_vec / combined_norm).tolist()
        
        return goal_direction.tolist()

    def get_collision_zone(self):
        '''Return the collision zone as a circle with the drone's position and collision radius.'''
        return (self.position, self.collision_radius)

    def info(self):
        return f"Quadcopter at {self.position}"
