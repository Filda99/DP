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
        self.mass = 0.496,                 # kg
        self.max_speed = 14.7,             # m/s
        self.ascent_speed = 4.0            # m/s
        self.descent_speed = 4.0           # m/s
        self.wind_resistance = 14.7,       # m/s
        self.max_flight_time = 1920,       # s
        self.collision_radius = calculate_collision_radius(0.373, 0.282)  # m

    def move(self, action):
        '''Move the drone based on the action vector.
        The action is a vector with x and y components, each in the range [-1, 1].
        The drone's speed is scaled by its maximum speed.'''
        dx, dy = np.clip(action, -1, 1) # Ensure action is within [-1, 1]
        vx = dx * self.max_speed
        vy = dy * self.max_speed
        self.position[0] += vx * self.dt 
        self.position[1] += vy * self.dt
        self.flight_time += self.dt

    def get_collision_zone(self):
        '''Return the collision zone as a circle with the drone's position and collision radius.'''
        return (self.position, self.collision_radius)

    def info(self):
        return f"Quadcopter at {self.position}"
