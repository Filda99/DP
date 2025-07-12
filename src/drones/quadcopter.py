from .base_drone import BaseDrone
import numpy as np

class Quadcopter(BaseDrone):
    def __init__(self, position, heading):
        super().__init__(position, heading)
        self.mass = 0.246
        self.max_speed = 16.0
        self.ascent_speed = 5.0
        self.descent_speed = 3.5
        self.wind_resistance = 10.7
        self.max_flight_time = 1860  # s
        self.collision_radius = 2.0

    def move(self, action):
        dx, dy = np.clip(action, -1, 1)
        vx = dx * self.max_speed
        vy = dy * self.max_speed
        self.position[0] += vx * self.dt
        self.position[1] += vy * self.dt
        self.flight_time += self.dt

    def get_collision_zone(self):
        return (self.position, self.collision_radius)

    def info(self):
        return f"Quadcopter at {self.position}"
