from .base_drone import BaseDrone
import math

class FixedWing(BaseDrone):
    def __init__(self, position, heading):
        super().__init__(position, heading)
        self.speed = 10.0
        self.min_turn_radius = 30.0

    def move(self, action):
        delta_heading = max(min(action, 30), -30)
        self.heading += delta_heading
        rad = math.radians(self.heading)
        dx = self.speed * math.cos(rad)
        dy = self.speed * math.sin(rad)
        self.position[0] += dx * self.dt
        self.position[1] += dy * self.dt
        self.flight_time += self.dt

    def get_collision_zone(self):
        rad = math.radians(self.heading)
        x, y = self.position
        dx = math.cos(rad) * 20
        dy = math.sin(rad) * 20
        front = [x + dx, y + dy]
        return (self.position, front, 4.0)

    def info(self):
        return f"FixedWing at {self.position}, heading {self.heading:.1f}°"
