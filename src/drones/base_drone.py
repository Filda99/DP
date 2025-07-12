from abc import ABC, abstractmethod

class BaseDrone(ABC):
    def __init__(self, position, heading):
        self.position = position  # [x, y]
        self.heading = heading
        self.speed = 0.0
        self.dt = 0.5
        self.flight_time = 0.0

    @abstractmethod
    def move(self, action):
        pass

    @abstractmethod
    def get_collision_zone(self):
        pass

    @abstractmethod
    def info(self):
        pass
