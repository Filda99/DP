from abc import ABC, abstractmethod
from typing import List, Union, Tuple

class BaseDrone(ABC):
    '''Base class for all drones. Provides basic properties and methods.'''

    def __init__(self, position: List[float], heading: float) -> None:
        '''Initialize the drone with position and heading.
        
        Args:
            position: List containing [x, y] coordinates of the drone
            heading: Initial heading angle in degrees (will be normalized to 0-360 range)
        '''
        self.position = position        # [x, y]
        self.heading  = heading % 360   # Normalize heading to [0, 360), works even if heading < 0
        self.speed = 0.0    # Initial speed is 0
        self.dt = 0.5       # Time step for movement updates measured in seconds
        self.flight_time = 0.0          # Total flight time in seconds

    @abstractmethod
    def move(self, action: Union[List[float], float]) -> None:
        """Execute movement based on the given action.
        
        Args:
            action: Movement command - format depends on drone type:
                   - For quadcopters: [x_velocity, y_velocity] 
                   - For fixed-wing: steering_angle (float)
        """
        pass

    @abstractmethod
    def get_collision_zone(self) -> Union[List[Union[List[float], float]], List[List[float]]]:
        """Get the collision detection zone for this drone.
        
        Returns: 
            Collision zone definition - format depends on drone type:
            - For circular zones: [center_point, radius] where center_point is [x, y]
            - For rectangular zones: [point1, point2, width] where points are [x, y] coordinates
        """
        pass

    @abstractmethod
    def info(self) -> str:
        """Get information string about the drone's current state.
        
        Returns:
            String containing drone type, position, heading, and other relevant information
        """
        pass
