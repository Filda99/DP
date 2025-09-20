from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np

class BaseDrone(ABC):
    '''Base class for all drones. Provides basic properties and methods.'''

    def __init__(self, position: List[float], heading: float, altitude: float = 50.0) -> None:
        '''
        Initialize the drone with position, heading, and altitude.
        
        Args:
            position: List containing [x, y] coordinates of the drone (for backward compatibility)
                     or [x, y, z] coordinates for 3D operation
            heading: Initial heading angle in degrees (will be normalized to 0-360 range)
            altitude: Initial altitude in meters (default: 50.0m) - used if position is 2D
        '''
        # Handle both 2D and 3D position inputs for backward compatibility
        if len(position) == 2:
            # 2D input: [x, y] - add default altitude
            self.position = [position[0], position[1], altitude]
        elif len(position) == 3:
            # 3D input: [x, y, z] - use as provided
            self.position = list(position)
        else:
            raise ValueError("Position must be either [x, y] or [x, y, z]")
            
        self.heading = heading % 360   # Normalize heading to [0, 360), works even if heading < 0
        self.pitch = 0.0               # Pitch angle in degrees (nose up/down)
        self.roll = 0.0                # Roll angle in degrees (wing tilt)
        self.speed = 0.0               # Initial speed is 0
        self.dt = 0.5                  # Time step for movement updates measured in seconds
        self.flight_time = 0.0         # Total flight time in seconds
        
        # Flight envelope constraints
        self.min_altitude = 0.0        # Minimum safe altitude (ground level)
        self.max_altitude = 1000.0     # Maximum operational altitude

    @abstractmethod
    def move(self, action: Union[List[float], float]) -> None:
        """
        Execute movement based on the given action.
        
        Args:
            action: Movement command - format depends on drone type:
                   - For quadcopters: [x_velocity, y_velocity] (2D) or [x_velocity, y_velocity, z_velocity] (3D)
                   - For fixed-wing: steering_angle (float) or [steering_angle, climb_rate] (3D)
        """
        pass

    @abstractmethod
    def get_collision_zone(self) -> Union[List[Union[List[float], float]], List[List[float]]]:
        """
        Get the collision detection zone for this drone.
        
        Returns: 
            Collision zone definition - format depends on drone type:
            - For circular zones (2D): [center_point, radius] where center_point is [x, y]
            - For spherical zones (3D): [center_point, radius] where center_point is [x, y, z]
            - For rectangular zones (2D): [point1, point2, width] where points are [x, y] coordinates
            - For box zones (3D): [point1, point2, width, height] where points are [x, y, z] coordinates
        """
        pass

    @abstractmethod
    def compute_action(self, goal: np.ndarray, avoid: bool = False, 
                      other_drones: List['BaseDrone'] = None) -> Union[List[float], float]:
        """
        Compute the action for this drone based on its goal and environment.
        
        Args:
            goal: Target position coordinates as numpy array (2D or 3D)
            avoid: If True, performs collision avoidance behavior
            other_drones: List of other drones to avoid (for collision avoidance)
            
        Returns:
            Action command - format depends on drone type:
            - For quadcopters: [x_velocity, y_velocity] (2D) or [x_velocity, y_velocity, z_velocity] (3D)
            - For fixed-wing: steering_angle (float) or [steering_angle, climb_rate] (3D)
        """
        pass
    
    def get_2d_position(self) -> List[float]:
        """
        Get the 2D position for backward compatibility.
        
        Returns:
            List containing [x, y] coordinates
        """
        return [self.position[0], self.position[1]]
    
    def get_3d_position(self) -> List[float]:
        """
        Get the full 3D position.
        
        Returns:
            List containing [x, y, z] coordinates
        """
        return self.position.copy()
    
    def get_altitude(self) -> float:
        """
        Get the current altitude.
        
        Returns:
            Current altitude in meters
        """
        return self.position[2]
    
    def set_altitude(self, altitude: float) -> None:
        """
        Set the altitude while respecting flight envelope constraints.
        
        Args:
            altitude: Target altitude in meters
        """
        self.position[2] = max(self.min_altitude, min(altitude, self.max_altitude))

    @abstractmethod
    def info(self) -> str:
        """
        Get information string about the drone's current state.
        
        Returns:
            String containing drone type, position, heading, altitude, and other relevant information
        """
        pass
        pass
