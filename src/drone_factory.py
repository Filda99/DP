from typing import List
from drones.quadcopter import Quadcopter
from drones.fixedwing import FixedWing
from drones.base_drone import BaseDrone

def create_drone(drone_type: str, position: List[float], heading: float) -> BaseDrone:
    """Factory function to create different types of drones.
    
    This function serves as a factory pattern implementation for creating drone instances.
    It abstracts the drone creation process and allows for easy extension with new drone types.
    
    Args:
        drone_type: String identifier for the type of drone to create.
                   Supported values: "quadcopter", "fixedwing"
        position: List containing [x, y, z] coordinates for the initial drone position
        heading: Initial heading angle in degrees (0-360, where 0 is East, 90 is North)
    
    Returns:
        BaseDrone: An instance of the requested drone type (Quadcopter or FixedWing)
    
    Raises:
        ValueError: If the drone_type is not recognized or supported
    
    Example:
        >>> quad = create_drone("quadcopter", [0.0, 0.0, 30.0], 45.0)
        >>> wing = create_drone("fixedwing", [10.0, 20.0, 100.0], 180.0)
    """
    # Create quadcopter drone with hovering capabilities
    if drone_type == "quadcopter":
        return Quadcopter(position, heading)
    
    # Create fixed-wing drone with forward flight characteristics
    elif drone_type == "fixedwing":
        return FixedWing(position, heading)
    
    # Unsupported drone type - raise informative error
    else:
        raise ValueError(f"Unknown drone type: {drone_type}")
