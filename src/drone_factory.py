from drones.quadcopter import Quadcopter
from drones.fixedwing import FixedWing

def create_drone(drone_type, position, heading):
    if drone_type == "quadcopter":
        return Quadcopter(position, heading)
    elif drone_type == "fixedwing":
        return FixedWing(position, heading)
    else:
        raise ValueError(f"Unknown drone type: {drone_type}")
