from typing import List, Union, Tuple
from .base_drone import BaseDrone
import math
import numpy as np

class FixedWing(BaseDrone):
    """Fixed-wing drone implementation with forward flight characteristics.
    
    This class represents a fixed-wing aircraft that must maintain forward motion
    and has limited turning capabilities due to physical constraints.
    
    TBD: Fine-tune flight physics parameters based on real aircraft specifications.
    """
    
    # TBD: Collision detection improvements
    # Current collision computation is basic - could be enhanced with:
    # - Dynamic reaction distance based on speed: reaction_distance = v * reaction_time
    # - If drone flies at v = 14.7 m/s with reaction time of 0.5s: reaction_distance = v * 0.5 
    # - Consider wind effects and maneuverability constraints

    def __init__(self, position: List[float], heading: float) -> None:
        """Initialize fixed-wing drone with flight-specific parameters.
        
        Args:
            position: List containing [x, y] coordinates for initial position
            heading: Initial heading angle in degrees
        """
        super().__init__(position, heading)
        # TBD: These values should be calibrated based on actual aircraft specifications
        self.speed = 10.0            # Constant forward speed in units/second
        self.min_turn_radius = 30.0  # Minimum turning radius constraint (currently unused)

    def move(self, action: float) -> None:
        """Execute movement based on steering angle input.
        
        Fixed-wing aircraft must maintain forward motion and can only change direction
        through steering adjustments, unlike quadcopters that can hover or move laterally.
        
        Args:
            action: Steering angle in degrees (positive = right turn, negative = left turn)
                   Clamped to ±30 degrees to simulate realistic aircraft limitations
        
        TBD: 
        - Implement minimum turn radius constraint using self.min_turn_radius
        - Add realistic acceleration/deceleration curves
        - Consider stall speed limitations
        """
        # Limit steering to realistic aircraft turning capabilities (±30 degrees)
        delta_heading = max(min(action, 30), -30)
        
        # Update heading based on steering input
        self.heading += delta_heading
        
        # Calculate movement vector based on new heading
        rad = math.radians(self.heading)
        dx = self.speed * math.cos(rad)  # East-West component
        dy = self.speed * math.sin(rad)  # North-South component
        
        # Update position based on velocity and time step
        self.position[0] += dx * self.dt
        self.position[1] += dy * self.dt
        
        # Track total flight time
        self.flight_time += self.dt

    def get_collision_zone(self) -> Tuple[List[float], List[float], float]:
        """Get rectangular collision zone extending forward from the aircraft.
        
        Fixed-wing aircraft have an elongated collision zone due to their forward motion
        and limited maneuverability compared to quadcopters.
        
        Returns:
            Tuple containing (rear_point, front_point, width):
            - rear_point: [x, y] coordinates of aircraft's current position
            - front_point: [x, y] coordinates 20 units ahead in flight direction
            - width: Collision zone width (4.0 units)
        
        TBD:
        - Make collision zone length proportional to current speed
        - Consider aircraft wingspan and length for more accurate dimensions
        - Add separate zones for different threat levels (warning vs critical)
        """
        # Calculate forward direction vector
        rad = math.radians(self.heading)
        x, y = self.position
        
        # TBD: Collision zone length (20 units) should be speed-dependent
        collision_length = 20  # Currently fixed, should be: speed * reaction_time + safety_margin
        dx = math.cos(rad) * collision_length
        dy = math.sin(rad) * collision_length
        
        # Define front point of collision rectangle
        front = [x + dx, y + dy]
        
        # TBD: Width (4.0) should reflect actual aircraft dimensions
        collision_width = 4.0  # Should consider wingspan + safety margins
        
        return (self.position, front, collision_width)

    def compute_action(self, goal, avoid=False, other_drones=None):
        """
        Fixed-wing aircraft steering strategy.
        
        Args:
            goal: Target position coordinates as numpy array
            avoid: If True, performs collision avoidance behavior
            other_drones: List of other drones to avoid (currently not used for fixed-wing)
            
        Returns:
            Steering angle in degrees (positive = right turn, negative = left turn)
        """
        if avoid:
            return -15  # Sharp left turn when avoiding collision
        
        # Calculate desired heading to goal
        pos = np.array(self.position)
        vec = np.array(goal) - pos
        target_angle = math.degrees(math.atan2(vec[1], vec[0]))
        
        # Calculate angular difference between current heading and target
        delta = (target_angle - self.heading + 360) % 360
        if delta > 180:
            delta -= 360
        
        # Limit steering angle to realistic aircraft constraints
        return max(min(delta, 15), -15)

    def info(self) -> str:
        """Get current aircraft status information.
        
        Returns:
            String containing aircraft type, position, and heading information
            
        TBD: Add more detailed flight information like speed, altitude, fuel level
        """
        return f"FixedWing at {self.position}, heading {self.heading:.1f}°"
