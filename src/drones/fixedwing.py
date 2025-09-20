from typing import List, Tuple
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
            position: List containing [x, y, z] coordinates for initial position
            heading: Initial heading angle in degrees
        """
        super().__init__(position, heading)
        # TBD: These values should be calibrated based on actual aircraft specifications
        self.speed = 10.0            # Constant forward speed in units/second
        self.min_turn_radius = 30.0  # Minimum turning radius constraint (currently unused)
        self.max_climb_rate = 5.0    # Maximum climb rate in m/s
        self.max_descent_rate = 3.0  # Maximum descent rate in m/s
        self.stall_speed = 8.0       # Minimum forward speed to maintain flight
        
        # 3D flight envelope constraints for fixed-wing
        self.min_altitude = 10.0     # Minimum safe altitude (higher than quadcopters)
        self.max_altitude = 2000.0   # Higher operational ceiling

    def move(self, action: List[float]) -> None:
        """Execute movement based on steering and climb input.
        
        Fixed-wing aircraft must maintain forward motion and can change direction
        through steering adjustments and altitude through climb/descent rates.
        
        Args:
            action: [steering_angle, climb_rate] - steering in degrees, climb rate in [-1, 1]
                   
        TBD: 
        - Implement minimum turn radius constraint using self.min_turn_radius
        - Add realistic acceleration/deceleration curves
        - Consider stall speed limitations
        """
        if len(action) != 2:
            raise ValueError("Action must be [steering_angle, climb_rate]")
        
        steering_angle, climb_rate = action
        delta_heading = max(min(steering_angle, 30), -30)
        climb_input = max(min(climb_rate, 1.0), -1.0)  # Normalize climb input
        
        # Update heading based on steering input
        self.heading += delta_heading
        
        # Calculate horizontal movement vector based on heading
        rad = math.radians(self.heading)
        dx = self.speed * math.cos(rad)  # East-West component
        dy = self.speed * math.sin(rad)  # North-South component
        
        # Calculate vertical movement based on climb input
        if climb_input > 0:
            dz = climb_input * self.max_climb_rate  # Climbing
        else:
            dz = climb_input * self.max_descent_rate  # Descending (negative)
        
        # Update position based on velocity and time step
        self.position[0] += dx * self.dt
        self.position[1] += dy * self.dt
        self.position[2] += dz * self.dt
        
        # Enforce altitude constraints
        self.position[2] = max(self.min_altitude, min(self.position[2], self.max_altitude))
        
        # Track total flight time
        self.flight_time += self.dt

    def get_collision_zone(self) -> Tuple[List[float], List[float], float, float]:
        """Get 3D box collision zone extending forward from the aircraft.
        
        Fixed-wing aircraft have an elongated collision zone due to their forward motion
        and limited maneuverability compared to quadcopters.
        
        Returns:
            Tuple containing (rear_point, front_point, width, height):
            - rear_point: [x, y, z] coordinates of aircraft's current position
            - front_point: [x, y, z] coordinates ahead in flight direction
            - width: Collision zone width (horizontal span)
            - height: Collision zone height (vertical span)
        
        TBD:
        - Make collision zone length proportional to current speed
        - Consider aircraft wingspan and length for more accurate dimensions
        - Add separate zones for different threat levels (warning vs critical)
        """
        # Calculate forward direction vector
        rad = math.radians(self.heading)
        x, y, z = self.position
        
        # TBD: Collision zone length (20 units) should be speed-dependent
        collision_length = 20  # Currently fixed, should be: speed * reaction_time + safety_margin
        dx = math.cos(rad) * collision_length
        dy = math.sin(rad) * collision_length
        
        # Define front point of collision box (same altitude as current)
        front = [x + dx, y + dy, z]
        
        # TBD: Dimensions should reflect actual aircraft dimensions
        collision_width = 4.0   # Should consider wingspan + safety margins
        collision_height = 3.0  # Should consider aircraft height + safety margins
        
        return (self.position.copy(), front, collision_width, collision_height)

    def compute_action(self, goal, avoid=False, other_drones=None):
        """
        Fixed-wing aircraft 3D steering and climb strategy.
        
        Args:
            goal: Target position coordinates as numpy array [x, y, z]
            avoid: If True, performs collision avoidance behavior
            other_drones: List of other drones to avoid
            
        Returns:
            [steering_angle, climb_rate] (list)
        """
        # Convert goal to 3D
        goal_3d = np.array(goal)
        if len(goal_3d) == 2:
            # If 2D goal provided, maintain current altitude
            goal_3d = np.array([goal[0], goal[1], self.position[2]])
        
        if avoid:
            # Basic avoidance maneuver
            return [-15, 0.5]  # Sharp left turn and climb
        
        # Calculate desired heading to goal (horizontal component)
        pos = np.array(self.position)
        vec_horizontal = goal_3d[:2] - pos[:2]  # Only x, y components
        target_angle = math.degrees(math.atan2(vec_horizontal[1], vec_horizontal[0]))
        
        # Calculate angular difference between current heading and target
        delta = (target_angle - self.heading + 360) % 360
        if delta > 180:
            delta -= 360
        
        # Limit steering angle to realistic aircraft constraints
        steering = max(min(delta, 15), -15)
        
        # Calculate desired climb rate for 3D operation
        altitude_diff = goal_3d[2] - pos[2]
        # Simple climb rate calculation
        if abs(altitude_diff) < 2.0:  # Close to target altitude
            climb_rate = 0.0
        elif altitude_diff > 0:  # Need to climb
            climb_rate = min(altitude_diff / 20.0, 1.0)  # Gradual climb
        else:  # Need to descend
            climb_rate = max(altitude_diff / 20.0, -1.0)  # Gradual descent
        
        return [steering, climb_rate]

    def info(self) -> str:
        """Get current aircraft status information.
        
        Returns:
            String containing aircraft type, position, heading, and altitude information
            
        TBD: Add more detailed flight information like speed, fuel level
        """
        x, y, z = self.position
        return f"FixedWing at [{x:.1f}, {y:.1f}, {z:.1f}m], heading {self.heading:.1f}°, altitude {z:.1f}m"
