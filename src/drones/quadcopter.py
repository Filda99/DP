"""
Quadcopter Drone Class

Quadcopter implementation with hover capabilities and omnidirectional movement.
Based on realistic flight dynamics with PyBullet physics.
"""

import pybullet as p
import numpy as np
from .base_drone import BaseDrone


class Quadcopter(BaseDrone):
    """Quadcopter drone with hover capabilities."""
    
    def __init__(self, position=[0, 0, 5], mass=0.5, max_horizontal_force=10.0, max_vertical_force=15.0):
        """Initialize quadcopter."""
        # Create PyBullet body first
        self.drone_id = self._create_pybullet_body(position, mass)
        
        # Initialize base class
        super().__init__(self.drone_id, position, mass)
        
        # Quadcopter-specific parameters
        self.max_horizontal_force = max_horizontal_force
        self.max_vertical_force = max_vertical_force
        
        # Control limits
        self.max_tilt_angle = np.pi / 6  # 30 degrees max tilt
        
    def _create_pybullet_body(self, position, mass):
        """Create quadcopter body in PyBullet."""
        # Create collision and visual shapes
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.1, 0.1, 0.05], 
            rgbaColor=[1, 0, 0, 1]  # Red quadcopter
        )
        
        # Create multibody
        drone_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        return drone_id
    
    def joystick_to_forces(self, joystick_input):
        """
        Convert joystick input to forces for quadcopter.
        
        Args:
            joystick_input: [left_right, forward_back, up_down] in range [-1, 1]
            
        Returns:
            forces: [force_x, force_y, force_z] in Newtons
        """
        # Horizontal forces (X, Y)
        force_x = joystick_input[0] * self.max_horizontal_force
        force_y = joystick_input[1] * self.max_horizontal_force
        
        # Vertical force (Z) - hover compensation + input
        hover_force = self.gravity_compensation
        vertical_input = joystick_input[2] * self.max_vertical_force
        force_z = hover_force + vertical_input
        
        return np.array([force_x, force_y, force_z])
    
    def apply_control(self, joystick_input):
        """Apply quadcopter control forces."""
        # Convert joystick to forces
        forces = self.joystick_to_forces(joystick_input)
        
        # Apply force to center of mass
        p.applyExternalForce(
            self.drone_id,
            -1,  # Apply to base link
            forces.tolist(),
            [0, 0, 0],  # Force position (center)
            p.WORLD_FRAME
        )
        
        return forces
    
    def apply_wind_effect(self, wind_velocity):
        """Apply wind forces with automatic compensation (like real drones)."""
        wind_force = np.array(wind_velocity) * 0.667  # Realistic wind resistance coefficient
        
        # Real drone automatic wind compensation capability
        # Modern drones can compensate up to ~80% of their max thrust for wind
        max_wind_compensation = self.max_horizontal_force * 0.8  # 8N compensation capability
        
        # Calculate horizontal wind force (ignore vertical for position hold)
        horizontal_wind = wind_force[:2]  # Only X, Y components
        wind_magnitude = np.linalg.norm(horizontal_wind)
        
        if wind_magnitude <= max_wind_compensation:
            # Wind is within capability - drone automatically compensates and holds position
            # Apply ONLY vertical wind component (horizontal is fully compensated)
            remaining_wind = np.array([0, 0, wind_force[2]])
            
            # No net horizontal wind force applied - drone holds position!
            
        else:
            # Wind exceeds compensation capability - partial compensation + remaining drift
            compensation_factor = max_wind_compensation / wind_magnitude
            
            # Calculate net wind effect after partial compensation
            compensated_horizontal = horizontal_wind * (1.0 - compensation_factor)
            remaining_wind = np.array([compensated_horizontal[0], compensated_horizontal[1], wind_force[2]])
        
        # Apply only the remaining (uncompensated) wind effects
        if np.linalg.norm(remaining_wind) > 0.05:
            p.applyExternalForce(
                self.drone_id,
                -1,
                remaining_wind.tolist(),
                [0, 0, 0],
                p.WORLD_FRAME
            )
        
        return wind_force
    
    def get_drone_type(self):
        """Get drone type string."""
        return "Quadcopter"
    
    def get_hover_force(self):
        """Get the force needed to hover."""
        return self.gravity_compensation
    
    def can_hover(self):
        """Check if drone can hover."""
        return True
    
    def get_max_speed(self):
        """Get theoretical maximum speed."""
        # Based on max horizontal force and mass
        max_acceleration = self.max_horizontal_force / self.mass
        return max_acceleration * 2.0  # Rough estimate
    
    def get_flight_characteristics(self):
        """Get flight characteristics dictionary."""
        return {
            'type': self.get_drone_type(),
            'mass': self.mass,
            'can_hover': self.can_hover(),
            'max_horizontal_force': self.max_horizontal_force,
            'max_vertical_force': self.max_vertical_force,
            'hover_force': self.get_hover_force(),
            'max_speed_estimate': self.get_max_speed()
        }