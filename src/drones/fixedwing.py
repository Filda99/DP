"""
Fixed-Wing Drone Class

Fixed-wing aircraft implementation with forward flight requirements.
Cannot hover, requires forward motion for lift generation.
"""

import pybullet as p
import numpy as np
from .base_drone import BaseDrone


class FixedWing(BaseDrone):
    """Fixed-wing aircraft with forward flight requirements."""
    
    def __init__(self, position=[0, 0, 5], mass=1.0, max_thrust=20.0, min_speed=3.0):
        """Initialize fixed-wing aircraft."""
        # Create PyBullet body first
        self.drone_id = self._create_pybullet_body(position, mass)
        
        # Initialize base class
        super().__init__(self.drone_id, position, mass)
        
        # Fixed-wing specific parameters
        self.max_thrust = max_thrust
        self.min_speed = min_speed  # Minimum speed to maintain lift
        self.current_heading = 0.0  # Current heading in radians
        self.turn_rate = 1.0  # Max turn rate (rad/s)
        
        # Aerodynamic properties
        self.lift_coefficient = 0.8
        self.drag_coefficient = 0.05
        self.wing_area = 0.5  # m²
        self.current_air_density = 1.225  # kg/m³ (default at sea level, updated by atmospheric conditions)
        
    def _create_pybullet_body(self, position, mass):
        """Create fixed-wing body in PyBullet."""
        # Create a more aerodynamic shape (elongated box)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.1, 0.05])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.3, 0.1, 0.05], 
            rgbaColor=[0, 0, 1, 1]  # Blue fixed-wing
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
        Convert joystick input to forces for fixed-wing aircraft.
        
        Args:
            joystick_input: [turn_left_right, throttle, climb_dive] in range [-1, 1]
            
        Returns:
            forces: [force_x, force_y, force_z] in Newtons
        """
        # Fixed-wing control interpretation:
        # [0] = turn left/right (changes heading)
        # [1] = throttle (forward thrust)
        # [2] = climb/dive (elevator)
        
        turn_input = joystick_input[0]
        throttle_input = joystick_input[1]
        elevator_input = joystick_input[2]
        
        # Update heading based on turn input
        self.current_heading += turn_input * self.turn_rate * (1/240)  # Assuming 240 FPS
        
        # Forward thrust (always needs some thrust to maintain flight)
        base_thrust = self.max_thrust * 0.3  # Minimum thrust to maintain flight
        thrust = base_thrust + (throttle_input * 0.5 + 0.5) * (self.max_thrust - base_thrust)
        
        # Calculate forces in world coordinates
        force_x = thrust * np.cos(self.current_heading)
        force_y = thrust * np.sin(self.current_heading)
        
        # Vertical force (elevator control + basic lift)
        current_speed = self.get_speed()
        
        # Generate lift based on forward speed
        if current_speed > 0.1:
            lift = 0.5 * self.current_air_density * (current_speed ** 2) * self.wing_area * self.lift_coefficient
        else:
            lift = 0
        
        # Combine lift with elevator input
        force_z = lift + elevator_input * self.max_thrust * 0.3 - self.gravity_compensation
        
        return np.array([force_x, force_y, force_z])
    
    def apply_control(self, joystick_input):
        """Apply fixed-wing control forces."""
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
        
        # Apply drag
        self._apply_drag()
        
        return forces
    
    def _apply_drag(self):
        """Apply aerodynamic drag."""
        velocity = self.get_velocity()
        speed = np.linalg.norm(velocity)
        
        if speed > 0.1:
            # Drag force opposes motion
            drag_magnitude = 0.5 * self.current_air_density * (speed ** 2) * self.wing_area * self.drag_coefficient
            drag_direction = -velocity / speed
            drag_force = drag_direction * drag_magnitude
            
            p.applyExternalForce(
                self.drone_id,
                -1,
                drag_force.tolist(),
                [0, 0, 0],
                p.WORLD_FRAME
            )
    
    def apply_wind_effect(self, wind_velocity):
        """Apply wind forces to the fixed-wing aircraft."""
        # Fixed-wing is more affected by wind due to larger surface area
        wind_force = np.array(wind_velocity) * 0.2  # Higher wind resistance
        
        p.applyExternalForce(
            self.drone_id,
            -1,
            wind_force.tolist(),
            [0, 0, 0],
            p.WORLD_FRAME
        )
        
        return wind_force
    
    def get_drone_type(self):
        """Get drone type string."""
        return "Fixed-Wing"
    
    def can_hover(self):
        """Check if drone can hover."""
        return False
    
    def is_stalling(self):
        """Check if aircraft is stalling (speed too low)."""
        return self.get_speed() < self.min_speed
    
    def get_heading(self):
        """Get current heading in radians."""
        return self.current_heading
    
    def get_heading_degrees(self):
        """Get current heading in degrees."""
        return np.degrees(self.current_heading)
    
    def get_flight_characteristics(self):
        """Get flight characteristics dictionary."""
        return {
            'type': self.get_drone_type(),
            'mass': self.mass,
            'can_hover': self.can_hover(),
            'max_thrust': self.max_thrust,
            'min_speed': self.min_speed,
            'current_heading_deg': self.get_heading_degrees(),
            'is_stalling': self.is_stalling(),
            'lift_coefficient': self.lift_coefficient,
            'drag_coefficient': self.drag_coefficient
        }
    
    def apply_environmental_effects(self, atmospheric_conditions: dict):
        """
        Apply drag force based on air relative velocity and local atmospheric density.
        
        Args:
            atmospheric_conditions: dict with keys:
                - 'velocity': [u, v, w] airflow vector (m/s)
                - 'temperature': local temperature (K)
                - 'density': local air density (kg/m³)
        """
        # Extract atmospheric conditions
        local_airflow = atmospheric_conditions['velocity']
        local_density = atmospheric_conditions.get('density', 1.225)  # kg/m³
        
        # Store density for use in lift calculation
        self.current_air_density = local_density
        
        # Aircraft velocity relative to the air (v_air_relative = v_aircraft - u_air)
        aircraft_velocity = self.get_velocity()
        air_relative_velocity = aircraft_velocity - local_airflow
        v_air = np.linalg.norm(air_relative_velocity)
        
        # Drag Force: F_drag = 0.5 * ρ * v²_air * C_D * A * (-v̂_air_relative)
        drag_magnitude = 0.5 * local_density * (v_air ** 2) * self.drag_coefficient * self.wing_area
        
        # Drag acts opposite to the direction of relative air velocity
        drag_force = -drag_magnitude * (air_relative_velocity / v_air) if v_air > 0.1 else np.array([0., 0., 0.])

        # Apply the total external force
        p.applyExternalForce(
            self.drone_id,
            -1,
            drag_force.tolist(),
            [0, 0, 0],
            p.WORLD_FRAME
        )