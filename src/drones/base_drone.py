"""
Base Drone Class

Abstract base class for all drone types with common physics and control interfaces.
"""

import pybullet as p
import numpy as np
from abc import ABC, abstractmethod


class BaseDrone(ABC):
    """Abstract base class for all drone types."""
    
    def __init__(self, drone_id, position=[0, 0, 5], mass=0.5, environment=None):
        """Initialize base drone with PyBullet physics."""
        self.drone_id = drone_id
        self.position = np.array(position)
        self.mass = mass
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Flight data logging
        self.flight_log = {
            'positions': [],
            'forces': [],
            'velocities': [],
            'times': []
        }
        
        # Physics properties
        self.gravity_compensation = mass * 9.81
        
    def get_position(self):
        """Get current position from PyBullet."""
        if self.drone_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.drone_id)
            self.position = np.array(pos)
        return self.position
    
    def get_velocity(self):
        """Get current velocity from PyBullet."""
        if self.drone_id is not None:
            vel, _ = p.getBaseVelocity(self.drone_id)
            self.velocity = np.array(vel)
        return self.velocity
    
    def get_speed(self):
        """Get current speed (velocity magnitude)."""
        return np.linalg.norm(self.get_velocity())
    
    def log_flight_data(self, forces, time_step):
        """Log flight data for analysis."""
        self.flight_log['positions'].append(self.get_position().copy())
        self.flight_log['forces'].append(forces.copy())
        self.flight_log['velocities'].append(self.get_velocity().copy())
        self.flight_log['times'].append(time_step)
    
    def get_flight_log(self):
        """Get complete flight log."""
        return self.flight_log
    
    def clear_flight_log(self):
        """Clear flight log."""
        self.flight_log = {
            'positions': [],
            'forces': [],
            'velocities': [],
            'times': []
        }
    
    @abstractmethod
    def joystick_to_forces(self, joystick_input):
        """Convert joystick input to forces. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def apply_control(self, joystick_input):
        """Apply control forces. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_drone_type(self):
        """Get drone type string. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def apply_environmental_effects(self, atmospheric_conditions: dict):
        """
        Apply environmental forces based on local atmospheric conditions.
        
        Args:
            atmospheric_conditions: dict with keys:
                - 'velocity': [u, v, w] airflow vector (m/s)
                - 'temperature': local temperature (K)
                - 'density': local air density (kg/m³)
        """
        pass