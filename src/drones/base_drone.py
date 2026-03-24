"""
Base Drone Class

Abstract base class for all drone types with common physics and control interfaces.
Includes PID Controller implementation for stable flight dynamics.
"""

import pybullet as p
import numpy as np
from abc import ABC, abstractmethod
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class PIDController:
    """Standard PID Controller implementation."""
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0
        
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error, dt, debug=False):
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * dt
        if self.integral_limit:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative
        self.prev_error = error
        
        output = p_term + i_term + d_term
        
        if self.output_limit:
            output = np.clip(output, -self.output_limit, self.output_limit)
            
        if debug:
            return f"P:{p_term:.2f} I:{i_term:.2f} D:{d_term:.2f} Out:{output:.2f}"
            
        return output

class BaseDrone(ABC):
    """Abstract base class for all drone types."""
    
    def __init__(self, drone_id, position=[0, 0, 5], mass=0.5, environment=None):
        """Initialize base drone with PyBullet physics."""
        self.drone_id = drone_id
        self.position = np.array(position)
        self.mass = mass
        self.environment = environment
        
        # Water tank properties
        self.water_capacity = 0.0
        self.current_water = 0.0
        self.water_valve_open = False
        
        # Physics properties
        self.gravity = 9.81
        self.gravity_compensation = mass * self.gravity
        
        # Logging
        self.flight_log = {'positions': [], 'forces': [], 'velocities': [], 'times': []}
        
    def get_position(self):
        if self.drone_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.drone_id)
            return np.array(pos)
        return np.zeros(3)
    
    def get_velocity(self):
        if self.drone_id is not None:
            vel, _ = p.getBaseVelocity(self.drone_id)
            return np.array(vel)
        return np.zeros(3)
        
    def get_orientation_quaternion(self):
        if self.drone_id is not None:
            _, orn = p.getBasePositionAndOrientation(self.drone_id)
            return np.array(orn)
        return np.array([0, 0, 0, 1])
        
    def get_orientation_rpy(self):
        """Get Roll, Pitch, Yaw in radians."""
        quat = self.get_orientation_quaternion()
        return np.array(p.getEulerFromQuaternion(quat))
    
    def get_speed(self):
        return np.linalg.norm(self.get_velocity())

    def open_water_valve(self):
        self.water_valve_open = True
        
    def close_water_valve(self):
        self.water_valve_open = False
        
    def can_drop_water(self):
        return self.water_valve_open and self.current_water > 0
        
    def consume_water(self, amount):
        dropped = min(self.current_water, amount)
        self.current_water -= dropped
        return dropped
    
    def refill_tank(self):
        """Refills the water tank to maximum capacity."""
        if self.current_water < self.water_capacity:
            self.current_water = self.water_capacity
            # print(f"{self.drone_id}: Tank refilled to {self.water_capacity}L")
            return True
        return False

    @abstractmethod
    def apply_control(self, joystick_input, dt):
        pass
    
    @abstractmethod
    def apply_environmental_effects(self, atmospheric_conditions: dict):
        pass
    
    @abstractmethod
    def get_drone_type(self):
        pass
        
    @abstractmethod
    def get_flight_characteristics(self):
        pass