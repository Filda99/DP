"""
Fixed-Wing Drone Class

Aerodynamic implementation based on Lift/Drag coefficients and Angle of Attack.
Controls: Autopilot assisted (Bank Angle, Pitch Angle, Throttle).
CORRECTED: Aerodynamics now depends on Airspeed (Velocity - Wind), not Ground Speed.
"""

import pybullet as p
import numpy as np
from .base_drone import BaseDrone, PIDController

class FixedWing(BaseDrone):
    def __init__(self, position=[0, 0, 5], mass=1.0, max_thrust=30.0, min_speed=5.0, water_capacity=50.0):
        self.drone_id = self._create_pybullet_body(position, mass)
        super().__init__(self.drone_id, position, mass)
        
        self.water_capacity = water_capacity
        self.current_water = water_capacity
        
        # Flight Envelope
        self.max_thrust = max_thrust
        self.max_speed = 25.0
        self.stall_speed = 6.0
        
        # Aerodynamics (Cessna-like approximation)
        self.wing_area = 0.8
        self.wing_span = 1.5
        self.aspect_ratio = (self.wing_span**2) / self.wing_area
        
        # Controllers (Autopilot)
        self.pid_bank = PIDController(kp=2.0, ki=0.05, kd=0.5)
        self.pid_pitch = PIDController(kp=2.0, ki=0.05, kd=0.5)
        
    def _create_pybullet_body(self, position, mass):
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.8, 0.05]) 
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.8, 0.05], rgbaColor=[0.2, 0.2, 0.9, 1])
        drone_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        return drone_id

    def _calculate_aerodynamics(self, velocity, orientation_quat, air_density):
        """
        Calculates Lift and Drag forces based on Airspeed and Angle of Attack.
        """
        # 1. Get Wind Vector
        wind_vec = np.zeros(3)
        if self.environment:
            # Note: We assume uniform wind for simplicity, or query specific pos
            wind_vec = self.environment.get_wind_at_position(self.get_position())
            
        # 2. Calculate Airspeed Vector (Relative Velocity)
        # V_air = V_ground - V_wind
        airspeed_vec_world = velocity - wind_vec
        airspeed = np.linalg.norm(airspeed_vec_world)
        
        # If too slow, no aerodynamics
        if airspeed < 0.1: return np.zeros(3), np.zeros(3)

        # Rotation matrix: Body -> World
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation_quat)).reshape(3, 3)
        
        # Transform Airspeed to Body Frame (to calculate Alpha)
        airspeed_body = rot_matrix.T @ airspeed_vec_world
        
        # Angle of Attack (Alpha) = atan(-w / u)
        # Using body-frame vertical and forward airspeed components
        alpha = np.arctan2(-airspeed_body[2], airspeed_body[0])
        
        # Aerodynamic Coefficients
        stall_angle = np.radians(15)
        if abs(alpha) < stall_angle:
            CL = 2.0 * np.pi * alpha + 0.3
        else:
            CL = 0.8 * np.sign(alpha) * np.exp(-5 * (abs(alpha) - stall_angle))
            
        CD = 0.05 + (CL**2) / (np.pi * self.aspect_ratio * 0.8)
        
        # Dynamic Pressure (q = 0.5 * rho * v^2)
        # CRITICAL: Using AIRSPEED, not ground speed
        q = 0.5 * air_density * (airspeed**2) * self.wing_area
        
        lift_mag = q * CL
        drag_mag = q * CD
        
        # Force Vectors in Body Frame
        # Simplified: Lift is perpendicular to airflow, Drag is parallel
        # We approximate Lift as mostly Up (Z) in body frame for small alphas
        F_lift_body = np.array([-lift_mag * np.sin(alpha), 0, lift_mag * np.cos(alpha)])
        F_drag_body = np.array([-drag_mag * np.cos(alpha), 0, -drag_mag * np.sin(alpha)])
        
        # Rotate back to World Frame
        F_lift_world = rot_matrix @ F_lift_body
        F_drag_world = rot_matrix @ F_drag_body
        
        return F_lift_world, F_drag_world

    def apply_control(self, joystick_input):
        """
        Autopilot Control.
        Input: [Target_Bank_Input, Throttle, Target_Pitch_Input]
        """
        dt = 1/60.0
        
        # Inputs
        target_bank = -joystick_input[0] * np.radians(45) 
        throttle = np.clip(joystick_input[1], 0.0, 1.0)
        target_pitch = joystick_input[2] * np.radians(20)
        
        # Thrust (World Frame)
        thrust_scalar = throttle * self.max_thrust
        quat = self.get_orientation_quaternion()
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        thrust_vec = rot_matrix @ np.array([thrust_scalar, 0, 0])
        
        # Aerodynamics (Using Airspeed via _calculate_aerodynamics)
        vel_world = self.get_velocity()
        lift_vec, drag_vec = self._calculate_aerodynamics(vel_world, quat, 1.225)
        
        # Stabilization (PID)
        rpy = self.get_orientation_rpy()
        torque_x = self.pid_bank.update(target_bank - rpy[0], dt)
        torque_y = self.pid_pitch.update(target_pitch - rpy[1], dt)
        
        # Yaw Damping
        ang_vel = np.array(p.getBaseVelocity(self.drone_id)[1])
        torque_z = -2.0 * ang_vel[2] 
        
        # Apply Forces
        p.applyExternalForce(self.drone_id, -1, thrust_vec, [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalForce(self.drone_id, -1, lift_vec, [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalForce(self.drone_id, -1, drag_vec, [0, 0, 0], p.WORLD_FRAME)
        
        # Apply Torques
        p.applyExternalTorque(self.drone_id, -1, [torque_x, torque_y, torque_z], p.WORLD_FRAME)
        
        return thrust_vec 

    def apply_environmental_effects(self, atmospheric_conditions: dict):
        # Environment is already handled inside apply_control -> _calculate_aerodynamics
        pass

    def get_drone_type(self): return "Fixed-Wing"
    def get_flight_characteristics(self): return {'type': 'Fixed-Wing (Aero)', 'stall_speed': self.stall_speed}