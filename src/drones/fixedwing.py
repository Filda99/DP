"""
Fixed-Wing Drone Class

Aerodynamic implementation based on Lift/Drag coefficients and Angle of Attack.
Controls: Autopilot assisted (Bank Angle, Pitch Angle, Throttle).
"""

import pybullet as p
import numpy as np
from .base_drone import BaseDrone, PIDController

class FixedWing(BaseDrone):
    def __init__(self, position=[0, 0, 5], mass=1.0, max_thrust=30.0, min_speed=5.0, water_capacity=50.0, environment=None):
        self.drone_id = self._create_pybullet_body(position, mass)
        super().__init__(self.drone_id, position, mass, environment)
        
        self.water_capacity = water_capacity
        self.current_water = water_capacity
        self.max_thrust = max_thrust
        self.max_speed = 25.0
        self.stall_speed = 6.0
        
        self.wing_area = 0.8
        self.wing_span = 1.5
        self.aspect_ratio = (self.wing_span**2) / self.wing_area
        
        # Autopilot Controllers
        self.pid_bank = PIDController(kp=2.0, ki=0.05, kd=0.5, output_limit=10.0)
        self.pid_pitch = PIDController(kp=2.0, ki=0.05, kd=0.5, output_limit=10.0)
        self.pid_yaw = PIDController(kp=1.0, ki=0.0, kd=0.1, output_limit=5.0) # NEW: Rudder control
        
    def _create_pybullet_body(self, position, mass):
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.8, 0.05]) 
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.8, 0.05], rgbaColor=[0.2, 0.2, 0.9, 1])
        drone_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        # p.changeDynamics(drone_id, -1, linearDamping=0, angularDamping=0)
        return drone_id

    def _calculate_aerodynamics(self, velocity, orientation_quat, air_density):
        # 1. Wind
        wind_vec = np.zeros(3)
        if self.environment:
            wind_vec = self.environment.get_wind_at_position(self.get_position())
            
        # 2. Airspeed
        airspeed_vec_world = velocity - wind_vec
        airspeed = np.linalg.norm(airspeed_vec_world)
        
        if airspeed < 0.1: return np.zeros(3), np.zeros(3)

        # Body Transformation
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation_quat)).reshape(3, 3)
        airspeed_body = rot_matrix.T @ airspeed_vec_world
        
        if abs(airspeed_body[0]) < 0.1:
            return np.zeros(3), np.zeros(3)

        # Alpha
        alpha = np.arctan2(-airspeed_body[2], airspeed_body[0])
        
        # Coefficients
        stall_angle = np.radians(15)
        if abs(alpha) < stall_angle:
            CL = 2.0 * np.pi * alpha + 0.3
        else:
            CL = 0.8 * np.sign(alpha) * np.exp(-5 * (abs(alpha) - stall_angle))
        CD = 0.05 + (CL**2) / (np.pi * self.aspect_ratio * 0.8)
        
        # Forces
        q = 0.5 * air_density * (airspeed**2) * self.wing_area
        lift_mag = q * CL 
        drag_mag = q * CD
        
        F_lift_body = np.array([-lift_mag * np.sin(alpha), 0, lift_mag * np.cos(alpha)])
        F_drag_body = np.array([-drag_mag * np.cos(alpha), 0, -drag_mag * np.sin(alpha)])
        
        return rot_matrix @ F_lift_body, rot_matrix @ F_drag_body

    def apply_control(self, joystick_input):
        dt = 1/60.0
        
        # Inputs: [Bank, Throttle, Pitch, WATER_DROP]
        # Inputs: Bank (-1..1), Throttle (0..1), Pitch (-1..1)
        target_bank = -joystick_input[0] * np.radians(45) 
        throttle = np.clip(joystick_input[1], 0.0, 1.0)
        target_pitch = joystick_input[2] * np.radians(20)
        if len(joystick_input) > 3 and joystick_input[3] > 0.5:
            self.open_water_valve()
        else:
            self.close_water_valve()
        
        # Thrust
        thrust_scalar = throttle * self.max_thrust
        quat = self.get_orientation_quaternion()
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        thrust_vec = rot_matrix @ np.array([thrust_scalar, 0, 0])
        
        # Aerodynamics
        vel_world = self.get_velocity()
        lift_vec, drag_vec = self._calculate_aerodynamics(vel_world, quat, 1.225)
        
        # Stabilization
        rpy = self.get_orientation_rpy()
        torque_x = self.pid_bank.update(target_bank - rpy[0], dt)
        torque_y = self.pid_pitch.update(target_pitch - rpy[1], dt)
        
        # COORDINATED TURN LOGIC (Turn Nose into Bank)
        speed = np.linalg.norm(vel_world)
        if speed > 2.0:
            # Ideal Yaw Rate = (g / V) * tan(BankAngle)
            target_yaw_rate = (9.81 / speed) * np.tan(rpy[0])
        else:
            target_yaw_rate = 0.0
            
        ang_vel = np.array(p.getBaseVelocity(self.drone_id)[1])
        torque_z = self.pid_yaw.update(target_yaw_rate - ang_vel[2], dt)
        
        # Apply Forces
        p.applyExternalForce(self.drone_id, -1, thrust_vec, [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalForce(self.drone_id, -1, lift_vec, [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalForce(self.drone_id, -1, drag_vec, [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1, [torque_x, torque_y, torque_z], p.WORLD_FRAME)
        
        return thrust_vec 

    def apply_environmental_effects(self, atmospheric_conditions: dict):
        pass

    def get_drone_type(self): return "Fixed-Wing"
    def get_flight_characteristics(self): return {'type': 'Fixed-Wing (Aero)', 'stall_speed': self.stall_speed}