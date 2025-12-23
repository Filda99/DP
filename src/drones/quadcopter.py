"""
Quadcopter Drone Class - Joystick Control Model with Wind Bypass

Simplified physics model for MARL.
Includes a "Perfect Stability" zone for winds below 14.7 m/s (ANAFI USA specs).
The drone acts like a commercially stabilized platform (e.g., DJI/Parrot).
"""

import pybullet as p
import numpy as np
from .base_drone import BaseDrone

class Quadcopter(BaseDrone):
    def __init__(self, position=[0, 0, 5], mass=0.5, max_horizontal_force=None, max_vertical_force=None):
        """
        Initialize quadcopter with Joystick Control.
        """
        self.drone_id = self._create_pybullet_body(position, mass)
        super().__init__(self.drone_id, position, mass)
        
        # Flight Characteristics
        self.max_xy_velocity = 15.0  # m/s (Max speed)
        self.max_z_velocity = 4.0    # m/s (Max climb/descend)
        self.max_yaw_rate = 2.0      # rad/s
        
        # Wind Resistance Limit (ANAFI USA Spec)
        self.max_wind_resistance = 14.7 # m/s
        
        # Control Parameters (P-Controller is sufficient thanks to Wind Bypass)
        # Higher = Snappier, Lower = Smoother
        self.kp_xy = 2.0  
        self.kp_z = 4.0   
        self.kp_yaw = 5.0 

    def _create_pybullet_body(self, position, mass):
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.04])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.04], rgbaColor=[0.8, 0.2, 0.2, 1])
        
        drone_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        return drone_id

    def apply_control(self, joystick_input):
        """
        Apply high-level joystick control.
        Input: [Roll_Cmd, Pitch_Cmd, Yaw_Rate, Vertical_Cmd] (Ranges -1 to 1)
        """
        # Get current state
        lin_vel = self.get_velocity()
        ang_vel = np.array(p.getBaseVelocity(self.drone_id)[1])
        rpy = self.get_orientation_rpy()
        current_yaw = rpy[2]
        
        # 1. Map Inputs -> Target Velocities (Local Frame)
        target_vel_x_local = joystick_input[1] * self.max_xy_velocity # Pitch -> Forward
        target_vel_y_local = joystick_input[0] * self.max_xy_velocity # Roll -> Right
        target_vel_z = joystick_input[3] * self.max_z_velocity        # Vertical
        target_yaw_rate = joystick_input[2] * self.max_yaw_rate
        
        # 2. Transform to World Frame
        c, s = np.cos(current_yaw), np.sin(current_yaw)
        target_vel_x_world = target_vel_x_local * c - target_vel_y_local * s
        target_vel_y_world = target_vel_x_local * s + target_vel_y_local * c
        
        # 3. Calculate Forces (P-Controller)
        # Force = Mass * Gain * Error
        force_x = self.mass * self.kp_xy * (target_vel_x_world - lin_vel[0])
        force_y = self.mass * self.kp_xy * (target_vel_y_world - lin_vel[1])
        force_z = self.mass * self.kp_z  * (target_vel_z - lin_vel[2])
        
        # Add Gravity Compensation
        force_z += self.gravity_compensation
        
        # 4. Apply Forces
        total_force = [force_x, force_y, force_z]
        p.applyExternalForce(self.drone_id, -1, total_force, [0, 0, 0], p.WORLD_FRAME)
        
        # 5. Handle Yaw
        torque_z = self.kp_yaw * (target_yaw_rate - ang_vel[2])
        p.applyExternalTorque(self.drone_id, -1, [0, 0, torque_z], p.WORLD_FRAME)
        
        # 6. Visual Stabilization (Keep drone visually flat)
        # p.applyExternalTorque(self.drone_id, -1, [-2.0 * rpy[0] - 0.5*ang_vel[0], -2.0 * rpy[1] - 0.5*ang_vel[1], 0], p.WORLD_FRAME)

        return np.array(total_force)

    def apply_environmental_effects(self, atmospheric_conditions: dict):
        """
        Apply wind effects with Bypass Logic.
        """
        local_airflow = atmospheric_conditions['velocity']
        wind_speed = np.linalg.norm(local_airflow)
        
        # --- LOGIC: WIND BYPASS ---
        # If wind is within operational limits (14.7 m/s), assume perfect stabilization.
        # We apply NO external wind force, so the P-controller holds position perfectly.
        if wind_speed <= self.max_wind_resistance:
            return np.zeros(3)
            
        # --- LOGIC: EXTREME WIND ---
        # If wind exceeds limits, physics takes over and drone will drift.
        vel_world = self.get_velocity()
        relative_vel = vel_world - local_airflow
        
        # Calculate Drag Force
        # F = 0.5 * rho * v^2 * Cd * A
        drag_force = -0.5 * 1.225 * 0.1 * relative_vel * np.linalg.norm(relative_vel)
        
        p.applyExternalForce(self.drone_id, -1, drag_force.tolist(), [0, 0, 0], p.WORLD_FRAME)
        return drag_force

    def get_drone_type(self): return "Quadcopter"
    def can_hover(self): return True
    def get_max_speed(self): return self.max_xy_velocity
    def get_flight_characteristics(self):
        return {'type': "Quadcopter (Wind Bypass)", 'max_speed': self.max_xy_velocity}