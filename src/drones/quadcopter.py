"""
Quadcopter Drone Class - High-Level Velocity Control Model

Simplified physics model for MARL.
Includes a "Perfect Stability" zone for winds below 14.7 m/s (ANAFI USA specs).
The drone acts like a commercially stabilized platform (e.g., DJI/Parrot).

Implements a high-level velocity controller (similar to DJI/commercial drones):
- Pitch Input -> Target Velocity Forward (Body Frame)
- Roll Input  -> Target Velocity Right (Body Frame)
- Yaw Input   -> Target Yaw Rate
- Throttle    -> Target Vertical Velocity

Includes logic to rotate commands based on current heading so controls
always feel relative to the drone's front.

Implementation based on:
1. Input Mapping -> Target Velocity (Body Frame)
2. Frame Transformation (Body -> World) using Rotation Matrix R_z(psi)
3. Proportional Force Control (P-Controller) + Gravity Compensation
4. Discrete Environmental Stability (Wind Bypass)

As defined in Thesis Section 2.2.2.
"""

import pybullet as p
import numpy as np
from .base_drone import BaseDrone

class Quadcopter(BaseDrone):
    def __init__(self, position=[0, 0, 5], mass=0.5, max_horizontal_force=None, max_vertical_force=None):
        """
        Initialize quadcopter with High-Level Velocity Control.
        """
        self.drone_id = self._create_pybullet_body(position, mass)
        super().__init__(self.drone_id, position, mass)
        
        # --- 1. Model Parameters (from Thesis) ---
        self.max_xy_velocity = 15.0  # V_xy_max [m/s]
        self.max_z_velocity = 4.0    # V_z_max [m/s]
        self.max_yaw_rate = 2.5      # rad/s
        
        self.max_wind_resistance = 14.7 # V_max_stable [m/s]
        
        # --- 2. Controller Gains (K_p matrix) ---
        self.kp_xy = 2.0   # Horizontal velocity gain
        self.kp_z = 10.0   # Vertical velocity gain
        self.kp_yaw = 1.0  # Yaw rate gain
        self.kp_attitude = 5.0 # Torque to keep level

    def _create_pybullet_body(self, position, mass):
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[0, 1, 1, 1])
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        p.changeDynamics(body_id, -1, linearDamping=0.05, angularDamping=0.05)
        return body_id

    def apply_control(self, inputs, dt):
        """
        Applies control forces based on High-Level Velocity Control Model.
        """
        # --- Stage 1: Input Mapping (v_local_cmd) ---
        u_roll, u_pitch, u_yaw, u_vert = inputs
        
        v_local_x = u_pitch * self.max_xy_velocity  # Forward
        v_local_y = u_roll * self.max_xy_velocity   # Right
        v_local_z = u_vert * self.max_z_velocity    # Up
        target_yaw_rate = u_yaw * self.max_yaw_rate

        # --- Stage 2: Frame Transformation (v_world_cmd) ---
        current_rpy = self.get_orientation_rpy()
        psi = current_rpy[2]
        c, s = np.cos(psi), np.sin(psi)
        
        v_world_x = (v_local_x * c) - (v_local_y * s)
        v_world_y = (v_local_x * s) + (v_local_y * c)
        v_world_z = v_local_z
        
        # --- Stage 3: Proportional Force Control (F_cmd) ---
        vel_current = self.get_velocity()
        
        force_x = self.mass * self.kp_xy * (v_world_x - vel_current[0])
        force_y = self.mass * self.kp_xy * (v_world_y - vel_current[1])
        
        gravity_comp = self.mass * 9.81
        force_z = (self.mass * self.kp_z * (v_world_z - vel_current[2])) + gravity_comp

        # --- Attitude & Yaw Control ---
        ang_vel = np.array(p.getBaseVelocity(self.drone_id)[1])
        torque_z = self.kp_yaw * (target_yaw_rate - ang_vel[2])
        
        torque_x = -self.kp_attitude * current_rpy[0] - 1.0 * ang_vel[0]
        torque_y = -self.kp_attitude * current_rpy[1] - 1.0 * ang_vel[1]

        # --- Apply Forces (CORRECTED) ---
        # Get current position to apply force at Center of Mass
        current_pos = self.get_position()
        
        p.applyExternalForce(self.drone_id, -1, [force_x, force_y, force_z], current_pos, p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1, [torque_x, torque_y, torque_z], p.WORLD_FRAME)
        
        return np.array([force_x, force_y, force_z])

    def apply_environmental_effects(self, atmospheric_conditions: dict):
        """Discrete Environmental Stability (Wind Bypass)"""
        wind_vec = atmospheric_conditions.get('velocity', np.zeros(3))
        wind_speed = np.linalg.norm(wind_vec)
        
        if wind_speed <= self.max_wind_resistance:
            return np.zeros(3)
            
        drone_vel = self.get_velocity()
        v_rel = drone_vel - wind_vec
        v_rel_mag = np.linalg.norm(v_rel)
        
        if v_rel_mag < 0.1: return np.zeros(3)
        
        rho = 1.225
        Cd = 0.8
        A = 0.05
        
        drag_mag = 0.5 * rho * Cd * A * (v_rel_mag**2)
        drag_force = - (v_rel / v_rel_mag) * drag_mag
        
        # CORRECTED: Apply at current position
        current_pos = self.get_position()
        p.applyExternalForce(self.drone_id, -1, drag_force.tolist(), current_pos, p.WORLD_FRAME)
        
        return drag_force

    def get_drone_type(self): return "Quadcopter"
    def can_hover(self): return True
    def get_max_speed(self): return self.max_xy_velocity
    def get_flight_characteristics(self):
        return {'type': "Quadcopter (Wind Bypass)", 'max_speed': self.max_xy_velocity}