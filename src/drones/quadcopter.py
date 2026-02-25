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
        self.kp_xy = 6.0   # Horizontal velocity gain
        self.kp_z = 25.0   # Vertical velocity gain
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
        p.changeDynamics(body_id, -1, linearDamping=0.5, angularDamping=0.5)
        # p.changeDynamics(body_id, -1, linearDamping=0.05, angularDamping=0.05)
        return body_id

    # def apply_control(self, inputs, dt):
    #     """
    #     Improved control with better physics and stability.
    #     """
    #     # --- Stage 1: Input Mapping ---
    #     u_roll, u_pitch, u_yaw, u_vert = inputs
        
    #     # IMPROVED CONTROL - better force scaling and stability
    #     # Reduced force scale for more gentle control
    #     force_scale = 1.5  # Reduced from 2.0 for better stability
        
    #     force_x = u_roll * force_scale  # Roll force (X axis)
    #     force_y = u_pitch * force_scale  # Pitch force (Y axis) - CORRECTED: removed negative sign
        
    #     # IMPROVED VERTICAL CONTROL with better hover stability
    #     gravity_comp = self.mass * 9.81
    #     # Add velocity damping for better hover stability
    #     current_vel = self.get_velocity()
    #     z_damping = -0.5 * current_vel[2]  # Vertical velocity damping
        
    #     # Hover point is at u_vert=0, with improved stability
    #     throttle_input = u_vert * force_scale * 0.5  # Reduced throttle sensitivity
    #     force_z = gravity_comp + throttle_input + z_damping
        
    #     # IMPROVED YAW CONTROL
    #     target_yaw_rate = u_yaw * self.max_yaw_rate
    #     ang_vel = np.array(p.getBaseVelocity(self.drone_id)[1])
    #     torque_z = self.kp_yaw * (target_yaw_rate - ang_vel[2])
        
    #     # MUCH STRONGER ATTITUDE STABILIZATION
    #     current_rpy = self.get_orientation_rpy()
    #     # Increased stabilization gains significantly
    #     torque_x = -8.0 * current_rpy[0] - 2.0 * ang_vel[0]  # Roll stabilization + damping
    #     torque_y = -8.0 * current_rpy[1] - 2.0 * ang_vel[1]  # Pitch stabilization + damping

    #     # --- Apply Forces ---
    #     current_pos = self.get_position()
        
    #     p.applyExternalForce(self.drone_id, -1, [force_x, force_y, force_z], current_pos, p.WORLD_FRAME)
    #     p.applyExternalTorque(self.drone_id, -1, [torque_x, torque_y, torque_z], p.WORLD_FRAME)
        
    #     return np.array([force_x, force_y, force_z])

    def apply_control(self, inputs, dt):
        """
        Implementation strictly following Thesis Section 2.2.2
        (High-Level Velocity Control Model).
        """
        u_roll, u_pitch, u_yaw, u_vert = inputs
        
        current_vel = self.get_velocity()
        current_pos = self.get_position()
        current_rpy = self.get_orientation_rpy()
        yaw = current_rpy[2]
        
        # --- Eq 2.12: Input Mapping to Target Body Velocity ---
        target_v_body_x = u_pitch * self.max_xy_velocity # Dopředu/Dozadu
        target_v_body_y = u_roll * self.max_xy_velocity  # Doleva/Doprava
        target_v_z = u_vert * self.max_z_velocity        # Nahoru/Dolu
        
        # --- Eq 2.13: Frame Transformation R_z(psi) ---
        # Převedení rychlosti z pohledu dronu do pohledu mapy (aby vždy letěl dopředu tam, kam se dívá)
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        target_v_world_x = target_v_body_x * cos_y - target_v_body_y * sin_y
        target_v_world_y = target_v_body_x * sin_y + target_v_body_y * cos_y
        
        # --- Eq 2.14: Proportional Force Control ---
        # Vytvoření síly na základě rozdílu mezi cílovou a skutečnou rychlostí
        force_x = self.mass * self.kp_xy * (target_v_world_x - current_vel[0])
        force_y = self.mass * self.kp_xy * (target_v_world_y - current_vel[1])
        
        # Osa Z obsahuje kompenzaci gravitace (g_comp = m * g)
        gravity_comp = self.mass * 9.81
        force_z = self.mass * self.kp_z * (target_v_z - current_vel[2]) + gravity_comp
        
        # --- Eq 2.15: Attitude Stabilization and Yaw Control ---
        target_yaw_rate = u_yaw * self.max_yaw_rate
        ang_vel = p.getBaseVelocity(self.drone_id)[1]
        
        torque_z = self.kp_yaw * (target_yaw_rate - ang_vel[2])
        # PD regulátor pro udržení roviny (Roll a Pitch se snaží být 0)
        torque_x = -self.kp_attitude * current_rpy[0] - 1.0 * ang_vel[0]
        torque_y = -self.kp_attitude * current_rpy[1] - 1.0 * ang_vel[1]

        # --- Aplikace sil ---
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