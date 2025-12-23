"""
Fixed-Wing Drone Class - Kinematic Guidance Model
Implemented based on MathWorks UAV Guidance Model equations.
"""

import pybullet as p
import numpy as np
import math
from .base_drone import BaseDrone

class FixedWing(BaseDrone):
    def __init__(self, position=[0, 0, 5], mass=1.0, water_capacity=50.0, environment=None):
        # Initialize PyBullet Body
        self.drone_id = self._create_pybullet_body(position, mass)
        super().__init__(self.drone_id, position, mass, environment)
        
        self.water_capacity = water_capacity
        self.current_water = water_capacity
        
        # --- Guidance Model Parameters (Gains) ---
        self.kp_h = 2.0       # Proportional gain for height error
        self.kp_gamma = 5.0   # Proportional gain for flight path angle
        self.kp_va = 2.0      # Proportional gain for airspeed
        self.kp_phi = 5.0     # Proportional gain for roll angle
        self.kd_phi = 2.0     # Derivative gain for roll rate
        
        # --- Internal State Initialization ---
        # State vector: [x, y, h, Va, chi, gamma, phi, phi_dot]
        # x, y, h: Inertial Position
        # Va: Airspeed
        # chi: Course angle (heading)
        # gamma: Flight path angle (climb angle)
        # phi: Roll angle
        self.state_pos = np.array(position, dtype=float)  # x, y, h
        self.state_va = 15.0  # Initial airspeed (m/s)
        self.state_chi = 0.0  # Initial course (radians)
        self.state_gamma = 0.0
        self.state_phi = 0.0
        self.state_phi_dot = 0.0
        
        # --- Control Targets ---
        self.target_h = position[2]
        self.target_va = 15.0
        self.target_phi = 0.0
        
        # Limits
        self.max_speed = 30.0
        self.min_speed = 10.0
        self.max_roll = np.radians(45)
        
        # Configure PyBullet dynamics to be kinematic (we control motion manually)
        p.changeDynamics(self.drone_id, -1, linearDamping=0, angularDamping=0)

    def _create_pybullet_body(self, position, mass):
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.1, 0.05])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.1, 0.05], rgbaColor=[0.8, 0.2, 0.2, 1])
        # Add wings for visual reference
        p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.02], rgbaColor=[0.8, 0.2, 0.2, 1], visualFramePosition=[0, 0, 0])
        
        uid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=position)
        return uid

    def apply_control(self, inputs, dt):
        """
        Updates the drone state using the Guidance Model differential equations.
        
        Inputs: [Roll, Throttle, Pitch, Water_Trigger]
        - Roll (-1..1): Sets Commanded Roll (phi_c)
        - Throttle (0..1): Sets Commanded Airspeed (Va_c)
        - Pitch (-1..1): Adjusts Commanded Altitude (h_c)
        """

        # --- 1. Parse Inputs ---
        
        # 1. Map Inputs to Control Commands (c superscripts in equations)
        roll_input = inputs[0]
        throttle_input = inputs[1]
        pitch_input = inputs[2]
        water_input = inputs[3] if len(inputs) > 3 else 0

        # Command: Roll (Phi_c)
        self.target_phi = np.clip(roll_input, -1.0, 1.0) * self.max_roll # Max roll angle on both sides

        # Command: Flight Path Angle (Direct Pitch)
        # Instead of changing altitude, we set the climb angle directly.
        # Positive input = Climb (+45 deg), Negative = Dive (-45 deg)
        max_pitch = np.radians(45)
        gamma_c = np.clip(pitch_input, -1.0, 1.0) * max_pitch

        # Command: Airspeed (Va_c)
        # If throttle_input = 0 -> min_speed, throttle_input = 1 -> max_speed (that's why we need to
        # subtract min from max, because we add min in every case)
        # self.target_va = self.min_speed + (throttle_input * (self.max_speed - self.min_speed))
        self.target_va = throttle_input * self.max_speed

        # Command: Altitude (h_c) - Pitch adjusts target height rate
        # self.target_h += pitch_input * 0.1 # Rate of changing target altitude
        
        # Water Mechanism
        self.open_water_valve() if water_input > 0.5 else self.close_water_valve()

        # --- 2. Guidance Model Equations ---
        
        # Constants
        g = 9.81
        
        # Get Current State
        x, y, h = self.state_pos
        va = self.state_va
        chi = self.state_chi
        gamma = self.state_gamma
        phi = self.state_phi
        phi_dot = self.state_phi_dot

        # STALL LOGIC:
        # If we are too slow, we lose lift and gravity takes over.
        stall_speed = 8.0  # m/s
        is_stalled = va < stall_speed

        if is_stalled:
            # If stalled, we ignore the commanded gamma_c.
            # The nose drops (-30 degrees) and we accelerate downwards.
            gamma_c = np.radians(-30) 
            # Damping the roll control when stalled (harder to turn)
            self.target_phi *= 0.2

        # --- 3. Update State Derivatives (MathWorks Equations Modified) ---
        
        # Update Airspeed (with gravity assist if diving/stalled)
        # If stalled, gravity helps us accelerate (dive)
        gravity_assist = -g * np.sin(gamma) if is_stalled else 0
        va_dot = self.kp_va * (self.target_va - va) + gravity_assist

        # Update Gamma (Flight Path Angle)
        gamma_dot = self.kp_gamma * (gamma_c - gamma)
        
        # Update Roll
        phi_ddot = self.kp_phi * (self.target_phi - phi) + self.kd_phi * (-phi_dot)

        # --- 4. Wind Triangle & Ground Speed ---

        # Update Course (Heading) - standard coordinated turn
        # We need Ground Speed (Vg) for this.
        # Simplified wind handling for clarity:
        if self.environment:
             v_wind = self.environment.weather.get('wind_velocity', np.zeros(3))
        else:
             v_wind = np.zeros(3)
             
        # 3.1. Calculate Direction Vector of the Drone (Unit Vector based on chi/gamma)
        # This represents the direction the drone is traveling
        dir_x = np.cos(chi) * np.cos(gamma)
        dir_y = np.sin(chi) * np.cos(gamma)
        dir_z = np.sin(gamma)
        
        # 3.2. Project Wind onto Flight Path
        # Dot product: How much is the wind helping (+) or hurting (-) our speed?
        wind_along_path = (v_wind[0] * dir_x) + (v_wind[1] * dir_y) + (v_wind[2] * dir_z)
        
        # 3.3. Update Ground Speed (Vg)
        # Vg is Airspeed (engines) + Wind Component
        vg = va + wind_along_path

        # --- 5. Turn Rate (Chi_dot) with Drift Correction ---

        # d(chi)/dt (Course Rate / Coordinated Turn)
        # Equation: chi_dot = (g * cos(chi - psi) * tan(phi)) / Vg
        if vg > 1.0 and not is_stalled:
            # 1. Calculate the drift angle correction term: cos(chi - psi)
            # We use the cross-track wind component to find sin(chi - psi)
            w_x, w_y = v_wind[0], v_wind[1]
            
            # Cross-wind component perpendicular to the course
            cross_wind = -w_x * np.sin(chi) + w_y * np.cos(chi)
            
            # Drift Angle (Wind Triangle): Va * sin(drift) = CrossWind
            sin_drift = np.clip(cross_wind / max(va, 0.1), -1.0, 1.0)
            cos_drift = np.sqrt(1.0 - sin_drift**2)
            
            # 2. Apply equation
            chi_dot = (g * cos_drift * np.tan(phi)) / vg
        else:
            chi_dot = 0.0

        # --- 4. Integration ---
        x_dot = vg * np.cos(chi) * np.cos(gamma)
        y_dot = vg * np.sin(chi) * np.cos(gamma)
        h_dot = vg * np.sin(gamma)

        self.state_pos[0] += x_dot * dt
        self.state_pos[1] += y_dot * dt
        self.state_pos[2] += h_dot * dt
        
        self.state_va += va_dot * dt
        self.state_chi += chi_dot * dt
        self.state_gamma += gamma_dot * dt
        self.state_phi_dot += phi_ddot * dt
        self.state_phi += self.state_phi_dot * dt

        # --- 5. PyBullet Update ---
        new_quat = p.getQuaternionFromEuler([self.state_phi, self.state_gamma, self.state_chi])
        p.resetBasePositionAndOrientation(self.drone_id, self.state_pos, new_quat)
        p.resetBaseVelocity(self.drone_id, linearVelocity=[x_dot, y_dot, h_dot], angularVelocity=[0, 0, chi_dot])

        return np.array([x_dot, y_dot, h_dot])

        # # --- 3. Calculate Ground Speed (Vg) considering Wind ---

        # # 3.1. Get Wind Vector from Environment
        # if self.environment:
        #     # Get wind (vx, vy, vz)
        #     v_wind = self.environment.weather.get('wind_velocity', np.zeros(3))
        # else:
        #     v_wind = np.zeros(3)

        # # 3.2. Calculate Direction Vector of the Drone (Unit Vector based on chi/gamma)
        # # This represents the direction the drone is traveling
        # dir_x = np.cos(chi) * np.cos(gamma)
        # dir_y = np.sin(chi) * np.cos(gamma)
        # dir_z = np.sin(gamma)
        
        # # 3.3. Project Wind onto Flight Path
        # # Dot product: How much is the wind helping (+) or hurting (-) our speed?
        # wind_along_path = (v_wind[0] * dir_x) + (v_wind[1] * dir_y) + (v_wind[2] * dir_z)
        
        # # 3.4. Update Ground Speed (Vg)
        # # Vg is Airspeed (engines) + Wind Component
        # vg = va + wind_along_path
        
        # # --- 4. Calculate Commanded Flight Path Angle (gamma_c) ---

        # # Equation: Vg * sin(gamma_c) = min(max(kp_h * (h_c - h), -Vg), Vg)
        # h_err = self.target_h - h
        # climb_term = np.clip(self.kp_h * h_err, -vg, vg)
        
        # # Avoid division by zero
        # if vg > 0.1:
        #     sin_gamma_c = climb_term / vg
        #     # Clamp for numerical stability inside arcsin
        #     sin_gamma_c = np.clip(sin_gamma_c, -1.0, 1.0) 
        #     gamma_c = np.arcsin(sin_gamma_c)
        # else:
        #     gamma_c = 0.0

        # # --- 5. Calculate Derivatives ---
        
        # # d(gamma)/dt = kp_gamma * (gamma_c - gamma)
        # gamma_dot = self.kp_gamma * (gamma_c - gamma)
        
        # # d(Va)/dt = kp_va * (Va_c - Va)
        # va_dot = self.kp_va * (self.target_va - va)
        
        # # d(phi)/dt (Roll Dynamics - 2nd Order)
        # # phi_ddot = kp_phi * (phi_c - phi) + kd_phi * (-phi_dot)
        # phi_ddot = self.kp_phi * (self.target_phi - phi) + self.kd_phi * (-phi_dot)
        
        # # d(chi)/dt (Course Rate / Coordinated Turn)
        # # Equation: chi_dot = (g * cos(chi - psi) * tan(phi)) / Vg
        # if vg > 1.0:
        #     # 1. Calculate the drift angle correction term: cos(chi - psi)
        #     # We use the cross-track wind component to find sin(chi - psi)
        #     w_x, w_y = v_wind[0], v_wind[1]
            
        #     # Cross-wind component perpendicular to the course
        #     cross_wind = -w_x * np.sin(chi) + w_y * np.cos(chi)
            
        #     # sin(chi - psi) = Cross_Wind / Va
        #     sin_drift = np.clip(cross_wind / va, -1.0, 1.0)
            
        #     # cos(chi - psi) = sqrt(1 - sin^2(chi - psi))
        #     cos_drift = np.sqrt(1.0 - sin_drift**2)
            
        #     # 2. Apply equation
        #     chi_dot = (g * cos_drift * np.tan(phi)) / vg
        # else:
        #     chi_dot = 0.0
            
        # # Kinematics (Position Derivatives)
        # # x_dot = Vg * cos(chi) * cos(gamma)
        # # y_dot = Vg * sin(chi) * cos(gamma)
        # # h_dot = Vg * sin(gamma)
        # x_dot = vg * np.cos(chi) * np.cos(gamma)
        # y_dot = vg * np.sin(chi) * np.cos(gamma)
        # h_dot = vg * np.sin(gamma)

        # # Integration (Euler)
        # self.state_pos[0] += x_dot * dt
        # self.state_pos[1] += y_dot * dt
        # self.state_pos[2] += h_dot * dt
        
        # self.state_va += va_dot * dt
        # self.state_chi += chi_dot * dt
        # self.state_gamma += gamma_dot * dt
        
        # self.state_phi_dot += phi_ddot * dt
        # self.state_phi += self.state_phi_dot * dt

        # # --- 6. Update PyBullet Visuals/Physics State ---

        # # We explicitly force the position/orientation because this is a Guidance Model, not a Rigid Body Force simulation.
        
        # # Orientation: Convert (Roll, Pitch=Gamma, Yaw=Chi) to Quaternion
        # # Note: Gamma is Flight Path Angle, which roughly approximates Pitch in steady level flight 
        # new_quat = p.getQuaternionFromEuler([self.state_phi, self.state_gamma, self.state_chi])
        
        # p.resetBasePositionAndOrientation(self.drone_id, self.state_pos, new_quat)
        
        # # Set velocity for other sensors/visuals (trails) to work
        # p.resetBaseVelocity(self.drone_id, linearVelocity=[x_dot, y_dot, h_dot], angularVelocity=[0, 0, chi_dot])

        # return np.array([x_dot, y_dot, h_dot])

    def apply_environmental_effects(self, atmospheric_conditions: dict):
        # The guidance model assumes intrinsic wind handling in Vg calculation.
        # For this implementation, we only use wind to update effective ground speed if needed.
        pass

    def get_flight_characteristics(self):
        return {
            'type': "Fixed-Wing (Guidance Model)",
            'control_mode': "Kinematic", 
            'max_speed': self.max_speed
        }

    def get_drone_type(self):
        return "Fixed-Wing"