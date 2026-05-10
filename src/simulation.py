"""
Simulation Manager

Manages the complete simulation with multiple drones, environment, and physics.
"""
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter

try:
    from .environment import Environment
    from .drones import Quadcopter, FixedWing
    from ..tools.visualizer import SimulationVisualizer
    from .map_importer import load_environment_from_osm
except ImportError:
    from src.environment import Environment
    from src.drones import Quadcopter, FixedWing
    from tools.visualizer import SimulationVisualizer
    from src.map_importer import load_environment_from_osm

class Simulation:
    """Complete simulation manager."""
    
    # ============================================================================
    # INITIALIZATION & LIFECYCLE
    # ============================================================================
    
    def __init__(self, log_file=None):
        """Initialize simulation."""
        self.gui = False
        self.physics_client = None
        self.drones = {}
        self.destroyed_drones = []
        self.environment = Environment()
        self.simulation_time = 0.0
        self.fps = 30
        self.timestep = 1/self.fps
        
        # --- Airflow model variables ---
        self.airflow_H = 100.0
        self.convection_gain = 8.0
        self.plume_radius_factor = 2.0
        self.radial_flow_factor = 0.3
        
        # --- Temperature grid ---
        self.temperature_grid = None
        self.base_temperature = 293.15
        
        # Initialize visualizer
        self.visualizer = SimulationVisualizer()
        
        # Simulation data logging
        self.simulation_log = {
            'drones': {},
            'environment_effects': [],
            'collisions': [],
            'destroyed_drones': [],
            'fire_states': [],
            'water_levels': [],
            'events': [],
            'times': []
        }
        
        # Trajectory tracking
        self.drone_trajectories = {}
        self.trajectory_sample_rate = 3
        self._step_counter = 0

        # Per-step effective water drop stats (for reward computation)
        self.drone_extinguish_stats = {}
        
        # Setup file logging
        # self._setup_logging(log_file)
    
    def _setup_logging(self, log_file):
        """Setup file logging for simulation events."""
        os.makedirs('logs', exist_ok=True)
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'logs/simulation_{timestamp}.json'
        
        self.log_file = log_file
        self.log_entries = []
        # print(f"📝 Logging to: {self.log_file}")
        
    def start_simulation(self):
        """Start PyBullet simulation."""
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        self.environment.create_ground()
        self.environment.create_refill_zone()
        
    def stop_simulation(self):
        """Stop PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
    
    def _log_event(self, event_type, data):
        """Log an event to the log file."""
        log_entry = {
            'time': self.simulation_time,
            'event': event_type,
            'data': data
        }
        self.log_entries.append(log_entry)
        self.simulation_log['events'].append(log_entry)
    
    def _save_log(self):
        """Save all log entries to file."""
        log_data = {
            'metadata': {
                'timestep': self.timestep,
                'total_time': self.simulation_time,
                'drones': list(self.drones.keys()),
                'destroyed_drones': self.destroyed_drones
            },
            'events': self.log_entries,
            'simulation_log': self.simulation_log
        }
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        # print(f"📝 Log saved to: {self.log_file}")
    
    # ============================================================================
    # DRONE MANAGEMENT
    # ============================================================================
    
    def add_quadcopter(self, name, position=[0, 0, 5], mass=0.5):
        """Add a quadcopter drone to the simulation."""
        quad = Quadcopter(position, mass)
        self.drones[name] = quad
        self.simulation_log['drones'][name] = {
            'type': 'quadcopter',
            'positions': [], 'forces': [], 'velocities': [], 'control_inputs': []
        }
        self.drone_trajectories[name] = [[position[0], position[1], position[2], 0.0]]
        self.drone_extinguish_stats[name] = 0.0
        return quad
    
    def add_fixedwing(self, name, position=[0, 0, 5], mass=1.0, water_capacity=0.0, yaw=0.0):
        """Add a fixed-wing drone to the simulation."""
        fw = FixedWing(position=position, mass=mass, water_capacity=water_capacity, environment=self.environment, initial_chi=yaw)
        self.drones[name] = fw
        self.simulation_log['drones'][name] = {
            'type': 'fixedwing',
            'positions': [], 'forces': [], 'velocities': [], 'control_inputs': [], 'water_levels': []
        }
        self.drone_trajectories[name] = [[position[0], position[1], position[2], 0.0]]
        self.drone_extinguish_stats[name] = 0.0
        return fw
    
    def get_drone_status(self, drone_name):
        """Return position, velocity and characteristics for a single drone."""
        if drone_name not in self.drones: return None
        drone = self.drones[drone_name]
        return {
            'name': drone_name, 'type': drone.get_drone_type(),
            'position': drone.get_position(), 'velocity': drone.get_velocity(),
            'speed': drone.get_speed(), 'characteristics': drone.get_flight_characteristics()
        }
    
    def get_all_drone_status(self):
        """Return status of every active drone."""
        return {name: self.get_drone_status(name) for name in self.drones.keys()}
    
    # ============================================================================
    # ENVIRONMENT SETUP
    # ============================================================================
    
    def setup_osm_environment(self, location_query: str, default_building_height: float = 10.0,
                            distance_m: float = 2000):
        """Load environment geometry from OpenStreetMap."""
        load_environment_from_osm(self.environment, location_query, default_building_height, distance_m)
    
    def set_wind(self, wind_velocity):
        """Manually override wind velocity [vx, vy, vz] in m/s."""
        self.environment.set_wind(wind_velocity)
    
    def set_weather(self, visibility=1000.0, precipitation=0.0):
        """Set weather conditions (visibility in metres, precipitation 0–1)."""
        self.environment.set_weather(visibility, precipitation)
    
    # ============================================================================
    # FIRE SIMULATION
    # ============================================================================
    
    def enable_fire_simulation(self, grid_width_m=100, grid_height_m=100, cell_size_m=2.0, dt=None):
        """Enable wildfire simulation and initialise the temperature grid."""
        if dt is None: dt = self.timestep
        self.environment.enable_fire_simulation(grid_width_m, grid_height_m, cell_size_m, dt=dt)
        
        if hasattr(self.environment, 'fire_grid') and self.environment.fire_grid is not None:
            H, W = self.environment.fire_grid.H, self.environment.fire_grid.W
            height_levels = 20
            self.temperature_grid = np.full((height_levels, H, W), self.base_temperature, dtype=np.float32)
    
    def start_fire(self, world_pos, intensity=0.2, radius_m=5.0):
        """Ignite fire at *world_pos* with the given intensity and radius."""
        return self.environment.start_fire_at_position(world_pos, intensity, radius_m=radius_m)
    
    def _update_temperature_grid(self):
        if self.temperature_grid is None: return
        height_levels, H, W = self.temperature_grid.shape
        new_temp = self.temperature_grid.copy()
        
        fire_intensities = self.environment.fire_grid.I
        new_temp[0, :, :] = self.base_temperature + fire_intensities * 500.0
        
        for layer in range(1, height_levels):
            decay = 0.85 ** layer
            new_temp[layer, :, :] = self.base_temperature + (new_temp[0, :, :] - self.base_temperature) * decay
        
        new_temp = np.clip(new_temp, self.base_temperature, self.base_temperature + 1000.0)
        self.temperature_grid = new_temp
    
    def _calculate_water_drops(self):
        """
        OPTIMIZED VERSION: Calculates water dispersion only at impact sites.
        """
        # Reset stats for this step
        for name in self.drone_extinguish_stats:
            self.drone_extinguish_stats[name] = 0.0

        # 1. Quick check if grid exists
        if self.environment.fire_grid is None or self.environment.grid_mapper is None:
            return {}

        water_drops = {}
        dt = self.timestep
        mapper = self.environment.grid_mapper
        
        # Grid dimensions for boundary checks
        H, W = self.environment.fire_grid.H, self.environment.fire_grid.W

        # Direct access to burning state (for reward computation)
        fire_active = self.environment.fire_grid.B

        for drone_name, drone in self.drones.items():
            # If drone is not dropping water, skip (saves CPU)
            if not drone.can_drop_water(): continue

            # 2. Get Position and Water Amount
            pos = drone.get_position()
            altitude = pos[2]
            
            # Water effectiveness decreases with height
            effectiveness = max(0.0, 1.0 - (altitude / 200.0))
            # 240 RL steps until the water is completely off
            water_amount = drone.consume_water(100.0 * dt) * effectiveness
            
            if water_amount <= 0: continue

            # 3. Where did it hit? (Center cell)
            try:
                center_r, center_c = mapper.world_to_cell((pos[0], pos[1]))
            except:
                continue # Out of bounds

            # 4. Dispersion Parameters
            effective_radius_m = 10.0 + 0.3 * altitude
            sigma_cells = effective_radius_m / mapper.cell_size_m / 2.5
            
            # Optimization: Calculate only within 3*sigma radius (captures 99% of water)
            influence_radius = int(sigma_cells * 3) + 1
            
            # 5. Iterate only over a small Bounding Box
            # Instead of 160,000 cells, we check only ~100 cells
            r_min = max(0, center_r - influence_radius)
            r_max = min(H, center_r + influence_radius + 1)
            c_min = max(0, center_c - influence_radius)
            c_max = min(W, center_c + influence_radius + 1)

            two_sigma_sq = 2 * sigma_cells**2

            # Effective water on burning cells (for reward tracking)
            drone_effective_drop = 0.0

            for r in range(r_min, r_max):
                for c in range(c_min, c_max):
                    # Squared distance from center
                    dist_sq = (r - center_r)**2 + (c - center_c)**2
                    
                    # Gaussian function: exp(-x^2 / 2sigma^2)
                    weight = np.exp(-dist_sq / two_sigma_sq)
                    
                    # Ignore negligible contributions
                    if weight < 0.01: continue
                    
                    dropped_value = water_amount * weight * 10.0 # Scaling factor
                    
                    # 1. Store in dictionary (accumulate if multiple drones hit same spot)
                    coord = (r, c)
                    if coord in water_drops:
                        water_drops[coord] += dropped_value
                    else:
                        water_drops[coord] = dropped_value

                    # Count reward if cell is burning
                    if fire_active[r, c]:
                        drone_effective_drop += dropped_value

            # Store stats for reward
            self.drone_extinguish_stats[drone_name] = drone_effective_drop

        # 6. Clip to max 1.0 (Saturation)
        for k in water_drops:
            water_drops[k] = min(1.0, water_drops[k])

        return water_drops
    
    # ============================================================================
    # ATMOSPHERIC PHYSICS
    # ============================================================================
    
    def get_local_atmospheric_conditions(self, world_pos: np.ndarray) -> dict:
        """
        Returns local atmospheric conditions at a given world position.

        The local airflow is computed as the superposition of:
        (1) global ambient wind
        (2) fire-induced buoyant convection generated by nearby burning cells

        Fire-induced effects are modeled using a simplified, parameterized
        plume formulation intended to capture first-order aerodynamic effects.
        """

        # ------------------------------------------------------------------
        # 1. Ambient atmospheric conditions (baseline)
        # ------------------------------------------------------------------
        velocity = self.environment.weather['wind_velocity'].copy()
        temperature = self.base_temperature
        density = 1.225  # Reference air density at sea level (kg/m^3)

        gm = self.environment.grid_mapper
        fg = self.environment.fire_grid

        # If no spatial mapping exists, fire influence cannot be computed
        if gm is None:
            return {'velocity': velocity, 'temperature': temperature, 'density': density}

        # ------------------------------------------------------------------
        # 2. Temperature and density adjustment from temperature field
        # ------------------------------------------------------------------
        # Uses a discretized vertical temperature grid, if available,
        # to model buoyancy-related density changes.
        if self.temperature_grid is not None:
            # Map horizontal position to grid indices
            i, j = gm.world_to_cell(world_pos[:2])

            # Convert continuous height to vertical layer index
            z_idx = int(np.clip(
                world_pos[2] / self.airflow_H * self.temperature_grid.shape[0],
                0, self.temperature_grid.shape[0] - 1
            ))

            # Update local temperature and density if within grid bounds
            if 0 <= i < self.temperature_grid.shape[1] and 0 <= j < self.temperature_grid.shape[2]:
                temperature = self.temperature_grid[z_idx, i, j]

                # Ideal gas approximation: warmer air is less dense
                density *= 293.15 / temperature  # Reference temperature = 20°C

        # Above the modeled fire-atmosphere coupling height,
        # airflow is assumed to be unaffected by the fire
        if world_pos[2] >= self.airflow_H or fg is None:
            return {'velocity': velocity, 'temperature': temperature, 'density': density}

        # ------------------------------------------------------------------
        # 3. Fire-induced convection (buoyant plume model)
        # ------------------------------------------------------------------
        # Compute fire influence only from nearby burning cells
        ci, cj = gm.world_to_cell(world_pos[:2])
        plume_radius = gm.cell_size_m * self.plume_radius_factor

        # Normalize height to [0, 1] within the fire influence layer
        z_norm = world_pos[2] / self.airflow_H

        # Accumulator for fire-induced airflow
        convection = np.zeros(3)

        # Height-dependent plume strength:
        # weak near ground, strongest in lower atmosphere, decays aloft
        height_taper = (
            z_norm / 0.3 if z_norm < 0.3 else
            (1.0 - z_norm) / 0.7
        )

        # Iterate over local neighborhood of fire cells
        for di in range(-5, 5):
            for dj in range(-5, 5):
                i, j = ci + di, cj + dj

                # Skip cells outside the fire grid
                if not (0 <= i < fg.H and 0 <= j < fg.W):
                    continue

                # Skip non-burning cells
                fire_intensity = fg.I[i, j]
                if fire_intensity <= 0:
                    continue

                # ----------------------------------------------------------
                # Geometry: distance from fire cell center to query point
                # ----------------------------------------------------------
                fx = gm.origin_x + (j + 0.5) * gm.cell_size_m
                fy = gm.origin_y + (i + 0.5) * gm.cell_size_m
                dx, dy = world_pos[0] - fx, world_pos[1] - fy
                r = np.hypot(dx, dy)

                # ----------------------------------------------------------
                # Vertical buoyant updraft (Gaussian plume)
                # ----------------------------------------------------------
                # Fire intensity controls plume strength,
                # Gaussian decay controls spatial influence
                w = (
                    fire_intensity *
                    self.convection_gain *
                    height_taper *
                    np.exp(-0.5 * (r / plume_radius) ** 2)
                )

                # Accumulate vertical convection from all nearby fires
                convection[2] += w

                # ----------------------------------------------------------
                # Radial inflow / outflow (mass continuity)
                # ----------------------------------------------------------
                # Air flows inward near the ground and outward aloft
                if r < 1e-6:
                    continue

                # Unit vector pointing from fire cell to query point
                ux, uy = dx / r, dy / r

                # Strength of radial flow depends on height
                radial = (
                    -w if z_norm < 0.5 else w
                ) * self.radial_flow_factor * abs(z_norm - 0.5) / 0.5

                # Accumulate horizontal airflow contribution
                convection[0] += ux * radial
                convection[1] += uy * radial

        # ------------------------------------------------------------------
        # 4. Superposition of ambient wind and fire-induced airflow
        # ------------------------------------------------------------------
        velocity += convection

        return {'velocity': velocity, 'temperature': temperature, 'density': density}



    # ============================================================================
    # SIMULATION STEPPING & COLLISION DETECTION
    # ============================================================================

    def step_simulation(self, drone_controls):
        """Step the simulation forward."""

        zone = self.environment.refill_zone

        # Apply controls to each drone
        for drone_name, control_input in drone_controls.items():
            if drone_name in self.drones:
                drone = self.drones[drone_name]

                # 0. Check for refill zone (only for FixedWing with water tank)
                if zone and drone.water_capacity > 0: # FixedWing drones have water tanks
                    d_pos = drone.get_position()
                    z_pos = zone['position']
                    
                    # 2D (XY) distance only — zone is at z=0 but FW flies at 40–150 m,
                    # so including the z-component would make the radius impossible to hit.
                    dist_sq = (d_pos[0]-z_pos[0])**2 + (d_pos[1]-z_pos[1])**2
                    
                    if dist_sq < zone['radius_sq']:
                        drone.refill_tank()
                
                # 1. Atmospheric effects
                atmos = self.get_local_atmospheric_conditions(drone.get_position())
                drone.apply_environmental_effects(atmos)

                # 2. Control
                forces = drone.apply_control(control_input, self.timestep)
                
                # 3. Log data

        # Step physics
        p.stepSimulation()
        self.simulation_time += self.timestep
        
        # Update trajectories (downsampled)
        self._step_counter += 1
        if self._step_counter % self.trajectory_sample_rate == 0:
            for drone_name, drone in self.drones.items():
                pos = drone.get_position().copy()
                self.drone_trajectories[drone_name].append([pos[0], pos[1], pos[2], self.simulation_time])
        
        # Environment updates
        water_drops = self._calculate_water_drops()
        self.environment.update_fire_simulation(water_drops=water_drops, real_dt=self.timestep)
        
        self._check_collisions()
    
    def _check_collisions(self):
        """Detect and handle obstacle/ground collisions for all drones."""
        drones_to_destroy = []
        for drone_name, drone in self.drones.items():
            position = drone.get_position()
            
            collision, obstacle = self.environment.is_position_in_obstacle(position)
            if collision:
                drones_to_destroy.append(drone_name)
                continue
            
            if position[2] < 0.5:
                drones_to_destroy.append(drone_name)
                continue
                
        for drone_name in set(drones_to_destroy):
            self._destroy_drone(drone_name)
    
    def _destroy_drone(self, drone_name):
        """Remove a drone from the simulation."""
        if drone_name not in self.drones: return
        drone = self.drones[drone_name]
        try: p.removeBody(drone.drone_id)
        except: pass
        self.destroyed_drones.append(drone_name)
        del self.drones[drone_name]
        
        if drone_name in self.drone_extinguish_stats:
             del self.drone_extinguish_stats[drone_name]
    
    def run_scenario(self, scenario_function, steps=1000):
        """Run a scripted scenario for *steps* simulation steps."""
        for step in range(steps):
            controls = scenario_function(step, self.simulation_time, self.drones)
            self.step_simulation(controls)
            if step % 100 == 0: print(f"  Step {step}/{steps} - Time: {self.simulation_time:.2f}s")
    
    # ============================================================================
    # VISUALIZATION & ANALYSIS
    # ============================================================================
    
    def create_multi_drone_visualization(self, title="Multi-Drone Analysis"):
        """Create visualization showing all drones together."""
        if self.visualizer:
            self.visualizer.create_multi_drone_visualization(
                self.simulation_log, 
                self.drones, 
                self.environment, 
                title
            )
        else:
            print("Visualizer not initialized")

    def create_visualization(self, drone_name=None, title="Simulation Analysis"):
        """Create comprehensive visualization for a single drone."""
        if self.visualizer:
            self.visualizer.create_single_drone_visualization(
                self.simulation_log,
                self.drones,
                self.environment,
                drone_name,
                title
            )
        else:
            print("Visualizer not initialized")
    
    # ============================================================================
    # TRAJECTORY ACCESS METHODS
    # ============================================================================
    
    def get_drone_trajectory(self, drone_name, flatten_2d=False):
        """Return recorded trajectory as a NumPy array."""
        if drone_name not in self.drone_trajectories: return np.array([])
        traj = np.array(self.drone_trajectories[drone_name])
        return traj[:, :2] if flatten_2d and len(traj) > 0 else traj
    
    def get_all_trajectories(self, flatten_2d=False):
        """Return trajectories of all drones."""
        return {name: self.get_drone_trajectory(name, flatten_2d) for name in self.drone_trajectories.keys()}
    
    def clear_trajectories(self):
        """Reset trajectory buffers for all active drones."""
        for name in self.drone_trajectories:
            if name in self.drones:
                pos = self.drones[name].get_position().copy()
                self.drone_trajectories[name] = [[pos[0], pos[1], pos[2], self.simulation_time]]
    
    def get_simulation_summary(self):
        """Return a summary dict with timing, drone status and fire stats."""
        summary = {
            'total_time': self.simulation_time,
            'total_steps': len(self.simulation_log['times']),
            'active_drones': list(self.drones.keys()),
            'destroyed_drones': self.destroyed_drones,
            'environment': self.environment.get_environment_info(),
            'collisions': len(self.simulation_log['collisions'])
        }
        fire_state = self.environment.get_fire_state()
        if fire_state: summary['fire'] = fire_state['fire_stats']
        return summary