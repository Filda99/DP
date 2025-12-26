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
    from .visualizer import SimulationVisualizer
    from .map_importer import load_environment_from_osm
except ImportError:
    from src.environment import Environment
    from src.drones import Quadcopter, FixedWing
    from src.visualizer import SimulationVisualizer
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
        self.fps = 60
        self.timestep = 1/60.0
        
        # --- Airflow model variables ---
        self.airflow_H = 50.0
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
        
        # Setup file logging
        self._setup_logging(log_file)
    
    def _setup_logging(self, log_file):
        """Setup file logging for simulation events."""
        os.makedirs('logs', exist_ok=True)
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'logs/simulation_{timestamp}.json'
        
        self.log_file = log_file
        self.log_entries = []
        print(f"📝 Logging to: {self.log_file}")
        
    def start_simulation(self):
        """Start PyBullet simulation."""
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)  # 1/60s
        self.environment.create_ground()
        print(f"✅ Simulation started")
        self._log_event('simulation_start ', {'timestep': self.timestep})
        
    def stop_simulation(self):
        """Stop PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
        self._save_log()
        print("✅ Simulation stopped")
    
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
        print(f"📝 Log saved to: {self.log_file}")
    
    # ============================================================================
    # DRONE MANAGEMENT
    # ============================================================================
    
    def add_quadcopter(self, name, position=[0, 0, 5], mass=0.5):
        quad = Quadcopter(position, mass)
        self.drones[name] = quad
        self.simulation_log['drones'][name] = {
            'type': 'quadcopter',
            'positions': [], 'forces': [], 'velocities': [], 'control_inputs': []
        }
        self.drone_trajectories[name] = [[position[0], position[1], position[2], 0.0]]
        print(f"✅ Added quadcopter '{name}' at {position}")
    
    def add_fixedwing(self, name, position=[0, 0, 5], mass=1.0, water_capacity=0.0):
        fw = FixedWing(position=position, mass=mass, water_capacity=water_capacity, environment=self.environment)
        self.drones[name] = fw
        self.simulation_log['drones'][name] = {
            'type': 'fixedwing',
            'positions': [], 'forces': [], 'velocities': [], 'control_inputs': [], 'water_levels': []
        }
        self.drone_trajectories[name] = [[position[0], position[1], position[2], 0.0]]
        start_orientation = p.getQuaternionFromEuler([0, 0.2, 0.0])
        # p.resetBasePositionAndOrientation(fw.drone_id, position, start_orientation)

        # p.changeDynamics(fw.drone_id, -1, linearDamping=0.0, angularDamping=0.0)

        print(f"✅ Added fixed-wing '{name}' at {position} (water: {water_capacity}L)")
        return fw
    
    def get_drone_status(self, drone_name):
        if drone_name not in self.drones: return None
        drone = self.drones[drone_name]
        return {
            'name': drone_name, 'type': drone.get_drone_type(),
            'position': drone.get_position(), 'velocity': drone.get_velocity(),
            'speed': drone.get_speed(), 'characteristics': drone.get_flight_characteristics()
        }
    
    def get_all_drone_status(self):
        return {name: self.get_drone_status(name) for name in self.drones.keys()}
    
    # ============================================================================
    # ENVIRONMENT SETUP
    # ============================================================================
    
    def setup_osm_environment(self, location_query: str, default_building_height: float = 10.0,
                            distance_m: float = 2000):
        print(f"🌍 Loading environment from OSM: {location_query}")
        load_environment_from_osm(self.environment, location_query, default_building_height, distance_m)
    
    def set_wind(self, wind_velocity):
        self.environment.set_wind(wind_velocity)
        print(f"✅ Wind set to {wind_velocity} m/s")
    
    def set_weather(self, visibility=1000.0, precipitation=0.0):
        self.environment.set_weather(visibility, precipitation)
        print(f"✅ Weather set - visibility: {visibility}m, precipitation: {precipitation}")
    
    # ============================================================================
    # FIRE SIMULATION
    # ============================================================================
    
    def enable_fire_simulation(self, grid_width_m=100, grid_height_m=100, cell_size_m=2.0, dt=None):
        if dt is None: dt = self.timestep
        self.environment.enable_fire_simulation(grid_width_m, grid_height_m, cell_size_m, dt=dt)
        
        if hasattr(self.environment, 'fire_grid') and self.environment.fire_grid is not None:
            H, W = self.environment.fire_grid.H, self.environment.fire_grid.W
            height_levels = 20
            self.temperature_grid = np.full((height_levels, H, W), self.base_temperature, dtype=float)
            print(f"✅ Temperature grid initialized: {height_levels}×{H}×{W}")
        print(f"✅ Fire simulation enabled in environment")
    
    def start_fire(self, world_pos, intensity=0.2):
        return self.environment.start_fire_at_position(world_pos, intensity)
    
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
        # FIX: Check if grid exists
        if self.environment.fire_grid is None or self.environment.grid_mapper is None:
            return {}
            
        H, W = self.environment.fire_grid.H, self.environment.fire_grid.W
        water_grid = np.zeros((H, W), dtype=float)
        dt = self.timestep
        cell_size = self.environment.grid_mapper.cell_size_m
        max_sigma = 0.0
        
        for drone in self.drones.values():
            if not drone.can_drop_water(): continue
            pos = drone.get_position()
            altitude = pos[2]
            
            effectiveness = 1.0 - (altitude / 50.0)
            water = drone.consume_water(200.0 * dt) * effectiveness
            if water <= 0: continue
            
            try:
                i, j = self.environment.grid_mapper.world_to_cell((pos[0], pos[1]))
                if 0 <= i < H and 0 <= j < W:
                    water_grid[i, j] += water
                    effective_radius = 10.0 + 0.3 * altitude
                    sigma = effective_radius / cell_size / 2.5
                    max_sigma = max(max_sigma, sigma)
            except: continue
        
        if max_sigma > 0:
            water_grid = gaussian_filter(water_grid, sigma=max_sigma, mode='constant')
        
        water_drops = {}
        nonzero = np.argwhere(water_grid > 1e-6)
        for i, j in nonzero:
            scaled_water = float(water_grid[i, j]) * 10.0
            water_drops[(int(i), int(j))] = min(1.0, scaled_water)
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
        for di in range(-2, 3):
            for dj in range(-2, 3):
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
        # Apply controls to each drone
        for drone_name, control_input in drone_controls.items():
            if drone_name in self.drones:
                drone = self.drones[drone_name]
                
                # 1. Atmospheric effects
                atmos = self.get_local_atmospheric_conditions(drone.get_position())
                drone.apply_environmental_effects(atmos)

                # 2. Control
                forces = drone.apply_control(control_input, self.timestep)
                
                # 3. Log data (Every step - FIXED to match time array length)
                self.simulation_log['drones'][drone_name]['positions'].append(drone.get_position().copy())
                self.simulation_log['drones'][drone_name]['forces'].append(forces.copy())
                self.simulation_log['drones'][drone_name]['velocities'].append(drone.get_velocity().copy())
                self.simulation_log['drones'][drone_name]['control_inputs'].append(control_input.copy())
                
                # Check if drone has water attribute (FixedWing does, Quadcopter does not)
                if drone.can_drop_water():
                    self.simulation_log['drones'][drone_name]['water_levels'].append(drone.current_water)
                # else:
                #     self.simulation_log['drones'][drone_name]['water_levels'].append(0.0)

        # Step physics
        p.stepSimulation()
        self.simulation_time += self.timestep
        self.simulation_log['times'].append(self.simulation_time)
        
        # Update trajectories (downsampled)
        self._step_counter += 1
        if self._step_counter % self.trajectory_sample_rate == 0:
            for drone_name, drone in self.drones.items():
                pos = drone.get_position().copy()
                self.drone_trajectories[drone_name].append([pos[0], pos[1], pos[2], self.simulation_time])
        
        # Environment updates
        water_drops = self._calculate_water_drops()
        self.environment.update_fire_simulation(water_drops=water_drops, real_dt=self.timestep)
        self._update_temperature_grid()
        
        if len(self.simulation_log['times']) % 10 == 0:
            self.environment.visualize_fire_in_simulation()
        
        fire_state = self.environment.get_fire_state()
        self.simulation_log['fire_states'].append(fire_state)
        
        self._check_collisions()
    
    def _check_collisions(self):
        drones_to_destroy = []
        for drone_name, drone in self.drones.items():
            position = drone.get_position()
            
            collision, obstacle = self.environment.is_position_in_obstacle(position)
            if collision:
                print(f"💥 COLLISION: {drone_name} hit {obstacle['type']}")
                self._log_event('collision', {'drone': drone_name, 'obstacle': obstacle['type']})
                drones_to_destroy.append(drone_name)
                continue
            
            if position[2] < 0.5:
                print(f"💥 GROUND CRASH: {drone_name}")
                self._log_event('ground_crash', {'drone': drone_name})
                drones_to_destroy.append(drone_name)
                continue
                
        for drone_name in set(drones_to_destroy):
            self._destroy_drone(drone_name)
    
    def _destroy_drone(self, drone_name):
        if drone_name not in self.drones: return
        drone = self.drones[drone_name]
        try: p.removeBody(drone.drone_id)
        except: pass
        self.destroyed_drones.append(drone_name)
        self._log_event('drone_destroyed', {'drone': drone_name})
        del self.drones[drone_name]
        print(f"🔥 DESTROYED: {drone_name}")
    
    def run_scenario(self, scenario_function, steps=1000):
        print(f"🚁 Running scenario for {steps} steps...")
        for step in range(steps):
            controls = scenario_function(step, self.simulation_time, self.drones)
            self.step_simulation(controls)
            if step % 100 == 0: print(f"  Step {step}/{steps} - Time: {self.simulation_time:.2f}s")
        print(f"✅ Scenario completed")
    
    # ============================================================================
    # VISUALIZATION & ANALYSIS
    # ============================================================================
    
    def create_multi_drone_visualization(self, title="Multi-Drone Analysis"):
        """Create visualization showing all drones together."""
        print("📊 Generating multi-drone visualization...")
        if self.visualizer:
            self.visualizer.create_multi_drone_visualization(
                self.simulation_log, 
                self.drones, 
                self.environment, 
                title
            )
            print("✅ Visualization complete")
        else:
            print("❌ Visualizer not initialized")

    def create_visualization(self, drone_name=None, title="Simulation Analysis"):
        """Create comprehensive visualization for a single drone."""
        print("📊 Generating visualization...")
        if self.visualizer:
            self.visualizer.create_single_drone_visualization(
                self.simulation_log,
                self.drones,
                self.environment,
                drone_name,
                title
            )
            print("✅ Visualization complete")
        else:
            print("❌ Visualizer not initialized")
    
    # ============================================================================
    # TRAJECTORY ACCESS METHODS
    # ============================================================================
    
    def get_drone_trajectory(self, drone_name, flatten_2d=False):
        if drone_name not in self.drone_trajectories: return np.array([])
        traj = np.array(self.drone_trajectories[drone_name])
        return traj[:, :2] if flatten_2d and len(traj) > 0 else traj
    
    def get_all_trajectories(self, flatten_2d=False):
        return {name: self.get_drone_trajectory(name, flatten_2d) for name in self.drone_trajectories.keys()}
    
    def clear_trajectories(self):
        for name in self.drone_trajectories:
            if name in self.drones:
                pos = self.drones[name].get_position().copy()
                self.drone_trajectories[name] = [[pos[0], pos[1], pos[2], self.simulation_time]]
    
    def get_simulation_summary(self):
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