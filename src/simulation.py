"""
Simulation Manager

Manages the complete simulation with multiple drones, environment, and physics.
"""
import pybullet as p
import pybullet_data
import numpy as np
import json
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter

# Clean imports assuming standard project structure
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
    
    def __init__(self, log_file=None):
        self.gui = False
        self.physics_client = None
        self.drones = {}
        self.destroyed_drones = []
        
        # Physics Constants
        self.fps = 60
        self.timestep = 1.0 / self.fps
        
        # Atmospheric Constants
        self.AIRFLOW_H_LIMIT = 50.0
        self.BASE_TEMP = 293.15
        
        self.environment = Environment()
        self.simulation_time = 0.0
        self.temperature_grid = None 
        
        self.visualizer = SimulationVisualizer()
        
        # Data Logging
        self.log_entries = []
        self.simulation_log = {
            'drones': {},
            'environment_effects': [],
            'collisions': [],
            'destroyed_drones': [],
            'fire_states': [],
            'times': []
        }
        self.drone_trajectories = {}
        self._step_counter = 0
        
        self._setup_logging(log_file)
    
    def _setup_logging(self, log_file):
        os.makedirs('logs', exist_ok=True)
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'logs/simulation_{timestamp}.json'
        self.log_file = log_file
        print(f"📝 Logging to: {self.log_file}")
        
    def start_simulation(self, gui=False):
        """Start PyBullet simulation."""
        self.gui = gui
        connection_mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.environment.create_ground()
        
        print(f"✅ Simulation started (GUI={gui})")
        
    def stop_simulation(self):
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
        self._save_log()
        print("✅ Simulation stopped")
        
    def add_quadcopter(self, name, position=[0, 0, 5], mass=0.5):
        quad = Quadcopter(position, mass)
        self._register_drone(name, quad, 'quadcopter')
        print(f"✅ Added quadcopter '{name}'")
        
    def add_fixedwing(self, name, position=[0, 0, 5], mass=1.0, max_thrust=20.0, water_capacity=0.0):
        fw = FixedWing(position, mass, max_thrust, water_capacity=water_capacity)
        self._register_drone(name, fw, 'fixedwing')
        print(f"✅ Added fixed-wing '{name}'")
        return fw
        
    def _register_drone(self, name, drone, type_str):
        self.drones[name] = drone
        self.simulation_log['drones'][name] = {
            'type': type_str,
            'positions': [], 'forces': [], 'velocities': [], 'control_inputs': []
        }
        self.drone_trajectories[name] = [list(drone.get_position()) + [0.0]]

    # --- ENVIRONMENT WRAPPERS ---
    def setup_osm_environment(self, location, **kwargs):
        load_environment_from_osm(self.environment, location, **kwargs)
        
    def enable_fire_simulation(self, **kwargs):
        dt = kwargs.pop('dt', self.timestep)
        self.environment.enable_fire_simulation(dt=dt, **kwargs)
        # Init temp grid (Layers, H, W)
        if self.environment.fire_grid:
            shape = (20, self.environment.fire_grid.H, self.environment.fire_grid.W)
            self.temperature_grid = np.full(shape, self.BASE_TEMP)

    def start_fire(self, position, intensity=0.2):
        """Wrapper to start fire in environment."""
        return self.environment.start_fire_at_position(position, intensity)

    def set_wind(self, velocity):
        """Set wind velocity vector [x, y, z]."""
        # Update both current and target to prevent immediate interpolation override
        vel = np.array(velocity, dtype=float)
        self.environment.weather['wind_velocity'] = vel
        self.environment.target_wind = vel
        self.environment.wind_velocity = vel
        print(f"✅ Wind set to {velocity} m/s")

    def set_weather(self, visibility=1000.0, precipitation=0.0):
        """Set weather parameters."""
        self.environment.weather['visibility'] = visibility
        self.environment.weather['precipitation'] = precipitation

    # --- PHYSICS STEP ---
    def step_simulation(self, drone_controls):
        """Main loop step."""
        
        # 1. Update Drones
        for name, drone in self.drones.items():
            if name not in drone_controls: continue
            
            # Atmospheric effects
            atmos = self.get_local_atmospheric_conditions(drone.get_position())
            drone.apply_environmental_effects(atmos)
            
            # Control
            forces = drone.apply_control(drone_controls[name])
            
            # Logging
            if self._step_counter % 2 == 0: # Log every 2nd step to save space
                self._log_drone_state(name, drone, forces, drone_controls[name])

        # 2. Physics Step
        p.stepSimulation()
        self.simulation_time += self.timestep
        self._step_counter += 1
        self.simulation_log['times'].append(self.simulation_time)

        # 3. Environment (Fire/Water)
        water_drops = self._calculate_water_drops()
        self.environment.update_fire_simulation(water_drops=water_drops, real_dt=self.timestep)
        
        # 4. Updates
        if self.environment.fire_enabled and self._step_counter % 10 == 0:
            self._update_temperature_grid()
            
        # 5. Collision Check
        self._check_collisions()
    
    def get_all_trajectories(self, flatten_2d=False):
        """Helper for demos to get trajectories."""
        trajs = {}
        for name, points in self.drone_trajectories.items():
            arr = np.array(points)
            if flatten_2d and len(arr) > 0:
                trajs[name] = arr[:, :2]
            else:
                trajs[name] = arr
        return trajs

    def get_local_atmospheric_conditions(self, pos):
        """Simplified atmospheric query."""
        # Default
        cond = {
            'velocity': self.environment.weather['wind_velocity'].copy(),
            'temperature': self.BASE_TEMP,
            'density': 1.225
        }
        
        # If fire is active and we are low enough
        if self.environment.fire_enabled and pos[2] < self.AIRFLOW_H_LIMIT:
            # Here we could query the temp grid
            # For simplicity in this optimized version, we return base + wind
            pass 
            
        return cond

    def _calculate_water_drops(self):
        """Calculates water dropped by drones."""
        drops = {}
        dt = self.timestep
        cell_size = self.environment.grid_mapper.cell_size_m if self.environment.grid_mapper else 2.0
        
        for name, drone in self.drones.items():
            if not drone.can_drop_water(): continue
            
            # Rate: 200L/s for large tank, scaled down by dt
            drop_rate = 200.0 * dt
            amount = drone.consume_water(drop_rate)
            
            if amount <= 0: continue
            
            # Map to grid
            pos = drone.get_position()
            if self.environment.grid_mapper:
                try:
                    i, j = self.environment.grid_mapper.world_to_cell((pos[0], pos[1]))
                    # Simple 1-cell drop for speed (gaussian blur removed for optimization)
                    drops[(i, j)] = drops.get((i, j), 0.0) + (amount * 0.1) 
                except:
                    pass
        return drops

    def _update_temperature_grid(self):
        """Updates 3D temperature grid based on fire intensity."""
        if self.temperature_grid is None or not self.environment.fire_grid: return
        
        # Heat rises
        # Bottom layer = Base + Fire Intensity
        intensities = self.environment.fire_grid.I
        self.temperature_grid[0] = self.BASE_TEMP + intensities * 500.0
        
        # Vertical diffusion (simple decay)
        for i in range(1, len(self.temperature_grid)):
            self.temperature_grid[i] = self.BASE_TEMP + (self.temperature_grid[0] - self.BASE_TEMP) * (0.8 ** i)

    def _check_collisions(self):
        """Checks collisions."""
        to_destroy = []
        for name, drone in self.drones.items():
            pos = drone.get_position()
            
            # Ground
            if pos[2] < 0.1:
                print(f"💥 {name} crashed into ground.")
                to_destroy.append(name)
                continue
                
            # Environment
            hit, obs = self.environment.is_position_in_obstacle(pos)
            if hit:
                print(f"💥 {name} hit {obs.get('type', 'obstacle')}.")
                to_destroy.append(name)

        for name in to_destroy:
            self._destroy_drone(name)

    def _destroy_drone(self, name):
        if name in self.drones:
            # PyBullet remove
            try:
                p.removeBody(self.drones[name].drone_id)
            except:
                pass
            del self.drones[name]
            self.destroyed_drones.append(name)

    def _log_drone_state(self, name, drone, forces, inputs):
        # Update trajectory
        pos = drone.get_position()
        self.drone_trajectories[name].append(list(pos) + [self.simulation_time])
        
        # Log details
        self.simulation_log['drones'][name]['positions'].append(pos)
        self.simulation_log['drones'][name]['velocities'].append(drone.get_velocity())
        self.simulation_log['drones'][name]['forces'].append(forces)
        self.simulation_log['drones'][name]['control_inputs'].append(inputs)

    def _save_log(self):
        with open(self.log_file, 'w') as f:
            # Convert numpy arrays to lists for JSON
            json.dump(self.simulation_log, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        print(f"📝 Log saved.")