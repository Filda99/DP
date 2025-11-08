"""
Simulation Manager

Manages the complete simulation with multiple drones, environment, and physics.
"""

import pybullet as p
import pybullet_data
import numpy as np

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
    
    def __init__(self, gui=False):
        """Initialize simulation."""
        self.gui = gui
        self.physics_client = None
        self.drones = {}
        self.environment = Environment()
        self.simulation_time = 0.0
        self.fps = 60  # Simulation FPS
        self.timestep = 1/60.0  # 60 FPS (reduced from 240 for performance)
        # --- Airflow model variables (corrected for physical accuracy) ---
        self.airflow_H = 50.0  # Convection height limit (m)
        self.convection_gain = 8.0  # Base updraft velocity [m/s] for unit fire intensity (realistic: 5-20 m/s)
        self.plume_radius_factor = 2.0  # Plume extends ~2x cell size
        self.radial_flow_factor = 0.3  # Radial flow is ~30% of vertical velocity
        
        # --- Temperature grid for heat diffusion ---
        self.temperature_grid = None  # Will be initialized when fire is enabled
        self.base_temperature = 293.15  # 20°C in Kelvin
        
        # Initialize visualizer
        self.visualizer = SimulationVisualizer()
        
        # Simulation data
        self.simulation_log = {
            'drones': {},
            'environment_effects': [],
            'collisions': [],
            'fire_states': [],  # Add fire state logging
            'times': []
        }
        
    def start_simulation(self):
        """Start PyBullet simulation."""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Create ground
        self.environment.create_ground()
        
        print(f"✅ Simulation started ({'GUI' if self.gui else 'headless'} mode)")
        
    def stop_simulation(self):
        """Stop PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
        print("✅ Simulation stopped")
    
    def add_quadcopter(self, name, position=[0, 0, 5], mass=0.5):
        """Add a quadcopter to the simulation."""
        quad = Quadcopter(position, mass)
        self.drones[name] = quad
        
        # Initialize logging for this drone
        self.simulation_log['drones'][name] = {
            'type': 'quadcopter',
            'positions': [],
            'forces': [],
            'velocities': [],
            'control_inputs': []
        }
        
        print(f"✅ Added quadcopter '{name}' at {position}")
    
    def add_fixedwing(self, name, position=[0, 0, 5], mass=1.0, max_thrust=20.0, water_capacity=0.0):
        """Add a fixed-wing drone to the simulation.
        
        Args:
            name: Drone identifier
            position: Initial position [x, y, z]
            mass: Aircraft mass in kg
            max_thrust: Maximum thrust in Newtons
            water_capacity: Water tank capacity in liters (0 = no firefighting)
        """
        fw = FixedWing(position, mass, max_thrust, water_capacity=water_capacity)
        self.drones[name] = fw
        
        # Initialize logging for this drone
        self.simulation_log['drones'][name] = {
            'type': 'fixedwing',
            'positions': [],
            'forces': [],
            'velocities': [],
            'control_inputs': []
        }
        
        print(f"✅ Added fixed-wing '{name}' at {position} (water: {water_capacity}L)")
        return fw
    
    def setup_osm_environment(self, location_query: str, default_building_height: float = 10.0,
                            distance_m: float = 2000, use_city_boundaries: bool = True):
        """
        Setup environment from OpenStreetMap data.
        
        Args:
            location_query: Location to load (e.g., "Prague, Czech Republic")
            default_building_height: Default height for buildings without height data (meters)
            distance_m: Radius in meters to download around the location (default: 2000m = 2km)
            use_city_boundaries: If True, use city/residential boundaries instead of individual buildings (MUCH faster!)
        """
        print(f"🌍 Loading environment from OSM: {location_query}")
        load_environment_from_osm(self.environment, location_query, default_building_height, distance_m, use_city_boundaries)
    
    def enable_fire_simulation(self, grid_width_m=100, grid_height_m=100, cell_size_m=2.0, dt=None):
        """Enable wildfire simulation in the environment."""
        # Use simulation timestep if dt not specified
        if dt is None:
            dt = self.timestep
        
        self.environment.enable_fire_simulation(grid_width_m, grid_height_m, cell_size_m, dt=dt)
        
        # Initialize temperature grid (3D: height × rows × cols)
        if hasattr(self.environment, 'fire_grid') and self.environment.fire_grid is not None:
            H, W = self.environment.fire_grid.H, self.environment.fire_grid.W
            height_levels = 20  # 20 vertical layers
            self.temperature_grid = np.full((height_levels, H, W), self.base_temperature, dtype=float)
            print(f"✅ Temperature grid initialized: {height_levels}×{H}×{W}")
        
        print(f"✅ Fire simulation enabled in environment")
    
    def start_fire(self, world_pos, intensity=0.2):
        """Start a fire at a world position."""
        return self.environment.start_fire_at_position(world_pos, intensity)
    
    def set_wind(self, wind_velocity, turbulence=0.0):
        """Set wind conditions."""
        self.environment.set_wind(wind_velocity, turbulence)
        print(f"✅ Wind set to {wind_velocity} m/s, turbulence: {turbulence}")
    
    def set_weather(self, visibility=1000.0, precipitation=0.0):
        """Set weather conditions."""
        self.environment.set_weather(visibility, precipitation)
        print(f"✅ Weather set - visibility: {visibility}m, precipitation: {precipitation}")
    
    def _update_temperature_grid(self):
        """
        Update temperature grid based on fire intensity and heat diffusion.
        
        Physics:
        - Bottom layer heats up from burning cells: T = base_temp + intensity × 500K
        - Heat diffuses to neighbors (simple 3D diffusion)
        - Heat rises (convection approximated by vertical diffusion bias)
        """
        if self.temperature_grid is None:
            return
        
        height_levels, H, W = self.temperature_grid.shape
        new_temp = self.temperature_grid.copy()
        
        # 1. Heat bottom layer from fire
        fire_intensities = self.environment.fire_grid.I  # (H, W)
        new_temp[0, :, :] = self.base_temperature + fire_intensities * 500.0
        
        # 2. Simplified heat diffusion - only vertical (heat rises)
        # Much faster than full 3D diffusion
        for layer in range(1, height_levels):
            # Heat from layer below rises up with decay
            decay = 0.85 ** layer  # Exponential decay with height
            new_temp[layer, :, :] = self.base_temperature + (new_temp[0, :, :] - self.base_temperature) * decay
        
        # Clamp to realistic range
        new_temp = np.clip(new_temp, self.base_temperature, self.base_temperature + 1000.0)
        
        self.temperature_grid = new_temp
    
    def step_simulation(self, drone_controls):
        """
        Step the simulation forward.
        
        Args:
            drone_controls: Dict with drone names as keys and joystick inputs as values
                          e.g., {'drone1': [0.5, 0.0, 0.1], 'drone2': [-0.3, 1.0, 0.0]}
        """
        # Apply controls to each drone
        for drone_name, control_input in drone_controls.items():
            if drone_name in self.drones:
                drone = self.drones[drone_name]
                
                # 1. Get local atmospheric conditions (Fire -> Temperature -> Density + Airflow)
                atmospheric_conditions = self.get_local_atmospheric_conditions(drone.get_position())
                
                # 2. Apply environmental effects (Atmospheric conditions -> Aircraft)
                drone.apply_environmental_effects(atmospheric_conditions)

                # Apply drone control
                forces = drone.apply_control(control_input)
                
                # Apply environmental effects
                position = drone.get_position()
                wind = self.environment.get_wind_at_position(position)
                wind_forces = drone.apply_wind_effect(wind)
                
                # Log data
                self.simulation_log['drones'][drone_name]['positions'].append(position.copy())
                self.simulation_log['drones'][drone_name]['forces'].append(forces.copy())
                self.simulation_log['drones'][drone_name]['velocities'].append(drone.get_velocity().copy())
                self.simulation_log['drones'][drone_name]['control_inputs'].append(control_input.copy())
        
        # Step physics
        p.stepSimulation()
        self.simulation_time += self.timestep
        self.simulation_log['times'].append(self.simulation_time)
        
        # Calculate water drops from drones (if any)
        water_drops = self._calculate_water_drops()
        
        # Update fire simulation with water drops (pass real timestep)
        self.environment.update_fire_simulation(water_drops=water_drops, real_dt=self.timestep)
        
        # Update temperature grid from fire (for physics demos)
        self._update_temperature_grid()
        
        # Visualize less frequently
        if len(self.simulation_log['times']) % 10 == 0:
            self.environment.visualize_fire_in_simulation()
        
        # Log fire state
        fire_state = self.environment.get_fire_state()
        self.simulation_log['fire_states'].append(fire_state)
        
        # Check for collisions with environment
        self._check_collisions()
    
    def _calculate_water_drops(self):
        """
        Calculate water drops from drone positions.
        Only drones with firefighting capability (water_capacity > 0) and open valve can drop water.
        Water consumption is limited by tank capacity.
        
        Returns:
            Dict mapping (i, j) cell coordinates to total water amount (0.0 to 1.0+)
        """
        water_drops = {}
        
        # Realistic water drop model:
        # - Only drones with water tanks (water_capacity > 0) can drop water
        # - Valve must be open (water_valve_open = True)
        # - Drones must be low altitude to drop water effectively (< 50m)
        # - Water distributed over area, total consumption per step
        drop_radius = 15.0  # meters - effective drop radius (WIDER coverage for aerial firefighting)
        water_consumption_rate = 200.0  # liters per second when valve open (REALISTIC for aerial firefighting)
        max_altitude = 50.0  # meters - maximum altitude for effective drops (HIGHER ceiling to accommodate climb)
        
        for drone_name, drone in self.drones.items():
            # Check if drone has firefighting capability
            if not drone.can_drop_water():
                continue  # Skip drones without water tank, closed valve, or empty tank
            
            drone_pos = drone.get_position()
            
            # Altitude check - drones too high cannot drop water effectively
            if drone_pos[2] > max_altitude:
                continue  # Skip this drone, too high
            
            # Calculate water consumption for this step
            dt = 1.0 / self.fps
            water_to_drop = water_consumption_rate * dt  # liters consumed this frame
            
            # Try to consume water from tank
            actual_water = drone.consume_water(water_to_drop)
            if actual_water <= 0:
                continue  # No water left
            
            # Altitude factor - effectiveness decreases with altitude
            altitude_factor = 1.0 - (drone_pos[2] / max_altitude)  # 1.0 at ground, 0.0 at max_altitude
            
            # Check if drone is close to any cells
            if self.environment.grid_mapper.is_position_in_bounds((drone_pos[0], drone_pos[1])):
                # Get nearby cells within drop radius
                center_i, center_j = self.environment.grid_mapper.world_to_cell((drone_pos[0], drone_pos[1]))
                
                # Check cells in a radius around the drone
                search_radius = int(np.ceil(drop_radius / self.environment.grid_mapper.cell_size_m))
                
                for di in range(-search_radius, search_radius + 1):
                    for dj in range(-search_radius, search_radius + 1):
                        i = center_i + di
                        j = center_j + dj
                        
                        # Check bounds
                        H, W = self.environment.grid_mapper.get_grid_dimensions()
                        if 0 <= i < H and 0 <= j < W:
                            # Calculate distance
                            cell_world_pos = self.environment.grid_mapper.cell_to_world(i, j)
                            distance = np.sqrt((drone_pos[0] - cell_world_pos[0])**2 + 
                                             (drone_pos[1] - cell_world_pos[1])**2)
                            
                            if distance <= drop_radius:
                                # Calculate water amount based on distance, altitude, and actual water available
                                # Scale actual_water (liters) to moisture units
                                distance_factor = (1.0 - distance / drop_radius)
                                water_amount = actual_water * distance_factor * altitude_factor
                                
                                # Accumulate water from multiple drones
                                if (i, j) not in water_drops:
                                    water_drops[(i, j)] = 0.0
                                water_drops[(i, j)] += water_amount
        
        return water_drops
    
    def _check_collisions(self):
        """Check for collisions between drones and environment."""
        for drone_name, drone in self.drones.items():
            position = drone.get_position()
            
            # Check obstacle collision
            collision, obstacle = self.environment.is_position_in_obstacle(position)
            if collision:
                collision_data = {
                    'time': self.simulation_time,
                    'drone': drone_name,
                    'obstacle': obstacle['type'],
                    'position': position.copy()
                }
                self.simulation_log['collisions'].append(collision_data)
                print(f"⚠️ Collision detected: {drone_name} hit {obstacle['type']} at {position}")
    
    def run_scenario(self, scenario_function, steps=1000):
        """Run a complete scenario."""
        print(f"🚁 Running scenario for {steps} steps...")
        
        for step in range(steps):
            # Get control inputs from scenario function
            controls = scenario_function(step, self.simulation_time, self.drones)
            
            # Step simulation
            self.step_simulation(controls)
            
            # Progress reporting
            if step % 100 == 0:
                print(f"  Step {step}/{steps} - Time: {self.simulation_time:.2f}s")
        
        print(f"✅ Scenario completed after {steps} steps ({self.simulation_time:.2f}s)")
    
    def get_drone_status(self, drone_name):
        """Get current status of a drone."""
        if drone_name not in self.drones:
            return None
        
        drone = self.drones[drone_name]
        return {
            'name': drone_name,
            'type': drone.get_drone_type(),
            'position': drone.get_position(),
            'velocity': drone.get_velocity(),
            'speed': drone.get_speed(),
            'characteristics': drone.get_flight_characteristics()
        }
    
    def get_all_drone_status(self):
        """Get status of all drones."""
        return {name: self.get_drone_status(name) for name in self.drones.keys()}
    

    def get_local_atmospheric_conditions(self, world_pos: np.ndarray) -> dict:
        """
        Calculates the local atmospheric conditions including airflow, temperature, and air density.
        
        Implements physically accurate fire-driven convection model:
        - Vertical convection (w): Fire heat creates buoyant upward flow with peaked velocity profile
        - Radial flow (u, v): INWARD at low altitude (<50% height), OUTWARD at high altitude (>50% height)
        - Radial attenuation: Velocities decay with distance from fire center (Gaussian plume)
        - Temperature: Heat from fire diffuses through 3D temperature grid
        - Density: Calculated using ideal gas law approximation
        - Superposition with global wind
        
        Args:
            world_pos: [x, y, z] position in world coordinates (meters)
            
        Returns:
            dict: {
                'velocity': [u, v, w] airflow velocity vector (m/s),
                'temperature': local temperature (K),
                'density': local air density (kg/m³)
            }
        """
        # Start with global wind
        local_airflow = self.environment.weather['wind_velocity'].copy()
        
        # Default atmospheric conditions
        local_temp = self.base_temperature  # 293.15 K (20°C)
        local_density = 1.225  # kg/m³ at sea level, 20°C
        
        # Get temperature from grid if available
        if self.temperature_grid is not None:
            # Map world position to temperature grid
            try:
                center_i, center_j = self.environment.grid_mapper.world_to_cell((world_pos[0], world_pos[1]))
                # Map height to vertical layer (0 to height_levels-1)
                height_levels = self.temperature_grid.shape[0]
                layer_height = self.airflow_H / height_levels
                layer_idx = int(np.clip(world_pos[2] / layer_height, 0, height_levels - 1))
                
                if 0 <= center_i < self.temperature_grid.shape[1] and 0 <= center_j < self.temperature_grid.shape[2]:
                    local_temp = self.temperature_grid[layer_idx, center_i, center_j]
                    
                    # Calculate density using ideal gas approximation
                    # ρ = ρ₀ × (T₀ / T)
                    initial_density = 1.225  # kg/m³
                    initial_temp = 293.15  # K
                    local_density = initial_density * (initial_temp / local_temp)
            except (IndexError, AttributeError):
                pass  # Use default values
        
        # Only apply convection effects below convection height and if fire enabled
        if world_pos[2] >= self.airflow_H:
            return {
                'velocity': local_airflow,
                'temperature': local_temp,
                'density': local_density
            }
        
        # Map world position to grid
        center_i, center_j = self.environment.grid_mapper.world_to_cell((world_pos[0], world_pos[1]))
        
        # Accumulate contributions from nearby burning cells
        influence_radius_cells = 2  # Consider cells within 2-cell radius
        total_convection_z = 0.0
        total_convection_x = 0.0
        total_convection_y = 0.0
        
        for di in range(-influence_radius_cells, influence_radius_cells + 1):
            for dj in range(-influence_radius_cells, influence_radius_cells + 1):
                i = center_i + di
                j = center_j + dj
                
                # Check bounds
                if not (0 <= i < self.environment.fire_grid.H and 0 <= j < self.environment.fire_grid.W):
                    continue
                
                fire_intensity = self.environment.fire_grid.I[i, j]
                if fire_intensity <= 0:
                    continue
                
                # Calculate distance from this fire cell center
                fire_center_x = self.environment.grid_mapper.origin_x + (j + 0.5) * self.environment.grid_mapper.cell_size_m
                fire_center_y = self.environment.grid_mapper.origin_y + (i + 0.5) * self.environment.grid_mapper.cell_size_m
                
                dx = world_pos[0] - fire_center_x
                dy = world_pos[1] - fire_center_y
                horizontal_dist = np.linalg.norm([dx, dy])
                
                # --- VERTICAL CONVECTION (w component) ---
                # Height profile: peaked at ~30% of total height (more realistic than linear)
                normalized_height = world_pos[2] / self.airflow_H
                if normalized_height < 0.3:
                    # Rising phase: velocity increases from ground to peak
                    height_taper = normalized_height / 0.3
                else:
                    # Decaying phase: velocity decreases from peak to top
                    height_taper = (1.0 - normalized_height) / 0.7
                
                # Radial attenuation: Gaussian plume (fire effects decay with horizontal distance)
                plume_radius = self.environment.grid_mapper.cell_size_m * self.plume_radius_factor
                radial_taper = np.exp(-0.5 * (horizontal_dist / plume_radius)**2)
                
                # Calculate upward velocity with realistic magnitude
                cell_convection_z = fire_intensity * self.convection_gain * height_taper * radial_taper
                total_convection_z += cell_convection_z
                
                # --- RADIAL FLOW (u, v components) ---
                # Physical principle: Air flows INWARD at low altitude to replace rising air,
                # then flows OUTWARD at high altitude as the plume spreads
                if normalized_height < 0.5:
                    # LOW ALTITUDE: INWARD flow (air rushes toward fire)
                    inward_strength = (0.5 - normalized_height) / 0.5  # 1.0 at ground, 0.0 at mid-height
                    radial_velocity = -cell_convection_z * self.radial_flow_factor * inward_strength  # Negative = inward
                else:
                    # HIGH ALTITUDE: OUTWARD flow (air spreads out from plume top)
                    outward_strength = (normalized_height - 0.5) / 0.5  # 0.0 at mid-height, 1.0 at top
                    radial_velocity = cell_convection_z * self.radial_flow_factor * outward_strength  # Positive = outward
                
                # Apply radial direction (unit vector from fire center to position)
                if horizontal_dist > 1e-6:
                    radial_dir_x = dx / horizontal_dist
                    radial_dir_y = dy / horizontal_dist
                    total_convection_x += radial_dir_x * radial_velocity
                    total_convection_y += radial_dir_y * radial_velocity
        
        # Add fire convection to global wind (superposition principle)
        local_airflow[0] += total_convection_x
        local_airflow[1] += total_convection_y
        local_airflow[2] += total_convection_z
        
        return {
            'velocity': local_airflow,
            'temperature': local_temp,
            'density': local_density
        }
    
    def create_multi_drone_visualization(self, title="Multi-Drone Analysis"):
        """Create visualization showing all drones together."""
        # Delegate to visualizer
        self.visualizer.create_multi_drone_visualization(
            self.simulation_log, 
            self.drones, 
            self.environment, 
            title
        )
        
        # Generate fire analysis if enabled
        if self.simulation_log['fire_states']:
            self.visualizer.generate_fire_analysis(
                self.simulation_log, 
                self.environment, 
                "multi_drone_combined"
            )

    def create_visualization(self, drone_name=None, title="Simulation Analysis"):
        """Create comprehensive visualization of simulation results."""
        # Delegate to visualizer
        self.visualizer.create_single_drone_visualization(
            self.simulation_log,
            self.drones,
            self.environment,
            drone_name,
            title
        )
    
    def get_simulation_summary(self):
        """Get complete simulation summary."""
        summary = {
            'total_time': self.simulation_time,
            'total_steps': len(self.simulation_log['times']),
            'drones': {name: self.get_drone_status(name) for name in self.drones.keys()},
            'environment': self.environment.get_environment_info(),
            'collisions': len(self.simulation_log['collisions'])
        }
        
        # Add fire information if enabled
        fire_state = self.environment.get_fire_state()
        if fire_state:
            summary['fire'] = fire_state['fire_stats']
        
        return summary