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
except ImportError:
    from src.environment import Environment
    from src.drones import Quadcopter, FixedWing
    from src.visualizer import SimulationVisualizer


class Simulation:
    """Complete simulation manager."""
    
    def __init__(self, gui=False):
        """Initialize simulation."""
        self.gui = gui
        self.physics_client = None
        self.drones = {}
        self.environment = Environment()
        self.simulation_time = 0.0
        self.timestep = 1/240.0  # 240 FPS
        # --- Airflow model variables (corrected for physical accuracy) ---
        self.airflow_H = 50.0  # Convection height limit (m)
        self.convection_gain = 8.0  # Base updraft velocity [m/s] for unit fire intensity (realistic: 5-20 m/s)
        self.plume_radius_factor = 2.0  # Plume extends ~2x cell size
        self.radial_flow_factor = 0.3  # Radial flow is ~30% of vertical velocity
        
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
        return quad
    
    def add_fixedwing(self, name, position=[0, 0, 5], mass=1.0):
        """Add a fixed-wing aircraft to the simulation."""
        fw = FixedWing(position, mass)
        self.drones[name] = fw
        
        # Initialize logging for this drone
        self.simulation_log['drones'][name] = {
            'type': 'fixedwing', 
            'positions': [],
            'forces': [],
            'velocities': [],
            'control_inputs': []
        }
        
        print(f"✅ Added fixed-wing '{name}' at {position}")
        return fw
    
    def setup_city_environment(self):
        """Setup city environment with buildings."""
        self.environment.create_city_environment()
        
    def setup_natural_environment(self):
        """Setup natural environment with forests and lakes."""
        self.environment.create_natural_environment()
        
    def setup_mixed_environment(self):
        """Setup mixed urban/natural environment."""
        self.environment.create_mixed_environment()
    
    def enable_fire_simulation(self, grid_width_m=100, grid_height_m=100, cell_size_m=2.0):
        """Enable wildfire simulation in the environment."""
        self.environment.enable_fire_simulation(grid_width_m, grid_height_m, cell_size_m)
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
                
                # 1. Get local airflow (Fire -> Airflow)
                local_airflow = self.get_local_airflow(drone.get_position())
                
                # 2. Apply environmental effects (Airflow -> Aircraft)
                drone.apply_environmental_effects(local_airflow)

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
        
        # Update fire simulation if enabled
        if self.environment.fire_enabled:
            # Calculate suppression from drones (if any)
            suppression_assignments = self._calculate_drone_suppression()
            
            # Update fire simulation
            self.environment.update_fire_simulation(suppression_assignments)
            
            # Update fire visualization every 10 steps (for performance)
            if len(self.simulation_log['times']) % 10 == 0:
                self.environment.visualize_fire_in_simulation()
            
            # Log fire state
            fire_state = self.environment.get_fire_state()
            self.simulation_log['fire_states'].append(fire_state)
        
        # Check for collisions with environment
        self._check_collisions()
    
    def _calculate_drone_suppression(self):
        """Calculate fire suppression effects from drone positions."""
        if not self.environment.fire_enabled:
            return {}
        
        suppression_assignments = {}
        
        # Realistic suppression model:
        # - Drones must be low altitude to suppress (< 15m)
        # - Small suppression radius (only 5m)
        # - Low effectiveness (15% base probability)
        suppression_radius = 5.0  # meters - drones must be close
        base_effectiveness = 0.15  # base suppression probability per drone
        max_altitude = 15.0  # meters - maximum altitude for effective suppression
        
        for drone_name, drone in self.drones.items():
            drone_pos = drone.get_position()
            
            # Altitude check - drones too high cannot suppress fires
            if drone_pos[2] > max_altitude:
                continue  # Skip this drone, too high to suppress
            
            # Altitude factor - effectiveness decreases with altitude
            altitude_factor = 1.0 - (drone_pos[2] / max_altitude)  # 1.0 at ground, 0.0 at max_altitude
            
            # Check if drone is close to any burning cells
            if self.environment.grid_mapper.is_position_in_bounds((drone_pos[0], drone_pos[1])):
                # Get nearby cells within suppression radius
                center_i, center_j = self.environment.grid_mapper.world_to_cell((drone_pos[0], drone_pos[1]))
                
                # Check cells in a radius around the drone
                search_radius = int(np.ceil(suppression_radius / self.environment.grid_mapper.cell_size_m))
                
                for di in range(-search_radius, search_radius + 1):
                    for dj in range(-search_radius, search_radius + 1):
                        i = center_i + di
                        j = center_j + dj
                        
                        # Check bounds
                        H, W = self.environment.grid_mapper.get_grid_dimensions()
                        if 0 <= i < H and 0 <= j < W:
                            # Check if cell is burning
                            if self.environment.fire_grid.B[i, j]:
                                # Calculate distance
                                cell_world_pos = self.environment.grid_mapper.cell_to_world(i, j)
                                distance = np.sqrt((drone_pos[0] - cell_world_pos[0])**2 + 
                                                 (drone_pos[1] - cell_world_pos[1])**2)
                                
                                if distance <= suppression_radius:
                                    # Add suppression assignment
                                    if (i, j) not in suppression_assignments:
                                        suppression_assignments[(i, j)] = []
                                    
                                    # Effectiveness decreases with both distance and altitude
                                    distance_factor = (1.0 - distance / suppression_radius)
                                    effectiveness = base_effectiveness * distance_factor * altitude_factor
                                    suppression_assignments[(i, j)].append(effectiveness)
        
        return suppression_assignments
    
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
    

    def get_local_airflow(self, world_pos: np.ndarray) -> np.ndarray:
        """
        Calculates the local airflow vector (u, v, w) based on global wind and fire convection.
        
        Implements physically accurate fire-driven convection model:
        - Vertical convection (w): Fire heat creates buoyant upward flow with peaked velocity profile
        - Radial flow (u, v): INWARD at low altitude (<50% height), OUTWARD at high altitude (>50% height)
        - Radial attenuation: Velocities decay with distance from fire center (Gaussian plume)
        - Superposition with global wind
        
        Args:
            world_pos: [x, y, z] position in world coordinates (meters)
            
        Returns:
            [u, v, w] airflow velocity vector (m/s)
        """
        # Start with global wind
        local_airflow = self.weather['wind_velocity'].copy()
        
        if not self.fire_enabled:
            return local_airflow
        
        # Only apply fire effects below convection height
        if world_pos[2] >= self.airflow_H:
            return local_airflow
        
        # Map world position to grid
        center_i, center_j = self.grid_mapper.world_to_cell((world_pos[0], world_pos[1]))
        
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
                if not (0 <= i < self.fire_grid.H and 0 <= j < self.fire_grid.W):
                    continue
                
                fire_intensity = self.fire_grid.I[i, j]
                if fire_intensity <= 0:
                    continue
                
                # Calculate distance from this fire cell center
                fire_center_x = self.grid_mapper.origin_x + (j + 0.5) * self.grid_mapper.cell_size_m
                fire_center_y = self.grid_mapper.origin_y + (i + 0.5) * self.grid_mapper.cell_size_m
                
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
                plume_radius = self.grid_mapper.cell_size_m * self.plume_radius_factor
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
        
        return local_airflow
    
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
        if self.environment.fire_enabled and self.simulation_log['fire_states']:
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
        if self.environment.fire_enabled:
            fire_state = self.environment.get_fire_state()
            if fire_state:
                summary['fire'] = fire_state['fire_stats']
        
        return summary