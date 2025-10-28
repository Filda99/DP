"""
Environment System

Creates environmental features like obstacles, terrain, weather effects.
Manages cities, forests, wind, and other environmental factors.
Includes wildfire simulation capabilities.
"""

import pybullet as p
import numpy as np
import random
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.fire_grid import FireGrid
from src.grid_mapper import GridMapper


class Environment:
    """Environment system with obstacles, terrain, and weather."""
    
    def __init__(self):
        """Initialize environment."""
        self.obstacles = []
        self.terrain_zones = []
        
        # Initialize random wind (will vary over time)
        self._initialize_random_wind()
        
        self.weather = {
            'wind_velocity': self.wind_velocity,
            'wind_turbulence': self.wind_turbulence,
            'visibility': 1000.0,  # meters
            'precipitation': 0.0   # 0.0 = none, 1.0 = heavy
        }
        
        # Wind dynamics (for temporal variation)
        self.wind_change_timer = 0.0
        self.wind_change_interval = random.uniform(5.0, 15.0)  # Change wind every 5-15 seconds
        self.target_wind = self.wind_velocity.copy()
        
        # Fire simulation components
        self.fire_grid = None
        self.grid_mapper = None
        self.fire_enabled = False
    
    def _initialize_random_wind(self):
        """Initialize random wind direction and speed."""
        # Random wind speed between 3-12 m/s (light to strong breeze)
        wind_speed = random.uniform(3.0, 12.0)
        
        # Random direction (angle in radians)
        wind_angle = random.uniform(0, 2 * np.pi)
        
        # Convert to velocity vector (x=east, y=north, z=up)
        wind_x = wind_speed * np.cos(wind_angle)
        wind_y = wind_speed * np.sin(wind_angle)
        wind_z = 0.0  # No vertical wind
        
        self.wind_velocity = np.array([wind_x, wind_y, wind_z])
        self.wind_turbulence = random.uniform(0.2, 0.5)  # Random turbulence
        
        print(f"🌬️  Initial wind: {wind_speed:.1f} m/s at {np.degrees(wind_angle):.0f}°")
        print(f"   Vector: [{wind_x:.1f}, {wind_y:.1f}, {wind_z:.1f}] m/s, turbulence: {self.wind_turbulence:.2f}")
        self.fire_visual_objects = []
        
    def create_ground(self):
        """Create ground plane."""
        ground_id = p.loadURDF("plane.urdf")
        return ground_id
    
    def add_city_block(self, position, size=[5, 5, 10], color=[0.7, 0.7, 0.7, 1]):
        """Add a city building/block obstacle."""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[size[0]/2, size[1]/2, size[2]/2]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[size[0]/2, size[1]/2, size[2]/2], 
            rgbaColor=color
        )
        
        building_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], size[2]/2]  # Ground level
        )
        
        obstacle = {
            'id': building_id,
            'type': 'city_block',
            'position': position,
            'size': size,
            'bounds': self._calculate_bounds(position, size)
        }
        
        self.obstacles.append(obstacle)
        return building_id
    
    def add_forest_area(self, center, radius, tree_count=20):
        """Add a forest area with multiple trees."""
        trees = []
        
        for _ in range(tree_count):
            # Random position within radius
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, radius)
            
            tree_x = center[0] + distance * np.cos(angle)
            tree_y = center[1] + distance * np.sin(angle)
            tree_height = random.uniform(8, 15)
            
            tree_id = self._create_tree([tree_x, tree_y, 0], tree_height)
            trees.append(tree_id)
        
        forest = {
            'type': 'forest',
            'center': center,
            'radius': radius,
            'trees': trees,
            'bounds': {
                'min': [center[0] - radius, center[1] - radius, 0],
                'max': [center[0] + radius, center[1] + radius, 20]
            }
        }
        
        self.terrain_zones.append(forest)
        return trees
    
    def _create_tree(self, position, height):
        """Create a single tree."""
        # Tree trunk
        trunk_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=0.3, 
            height=height * 0.7
        )
        trunk_visual = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=0.3, 
            length=height * 0.7,
            rgbaColor=[0.6, 0.3, 0.1, 1]  # Brown
        )
        
        # Tree crown
        crown_collision = p.createCollisionShape(
            p.GEOM_SPHERE, 
            radius=height * 0.3
        )
        crown_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=height * 0.3,
            rgbaColor=[0.1, 0.6, 0.1, 1]  # Green
        )
        
        # Create trunk
        trunk_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=trunk_collision,
            baseVisualShapeIndex=trunk_visual,
            basePosition=[position[0], position[1], height * 0.35]
        )
        
        # Create crown
        crown_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=crown_collision,
            baseVisualShapeIndex=crown_visual,
            basePosition=[position[0], position[1], height * 0.8]
        )
        
        return {'trunk': trunk_id, 'crown': crown_id}
    
    def add_lake(self, center, radius):
        """Add a lake area (visual only, affects flight characteristics)."""
        # Create blue circular lake
        lake_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=radius, 
            height=0.1
        )
        lake_visual = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=radius, 
            length=0.1,
            rgbaColor=[0.1, 0.5, 0.9, 0.8]  # Blue water
        )
        
        lake_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=lake_collision,
            baseVisualShapeIndex=lake_visual,
            basePosition=[center[0], center[1], 0.05]
        )
        
        lake = {
            'id': lake_id,
            'type': 'lake',
            'center': center,
            'radius': radius,
            'bounds': {
                'min': [center[0] - radius, center[1] - radius, 0],
                'max': [center[0] + radius, center[1] + radius, 0.1]
            }
        }
        
        self.terrain_zones.append(lake)
        return lake_id
    
    def set_wind(self, wind_velocity, turbulence=0.0):
        """Set wind conditions."""
        self.weather['wind_velocity'] = np.array(wind_velocity)
        self.weather['wind_turbulence'] = turbulence
    
    def set_weather(self, visibility=1000.0, precipitation=0.0):
        """Set weather conditions."""
        self.weather['visibility'] = visibility
        self.weather['precipitation'] = precipitation
    
    def get_wind_at_position(self, position):
        """Get wind velocity at a specific position with turbulence."""
        base_wind = self.weather['wind_velocity'].copy()
        
        # Add turbulence (primarily horizontal, minimal vertical)
        if self.weather['wind_turbulence'] > 0:
            # Realistic turbulence: 90% horizontal, 10% vertical intensity
            horizontal_turbulence = np.random.normal(0, self.weather['wind_turbulence'], 2)
            vertical_turbulence = np.random.normal(0, self.weather['wind_turbulence'] * 0.1, 1)
            turbulence = np.array([horizontal_turbulence[0], horizontal_turbulence[1], vertical_turbulence[0]])
            base_wind += turbulence
        
        # Height effect - wind increases with altitude (reduced effect)
        height_factor = 1.0 + position[2] * 0.01  # Reduced from 0.05 to 0.01
        base_wind *= height_factor
        
        return base_wind
    
    def is_position_in_obstacle(self, position):
        """Check if position collides with any obstacle."""
        for obstacle in self.obstacles:
            bounds = obstacle['bounds']
            if (bounds['min'][0] <= position[0] <= bounds['max'][0] and
                bounds['min'][1] <= position[1] <= bounds['max'][1] and
                bounds['min'][2] <= position[2] <= bounds['max'][2]):
                return True, obstacle
        return False, None
    
    def get_terrain_type_at_position(self, position):
        """Get terrain type at a specific position."""
        for zone in self.terrain_zones:
            if zone['type'] == 'forest':
                center = zone['center']
                radius = zone['radius']
                distance = np.sqrt((position[0] - center[0])**2 + (position[1] - center[1])**2)
                if distance <= radius:
                    return 'forest'
            
            elif zone['type'] == 'lake':
                center = zone['center']
                radius = zone['radius']
                distance = np.sqrt((position[0] - center[0])**2 + (position[1] - center[1])**2)
                if distance <= radius and position[2] <= 5:  # Low altitude over water
                    return 'lake'
        
        return 'open'
    
    def _calculate_bounds(self, position, size):
        """Calculate bounding box for an obstacle."""
        return {
            'min': [position[0] - size[0]/2, position[1] - size[1]/2, 0],
            'max': [position[0] + size[0]/2, position[1] + size[1]/2, size[2]]
        }
    
    def get_environment_info(self):
        """Get complete environment information."""
        return {
            'obstacles': len(self.obstacles),
            'terrain_zones': len(self.terrain_zones),
            'weather': self.weather.copy()
        }
    
    def create_city_environment(self):
        """Create a pre-configured city environment."""
        # Create buildings in a grid pattern
        building_positions = [
            [-20, -10], [-10, -10], [10, -10], [20, -10],
            [-20, 10], [-10, 10], [10, 10], [20, 10],
            [-15, 0], [0, 0], [15, 0]
        ]
        
        for pos in building_positions:
            height = random.uniform(15, 25)
            width = random.uniform(4, 8)
            self.add_city_block(pos, [width, width, height])
        
        # Add some parks (small forest areas)
        self.add_forest_area([-5, -20], 8, 10)
        self.add_forest_area([25, 15], 6, 8)
        
        print(f"✅ City environment created with {len(self.obstacles)} buildings")
    
    def create_natural_environment(self):
        """Create a pre-configured natural environment."""
        # Large forest areas
        self.add_forest_area([0, 0], 15, 30)
        self.add_forest_area([30, 20], 12, 25)
        self.add_forest_area([-25, -15], 10, 20)
        
        # Lakes
        self.add_lake([15, -20], 8)
        self.add_lake([-10, 25], 6)
        
        print(f"✅ Natural environment created with {len(self.terrain_zones)} terrain zones")
    
    def create_mixed_environment(self):
        """Create a mixed urban/natural environment."""
        # Small city center
        for x in [-10, 0, 10]:
            for y in [-5, 5]:
                self.add_city_block([x, y], [6, 6, random.uniform(12, 20)])
        
        # Surrounding nature
        self.add_forest_area([20, 20], 12, 20)
        self.add_forest_area([-25, 0], 10, 15)
        self.add_lake([0, -25], 10)
        
        print(f"✅ Mixed environment created with {len(self.obstacles)} buildings and {len(self.terrain_zones)} natural zones")
    
    def enable_fire_simulation(self, grid_width_m=100, grid_height_m=100, cell_size_m=2.0, 
                             dt=0.1, alpha=1.0, k_wind=1.5, wind_dir=0.0):
        """
        Enable wildfire simulation in the environment.
        
        Args:
            grid_width_m: Width of fire grid in meters
            grid_height_m: Height of fire grid in meters  
            cell_size_m: Size of each fire cell in meters
            dt: Fire simulation time step
            alpha: Distance decay factor for fire spread
            k_wind: Wind influence factor
            wind_dir: Wind direction in radians
        """
        # Create grid mapper
        self.grid_mapper = GridMapper(grid_width_m, grid_height_m, cell_size_m)
        
        # Get grid dimensions
        H, W = self.grid_mapper.get_grid_dimensions()
        
        # Create fire grid with wind from weather system
        wind_velocity = self.weather['wind_velocity'][:2]  # Only x,y components
        wind_speed = np.linalg.norm(wind_velocity)
        
        if wind_speed > 0.01:
            wind_angle = np.arctan2(wind_velocity[1], wind_velocity[0])
        else:
            wind_angle = 0.0
        
        # Create base lambda values for fire spread
        l_base = np.ones(H) * 0.5  # Base fire spread rate (probability ~20% per 0.5s step)
        
        self.fire_grid = FireGrid(
            H=H, W=W, dt=dt, alpha=alpha, 
            k_wind=k_wind, k_slope=1.0,
            wind_dir=wind_angle, l_base=l_base
        )
        
        # Enable fire BEFORE initializing fuel from terrain
        self.fire_enabled = True
        
        # Modify fuel based on terrain
        self._initialize_fire_fuel_from_terrain()
        
        # IMPORTANT: Clear any random fires that were started in reset_random()
        # We want to start fires manually, not have random initial fires
        self.fire_grid.B[:] = False
        self.fire_grid.I[:] = 0.0
        
        print(f"✅ Fire simulation enabled: {H}x{W} grid, {cell_size_m}m cells")
    
    def _initialize_fire_fuel_from_terrain(self):
        """Initialize fire fuel levels based on terrain zones."""
        if not self.fire_enabled or self.fire_grid is None:
            return
        
        H, W = self.grid_mapper.get_grid_dimensions()
        
        # Debug counters
        debug_counts = {'building': 0, 'forest': 0, 'lake': 0, 'default': 0}
        
        # Set fuel based on terrain type
        for i in range(H):
            for j in range(W):
                world_pos = self.grid_mapper.cell_to_world(i, j)
                fuel_level = 0.3  # Default fuel level (open terrain)
                burn_rate = 0.08  # Default burn rate (grass burns fast)
                terrain_type = 'default'
                
                # FIRST: Check if position is inside a building - NO FUEL
                in_building = False
                for obstacle in self.obstacles:
                    if obstacle['type'] == 'city_block':
                        bounds = obstacle['bounds']
                        if (bounds['min'][0] <= world_pos[0] <= bounds['max'][0] and
                            bounds['min'][1] <= world_pos[1] <= bounds['max'][1]):
                            fuel_level = 0.0  # Buildings have no fuel!
                            burn_rate = 0.0   # Buildings don't burn
                            in_building = True
                            terrain_type = 'building'
                            break
                
                # Only check terrain zones if not in a building
                if not in_building:
                    for zone in self.terrain_zones:
                        if zone['type'] == 'forest':
                            center = zone['center']
                            radius = zone['radius']
                            distance = np.sqrt((world_pos[0] - center[0])**2 + 
                                             (world_pos[1] - center[1])**2)
                            if distance <= radius:
                                fuel_level = 0.8  # High fuel in forests
                                burn_rate = 0.03  # Forest burns slowly but longer
                                terrain_type = 'forest'
                                break
                        elif zone['type'] == 'lake':
                            center = zone['center']
                            radius = zone['radius']
                            distance = np.sqrt((world_pos[0] - center[0])**2 + 
                                             (world_pos[1] - center[1])**2)
                            if distance <= radius:
                                fuel_level = 0.0  # No fuel in water
                                burn_rate = 0.0   # Water doesn't burn
                                terrain_type = 'lake'
                                break
                
                self.fire_grid.F[i, j] = fuel_level
                self.fire_grid.fuel_burn_rate[i, j] = burn_rate
                debug_counts[terrain_type] += 1
        
        print(f"✅ Fire fuel levels initialized from terrain:")
        print(f"   Buildings: {debug_counts['building']} cells")
        print(f"   Forests: {debug_counts['forest']} cells")
        print(f"   Lakes: {debug_counts['lake']} cells")
        print(f"   Open terrain: {debug_counts['default']} cells")

    
    def start_fire_at_position(self, world_pos, intensity=0.2):
        """
        Start a fire at a specific world position.
        
        Args:
            world_pos: (x, y) position in world coordinates
            intensity: Initial fire intensity (used for visualization only)
        """
        if not self.fire_enabled:
            print("❌ Fire simulation not enabled")
            return False
        
        i, j = self.grid_mapper.world_to_cell(world_pos)
        
        if self.fire_grid.F[i, j] > 0:  # Only if there's fuel
            self.fire_grid.B[i, j] = True
            self.fire_grid.I[i, j] = np.minimum(1.0, self.fire_grid.F[i, j])
            print(f"✅ Fire started at world pos {world_pos} -> cell ({i}, {j})")
            return True
        else:
            print(f"❌ Cannot start fire at {world_pos} - no fuel")
            return False
    
    def _update_wind_dynamics(self, dt=0.1):
        """
        Update wind velocity over time with smooth transitions.
        Wind changes gradually to create realistic temporal variation.
        
        Args:
            dt: Time step in seconds (default 0.1s per simulation step)
        """
        self.wind_change_timer += dt
        
        # Check if it's time to set a new target wind
        if self.wind_change_timer >= self.wind_change_interval:
            # Generate new target wind
            wind_speed = random.uniform(3.0, 12.0)  # 3-12 m/s
            wind_angle = random.uniform(0, 2 * np.pi)
            
            self.target_wind[0] = wind_speed * np.cos(wind_angle)
            self.target_wind[1] = wind_speed * np.sin(wind_angle)
            self.target_wind[2] = 0.0
            
            # Reset timer with new random interval
            self.wind_change_timer = 0.0
            self.wind_change_interval = random.uniform(5.0, 15.0)
            
            # Also vary turbulence
            self.wind_turbulence = random.uniform(0.2, 0.5)
        
        # Smoothly interpolate current wind toward target (takes ~2-3 seconds to transition)
        blend_factor = 0.02  # Smooth interpolation rate
        self.wind_velocity = (1 - blend_factor) * self.wind_velocity + blend_factor * self.target_wind
        
        # Update weather dictionary
        self.weather['wind_velocity'] = self.wind_velocity
        self.weather['wind_turbulence'] = self.wind_turbulence
    
    def update_fire_simulation(self, suppression_assignments=None, water_drops=None):
        """
        Update the fire simulation by one step.
        
        Args:
            suppression_assignments: Dict mapping (i,j) to list of suppression probabilities (deprecated)
            water_drops: Dict mapping (i,j) to water amount (0.0 to 1.0+)
        """
        if not self.fire_enabled:
            return
        
        # Update wind dynamics (gradual changes over time)
        self._update_wind_dynamics(dt=0.1)
        
        # Update wind from unified weather system
        wind_velocity = self.weather['wind_velocity'][:2]  # Only x, y components
        wind_speed = np.linalg.norm(wind_velocity)
        
        if wind_speed > 0.01:  # Avoid division by zero
            wind_angle = np.arctan2(wind_velocity[1], wind_velocity[0])
        else:
            wind_angle = 0.0
        
        # Update fire grid wind parameters
        self.fire_grid.wind_dir = wind_angle
        
        # Optional: Scale wind influence by wind speed
        # Stronger wind = more spread influence
        # base_k_wind = 1.5
        # self.fire_grid.k_wind = base_k_wind * min(2.0, wind_speed / 5.0)
        
        # Step the fire simulation with water drops
        self.fire_grid.step(suppression_assignments, water_drops)
    
    def get_fire_state(self):
        """Get current fire simulation state."""
        if not self.fire_enabled:
            return None
        
        return {
            'fire_grid_state': self.fire_grid.get_state(),
            'fire_stats': self.fire_grid.get_stats(),
            'grid_bounds': self.grid_mapper.get_grid_bounds(),
            'cell_size': self.grid_mapper.cell_size_m
        }
    
    def visualize_fire_in_simulation(self):
        """Create visual objects for fire in PyBullet simulation."""
        if not self.fire_enabled:
            return
        
        # Remove old fire visualizations
        for obj_id in self.fire_visual_objects:
            try:
                p.removeBody(obj_id)
            except:
                pass
        self.fire_visual_objects.clear()
        
        # Create new fire visualizations
        H, W = self.fire_grid.B.shape
        
        for i in range(H):
            for j in range(W):
                if self.fire_grid.B[i, j]:  # If cell is burning
                    world_pos = self.grid_mapper.cell_to_world(i, j)
                    intensity = self.fire_grid.I[i, j]
                    
                    # Create fire visual (red cylinder)
                    fire_height = 0.5 + intensity * 2.0  # Height based on intensity
                    fire_radius = self.grid_mapper.cell_size_m * 0.4
                    
                    # Color based on intensity (yellow to red)
                    red = 1.0
                    green = max(0.0, 1.0 - intensity * 2.0)
                    blue = 0.0
                    alpha = 0.8
                    
                    collision_shape = p.createCollisionShape(
                        p.GEOM_CYLINDER,
                        radius=fire_radius,
                        height=fire_height
                    )
                    visual_shape = p.createVisualShape(
                        p.GEOM_CYLINDER,
                        radius=fire_radius,
                        length=fire_height,
                        rgbaColor=[red, green, blue, alpha]
                    )
                    
                    fire_obj = p.createMultiBody(
                        baseMass=0,  # Static
                        baseCollisionShapeIndex=collision_shape,
                        baseVisualShapeIndex=visual_shape,
                        basePosition=[world_pos[0], world_pos[1], fire_height/2]
                    )
                    
                    self.fire_visual_objects.append(fire_obj)