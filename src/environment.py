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
import time

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.fire_grid import FireGrid
from src.grid_mapper import GridMapper


class Environment:
    """Environment system with obstacles, terrain, and weather."""
    
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    
    def __init__(self):
        """Initialize environment."""
        self.obstacles = []
        self.terrain_zones = []
        self.fire_visual_objects = []
        
        # Initialize random wind (will vary over time)
        self._initialize_random_wind()
        
        self.weather = {
            'wind_velocity': self.wind_velocity,
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
        self.fire_time_accumulator = 0.0  # Accumulates real time for fire updates
        
        # Spatial index for fast terrain lookups (populated when needed)
        self.spatial_hash = None  # Will be dict: (grid_x, grid_y) -> list of objects
        self.spatial_hash_cell_size = 10.0  # meters per hash cell
    
    # ============================================================================
    # SPATIAL INDEX FOR PERFORMANCE
    # ============================================================================
    
    def _build_spatial_hash(self, cell_size=10.0):
        """
        Build a spatial hash for fast terrain lookups.
        Converts O(N) lookup to O(1) by pre-indexing objects into grid cells.
        
        Args:
            cell_size: Size of hash grid cells in meters (default 10m)
        """
        print(f"   ⏱️  Building spatial index (cell size: {cell_size}m)...")
        t_start = time.time()
        
        self.spatial_hash = {}
        self.spatial_hash_cell_size = cell_size
        
        # Index obstacles (buildings)
        for obstacle in self.obstacles:
            if obstacle['type'] == 'city_block':
                bounds = obstacle['bounds']
                # Find all hash cells this obstacle overlaps
                x_min, x_max = bounds['min'][0], bounds['max'][0]
                y_min, y_max = bounds['min'][1], bounds['max'][1]
                
                # Calculate hash cell range
                hx_min = int(np.floor(x_min / cell_size))
                hx_max = int(np.ceil(x_max / cell_size))
                hy_min = int(np.floor(y_min / cell_size))
                hy_max = int(np.ceil(y_max / cell_size))
                
                # Add obstacle to all overlapping cells
                for hx in range(hx_min, hx_max + 1):
                    for hy in range(hy_min, hy_max + 1):
                        key = (hx, hy)
                        if key not in self.spatial_hash:
                            self.spatial_hash[key] = []
                        self.spatial_hash[key].append(('obstacle', obstacle))
        
        # Index terrain zones (forests, lakes)
        for zone in self.terrain_zones:
            if zone['type'] in ['forest', 'lake']:
                bounds = zone['bounds']
                x_min, x_max = bounds['min'][0], bounds['max'][0]
                y_min, y_max = bounds['min'][1], bounds['max'][1]
                
                # Calculate hash cell range
                hx_min = int(np.floor(x_min / cell_size))
                hx_max = int(np.ceil(x_max / cell_size))
                hy_min = int(np.floor(y_min / cell_size))
                hy_max = int(np.ceil(y_max / cell_size))
                
                # Add zone to all overlapping cells
                for hx in range(hx_min, hx_max + 1):
                    for hy in range(hy_min, hy_max + 1):
                        key = (hx, hy)
                        if key not in self.spatial_hash:
                            self.spatial_hash[key] = []
                        self.spatial_hash[key].append(('zone', zone))
        
        num_cells = len(self.spatial_hash)
        total_objects = len(self.obstacles) + len(self.terrain_zones)
        print(f"      ✅ Spatial index built in {time.time() - t_start:.3f}s")
        print(f"         {total_objects} objects indexed into {num_cells} hash cells")
    
    def _get_hash_cell(self, world_pos):
        """Get the spatial hash cell coordinates for a world position."""
        hx = int(np.floor(world_pos[0] / self.spatial_hash_cell_size))
        hy = int(np.floor(world_pos[1] / self.spatial_hash_cell_size))
        return (hx, hy)
    
    def get_fuel_at_position(self, world_pos):
        """
        Get fuel properties at a world position using spatial index for O(1) lookup.
        
        This method is the single source of truth for fuel properties.
        It encapsulates all terrain logic and provides a clean interface.
        
        Args:
            world_pos: (x, y) position in world coordinates
            
        Returns:
            tuple: (fuel_level, burn_rate) 
                   fuel_level: 0.0 to 1.0
                   burn_rate: rate at which fuel is consumed
        """
        # Fuel type constants (matching FireGrid constants)
        FUEL_WATER = (0.0, 0.0)
        FUEL_BUILDING = (0.9, 0.0001)
        FUEL_FOREST = (0.8, 0.03)
        FUEL_GRASS = (0.3, 0.08)
        
        # If no spatial hash, fall back to linear search (slow but works)
        if self.spatial_hash is None:
            return self._get_fuel_at_position_slow(world_pos)
        
        # Get hash cell and retrieve nearby objects (O(1) lookup!)
        hash_key = self._get_hash_cell(world_pos)
        nearby_objects = self.spatial_hash.get(hash_key, [])
        
        # Priority 1: Check for buildings (buildings can burn, but very slowly)
        for obj_type, obj in nearby_objects:
            if obj_type == 'obstacle' and obj['type'] == 'city_block':
                bounds = obj['bounds']
                if (bounds['min'][0] <= world_pos[0] <= bounds['max'][0] and
                    bounds['min'][1] <= world_pos[1] <= bounds['max'][1]):
                    return FUEL_BUILDING
        
        # Priority 2: Check for water (complete fire break)
        for obj_type, obj in nearby_objects:
            if obj_type == 'zone' and obj['type'] == 'lake':
                if obj.get('shape') == 'rectangle':
                    # Rotated rectangle - use point-in-polygon test
                    if self._point_in_rectangle_env(world_pos, obj['corners']):
                        return FUEL_WATER
                else:
                    # Circular lake
                    if self._point_in_circle_env(world_pos, obj['center'], obj['radius']):
                        return FUEL_WATER
        
        # Priority 3: Check for forest (high fuel)
        for obj_type, obj in nearby_objects:
            if obj_type == 'zone' and obj['type'] == 'forest':
                if self._point_in_circle_env(world_pos, obj['center'], obj['radius']):
                    return FUEL_FOREST
        
        # Default: Open terrain (grass)
        return FUEL_GRASS
    
    def _get_fuel_at_position_slow(self, world_pos):
        """Fallback method when spatial hash is not available (slow O(N) version)."""
        FUEL_WATER = (0.0, 0.0)
        FUEL_BUILDING = (0.9, 0.0001)
        FUEL_FOREST = (0.8, 0.03)
        FUEL_GRASS = (0.3, 0.08)
        
        # Check buildings
        for obstacle in self.obstacles:
            if obstacle['type'] == 'city_block':
                bounds = obstacle['bounds']
                if (bounds['min'][0] <= world_pos[0] <= bounds['max'][0] and
                    bounds['min'][1] <= world_pos[1] <= bounds['max'][1]):
                    return FUEL_BUILDING
        
        # Check water
        for zone in self.terrain_zones:
            if zone['type'] == 'lake':
                if zone.get('shape') == 'rectangle':
                    if self._point_in_rectangle_env(world_pos, zone['corners']):
                        return FUEL_WATER
                else:
                    if self._point_in_circle_env(world_pos, zone['center'], zone['radius']):
                        return FUEL_WATER
        
        # Check forest
        for zone in self.terrain_zones:
            if zone['type'] == 'forest':
                if self._point_in_circle_env(world_pos, zone['center'], zone['radius']):
                    return FUEL_FOREST
        
        return FUEL_GRASS
    
    def _point_in_circle_env(self, point, center, radius):
        """Check if point is inside a circle."""
        distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        return distance <= radius
    
    def _point_in_rectangle_env(self, point, corners):
        """Check if point is inside a rotated rectangle using ray casting algorithm."""
        inside = False
        n = len(corners)
        for k in range(n):
            k_next = (k + 1) % n
            xi, yi = corners[k]
            xj, yj = corners[k_next]
            if ((yi > point[1]) != (yj > point[1])) and \
               (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi):
                inside = not inside
        return inside
    
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
        
        print(f"🌬️  Initial wind: {wind_speed:.1f} m/s at {np.degrees(wind_angle):.0f}°")
        
    # ============================================================================
    # BASIC TERRAIN CREATION
    # ============================================================================
    
    def create_ground(self):
        """Create ground plane."""
        ground_id = p.loadURDF("../urdf/plane.urdf")
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
    
    def add_water_rectangle(self, center, length, width, angle):
        """
        Add a rectangular water body (for rivers/streams).
        
        Args:
            center: [x, y] center position
            length: Length along the flow direction (m)
            width: Width perpendicular to flow (m)
            angle: Rotation angle in radians (flow direction)
        """
        # Create rectangular water body
        water_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, 0.05]
        )
        water_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, 0.05],
            rgbaColor=[0.1, 0.5, 0.9, 0.8]  # Blue water
        )
        
        # Calculate quaternion from angle (rotation around Z-axis)
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        quaternion = [0, 0, sin_half, cos_half]  # [x, y, z, w]
        
        water_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=water_collision,
            baseVisualShapeIndex=water_visual,
            basePosition=[center[0], center[1], 0.05],
            baseOrientation=quaternion
        )
        
        # Calculate bounding box (rotated rectangle)
        # For fire grid purposes, we'll use a conservative axis-aligned bounding box
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Corner offsets in local frame
        corners_local = [
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ]
        
        # Rotate corners to world frame
        corners_world = []
        for lx, ly in corners_local:
            wx = center[0] + lx * cos_a - ly * sin_a
            wy = center[1] + lx * sin_a + ly * cos_a
            corners_world.append([wx, wy])
        
        # Find axis-aligned bounding box
        x_coords = [c[0] for c in corners_world]
        y_coords = [c[1] for c in corners_world]
        
        water = {
            'id': water_id,
            'type': 'lake',  # Treated as lake for fire purposes
            'center': center,
            'length': length,
            'width': width,
            'angle': angle,
            'shape': 'rectangle',
            'corners': corners_world,  # For precise collision detection
            'bounds': {
                'min': [min(x_coords), min(y_coords), 0],
                'max': [max(x_coords), max(y_coords), 0.1]
            }
        }
        
        self.terrain_zones.append(water)
        return water_id
    
    # ============================================================================
    # WEATHER & WIND SYSTEM
    # ============================================================================
    
    def set_wind(self, wind_velocity, turbulence=0.0):
        """Set wind conditions."""
        self.weather['wind_velocity'] = np.array(wind_velocity)
    
    def set_weather(self, visibility=1000.0, precipitation=0.0):
        """Set weather conditions."""
        self.weather['visibility'] = visibility
        self.weather['precipitation'] = precipitation
    
    def get_wind_at_position(self, position):
        """Get wind velocity at a specific position with turbulence."""
        base_wind = self.weather['wind_velocity'].copy()
        
        # Height effect - wind increases with altitude (reduced effect)
        height_factor = 1.0 + position[2] * 0.01  # Reduced from 0.05 to 0.01
        base_wind *= height_factor
        
        return base_wind
    
    def _update_wind_dynamics(self, dt=0.1):
        """
        Update wind velocity over time with smooth transitions.
        Wind changes gradually by small increments to create realistic temporal variation.
        
        Args:
            dt: Time step in seconds (default 0.1s per simulation step)
        """
        self.wind_change_timer += dt
        
        # Check if it's time to make a small wind adjustment
        if self.wind_change_timer >= self.wind_change_interval:
            # Generate SMALL changes to current wind instead of new random wind
            current_speed = np.linalg.norm(self.target_wind[:2])
            current_angle = np.arctan2(self.target_wind[1], self.target_wind[0])
            
            # Small random changes (±10-20% speed, ±15 degrees direction)
            speed_change = random.uniform(-0.15, 0.15) * current_speed  # ±15% speed change
            angle_change = random.uniform(-np.pi/12, np.pi/12)  # ±15 degrees
            
            # Apply changes with bounds
            new_speed = np.clip(current_speed + speed_change, 2.0, 15.0)  # Keep in realistic range
            new_angle = current_angle + angle_change
            
            # Update target wind
            self.target_wind[0] = new_speed * np.cos(new_angle)
            self.target_wind[1] = new_speed * np.sin(new_angle)
            self.target_wind[2] = 0.0
            
            # Reset timer with new random interval (change every 3-8 seconds)
            self.wind_change_timer = 0.0
            self.wind_change_interval = random.uniform(3.0, 8.0)  # More frequent, smaller changes
        
        # Smoothly interpolate current wind toward target
        blend_factor = 0.02  # Smooth interpolation rate
        self.wind_velocity = (1 - blend_factor) * self.wind_velocity + blend_factor * self.target_wind
        
        # Update weather dictionary
        self.weather['wind_velocity'] = self.wind_velocity
        
    # ============================================================================
    # SPATIAL QUERIES & COLLISION DETECTION
    # ============================================================================
    
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
                # Check if it's a rectangle or circle
                if zone.get('shape') == 'rectangle':
                    # Point-in-polygon test for rotated rectangle
                    point = np.array([position[0], position[1]])
                    corners = zone['corners']
                    
                    # Use ray casting algorithm
                    inside = False
                    n = len(corners)
                    for i in range(n):
                        j = (i + 1) % n
                        xi, yi = corners[i]
                        xj, yj = corners[j]
                        
                        if ((yi > position[1]) != (yj > position[1])) and \
                           (position[0] < (xj - xi) * (position[1] - yi) / (yj - yi) + xi):
                            inside = not inside
                    
                    if inside and position[2] <= 5:  # Low altitude over water
                        return 'lake'
                else:
                    # Original circular lake
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
    
    # ============================================================================
    # FIRE SIMULATION - INITIALIZATION
    # ============================================================================
    
    def enable_fire_simulation(self, grid_width_m=100, grid_height_m=100, cell_size_m=2.0, 
                             dt=0.1, alpha=1.0, k_wind=1.5, wind_dir=0.0, lazy_fuel=True):
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
            lazy_fuel: If True, load fuel on-demand (fast startup). If False, preload all fuel (slow startup)
        """
        t_total_start = time.time()
        
        # Build spatial hash for fast terrain lookups
        self._build_spatial_hash(cell_size=10.0)
        
        # Create grid mapper
        print("    Creating grid mapper...")
        t_start = time.time()
        self.grid_mapper = GridMapper(grid_width_m, grid_height_m, cell_size_m)
        
        # Get grid dimensions
        H, W = self.grid_mapper.get_grid_dimensions()
        
        # Create fire grid with wind from weather system
        print("    Calculating wind parameters...")
        t_start = time.time()
        wind_velocity = self.weather['wind_velocity'][:2]  # Only x,y components
        wind_speed = np.linalg.norm(wind_velocity)
        
        if wind_speed > 0.01:
            wind_angle = np.arctan2(wind_velocity[1], wind_velocity[0])
        else:
            wind_angle = 0.0
        
        # Create base lambda values for fire spread
        # REDUCED from 0.5 to 0.1 for much slower spread
        l_base = np.ones(H) * 0.1  # Base fire spread rate (SLOW - realistic wildfire)
        
        print(f"    Initializing FireGrid ({H}x{W} cells, dt={dt:.4f}s)...")
        t_start = time.time()
        self.fire_grid = FireGrid(
            H=H, W=W, dt=dt, alpha=alpha, 
            k_wind=k_wind, k_slope=1.0,
            wind_dir=wind_angle, l_base=l_base
        )
        
        self.fire_enabled = True

        # Choose fuel loading strategy
        if lazy_fuel:
            # LAZY LOADING: Load fuel on-demand (fast startup)
            print("    Enabling lazy fuel loading (on-demand)...")
            t_start = time.time()
            self.fire_grid.enable_lazy_fuel_loading(self, self.grid_mapper)
        else:
            # EAGER LOADING: Preload all fuel (slow startup)
            print("    Mapping fuel from terrain (buildings, forests, water)...")
            t_start = time.time()
            self._initialize_fire_fuel_from_terrain()
        
        # IMPORTANT: Clear any random fires that were started in reset_random()
        # We want to start fires manually, not have random initial fires
        self.fire_grid.B[:] = False
        self.fire_grid.I[:] = 0.0
        
        print(f"✅ Fire simulation enabled: {H}x{W} grid, {cell_size_m}m cells (total: {time.time() - t_total_start:.3f}s)")
    
    def _initialize_fire_fuel_from_terrain(self):
        """Initialize fire fuel levels based on terrain zones."""
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
                
                # FIRST: Check if position is inside a building
                in_building = False
                for obstacle in self.obstacles:
                    if obstacle['type'] == 'city_block':
                        bounds = obstacle['bounds']
                        if (bounds['min'][0] <= world_pos[0] <= bounds['max'][0] and
                            bounds['min'][1] <= world_pos[1] <= bounds['max'][1]):
                            fuel_level = 1.0
                            burn_rate = 0.0001  # Buildings burn very slowly
                            in_building = True
                            terrain_type = 'building'
                            break
                
                # Only check terrain zones if not in a building
                # PRIORITY ORDER: Water > Forest > Grass
                # Water should override everything (fire break)
                if not in_building:
                    # First pass: Check for water (HIGHEST PRIORITY for terrain)
                    in_water = False
                    for zone in self.terrain_zones:
                        if zone['type'] == 'lake':
                            # Check if it's a rectangle or circle
                            if zone.get('shape') == 'rectangle':
                                # Point-in-polygon test for rotated rectangle
                                point = [world_pos[0], world_pos[1]]
                                corners = zone['corners']
                                
                                # Use ray casting algorithm for point-in-polygon
                                inside = False
                                n = len(corners)
                                for k in range(n):
                                    k_next = (k + 1) % n
                                    xi, yi = corners[k]
                                    xj, yj = corners[k_next]
                                    
                                    if ((yi > point[1]) != (yj > point[1])) and \
                                       (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi):
                                        inside = not inside
                                
                                if inside:
                                    fuel_level = 0.0  # No fuel in water
                                    burn_rate = 0.0   # Water doesn't burn
                                    terrain_type = 'lake'
                                    in_water = True
                                    break
                            else:
                                # Original circular lake
                                center = zone['center']
                                radius = zone['radius']
                                distance = np.sqrt((world_pos[0] - center[0])**2 + 
                                                 (world_pos[1] - center[1])**2)
                                if distance <= radius:
                                    fuel_level = 0.0  # No fuel in water
                                    burn_rate = 0.0   # Water doesn't burn
                                    terrain_type = 'lake'
                                    in_water = True
                                    break
                        
                        if in_water:
                            break
                    
                    # Second pass: Check for forest (ONLY if not in water)
                    if not in_water:
                        for zone in self.terrain_zones:
                            if zone['type'] == 'forest':
                                center = zone['center']
                                radius = zone['radius']
                                distance = np.sqrt((world_pos[0] - center[0])**2 + 
                                                 (world_pos[1] - center[1])**2)
                                if distance <= radius:
                                    fuel_level = 0.8  # High fuel in forests
                                    burn_rate = 0.03  # Forest burns slowly
                                    terrain_type = 'forest'
                                    break
                
                self.fire_grid.F[i, j] = fuel_level
                self.fire_grid.fuel_burn_rate[i, j] = burn_rate
                debug_counts[terrain_type] += 1
        
        print(f" Fire fuel levels initialized from terrain:")
        print(f"   Buildings: {debug_counts['building']} cells")
        print(f"   Forests: {debug_counts['forest']} cells")
        print(f"   Lakes: {debug_counts['lake']} cells")
        print(f"   Open terrain: {debug_counts['default']} cells")
    
    # ============================================================================
    # FIRE SIMULATION - RUNTIME CONTROL
    # ============================================================================
    
    def start_fire_at_position(self, world_pos, intensity=0.2):
        """
        Start a fire at a specific world position.
        
        Args:
            world_pos: (x, y) position in world coordinates
            intensity: Initial fire intensity (used for visualization only)
        """
        i, j = self.grid_mapper.world_to_cell(world_pos)
        
        # LAZY FUEL LOADING: Load fuel for starting cell
        if self.fire_grid.lazy_fuel_enabled:
            fuel_level, burn_rate = self.fire_grid._get_fuel_at_cell(i, j)
            self.fire_grid.F[i, j] = fuel_level
            self.fire_grid.fuel_burn_rate[i, j] = burn_rate
        
        if self.fire_grid.F[i, j] > 0:  # Only if there's fuel
            self.fire_grid.B[i, j] = True
            self.fire_grid.I[i, j] = np.minimum(1.0, self.fire_grid.F[i, j])
            print(f"✅ Fire started at world pos {world_pos} -> cell ({i}, {j})")
            return True
        else:
            print(f"❌ Cannot start fire at {world_pos} - no fuel")
            return False
    
    def update_fire_simulation(self, suppression_assignments=None, water_drops=None, real_dt=None):
        """
        Update the fire simulation by one step.
        
        Args:
            suppression_assignments: Dict mapping (i,j) to list of suppression probabilities (deprecated)
            water_drops: Dict mapping (i,j) to water amount (0.0 to 1.0+)
            real_dt: Real time step in seconds (if None, uses fire_grid.dt)
        """
        # Accumulate time - only update fire grid when enough time has passed
        if real_dt is None:
            real_dt = self.fire_grid.dt
        
        self.fire_time_accumulator += real_dt
        
        # Only update fire if accumulated time >= fire grid timestep
        if self.fire_time_accumulator < self.fire_grid.dt:
            return  # Skip this update - not enough time has passed
        
        # Reset accumulator (keep remainder for next time)
        self.fire_time_accumulator -= self.fire_grid.dt
        
        # Update wind dynamics (gradual changes over time)
        self._update_wind_dynamics(dt=1)
        
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
        return {
            'fire_grid_state': self.fire_grid.get_state(),
            'fire_stats': self.fire_grid.get_stats(),
            'grid_bounds': self.grid_mapper.get_grid_bounds(),
            'cell_size': self.grid_mapper.cell_size_m
        }
    
    # ============================================================================
    # VISUALIZATION & DEBUGGING
    # ============================================================================
    
    def visualize_fire_in_simulation(self):
        """Create visual objects for fire in PyBullet simulation."""
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
    
    def save_environment_map(self, filename="environment_map.png", show_fire_grid=True, 
                           detailed=False):
        """
        Save a top-down visualization of the environment showing:
        - Buildings (gray rectangles)
        - Forests (green circles)
        - Water bodies (blue circles)
        - Fire grid overlay (fuel levels as color map)
        
        Args:
            filename: Output filename
            show_fire_grid: Whether to show the fire grid fuel levels
            detailed: If True, draw individual buildings/forests (SLOW for large areas).
                     If False, only show fire grid colors (FAST).
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        
        print(f"   Creating visualization (detailed={detailed})...")
        
        # --- 1. Show fire grid if enabled and available ---
        if show_fire_grid and self.fire_grid is not None:
            # Get fire grid bounds and fuel data
            x_min, x_max, y_min, y_max = self.grid_mapper.get_grid_bounds()
            fuel_map = self.fire_grid.fuel_burn_rate
            
            # Create colored fuel map - OPTIMIZED VERSION
            H, W = fuel_map.shape
            colored_fuel = np.zeros((H, W, 4))  # RGBA
            
            # Vectorized fuel coloring (NO LOOPS - much faster!)
            # Default: grass/open (light yellow-green)
            colored_fuel[:, :] = [0.8, 0.9, 0.6, 1.0]
            
            # Zero fuel mask - could be buildings OR water
            zero_mask = (fuel_map == 0.0)
            
            # First, mark all zero-fuel as gray (buildings)
            colored_fuel[zero_mask] = [0.5, 0.5, 0.5, 1.0]
            
            # Now check which zero-fuel cells are actually water (lakes)
            # This requires checking terrain_zones
            for i in range(H):
                for j in range(W):
                    if zero_mask[i, j]:  # Only check zero-fuel cells
                        # Convert cell to world coordinates
                        world_x = (j - W // 2) * self.grid_mapper.cell_size_m
                        world_y = (i - H // 2) * self.grid_mapper.cell_size_m
                        world_pos = [world_x, world_y]
                        
                        # Check if this position is in a lake
                        for zone in self.terrain_zones:
                            if zone['type'] == 'lake':
                                is_in_water = False
                                
                                # Check if it's a rectangle or circle
                                if zone.get('shape') == 'rectangle':
                                    # Point-in-polygon test
                                    corners = zone['corners']
                                    inside = False
                                    n = len(corners)
                                    for k in range(n):
                                        k_next = (k + 1) % n
                                        xi, yi = corners[k]
                                        xj, yj = corners[k_next]
                                        
                                        if ((yi > world_pos[1]) != (yj > world_pos[1])) and \
                                           (world_pos[0] < (xj - xi) * (world_pos[1] - yi) / (yj - yi) + xi):
                                            inside = not inside
                                    
                                    is_in_water = inside
                                else:
                                    # Original circular lake
                                    center = zone['center']
                                    radius = zone['radius']
                                    distance = np.sqrt((world_pos[0] - center[0])**2 + 
                                                     (world_pos[1] - center[1])**2)
                                    is_in_water = (distance <= radius)
                                
                                if is_in_water:
                                    # It's water! Color it blue
                                    colored_fuel[i, j] = [0.2, 0.5, 0.9, 1.0]  # Blue for water
                                    break
            
            # High fuel: dark green (forests)
            forest_mask = (fuel_map >= 0.05)
            colored_fuel[forest_mask] = [0.2, 0.5, 0.15, 1.0]
            
            # Display the fuel map
            ax.imshow(colored_fuel, extent=[x_min, x_max, y_min, y_max], 
                     origin='lower', interpolation='nearest', alpha=0.8)
        
        # --- 2. Draw individual objects (ONLY if detailed mode - can be slow) ---
        if detailed:
            # Instead of drawing ALL buildings, sample a representative set
            if len(self.obstacles) > 500:
                print(f"   Sampling {min(500, len(self.obstacles))} of {len(self.obstacles)} buildings (too many to draw all)...")
                # Sample evenly distributed buildings
                step = max(1, len(self.obstacles) // 500)
                sampled_obstacles = self.obstacles[::step][:500]
            else:
                print(f"   Drawing {len(self.obstacles)} buildings...")
                sampled_obstacles = self.obstacles
            
            for obstacle in sampled_obstacles:
                if obstacle['type'] == 'city_block':
                    pos = obstacle['position']
                    size = obstacle['size']
                    
                    rect = mpatches.Rectangle(
                        (pos[0] - size[0]/2, pos[1] - size[1]/2),
                        size[0], size[1],
                        linewidth=0.3, edgecolor='black', facecolor='darkgray', alpha=0.5
                    )
                    ax.add_patch(rect)
            
            # Draw forests (limit if too many)
            forest_zones = [z for z in self.terrain_zones if z['type'] == 'forest']
            if len(forest_zones) > 200:
                print(f"   Sampling {min(200, len(forest_zones))} of {len(forest_zones)} forests...")
                step = max(1, len(forest_zones) // 200)
                sampled_forests = forest_zones[::step][:200]
            else:
                print(f"   Drawing {len(forest_zones)} forests...")
                sampled_forests = forest_zones
            
            for zone in sampled_forests:
                center = zone['center']
                radius = zone['radius']
                
                circle = mpatches.Circle(
                    center, radius,
                    linewidth=0.3, edgecolor='darkgreen', facecolor='green', alpha=0.3
                )
                ax.add_patch(circle)
            
            # Draw water bodies (limit if too many)
            lake_zones = [z for z in self.terrain_zones if z['type'] == 'lake']
            if len(lake_zones) > 100:
                print(f"   Sampling {min(100, len(lake_zones))} of {len(lake_zones)} water bodies...")
                step = max(1, len(lake_zones) // 100)
                sampled_lakes = lake_zones[::step][:100]
            else:
                print(f"   Drawing {len(lake_zones)} water bodies...")
                sampled_lakes = lake_zones
            
            for zone in sampled_lakes:
                if zone.get('shape') == 'rectangle':
                    # Draw rectangle for rivers
                    from matplotlib.patches import Polygon
                    corners = zone['corners']
                    poly = Polygon(corners, linewidth=0.3, edgecolor='darkblue', 
                                  facecolor='blue', alpha=0.4)
                    ax.add_patch(poly)
                else:
                    # Draw circle for lakes
                    center = zone['center']
                    radius = zone['radius']
                    circle = mpatches.Circle(
                        center, radius,
                        linewidth=0.3, edgecolor='darkblue', facecolor='blue', alpha=0.4
                    )
                    ax.add_patch(circle)
        
        # --- 3. Set up the plot ---
        if self.fire_grid is not None:
            x_min, x_max, y_min, y_max = self.grid_mapper.get_grid_bounds()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            # Auto-scale to show all objects
            ax.autoscale()
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title('Environment Map (Top-Down View)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='darkgray', edgecolor='black', label='Buildings'),
            mpatches.Patch(facecolor='green', alpha=0.4, edgecolor='darkgreen', label='Forests'),
            mpatches.Patch(facecolor='blue', alpha=0.6, edgecolor='darkblue', label='Water'),
        ]
        
        if show_fire_grid and self.fire_grid is not None:
            legend_elements.extend([
                mpatches.Patch(facecolor=[0.8, 0.9, 0.6, 1.0], label='Grass/Open (Low Fuel)'),
                mpatches.Patch(facecolor=[0.4, 0.6, 0.1, 1.0], label='Forest (High Fuel)')
            ])
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add info text
        info_text = f"Buildings: {len(self.obstacles)}\n"
        forests = sum(1 for z in self.terrain_zones if z['type'] == 'forest')
        lakes = sum(1 for z in self.terrain_zones if z['type'] == 'lake')
        info_text += f"Forest areas: {forests}\n"
        info_text += f"Water bodies: {lakes}"
        
        if self.fire_grid is not None:
            H, W = self.fire_grid.H, self.fire_grid.W
            info_text += f"\nFire grid: {H}×{W} cells"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Environment map saved to: {filename}")