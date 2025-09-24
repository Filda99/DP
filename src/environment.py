"""
Environment System

Creates environmental features like obstacles, terrain, weather effects.
Manages cities, forests, wind, and other environmental factors.
"""

import pybullet as p
import numpy as np
import random


class Environment:
    """Environment system with obstacles, terrain, and weather."""
    
    def __init__(self):
        """Initialize environment."""
        self.obstacles = []
        self.terrain_zones = []
        self.weather = {
            'wind_velocity': np.array([0.0, 0.0, 0.0]),
            'wind_turbulence': 0.0,
            'visibility': 1000.0,  # meters
            'precipitation': 0.0   # 0.0 = none, 1.0 = heavy
        }
        
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
        
        # Add turbulence
        if self.weather['wind_turbulence'] > 0:
            turbulence = np.random.normal(0, self.weather['wind_turbulence'], 3)
            base_wind += turbulence
        
        # Height effect - wind increases with altitude
        height_factor = 1.0 + position[2] * 0.05
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