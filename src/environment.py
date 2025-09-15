import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random

class TerrainType(Enum):
    """Enumeration of different terrain types."""
    OPEN_FIELD = "open_field"
    FOREST = "forest"
    LAKE = "lake"
    RIVER = "river"
    MOUNTAIN = "mountain"
    URBAN = "urban"
    NO_FLY_ZONE = "no_fly_zone"

class TerrainZone:
    """Represents a terrain zone with specific properties."""
    
    def __init__(self, 
                 terrain_type: TerrainType,
                 bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
                 altitude_restriction: Optional[Tuple[float, float]] = None,  # (min_alt, max_alt)
                 speed_modifier: float = 1.0,
                 avoidance_priority: int = 0):
        """
        Initialize a terrain zone.
        
        Args:
            terrain_type: Type of terrain
            bounds: (x_min, x_max, y_min, y_max) boundaries of the zone
            altitude_restriction: Optional altitude limits (min, max) in meters
            speed_modifier: Speed multiplier for this terrain (1.0 = normal speed)
            avoidance_priority: Higher values mean drones should avoid this area more
        """
        self.terrain_type = terrain_type
        self.bounds = bounds
        self.altitude_restriction = altitude_restriction
        self.speed_modifier = speed_modifier
        self.avoidance_priority = avoidance_priority
        
        # Terrain-specific properties
        self.color = self._get_terrain_color()
        self.alpha = self._get_terrain_alpha()
        
    def _get_terrain_color(self) -> str:
        """Get the color associated with this terrain type."""
        color_map = {
            TerrainType.OPEN_FIELD: 'lightgreen',
            TerrainType.FOREST: 'darkgreen',
            TerrainType.LAKE: 'lightblue',
            TerrainType.RIVER: 'blue',
            TerrainType.MOUNTAIN: 'gray',
            TerrainType.URBAN: 'lightgray',
            TerrainType.NO_FLY_ZONE: 'red'
        }
        return color_map.get(self.terrain_type, 'white')
    
    def _get_terrain_alpha(self) -> float:
        """Get the transparency level for this terrain type."""
        alpha_map = {
            TerrainType.OPEN_FIELD: 0.2,
            TerrainType.FOREST: 0.4,
            TerrainType.LAKE: 0.6,
            TerrainType.RIVER: 0.5,
            TerrainType.MOUNTAIN: 0.3,
            TerrainType.URBAN: 0.3,
            TerrainType.NO_FLY_ZONE: 0.7
        }
        return alpha_map.get(self.terrain_type, 0.2)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this terrain zone."""
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def get_flight_constraints(self) -> Dict:
        """Get flight constraints for this terrain."""
        constraints = {
            'can_fly': True,
            'min_altitude': 0.0,
            'max_altitude': 1000.0,
            'speed_modifier': self.speed_modifier,
            'avoidance_cost': self.avoidance_priority
        }
        
        # Terrain-specific constraints
        if self.terrain_type == TerrainType.NO_FLY_ZONE:
            constraints['can_fly'] = False
            constraints['avoidance_cost'] = 1000
        elif self.terrain_type == TerrainType.FOREST:
            constraints['min_altitude'] = 20.0  # Trees are tall
            constraints['speed_modifier'] = 0.8  # Slower due to turbulence
        elif self.terrain_type == TerrainType.MOUNTAIN:
            constraints['min_altitude'] = 50.0  # High terrain
            constraints['speed_modifier'] = 0.7  # Slower due to wind
        elif self.terrain_type == TerrainType.LAKE:
            constraints['min_altitude'] = 5.0   # Low flight over water allowed
            constraints['speed_modifier'] = 1.1  # Slightly faster over water
        elif self.terrain_type == TerrainType.URBAN:
            constraints['min_altitude'] = 30.0  # Buildings
            constraints['avoidance_cost'] = 5   # Prefer to avoid populated areas
        
        if self.altitude_restriction:
            constraints['min_altitude'] = max(constraints['min_altitude'], self.altitude_restriction[0])
            constraints['max_altitude'] = min(constraints['max_altitude'], self.altitude_restriction[1])
        
        return constraints

class Environment:
    """Enhanced environment with terrain features."""
    
    def __init__(self, 
                 bounds: Tuple[float, float, float, float] = (-10, 100, -10, 100),
                 name: str = "Drone Environment"):
        """
        Initialize the environment.
        
        Args:
            bounds: (x_min, x_max, y_min, y_max) overall environment boundaries
            name: Name of the environment
        """
        self.bounds = bounds
        self.name = name
        self.terrain_zones: List[TerrainZone] = []
        self.weather_conditions = {
            'wind_speed': 0.0,      # m/s
            'wind_direction': 0.0,  # degrees
            'visibility': 1000.0,   # meters
            'precipitation': 0.0    # 0.0 = none, 1.0 = heavy
        }
    
    def add_terrain_zone(self, terrain_zone: TerrainZone):
        """Add a terrain zone to the environment."""
        self.terrain_zones.append(terrain_zone)
    
    def generate_natural_environment(self):
        """Generate a natural environment with various terrain features."""
        x_min, x_max, y_min, y_max = self.bounds
        width = x_max - x_min
        height = y_max - y_min
        
        # Clear existing terrain
        self.terrain_zones.clear()
        
        # Add a large forest area
        forest_x = x_min + width * 0.2
        forest_y = y_min + height * 0.3
        forest_w = width * 0.3
        forest_h = height * 0.4
        self.add_terrain_zone(TerrainZone(
            TerrainType.FOREST,
            (forest_x, forest_x + forest_w, forest_y, forest_y + forest_h),
            speed_modifier=0.8,
            avoidance_priority=2
        ))
        
        # Add a lake
        lake_x = x_min + width * 0.6
        lake_y = y_min + height * 0.1
        lake_w = width * 0.25
        lake_h = height * 0.3
        self.add_terrain_zone(TerrainZone(
            TerrainType.LAKE,
            (lake_x, lake_x + lake_w, lake_y, lake_y + lake_h),
            speed_modifier=1.1
        ))
        
        # Add a river connecting through the environment
        river_width = width * 0.05
        self.add_terrain_zone(TerrainZone(
            TerrainType.RIVER,
            (x_min + width * 0.45, x_min + width * 0.45 + river_width, y_min, y_max),
            speed_modifier=1.05
        ))
        
        # Add some open fields
        field1_x = x_min + width * 0.05
        field1_y = y_min + width * 0.05
        field1_w = width * 0.15
        field1_h = height * 0.25
        self.add_terrain_zone(TerrainZone(
            TerrainType.OPEN_FIELD,
            (field1_x, field1_x + field1_w, field1_y, field1_y + field1_h),
            speed_modifier=1.0
        ))
        
        # Add a no-fly zone (restricted airspace)
        nfz_x = x_min + width * 0.75
        nfz_y = y_min + height * 0.6
        nfz_w = width * 0.2
        nfz_h = height * 0.25
        self.add_terrain_zone(TerrainZone(
            TerrainType.NO_FLY_ZONE,
            (nfz_x, nfz_x + nfz_w, nfz_y, nfz_y + nfz_h),
            avoidance_priority=1000
        ))
        
        # Add a mountain area
        mountain_x = x_min + width * 0.1
        mountain_y = y_min + height * 0.75
        mountain_w = width * 0.3
        mountain_h = height * 0.2
        self.add_terrain_zone(TerrainZone(
            TerrainType.MOUNTAIN,
            (mountain_x, mountain_x + mountain_w, mountain_y, mountain_y + mountain_h),
            altitude_restriction=(50.0, 1000.0),
            speed_modifier=0.7,
            avoidance_priority=3
        ))
    
    def generate_urban_environment(self):
        """Generate an urban environment with buildings and restricted zones."""
        x_min, x_max, y_min, y_max = self.bounds
        width = x_max - x_min
        height = y_max - y_min
        
        # Clear existing terrain
        self.terrain_zones.clear()
        
        # Add urban areas (buildings)
        for i in range(5):
            for j in range(4):
                if random.random() > 0.3:  # Not all grid cells have buildings
                    building_x = x_min + (i * width / 5) + width * 0.01
                    building_y = y_min + (j * height / 4) + height * 0.01
                    building_w = width / 5 - width * 0.02
                    building_h = height / 4 - height * 0.02
                    
                    self.add_terrain_zone(TerrainZone(
                        TerrainType.URBAN,
                        (building_x, building_x + building_w, building_y, building_y + building_h),
                        altitude_restriction=(30.0, 1000.0),
                        speed_modifier=0.9,
                        avoidance_priority=5
                    ))
        
        # Add a park (open field)
        park_x = x_min + width * 0.35
        park_y = y_min + height * 0.4
        park_w = width * 0.3
        park_h = height * 0.2
        self.add_terrain_zone(TerrainZone(
            TerrainType.OPEN_FIELD,
            (park_x, park_x + park_w, park_y, park_y + park_h),
            speed_modifier=1.0
        ))
    
    def get_terrain_at_position(self, x: float, y: float) -> Optional[TerrainZone]:
        """Get the terrain type at a specific position."""
        # Return the terrain with highest avoidance priority if multiple overlap
        matching_zones = [zone for zone in self.terrain_zones if zone.contains_point(x, y)]
        if matching_zones:
            return max(matching_zones, key=lambda z: z.avoidance_priority)
        return None
    
    def get_flight_constraints_at_position(self, x: float, y: float) -> Dict:
        """Get flight constraints at a specific position."""
        terrain = self.get_terrain_at_position(x, y)
        if terrain:
            return terrain.get_flight_constraints()
        else:
            # Default open airspace
            return {
                'can_fly': True,
                'min_altitude': 0.0,
                'max_altitude': 1000.0,
                'speed_modifier': 1.0,
                'avoidance_cost': 0
            }
    
    def is_position_safe(self, x: float, y: float, altitude: float = 50.0) -> bool:
        """Check if a position is safe for flight."""
        constraints = self.get_flight_constraints_at_position(x, y)
        return (constraints['can_fly'] and 
                constraints['min_altitude'] <= altitude <= constraints['max_altitude'])
    
    def visualize_terrain(self, ax):
        """Add terrain visualization to a matplotlib axis."""
        for zone in self.terrain_zones:
            x_min, x_max, y_min, y_max = zone.bounds
            width = x_max - x_min
            height = y_max - y_min
            
            # Create rectangle for terrain zone
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                facecolor=zone.color,
                alpha=zone.alpha,
                edgecolor='black',
                linewidth=0.5
            )
            # Mark as terrain zone for later identification
            rect._terrain_zone = True
            ax.add_patch(rect)
            
            # Add terrain label
            center_x = x_min + width / 2
            center_y = y_min + height / 2
            label = zone.terrain_type.value.replace('_', ' ').title()
            ax.text(center_x, center_y, label, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=8,
                   weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    def set_weather(self, wind_speed: float = 0.0, wind_direction: float = 0.0, 
                   visibility: float = 1000.0, precipitation: float = 0.0):
        """Set weather conditions."""
        self.weather_conditions = {
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'visibility': visibility,
            'precipitation': precipitation
        }
    
    def get_weather_effects(self) -> Dict:
        """Get weather effects on flight performance."""
        effects = {
            'speed_modifier': 1.0,
            'collision_detection_range': 1.0,
            'avoidance_time_penalty': 0.0
        }
        
        # Wind effects
        if self.weather_conditions['wind_speed'] > 5.0:
            effects['speed_modifier'] *= 0.9
        if self.weather_conditions['wind_speed'] > 10.0:
            effects['speed_modifier'] *= 0.8
        
        # Visibility effects
        if self.weather_conditions['visibility'] < 500.0:
            effects['collision_detection_range'] *= 0.7
        if self.weather_conditions['visibility'] < 200.0:
            effects['collision_detection_range'] *= 0.5
        
        # Precipitation effects
        if self.weather_conditions['precipitation'] > 0.3:
            effects['speed_modifier'] *= 0.85
            effects['avoidance_time_penalty'] += 1.0
        
        return effects
    
    def __str__(self) -> str:
        """String representation of the environment."""
        return f"Environment '{self.name}' with {len(self.terrain_zones)} terrain zones"
