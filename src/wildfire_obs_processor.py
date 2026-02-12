import numpy as np
import torch
from typing import Dict, List, Tuple
# Note: We use simple indexing here, but for production, 
# 'cv2' or 'skimage' is recommended for smooth resampling.
try:
    import cv2
except ImportError:
    cv2 = None
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import ObservationProcessorConfig

class WildfireObsProcessor:
    """
    Modular middleware to process simulation state into 
    standardized neural network inputs (CNN + MLP).
    """
    def __init__(self, 
                 window_size_m: float = ObservationProcessorConfig.DEFAULT_WINDOW_SIZE,
                 resolution_px: int = ObservationProcessorConfig.DEFAULT_RESOLUTION,
                 lidar_rays: int = ObservationProcessorConfig.DEFAULT_LIDAR_RAYS,
                 lidar_dist: float = ObservationProcessorConfig.DEFAULT_LIDAR_DISTANCE):
        self.window_size_m = window_size_m
        self.res_px = resolution_px
        self.lidar_rays = lidar_rays
        self.lidar_dist = lidar_dist

    def is_out_of_bounds(self, pos, map_bounds_x, map_bounds_y) -> bool:
        """Check if drone position is outside map boundaries"""
        return (abs(pos[0]) > map_bounds_x or abs(pos[1]) > map_bounds_y)

    def get_local_fire_map(self, sim, drone_name, boundary_distances=None) -> np.ndarray:
        """
        Extracts a meter-based local crop of the fire intensity grid.
        Returns: (1, 32, 32) float32 array normalized [0, 1].
        """
        drone = sim.drones[drone_name]
        pos = drone.get_position()
        mapper = sim.environment.grid_mapper
        fire_grid = sim.environment.fire_grid.I # Intensity layer

        # Get map bounds for boundary checking
        map_size_x = mapper.width * mapper.resolution
        map_size_y = mapper.height * mapper.resolution
        map_bounds_x = map_size_x / 2
        map_bounds_y = map_size_y / 2

        # Check if drone is out of bounds
        if self.is_out_of_bounds(pos, map_bounds_x, map_bounds_y):
            print(f"Warning: Drone {drone_name} is out of bounds at {pos}")
            return np.zeros((1, self.res_px, self.res_px), dtype=np.float32)

        # 1. Calculate the bounding box in meters
        half_w = self.window_size_m / 2
        min_world = (pos[0] - half_w, pos[1] - half_w)
        max_world = (pos[0] + half_w, pos[1] + half_w)

        # 2. Map world bounds to grid indices
        r_min, c_min = mapper.world_to_cell(min_world)
        r_max, c_max = mapper.world_to_cell(max_world)

        # 3. Clamp indices to valid grid bounds
        r_min = max(0, min(r_min, mapper.height - 1))
        r_max = max(0, min(r_max, mapper.height - 1))
        c_min = max(0, min(c_min, mapper.width - 1))
        c_max = max(0, min(c_max, mapper.width - 1))

        print(f"Drone {drone_name} at {pos} -> Fire map indices: rows [{r_min}:{r_max}], cols [{c_min}:{c_max}]") # Debug: Print the calculated indices for the fire map crop

        # 4. Extract the crop
        # We add 1 to max to ensure we get the full range
        # Note: In a production environment, you should add boundary checks here
        # to pad with zeros if the drone flies off the map.
        crop = fire_grid[r_min:r_max+1, c_min:c_max+1]

        # 5. Standardize resolution to fixed pixels (32x32)
        if crop.size == 0:
             # Safety fallback if crop is empty (off-map)
             processed_map = np.zeros((self.res_px, self.res_px), dtype=np.float32)
        elif cv2:
            processed_map = cv2.resize(crop, (self.res_px, self.res_px), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback: Simple nearest-neighbor if CV2 is missing
            indices_i = np.linspace(0, crop.shape[0]-1, self.res_px).astype(int)
            indices_j = np.linspace(0, crop.shape[1]-1, self.res_px).astype(int)
            processed_map = crop[np.ix_(indices_i, indices_j)]

        return processed_map[np.newaxis, ...].astype(np.float32)

    def get_lidar_data(self, sim, drone_name) -> np.ndarray:
        """
        Simple 2D Lidar: Distance to fire/obstacles in N directions.
        Returns: (8,) array of distances.
        """
        # This can be expanded to use PyBullet's rayTest for real obstacles
        # For now, we simulate 8 directions
        angles = np.linspace(0, 2*np.pi, self.lidar_rays, endpoint=False)
        return np.ones(self.lidar_rays) * self.lidar_dist # Placeholder logic

    def fetch(self, sim, drone_name) -> Dict[str, np.ndarray]:
        """
        Generates the full dictionary of observations for one agent.
        """
        drone = sim.drones[drone_name]
        pos = drone.get_position()
        vel = drone.get_velocity()
        rpy = drone.get_orientation_rpy()

        if drone.water_capacity > 0:
            water_norm = drone.current_water / drone.water_capacity
        else:
            water_norm = 0.0 # Handle 0-capacity drones safely

        # Vector state (Self-State) - ROZŠÍŘENO pro map awareness a exploration tracking
        # [z, vx, vy, vz, roll, pitch, water_level, norm_x, norm_y, dist_to_boundary, exploration_ratio, fire_discovery_ratio]
        # We add world position for map awareness
        
        # Get actual map dimensions from environment
        grid_mapper = sim.environment.grid_mapper
        map_size_x = grid_mapper.width * grid_mapper.resolution  # Real world width in meters
        map_size_y = grid_mapper.height * grid_mapper.resolution  # Real world height in meters
        map_bounds_x = map_size_x / 2  # Assuming map centered at origin
        map_bounds_y = map_size_y / 2

        # Normalized position calculation
        norm_x = pos[0] / map_bounds_x
        norm_y = pos[1] / map_bounds_y
        
        # Distance to each boundary (normalized 0-1, where 1 = at center, 0 = at boundary)
        dist_to_left = (map_bounds_x + pos[0]) / (2 * map_bounds_x)    # Distance to left boundary
        dist_to_right = (map_bounds_x - pos[0]) / (2 * map_bounds_x)   # Distance to right boundary  
        dist_to_bottom = (map_bounds_y + pos[1]) / (2 * map_bounds_y)  # Distance to bottom boundary
        dist_to_top = (map_bounds_y - pos[1]) / (2 * map_bounds_y)     # Distance to top boundary
        boundary_distances = [dist_to_left, dist_to_right, dist_to_bottom, dist_to_top]

        # Check if out of bounds
        is_outside = self.is_out_of_bounds(pos, map_bounds_x, map_bounds_y)
        if is_outside:
            print(f"Warning: Drone {drone_name} is outside map boundaries!")

        print(f"Drone {drone_name} Position: {pos}, Normalized: ({norm_x:.2f}, {norm_y:.2f}), Distances to boundaries: L={dist_to_left:.2f}, R={dist_to_right:.2f}, B={dist_to_bottom:.2f}, T={dist_to_top:.2f}, OutOfBounds: {is_outside}") # Debug: Print position and boundary distances
        
        # Minimum boundary distance (most critical for collision avoidance)
        min_boundary_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        # Exploration progress (will be updated by wrapper)
        exploration_ratio = 0.0  # Placeholder - wrapper will override
        fire_discovery_ratio = 0.0  # Placeholder - wrapper will override
        
        self_state = np.array([
            pos[2],                    # 0: Altitude
            vel[0], vel[1], vel[2],    # 1-3: World velocities
            rpy[0], rpy[1],            # 4-5: Attitude (roll, pitch)
            water_norm,                # 6: Water level
            norm_x, norm_y,            # 7-8: Normalized map position
            dist_to_left, dist_to_right, dist_to_bottom, dist_to_top,  # 9-12: Individual boundary distances
            min_boundary_dist,         # 13: Minimum boundary distance
            exploration_ratio,         # 14: Exploration progress (placeholder)
            fire_discovery_ratio,      # 15: Fire discovery progress (placeholder)
        ], dtype=np.float32)

        print(f"Generated self_state for {drone_name}: {self_state}") # Debug: Print the generated self_state vector

        return {
            "local_map": self.get_local_fire_map(sim, drone_name, boundary_distances),
            "self_state": self_state,
            "lidar": self.get_lidar_data(sim, drone_name)
        }
    
    def get_self_state_size(self):
        """Returns the size of self_state vector for consistent configuration"""
        return ObservationProcessorConfig.SELF_STATE_SIZE