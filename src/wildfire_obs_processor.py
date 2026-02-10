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

class WildfireObsProcessor:
    """
    Modular middleware to process simulation state into 
    standardized neural network inputs (CNN + MLP).
    """
    def __init__(self, 
                 window_size_m: float = 40.0, 
                 resolution_px: int = 32,
                 lidar_rays: int = 8,
                 lidar_dist: float = 50.0):
        self.window_size_m = window_size_m
        self.res_px = resolution_px
        self.lidar_rays = lidar_rays
        self.lidar_dist = lidar_dist

    def get_local_fire_map(self, sim, drone_name) -> np.ndarray:
        """
        Extracts a meter-based local crop of the fire intensity grid.
        Returns: (1, 32, 32) float32 array normalized [0, 1].
        """
        drone = sim.drones[drone_name]
        pos = drone.get_position()
        mapper = sim.environment.grid_mapper
        fire_grid = sim.environment.fire_grid.I # Intensity layer

        # 1. Calculate the bounding box in meters
        half_w = self.window_size_m / 2
        min_world = (pos[0] - half_w, pos[1] - half_w)
        max_world = (pos[0] + half_w, pos[1] + half_w)

        # 2. Map world bounds to grid indices
        r_min, c_min = mapper.world_to_cell(min_world)
        r_max, c_max = mapper.world_to_cell(max_world)

        # 3. Extract the crop
        # We add 1 to max to ensure we get the full range
        # Note: In a production environment, you should add boundary checks here
        # to pad with zeros if the drone flies off the map.
        crop = fire_grid[r_min:r_max+1, c_min:c_max+1]

        # 4. Standardize resolution to fixed pixels (32x32)
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

        # Vector state (Self-State)
        # [z, vx, vy, vz, roll, pitch, yaw_rate, water_level]
        # We omit X,Y world coords for better generalization
        self_state = np.array([
            pos[2],                    # Altitude
            vel[0], vel[1], vel[2],    # World velocities
            rpy[0], rpy[1],            # Attitude
            water_norm,                # Water level
        ], dtype=np.float32)

        return {
            "local_map": self.get_local_fire_map(sim, drone_name),
            "self_state": self_state,
            "lidar": self.get_lidar_data(sim, drone_name)
        }