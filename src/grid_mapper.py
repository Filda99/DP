"""
Grid Mapper for FireGrid Integration

Maps between PyBullet world coordinates and FireGrid cell indices.
Handles coordinate transformations and visualization support.
"""

import numpy as np
from typing import Tuple, List


class GridMapper:
    """
    Helper class for mapping between PyBullet world coordinates and FireGrid indices.
    
    The grid is centered at the origin of the world coordinate system.
    """
    
    def __init__(self, grid_width_m: float, grid_height_m: float, cell_size_m: float):
        """
        Initialize the grid mapper.
        
        Args:
            grid_width_m (float): Total width of the grid in meters
            grid_height_m (float): Total height of the grid in meters  
            cell_size_m (float): Size of each cell in meters
        """
        self.grid_width_m = grid_width_m
        self.grid_height_m = grid_height_m
        self.cell_size_m = cell_size_m
        
        # Calculate grid dimensions in cells
        self.grid_width_cells = int(np.ceil(grid_width_m / cell_size_m))
        self.grid_height_cells = int(np.ceil(grid_height_m / cell_size_m))
        
        # Calculate actual grid size (may be slightly larger due to integer cells)
        self.actual_width_m = self.grid_width_cells * cell_size_m
        self.actual_height_m = self.grid_height_cells * cell_size_m
        
        # Grid origin (bottom-left corner) in world coordinates
        self.origin_x = -self.actual_width_m / 2
        self.origin_y = -self.actual_height_m / 2
        
        print(f"✅ GridMapper initialized:")
        print(f"   Grid size: {self.grid_width_cells}x{self.grid_height_cells} cells")
        print(f"   Cell size: {cell_size_m}m")
        print(f"   World bounds: [{self.origin_x:.1f}, {-self.origin_x:.1f}] x [{self.origin_y:.1f}, {-self.origin_y:.1f}]")
    
    def world_to_cell(self, pos_xy: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell indices.
        
        Args:
            pos_xy: (x, y) position in world coordinates
            
        Returns:
            (i, j) cell indices (row, column) with clipping to grid bounds
        """
        x, y = pos_xy
        
        # Convert to grid coordinates (relative to origin)
        grid_x = x - self.origin_x
        grid_y = y - self.origin_y
        
        # Convert to cell indices
        j = int(np.floor(grid_x / self.cell_size_m))  # Column index
        i = int(np.floor(grid_y / self.cell_size_m))  # Row index
        
        # Clip to grid bounds
        i = max(0, min(self.grid_height_cells - 1, i))
        j = max(0, min(self.grid_width_cells - 1, j))
        
        return (i, j)
    
    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """
        Convert grid cell indices to world coordinates (cell center).
        
        Args:
            i: Row index
            j: Column index
            
        Returns:
            (x, y) center coordinates of the cell in world space
        """
        # Clamp indices to valid range
        i = max(0, min(self.grid_height_cells - 1, i))
        j = max(0, min(self.grid_width_cells - 1, j))
        
        # Calculate center coordinates
        x = self.origin_x + (j + 0.5) * self.cell_size_m
        y = self.origin_y + (i + 0.5) * self.cell_size_m
        
        return (x, y)
    
    def get_all_cell_centers(self) -> List[Tuple[float, float]]:
        """
        Get world coordinates for all cell centers.
        
        Returns:
            List of (x, y) coordinates for all cell centers
        """
        centers = []
        for i in range(self.grid_height_cells):
            for j in range(self.grid_width_cells):
                centers.append(self.cell_to_world(i, j))
        return centers
    
    def get_cell_corners(self, i: int, j: int) -> List[Tuple[float, float]]:
        """
        Get the four corner coordinates of a specific cell.
        
        Args:
            i: Row index
            j: Column index
            
        Returns:
            List of (x, y) coordinates for cell corners [bottom-left, bottom-right, top-right, top-left]
        """
        # Clamp indices
        i = max(0, min(self.grid_height_cells - 1, i))
        j = max(0, min(self.grid_width_cells - 1, j))
        
        # Calculate corner coordinates
        x_min = self.origin_x + j * self.cell_size_m
        x_max = self.origin_x + (j + 1) * self.cell_size_m
        y_min = self.origin_y + i * self.cell_size_m
        y_max = self.origin_y + (i + 1) * self.cell_size_m
        
        return [
            (x_min, y_min),  # Bottom-left
            (x_max, y_min),  # Bottom-right
            (x_max, y_max),  # Top-right
            (x_min, y_max)   # Top-left
        ]
    
    def is_position_in_bounds(self, pos_xy: Tuple[float, float]) -> bool:
        """
        Check if a world position is within the grid bounds.
        
        Args:
            pos_xy: (x, y) position in world coordinates
            
        Returns:
            True if position is within grid bounds
        """
        x, y = pos_xy
        return (self.origin_x <= x <= -self.origin_x and 
                self.origin_y <= y <= -self.origin_y)
    
    def get_grid_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the world coordinate bounds of the grid.
        
        Returns:
            (x_min, x_max, y_min, y_max) in world coordinates
        """
        print(f"Grid bounds: x[{self.origin_x}, {-self.origin_x}], y[{self.origin_y}, {-self.origin_y}]")
        return (self.origin_x, -self.origin_x, self.origin_y, -self.origin_y)
    
    def get_grid_dimensions(self) -> Tuple[int, int]:
        """
        Get the grid dimensions in cells.
        
        Returns:
            (height, width) in number of cells
        """
        return (self.grid_height_cells, self.grid_width_cells)