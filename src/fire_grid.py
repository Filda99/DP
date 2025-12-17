import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class FireGrid:
    """
    Simulates wildfire spread on a 2D grid using optimized Bounding Box Vectorization.
    
    Optimization:
    Instead of calculating physics for the entire grid (which is slow for large maps),
    we dynamically detect the 'Active Fire Window' - a rectangular region bounding
    all currently burning cells plus a safety margin. Physics is computed only
    within this window.
    """
    
    # Fuel properties: (fuel_level, burn_rate)
    FUEL_WATER = (0.0, 0.0) 
    FUEL_BUILDING = (0.9, 0.0005)
    FUEL_FOREST = (0.8, 0.03)
    FUEL_GRASS = (0.3, 0.08)
    
    def __init__(self, H: int, W: int, dt: float = 0.1, alpha: float = 1.0,
                 k_wind: float = 1.0, k_slope: float = 1.0, wind_dir: float = 0.0,
                 l_base: Optional[np.ndarray] = None):
        
        self.H = H
        self.W = W
        self.dt = dt
        
        # Physics Parameters
        self.alpha = alpha      # Distance decay
        self.k_wind = k_wind    # Wind strength
        self.k_slope = k_slope  # Slope effect
        self.wind_dir = wind_dir
        
        # Base spread rate (lambda)
        self.l_base = np.ones(H) if l_base is None else np.array(l_base)
        
        # Grid State (H x W)
        self.B = np.zeros((H, W), dtype=bool)    # Burning
        self.F = np.zeros((H, W), dtype=float)   # Fuel
        self.I = np.zeros((H, W), dtype=float)   # Intensity
        self.M = np.zeros((H, W), dtype=float)   # Moisture
        self.fuel_burn_rate = np.zeros((H, W), dtype=float)

        # Lazy Loading hooks
        self.lazy_fuel_enabled = False
        self.environment = None
        self.grid_mapper = None
        self.fuel_cache = {}

    def enable_lazy_fuel_loading(self, environment, grid_mapper):
        """Enable on-demand fuel loading."""
        self.environment = environment
        self.grid_mapper = grid_mapper
        self.lazy_fuel_enabled = True

    def _load_fuel_for_mask(self, mask, r_offset, c_offset):
        """Load fuel from environment for specific cells (Lazy Loading).
        Adjusts global indices based on ROI offset.
        """
        if not self.lazy_fuel_enabled: return
        
        # Indices relative to the ROI slice
        local_indices = np.argwhere(mask)
        if len(local_indices) == 0: return

        # Convert to global coordinates
        global_indices = local_indices + [r_offset, c_offset]
        
        for i, j in global_indices:
            if (i, j) in self.fuel_cache:
                self.F[i, j], self.fuel_burn_rate[i, j] = self.fuel_cache[(i, j)]
                continue
            # Note: In real setup, we would call environment.get_fuel_at(pos) here
            # But typically this is handled by map_importer pre-filling the grid
            pass 

    def step(self, suppression_assignments=None, water_drops=None):
        """
        Perform one simulation step using ROI (Region of Interest) optimization.
        """
        # 1. APPLY WATER (Global or Sparse update)
        if water_drops:
            indices = list(water_drops.keys())
            amounts = list(water_drops.values())
            if indices:
                rows, cols = zip(*indices)
                water_array = np.array(amounts)
                
                # Update Moisture
                self.M[rows, cols] = np.minimum(1.0, self.M[rows, cols] + water_array)
                # Suppress Intensity
                self.I[rows, cols] *= (1.0 - water_array * 0.8)
                # Extinguish
                extinguish_mask = np.random.random(len(water_array)) < water_array
                ext_rows = np.array(rows)[extinguish_mask]
                ext_cols = np.array(cols)[extinguish_mask]
                self.B[ext_rows, ext_cols] = False
                self.I[ext_rows, ext_cols] = 0.0

        # --- OPTIMIZATION: FIND ACTIVE WINDOW ---
        # Find bounds of current fire
        burning_rows, burning_cols = np.where(self.B)
        
        if len(burning_rows) == 0:
            return # No fire, nothing to calculate

        # Define Region of Interest (ROI) with padding
        # Padding allows fire to spread into neighbors
        PAD = 2 
        r_min = max(0, np.min(burning_rows) - PAD)
        r_max = min(self.H, np.max(burning_rows) + 1 + PAD)
        c_min = max(0, np.min(burning_cols) - PAD)
        c_max = min(self.W, np.max(burning_cols) + 1 + PAD)
        
        # EXTRACT SLICES (Views into the main grid)
        # We only work with these small arrays!
        B_roi = self.B[r_min:r_max, c_min:c_max]
        F_roi = self.F[r_min:r_max, c_min:c_max]
        M_roi = self.M[r_min:r_max, c_min:c_max]
        rate_roi = self.fuel_burn_rate[r_min:r_max, c_min:c_max]
        
        # Base spread rates for this slice (needed for rows)
        l_base_roi = self.l_base[r_min:r_max]

        # 2. IGNITION (Vectorized on ROI)
        h_roi, w_roi = B_roi.shape
        ignition_potential = np.zeros((h_roi, w_roi))
        
        # Directions: (di, dj)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        cell_dist = 1.0 
        
        for di, dj in directions:
            # We need to shift the ROI to see neighbors.
            # Instead of np.roll (which wraps), we use slicing on the GLOBAL grid 
            # to get the exact neighbors for the current ROI.
            
            # Target (Center) coords in global grid: r_min:r_max, c_min:c_max
            # Neighbor coords in global grid: r_min-di : r_max-di, c_min-dj : c_max-dj
            
            # Check bounds for neighbor slice
            n_r_min, n_r_max = r_min - di, r_max - di
            n_c_min, n_c_max = c_min - dj, c_max - dj
            
            # If neighbor slice is completely out of bounds, skip
            if n_r_min >= self.H or n_r_max <= 0 or n_c_min >= self.W or n_c_max <= 0:
                continue
                
            # Handle partial overlaps (clipping)
            # This effectively pads with False/Zero where active window touches map edge
            # For simplicity in this optimization, we can just use zero-padding logic
            # or simply perform the calculation only on valid overlaps.
            
            # SIMPLIFIED SHIFT LOGIC FOR ROI:
            # Since we added PAD=2, and we look at distance 1 neighbors, 
            # the ROI is guaranteed to contain the immediate neighbors of any burning cell
            # EXCEPT at the very edges of the ROI.
            # But the edges of ROI are non-burning by definition (because we padded burning_cells).
            # So simple slicing within ROI is safe!
            
            # Shift within ROI using slicing
            # Target: Where we compute ignition
            # Source: The neighbor cell
            
            # We want B_shifted[r, c] to equal B_roi[r-di, c-dj]
            # Valid range for r is where r-di is inside [0, h_roi)
            
            t_r_start = max(0, di)
            t_r_end = min(h_roi, h_roi + di)
            t_c_start = max(0, dj)
            t_c_end = min(w_roi, w_roi + dj)
            
            s_r_start = max(0, -di)
            s_r_end = min(h_roi, h_roi - di)
            s_c_start = max(0, -dj)
            s_c_end = min(w_roi, w_roi - dj)
            
            # Check if valid slice
            if t_r_end <= t_r_start or t_c_end <= t_c_start:
                continue
                
            B_neighbor_slice = B_roi[s_r_start:s_r_end, s_c_start:s_c_end]
            
            # Wind Gain
            spread_angle = np.arctan2(di, dj)
            angle_diff = np.abs(spread_angle - self.wind_dir)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
            wind_gain = 1.0 + self.k_wind * np.cos(angle_diff)
            wind_gain = max(0.1, wind_gain)
            
            # Spread Probability
            # Use l_base corresponding to the target rows
            l_slice = l_base_roi[t_r_start:t_r_end]
            lambda_matrix = l_slice[:, np.newaxis] * np.exp(-self.alpha * cell_dist) * wind_gain
            
            prob_ignite = (1.0 - np.exp(-lambda_matrix * self.dt)) * B_neighbor_slice
            
            # Accumulate to target area in ignition_potential
            ignition_potential[t_r_start:t_r_end, t_c_start:t_c_end] += prob_ignite

        # Final Ignition Check (on ROI)
        ignition_prob = ignition_potential * ((1.0 - M_roi) ** 2)
        
        random_matrix = np.random.random((h_roi, w_roi))
        should_ignite = (ignition_prob > random_matrix) & (~B_roi) & (F_roi > 0)
        
        # Update ROI state
        B_roi[should_ignite] = True
        # Intensity = Fuel (clipped)
        I_roi_new = np.zeros_like(self.I[r_min:r_max, c_min:c_max])
        
        # Only update intensity for newly ignited or currently burning
        # We need to copy the *current* intensity first to preserve physics?
        # Actually, intensity is derived from fuel/burn rate each step.
        
        # 3. FUEL CONSUMPTION & UPDATE (Vectorized on ROI)
        burn_mask = B_roi
        consumed = rate_roi[burn_mask] * self.dt
        F_roi[burn_mask] = np.maximum(0.0, F_roi[burn_mask] - consumed)
        
        # Burnout
        burnout = burn_mask & (F_roi <= 1e-3)
        B_roi[burnout] = False
        
        # Update Intensity
        active_fire = B_roi
        m_factor = (1.0 - M_roi[active_fire]) ** 1.5
        
        # Direct write to grid arrays is not needed because slicing numpy arrays 
        # produces a view (mostly), but to be safe with advanced slicing we write back.
        # However, for simple basic slices, assignment works in place.
        # But 'F_roi[burn_mask] = ...' modifies F_roi in place.
        # Does F_roi view modify self.F? Yes, basic slicing returns a view.
        
        # Update Intensity Grid (Full ROI calculation)
        # Reset intensity in ROI
        I_roi = self.I[r_min:r_max, c_min:c_max]
        I_roi[:] = 0.0
        
        # Set intensity for active fires
        # I = Fuel * Moisture_Factor
        new_intensities = F_roi[active_fire] * m_factor
        I_roi[active_fire] = np.clip(new_intensities, 0.0, 1.0)

        # 4. EVAPORATION (Only on ROI? No, technically everywhere, but water drops are local)
        # To be purely lazy, we only evaporate in ROI. Distant wet cells won't dry, 
        # but that's acceptable approx or we track wet cells separately.
        # For Demo, ROI evaporation is fine.
        evap_rate = np.where(B_roi, 0.005, 0.01)
        M_roi[:] = np.maximum(0.0, M_roi - evap_rate * self.dt)

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'B': self.B.copy(),
            'F': self.F.copy(),
            'I': self.I.copy(),
            'M': self.M.copy()
        }

    def get_stats(self) -> Dict[str, Any]:
        burning = np.sum(self.B)
        total_cells = self.H * self.W
        return {
            'burning_cells': int(burning),
            'burn_percentage': float(burning / total_cells * 100),
            'total_fuel': float(np.sum(self.F)),
            'avg_intensity': float(np.mean(self.I[self.B])) if burning > 0 else 0.0
        }