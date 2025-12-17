import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class FireGrid:
    """
    Simulates wildfire spread on a 2D grid using fully vectorized NumPy operations.
    
    Physics:
    - Cellular Automata approach based on Griffith et al. (2017) MDP formulation.
    - Fully vectorized: Uses array shifting instead of iterating over cells.
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

    def _load_fuel_for_mask(self, mask):
        """Load fuel from environment for specific cells (Lazy Loading)."""
        if not self.lazy_fuel_enabled: return
        
        indices = np.argwhere(mask)
        for i, j in indices:
            if (i, j) in self.fuel_cache:
                self.F[i, j], self.fuel_burn_rate[i, j] = self.fuel_cache[(i, j)]
                continue
            # Note: In real setup, we would call environment.get_fuel_at(pos) here
            pass 

    def step(self, suppression_assignments=None, water_drops=None):
        """
        Perform one simulation step using vectorized physics.
        Order: Water -> Ignition -> Suppression -> Burn -> Evaporation
        """
        # 1. APPLY WATER (Vectorized)
        if water_drops:
            indices = list(water_drops.keys())
            amounts = list(water_drops.values())
            if indices:
                rows, cols = zip(*indices)
                water_array = np.array(amounts)
                
                # Increase moisture
                self.M[rows, cols] = np.minimum(1.0, self.M[rows, cols] + water_array)
                
                # Suppress existing fire intensity
                self.I[rows, cols] *= (1.0 - water_array * 0.8)
                
                # Chance to extinguish
                extinguish_mask = np.random.random(len(water_array)) < water_array
                ext_rows = np.array(rows)[extinguish_mask]
                ext_cols = np.array(cols)[extinguish_mask]
                self.B[ext_rows, ext_cols] = False
                self.I[ext_rows, ext_cols] = 0.0

        # 2. IGNITION (Full Vectorization using Shifts)
        ignition_potential = np.zeros((self.H, self.W))
        
        # Directions: (di, dj)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        cell_dist = 1.0 
        
        for di, dj in directions:
            # Shift the BURNING mask (B) to represent neighbors affecting the center
            B_shifted = np.roll(self.B, shift=(di, dj), axis=(0, 1))
            
            # Mask out wraparound artifacts
            if di == 1: B_shifted[0, :] = False
            elif di == -1: B_shifted[-1, :] = False
            if dj == 1: B_shifted[:, 0] = False
            elif dj == -1: B_shifted[:, -1] = False
            
            # Wind Gain
            spread_angle = np.arctan2(di, dj)
            angle_diff = np.abs(spread_angle - self.wind_dir)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
            wind_gain = 1.0 + self.k_wind * np.cos(angle_diff)
            wind_gain = max(0.1, wind_gain)
            
            # Spread Probability
            # lambda_matrix = base_rate * distance_decay * wind
            lambda_matrix = self.l_base[:, np.newaxis] * np.exp(-self.alpha * cell_dist) * wind_gain
            
            # Prob = 1 - exp(-lambda * dt)
            prob_ignite_neighbor = (1.0 - np.exp(-lambda_matrix * self.dt)) * B_shifted
            
            ignition_potential += prob_ignite_neighbor

        # 2c. Final Ignition Check
        # Moisture reduction: (1-M)^2 -> High moisture blocks ignition
        ignition_prob = ignition_potential * ((1.0 - self.M) ** 2)
        
        random_matrix = np.random.random((self.H, self.W))
        should_ignite = (ignition_prob > random_matrix) & (~self.B) & (self.F > 0)
        
        if self.lazy_fuel_enabled:
            self._load_fuel_for_mask(should_ignite)

        self.B[should_ignite] = True
        # Set initial intensity equal to fuel (clipped to 1.0)
        self.I[should_ignite] = np.minimum(1.0, self.F[should_ignite])

        # 3. SUPPRESSION (Implicit via Moisture/Water logic above)
        
        # 4. FUEL CONSUMPTION & INTENSITY UPDATE
        burn_mask = self.B
        consumed = self.fuel_burn_rate[burn_mask] * self.dt
        self.F[burn_mask] = np.maximum(0.0, self.F[burn_mask] - consumed)
        
        # Burnout (Fuel depleted)
        burnout = burn_mask & (self.F <= 1e-3)
        self.B[burnout] = False
        self.I[burnout] = 0.0
        
        # Update Intensity for visualization
        # I = Fuel * Moisture_Factor.
        # FIX: Removed * 10.0 to prevent clipping issues. Clipped to [0, 1].
        active_fire = self.B
        m_factor = (1.0 - self.M[active_fire]) ** 1.5
        new_intensities = self.F[active_fire] * m_factor
        self.I[active_fire] = np.clip(new_intensities, 0.0, 1.0)

        # 5. EVAPORATION
        evap_rate = np.where(self.B, 0.005, 0.01)
        self.M = np.maximum(0.0, self.M - evap_rate * self.dt)

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