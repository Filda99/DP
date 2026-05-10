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
    
    def __init__(self, H: int, W: int, dt: float = 0.1, alpha: float = 1.0,
                 k_wind: float = 1.0, k_slope: float = 1.0, wind_dir: float = 0.0,
                 l_base: Optional[np.ndarray] = None):
        """Initialise a fire grid.

        Parameters
        ----------
        H, W : int
            Grid dimensions (rows, columns).
        dt : float
            Simulation time step [s].
        alpha : float
            Distance decay for ignition probability.
        k_wind, k_slope : float
            Coefficients for wind and slope influence on spread.
        wind_dir : float
            Global wind direction [rad].
        l_base : array, optional
            Per-row base spread rate (lambda).  Defaults to 1.0 everywhere.
        """
        
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

    def step(self, suppression_assignments=None, water_drops=None):
        """Advance the fire simulation by one time step.

        Algorithm (ROI-optimised):
          1. **Water application** — sparse update of moisture, intensity and
             extinguishment at cells where water was dropped.
          2. **Active window detection** — compute the bounding box of all
             burning cells plus a ``PAD`` margin.  All subsequent physics
             operates only on this Region of Interest (ROI) for performance.
          3. **Ignition** — for each of the 8 neighbours, compute ignition
             probability using distance decay, wind alignment and moisture.
             Stochastic roll determines new ignitions.
          4. **Fuel consumption** — burning cells consume fuel at their
             local ``fuel_burn_rate``.  Cells with no fuel left burn out.
          5. **Intensity update** — ``I = clip(F * (1-M)^1.5, 0, 1)``.
          6. **Evaporation** — moisture slowly decreases.
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

        # ── ROI: bounding box of burning cells + PAD margin ──────────
        # Physics is computed only inside this window, which is typically
        # a small fraction of the full H×W grid.  PAD ensures that
        # fire can spread into unburnt neighbours at the window edge.
        burning_rows, burning_cols = np.where(self.B)
        
        if len(burning_rows) == 0:
            return

        PAD = 10
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

        # ── Ignition: 8-neighbour stochastic spread ──────────────────
        # For each direction (di, dj) we shift the ROI by that offset
        # and accumulate ignition potential from burning neighbours.
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
            # Shift logic: target[r,c] reads source[r-di, c-dj].
            # Computed via slicing rather than np.roll (avoids wrap-around).
            # The PAD guarantees that burning cells always have valid
            # neighbours inside the ROI.
            
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
                
            # Wind-aligned spread probability
            spread_angle = np.arctan2(di, dj)
            angle_diff = np.abs(spread_angle - self.wind_dir)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
            wind_gain = 1.0 + self.k_wind * np.cos(angle_diff)
            wind_gain = max(0.1, wind_gain)
            
            # Spread probability: P = (1 - exp(-λ·dt)) for burning neighbours
            l_slice = l_base_roi[t_r_start:t_r_end]
            lambda_matrix = l_slice[:, np.newaxis] * np.exp(-self.alpha * cell_dist) * wind_gain
            
            prob_ignite = (1.0 - np.exp(-lambda_matrix * self.dt)) * B_neighbor_slice
            
            # Accumulate to target area in ignition_potential
            ignition_potential[t_r_start:t_r_end, t_c_start:t_c_end] += prob_ignite

        # ── Stochastic ignition roll ─────────────────────────────────
        # Moisture reduces ignition probability quadratically.
        ignition_prob = ignition_potential * ((1.0 - M_roi) ** 2)
        
        random_matrix = np.random.random((h_roi, w_roi))
        should_ignite = (ignition_prob > random_matrix) & (~B_roi) & (F_roi > 0)
        B_roi[should_ignite] = True
        # ── Fuel consumption ─────────────────────────────────────────
        burn_mask = B_roi
        consumed = rate_roi[burn_mask] * self.dt
        F_roi[burn_mask] = np.maximum(0.0, F_roi[burn_mask] - consumed)
        
        # ── Burnout ──────────────────────────────────────────────────
        burnout = burn_mask & (F_roi <= 1e-3)
        B_roi[burnout] = False
        
        # ── Intensity = F · (1-M)^1.5, clipped to [0, 1] ────────────
        active_fire = B_roi
        m_factor = (1.0 - M_roi[active_fire]) ** 1.5

        # Basic numpy slicing returns a *view*, so writes to F_roi, B_roi
        # etc. modify self.F, self.B in place.
        I_roi = self.I[r_min:r_max, c_min:c_max]
        I_roi[:] = 0.0
        
        # Set intensity for active fires
        # I = Fuel * Moisture_Factor
        new_intensities = F_roi[active_fire] * m_factor
        I_roi[active_fire] = np.clip(new_intensities, 0.0, 1.0)

        # ── Moisture evaporation ─────────────────────────────────────
        evap_rate = np.where(B_roi, 0.005, 0.01)
        M_roi[:] = np.maximum(0.0, M_roi - evap_rate * self.dt)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return a copy of all grid arrays."""
        return {
            'B': self.B.copy(),
            'F': self.F.copy(),
            'I': self.I.copy(),
            'M': self.M.copy()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics (burning cells, fuel, intensity)."""
        burning = np.sum(self.B)
        total_cells = self.H * self.W
        return {
            'burning_cells': int(burning),
            'burn_percentage': float(burning / total_cells * 100),
            'total_fuel': float(np.sum(self.F)),
            'avg_intensity': float(np.mean(self.I[self.B])) if burning > 0 else 0.0
        }