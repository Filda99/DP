import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class FireGrid:
    """
    Simulates wildfire spread on a 2D grid using NumPy arrays.
    
    Fire spread is modeled using probabilistic rules that account for:
    - Distance-based decay
    - Wind effects
    - Slope effects
    - Fuel availability
    - Suppression activities
    
    Based on the MDP formulation from Griffith et al. (2017)
    """
    
    # ============================================================================
    # FUEL TYPE CONSTANTS
    # ============================================================================
    
    # Fuel properties: (fuel_level, burn_rate)
    FUEL_WATER = (0.0, 0.0)  # Water - no fuel, no burning
    FUEL_BUILDING = (0.9, 0.0001)  # Buildings - high fuel, very slow burn
    FUEL_FOREST = (0.8, 0.03)  # Forest - high fuel, moderate burn
    FUEL_GRASS = (0.3, 0.08)  # Grassland/open terrain - low fuel, fast burn
    
    # ============================================================================
    # INITIALIZATION & CONFIGURATION
    # ============================================================================
    
    def __init__(self, H: int, W: int, dt: float = 0.1, alpha: float = 1.0,
                 k_wind: float = 1.0, k_slope: float = 1.0, wind_dir: float = 0.0,
                 l_base: Optional[np.ndarray] = None):
        """
        Initialize the FireGrid.
        
        Args:
            H (int): Height of the grid
            W (int): Width of the grid
            dt (float): Time step for simulation
            alpha (float): Distance decay factor
            k_wind (float): Wind influence factor
            k_slope (float): Slope influence factor
            wind_dir (float): Wind direction in radians
            l_base (np.ndarray): Base lambda values for each row (if None, defaults to ones)
        """
        self.H = H
        self.W = W
        self.dt = dt
        self.alpha = alpha
        self.k_wind = k_wind
        self.k_slope = k_slope
        self.wind_dir = wind_dir
        
        # Base lambda values (one per row)
        if l_base is None:
            self.l_base = np.ones(H)
        else:
            assert len(l_base) == H, f"l_base must have length {H}"
            self.l_base = np.array(l_base)
        
        # Initialize state arrays
        self.reset_random()
        
        # Lazy fuel loading
        self.fuel_cache = {}  # Cache for already-loaded fuel values
        self.environment = None  # Will be set later for lazy loading
        self.grid_mapper = None
        self.lazy_fuel_enabled = False  # Whether to use lazy loading
    
    def reset_random(self, seed: Optional[int] = None) -> None:
        """Reset the grid to a random initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Burning flag (bool array)
        self.B = np.zeros((self.H, self.W), dtype=bool)
        
        # Remaining fuel (0.0 to 1.0)
        self.F = np.random.uniform(0.3, 1.0, (self.H, self.W))
        
        # Fuel burn rate (terrain-dependent, default 0.05)
        self.fuel_burn_rate = np.full((self.H, self.W), 0.05, dtype=float)
        
        # Moisture level (0.0 to 1.0) - reduces ignition probability
        self.M = np.zeros((self.H, self.W), dtype=float)
        
        # Intensity
        self.I = np.zeros((self.H, self.W))
        
        # Start with a few random burning cells
        num_initial_fires = max(1, min(5, (self.H * self.W) // 100))
        for _ in range(num_initial_fires):
            i = np.random.randint(0, self.H)
            j = np.random.randint(0, self.W)
            if self.F[i, j] > 0:
                self.B[i, j] = True
                self.I[i, j] = np.minimum(1.0, self.F[i, j])
    
    def enable_lazy_fuel_loading(self, environment, grid_mapper):
        """Enable lazy fuel loading - fuel values loaded on-demand when fire reaches cells.
        
        Args:
            environment: Environment object with terrain data
            grid_mapper: GridMapper for coordinate conversion
        """
        self.environment = environment
        self.grid_mapper = grid_mapper
        self.lazy_fuel_enabled = True
        print("   ✅ Lazy fuel loading enabled - fuel will be loaded on-demand")
    
    # ============================================================================
    # FUEL MANAGEMENT & TERRAIN INTERACTION
    # ============================================================================
    
    def _get_fuel_at_cell(self, i, j):
        """Get fuel value for cell, loading lazily from environment if needed.
        
        Args:
            i, j: Cell coordinates
            
        Returns:
            tuple: (fuel_level, burn_rate)
        """
        # Guard: If lazy loading disabled, return current values
        if not self.lazy_fuel_enabled:
            return self.F[i, j], self.fuel_burn_rate[i, j]
        
        # Guard: Check cache first
        if (i, j) in self.fuel_cache:
            return self.fuel_cache[(i, j)]
        
        # Delegate to Environment for fuel lookup (single source of truth)
        # Environment handles all terrain logic and spatial indexing
        world_pos = self.grid_mapper.cell_to_world(i, j)
        fuel_properties = self.environment.get_fuel_at_position(world_pos)
        
        # Cache and return
        self.fuel_cache[(i, j)] = fuel_properties
        return fuel_properties
    
    # ============================================================================
    # FIRE SPREAD CALCULATIONS
    # ============================================================================
    
    def _calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate Euclidean distance between two grid points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _calculate_wind_gain(self, from_i: int, from_j: int, to_i: int, to_j: int) -> float:
        """Calculate wind influence on fire spread from one cell to another."""
        if from_i == to_i and from_j == to_j:
            return 1.0
        
        # Direction vector from source to target
        di = to_i - from_i
        dj = to_j - from_j
        spread_dir = np.arctan2(di, dj)
        
        # Angle difference between wind direction and spread direction
        # Used for wind gain calculation (positive when spreading with the wind)
        # e.g., if wind is blowing east (0 radians) and fire spreads east, gain is maximized
        angle_diff = np.abs(spread_dir - self.wind_dir)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Use smaller angle
        
        # Wind gain: higher when spreading with the wind
        wind_gain = 1.0 + self.k_wind * np.cos(angle_diff)
        return max(0.1, wind_gain)  # Ensure positive gain
    
    def _calculate_slope_gain(self, from_i: int, from_j: int, to_i: int, to_j: int) -> float:
        """Calculate slope influence on fire spread (simplified model)."""
        # For simplicity, assume flat terrain with slight uphill bias
        return 1.0 + self.k_slope * 0.1  # Small uphill bias
    
    def _calculate_lambda_xy(self, from_i: int, from_j: int, to_i: int, to_j: int) -> float:
        """Calculate lambda_xy for fire spread from (from_i, from_j) to (to_i, to_j)."""
        if from_i == to_i and from_j == to_j:
            return 0.0
        
        # Distance decay
        d = self._calculate_distance(from_i, from_j, to_i, to_j)
        # Tells us how much the distance reduces the spread rate 
        # because fire spreads less effectively over longer distances.
        # Calculation: -alpha * d gives the decay exponent,
        # and exp(-alpha * d) gives the decay factor which is how much
        # the spread rate is reduced due to distance.
        distance_factor = np.exp(-self.alpha * d)
        
        # Environmental factors
        wind_gain = self._calculate_wind_gain(from_i, from_j, to_i, to_j)
        slope_gain = self._calculate_slope_gain(from_i, from_j, to_i, to_j)
        
        # Base lambda for the target row
        l_base_val = self.l_base[to_i]
        
        return l_base_val * distance_factor * wind_gain * slope_gain
    
    def _calculate_spread_probability(self, from_i: int, from_j: int, to_i: int, to_j: int) -> float:
        """Calculate P_xy = 1 - exp(-lambda_xy * dt)."""
        # Tells us the probability that fire spreads from cell (from_i, from_j)
        # to cell (to_i, to_j) during the time step dt.
        # It is derived from the rate lambda_xy and the time step dt.
        lambda_xy = self._calculate_lambda_xy(from_i, from_j, to_i, to_j)
        return 1.0 - np.exp(-lambda_xy * self.dt)
    
    # ============================================================================
    # IGNITION PROBABILITY CALCULATIONS
    # ============================================================================
    
    def _calculate_ignition_probability(self, target_i: int, target_j: int) -> float:
        """
        Calculate r1(x) = 1 - product(1 - P_xy * B[y]) for all burning neighbors.
        
        This is the probability that at least one burning neighbor ignites the target cell.
        """
        if self.B[target_i, target_j]:  # Already burning
            return 0.0
        
        if self.F[target_i, target_j] <= 0:  # No fuel
            return 0.0
        
        # Consider all cells in a reasonable neighborhood (e.g., 3x3 or 5x5)
        neighborhood_size = 3  # Can be adjusted for performance vs accuracy
        half_size = neighborhood_size // 2
        
        # Calculate bounds
        i_min = max(0, target_i - half_size)
        i_max = min(self.H, target_i + half_size + 1)
        j_min = max(0, target_j - half_size)
        j_max = min(self.W, target_j + half_size + 1)
        
        # Product of (1 - P_xy * B[y]) for all burning neighbors
        product_term = 1.0
        
        for source_i in range(i_min, i_max):
            for source_j in range(j_min, j_max):
                if source_i == target_i and source_j == target_j:
                    continue
                
                if self.B[source_i, source_j]:  # Source is burning
                    P_xy = self._calculate_spread_probability(source_i, source_j, target_i, target_j)
                    product_term *= (1.0 - P_xy)
        
        ignition_prob = 1.0 - product_term
        
        # STRONG moisture reduction: High moisture dramatically reduces ignition
        # - At 50% moisture: ignition reduced by 75% (0.5^2 = 0.25 multiplier)
        # - At 80% moisture: ignition reduced by 96% (0.2^2 = 0.04 multiplier)
        moisture_factor = (1.0 - self.M[target_i, target_j]) ** 2
        ignition_prob *= moisture_factor
        
        return ignition_prob
    
    def _calculate_ignition_probabilities_vectorized(self) -> np.ndarray:
        """
        Calculate ignition probabilities for all cells using vectorized operations where possible.
        Returns array of ignition probabilities for non-burning cells.
        
        OPTIMIZATION: Only check cells within 'burn_radius' of active fires to avoid
        checking all 40,000+ cells when most are far from any fire.
        """
        ignition_probs = np.zeros((self.H, self.W))
        
        # Only calculate for non-burning cells with fuel
        non_burning_mask = ~self.B & (self.F > 0)
        
        # Find all burning cells
        burning_cells = np.argwhere(self.B)
        
        if len(burning_cells) == 0:
            return ignition_probs  # No fires, no ignition
        
        # Define burn radius: fire can only spread ~5-10 cells in reasonable conditions
        # (based on 3x3 neighborhood in _calculate_ignition_probability)
        burn_radius = 10  # cells
        
        # Create active region mask: cells within burn_radius of any fire
        active_region = np.zeros((self.H, self.W), dtype=bool)
        
        for fire_i, fire_j in burning_cells:
            i_min = max(0, fire_i - burn_radius)
            i_max = min(self.H, fire_i + burn_radius + 1)
            j_min = max(0, fire_j - burn_radius)
            j_max = min(self.W, fire_j + burn_radius + 1)
            active_region[i_min:i_max, j_min:j_max] = True
        
        # Only check cells that are:
        # 1. Non-burning with fuel (non_burning_mask)
        # 2. Within burn_radius of active fire (active_region)
        cells_to_check = non_burning_mask & active_region
        
        # for cell_to_check in cells_to_check:
        #     self._calculate_ignition_probability()

        for i in range(self.H):
            for j in range(self.W):
                if cells_to_check[i, j]:
                    ignition_probs[i, j] = self._calculate_ignition_probability(i, j)
        
        return ignition_probs
    
    # ============================================================================
    # SUPPRESSION CALCULATIONS
    # ============================================================================
    
    def _calculate_suppression_probability(self, i: int, j: int, 
                                         suppression_assignments: Dict[Tuple[int, int], List[float]]) -> float:
        """
        Calculate r2(x) = 1 - product(1 - Q_i(x)) for suppression at cell (i, j).
        
        Args:
            i, j: Target cell coordinates
            suppression_assignments: Dict mapping (i,j) to list of suppression probabilities
        
        Returns:
            Probability that at least one suppression effort succeeds
        """
        if (i, j) not in suppression_assignments:
            return 0.0
        
        Q_values = suppression_assignments[(i, j)]
        if not Q_values:
            return 0.0
        
        # Product of (1 - Q_i) for all suppression efforts
        product_term = 1.0
        for Q_i in Q_values:
            product_term *= (1.0 - max(0.0, min(1.0, Q_i)))  # Clamp to [0, 1]
        
        return 1.0 - product_term
    
    # ============================================================================
    # SIMULATION STEPPING
    # ============================================================================
    
    def step(self, suppression_assignments: Optional[Dict[Tuple[int, int], List[float]]] = None,
             water_drops: Optional[Dict[Tuple[int, int], float]] = None) -> None:
        """
        Perform one simulation step with the following update order:
        1. Water application (increase moisture, suppress fire)
        2. Ignition (new fires start)
        3. Suppression (fires are extinguished)
        4. Fuel decrease (burning cells consume fuel at CONSTANT rate)
        5. Burn-out (cells with no fuel stop burning)
        6. Moisture evaporation
        
        Args:
            suppression_assignments: Dict mapping (i,j) to list of suppression probabilities (deprecated)
            water_drops: Dict mapping (i,j) to water amount (0.0 to 1.0+)
        """
        if suppression_assignments is None:
            suppression_assignments = {}
        
        # Create copies for simultaneous updates
        new_B = self.B.copy()
        new_F = self.F.copy()
        new_I = self.I.copy()
        new_M = self.M.copy()
        
        # 0. WATER APPLICATION: Apply water drops to increase moisture and suppress fire
        if water_drops:
            for (i, j), water_amount in water_drops.items():
                # Increase moisture (capped at 1.0)
                new_M[i, j] = min(1.0, new_M[i, j] + water_amount)
                
                # Immediate fire suppression effect
                if new_B[i, j]:
                    # Water reduces intensity immediately
                    new_I[i, j] *= (1.0 - water_amount * 0.8)
                    # Chance to extinguish fire completely
                    if np.random.random() < water_amount:
                        new_B[i, j] = False
        
        # 1. IGNITION: New fires start based on burning neighbors
        ignition_probs = self._calculate_ignition_probabilities_vectorized()
        ignition_random = np.random.random((self.H, self.W))
        
        # Cells ignite if random value < ignition probability
        ignition_mask = (ignition_probs > ignition_random) & (~self.B) & (self.F > 0)
        
        # LAZY FUEL LOADING: Load fuel for cells about to ignite
        if self.lazy_fuel_enabled:
            igniting_cells = np.argwhere(ignition_mask)
            for i, j in igniting_cells:
                fuel_level, burn_rate = self._get_fuel_at_cell(i, j)
                self.F[i, j] = fuel_level
                self.fuel_burn_rate[i, j] = burn_rate
        
        new_B[ignition_mask] = True
        # Set initial intensity for newly ignited cells based on fuel
        new_I[ignition_mask] = np.minimum(1.0, new_F[ignition_mask])
        
        # 2. SUPPRESSION: Fires are extinguished
        suppression_random = np.random.random((self.H, self.W))
        
        for i in range(self.H):
            for j in range(self.W):
                if new_B[i, j]:  # Only burning cells can be suppressed
                    suppression_prob = self._calculate_suppression_probability(i, j, suppression_assignments)
                    if suppression_prob > suppression_random[i, j]:
                        new_B[i, j] = False
                        new_I[i, j] = 0.0
        
        # 3. FUEL DECREASE: Burning cells consume fuel at terrain-dependent rate
        # Use specific burn rate for each cell based on terrain type
        burning_mask = new_B
        
        # Get burn rates for burning cells only
        burn_rates_for_burning_cells = self.fuel_burn_rate[burning_mask]
        new_F[burning_mask] = np.maximum(0.0, new_F[burning_mask] - burn_rates_for_burning_cells * self.dt)
        
        # Update intensity for visualization (combination of remaining fuel and burn rate)
        # Moisture STRONGLY reduces fire intensity
        moisture_intensity_reduction = (1.0 - new_M[burning_mask]) ** 1.5
        new_I[burning_mask] = (np.minimum(1.0, new_F[burning_mask]) * 
                               self.fuel_burn_rate[burning_mask] * 10.0 * 
                               moisture_intensity_reduction)
        
        # 4. BURN-OUT: Cells with no fuel stop burning
        burnout_mask = burning_mask & (new_F <= 0)
        new_B[burnout_mask] = False
        new_I[burnout_mask] = 0.0
        
        # 4b. MOISTURE SUPPRESSION: High moisture can extinguish fires
        # Fires in very wet cells have a chance to be extinguished each step
        wet_burning_mask = burning_mask & (new_M > 0.5)
        extinguish_prob = (new_M[wet_burning_mask] - 0.5) * 0.4  # 0-20% chance per step
        extinguish_random = np.random.random(np.sum(wet_burning_mask))
        extinguished = extinguish_prob > extinguish_random
        
        # Apply extinguishment
        wet_burning_indices = np.argwhere(wet_burning_mask)
        for idx, should_extinguish in zip(wet_burning_indices, extinguished):
            if should_extinguish:
                new_B[idx[0], idx[1]] = False
                new_I[idx[0], idx[1]] = 0.0
        
        # 5. MOISTURE EVAPORATION: Moisture gradually evaporates over time
        # Evaporation is slower in burning areas (steam effect)
        evaporation_rate = np.where(new_B, 0.005, 0.01)  # Half speed in burning areas
        new_M = np.maximum(0.0, new_M - evaporation_rate * self.dt)
        
        # Apply all updates simultaneously
        self.B = new_B
        self.F = new_F
        self.I = new_I
        self.M = new_M
    
    # ============================================================================
    # STATE QUERIES & STATISTICS
    # ============================================================================
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state of the fire grid.
        
        Returns:
            Dict containing copies of the state arrays
        """
        return {
            'B': self.B.copy(),  # Burning flags
            'F': self.F.copy(),  # Fuel levels
            'I': self.I.copy(),  # Intensities (for visualization)
            'M': self.M.copy()   # Moisture levels
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the current fire state."""
        total_cells = self.H * self.W
        burning_cells = np.sum(self.B)
        total_fuel = np.sum(self.F)
        avg_intensity = np.mean(self.I[self.B]) if burning_cells > 0 else 0.0
        
        return {
            'total_cells': total_cells,
            'burning_cells': int(burning_cells),
            'burn_percentage': float(burning_cells / total_cells * 100),
            'total_fuel': float(total_fuel),
            'avg_fuel_per_cell': float(total_fuel / total_cells),
            'avg_intensity': float(avg_intensity),
            'max_intensity': float(np.max(self.I)) if burning_cells > 0 else 0.0
        }