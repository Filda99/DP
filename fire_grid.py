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
    
    def reset_random(self, seed: Optional[int] = None) -> None:
        """Reset the grid to a random initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Burning flag (bool array)
        self.B = np.zeros((self.H, self.W), dtype=bool)
        
        # Remaining fuel (0.0 to 1.0)
        self.F = np.random.uniform(0.3, 1.0, (self.H, self.W))
        
        # Intensity (for visualization only - not used in fuel consumption)
        self.I = np.zeros((self.H, self.W))
        
        # Start with a few random burning cells
        num_initial_fires = max(1, min(5, (self.H * self.W) // 100))
        for _ in range(num_initial_fires):
            i = np.random.randint(0, self.H)
            j = np.random.randint(0, self.W)
            if self.F[i, j] > 0:
                self.B[i, j] = True
                self.I[i, j] = np.minimum(1.0, self.F[i, j])
    
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
        angle_diff = np.abs(spread_dir - self.wind_dir)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Use smaller angle
        
        # Wind gain: higher when spreading with the wind
        wind_gain = 1.0 + self.k_wind * np.cos(angle_diff)
        return max(0.1, wind_gain)  # Ensure positive gain
    
    def _calculate_slope_gain(self, from_i: int, from_j: int, to_i: int, to_j: int) -> float:
        """Calculate slope influence on fire spread (simplified model)."""
        # For simplicity, assume flat terrain with slight uphill bias
        # TODO: In a real implementation, this would use actual elevation data
        return 1.0 + self.k_slope * 0.1  # Small uphill bias
    
    def _calculate_lambda_xy(self, from_i: int, from_j: int, to_i: int, to_j: int) -> float:
        """Calculate lambda_xy for fire spread from (from_i, from_j) to (to_i, to_j)."""
        if from_i == to_i and from_j == to_j:
            return 0.0
        
        # Distance decay
        d = self._calculate_distance(from_i, from_j, to_i, to_j)
        distance_factor = np.exp(-self.alpha * d)
        
        # Environmental factors
        wind_gain = self._calculate_wind_gain(from_i, from_j, to_i, to_j)
        slope_gain = self._calculate_slope_gain(from_i, from_j, to_i, to_j)
        
        # Base lambda for the target row
        l_base_val = self.l_base[to_i]
        
        return l_base_val * distance_factor * wind_gain * slope_gain
    
    def _calculate_spread_probability(self, from_i: int, from_j: int, to_i: int, to_j: int) -> float:
        """Calculate P_xy = 1 - exp(-lambda_xy * dt)."""
        lambda_xy = self._calculate_lambda_xy(from_i, from_j, to_i, to_j)
        return 1.0 - np.exp(-lambda_xy * self.dt)
    
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
        
        return 1.0 - product_term
    
    def _calculate_ignition_probabilities_vectorized(self) -> np.ndarray:
        """
        Calculate ignition probabilities for all cells using vectorized operations where possible.
        Returns array of ignition probabilities for non-burning cells.
        """
        ignition_probs = np.zeros((self.H, self.W))
        
        # Only calculate for non-burning cells with fuel
        non_burning_mask = ~self.B & (self.F > 0)
        
        for i in range(self.H):
            for j in range(self.W):
                if non_burning_mask[i, j]:
                    ignition_probs[i, j] = self._calculate_ignition_probability(i, j)
        
        return ignition_probs
    
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
    
    def step(self, suppression_assignments: Optional[Dict[Tuple[int, int], List[float]]] = None) -> None:
        """
        Perform one simulation step with the following update order:
        1. Ignition (new fires start)
        2. Suppression (fires are extinguished)
        3. Fuel decrease (burning cells consume fuel at CONSTANT rate)
        4. Burn-out (cells with no fuel stop burning)
        
        Args:
            suppression_assignments: Dict mapping (i,j) to list of suppression probabilities
        """
        if suppression_assignments is None:
            suppression_assignments = {}
        
        # Create copies for simultaneous updates
        new_B = self.B.copy()
        new_F = self.F.copy()
        new_I = self.I.copy()
        
        # 1. IGNITION: New fires start based on burning neighbors
        ignition_probs = self._calculate_ignition_probabilities_vectorized()
        ignition_random = np.random.random((self.H, self.W))
        
        # Cells ignite if random value < ignition probability
        ignition_mask = (ignition_probs > ignition_random) & (~self.B) & (self.F > 0)
        
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
        
        # 3. FUEL DECREASE: Burning cells consume fuel at CONSTANT rate
        # Paper (p.41): "The fuel in an ignited cell decreases at a constant rate"
        # "We can rescale the units of fuel to assume that this rate is one unit"
        burning_mask = new_B
        new_F[burning_mask] = np.maximum(0.0, new_F[burning_mask] - 1.0 * self.dt)
        
        # Update intensity for visualization (based on remaining fuel)
        new_I[burning_mask] = np.minimum(1.0, new_F[burning_mask])
        
        # 4. BURN-OUT: Cells with no fuel stop burning
        burnout_mask = burning_mask & (new_F <= 0)
        new_B[burnout_mask] = False
        new_I[burnout_mask] = 0.0
        
        # Apply all updates simultaneously
        self.B = new_B
        self.F = new_F
        self.I = new_I
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state of the fire grid.
        
        Returns:
            Dict containing copies of the state arrays
        """
        return {
            'B': self.B.copy(),  # Burning flags
            'F': self.F.copy(),  # Fuel levels
            'I': self.I.copy()   # Intensities (for visualization)
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