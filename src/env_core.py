# =============================================================================
# env_core.py — Main Multi-Agent Environment
# =============================================================================
# This file implements the DroneFireEnv class, which is a PettingZoo
# ParallelEnv environment.  The environment wraps the PyBullet-based
# Simulation class and exposes the standard RL interface:
#   reset()  → start a new episode, return initial observations
#   step()   → apply one set of actions, return next obs/rewards/done flags
#   state()  → return the global state used by the centralised critic (MAPPO)
#
# Two heterogeneous agent types are supported:
#   quad_*   (quadrotor scouts)     — local map + self-state + neighbour states
#   fixed_*  (fixed-wing commanders) — self-state only (messages come from train.py)
# =============================================================================

import os, sys

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so that `from src.simulation
# import Simulation` works regardless of where the script is launched from.
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import functools
import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv
from src.simulation import Simulation
import random
from reward_config import QUAD, FIXED, SHARED

# Fixed normalisation constant — independent of map size so that networks
# trained on one map generalise to others.  All positions and distances
# in observations are divided by this value instead of map_bounds.
NORM_DIST = 1000.0

class DroneFireEnv(ParallelEnv):
    # PettingZoo requires a metadata dict on every ParallelEnv subclass.
    metadata = {"render_modes": ["human"], "name": "drone_fire_v2"}

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, num_quads=1, num_fixed=1, grid_size_m=2000.0, local_map_size=32, max_steps=500):
        """Initialise the environment configuration.

        Parameters
        ----------
        num_quads : int
            Number of quadrotor scout drones.
        num_fixed : int
            Number of fixed-wing commander aircraft.
        grid_size_m : float
            Side length of the square simulation area in metres.
        local_map_size : int
            Width/height (in pixels) of each scout's local fire map.
        max_steps : int
            Maximum number of RL steps per episode before truncation.
        """
        super().__init__()

        # Episode time limit and map geometry
        self.max_steps = max_steps
        self.num_quads = num_quads
        self.num_fixed = num_fixed
        self.grid_size_m = grid_size_m
        # map_bounds is the coordinate of each edge: x ∈ [-map_bounds, +map_bounds]
        self.map_bounds = self.grid_size_m / 2.0

        # -----------------------------------------------------------------
        # Agent lists
        # PettingZoo distinguishes between possible_agents (all agents that
        # could ever appear) and agents (those currently alive in the
        # episode).  We populate agents from possible_agents at each reset.
        # -----------------------------------------------------------------
        self.quad_agents = [f"quad_{i}" for i in range(self.num_quads)]
        self.fixed_agents = [f"fixed_{i}" for i in range(self.num_fixed)]
        self.possible_agents = self.quad_agents + self.fixed_agents
        self.agents = self.possible_agents[:]  # shallow copy — will be pruned as agents die
        
        # =================================================================
        # ACTION SPACE
        # =================================================================
        # Both agent types output a 4-dimensional continuous action vector
        # in the range [-1, 1].  The *meaning* of each dimension differs:
        #
        #   Quad  (scout):     [Roll, Pitch, Yaw,      Throttle]
        #   Fixed (commander): [Roll, Pitch, Throttle, Water_Trigger]
        #
        # The mapping from this normalised range to actual physics controls
        # happens inside step() just before the physics is stepped.
        # Using a uniform Box space for both types keeps the network
        # architecture simple (separate actor heads, same output size).
        # =================================================================
        self._action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # =================================================================
        # OBSERVATION SPACE (ASYMMETRIC)
        # =================================================================
        # The two agent types receive fundamentally different observations,
        # which is why we store them in a dict keyed by agent name.
        #
        # --- 1. Scout (Quad) observation ---
        # self_state vector (15 floats):
        #   0-2  : normalised position  x, y, z   (divided by map_bounds)
        #   3-5  : normalised velocity  vx, vy, vz (divided by 20 m/s)
        #   6-9  : normalised distances to the four map edges
        #           (north, south, east, west — divided by grid_size_m)
        #   10-11: static fire start position relative to drone
        #           (used as a coarse compass pointing to the fire origin)
        #   12-14: dynamic fire info extracted from the local map:
        #           rel_x, rel_y (centroid in [-1,1]), intensity (mean)
        self.quad_self_dim = 15
        # max_neighbors is the number of *other* quads whose relative
        # position the attention mechanism can attend to.
        # Guard against num_quads == 1 so we never get a zero-sized tensor.
        self.max_neighbors = self.num_quads - 1 if self.num_quads > 1 else 1

        quad_obs_space = gym.spaces.Dict({
            # Grayscale local fire occupancy map, shape (1, H, W).
            # The channel-first format matches PyTorch CNN conventions.
            "local_map": gym.spaces.Box(low=0.0, high=1.0, shape=(1, local_map_size, local_map_size), dtype=np.float32),
            "self_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.quad_self_dim,), dtype=np.float32),
            # Relative positions of neighbouring scouts (for self-attention).
            # Shape: (max_neighbors, 3) — each row is [rel_x, rel_y, rel_z].
            "neighbor_states": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_neighbors, 3), dtype=np.float32),
            # Boolean attention mask: True = ignore this slot (dead / padding),
            # False = valid neighbour.  Passed directly to MultiheadAttention
            # as key_padding_mask so dead neighbours are never attended to.
            "neighbor_mask": gym.spaces.Box(low=0, high=1, shape=(self.max_neighbors,), dtype=bool)
        })

        # --- 2. Commander (Fixed-wing) observation ---
        # self_state vector (19 floats):
        #   0-2  : normalised position  x, y, z
        #   3-5  : normalised velocity  vx, vy, vz
        #   6-9  : normalised distances to the four map edges
        #   10   : water level in [0, 1]  (current_water / water_capacity)
        #   11-12: relative compass to the refill zone  (rel_x, rel_y)
        #   13-15: orientation angles  (roll, pitch, yaw) normalised by π
        #   16   : danger flag — 1.0 if within 300 m of any edge, else 0.0
        #   17-18: compass to fire start position
        #
        # NOTE: scout messages are NOT part of this observation dict.
        # They are concatenated in train.py and injected as a separate
        # input to the CommanderActor network.
        self.fixed_self_dim = 19

        fixed_obs_space = gym.spaces.Dict({
            "self_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.fixed_self_dim,), dtype=np.float32)
        })

        # Assign each agent its own observation space.
        self._observation_spaces = {}
        for agent in self.quad_agents:
            self._observation_spaces[agent] = quad_obs_space
        for agent in self.fixed_agents:
            self._observation_spaces[agent] = fixed_obs_space

        # =================================================================
        # GLOBAL STATE SPACE (for the centralised critic in MAPPO)
        # =================================================================
        # The critic sees a flattened global state that contains:
        #   - A 16×16 downsampled fire intensity map  →  256 values
        #   - self_state of every quad agent          →  quad_self_dim × N
        #   - self_state of every fixed-wing agent    →  fixed_self_dim × M
        # Using a fixed-size vector (rather than a dict) keeps the critic
        # architecture a simple MLP without any masking logic.
        self.global_state_size = (16 * 16) + (self.num_quads * self.quad_self_dim) + (self.num_fixed * self.fixed_self_dim)
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.global_state_size,), dtype=np.float32)

        # Action history for smoothing.
        # Used in step() to blend the previous action with the new one,
        # preventing jerky or oscillatory control signals.
        # Initialised lazily per agent on first step (action dims differ).
        self.last_actions = {}
        # Raw action history to calculate Jerk rewards
        self.last_raw_actions = {}

    # =========================================================================
    # PettingZoo required property accessors
    # =========================================================================
    # PettingZoo requires these to be decorated with @functools.lru_cache so
    # that repeated calls (e.g. one per agent per step) are O(1) dict lookups
    # instead of fresh object construction.  lru_cache(maxsize=None) means
    # "cache every unique argument infinitely" — safe here because the number
    # of distinct agent names is small and fixed at construction time.
    # =========================================================================

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    # =========================================================================
    # Observation builders
    # =========================================================================

    def _get_quad_obs(self, agent_name):
        """Build the observation dict for a quadrotor scout.

        Returns a dict with keys: local_map, self_state, neighbor_states,
        neighbor_mask.  See __init__ for the full layout description.
        """
        drone = self.sim.drones[agent_name]
        pos = drone.get_position()  # [x, y, z]  in metres
        vel = drone.get_velocity()  # [vx, vy, vz] in m/s

        # --- 1. Normalise position and velocity to roughly [-1, 1] ---
        # Dividing position by map_bounds ensures 0 = centre, ±1 = edge.
        # Dividing velocity by 20.0 assumes max useful speed ~20 m/s.
        norm_pos = pos / NORM_DIST
        norm_vel = vel / 20.0

        # --- 2. Static fire start position (Unit Vector Compass) ---
        vec_x = self.fire_x - pos[0]
        vec_y = self.fire_y - pos[1]
        dist_to_fire = np.hypot(vec_x, vec_y)
        
        if dist_to_fire > 0.1: # Ochrana proti dělení nulou
            rel_fire_start_x = vec_x / dist_to_fire
            rel_fire_start_y = vec_y / dist_to_fire
        else:
            rel_fire_start_x, rel_fire_start_y = 0.0, 0.0

        # --- 3. Normalised distances to each map boundary ---
        dist_measurements = self._get_boundary_measurements_norm(pos)

        # --- 4. Dynamic fire info from the live local fire map ---
        # The scout looks "downward" and sees a 32×32 fire intensity grid.
        # _calculate_fire_info() returns the centroid and mean intensity.
        local_map = self._extract_local_fire_map(pos)
        dyn_x, dyn_y, dyn_intensity = self._calculate_fire_info(local_map)

        # --- 5. Assemble the 15-element self_state vector ---
        self_state = np.array([
            norm_pos[0], norm_pos[1], norm_pos[2],                             # 0-2  : normalised position
            norm_vel[0], norm_vel[1], norm_vel[2],                             # 3-5  : normalised velocity
            dist_measurements[0], dist_measurements[1],                        # 6-7  : distance to north / south edge
            dist_measurements[2], dist_measurements[3],                        # 8-9  : distance to east / west edge
            rel_fire_start_x, rel_fire_start_y,                                # 10-11: static fire compass
            dyn_x, dyn_y, dyn_intensity                                        # 12-14: dynamic fire centroid + intensity
        ], dtype=np.float32)

        # --- 6. Build neighbour tensors for self-attention ---
        neighbor_states = []
        neighbor_mask = []
        
        # Iterate over all other quads
        for other in self.quad_agents:
            if other == agent_name: continue  # Skip self
            
            if other in self.sim.drones:
                # Live neighbour: compute relative position
                other_pos = self.sim.drones[other].get_position()
                rel_pos = other_pos - pos
                # Normalise relative position by arena size
                neighbor_states.append(rel_pos / NORM_DIST)
                neighbor_mask.append(False)  # False = valid, include in attention
            else:
                # Dead neighbour: zero-pad slot and mask it out
                neighbor_states.append(np.zeros(3, dtype=np.float32))
                neighbor_mask.append(True)   # True = masked out in attention

        # If only 1 quad exists, no loop iterations ran — pad to max_neighbors
        while len(neighbor_states) < self.max_neighbors:
            neighbor_states.append(np.zeros(3, dtype=np.float32))
            neighbor_mask.append(True)

        return {
            "local_map": local_map,
            "self_state": self_state,
            "neighbor_states": np.array(neighbor_states, dtype=np.float32),
            "neighbor_mask": np.array(neighbor_mask, dtype=bool)
        }

    def _calculate_fire_info(self, local_map):
        """Extract a compact fire summary from the 32×32 local fire map.

        Returns
        -------
        rel_x : float
            Horizontal centroid of fire pixels in [-1, 1] relative to map
            centre.  Positive = fire is to the right.
        rel_y : float
            Vertical centroid in [-1, 1].  Positive = fire is downward.
        intensity : float
            Mean pixel intensity across the whole local map.

        This triplet is embedded into the scout's self_state (indices 12-14)
        and is also broadcast as the scout's *message* to the commander.
        """
        # Mean intensity over all 32×32 pixels (including zeros for empty cells)
        intensity = np.mean(local_map)

        if intensity > 0.001:
            # local_map has shape (1, 32, 32) — we index channel 0.
            # np.where returns row indices (i) and column indices (j) of
            # all pixels above the threshold.
            grid_i, grid_j = np.where(local_map[0] > 0.1)

            if len(grid_i) > 0:
                # Pixel centroid of burning pixels
                c_i = np.mean(grid_i)  # row    → y direction
                c_j = np.mean(grid_j)  # column → x direction

                # Map pixel [0..31] → normalised [-1, 1] using centre 15.5
                rel_x = (c_j - 15.5) / 15.5
                rel_y = (c_i - 15.5) / 15.5
                return rel_x, rel_y, intensity

        # No significant fire in the local map — return neutral values
        return 0.0, 0.0, intensity
    
    def _get_boundary_measurements_norm(self, pos):
        """Return distances to all four map edges normalised by grid_size_m."""
        distances = self._get_boundary_measurements(pos)
        return np.array(distances) / NORM_DIST

    def _get_boundary_measurements(self, pos):
        """Return raw (metre) distances from pos to each map edge.

        Returns a 4-tuple: (dist_north, dist_south, dist_east, dist_west).
        All values are positive when the drone is inside the map boundaries.
        A negative value means the drone has already crossed an edge.
        """
        dist_north = self.map_bounds - pos[1]   # distance to +Y edge
        dist_south = pos[1] - (-self.map_bounds) # distance to -Y edge
        dist_east  = self.map_bounds - pos[0]   # distance to +X edge
        dist_west  = pos[0] - (-self.map_bounds) # distance to -X edge
        return dist_north, dist_south, dist_east, dist_west
    
    def _get_fixed_obs(self, agent_name):
        """Build the observation dict for a fixed-wing commander.

        Returns a dict with a single key: self_state (19-float vector).
        See __init__ for the full layout description.
        """
        drone = self.sim.drones[agent_name]
        pos = drone.get_position()
        vel = drone.get_velocity()
        rpy = drone.get_orientation_rpy()  # [roll, pitch, yaw] in radians

        # Water level as a fraction in [0, 1]
        water_lvl = drone.current_water / drone.water_capacity if drone.water_capacity > 0 else 0.0
        
        # Relative compass direction to the refill (water-replenishment) zone
        if self.sim.environment.refill_zone is not None:
            refill_pos = self.sim.environment.refill_zone['position']
            rel_base_x = (refill_pos[0] - pos[0]) / NORM_DIST
            rel_base_y = (refill_pos[1] - pos[1]) / NORM_DIST
        else:
            rel_base_x = 0.0
            rel_base_y = 0.0

        # --- Normalise ---
        norm_pos = pos / NORM_DIST               # fixed normalisation
        norm_vel = vel / 20.0                   # vx,vy,vz in roughly [-1, 1]
        norm_rpy = rpy / np.pi                  # angles from [-pi,pi] to [-1,1]

        # Normalised distances to each map edge
        dist_boundaries = self._get_boundary_measurements_norm(pos)

        # --- Danger flag ---
        # 1.0 when the aircraft is within 20% of map half-size of any edge.
        # At ~20 m/s cruise speed this gives enough lead time to turn.
        danger_threshold = FIXED["boundary_threshold_m"]  # fixed absolute distance, same as penalty zone
        danger_flag = 1.0 if min(self._get_boundary_measurements(pos)) < danger_threshold else 0.0
        
        # --- Fire compass (unit vector from drone toward fire) ---
        vec_fire_x = self.fire_x - pos[0]
        vec_fire_y = self.fire_y - pos[1]
        dist_to_fire = np.hypot(vec_fire_x, vec_fire_y)
        if dist_to_fire > 1.0:
            fire_dir_x = vec_fire_x / dist_to_fire
            fire_dir_y = vec_fire_y / dist_to_fire
        else:
            fire_dir_x, fire_dir_y = 0.0, 0.0

        self_state = np.array([
            norm_pos[0], norm_pos[1], norm_pos[2],                            # 0-2  : position
            norm_vel[0], norm_vel[1], norm_vel[2],                            # 3-5  : velocity
            dist_boundaries[0], dist_boundaries[1],                           # 6-7  : north / south edge
            dist_boundaries[2], dist_boundaries[3],                           # 8-9  : east / west edge
            water_lvl,                                                        # 10   : water level
            rel_base_x, rel_base_y,                                           # 11-12: compass to refill zone
            norm_rpy[0], norm_rpy[1], norm_rpy[2],                            # 13-15: roll, pitch, yaw
            danger_flag,                                                      # 16   : proximity alert
            fire_dir_x, fire_dir_y                                            # 17-18: compass to fire start position
        ], dtype=np.float32)

        return {"self_state": self_state}

    def _get_obs(self, agent_name):
        """Route observation generation to the correct agent-type method.

        Returns a zero-filled observation dict if the agent is already dead,
        so that the MAPPO policy never receives None or missing keys.
        """
        if "fixed" in agent_name:
            if agent_name in self.sim.drones:
                return self._get_fixed_obs(agent_name)
            else:
                # Dead fixed-wing: return a zero state vector
                return {"self_state": np.zeros(self.fixed_self_dim, dtype=np.float32)}
        else:
            if agent_name in self.sim.drones:
                return self._get_quad_obs(agent_name)
            else:
                # Dead quadrotor: zero-fill every tensor in the observation dict.
                # neighbor_mask is all-True so attention ignores every slot.
                return {
                    "local_map": np.zeros((1, 32, 32), dtype=np.float32),
                    "self_state": np.zeros(self.quad_self_dim, dtype=np.float32),
                    "neighbor_states": np.zeros((self.max_neighbors, 3), dtype=np.float32),
                    "neighbor_mask": np.ones((self.max_neighbors,), dtype=bool)
                }

    def get_privileged_state(self, agent_name):
        """Build privileged global-state vector for the critic (CTDE).

        Contains the agent's own obs + privileged extras that the actor
        never sees: fire position, fire intensity, other agent's position.

        Scout  critic input: 15 (self) + 6 (priv) = 21
        Cmdr   critic input: 17 (self) + 6 (priv) = 23
        """
        own_obs = self._get_obs(agent_name)
        own_state = own_obs["self_state"]  # 15D (scout) or 17D (cmdr)

        # Fire info (normalised by map_bounds)
        fire_x_norm = self.fire_x / NORM_DIST
        fire_y_norm = self.fire_y / NORM_DIST

        # Current fire intensity (mean of fire grid, 0 if no grid)
        fire_intensity = 0.0
        if self.sim.environment.fire_grid is not None:
            fire_intensity = float(np.mean(self.sim.environment.fire_grid.I))

        # Other agent's position (normalised)
        other_pos = np.zeros(3)
        if "fixed" in agent_name:
            # Commander critic sees scout position
            for q in self.quad_agents:
                if q in self.sim.drones:
                    other_pos = self.sim.drones[q].get_position() / NORM_DIST
                    break
        else:
            # Scout critic sees commander position
            for f in self.fixed_agents:
                if f in self.sim.drones:
                    other_pos = self.sim.drones[f].get_position() / NORM_DIST
                    break

        priv = np.array([fire_x_norm, fire_y_norm, fire_intensity,
                         other_pos[0], other_pos[1], other_pos[2]],
                        dtype=np.float32)
        return np.concatenate([own_state, priv])

    def _extract_local_fire_map(self, pos, resolution_px=32):
        """Extract and resize the fire intensity map local to the drone's position.

        The field-of-view (FOV) window scales with altitude: the higher the
        drone flies, the larger the ground area it observes, but each pixel
        covers more territory (lower detail).  This creates a natural
        altitude trade-off the agent must learn:
          - Fly *low* for precise, high-resolution fire targeting.
          - Fly *high* for wider situational awareness and faster discovery.

        The raw crop is resized to (resolution_px, resolution_px) using
        bilinear interpolation so the CNN input is always the same shape.
        Output form is (1, H, W) — channel-first, as expected by PyTorch
        Conv2d.  The singleton leading dimension is added with `np.newaxis`
        (equivalent to crop[None, ...]).
        """
        if self.sim.environment.fire_grid is None:
            return np.zeros((1, resolution_px, resolution_px), dtype=np.float32)

        mapper = self.sim.environment.grid_mapper
        fire_grid = self.sim.environment.fire_grid.I  # 2-D fire intensity array

        # Altitude-adaptive FOV: window side = max(10 m, altitude x 1.5).
        # At z=10 m => 15 m window;  at z=80 m => 120 m window.
        adaptive_window = max(10.0, pos[2] * 1.5)

        # World-coordinate bounding box of the crop (metres)
        half_w = adaptive_window / 2.0
        min_world = (pos[0] - half_w, pos[1] - half_w)
        max_world = (pos[0] + half_w, pos[1] + half_w)

        # Convert world coordinates to integer grid cell indices
        r_min, c_min = mapper.world_to_cell(min_world)
        r_max, c_max = mapper.world_to_cell(max_world)

        # Clamp to valid grid bounds to avoid out-of-range array slicing
        r_min = max(0, min(r_min, mapper.grid_height_cells - 1))
        r_max = max(0, min(r_max, mapper.grid_height_cells - 1))
        c_min = max(0, min(c_min, mapper.grid_width_cells - 1))
        c_max = max(0, min(c_max, mapper.grid_width_cells - 1))

        crop = fire_grid[r_min:r_max+1, c_min:c_max+1]

        # Resize the variable-size crop to the fixed CNN input resolution.
        # Guard against empty crops (e.g. drone is exactly on the grid edge).
        if crop.size == 0:
            processed_map = np.zeros((resolution_px, resolution_px), dtype=np.float32)
        else:
            import cv2
            try:
                processed_map = cv2.resize(crop, (resolution_px, resolution_px), interpolation=cv2.INTER_LINEAR)
            except:
                processed_map = np.zeros((resolution_px, resolution_px), dtype=np.float32)

        # Add a channel dimension: (H, W) -> (1, H, W) for PyTorch Conv2d
        return processed_map[np.newaxis, ...].astype(np.float32)

    # =========================================================================
    # Global state (for centralised critic)
    # =========================================================================

    def state(self):
        """Return the global state vector used by the centralised MAPPO critic.

        MAPPO uses a centralised critic that sees the full world state while
        the individual actors only see their own local observations.  This
        method assembles that global state as a single flat numpy array.

        Layout:
          [ fire_summary (256) | quad_self_states (15 x N) | fixed_self_states (17 x M) ]

        The fire grid is downsampled to 16x16 using INTER_AREA (pixel-area
        averaging), which gives the critic a compact but representative
        overview of the fire.  INTER_AREA is preferred for downsampling
        because it averages all contributing source pixels, preserving total
        intensity better than bilinear sampling.
        """
        # --- 1. Compressed global fire map (16x16 = 256 values) ---
        if self.sim.environment.fire_grid is not None:
            full_grid = self.sim.environment.fire_grid.I  # full-res 2-D array
            import cv2
            # INTER_AREA = box-filter downsampling: each output pixel is the
            # mean intensity of the corresponding source region.
            small_grid = cv2.resize(full_grid, (16, 16), interpolation=cv2.INTER_AREA)
            # Flatten 2-D (16, 16) -> 1-D (256,) for concatenation
            fire_summary = small_grid.flatten()
        else:
            fire_summary = np.zeros(16 * 16, dtype=np.float32)

        # --- 2. Self-state vectors of every agent (dead agents get zeros) ---
        agent_states = []

        # Quads first
        for agent in self.quad_agents:
            if agent in self.sim.drones:
                agent_states.append(self._get_quad_obs(agent)["self_state"])
            else:
                agent_states.append(np.zeros(self.quad_self_dim, dtype=np.float32))

        # Fixed-wing second
        for agent in self.fixed_agents:
            if agent in self.sim.drones:
                agent_states.append(self._get_fixed_obs(agent)["self_state"])
            else:
                agent_states.append(np.zeros(self.fixed_self_dim, dtype=np.float32))

        # Concatenate fire summary + all agent states into one long 1-D vector
        global_state = np.concatenate([fire_summary] + agent_states)
        return global_state
        
    # =========================================================================
    # Episode lifecycle
    # =========================================================================

    def reset(self, seed=None, options=None, epizode_number=0):
        """Reset the environment to the start of a new episode.

        Performs a hard restart of the PyBullet physics engine, spawns a new
        fire, places the refill zone, and positions all drones.

        Parameters
        ----------
        seed : int or None
            If provided, seeds both numpy and Python's random module for
            reproducible rollouts.
        epizode_number : int
            Current training episode number, used by the curriculum scheduler.

        Returns
        -------
        observations : dict[str, obs]
            Initial observation for every agent.
        infos : dict[str, dict]
            Empty info dicts (required by PettingZoo API).
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # --- 0. Optional map-size randomisation ---
        # If map_size_range is set, randomly pick a new grid_size each episode
        # so the policy generalises across different arena scales.
        if hasattr(self, 'map_size_range') and self.map_size_range is not None:
            lo, hi = self.map_size_range
            self.grid_size_m = float(random.randint(int(lo) // 100, int(hi) // 100) * 100)
            self.map_bounds = self.grid_size_m / 2.0

        # --- 1. Restore the full agent list and reset action histories ---
        # self.agents starts as a copy of possible_agents and is pruned as
        # agents are terminated during the episode.
        self.agents = self.possible_agents[:]
        self.last_actions = {}
        self.last_raw_actions = {}

        # --- 2. Hard-restart the PyBullet engine ---
        # Calling stop_simulation() then creating a fresh Simulation() object
        # fully tears down the world and prevents memory leaks from accumulated
        # PyBullet objects across many episodes.
        if hasattr(self, 'sim') and self.sim is not None:
            self.sim.stop_simulation()

        self.sim = Simulation()
        self.sim.start_simulation()

        # --- 3. Initialise the fire simulation grid ---
        self.sim.enable_fire_simulation(
            grid_width_m=self.grid_size_m,
            grid_height_m=self.grid_size_m,
            cell_size_m=5.0,
            dt=0.1
        )

        # Oheň na náhodné pozici v bezpečné zóně
        safe_zone_fire = self.map_bounds * 0.4
        self.fire_x = random.uniform(-safe_zone_fire, safe_zone_fire)
        self.fire_y = random.uniform(-safe_zone_fire, safe_zone_fire)

        # =========================================================
        # PROBABILISTIC CURRICULUM (Smíšená obtížnost bez útesů)
        # =========================================================
        rand_diff = random.random()
        
        if rand_diff < 0.30:
            # 30 % šance: SNADNÁ MISE (Zrodí se ±20m od ohně)
            # Udržuje "svalovou paměť" pro visení, centruje oheň a drží výšku.
            # Kritik díky tomuto vždy vidí nějaké vysoké odměny a neexploduje.
            spawn_radius = 20.0
            
        elif rand_diff < 0.50:
            # 20 % šance: STŘEDNÍ MISE (Zrodí se ±150m od ohně)
            # Učí drona hledat oheň v okolí a hrubě používat kompas.
            spawn_radius = 150.0
            
        else:
            # 50 % šance: TĚŽKÁ MISE (Zrodí se kdekoli na mapě)
            # Učí drona vytrvalosti, letu přes půl mapy a maximální důvěře v kompas.
            spawn_radius = self.map_bounds * 0.8

        # Oheň na náhodné pozici v bezpečné zóně
        safe_zone_fire = self.map_bounds * 0.4
        self.fire_x = random.uniform(-safe_zone_fire, safe_zone_fire)
        self.fire_y = random.uniform(-safe_zone_fire, safe_zone_fire)
        # self.sim.start_fire([self.fire_x, self.fire_y], intensity=0.5)
        num_ignitions = random.randint(3, 6)
        for _ in range(num_ignitions):
            offset_x = random.uniform(-40, 40)
            offset_y = random.uniform(-40, 40)
            self.sim.start_fire([self.fire_x + offset_x, self.fire_y + offset_y], intensity=0.5)
        self.current_episode = epizode_number

        # Refill zona na náhodné pozici, ale s jistou korelací k ohni (aby nebyla úplně mimo mapu)
        refill_x = float(np.clip(-self.fire_x + random.uniform(-50, 50), -self.map_bounds * 0.8, self.map_bounds * 0.8))
        refill_y = float(np.clip(-self.fire_y + random.uniform(-50, 50), -self.map_bounds * 0.8, self.map_bounds * 0.8))
        self.sim.environment.create_refill_zone(center_pos=[refill_x, refill_y, 0.0])

        # --- 4. Spawn every agent at its starting position ---
        for agent in self.agents:
            if "fixed" in agent:
                # Normální spawn pro fixed-wing
                # fw_spawn_radius = random.uniform(0.0, self.map_bounds * 0.30)
                # spawn_angle  = random.uniform(0, 2 * np.pi)
                # fw_start_x   = float(fw_spawn_radius * np.cos(spawn_angle))
                # fw_start_y   = float(fw_spawn_radius * np.sin(spawn_angle))
                # Phase 1: spawn 150-400m od ohně
                fw_spawn_dist = random.uniform(150.0, 400.0)
                spawn_angle = random.uniform(0, 2 * np.pi)
                fw_start_x = float(np.clip(
                    self.fire_x + fw_spawn_dist * np.cos(spawn_angle),
                    -self.map_bounds * 0.85, self.map_bounds * 0.85))
                fw_start_y = float(np.clip(
                    self.fire_y + fw_spawn_dist * np.sin(spawn_angle),
                    -self.map_bounds * 0.85, self.map_bounds * 0.85))
                fw_yaw       = random.uniform(-np.pi, np.pi)

                self.sim.add_fixedwing(agent, position=[fw_start_x, fw_start_y, 100.0], water_capacity=200.0, yaw=fw_yaw)

                drone = self.sim.drones[agent]
                drone.state_va = 15.0

            else:
                # Quads: Tady vygenerujeme unikátní pozici pro KAŽDÉHO scouta zvlášť!
                # quad_start_x = self.fire_x + random.uniform(-spawn_radius, spawn_radius)
                # quad_start_y = self.fire_y + random.uniform(-spawn_radius, spawn_radius)
                quad_start_x = self.fire_x + random.uniform(-20, 20)
                quad_start_y = self.fire_y + random.uniform(-20, 20)
                
                # Pojistka proti spawnu za mapou
                quad_start_x = float(np.clip(quad_start_x, -self.map_bounds * 0.9, self.map_bounds * 0.9))
                quad_start_y = float(np.clip(quad_start_y, -self.map_bounds * 0.9, self.map_bounds * 0.9))
                quad_start_z = random.uniform(30.0, 60.0)
                
                self.sim.add_quadcopter(agent, position=[quad_start_x, quad_start_y, quad_start_z])

        # --- 5. Initialise per-episode tracking variables ---
        self.visited_cells = set()   # tracks which grid cells have been overflown
        self.fire_discovered = False
        self.current_step = 0

        # Dwell counter: tracks consecutive steps each quad spends over fire
        self._dwell_counter = {}
        # Jerk penalty: tracks previous actions for smoothness reward
        self._last_actions_rw = {}

        # Potential-based shaping: track previous distance to fire per scout/fw
        self._prev_fire_dists = {}
        for q in self.quad_agents:
            if q in self.sim.drones:
                pos = self.sim.drones[q].get_position()
                self._prev_fire_dists[q] = np.sqrt((pos[0] - self.fire_x)**2 + (pos[1] - self.fire_y)**2)
                
        for f in self.fixed_agents:
            if f in self.sim.drones:
                pos = self.sim.drones[f].get_position()
                self._prev_fire_dists[f] = np.hypot(pos[0] - self.fire_x, pos[1] - self.fire_y)

        # Fire spread tracking: how many cells are burning at episode start
        if self.sim.environment.fire_grid is not None:
            self._prev_burning_count = int(np.sum(self.sim.environment.fire_grid.B))
        else:
            self._prev_burning_count = 0

        # --- 6. Build and return initial observations (PettingZoo requirement) ---
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    # =========================================================================
    # Step
    # =========================================================================

    def step(self, actions):
        """Apply one set of agent actions and advance the simulation.

        The step proceeds as follows:
          1.  Action smoothing  — blend new action with the previous one to
              reduce jerky, oscillatory control.
          2.  Physics stepping  — call the simulator frame_skip times so the
              physics can settle between RL decisions.
          3.  Death detection   — check boundary and crash conditions.
          4.  Reward computation — per-agent reward using physics shaping +
              mission-specific reward functions.
          5.  Team reward       — commander extinguish bonus shared with scouts.
          6.  PettingZoo API compliance — populate all required output dicts.

        Parameters
        ----------
        actions : dict[str, np.ndarray]
            One 4-dim action vector per living agent, values in [-1, 1].

        Returns
        -------
        observations, rewards, terminations, truncations, infos : dicts
            All keyed by agent name as required by the PettingZoo ParallelEnv
            API.
        """
        self.current_step += 1
        # frame_skip: number of physics sub-steps per RL decision step.
        # 5 sub-steps at ~240 Hz physics = ~48 Hz effective RL frequency.
        # This is a common trick to keep physics stable while letting the
        # policy operate at a lower frequency.
        frame_skip = 5

        # --- 1. Action smoothing and mapping ---
        # Exponential moving average: new_smooth = 0.8 * old + 0.2 * new.
        # This is a simple low-pass filter on the action signal.  The 0.8/0.2
        # blend (tau ~ 5 steps) prevents the control surface commands from
        # jumping discontinuously, which would excite unphysical oscillations
        # in the rigid-body simulator.
        drone_controls = {}

        for agent_name, action in actions.items():
            # -----------------------------------------------------------------
            # NO smoothing for fixed-wing during training.
            # Smoothing breaks PPO credit assignment: PPO compares action A
            # with reward R, but R was caused by smooth(A) — a different
            # action.  The guidance model's inner loop (kp_gamma, kp_phi)
            # already provides physical smoothing of control surfaces.
            # Quad actions are still smoothed (direct motor control needs it).
            # -----------------------------------------------------------------
            if "fixed" in agent_name:
                smooth_action = action
            else:
                prev = self.last_actions.get(agent_name, np.zeros_like(action))
                smooth_action = 0.5 * prev + 0.5 * action
            self.last_actions[agent_name] = smooth_action

            if "fixed" in agent_name:
                drone = self.sim.drones.get(agent_name)
                if drone is not None:
                    # =========================================================
                    # HIERARCHICAL FLIGHT CONTROLLER
                    # =========================================================
                    # The RL network outputs HIGH-LEVEL strategy (3 dims):
                    #   [0] heading_delta  [-1, 1] → turn left/right
                    #   [1] target_alt     [-1, 1] → desired altitude
                    #   [2] water_trigger  [-1, 1] → drop water or not
                    #
                    # This deterministic controller converts strategy → physics
                    # commands [roll, pitch, throttle, water].  The aircraft
                    # CANNOT crash from bad RL actions — the controller clamps
                    # everything to safe ranges.  The RL agent only decides
                    # WHERE to fly, not HOW to fly.
                    # =========================================================

                    heading_delta_raw = float(action[0])  # [-1, 1]
                    target_alt_raw    = float(action[1])  # [-1, 1]
                    water_raw         = float(action[2])  # [-1, 1]

                    # --- 1. HEADING → ROLL ---
                    # heading_delta [-1, 1] → desired turn [-π, π]
                    # Proportional controller: larger turn desire → more roll
                    desired_turn = heading_delta_raw * np.pi
                    # Roll command: [-1, 1] (maps to ±45° in physics)
                    # Gain 2.0: When used with heading-hold (training loop
                    # passes heading_error/π as action[0]), this gives:
                    #   90° error → roll=1.0 (45° bank) — aggressive turn
                    #   45° error → roll=0.5 (22° bank) — moderate
                    #   10° error → roll=0.11 (5° bank) — gentle correction
                    roll_cmd = np.clip(2.0 * heading_delta_raw, -1.0, 1.0)

                    # --- 2. ALTITUDE → PITCH ---
                    # target_alt [-1, 1] → [40, 250] meters
                    target_alt = 40.0 + (target_alt_raw + 1.0) / 2.0 * 210.0
                    current_alt = drone.state_pos[2]
                    alt_error = target_alt - current_alt
                    # PD controller: proportional on error, derivative on climb rate
                    vz = drone.get_velocity()[2]
                    # pitch_cmd [-1, 1] maps to ±15° in physics (max_pitch)
                    pitch_cmd = np.clip(0.008 * alt_error - 0.02 * vz, -0.5, 0.5)

                    # --- 3. THROTTLE (fixed cruise) ---
                    throttle = 0.7  # → 0.7 × 30 = 21 m/s cruise

                    # --- 4. WATER ---
                    water_trigger = 1.0 if water_raw > 0.0 else 0.0

                    mapped_action = np.array([roll_cmd, pitch_cmd, throttle, water_trigger])
                    drone_controls[agent_name] = mapped_action
                else:
                    drone_controls[agent_name] = np.zeros(4)
            else:
                # Quad: actions are used directly by the physics backend
                drone_controls[agent_name] = smooth_action

        # --- 2. Step the physics engine frame_skip times ---
        for _ in range(frame_skip):
            self.sim.step_simulation(drone_controls)

        # --- 3. Prepare output dicts (required by PettingZoo API) ---
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Check episode time limit
        time_is_up = self.current_step >= getattr(self, 'max_steps', 500)

        # --- 4. Per-agent death check and reward computation ---
        agents_to_keep = []  # agents still alive after this step

        #########################################
        # MISSION
        #########################################

        for agent in self.agents:
            # Default to alive / not truncated
            terminations[agent] = False
            truncations[agent] = time_is_up
            infos[agent] = {}
            
            # Check whether the agent has crashed or left the map
            dead, crash_reward, death_cause = self._check_death(agent)

            rewards[agent] = 0

            if dead:
                terminations[agent] = True
                rewards[agent] += crash_reward
                infos[agent]["death_cause"] = death_cause
            else:
                rewards[agent] += self._apply_physics_shaping(agent)

                # Agent-type-specific mission reward
                # if "fixed" in agent:
                #     # Najdi oheň skrz zprávy od scoutů
                #     max_seen_intensity = 0.0
                #     best_fire_pos = np.array([self.fire_x, self.fire_y])  # fallback
                #     for q_name in self.quad_agents:
                #         if q_name in self.sim.drones:
                #             q_obs = self._get_quad_obs(q_name)
                #             q_intensity = q_obs["self_state"][14]
                #             if q_intensity > max_seen_intensity:
                #                 max_seen_intensity = q_intensity
                #                 rel_x = q_obs["self_state"][12]
                #                 rel_y = q_obs["self_state"][13]
                #                 q_pos = self.sim.drones[q_name].get_position()
                #                 fov_size = max(10.0, q_pos[2] * 1.5)
                #                 best_fire_pos = np.array([
                #                     q_pos[0] + rel_x * (fov_size / 2.0),
                #                     q_pos[1] + rel_y * (fov_size / 2.0)
                #                 ])

                #     if max_seen_intensity > 0.005:
                #         # MISSION: letět k ohni, hasit
                #         # Threshold 0.005 = oheň zabírá ~1% FOV při intensity 0.5
                #         rewards[agent] += self._get_fixed_reward(agent, best_fire_pos)
                #     else:
                #         # PATROL: zůstat blízko středu
                #         rewards[agent] += self._get_fixed_reward_patrol(agent)
                if "fixed" in agent:
                    rewards[agent] += self._get_fixed_reward_nav(agent)
                else:
                    rewards[agent] += self._get_quad_reward(agent)
                
                if time_is_up:
                    rewards[agent] += 2.0
                
             
            # If alive: store observation and keep the agent in the active list.
            # If dead: PettingZoo requires us to still return a final observation
            # for the terminated agent (the "terminal observation" convention).
            if not terminations[agent] and not truncations[agent]:
                observations[agent] = self._get_obs(agent)
                agents_to_keep.append(agent)
            else:
                # Terminal observation for a dead agent: all zeros.
                # The policy will never act on this observation, but the API
                # requires it so that train.py can safely access observations[agent]
                # after step() returns without a KeyError.
                if "fixed" in agent:
                    observations[agent] = {"self_state": np.zeros(self.fixed_self_dim, dtype=np.float32)}
                else:
                    observations[agent] = {
                        "local_map": np.zeros((1, 32, 32), dtype=np.float32),
                        "self_state": np.zeros(self.quad_self_dim, dtype=np.float32),
                        "neighbor_states": np.zeros((self.max_neighbors, 3), dtype=np.float32),
                        "neighbor_mask": np.ones((self.max_neighbors,), dtype=bool)
                    }

        # --- 5. Update the alive-agents list ---
        self.agents = agents_to_keep

        # --- 6. Team reward: commander extinguishing -> bonus for everyone ---
        for f_agent in self.fixed_agents:
            if f_agent not in drone_controls or f_agent not in rewards:
                continue

            eff = self.sim.drone_extinguish_stats.get(f_agent, 0.0)
            is_dropping = drone_controls[f_agent][3] > 0.5

            if eff > 0.0:
                fire_bonus = min(eff * 50.0, 3.0)  # cap per-step bonus
                rewards[f_agent] += fire_bonus

                # Scouts get 20% of the extinguish bonus — their messages guided commander
                for q_agent in [a for a in self.quad_agents if a in rewards]:
                    rewards[q_agent] += fire_bonus * 0.2

            elif is_dropping and f_agent in self.sim.drones:
                # Dense reward: dropping water near fire
                f_pos = self.sim.drones[f_agent].get_position()
                dist_to_fire = np.linalg.norm([f_pos[0] - self.fire_x, f_pos[1] - self.fire_y])
                if dist_to_fire < 100.0:
                    rewards[f_agent] += 2.0  # dense: +2/krok za správné hašení
                else:
                    # Penalty for wasting water far from fire
                    rewards[f_agent] -= 3.0

        # --- 7. Fire spread penalty — penalizace za šíření ohně (urgence) ---
        # Každý krok kde se oheň rozšíří o nové buňky, oba agenti dostanou trest.
        # Tohle dává agentům motivaci jednat rychle — čím déle čekají, tím víc
        # oheň roste a tím víc bodů ztrácejí. Sdílená penalizace zarovnává
        # incentivy scoutů a commandera směrem k jedinému cíli.
        # SKIP when no fixed-wing agents exist — scouts can't extinguish fire,
        # so penalising them for spread is pure noise that drowns the fire-finding signal.
        # if self.num_fixed > 0 and self.sim.environment.fire_grid is not None:
        #     current_burning = int(np.sum(self.sim.environment.fire_grid.B))
        #     delta_burned = max(0, current_burning - self._prev_burning_count)
        #     if delta_burned > 0:
        #         spread_penalty = delta_burned * 0.02  # ~0.5/krok when fire grows by 25 cells
        #         for agent in rewards:
        #             if "fixed" not in agent:
        #                 rewards[agent] -= spread_penalty
        #     self._prev_burning_count = current_burning

        for agent in rewards:
            rewards[agent] = np.clip(rewards[agent], SHARED["reward_clip_min"], SHARED["reward_clip_max"])

        return observations, rewards, terminations, truncations, infos



    # =========================================================================
    # Private reward helper methods
    # =========================================================================
    
    def _apply_physics_shaping(self, agent):
        """Fyzikální shaping — společný pro oba agenty.

        Obsahuje:
          - Survival bonus za přežití kroku
          - Boundary penalty (quadratická) za přiblížení k hranici mapy
          - Altitude penalty (flat) za létání mimo ideální rozsah výšky
        Žádné hodnoty nezávisí na velikosti mapy.
        """
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        reward = SHARED["survival_bonus"]

        # --- Boundary penalty ---
        dist_boundaries = self._get_boundary_measurements(pos)
        dist_to_edge = min(dist_boundaries)

        if "fixed" in agent:
            threshold = FIXED["boundary_threshold_m"]
        else:
            threshold = QUAD["boundary_threshold_m"]

        if dist_to_edge < threshold:
            reward -= SHARED["boundary_penalty"] * (1.0 - dist_to_edge / threshold) ** 2

        # --- Altitude penalty (flat, like old version) ---
        if "fixed" in agent:
            alt_min = FIXED["alt_ideal_min"]
            alt_max = FIXED["alt_ideal_max"]
            if pos[2] > alt_max or pos[2] < alt_min:
                reward -= 0.05
        else:
            if pos[2] > QUAD["alt_ideal_max"]:
                excess_alt = pos[2] - QUAD["alt_ideal_max"]
                reward -= (excess_alt * QUAD["alt_penalty"])
            elif pos[2] < QUAD["alt_ideal_min"]:
                excess_alt = QUAD["alt_ideal_min"] - pos[2]
                reward -= (excess_alt * QUAD["alt_penalty"])

            if QUAD["alt_sweet_min"] <= pos[2] <= QUAD["alt_sweet_max"]:
                reward += QUAD["alt_sweet_bonus"]

        return reward

    def _get_quad_reward(self, agent):
        """Compute the mission reward for a quadrotor scout.

        Jednoduchý design (osvědčený z commitu e571a46):
          - Vidí oheň → masivní odměna (flat + intensity)
          - Nevidí oheň → kompas k ohni (approach speed)
          - Speed penalty nad ohněm → nutí zastavit
        Žádné dwell, center, jerk, fire-lost — jen čistý silný signál.
        """
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        vel = drone.get_velocity()
        reward = 0.0

        local_map = self._extract_local_fire_map(pos)
        avg_fire_intensity = np.mean(local_map)

        if avg_fire_intensity > 0.001:
            # ── Vidí oheň → obrovská odměna ──
            reward += QUAD["fire_flat_bonus"]
            reward += avg_fire_intensity * QUAD["fire_intensity_k"]
            speed = np.linalg.norm(vel)
            reward -= speed * QUAD["fire_speed_pen"]
        else:
            # ── Nevidí oheň → kompas ──
            vec_to_fire = np.array([self.fire_x - pos[0], self.fire_y - pos[1]])
            dist_to_fire = np.linalg.norm(vec_to_fire)
            if dist_to_fire > 5.0:
                dir_to_fire = vec_to_fire / dist_to_fire
                approach_speed = np.dot(vel[:2], dir_to_fire)
                reward += approach_speed * QUAD["approach_k"]

        return reward
    
    def _get_fixed_reward_nav(self, agent):
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        water_lvl = drone.current_water / drone.water_capacity if drone.water_capacity > 0 else 0.0

        # Potential-based
        dist_to_fire = np.hypot(pos[0] - self.fire_x, pos[1] - self.fire_y)
        prev_dist = self._prev_fire_dists.get(agent, dist_to_fire)
        delta = prev_dist - dist_to_fire
        self._prev_fire_dists[agent] = dist_to_fire

        reward = delta * 0.01

        if water_lvl > 0.05:
            # MÁ VODU → zónové bonusy za blízkost k ohni
            if dist_to_fire < 300:
                reward += 0.3
            if dist_to_fire < 150:
                reward += 0.5
            # Altitude sweet spot
            if dist_to_fire < 300:
                alt = pos[2]
                if 60 < alt < 120:
                    reward += 0.1
                elif alt > 200 or alt < 30:
                    reward -= 0.1
        else:
            # PRÁZDNÝ TANK → gradient k refill zóně
            if self.sim.environment.refill_zone is not None:
                rz = self.sim.environment.refill_zone['position']
                dist_to_refill = np.hypot(pos[0] - rz[0], pos[1] - rz[1])
                # Normalizovaný gradient: max +0.3/step when heading to refill
                reward += max(0.0, 0.3 * (1.0 - dist_to_refill / 500.0))
                if dist_to_refill < 100:
                    reward += 0.5  # blízko refill zóny = velký bonus

        return reward

    def _get_fixed_reward_patrol(self, agent):
        """Patrol reward — malý tah ke středu mapy když scout nevidí oheň.

        Boundary penalty v _apply_physics_shaping už řeší okraje.
        Zde jen přidáme mírný gradient ke středu, aby commander nebloudil
        po mapě bez cíle.
        """
        pos = self.sim.drones[agent].get_position()
        dist_from_center = np.linalg.norm(pos[:2])
        # Lineárně klesá od 0.05 ve středu na 0.0 na hranici mapy
        return max(0.0, 0.05 * (1.0 - dist_from_center / self.map_bounds))

    def _get_fixed_reward(self, agent, best_fire_pos):
        """Mission reward pro commandera — přibližovací gradient k ohni.

        Lineární tah od 1500 m (0.0) až k 200 m (0.43), pak flat 0.5 uvnitř
        200 m — žádná dead zone, commander je vždy odměněn za to že je blízko
        ohni. Extinguish bonus (sekce 6) přidává další odměnu za skutečné hašení.
        """
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        dist_to_fire = np.linalg.norm(pos[:2] - best_fire_pos)
        close_radius = max(100.0, self.map_bounds * 0.13)   # at least 100m
        if dist_to_fire <= close_radius:
            return 0.5 + 0.5 * (1.0 - dist_to_fire / close_radius)
        return max(0.0, 0.5 * (1.0 - dist_to_fire / self.map_bounds))

    def _check_death(self, agent):
        """Detekce pádu agenta.

        Tři příčiny:
          1. Physics crash (PyBullet odstranil drone)
          2. Boundary violation (vyletěl z mapy)
          3. Ceiling violation (příliš vysoko)

        Ceiling je nastaven výrazně nad ideální výškou aby agent měl
        prostor reagovat — penalizace v _apply_physics_shaping ho tlačí
        dolů dřív než dosáhne ceilingu.
        """
        if agent not in self.sim.drones:
            return True, SHARED["crash_penalty"], "ground_crash"

        pos = self.sim.drones[agent].get_position()

        if abs(pos[0]) > self.map_bounds or abs(pos[1]) > self.map_bounds:
            self.sim._destroy_drone(agent)
            return True, SHARED["crash_penalty"], "boundary"

        max_ceiling = FIXED["alt_ceiling"] if "fixed" in agent else QUAD["alt_ceiling"]
        if pos[2] > max_ceiling:
            self.sim._destroy_drone(agent)
            return True, SHARED["crash_penalty"], "ceiling"

        return False, 0.0, ""