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
        # self_state vector (17 floats):
        #   0-2  : normalised position  x, y, z
        #   3-5  : normalised velocity  vx, vy, vz
        #   6-9  : normalised distances to the four map edges
        #   10   : water level in [0, 1]  (current_water / water_capacity)
        #   11-12: relative compass to the refill zone  (rel_x, rel_y)
        #   13-15: orientation angles  (roll, pitch, yaw) normalised by π
        #   16   : danger flag — 1.0 if within 300 m of any edge, else 0.0
        #
        # NOTE: scout messages are NOT part of this observation dict.
        # They are concatenated in train.py and injected as a separate
        # input to the CommanderActor network.
        self.fixed_self_dim = 17

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

        # Action history for smoothing (one 4-dim zero vector per agent).
        # Used in step() to blend the previous action with the new one,
        # preventing jerky or oscillatory control signals.
        self.last_actions = {agent: np.zeros(4, dtype=np.float32) for agent in self.possible_agents}
        # Raw action history to calculate Jerk rewards
        self.last_raw_actions = {agent: np.zeros(4, dtype=np.float32) for agent in self.possible_agents}

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
        norm_pos = pos / self.map_bounds
        norm_vel = vel / 20.0

        # --- 2. Static fire start position (coarse compass) ---
        # Even before the scout has visually found the fire it gets a
        # normalised vector pointing toward the fire's *initial* spawn
        # position.  This helps early exploration / curriculum learning.
        rel_fire_start_x = (self.fire_x - pos[0]) / self.map_bounds
        rel_fire_start_y = (self.fire_y - pos[1]) / self.map_bounds

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
                neighbor_states.append(rel_pos / self.grid_size_m)
                neighbor_mask.append(False)  # False = valid, include in attention
            else:
                # Dead neighbour: zero-pad slot and mask it out
                neighbor_states.append(np.zeros(3, dtype=np.float32))
                neighbor_mask.append(True)   # True = masked out in attention

        # If only 1 quad exists, no loop iterations ran — pad to max_neighbors
        while len(neighbor_states) < self.max_neighbors:
            neighbor_states.append(np.zeros(3, dtype=np.float32))
            neighbor_mask.append(True)
        
        # Extract local fire map (used for local_map output and fire_info above)
        local_map = self._extract_local_fire_map(pos)

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
        return np.array(distances) / 2000.0

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

        Returns a dict with a single key: self_state (17-float vector).
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
            rel_base_x = (refill_pos[0] - pos[0]) / self.map_bounds
            rel_base_y = (refill_pos[1] - pos[1]) / self.map_bounds
        else:
            rel_base_x = 0.0
            rel_base_y = 0.0

        # --- Normalise ---
        norm_pos = pos / self.map_bounds        # x,y,z in roughly [-1, 1]
        norm_vel = vel / 20.0                   # vx,vy,vz in roughly [-1, 1]
        norm_rpy = rpy / np.pi                  # angles from [-pi,pi] to [-1,1]

        # Normalised distances to each map edge
        dist_boundaries = self._get_boundary_measurements_norm(pos)

        # --- Danger flag ---
        # 1.0 when the aircraft is within 300 m of any edge, otherwise 0.0.
        # At ~20 m/s cruise speed, 300 m ≈ 15 s to the wall — enough lead time
        # to initiate a turn.  The flag gives the policy a clear binary signal
        # to start turning before the physics shaping penalty kicks in.
        danger_flag = 1.0 if min(self._get_boundary_measurements(pos)) < 300.0 else 0.0

        self_state = np.array([
            norm_pos[0], norm_pos[1], norm_pos[2],                            # 0-2  : position
            norm_vel[0], norm_vel[1], norm_vel[2],                            # 3-5  : velocity
            dist_boundaries[0], dist_boundaries[1],                           # 6-7  : north / south edge
            dist_boundaries[2], dist_boundaries[3],                           # 8-9  : east / west edge
            water_lvl,                                                        # 10   : water level
            rel_base_x, rel_base_y,                                           # 11-12: compass to refill zone
            norm_rpy[0], norm_rpy[1], norm_rpy[2],                            # 13-15: roll, pitch, yaw
            danger_flag                                                        # 16   : proximity alert
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

        # --- 1. Restore the full agent list and reset action histories ---
        # self.agents starts as a copy of possible_agents and is pruned as
        # agents are terminated during the episode.
        self.agents = self.possible_agents[:]
        self.last_actions = {agent: np.zeros(4, dtype=np.float32) for agent in self.possible_agents}
        self.last_raw_actions = {agent: np.zeros(4, dtype=np.float32) for agent in self.possible_agents}

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

        # --- Fire start position: curriculum learning ---
        # For the first 2000 episodes the fire is fixed at the map centre [0, 0]
        # so the agents can learn a basic "fly to fire" behaviour without the
        # added difficulty of searching a large arena.
        # After episode 2000 the fire spawns at a random position inside a
        # 60%-of-half-map radius, forcing generalisation across scenarios.
        # if epizode_number < 2000:
        #     safe_zone = self.map_bounds * 0.1
        #     self.fire_x = 0.0
        #     self.fire_y = 0.0
        # else:
        safe_zone = self.map_bounds * 0.6
        self.fire_x = random.uniform(-safe_zone, safe_zone)
        self.fire_y = random.uniform(-safe_zone, safe_zone)

        self.sim.start_fire([self.fire_x, self.fire_y], intensity=0.5)
        self.current_episode = epizode_number

        # --- Refill zone: placed on the opposite side of the map from the fire ---
        # This placement motivates the commander to shuttle between fire and base.
        # A small random offset prevents the policy from memorising a fixed landmark.
        refill_x = float(np.clip(-self.fire_x + random.uniform(-20, 20), -self.map_bounds * 0.8, self.map_bounds * 0.8))
        refill_y = float(np.clip(-self.fire_y + random.uniform(-20, 20), -self.map_bounds * 0.8, self.map_bounds * 0.8))
        self.sim.environment.create_refill_zone(center_pos=[refill_x, refill_y, 0.0], size=20.0)

        # --- Drone spawn positions: curriculum learning ---
        # For the first 300 episodes all drones start very close to the map
        # centre (within 10 m) so the fire is always visible immediately.
        # After episode 300 the spawn radius expands to safe_zone, requiring
        # agents to actively navigate to find the fire.
        if epizode_number < 300:
            start_x = random.uniform(-10, 10)
            start_y = random.uniform(-10, 10)
            start_z = random.uniform(10.0, 40.0)
        else:
            start_x = random.uniform(-safe_zone, safe_zone)
            start_y = random.uniform(-safe_zone, safe_zone)
            start_z = random.uniform(10.0, 40.0)

        # --- 4. Spawn every agent at its starting position ---
        for agent in self.agents:
            if "fixed" in agent:
                # Fixed-wing spawns at an offset from the quad start to avoid
                # mid-air collisions at the very beginning of the episode.
                fw_start_x = float(np.clip(start_x + random.uniform(-50, 50), -self.map_bounds * 0.6, self.map_bounds * 0.6))
                fw_start_y = float(np.clip(start_y + random.uniform(-50, 50), -self.map_bounds * 0.6, self.map_bounds * 0.6))
                # Point the aircraft toward the map centre so it immediately
                # starts flying inward rather than away from the action.
                to_center_vec = -np.array([fw_start_x, fw_start_y])
                yaw_to_center = np.arctan2(to_center_vec[1], to_center_vec[0])

                self.sim.add_fixedwing(agent, position=[fw_start_x, fw_start_y, 60.0], water_capacity=200.0, yaw=yaw_to_center)

                # Set a meaningful initial airspeed so the fixed-wing physics
                # are in a stable flight regime from frame 0.
                drone = self.sim.drones[agent]
                drone.state_va = 15.0
            else:
                # Quads spawn at a lower altitude — they hover in place so
                # altitude at start doesn't matter much.
                self.sim.add_quadcopter(agent, position=[start_x, start_y, start_z])

        # --- 5. Initialise per-episode tracking variables ---
        self.visited_cells = set()   # tracks which grid cells have been overflown
        self.fire_discovered = False
        self.current_step = 0

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
        jerk_penalties = {}

        for agent_name, action in actions.items():
            smooth_action = 0.8 * self.last_actions[agent_name] + 0.2 * action
            self.last_actions[agent_name] = smooth_action  # save for next step

            jerk_diff = np.sum(np.abs(action - self.last_raw_actions[agent_name]))
            jerk_penalties[agent_name] = jerk_diff * 0.02  # Weight of penalty
            self.last_raw_actions[agent_name] = np.copy(action)

            if "fixed" in agent_name:
                # Fixed-wing action mapping:
                #   [0] Roll    : pass through unchanged
                #   [1] Pitch   : pass through unchanged
                #   [2] Throttle: NN output [-1,1] -> physics range [0.4, 1.0]
                #       0.4 is the minimum throttle to avoid stall.
                #   [3] Water   : NN output [-1,1] -> trigger [0.0, 1.0]
                mapped_action = np.copy(smooth_action)
                mapped_action[2] = 0.4 + (smooth_action[2] + 1.0) * 0.3
                mapped_action[3] = (smooth_action[3] + 1.0) / 2.0
                drone_controls[agent_name] = mapped_action
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
        
        for agent in self.agents:
            # Default to alive / not truncated
            terminations[agent] = False
            truncations[agent] = time_is_up
            infos[agent] = {}
            
            # Check whether the agent has crashed or left the map
            dead, crash_reward = self._check_death(agent)

            rewards[agent] = 0
            
            if dead:
                terminations[agent] = True
                rewards[agent] += crash_reward
                    
            else:
                rewards[agent] += self._apply_physics_shaping(agent)
                rewards[agent] -= jerk_penalties.get(agent, 0.0)

                # Agent-type-specific mission reward
                if "fixed" in agent:
                    survival = self._get_fixed_reward_survival(agent)
                    mission = self._get_fixed_reward(agent)
                    # Curriculum blending: linearly increase mission weight with episode count
                    ep = getattr(self, 'current_episode', 0)
                    if ep < 500:
                        rewards[agent] += survival
                    elif ep < 2000:
                        blend = (ep - 500) / 1500.0  # linearly 0.0 -> 1.0
                        rewards[agent] += survival * (1.0 - blend * 0.7) + mission * (blend * 0.7)
                    else:
                        rewards[agent] += survival * 0.3 + mission * 0.7
                else:
                    rewards[agent] += self._get_quad_reward(agent)
                
                # if time_is_up:
                #     rewards[agent] += 10.0
                
                rewards[agent] *= 0.1  # scale down rewards to keep them in a reasonable range
                rewards[agent] = np.clip(rewards[agent], -10.0, 10.0)
            
                # If alive: store observation and keep the agent in the active list.
            # If dead: PettingZoo requires us to still return a final observation
            # for the terminated agent (the "terminal observation" convention).
            if not terminations[agent]:
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
        # When the commander successfully drops water on fire (eff > 0), both
        # the commander AND all scouts receive the bonus.  This is the key
        # credit-assignment mechanism: scouts only observe fire and generate
        # messages; they cannot extinguish directly.  By giving them a share of
        # the extinguish bonus we provide them with a training signal that their
        # messages (which guided the commander) were useful.
        for f_agent in self.fixed_agents:
            # Safety check: skip agents that were terminated this step or
            # never received an action (would cause KeyError otherwise).
            if f_agent not in drone_controls or f_agent not in rewards:
                continue

            eff = self.sim.drone_extinguish_stats.get(f_agent, 0.0)

            # Check if the commander is currently dropping water (trigger > 0.5)
            is_dropping = drone_controls[f_agent][3] > 0.5

            if eff > 0.0:
                # Extinguish bonus: proportional to how much fire was put out
                fire_bonus = eff * 0.2
                rewards[f_agent] += fire_bonus

                # Scouts get 100% of the same bonus — they guided the commander
                for q_agent in [a for a in self.quad_agents if a in rewards]:
                    rewards[q_agent] += fire_bonus

            elif is_dropping:
                # Waste penalty: aircraft is spraying but hitting nothing.
                # This discourages random water release when not on target.
                rewards[f_agent] += -0.05

        return observations, rewards, terminations, truncations, infos
    

    # =========================================================================
    # Private reward helper methods
    # =========================================================================

    def _apply_physics_shaping(self, agent):
        """Common physics-based shaping rewards applied to both agent types.

        Returns a small scalar reward value combining:
          - A survival bonus (tiny positive reward every step alive).
          - A boundary proximity penalty (quadratic, activates near map edges).
          - An altitude violation penalty (linear, for flying too high or low).
        """
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        reward = 0.01  # small per-step survival bonus

        # --- Boundary proximity penalty ---
        # As the agent approaches any map edge the penalty grows quadratically.
        # Threshold is 25% of map_bounds/2 for fixed-wing (they need more room
        # to turn) and 10% for quads (more manoeuvrable).
        dist_boundaries = self._get_boundary_measurements(pos)
        dist_to_edge = min(dist_boundaries)
        threshold = self.map_bounds / 2 * 0.25 if "fixed" in agent else self.map_bounds / 2 * 0.1
        if dist_to_edge < threshold:
            # (1 - d/T)^2 gives 0 at the threshold and 1 at d=0 (wall contact)
            reward -= 0.3 * (1.0 - dist_to_edge / threshold)**2

        # Stricter extra penalty for fixed-wing when very close to the edge
        # (< 50% of threshold).  Fixed-wing needs a larger safety margin
        # because it cannot stop or turn on the spot.
        if "fixed" in agent:
            if dist_to_edge < threshold * 0.5:
                reward -= 0.5

        # --- Altitude limit penalty ---
        max_alt = 200.0 if "fixed" in agent else 150.0
        min_alt = 15.0 if "fixed" in agent else 35.0
        if pos[2] > max_alt or pos[2] < min_alt:
            reward -= 0.05

        return reward

    def _get_quad_reward(self, agent):
        """Compute the mission reward for a quadrotor scout.

        The scout is rewarded for hovering over fire: the more intense the
        fire beneath the drone (as seen in its local fire map), the higher
        the reward.  A speed penalty discourages fast fly-overs and encourages
        the drone to slow down and loiter above the fire so its observations
        are more useful to the commander.
        """
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        reward = 0

        # Use the same altitude-adaptive view as the sensor to compute reward
        reward_zone = self._extract_local_fire_map(pos)
        avg_fire_intensity = np.mean(reward_zone)

        if avg_fire_intensity > 0.001:
            reward += 1                              # flat bonus for being above fire
            reward += (avg_fire_intensity * 10)     # proportional intensity bonus
            # Speed penalty: slow down over fire for better observation
            speed = np.linalg.norm(drone.get_velocity())
            reward -= speed * 0.05

        return reward

    def _get_fixed_reward_survival(self, agent):
        """Compute the survival/flight-quality reward for a fixed-wing commander.

        This reward function replaces the mission-specific reward during the
        early curriculum phase (episodes < 500) to teach the aircraft to fly
        stably before it learns the fire-fighting mission.

        It has two components:
          1. "Donut" bonus — a flat bonus for staying within 500 m of the map
             centre.  Beyond that the bonus decays linearly to zero at the
             map boundary, punishing extended excursions to the edge.
          2. "Rubber band" (approach speed) — when the aircraft is outside
             500 m, it receives a reward proportional to how fast it is flying
             *toward* the centre.  This is computed via the dot product between
             the velocity vector and the unit vector pointing to the centre:
               approach_speed = v . (centre - pos) / |centre - pos|
             Positive values mean the aircraft is heading inward (rewarded);
             negative values mean it is heading outward (penalised).  This
             reliably forces the policy to initiate a turn toward the centre
             once it crosses the 500 m ring, which is otherwise hard to learn
             purely from the boundary proximity penalty.
        """
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        vel = drone.get_velocity()
        reward = 0.05  # base per-step survival bonus

        dist_from_center = np.linalg.norm(pos[:2])

        # --- Component 1: Donut bonus ---
        if dist_from_center < 500.0:
            # Inside the safe ring: flat bonus regardless of exact position
            reward += 0.15
        else:
            # Outside the safe ring: linearly decaying area bonus
            reward += max(0.0, 0.15 * (1.0 - ((dist_from_center - 500.0) / (self.map_bounds - 500.0))))

            # --- Component 2: Rubber-band approach reward ---
            # Unit vector pointing from the aircraft toward the map centre [0, 0]
            vec_to_center = -pos[:2]
            dir_to_center = vec_to_center / dist_from_center
            # Dot product: positive = aircraft velocity has a component toward centre
            # Large positive approach_speed -> aircraft is actively flying back in
            approach_speed = np.dot(vel[:2], dir_to_center)
            reward += approach_speed * 0.02

        return reward


    def _calculate_orbital_reward(self, pos, vel, target_pos, ideal_radius=150.0):
        """Calculate reward for loitering / orbiting around a target point.

        The reward function divides the space around the target into three
        concentric zones based on distance:

        Zone 1 — Too close  (dist < 0.6 * ideal_radius)
            The aircraft is inside the minimum turn radius.  For a fixed-wing
            aircraft this means it cannot maintain a stable orbit and risks
            flying directly over / into the target.  Penalised.

        Zone 2 — Sweet spot  (0.6 * IR <= dist <= 1.5 * ideal_radius)
            Ideal loiter ring.  The aircraft receives a base bonus (0.2) plus
            a tangential-velocity bonus that encourages circular flight:
              tangential_bonus = (1 - |radial_speed| / max_speed) * 0.05
            When all velocity is tangential (radial_speed = 0), the tangential
            bonus is maximised.  When the aircraft is flying directly toward or
            away from the target (radial_speed = max_speed), the bonus is zero.

        Zone 3 — Too far  (dist > 1.5 * ideal_radius)
            The aircraft needs to navigate toward the target.  It is rewarded
            proportional to its approach speed (dot product of velocity with the
            unit vector toward the target) and a small proximity bonus.

        Parameters
        ----------
        pos        : position of the aircraft (uses only x,y)
        vel        : velocity of the aircraft (uses only x,y)
        target_pos : 2-D (x, y) world-coordinate target
        ideal_radius : desired loiter radius in metres
        """
        vec_to_target = target_pos - pos[:2]
        dist = np.linalg.norm(vec_to_target)

        if dist < ideal_radius * 0.6:
            # Zone 1: too close — inside minimum turn radius
            return -0.05
        elif ideal_radius * 0.6 <= dist <= ideal_radius * 1.5:
            # Zone 2: sweet spot — reward circular (tangential) flight
            dir_to_target = vec_to_target / (dist + 1e-6)
            radial_speed = np.dot(vel[:2], dir_to_target)  # positive = moving toward target
            return 0.2 + (1.0 - abs(radial_speed / 20.0)) * 0.05
        else:
            # Zone 3: too far — reward approach toward target
            dir_to_target = vec_to_target / (dist + 1e-6)
            approach_speed = np.dot(vel[:2], dir_to_target)
            return approach_speed * 0.05 + (1.0 - dist / self.grid_size_m) * 0.01

    def _get_fixed_reward(self, agent):
        """Compute the mission reward for a fixed-wing commander (3-state machine).

        The commander's mission reward is driven by a priority-ordered state
        machine with three possible states:

        State 1 — MISSION (highest priority)
            A scout can currently see fire (max_seen_intensity > 0.1).
            The commander should fly to the fire and orbit it so it can drop
            water.  Reward: orbital reward around the best known fire position.

        State 2 — REFILL
            No scout can see fire AND water level is below 10%.
            The commander must return to the refill zone to restock.
            Reward: orbital reward around the refill zone position.

        State 3 — PATROL (lowest priority / fallback)
            No fire visible AND water level is adequate.
            The commander patrols the map centre, staying ready for when a
            scout discovers the fire.
            Reward: orbital reward around the map centre [0, 0].

        Note on ground-truth vs messages:
            During training, the fire-seen signal is read *directly* from the
            scout observations (ground truth).  In deployment the scouts' 3-
            float messages would be used instead.  The policy must learn the
            correct correlation between messages and fire location during
            training, so that it can respond to messages in deployment.
        """
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        vel = drone.get_velocity()
        water_lvl = drone.current_water / drone.water_capacity if drone.water_capacity > 0 else 0.0

        # --- Find the best fire signal reported by any scout ---
        # We scan all quads and take the one with the highest dynamic intensity
        # (self_state[14]) as the most authoritative fire report.
        max_seen_intensity = 0.0
        best_fire_pos = None

        for q_name in self.quad_agents:
            if q_name in self.sim.drones:
                q_obs = self._get_quad_obs(q_name)
                q_intensity = q_obs["self_state"][14]  # dynamic intensity index
                if q_intensity > max_seen_intensity:
                    max_seen_intensity = q_intensity
                    # Reconstruct approximate world coordinates of the fire
                    # centroid from the scout's normalised local-map centroid.
                    rel_x = q_obs["self_state"][12]
                    rel_y = q_obs["self_state"][13]
                    q_pos = self.sim.drones[q_name].get_position()
                    fov_size = max(10.0, q_pos[2] * 1.5)
                    target_x = q_pos[0] + (rel_x * (fov_size / 2.0))
                    target_y = q_pos[1] + (rel_y * (fov_size / 2.0))
                    best_fire_pos = np.array([target_x, target_y])

        # --- State 1: MISSION — orbit the fire and drop water ---
        if max_seen_intensity > 0.1:
            return self._calculate_orbital_reward(pos, vel, best_fire_pos, ideal_radius=150.0)

        # --- State 2: REFILL — return to base when empty  ---
        elif water_lvl < 0.1:
            if self.sim.environment.refill_zone:
                target = self.sim.environment.refill_zone['position'][:2]
                return self._calculate_orbital_reward(pos, vel, target, ideal_radius=80.0)

        # --- State 3: PATROL — loiter near the map centre while waiting ---
        else:
            return self._calculate_orbital_reward(pos, vel, np.array([0.0, 0.0]), ideal_radius=300.0)

        return 0.0


    # =========================================================================
    # Death detection
    # =========================================================================

    def _check_death(self, agent):
        """Check whether an agent has crashed or left the map boundary.

        Returns
        -------
        dead : bool
            True if the agent should be removed from the simulation.
        penalty : float
            A large negative reward applied at the moment of termination.

        Two termination conditions are checked:
          1. Physics crash: the agent's drone is no longer in self.sim.drones.
             This happens when the PyBullet physics backend detects a hard
             collision and removes the drone object.
          2. Map boundary violation: the agent's (x, y) position exceeds
             map_bounds.  The drone is explicitly destroyed to keep the
             simulation consistent, then True is returned.
        """
        if agent not in self.sim.drones:
            # Drone was removed by the physics engine (crash)
            return True, -50

        pos = self.sim.drones[agent].get_position()
        if abs(pos[0]) > self.map_bounds or abs(pos[1]) > self.map_bounds:
            # Drone flew out of the allowed area — destroy it and penalise
            self.sim._destroy_drone(agent)
            return True, -50

        return False, 0.0