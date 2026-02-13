import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import sys

import torchrl.data as _data

# 1. Handle CompositeSpec
if hasattr(_data, "CompositeSpec"):
    CompositeSpec = _data.CompositeSpec
elif hasattr(_data, "Composite"):
    CompositeSpec = _data.Composite
else:
    raise ImportError("Could not find CompositeSpec or Composite in torchrl.data")

# 2. Handle UnboundedContinuousTensorSpec
if hasattr(_data, "UnboundedContinuousTensorSpec"):
    UnboundedContinuousTensorSpec = _data.UnboundedContinuousTensorSpec
elif hasattr(_data, "UnboundedContinuous"):
    UnboundedContinuousTensorSpec = _data.UnboundedContinuous
elif hasattr(_data, "UnboundedTensorSpec"): # Fallback for some versions
    UnboundedContinuousTensorSpec = _data.UnboundedTensorSpec
else:
    raise ImportError("Could not find UnboundedContinuousTensorSpec")

# 3. Handle BoundedContinuousTensorSpec
if hasattr(_data, "BoundedContinuousTensorSpec"):
    BoundedContinuousTensorSpec = _data.BoundedContinuousTensorSpec
elif hasattr(_data, "BoundedTensorSpec"):
    BoundedContinuousTensorSpec = _data.BoundedTensorSpec
elif hasattr(_data, "Bounded"):
    BoundedContinuousTensorSpec = _data.Bounded
else:
    raise ImportError("Could not find BoundedContinuousTensorSpec")

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.wildfire_obs_processor import WildfireObsProcessor
from config import WildfireGymConfig, MainConfig
class WildfireMARLEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, agents_config=["quad_1", "fw_1"], demo_mode=False):
        super().__init__()
        self.sim = Simulation()
        self.demo_mode = demo_mode  # Flag for unlimited runtime in demo
        
        # Observation window SYNCHRONIZED with processor defaults
        self.obs_proc = WildfireObsProcessor(window_size_m=30.0, resolution_px=32)  # CONSISTENT 30m window
        
        # BOUNDARY LIMITS PRO PENALIZACI - USE CONFIG!
        self.map_bounds = WildfireGymConfig.MAP_BOUNDS  # ±100m boundaries from config
        
        self.quad_agents = [a for a in agents_config if "quad" in a.lower()]
        self.fixed_agents = [a for a in agents_config if "fw" in a.lower() or "fixed" in a.lower()]
        self.all_agents = self.quad_agents + self.fixed_agents

        # DYNAMICKÁ KONSTRUKCE SPECS
        obs_dict = {}
        obs_spec_dict = {}
        action_dict = {}
        action_spec_dict = {}

        # Přidat kvadrokoptéry pouze pokud existují
        if self.quad_agents:
            n = len(self.quad_agents)
            self_state_size = self.obs_proc.get_self_state_size()  # Get actual size from processor
            obs_dict["quads"] = spaces.Dict({
                "local_map": spaces.Box(low=0.0, high=1.0, 
                    shape=(n, 1, WildfireGymConfig.OBSERVATION_RESOLUTION, WildfireGymConfig.OBSERVATION_RESOLUTION), 
                    dtype=np.float32),
                "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, self_state_size), dtype=np.float32),
                "hidden_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, MainConfig.ACTOR_HIDDEN_SIZE), dtype=np.float32),
            })
            obs_spec_dict["quads"] = CompositeSpec({
                "local_map": UnboundedContinuousTensorSpec(
                    shape=(n, 1, WildfireGymConfig.OBSERVATION_RESOLUTION, WildfireGymConfig.OBSERVATION_RESOLUTION)),
                "self_state": UnboundedContinuousTensorSpec(shape=(n, self_state_size)),
                "hidden_state": UnboundedContinuousTensorSpec(shape=(n, MainConfig.ACTOR_HIDDEN_SIZE)),
            })
            action_dict["quads"] = spaces.Dict({
                "action": spaces.Box(low=WildfireGymConfig.ACTION_LOW, high=WildfireGymConfig.ACTION_HIGH, 
                    shape=(n, WildfireGymConfig.ACTION_DIMENSIONS), dtype=np.float32)
            })
            action_spec_dict["quads"] = CompositeSpec({
                "action": BoundedContinuousTensorSpec(
                    low=WildfireGymConfig.ACTION_LOW, high=WildfireGymConfig.ACTION_HIGH, 
                    shape=(n, WildfireGymConfig.ACTION_DIMENSIONS))
            })

        # Přidat letadla pouze pokud existují
        if self.fixed_agents:
            n = len(self.fixed_agents)
            obs_dict["fixed"] = spaces.Dict({
                "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, 7), dtype=np.float32),
            })
            obs_spec_dict["fixed"] = CompositeSpec({
                "self_state": UnboundedContinuousTensorSpec(shape=(n, 7)),
            })
            action_dict["fixed"] = spaces.Dict({
                "action": spaces.Box(low=-1.0, high=1.0, shape=(n, 4), dtype=np.float32)
            })
            action_spec_dict["fixed"] = CompositeSpec({
                "action": BoundedContinuousTensorSpec(low=-1, high=1, shape=(n, 4))
            })

        # Globální pozorování zůstává vždy
        obs_dict["global_observation"] = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)
        obs_spec_dict["global_observation"] = UnboundedContinuousTensorSpec(shape=(512,))

        self.action_space = spaces.Dict(action_dict)
        self.observation_space = spaces.Dict(obs_dict)
        self.observation_spec = CompositeSpec(obs_spec_dict)
        self.action_spec = CompositeSpec(action_spec_dict)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        
        # CORRECTED exploration grid calculation: 1m cells on 200x200m map = 200x200 = 40000 cells
        grid_size = int((2 * self.map_bounds) / 1.0)  # 200m / 1m = 200 cells per side
        exploration_grid_cells = grid_size * grid_size  # 200 * 200 = 40000 cells total
        self.exploration_grid_cells = exploration_grid_cells
        
        # ===== EXPLORATION TRACKING PRO NOVÝ REWARD SYSTÉM =====
        self.visited_cells = set()  # Track visited 1x1m cells
        self.previous_positions = {}  # Track drone movement
        self.episode_start_time = 0.0
        self.total_fire_discovered = False
        self.exploration_reward_accumulated = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Set random seed for reproducible randomness
        if seed is not None:
            np.random.seed(seed)
        
        self.sim.start_simulation()
        
        # ===== ENABLE FIRE SIMULATION =====
        self.sim.enable_fire_simulation(
            grid_width_m=WildfireGymConfig.ENVIRONMENT_GRID_WIDTH,
            grid_height_m=WildfireGymConfig.ENVIRONMENT_GRID_HEIGHT,
            cell_size_m=WildfireGymConfig.ENVIRONMENT_CELL_SIZE,
            dt=WildfireGymConfig.ENVIRONMENT_DT
        )
        
        # ===== POSITION SETUP (RANDOM FOR TRAINING, FIXED FOR DEMO) =====
        if self.demo_mode and MainConfig.DEMO_FIXED_POSITIONS:
            # FIXED positions for consistent demo testing
            random_drone_pos = MainConfig.DEMO_DRONE_POSITION
            fire_x, fire_y = MainConfig.DEMO_FIRE_POSITION
        else:
            # RANDOM positions for training variety with CONTROLLED DISTANCE
            safe_margin = 20.0  # Keep 20m away from edges
            min_distance = 30.0  # Minimum 30m between drone and fire
            max_distance = 60.0  # Maximum 60m between drone and fire
            
            # Generate positions until we get good distance
            attempts = 0
            while attempts < 50:  # Safety limit
                # Random drone position
                drone_x = np.random.uniform(-self.map_bounds + safe_margin, self.map_bounds - safe_margin)
                drone_y = np.random.uniform(-self.map_bounds + safe_margin, self.map_bounds - safe_margin) 
                random_drone_pos = [drone_x, drone_y, 10.0]  # Always start at 10m height
                
                # Random fire position anywhere within boundaries
                fire_x = np.random.uniform(-self.map_bounds + 10, self.map_bounds - 10)  # 10m margin
                fire_y = np.random.uniform(-self.map_bounds + 10, self.map_bounds - 10)
                
                # Check distance
                distance = np.sqrt((drone_x - fire_x)**2 + (drone_y - fire_y)**2)
                
                if min_distance <= distance <= max_distance:
                    break
                    
                attempts += 1
            
            # If we couldn't find good positions after 50 attempts, use defaults
            if attempts >= 50:
                random_drone_pos = [-60.0, -60.0, 10.0]  # České komentáře: bezpečná pozice uvnitř hranic
                fire_x, fire_y = 0.0, 0.0  # Distance ~85m

        # Setup scenario with chosen positions
        if len(self.quad_agents) >= 1:
            self.sim.add_quadcopter(self.quad_agents[0], position=random_drone_pos)
        
        # Fire position
        self.sim.environment.ignite_fire(
            x=fire_x, 
            y=fire_y, 
            intensity=WildfireGymConfig.FIRE_INTENSITY
        )
        
        # ===== RESET EXPLORATION TRACKING =====
        self.visited_cells = set()
        self.previous_positions = {}
        self.episode_start_time = self.sim.simulation_time
        self.total_fire_discovered = False
        self.exploration_reward_accumulated = 0.0
        
        # Initialize previous positions for all drones
        for name in self.quad_agents:
            if name in self.sim.drones:
                self.previous_positions[name] = self.sim.drones[name].get_position()[:2]

        return self._get_obs(), {}

    def _get_obs(self):
        """Helper to fetch standardized observations and a centralized global state."""
        obs = {} # Inicializujeme prázdný slovník místo fixních klíčů
        
        # 1. Fetch Local Observations
        # Fetch Quads - pouze pokud existují
        if self.quad_agents:
            obs["quads"] = {}
            q_maps, q_states = [], []
            for name in self.quad_agents:
                if name in self.sim.drones:
                    data = self.obs_proc.fetch(self.sim, name)

                    # Full enhanced self_state with boundary awareness
                    full_state = data["self_state"].copy()  # Use actual size from processor
                    
                    # Update exploration and fire discovery ratios with real values
                    exploration_percentage = len(self.visited_cells) / self.exploration_grid_cells
                    fire_discovery_flag = 1.0 if self.total_fire_discovered else 0.0
                    
                    # Update exploration tracking in the state vector
                    full_state[14] = exploration_percentage  # Exploration ratio
                    full_state[15] = fire_discovery_flag     # Fire discovery flag
                    
                    q_maps.append(data["local_map"])
                    q_states.append(full_state)
                else:
                    q_maps.append(np.zeros((1, 32, 32), dtype=np.float32))
                    q_states.append(np.zeros(self.obs_proc.get_self_state_size(), dtype=np.float32))
            
            obs["quads"]["local_map"] = np.stack(q_maps)
            obs["quads"]["self_state"] = np.stack(q_states)

        # Fetch Fixed-Wings - pouze pokud existují
        if self.fixed_agents:
            obs["fixed"] = {}
            f_states = []
            for name in self.fixed_agents:
                if name in self.sim.drones:
                    data = self.obs_proc.fetch(self.sim, name)
                    f_states.append(data["self_state"])
                else:
                    f_states.append(np.zeros(7, dtype=np.float32))
            
            obs["fixed"]["self_state"] = np.stack(f_states)

        # 2. Construct Global Observation for the Critic
        full_fire_grid = self.sim.environment.fire_grid.I
        res = 16
        fire_summary = self._downsample_grid(full_fire_grid, res).flatten()

        agent_states = []
        key_features_count = 8  # First 8 most important features for global obs
        for name in self.all_agents:
            if name in self.sim.drones:
                # Use first N most important features for global obs (avoid making it too large)
                state = self.obs_proc.fetch(self.sim, name)["self_state"][:key_features_count]
                agent_states.append(state)
            else:
                agent_states.append(np.zeros(key_features_count, dtype=np.float32))
        
        max_agents = 8  # Maximum supported agents in global observation
        while len(agent_states) < max_agents:
            agent_states.append(np.zeros(key_features_count))
        all_agents_vector = np.concatenate(agent_states).astype(np.float32)

        wind = self.sim.environment.weather['wind_velocity']
        sim_time = np.array([self.sim.simulation_time])
        
        combined_global = np.concatenate([fire_summary, all_agents_vector, wind, sim_time])
        padding = np.zeros(512 - combined_global.shape[0])
        obs["global_observation"] = np.concatenate([combined_global, padding]).astype(np.float32)
        
        return obs

    def _downsample_grid(self, grid, target_res):
        """Utility to shrink the large fire grid into a small summary vector."""
        h, w = grid.shape
        s_h, s_w = h // target_res, w // target_res
        # Reshape and mean to downsample
        return grid[:target_res*s_h, :target_res*s_w].reshape(target_res, s_h, target_res, s_w).mean(axis=(1, 3))

    def step(self, actions):
        # 1. Map actions from RL to Simulation
        drone_controls = {}
        
        # Store actions for stop bonus calculation
        if not hasattr(self, 'last_actions'):
            self.last_actions = {}

        
        # Map Quads (Unpack from nested "action" key)
        if "quads" in actions and "action" in actions["quads"]:
            # GymWrapper passes this as a numpy array now
            quad_acts = actions["quads"]["action"]
            for i, name in enumerate(self.quad_agents):
                # Ensure we handle shape (1, 4) vs (4,)
                act = quad_acts[i]
                self.last_actions[name] = act  # Store for stop bonus
                drone_controls[name] = act
        
        # Map Fixed-Wings (Unpack from nested "action" key)
        if "fixed" in actions and "action" in actions["fixed"]:
            fixed_acts = actions["fixed"]["action"]
            for i, name in enumerate(self.fixed_agents):
                act = fixed_acts[i]
                va = (act[0] + 1) * WildfireGymConfig.FIXED_WING_VELOCITY_SCALE + WildfireGymConfig.FIXED_WING_VELOCITY_MIN
                gamma = act[1] * WildfireGymConfig.FIXED_WING_GAMMA_SCALE
                phi = act[2] * WildfireGymConfig.FIXED_WING_PHI_SCALE
                drop = 1.0 if act[3] > WildfireGymConfig.FIXED_WING_DROP_THRESHOLD else 0.0
                drone_controls[name] = [va, gamma, phi, drop]

        # 2. Physics Step
        self.sim.step_simulation(drone_controls)

        # 3. Collect new state
        new_obs = self._get_obs()

        # 4. ===== EXPLORATION-BASED REWARD SYSTÉM =====
        reward = 0.0  
        terminated = False  # Initialize termination flag
        
        # Tracking for fire discovery
        total_fire_visible = False
        
        for drone_idx, drone_name in enumerate(self.quad_agents):
            if drone_name and drone_name in self.sim.drones:
                drone_pos = self.sim.drones[drone_name].get_position()
                x, y, z = drone_pos
                
                # === 1. IMPROVED EXPLORATION SYSTEM ===
                # Convert position to grid cell (1m resolution - finer tracking!)
                cell_x = int(x // WildfireGymConfig.ENVIRONMENT_CELL_SIZE)
                cell_y = int(y // WildfireGymConfig.ENVIRONMENT_CELL_SIZE)
                cell_key = (cell_x, cell_y)

                # Check if drone is within map boundaries for exploration tracking
                within_boundaries = abs(x) <= self.map_bounds and abs(y) <= self.map_bounds

                if within_boundaries:
                    # Always reward exploration within boundaries
                    if cell_key not in self.visited_cells:
                        self.visited_cells.add(cell_key)
                        
                        if not self.total_fire_discovered:
                            # Pre-discovery: FLAT exploration reward (no distance bias)
                            base_exploration = 3.0
                            reward += base_exploration
                            self.exploration_reward_accumulated += base_exploration
                            
                            # Extra bonus for systematic exploration
                            exploration_count = len(self.visited_cells)
                            if exploration_count % 10 == 0:  # Every 10 new cells
                                milestone_bonus = 10.0
                                reward += milestone_bonus
                        else:
                            # Post-discovery: Still meaningful exploration (fire spreads!)
                            reward += 1.5  # Increased - fire monitoring is important!
                    
                    # REVISIT BONUS: Even visited cells can have new fire!
                    else:
                        # Small bonus for revisiting areas (fire monitoring/spread detection)
                        if self.total_fire_discovered:
                            revisit_bonus = 0.2  # Small but meaningful
                            reward += revisit_bonus
                        
                        # Extra bonus if returning to area where fire was previously detected
                        if "quads" in new_obs and "local_map" in new_obs["quads"] and len(new_obs["quads"]["local_map"]) > drone_idx:
                            local_fire = new_obs["quads"]["local_map"][drone_idx, 0]
                            fire_intensity = np.sum(local_fire)
                            if fire_intensity > 0.1:  # Fire detected in revisited area
                                monitoring_bonus = 2.0  # Good for tracking fire spread
                                reward += monitoring_bonus

                else:
                    # Outside boundaries - no exploration rewards, but use boundary return system
                     if drone_name in self.previous_positions:
                        prev_x, prev_y = self.previous_positions[drone_name]
                        
                        # Calculate movement towards boundaries
                        current_boundary_violation = max(abs(x) - self.map_bounds, abs(y) - self.map_bounds, 0)
                        prev_boundary_violation = max(abs(prev_x) - self.map_bounds, abs(prev_y) - self.map_bounds, 0)
                        
                        # MASSIVE bonus for moving towards boundaries when outside
                        if prev_boundary_violation > current_boundary_violation:
                            return_progress = prev_boundary_violation - current_boundary_violation
                            reward += return_progress * 50.0  # HUGE bonus for returning to boundaries!
                
                # === 2. LOCAL FIRE TRACKING SYSTEM (based on drone's local observation only) ===
                if drone_name in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[drone_name]
                    movement_distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                    
                    # Get current and previous fire observations from LOCAL MAP only
                    current_fire_visible = False
                    current_fire_intensity = 0.0
                    fire_center_x, fire_center_y = None, None
                    
                    if "quads" in new_obs and "local_map" in new_obs["quads"] and len(new_obs["quads"]["local_map"]) > drone_idx:
                        local_fire = new_obs["quads"]["local_map"][drone_idx, 0]  # Shape (32, 32)
                        current_fire_intensity = np.sum(local_fire)
                        
                        if current_fire_intensity > 0.1:  # Fire visible in local map
                            current_fire_visible = True
                            
                            # Calculate fire center within LOCAL MAP coordinates
                            fire_indices = np.where(local_fire > 0.1)
                            if len(fire_indices[0]) > 0:
                                # Convert local map indices to relative positions
                                window_size = self.obs_proc.window_size_m  # 30m window
                                resolution = self.obs_proc.res_px  # 32px resolution
                                
                                # Fire center in local map coordinates (relative to drone)
                                fire_center_i = np.mean(fire_indices[0])  # Row (Y direction)
                                fire_center_j = np.mean(fire_indices[1])  # Col (X direction)
                                
                                # Convert to world coordinates relative to drone position
                                meters_per_pixel = window_size / resolution
                                fire_center_x = x + (fire_center_j - resolution/2) * meters_per_pixel
                                fire_center_y = y + (fire_center_i - resolution/2) * meters_per_pixel
                    
                    # Track fire visibility history for this drone
                    if not hasattr(self, 'fire_visibility_history'):
                        self.fire_visibility_history = {}
                    if not hasattr(self, 'last_fire_positions'):
                        self.last_fire_positions = {}
                    
                    prev_fire_visible = self.fire_visibility_history.get(drone_name, False)
                    self.fire_visibility_history[drone_name] = current_fire_visible
                    
                    # FIRE TRACKING REWARDS/PENALTIES based on LOCAL observation only
                    if prev_fire_visible and current_fire_visible and fire_center_x is not None:
                        # Both current and previous: reward staying close to fire
                        if drone_name in self.last_fire_positions:
                            prev_fire_x, prev_fire_y = self.last_fire_positions[drone_name]
                            
                            # Calculate if drone moved towards or away from fire (within local observation)
                            prev_dist_to_fire = np.sqrt((prev_x - prev_fire_x)**2 + (prev_y - prev_fire_y)**2)
                            curr_dist_to_fire = np.sqrt((x - fire_center_x)**2 + (y - fire_center_y)**2)
                            
                            if curr_dist_to_fire < prev_dist_to_fire:
                                # Moving closer to fire within local view
                                tracking_bonus = (prev_dist_to_fire - curr_dist_to_fire) * 5.0
                                reward += tracking_bonus
                            else:
                                # Moving away from fire within local view
                                tracking_penalty = (curr_dist_to_fire - prev_dist_to_fire) * 2.0
                                reward -= tracking_penalty
                        
                        # Update last fire position
                        self.last_fire_positions[drone_name] = (fire_center_x, fire_center_y)
                        
                        # HOVERING BONUS when fire is visible and drone moves slowly
                        if movement_distance < 0.3:  # Very small movement when fire is visible
                            local_fire_dist = np.sqrt((fire_center_x - x)**2 + (fire_center_y - y)**2)
                            if local_fire_dist < 10.0:  # Close to fire within local view
                                hovering_bonus = (10.0 - local_fire_dist) * 2.0
                                reward += hovering_bonus
                    
                    elif prev_fire_visible and not current_fire_visible:
                        # LOST FIRE! Big penalty for losing sight of fire
                        fire_loss_penalty = 15.0
                        reward -= fire_loss_penalty
                        print(f"🔥❌ {drone_name} LOST FIRE from sight! Penalty: {fire_loss_penalty}")
                        
                        # Remove from tracking
                        if drone_name in self.last_fire_positions:
                            del self.last_fire_positions[drone_name]
                    
                    elif current_fire_visible and fire_center_x is not None:
                        # Just found fire - initialize tracking
                        self.last_fire_positions[drone_name] = (fire_center_x, fire_center_y)
                        
                        # HOVERING BONUS when just found fire
                        if movement_distance < 0.3:
                            hovering_bonus = 3.0
                            reward += hovering_bonus
                    
                    # GENERAL MOVEMENT PENALTIES/BONUSES
                    if movement_distance > 0.5:  # Meaningful movement
                        if current_fire_visible:
                            # Reduce movement reward when fire is visible (encourage tracking)
                            movement_multiplier = 0.3
                        else:
                            # INCREASED movement reward when exploring
                            movement_multiplier = 2.0  # Zvýšeno z 1.0 pro větší exploraci
                        reward += min(movement_distance * 0.5 * movement_multiplier, 2.0)  # Zvýšen bonus
                    elif movement_distance < 0.1:
                        if not current_fire_visible:
                            # Only penalize inactivity when no fire is visible
                            reward -= 0.5  # Zvýšena penalty za neačnost
                
                # SLOW FLYING BONUSES
                # if total_speed < 2.0:  # Very slow flight
                #     slow_bonus = (2.0 - total_speed) * 2.0  # Up to 4.0 bonus for nearly stationary
                #     reward += slow_bonus
                #     #print(f"Slow flight bonus: {slow_bonus:.1f}")
                # elif total_speed < 5.0:  # Moderate slow flight
                #     slow_bonus = (5.0 - total_speed) * 0.5  # Up to 1.5 bonus for moderate speed
                #     reward += slow_bonus
                
                # # FAST FLYING PENALTIES
                # if total_speed > 8.0:  # Too fast!
                #     speed_penalty = (total_speed - 8.0) * 3.0  # Progressive penalty
                #     reward -= speed_penalty
                #     #print(f"Speed penalty: {speed_penalty:.1f}")
                # elif total_speed > 5.0:  # Moderate speed warning
                #     speed_warning = (total_speed - 5.0) * 1.0
                #     reward -= speed_warning
                
                # # ADDITIONAL: Extra penalty for erratic vertical movement
                # if vertical_speed > 3.0:  # Too much up/down movement
                #     vertical_penalty = (vertical_speed - 3.0) * 2.0
                #     reward -= vertical_penalty
                    #print(f"Vertical speed penalty: {vertical_penalty:.1f}")
                
                # === 4. FIRE VISIBILITY & TRACKING REWARDS ===
                if "quads" in new_obs and "local_map" in new_obs["quads"] and len(new_obs["quads"]["local_map"]) > drone_idx:
                    local_fire = new_obs["quads"]["local_map"][drone_idx, 0]  # Shape (32, 32)
                    fire_intensity = np.sum(local_fire)
                    
                    if fire_intensity > 0.1:  # Vidí oheň!
                        total_fire_visible = True
                        
                        # === SIMPLE FIRE VISIBILITY REWARD ===
                        # Tracking flag but no discovery bonus
                        if not self.total_fire_discovered:
                            self.total_fire_discovered = True
                        
                        # Simple visibility bonus (only when actually seeing fire)
                        visibility_bonus = 10.0  # Flat bonus for seeing fire
                        reward += visibility_bonus
                
                # === 7. ENHANCED BOUNDARY ENFORCEMENT ===
                # IMMEDIATE EPISODE TERMINATION for boundary violations!
                
                # Check if outside boundaries
                outside_boundaries = abs(x) > self.map_bounds or abs(y) > self.map_bounds
                
                if outside_boundaries:
                    # CANCEL ALL POSITIVE REWARDS when outside!
                    reward = -200.0  # MASSIVE penalty for leaving boundaries
                    
                    # TERMINATE EPISODE IMMEDIATELY!
                    print(f"🚫 {drone_name} OUT OF BOUNDS at [{x:.1f}, {y:.1f}] - EPISODE TERMINATED!")
                    terminated = True
                    return new_obs, reward, terminated, {}  # 4 hodnoty pro starý gym format
                    
                else:
                    # STRONG warning when approaching boundary
                    boundary_buffer = 15.0  # Increased buffer zone
                    if abs(x) > (self.map_bounds - boundary_buffer) or abs(y) > (self.map_bounds - boundary_buffer):
                        boundary_distance = min(self.map_bounds - abs(x), self.map_bounds - abs(y))
                        if boundary_distance < boundary_buffer:
                            penalty_factor = (boundary_buffer - boundary_distance) / boundary_buffer
                            reward -= 20.0 * penalty_factor  # Much stronger approach penalty
                
                # Crash detection
                if z < 0.5:  # Dron se rozbil
                    reward -= 20.0  # Menší crash penalty
                    
                # Flying too high (ineffective exploration)
                if z > 30.0:
                    reward -= 0.1  # Velmi jemná penalizace
                    
        # Episode se ukončí při crash, takže nemusíme řešit missing drony
        
        # === 5. TEAM COORDINATION & SMART EXPLORATION ===
        # Reduced exploration bonus when fire is already found
        exploration_percentage = len(self.visited_cells) / 625.0  # 25x25 cells = 50x50m map
        if exploration_percentage > 0.1 and not self.total_fire_discovered:  # Only reward exploration before fire discovery
            reward += exploration_percentage * 1.0  # Smaller progressive bonus
        
        # === 6. COLLABORATIVE FIRE MONITORING ===
        # Bonus if multiple drones see fire simultaneously
        active_drones = sum(1 for name in self.quad_agents if name in self.sim.drones and self.sim.drones[name].get_position()[2] > 0.5)
        if total_fire_visible and active_drones >= 2:
            reward += 2.0  # Team coordination bonus
        
        self.last_fire_visible = total_fire_visible

        # === CRASH DETECTION - ukončit episode okamžitě ===
        crashed_drones = []
        for drone_name in self.quad_agents:
            if drone_name not in self.sim.drones:
                crashed_drones.append(drone_name)
            elif self.sim.drones[drone_name].get_position()[2] < 0.5:
                crashed_drones.append(drone_name)
        
        # Pokud nějaký dron crashnul, UKONČIT EPISODE
        episode_crashed = len(crashed_drones) > 0
        if episode_crashed:
            print(f"🛑 Episode terminated due to drone crash: {crashed_drones}")
        
        # Episode termination: crash always ends, time limit only in training
        if self.demo_mode:
            # Demo mode: only terminate on crash, no time limit
            terminated = episode_crashed
        else:
            # Training mode: terminate on crash OR timeout
            terminated = self.sim.simulation_time > 300 or episode_crashed
        
        # Ensure reward is always a valid number, never None
        reward = float(reward) if reward is not None else -10.0
        
        return new_obs, reward, terminated, {}