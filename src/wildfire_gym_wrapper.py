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
from wildfire_obs_processor import WildfireObsProcessor

class WildfireMARLEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, agents_config=["quad_1", "fw_1"], demo_mode=False):
        super().__init__()
        self.sim = Simulation()
        self.demo_mode = demo_mode  # Flag for unlimited runtime in demo
        
        # Observation window SYNCHRONIZED with processor defaults
        self.obs_proc = WildfireObsProcessor(window_size_m=30.0, resolution_px=32)  # CONSISTENT 30m window
        
        # BOUNDARY LIMITS PRO PENALIZACI
        self.map_bounds = 50.0  # Dron nemůže jít dál než ±50m od centra
        
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
                "local_map": spaces.Box(low=0.0, high=1.0, shape=(n, 1, 32, 32), dtype=np.float32),
                "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, self_state_size), dtype=np.float32),
                "hidden_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, 128), dtype=np.float32),
            })
            obs_spec_dict["quads"] = CompositeSpec({
                "local_map": UnboundedContinuousTensorSpec(shape=(n, 1, 32, 32)),
                "self_state": UnboundedContinuousTensorSpec(shape=(n, self_state_size)),
                "hidden_state": UnboundedContinuousTensorSpec(shape=(n, 128)),
            })
            action_dict["quads"] = spaces.Dict({
                "action": spaces.Box(low=-1.0, high=1.0, shape=(n, 4), dtype=np.float32)
            })
            action_spec_dict["quads"] = CompositeSpec({
                "action": BoundedContinuousTensorSpec(low=-1, high=1, shape=(n, 4))
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
        
        # CORRECTED exploration grid calculation: 2m cells on 100x100m map = 50x50 = 2500 cells
        grid_size = int((2 * self.map_bounds) / 2.0)  # 100m / 2m = 50 cells per side
        exploration_grid_cells = grid_size * grid_size  # 50 * 50 = 2500 cells total
        self.exploration_grid_cells = exploration_grid_cells
        
        # ===== EXPLORATION TRACKING PRO NOVÝ REWARD SYSTÉM =====
        self.visited_cells = set()  # Track visited 2x2m cells
        self.previous_positions = {}  # Track drone movement
        self.episode_start_time = 0.0
        self.total_fire_discovered = False
        self.exploration_reward_accumulated = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.start_simulation()
        
        # ===== ENABLE FIRE SIMULATION =====
        self.sim.enable_fire_simulation(
            grid_width_m=30,
            grid_height_m=30,
            cell_size_m=5.0,   # Větší buňky jako v demo pro lepší šíření
            dt=0.5  # Větší time step - fire sim nemusí být tak častá jako PyBullet
        )
        
        # Setup scenario: 1 dron ve středu mapy pro debugging
        if len(self.quad_agents) >= 1:
            # Jeden dron ve středu mapy pro lepší kontrolu
            self.sim.add_quadcopter(self.quad_agents[0], position=[0, 0, 10])
        
        # Fixed fire position - 1 oheň uprostřed mapy  
        self.sim.environment.ignite_fire(x=0, y=0, intensity=3.0)  # Střed mapy
        
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
                va = (act[0] + 1) * 7.5 + 10  # [10, 25] m/s
                gamma = act[1] * 0.5
                phi = act[2] * 0.8
                drop = 1.0 if act[3] > 0.5 else 0.0
                drone_controls[name] = [va, gamma, phi, drop]

        # 2. Physics Step
        self.sim.step_simulation(drone_controls)

        # 3. Collect new state
        new_obs = self._get_obs()

        # 4. ===== NOVÝ EXPLORATION-BASED REWARD SYSTÉM =====
        reward = 0.0  
        
        # Tracking for fire discovery
        total_fire_visible = False
        
        for drone_idx, drone_name in enumerate(self.quad_agents):
            if drone_name and drone_name in self.sim.drones:
                drone_pos = self.sim.drones[drone_name].get_position()
                x, y, z = drone_pos
                
                # === 1. BALANCED EXPLORATION ===
                # Convert position to grid cell (2m resolution)
                cell_x = int(x // 2.0)
                cell_y = int(y // 2.0)
                cell_key = (cell_x, cell_y)
                
                # BALANCED exploration rewards
                if cell_key not in self.visited_cells and abs(x) <= self.map_bounds and abs(y) <= self.map_bounds:
                    self.visited_cells.add(cell_key)
                    
                    if not self.total_fire_discovered:
                        # Pre-discovery: Moderate exploration reward
                        reward += 0.5  # Reduced from 1.0 to prevent reward inflation
                        self.exploration_reward_accumulated += 0.5
                    else:
                        # Post-discovery: Small exploration reward
                        reward += 0.1  # Minimal exploration reward
                
                # === 2. ANTI-MOVEMENT SYSTEM - Penalize large movements near fire ===
                if drone_name in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[drone_name]
                    movement_distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                    
                    # HOVERING BONUS when near fire and moving slowly
                    fire_distance = np.sqrt(x**2 + y**2)
                    if fire_distance < 15.0 and movement_distance < 0.2:  # Very small movement near fire
                        hovering_bonus = (15.0 - fire_distance) * 3.0  # Big bonus for staying put
                        reward += hovering_bonus
                        #print(f"Hovering bonus: {hovering_bonus:.1f}")
                    
                    # STANDARD movement reward (reduced when near fire)
                    if movement_distance > 0.5:  # Meaningful movement
                        movement_multiplier = 1.0
                        if fire_distance < 15.0:  # Reduce movement reward when near fire
                            movement_multiplier = 0.2  # Much less movement reward near fire
                        reward += min(movement_distance * 0.3 * movement_multiplier, 0.8)
                    elif movement_distance < 0.1 and fire_distance > 20.0:  # Only penalize inactivity when far from fire
                        reward -= 0.2  # Inactivity penalty only when not near fire
                
                # Update position tracking
                self.previous_positions[drone_name] = (x, y)
                
                # === NEW: RETURN HOME BONUS ===
                # Reward return to center after boundary violation  
                fire_distance = np.sqrt(x**2 + y**2)
                if drone_name in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[drone_name] 
                    prev_fire_distance = np.sqrt(prev_x**2 + prev_y**2)
                    
                    # Reward when returning towards fire/center from far away
                    if prev_fire_distance > 30.0 and fire_distance < prev_fire_distance:
                        return_progress = prev_fire_distance - fire_distance
                        reward += return_progress * 2.0  # Bonus for returning home
                        #print(f"Return home bonus: {return_progress * 2.0:.1f}")
                
                # === 3. FIRE VISIBILITY & TRACKING REWARDS ===
                fire_visible = False
                if "quads" in new_obs and "local_map" in new_obs["quads"] and len(new_obs["quads"]["local_map"]) > drone_idx:
                    local_fire = new_obs["quads"]["local_map"][drone_idx, 0]  # Shape (32, 32)
                    fire_intensity = np.sum(local_fire)
                    
                    if fire_intensity > 0.1:  # Vidí oheň!
                        fire_visible = True
                        total_fire_visible = True
                        
                        # === FIXED DISTANCE-BASED FIRE TRACKING REWARD ===
                        fire_distance = np.sqrt(x**2 + y**2)  # Distance to fire center (0,0)
                        
                        # EXTENDED LINEAR DECAY: No more dead zones!
                        max_tracking_reward = 15.0  # Maximum tracking reward when very close
                        tracking_distance_threshold = 42.0  # EXTENDED to boundary warning threshold (50-8)
                        
                        if fire_distance <= tracking_distance_threshold:
                            distance_factor = (tracking_distance_threshold - fire_distance) / tracking_distance_threshold
                            tracking_reward = max_tracking_reward * distance_factor
                            reward += tracking_reward
                            #print(f"Fire tracking reward: {tracking_reward:.1f} (distance: {fire_distance:.1f}m)")
                        
                        # One-time discovery bonus remains the same
                        if not self.total_fire_discovered:
                            reward += 50.0  # Discovery bonus
                            self.total_fire_discovered = True
                            print(f"🔥 FIRE DISCOVERED by {drone_name}!")
                        # === RADICAL STOP BONUS - Reward staying near fire with small actions ===
                        if fire_distance < 20.0:  # Extended stop zone
                            # Check action magnitudes if available
                            if hasattr(self, 'last_actions') and drone_name in self.last_actions:
                                action_magnitude = np.linalg.norm(self.last_actions[drone_name])
                                
                                # MASSIVE stop bonus for small actions near fire
                                if action_magnitude < 0.30:  # Small actions = hovering
                                    stop_bonus = (20.0 - fire_distance) * 5.0  # INCREASED from 1.0 to 5.0!
                                    reward += stop_bonus
                                    #print(f"MEGA Stop bonus: {stop_bonus:.1f}")
                                
                                # ANTI-MOVEMENT PENALTY for large actions near fire  
                                elif action_magnitude > 0.35 and fire_distance < 15.0:
                                    movement_penalty = action_magnitude * 20.0  # Strong penalty for big moves near fire
                                    reward -= movement_penalty
                                    #print(f"Anti-movement penalty: {movement_penalty:.1f}")
                        
                        # Moderate intensity bonus (unchanged) (unchanged)
                        intensity_bonus = min(fire_intensity * 2.0, 10.0)  # Reasonable intensity bonus
                        reward += intensity_bonus
                    
                    else:
                        # === DEAD ZONE ELIMINATION: Guidance back to fire when not visible ===
                        if self.total_fire_discovered:
                            fire_distance = np.sqrt(x**2 + y**2)
                            # Provide gentle guidance when fire not visible but within reasonable range
                            if 15.0 < fire_distance < 42.0:  # In the former "dead zone"
                                guidance_factor = (42.0 - fire_distance) / 27.0  # Linear from 0 to 1
                                guidance_reward = 2.0 * guidance_factor  # Gentle guidance back
                                reward += guidance_reward
                                #print(f"Fire guidance: {guidance_reward:.1f} at distance {fire_distance:.1f}m")
                            elif fire_distance >= 42.0:  # Far from fire - stronger guidance
                                reward -= 1.0  # Gentle penalty for being too far
                
                # === 4. SIMPLIFIED FIRE-SEEKING BEHAVIOR ===
                if self.total_fire_discovered and not fire_visible:
                    # Simple distance-based guidance (no movement comparison)
                    fire_distance = np.sqrt(x**2 + y**2)
                    
                    # Gentle guidance towards fire area
                    if fire_distance < 15.0:
                        reward += 2.0  # Small reward for being in fire area even if not visible
                    elif fire_distance > 30.0:
                        reward -= 0.5  # Very gentle guidance back to fire area
                
                # === 4. ENHANCED BOUNDARY AWARENESS & SAFETY ===
                # Use softer boundary enforcement with distance-based penalties
                # Since drone now knows boundary distances, it should learn to avoid them
                boundary_buffer = 8.0  # Increased safety buffer from 5m to 8m
                if abs(x) > (self.map_bounds - boundary_buffer) or abs(y) > (self.map_bounds - boundary_buffer):
                    # Soft penalty for approaching boundary (agent should learn to avoid)
                    boundary_distance = min(self.map_bounds - abs(x), self.map_bounds - abs(y))
                    if boundary_distance < boundary_buffer:
                        penalty_factor = (boundary_buffer - boundary_distance) / boundary_buffer
                        reward -= 5.0 * penalty_factor  # INCREASED from 2.0 to 5.0!
                
                # MUCH STRONGER penalty for actual boundary violation
                if abs(x) > self.map_bounds or abs(y) > self.map_bounds:
                    # Progressive penalty based on how far outside
                    excess_distance = max(abs(x) - self.map_bounds, abs(y) - self.map_bounds, 0)
                    boundary_penalty = 25.0 + excess_distance * 2.0  # Base 25 + progressive
                    reward -= boundary_penalty
                    #print(f"BOUNDARY VIOLATION! Penalty: {boundary_penalty:.1f}")
                
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