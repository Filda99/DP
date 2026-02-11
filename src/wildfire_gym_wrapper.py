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

    def __init__(self, agents_config=["quad_1", "fw_1"]):
        super().__init__()
        self.sim = Simulation()
        
        # Observation window 10m pro local_map
        self.obs_proc = WildfireObsProcessor(window_size_m=10.0, resolution_px=32)
        
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
            obs_dict["quads"] = spaces.Dict({
                "local_map": spaces.Box(low=0.0, high=1.0, shape=(n, 1, 32, 32), dtype=np.float32),
                "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, 12), dtype=np.float32),  # Rozšířeno na 12
                "hidden_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, 128), dtype=np.float32),
            })
            obs_spec_dict["quads"] = CompositeSpec({
                "local_map": UnboundedContinuousTensorSpec(shape=(n, 1, 32, 32)),
                "self_state": UnboundedContinuousTensorSpec(shape=(n, 12)),  # Rozšířeno na 12
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
                    
                    # Jen základní self_state
                    basic_state = data["self_state"][:6]  # Jen prvních 6 features
                    
                    q_maps.append(data["local_map"])
                    q_states.append(basic_state)
                else:
                    q_maps.append(np.zeros((1, 32, 32), dtype=np.float32))
                    q_states.append(np.zeros(6, dtype=np.float32))  # Jen 6 features
            
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
        for name in self.all_agents:
            if name in self.sim.drones:
                state = self.obs_proc.fetch(self.sim, name)["self_state"][:6]  # Jen 6 features
                agent_states.append(state)
            else:
                agent_states.append(np.zeros(6, dtype=np.float32))  # Jen 6 features
        
        while len(agent_states) < 8:
            agent_states.append(np.zeros(6))  # Jen 6 features
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
        
        # Map Quads (Unpack from nested "action" key)
        if "quads" in actions and "action" in actions["quads"]:
            # GymWrapper passes this as a numpy array now
            quad_acts = actions["quads"]["action"]
            for i, name in enumerate(self.quad_agents):
                # Ensure we handle shape (1, 4) vs (4,)
                act = quad_acts[i]
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
                
                # === 1. FIRE-CENTRIC EXPLORATION ===
                # Convert position to grid cell (2m resolution)
                cell_x = int(x // 2.0)
                cell_y = int(y // 2.0)
                cell_key = (cell_x, cell_y)
                
                # ADAPTIVE exploration rewards based on fire discovery status
                if cell_key not in self.visited_cells and abs(x) <= self.map_bounds and abs(y) <= self.map_bounds:
                    self.visited_cells.add(cell_key)
                    
                    if not self.total_fire_discovered:
                        # Pre-discovery: Encourage exploration
                        reward += 1.0
                        self.exploration_reward_accumulated += 1.0
                    else:
                        # Post-discovery: Discourage random exploration
                        reward += 0.2  # Much smaller exploration reward
                
                # === 2. MOVEMENT REWARD - Enhanced for active search ===
                if drone_name in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[drone_name]
                    movement_distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                    
                    if movement_distance > 0.5:  # Meaningful movement
                        reward += min(movement_distance * 0.3, 0.8)  # Slightly higher movement reward
                    elif movement_distance < 0.1:  # Standing still
                        reward -= 0.2  # Higher penalty for inactivity
                
                # Update position tracking
                self.previous_positions[drone_name] = (x, y)
                
                # === 3. FIRE VISIBILITY & TRACKING REWARDS ===
                fire_visible = False
                if "quads" in new_obs and "local_map" in new_obs["quads"] and len(new_obs["quads"]["local_map"]) > drone_idx:
                    local_fire = new_obs["quads"]["local_map"][drone_idx, 0]  # Shape (32, 32)
                    fire_intensity = np.sum(local_fire)
                    
                    if fire_intensity > 0.1:  # Vidí oheň!
                        fire_visible = True
                        total_fire_visible = True
                        
                        # MASSIVE continuous fire tracking reward
                        reward += 50.0  # Much higher than exploration
                        
                        # Extra bonus při prvním objevení ohně v epizodě
                        if not self.total_fire_discovered:
                            reward += 200.0  # HUGE discovery bonus
                            self.total_fire_discovered = True
                            print(f"🔥 FIRE DISCOVERED by {drone_name}!")
                        
                        # Bonus za intenzitu ohně (čím blíž, tím více vidí)
                        intensity_bonus = min(fire_intensity * 5.0, 50.0)  # Higher intensity bonus
                        reward += intensity_bonus
                        
                        # DISTANCE-BASED FIRE TRACKING
                        fire_distance = np.sqrt(x**2 + y**2)  # Distance to fire center (0,0)
                        if fire_distance < 15.0:  # Close to fire
                            proximity_reward = (15.0 - fire_distance) * 3.0  # Closer = higher reward
                            reward += proximity_reward
                    
                    else:
                        # If fire was previously visible but now lost
                        if self.total_fire_discovered:
                            # PENALTY for losing sight of discovered fire
                            fire_distance = np.sqrt(x**2 + y**2)
                            if fire_distance > 20.0:  # Too far from fire center
                                reward -= 5.0  # Penalty for being far from fire when it's known
                
                # === 4. FIRE-SEEKING BEHAVIOR (when fire is known but not visible) ===
                if self.total_fire_discovered and not fire_visible:
                    # Reward for moving towards fire center when fire is known
                    fire_distance = np.sqrt(x**2 + y**2)
                    
                    # Direction-based reward
                    if drone_name in self.previous_positions:
                        prev_x, prev_y = self.previous_positions[drone_name]
                        prev_distance = np.sqrt(prev_x**2 + prev_y**2)
                        
                        if fire_distance < prev_distance:  # Moving closer to fire
                            approach_reward = (prev_distance - fire_distance) * 10.0
                            reward += min(approach_reward, 20.0)  # Cap at 20
                        else:  # Moving away from fire
                            retreat_penalty = (fire_distance - prev_distance) * 5.0
                            reward -= min(retreat_penalty, 10.0)  # Cap penalty at 10
                
                # === 4. SAFETY PENALTIES (POUZE negativní události) ===
                # Boundary violation - umírněná penalizace
                if abs(x) > self.map_bounds or abs(y) > self.map_bounds:
                    reward -= 10.0  # Menší penalizace
                
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
        
        terminated = self.sim.simulation_time > 120 or episode_crashed  # Crash OR timeout
        
        # Ensure reward is always a valid number, never None
        reward = float(reward) if reward is not None else -10.0
        
        return new_obs, reward, terminated, {}