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
        # MENŠÍ OKNO PRO RYČLEJŠÍ EXPLORATION - 20m místo 40m
        self.obs_proc = WildfireObsProcessor(window_size_m=20.0, resolution_px=32)
        
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
        self.frame_pbar = None
        
        # ===== NOVÝ EXPLORATION TRACKING =====
        self.explored_cells = set()  # Už navštívené pozice (x,y)
        self.discovered_fire_cells = set()  # Už nalezené ohňové buňky
        self.last_fire_visible = False  # Vidí oheň v posledním kroku
        self.episode_start_time = 0.0  # Čas začátku epizody
        self.grid_resolution = 2.0  # Velikost buňky pro exploration tracking

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.start_simulation()
        
        # ===== KRITICKÉ: ENABLE FIRE SIMULATION FIRST! =====
        self.sim.enable_fire_simulation(
            grid_width_m=200,   # Dostatečně velký grid
            grid_height_m=200,
            cell_size_m=2.0,    # Stejné jako grid_resolution
            dt=0.5
        )
        
        # Setup scenario: Blížší start k ohni pro rychlejší nalezení
        for name in self.quad_agents:
            self.sim.add_quadcopter(name, position=[-3, 3, 8])  # BLÍŽE a níže
        for name in self.fixed_agents:
            self.sim.add_fixedwing(name, position=[0, 10, 20])

        # Větší a silnější oheň pro lepší viditelnost
        self.sim.environment.ignite_fire(x=0, y=8, intensity=3.0)  # Silnější, blíže
        # Přidej ještě jeden malý oheň pro větší šanci na nalezení
        self.sim.environment.ignite_fire(x=2, y=5, intensity=2.0)
        
        # Tichý fire setup - bez výpisů
        
        # ===== RESET EXPLORATION TRACKING =====
        self.explored_cells.clear()
        self.discovered_fire_cells.clear() 
        self.last_fire_visible = False
        self.episode_start_time = self.sim.simulation_time

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
                    
                    # Rozšíříme self_state o exploration informace
                    basic_state = data["self_state"]  # 12 features včetně map awareness
                    
                    # Aktualizuj exploration progress v self_state
                    exploration_ratio = len(self.explored_cells) / max(100, len(self.explored_cells))
                    fire_discovery_ratio = len(self.discovered_fire_cells) / max(1, len(self.discovered_fire_cells))
                    
                    # Aktualizuj posledních 2 features v basic_state
                    basic_state[-2] = exploration_ratio     # Exploration progress
                    basic_state[-1] = fire_discovery_ratio  # Fire discovery progress
                    
                    q_maps.append(data["local_map"])
                    q_states.append(basic_state)
                else:
                    q_maps.append(np.zeros((1, 32, 32), dtype=np.float32))
                    q_states.append(np.zeros(12, dtype=np.float32))  # Aktualizováno na 12
            
            obs["quads"]["local_map"] = np.stack(q_maps)
            obs["quads"]["self_state"] = np.stack(q_states)
            obs["quads"]["hidden_state"] = np.zeros((len(self.quad_agents), 128), dtype=np.float32)

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
                state = self.obs_proc.fetch(self.sim, name)["self_state"]
                # Aktualizuj exploration progress pro globální state
                if len(state) >= 12:  # Pokud má nové features
                    exploration_ratio = len(self.explored_cells) / max(100, len(self.explored_cells))
                    fire_discovery_ratio = len(self.discovered_fire_cells) / max(1, len(self.discovered_fire_cells))
                    state[-2] = exploration_ratio
                    state[-1] = fire_discovery_ratio
                agent_states.append(state)
            else:
                agent_states.append(np.zeros(12, dtype=np.float32))  # Aktualizováno na 12
        
        while len(agent_states) < 8:
            agent_states.append(np.zeros(12))  # Aktualizováno na 12 features
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

        if self.frame_pbar is not None:
            self.frame_pbar.update(1)
            self.sim_pbar.set_postfix({"Time": f"{self.sim.simulation_time:.1f}s"})

        # 3. Collect new state
        new_obs = self._get_obs()


        
        if drone_name and drone_name in self.sim.drones:
            drone_pos = self.sim.drones[drone_name].get_position()
            x, y, z = drone_pos
            
            # 1. BOUNDARY CHECK - VELKÁ PENALIZACE ZA VYLĚTANÍ Z MAPY
            if abs(x) > self.map_bounds or abs(y) > self.map_bounds:
                reward -= 10.0  # Velká penalizace za vylĚtení
                # Tichý boundary check
            
            # 2. VÝŠKOVÁ ODMĚNA - zachovej bezpečné létání
            if 3.0 < z < 20.0:  # Bezpečná výška
                reward += 0.2  # Základní odměna za létání
            elif z < 1.0:  # Příliš nízko
                reward -= 2.0
            elif z > 25.0:  # Příliš vysoko
                reward -= 1.0
            
            # 3. ODMĚNA ZA VIDĚNÍ OHNě V LOCAL MAPĚ
            if "quads" in new_obs and "local_map" in new_obs["quads"]:
                local_fire = new_obs["quads"]["local_map"][0] 
                fire_intensity = np.sum(local_fire)
                
                if fire_intensity > 0.1:  # Vidí oheň!
                    reward += fire_intensity * 2.0  # Větší odměna za videní ohně
                    # Tichý fire detection
                    
                    # 4. PROXIMITY BONUS - čím blíže, tím lépe
                    fire_positions = [[0, 8], [2, 5]]  # Známé pozice ohně
                    min_dist = float('inf')
                    for fire_pos in fire_positions:
                        dist = np.linalg.norm([x - fire_pos[0], y - fire_pos[1]])
                        min_dist = min(min_dist, dist)
                    
                    # Progresivní odměna za vzdálenost
                    if min_dist < 2.0:
                        reward += 5.0  # VELKÁ odměna za blízkost
                    elif min_dist < 5.0:
                        reward += 2.0  # Střední odměna
                    elif min_dist < 10.0:
                        reward += 0.5  # Malá odměna
                    
                    # Tichý proximity tracking
                else:
                    # Malá penalizace když nevidí oheň (motivace k hledání)
                    reward -= 0.1
            
            # 5. STABILITY BONUS - odměna za mírnou rychlost (anti-chaos)
            velocity = self.sim.drones[drone_name].get_velocity()
            speed = np.linalg.norm(velocity)
            if speed < 3.0:  # Mírná rychlost
                reward += 0.1
            elif speed > 10.0:  # Příliš rychlý
                reward -= 0.5
                
        else:
            # Dron neexistuje/crashnul - velká penalizace
            reward = -8.0
        
        # Malá časová penalizace (minimální)
        reward -= 0.02
        
        # 4. ===== NOVÝ EXPLORATION-BASED REWARD SYSTÉM =====
        reward = 0.0  # Žádná základní odměna - pouze za akce!
        
        # Získej pozici prvního drona
        drone_name = self.quad_agents[0] if self.quad_agents else None
        
        if drone_name and drone_name in self.sim.drones:
            drone_pos = self.sim.drones[drone_name].get_position()
            x, y, z = drone_pos
            
            # === 1. EXPLORATION REWARD (+2 za nové políčko) ===
            current_cell = (round(x / self.grid_resolution), round(y / self.grid_resolution))
            if current_cell not in self.explored_cells:
                self.explored_cells.add(current_cell)
                reward += 2.0  # +2 za nové políčko
            
            # === 2. FIRE DISCOVERY REWARDS ===
            fire_visible = False
            if "quads" in new_obs and "local_map" in new_obs["quads"]:
                local_fire = new_obs["quads"]["local_map"][0] 
                fire_intensity = np.sum(local_fire)
                
                if fire_intensity > 0.1:  # Vidí oheň!
                    fire_visible = True
                    
                    # Najdi konkrétní fire cells v local mapě
                    fire_cells_in_view = []
                    local_size = local_fire.shape[-1]  # Předpokládáme čtvercovou mapu
                    center = local_size // 2
                    
                    for i in range(local_size):
                        for j in range(local_size):
                            if local_fire[0, i, j] > 0.1:  # Burning cell
                                # Přepočítej lokální coords na world coords
                                world_fire_x = x + (i - center) * self.grid_resolution
                                world_fire_y = y + (j - center) * self.grid_resolution
                                fire_cell = (round(world_fire_x / self.grid_resolution), round(world_fire_y / self.grid_resolution))
                                fire_cells_in_view.append(fire_cell)
                    
                    # +10 za nové fire cells, +4 za známé
                    for fire_cell in fire_cells_in_view:
                        if fire_cell not in self.discovered_fire_cells:
                            self.discovered_fire_cells.add(fire_cell)
                            reward += 10.0  # +10 za novou hořící buňku
                        else:
                            reward += 4.0   # +4 za monitoring známé buňky
            
            # === 3. TIME PENALTY (-1 za vteřinu) ===
            time_elapsed = self.sim.simulation_time - self.episode_start_time
            reward -= time_elapsed * 0.1  # Mírná penalizace za čas
            
            # === 4. NO FIRE PENALTY (-3 když nevidí oheň) ===
            if not fire_visible:
                reward -= 3.0  # Silná motivace hledat oheň
            
            # === 5. BOUNDARY VIOLATION (-100) ===
            if abs(x) > self.map_bounds or abs(y) > self.map_bounds:
                reward -= 100.0  # Tvrdá penalizace za vylétnutí
            
            # === 6. CRASH DETECTION (-100) ===
            if z < 0.5:  # Dron se rozbil
                reward -= 100.0
            
            # === 7. EXTREME HEIGHT PENALTY ===
            if z > 30.0:
                reward -= 5.0  # Penalizace za extrémní výšky
            
            self.last_fire_visible = fire_visible
                
        else:
            # Dron neexistuje/crashnul - velká penalizace
            reward = -100.0

        
        terminated = self.sim.simulation_time > 120 # 2 minute timeout
        
        # Ensure reward is always a valid number, never None
        reward = float(reward) if reward is not None else -10.0
        
        return new_obs, reward, terminated, False, {}