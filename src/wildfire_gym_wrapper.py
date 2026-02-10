import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        self.obs_proc = WildfireObsProcessor(window_size_m=40.0, resolution_px=32)
        
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
                "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, 7), dtype=np.float32),
                "hidden_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n, 128), dtype=np.float32),
            })
            obs_spec_dict["quads"] = CompositeSpec({
                "local_map": UnboundedContinuousTensorSpec(shape=(n, 1, 32, 32)),
                "self_state": UnboundedContinuousTensorSpec(shape=(n, 7)),
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.start_simulation()
        
        # Setup scenario: Add drones at starting positions
        for name in self.quad_agents:
            self.sim.add_quadcopter(name, position=[0, 0, 5])
        for name in self.fixed_agents:
            self.sim.add_fixedwing(name, position=[0, 10, 20])

        # Ignite a small starting fire
        self.sim.environment.ignite_fire(x=0, y=10, intensity=1.0)

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
                    q_maps.append(data["local_map"])
                    q_states.append(data["self_state"])
                else:
                    q_maps.append(np.zeros((1, 32, 32), dtype=np.float32))
                    q_states.append(np.zeros(7, dtype=np.float32))
            
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
                agent_states.append(state)
            else:
                agent_states.append(np.zeros(7, dtype=np.float32))
        
        while len(agent_states) < 8:
            agent_states.append(np.zeros(7))
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
        
        # 4. Shared Reward Calculation
        # fire_state = self.sim.environment.get_fire_state()
        # reward = -0.1 * np.sum(fire_state['fire_grid_state']['B']) if fire_state else 0.0
        # Nová odměna pro trénink vznášení (v wildfire_gym_wrapper.py)
        local_fire = new_obs["quads"]["local_map"][0] # Mapa 32x32 pro prvního drona
        fire_intensity = np.sum(local_fire)
        # Odměna za to, že vidí oheň + penalizace za vzdálenost od středu ohně v jeho výhledu
        reward = fire_intensity * 0.5

        
        terminated = self.sim.simulation_time > 120 # 2 minute timeout
        
        return new_obs, np.array([reward]), terminated, False, {}