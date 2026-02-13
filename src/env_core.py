import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import functools
import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv
from src.simulation import Simulation  # Uprav cestu podle tvé složky!
import random

class DroneFireEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "drone_fire_v1"}

    def __init__(self, num_quads=1, grid_size_m=200.0, local_map_size=32):
        super().__init__()
        
        self.max_steps = 500

        self.num_quads = num_quads
        self.grid_size_m = grid_size_m
        self.map_bounds = self.grid_size_m / 2.0 
        
        self.possible_agents = [f"quad_{i}" for i in range(self.num_quads)]
        self.agents = self.possible_agents[:]
        
        # ACTION SPACE: [Roll, Pitch, Yaw, Throttle]
        self._action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32) 
            for agent in self.possible_agents
        }
        
        # OBSERVATION SPACE (Upraveno podle tvé specifikace!)
        # Co znamená 10 čísel v self_state:
        # 0-2: pozice x, y, z (relativně k mapě)
        # 3-5: rychlost vx, vy, vz
        # 6-9: vzdálenost k okrajům (sever, jih, východ, západ) -> super pro boundary awareness!
        self.self_state_size = 10 
        
        # Pokud máme více než 1 dron, přidáme (num_quads - 1) * 3 čísel pro relativní pozice ostatních
        # To nahradí Lidar a pomůže jim se nesrazit a koordinovat roj.
        self.other_drones_state_size = (self.num_quads - 1) * 3
        self.total_vector_size = self.self_state_size + self.other_drones_state_size

        self._observation_spaces = {
            agent: gym.spaces.Dict({
                "local_map": gym.spaces.Box(low=0.0, high=1.0, shape=(1, local_map_size, local_map_size), dtype=np.float32),
                "self_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_vector_size,), dtype=np.float32)
            })
            for agent in self.possible_agents
        }

        # STATE SPACE (Pohled boha pro Kritika)
        # Zmenšený fire_grid 16x16 pixelů + stavy všech dronů
        global_fire_cells = 16 * 16
        global_state_size = global_fire_cells + (self.num_quads * self.total_vector_size)
        
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(global_state_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    # Metody, které napíšeme v dalších krocích:
    def _get_obs(self, agent_name):
        """
        Vygeneruje pozorování pro konkrétního agenta (kvadrokoptéru).
        """
        drone = self.sim.drones[agent_name]
        pos = drone.get_position()  # [x, y, z]
        vel = drone.get_velocity()  # [vx, vy, vz]
        
        # 1. Výpočet vzdáleností k okrajům mapy (Boundary Awareness)
        # self.map_bounds je např. 100.0. Dron je na pozici x=80.
        # Vzdálenost k východu = 100 - 80 = 20m.
        dist_north = self.map_bounds - pos[1]
        dist_south = pos[1] - (-self.map_bounds)
        dist_east  = self.map_bounds - pos[0]
        dist_west  = pos[0] - (-self.map_bounds)
        
        # 2. Sestavení Self-State
        self_state = np.array([
            pos[0], pos[1], pos[2],    # 0-2: Pozice (x,y,z)
            vel[0], vel[1], vel[2],    # 3-5: Rychlost (vx,vy,vz)
            dist_north, dist_south, dist_east, dist_west  # 6-9: Vzdálenosti ke zdem
        ], dtype=np.float32)

        # 3. Získání pozic OSTATNÍCH dronů (pro tvůj Broadcast / Attention)
        other_drones_positions = []
        for other_agent in self.possible_agents:
            if other_agent != agent_name:
                if other_agent in self.sim.drones:
                    # Pokud dron žije, přidáme jeho relativní pozici vůči nám
                    other_pos = self.sim.drones[other_agent].get_position()
                    rel_pos = other_pos - pos
                    other_drones_positions.extend(rel_pos)
                else:
                    # Pokud dron naboural a neexistuje, dáme tam nuly (nebo hodnoty daleko mimo mapu)
                    other_drones_positions.extend([0.0, 0.0, 0.0])
        
        # Spojíme self_state a ostatní drony do jednoho vektoru
        full_vector_state = np.concatenate([self_state, other_drones_positions]).astype(np.float32)

        # 4. Získání lokální mapy (Local Map 32x32)
        # Použijeme zjednodušenou logiku z tvého starého obs_processor
        local_map = self._extract_local_fire_map(pos)

        return {
            "local_map": local_map,
            "self_state": full_vector_state
        }

    def _extract_local_fire_map(self, pos, window_size_m=30.0, resolution_px=32):
        """
        Vyřízne 30x30m čtverec z velkého FireGridu kolem dronu a zmenší ho na 32x32 pixelů.
        """
        if self.sim.environment.fire_grid is None:
            return np.zeros((1, resolution_px, resolution_px), dtype=np.float32)
            
        mapper = self.sim.environment.grid_mapper
        fire_grid = self.sim.environment.fire_grid.I  # Intenzita ohně
        
        # Hranice výřezu v metrech
        half_w = window_size_m / 2.0
        min_world = (pos[0] - half_w, pos[1] - half_w)
        max_world = (pos[0] + half_w, pos[1] + half_w)
        
        # Převod na indexy do gridu
        r_min, c_min = mapper.world_to_cell(min_world)
        r_max, c_max = mapper.world_to_cell(max_world)
        
        # Ořez proti pádům mimo pole
        r_min = max(0, min(r_min, mapper.grid_height_cells - 1))
        r_max = max(0, min(r_max, mapper.grid_height_cells - 1))
        c_min = max(0, min(c_min, mapper.grid_width_cells - 1))
        c_max = max(0, min(c_max, mapper.grid_width_cells - 1))
        
        crop = fire_grid[r_min:r_max+1, c_min:c_max+1]
        
        # Změna velikosti na fixních 32x32 (Pokud je crop prázdný, vrátí nuly)
        if crop.size == 0:
            processed_map = np.zeros((resolution_px, resolution_px), dtype=np.float32)
        else:
            import cv2  # Předpokládám, že máš OpenCV
            processed_map = cv2.resize(crop, (resolution_px, resolution_px), interpolation=cv2.INTER_LINEAR)
            
        return processed_map[np.newaxis, ...].astype(np.float32)

    def state(self):
        """
        MAPPO požadavek: Globální stav mapy pro centrálního Kritika.
        Vrací zploštělou (flattened) matici ohně + stavy všech agentů.
        """
        # 1. Zmenšený globální oheň (např. 16x16)
        if self.sim.environment.fire_grid is not None:
            full_grid = self.sim.environment.fire_grid.I
            import cv2
            small_grid = cv2.resize(full_grid, (16, 16), interpolation=cv2.INTER_AREA)
            fire_summary = small_grid.flatten()
        else:
            fire_summary = np.zeros(16 * 16, dtype=np.float32)

        # 2. Sezbíráme stavy všech dronů (abychom věděli, kde celý tým je)
        all_agent_states = []
        for agent in self.possible_agents:
            if agent in self.agents:
                # Agent žije, vezmeme jeho data (můžeme recyklovat naši _get_obs funkci)
                obs = self._get_obs(agent)
                all_agent_states.append(obs["self_state"])
            else:
                # Agent je mrtvý (naboural), pošleme nuly
                all_agent_states.append(np.zeros(self.total_vector_size, dtype=np.float32))

        # Spojíme oheň a stavy dronů do jednoho obřího 1D pole pro Kritika
        global_state = np.concatenate([fire_summary] + all_agent_states)
        return global_state
        
    def reset(self, seed=None, options=None):
        """
        Vyčistí mapu, vytvoří nový oheň a nahodí drony na start.
        Volá se na začátku každé epizody.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # 1. Obnovíme seznam žijících agentů
        self.agents = self.possible_agents[:]
        
        # 2. Tvrdý restart PyBullet enginu (Zabrání memory leakům)
        if hasattr(self, 'sim') and self.sim is not None:
            self.sim.stop_simulation()
            
        self.sim = Simulation()
        self.sim.start_simulation()
        
        # 3. Zapneme oheň (Zatím na fixní pozici [0,0] pro snazší učení)
        self.sim.enable_fire_simulation(
            grid_width_m=self.grid_size_m,
            grid_height_m=self.grid_size_m,
            cell_size_m=1.0,
            dt=0.1
        )
        self.sim.start_fire([0, 0], intensity=0.5)
        
        # 4. Přidáme drony na náhodné startovní pozice (např. 30m od ohně)
        for agent in self.agents:
            start_x = random.uniform(-30, 30)
            start_y = random.uniform(-30, 30)
            self.sim.add_quadcopter(agent, position=[start_x, start_y, 10.0])
            
        # 5. Inicializace sledování (Trackerů) pro Odměny
        self.visited_cells = set() # Sem si budeme ukládat, kde už drony byly
        self.fire_discovered = False
        self.current_step = 0
        
        # 6. PettingZoo vyžaduje vrátit slovník pozorování a slovník info
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        """
        Provede jeden krok v prostředí na základě akcí od RL agentů.
        Vrací: observations, rewards, terminations, truncations, infos
        """
        self.current_step += 1
        
        # 1. FYZIKÁLNÍ KROK (Frame Skip)
        # RL síť nepotřebuje rozhodovat 30x za vteřinu (to by ji mátlo, protože
        # by neviděla výsledek své akce). Zopakujeme stejnou akci např. 5x za sebou.
        frame_skip = 5
        
        # Převod formátu akcí pro tvou simulaci
        drone_controls = {}
        for agent_name, action in actions.items():
            # Akce je np.array o 4 hodnotách [-1.0 až 1.0]
            drone_controls[agent_name] = action
            
        # Pustíme fyziku
        for _ in range(frame_skip):
            self.sim.step_simulation(drone_controls)

        # 2. PŘÍPRAVA VÝSTUPNÍCH SLOVNÍKŮ (Vyžadováno PettingZoo API)
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Zkontrolujeme, zda nedošlo k vypršení času (Max steps)
        time_is_up = self.current_step >= getattr(self, 'max_steps', 500)

        # 3. KONTROLA STAVU AGENTŮ A VÝPOČET ODMĚN
        agents_to_keep = [] # Sem si uložíme ty, co přežijí tento krok
        
        for agent in self.agents:
            # Výchozí hodnoty
            terminations[agent] = False
            truncations[agent] = time_is_up
            infos[agent] = {}
            
            # Časová penalizace (Nutí je jednat rychle)
            step_reward = -0.1
            
            # A) Je dron zničený fyzikálně? (Tvoje simulace ho smazala nebo spadl)
            if agent not in self.sim.drones:
                terminations[agent] = True
                step_reward -= 100.0  # Trest za pád
                print(f"💥 {agent} havaroval!")
                
            else:
                # Získáme aktuální data dronu
                drone = self.sim.drones[agent]
                pos = drone.get_position()
                
                # B) Zkontrolujeme hranice mapy (Out of Bounds)
                if abs(pos[0]) > self.map_bounds or abs(pos[1]) > self.map_bounds:
                    terminations[agent] = True
                    step_reward -= 50.0  # Trest za uletění z mapy
                    print(f"🚫 {agent} uletěl z mapy na pozici {pos[:2]}")
                    
                # C) Kontrola objevení ohně (Pokud dron žije)
                else:
                    # Načteme lokální výřez ohně pro tohoto drona
                    local_map = self._extract_local_fire_map(pos)
                    fire_intensity = np.sum(local_map)
                    
                    if fire_intensity > 0.1 and not self.fire_discovered:
                        self.fire_discovered = True
                        step_reward += 50.0  # Velká odměna za první detekci ohně celým týmem
                        print(f"🔥 {agent} jako první objevil oheň!")
            
            # 4. ZÁPIS VÝSLEDKŮ PRO AGENTA
            rewards[agent] = step_reward
            
            # Pokud agent žije a neukončil epizodu, přidáme jeho pozorování a necháme si ho
            if not terminations[agent]:
                observations[agent] = self._get_obs(agent)
                agents_to_keep.append(agent)
            else:
                # PettingZoo API vyžaduje, abychom pro mrtvého agenta vrátili "terminální" pozorování
                # Použijeme prostě samé nuly
                observations[agent] = {
                    "local_map": np.zeros((1, 32, 32), dtype=np.float32),
                    "self_state": np.zeros(self.total_vector_size, dtype=np.float32)
                }
                
        # 5. AKTUALIZACE SEZNAMU ŽIJÍCÍCH AGENTŮ
        self.agents = agents_to_keep
        
        return observations, rewards, terminations, truncations, infos