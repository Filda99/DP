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
    metadata = {"render_modes": ["human"], "name": "drone_fire_v2"}

    def __init__(self, num_quads=1, num_fixed=1, grid_size_m=200.0, local_map_size=32, max_steps=500):
        super().__init__()
        
        self.max_steps = max_steps
        self.num_quads = num_quads
        self.num_fixed = num_fixed
        self.grid_size_m = grid_size_m
        self.map_bounds = self.grid_size_m / 2.0
        
        # Seznamy agentů
        self.quad_agents = [f"quad_{i}" for i in range(self.num_quads)]
        self.fixed_agents = [f"fixed_{i}" for i in range(self.num_fixed)]
        self.possible_agents = self.quad_agents + self.fixed_agents
        self.agents = self.possible_agents[:]
        
        # === ACTION SPACE ===
        # Quad: [Roll, Pitch, Yaw, Throttle]
        # Fixed: [Roll, Pitch, Throttle, Water_Trigger] (Mapování se děje ve step())
        self._action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        # === OBSERVATION SPACE (ASYMETRICKÝ) ===
        
        # 1. Konfigurace pro SCOUTA (Quad)
        # Co znamená 12 čísel v self_state:
        # 0-2: pozice x, y, z (relativně k mapě)
        # 3-5: rychlost vx, vy, vz
        # 6-9: vzdálenost k okrajům (sever, jih, východ, západ)
        # 10-11: informace o ohni (smer dx, dy)
        self.quad_self_dim = 12
        self.max_neighbors = self.num_quads - 1 if self.num_quads > 1 else 1 # Aby to nepadlo při 1 dronu
        
        quad_obs_space = gym.spaces.Dict({
            "local_map": gym.spaces.Box(low=0.0, high=1.0, shape=(1, local_map_size, local_map_size), dtype=np.float32),
            "self_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.quad_self_dim,), dtype=np.float32),
            # Pozice sousedů (pro Self-Attention)
            "neighbor_states": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_neighbors, 3), dtype=np.float32),
            # Maska: True = Ignorovat (Mrtvý/Padding), False = Validní
            "neighbor_mask": gym.spaces.Box(low=0, high=1, shape=(self.max_neighbors,), dtype=bool)
        })

        # 2. Konfigurace pro COMMANDERA (Fixed)
        # Pos(3) + Vel(3) + Walls(4) + WaterLvl(1) + pos_water_fill(2-[x,y]) + init fire(2) + [roll, pitch, yaw] + Danger zone activated (1)
        self.fixed_self_dim = 19
        
        fixed_obs_space = gym.spaces.Dict({
            "self_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.fixed_self_dim,), dtype=np.float32)
            # Zprávy (Messages) se sem nedávají, ty přijdou zvenčí v train.py!
        })

        self._observation_spaces = {}
        for agent in self.quad_agents:
            self._observation_spaces[agent] = quad_obs_space
        for agent in self.fixed_agents:
            self._observation_spaces[agent] = fixed_obs_space
        
        # === GLOBAL STATE SPACE (Kritik) ===
        # Mapa(256) + Všechny Quads(10 * N) + Všechny Fixed(11 * M)
        self.global_state_size = (16 * 16) + (self.num_quads * self.quad_self_dim) + (self.num_fixed * self.fixed_self_dim)
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.global_state_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    # Metody, které napíšeme v dalších krocích:
    def _get_quad_obs(self, agent_name):
        """
        Vygeneruje pozorování pro konkrétního agenta (kvadrokoptéru).
        """
        drone = self.sim.drones[agent_name]
        pos = drone.get_position()  # [x, y, z]
        vel = drone.get_velocity()  # [vx, vy, vz]

        # === NORMALIZACE VSTUPŮ (Kritické pro stabilitu sítě!) ===
        # Pozice dělíme hranicí mapy (aby byly cca -1..1)
        norm_pos = pos / self.map_bounds 
        # Rychlost dělíme 20 m/s (max speed)
        norm_vel = vel / 20.0 

        # KOMPAS: Relativní vektor k ohni (normalizovaný na mapu)
        # fire_x/y jsou definovány v reset()
        rel_fire_x = (self.fire_x - pos[0]) / self.map_bounds
        rel_fire_y = (self.fire_y - pos[1]) / self.map_bounds
        
        # 1. Výpočet vzdáleností k okrajům mapy (Boundary Awareness)
        dist_measurements = self._get_boundary_measurements_norm(pos)
        
        # 2. Sestavení Self-State
        self_state = np.array([
            norm_pos[0], norm_pos[1], norm_pos[2],    # 0-2: Pozice (x,y,z)
            norm_vel[0], norm_vel[1], norm_vel[2],    # 3-5: Rychlost (vx,vy,vz)
            dist_measurements[0], dist_measurements[1], dist_measurements[2], dist_measurements[3], # 6-9: Vzdálenosti ke zdem
            rel_fire_x, rel_fire_y # 10-11: Směr k ohni (dx, dy)
        ], dtype=np.float32)

        # 3. Získání pozic OSTATNÍCH dronů (Self-Attention Inputs)
        neighbor_states = []
        neighbor_mask = []
        
        # Projdeme ostatní Scouty
        for other in self.quad_agents:
            if other == agent_name: continue # Sebe nepočítám
            
            if other in self.sim.drones:
                # Validní soused
                other_pos = self.sim.drones[other].get_position()
                rel_pos = other_pos - pos
                # Normalizace relativní pozice
                neighbor_states.append(rel_pos / self.grid_size_m)
                neighbor_mask.append(False) # False = Validní (neignorovat)
            else:
                # Mrtvý soused (Padding)
                neighbor_states.append(np.zeros(3, dtype=np.float32))
                neighbor_mask.append(True) # True = IGNOROVAT v Attention

        # Pokud máš celkově jen 1 dron, doplň do max_neighbors
        while len(neighbor_states) < self.max_neighbors:
            neighbor_states.append(np.zeros(3, dtype=np.float32))
            neighbor_mask.append(True)
        
        # 4. Získání lokální mapy (Local Map 32x32)
        local_map = self._extract_local_fire_map(pos)

        return {
            "local_map": local_map,
            "self_state": self_state,
            "neighbor_states": np.array(neighbor_states, dtype=np.float32),
            "neighbor_mask": np.array(neighbor_mask, dtype=bool)
        }
    
    def _get_boundary_measurements_norm(self, pos):
        distances = self._get_boundary_measurements(pos)
        return np.array(distances) / 2000.0
    
    def _get_boundary_measurements(self, pos):
        """
        Výpočet vzdáleností k okrajům mapy.
        Vrací: (dist_north, dist_south, dist_east, dist_west) - všechny normalizované 0..1
        """
        dist_north = (self.map_bounds - pos[1])
        dist_south = (pos[1] - (-self.map_bounds))
        dist_east  = (self.map_bounds - pos[0])
        dist_west  = (pos[0] - (-self.map_bounds))
        return dist_north, dist_south, dist_east, dist_west
    
    def _get_fixed_obs(self, agent_name):
        """Generuje pozorování pro Commandera (Jen fyzika + Voda)."""
        drone = self.sim.drones[agent_name]
        pos = drone.get_position()
        vel = drone.get_velocity()
        rpy = drone.get_orientation_rpy() # [roll, pitch, yaw] v radiánech
        
        # Voda
        water_lvl = drone.current_water / drone.water_capacity if drone.water_capacity > 0 else 0.0
        
        # Vytáhneme skutečnou pozici zóny ze simulace
        if self.sim.environment.refill_zone is not None:
            refill_pos = self.sim.environment.refill_zone['position']
            # Výpočet relativního kompasu k základně
            rel_base_x = (refill_pos[0] - pos[0]) / self.map_bounds
            rel_base_y = (refill_pos[1] - pos[1]) / self.map_bounds
        else:
            rel_base_x = 0.0
            rel_base_y = 0.0
        
        # 2. NOVÉ: Kompas k ohni (aby letadlo vědělo, kam se vrátit po doplnění)
        rel_fire_x = (self.fire_x - pos[0]) / self.map_bounds
        rel_fire_y = (self.fire_y - pos[1]) / self.map_bounds

        # === NORMALIZACE ===
        norm_pos = pos / self.map_bounds
        norm_vel = vel / 20.0
        norm_rpy = rpy / np.pi # Převod z [-pi, pi] na cca [-1, 1]
        
        # 1. Výpočet reálných vzdáleností k okrajům (ne normalizovaných)
        dist_boundaries = self._get_boundary_measurements_norm(pos)

        # 2. DANGER FLAG: 1.0 pokud je blíž než 150m k okraji, jinak 0.0
        # 150 metrů je pro letadlo při rychlosti 20 m/s cca 7 sekund letu.
        danger_flag = 1.0 if min(self._get_boundary_measurements(pos)) < 300.0 else 0.0

        self_state = np.array([
            norm_pos[0], norm_pos[1], norm_pos[2],
            norm_vel[0], norm_vel[1], norm_vel[2],
            dist_boundaries[0], dist_boundaries[1], dist_boundaries[2], dist_boundaries[3],
            water_lvl,
            rel_base_x, rel_base_y,
            rel_fire_x, rel_fire_y,
            norm_rpy[0], norm_rpy[1], norm_rpy[2],
            danger_flag
        ], dtype=np.float32)
        
        return {
            "self_state": self_state
        }

    def _get_obs(self, agent_name):
        # Rozcestník podle typu agenta
        if "fixed" in agent_name:
            if agent_name in self.sim.drones:
                return self._get_fixed_obs(agent_name)
            else:
                # Mrtvé letadlo
                return {"self_state": np.zeros(self.fixed_self_dim, dtype=np.float32)}
        else:
            if agent_name in self.sim.drones:
                return self._get_quad_obs(agent_name)
            else:
                # Mrtvý dron
                return {
                    "local_map": np.zeros((1, 32, 32), dtype=np.float32),
                    "self_state": np.zeros(self.quad_self_dim, dtype=np.float32),
                    "neighbor_states": np.zeros((self.max_neighbors, 3), dtype=np.float32),
                    "neighbor_mask": np.ones((self.max_neighbors,), dtype=bool) # True = Ignorovat vše
                }

    def _extract_local_fire_map(self, pos, resolution_px=32):
        """
        Fixed field of view but based on drone's altitude (higher = wider view). Returns normalized intensity map.
        But the higher the drone is, the more it sees but with less detail (like a zoom out). 
        This way, the agent can learn to fly higher for better situational awareness or lower for precision.
        """
        if self.sim.environment.fire_grid is None:
            return np.zeros((1, resolution_px, resolution_px), dtype=np.float32)
            
        mapper = self.sim.environment.grid_mapper
        fire_grid = self.sim.environment.fire_grid.I  # Intenzita ohně
        
        # pos[2] je výška dronu (z)
        # Čím je výš, tím větší "okno" vidí. 
        # Např. při z=10m uvidí 15m, při z=80m uvidí 120m.
        adaptive_window = max(10.0, pos[2] * 1.5)
        
        # Hranice výřezu v metrech
        half_w = adaptive_window / 2.0
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
            import cv2
            # Přidáme ochranu: pokud je matice intenzity prázdná, resize selže
            try:
                processed_map = cv2.resize(crop, (resolution_px, resolution_px), interpolation=cv2.INTER_LINEAR)
            except:
                processed_map = np.zeros((resolution_px, resolution_px), dtype=np.float32)
            
        return processed_map[np.newaxis, ...].astype(np.float32)

    def state(self):
        """
        MAPPO požadavek: Globální stav mapy pro centrálního Kritika.
        Vrací zploštělou (flattened) matici ohně + stavy všech agentů.
        """
        # 1. Zmenšený globální oheň (např. 16x16)
        if self.sim.environment.fire_grid is not None:
            # TADY získáme aktuální matici ohně (např. těch tvých 200x200 buněk)
            full_grid = self.sim.environment.fire_grid.I
            import cv2
            # TADY proběhne ta rychlá matematika zmenšení na 16x16 pomocí zprůměrování plochy
            small_grid = cv2.resize(full_grid, (16, 16), interpolation=cv2.INTER_AREA)
            # TADY z 2D matice 16x16 uděláme 1D pole o 256 hodnotách, které umí sežrat Kritik
            fire_summary = small_grid.flatten()
        else:
            fire_summary = np.zeros(16 * 16, dtype=np.float32)

        # 2. Sezbíráme stavy všech dronů (abychom věděli, kde celý tým je)
        agent_states = []
        
        # Nejdřív Quady
        for agent in self.quad_agents:
            if agent in self.sim.drones:
                agent_states.append(self._get_quad_obs(agent)["self_state"])
            else:
                agent_states.append(np.zeros(self.quad_self_dim, dtype=np.float32))
                
        # Pak Fixed
        for agent in self.fixed_agents:
            if agent in self.sim.drones:
                agent_states.append(self._get_fixed_obs(agent)["self_state"])
            else:
                agent_states.append(np.zeros(self.fixed_self_dim, dtype=np.float32))

        # Spojíme oheň a stavy dronů do jednoho obřího 1D pole pro Kritika
        global_state = np.concatenate([fire_summary] + agent_states)
        return global_state
        
    def reset(self, seed=None, options=None, epizode_number=0):
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

        # Ohen startovni pozice
        if epizode_number < 2000:
            safe_zone = self.map_bounds * 0.1
            self.sim.start_fire([0, 0], intensity=0.5)
            self.fire_x = 0.0
            self.fire_y = 0.0
        else:
            safe_zone = self.map_bounds * 0.6
            self.fire_x = random.uniform(-safe_zone, safe_zone)
            self.fire_y = random.uniform(-safe_zone, safe_zone)
            self.sim.start_fire([self.fire_x, self.fire_y], intensity=0.5)
        # Dron startovni pozice        
        if epizode_number < 300:
            start_x = random.uniform(-10, 10)
            start_y = random.uniform(-10, 10)
            start_z = random.uniform(10.0, 40.0)
        else:
            start_x = random.uniform(-safe_zone, safe_zone)
            start_y = random.uniform(-safe_zone, safe_zone)
            start_z = random.uniform(10.0, 40.0)

        # 4. Přidáme drony na náhodné startovní pozice
        for agent in self.agents:
            
            
            if "fixed" in agent:
                # Vypočítáme vektor od startu do středu [0,0]
                to_center_vec = -np.array([start_x, start_y])
                yaw_to_center = np.arctan2(to_center_vec[1], to_center_vec[0])
                
                # Přidej do sim.add_fixedwing parametr pro orientaci (pokud ho tvá sim podporuje)
                self.sim.add_fixedwing(agent, position=[start_x, start_y, 60.0], water_capacity=200.0, yaw=yaw_to_center)

                drone = self.sim.drones[agent]
                drone.state_va = 15.0
            else:
                # Startuje níž
                self.sim.add_quadcopter(agent, position=[start_x, start_y, start_z])
            
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
        frame_skip = 5
        
        # Mapování akcí pro simulaci
        drone_controls = {}
        for agent_name, action in actions.items():
            if "fixed" in agent_name:
                # FixedWing: [Roll, Pitch, Throttle, Water]
                # NN vrací [-1, 1], my to mapujeme
                mapped_action = np.copy(action)
                mapped_action[2] = 0.4 + (action[2] + 1.0) * 0.3 # 0.4 až 1.0 (prevence stallu)
                mapped_action[3] = (action[3] + 1.0) / 2.0  # Water 0..1
                drone_controls[agent_name] = mapped_action
            else:
                # Quad: [Roll, Pitch, Yaw, Throttle]
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
            
            # Kontrola přežití (Fyzika + Hranice)
            dead, crash_reward = self._check_death(agent)

            rewards[agent] = 0
            
            if dead:
                terminations[agent] = True
                rewards[agent] += crash_reward
                    
            else:
                rewards[agent] += self._apply_physics_shaping(agent)

                # E) Specifické úkoly
                if "fixed" in agent:
                    # Kombinace: Letadlo musí nejdřív umět létat (survival) 
                    # a pak teprve řešit oheň (mission)
                    survival = self._get_fixed_reward_survival(agent)
                    # mission = self._get_fixed_reward(agent)
                    # rewards[agent] += (survival + (mission * 2.0)) / 2
                    rewards[agent] += survival
                else:
                    rewards[agent] += self._get_quad_reward(agent)
                
                if time_is_up:
                    rewards[agent] += 10.0
                
                rewards[agent] = np.clip(rewards[agent], -10.0, 10.0)
            
            # Pokud agent žije a neukončil epizodu, přidáme jeho pozorování a necháme si ho
            if not terminations[agent]:
                observations[agent] = self._get_obs(agent)
                agents_to_keep.append(agent)
            else:
                # PettingZoo API vyžaduje, abychom pro mrtvého agenta vrátili "terminální" pozorování
                if "fixed" in agent:
                    observations[agent] = {"self_state": np.zeros(self.fixed_self_dim, dtype=np.float32)}
                else:
                    observations[agent] = {
                        "local_map": np.zeros((1, 32, 32), dtype=np.float32),
                        "self_state": np.zeros(self.quad_self_dim, dtype=np.float32),
                        "neighbor_states": np.zeros((self.max_neighbors, 3), dtype=np.float32),
                        "neighbor_mask": np.ones((self.max_neighbors,), dtype=bool)
                    }
            
            
        # 5. AKTUALIZACE SEZNAMU ŽIJÍCÍCH AGENTŮ
        self.agents = agents_to_keep
        
        return observations, rewards, terminations, truncations, infos
    

    # === PRIVÁTNÍ METODY PRO ODMĚNY ===

    def _apply_physics_shaping(self, agent):
        """Společná pravidla fyziky: přežití, výška a hranice mapy."""
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        reward = 0.01

        # 1. Penalizace za přílišné přiblížení k hranicím (Boundary Proximity)
        dist_boundaries = self._get_boundary_measurements(pos)
        dist_to_edge = min(dist_boundaries)
        threshold = self.map_bounds / 2 * 0.25 if "fixed" in agent else self.map_bounds / 2 * 0.1 
        if dist_to_edge < threshold:
            reward -= 0.3 * (1.0 - dist_to_edge / threshold)**2

        # 1b. Přísnější penalizace pro letadlo, kdy se blíží k okraji
        if "fixed" in agent:
            if dist_to_edge < threshold * 0.5:
                reward -= 0.5

        # 2. Penalizace za výškový limit
        max_alt = 200.0 if "fixed" in agent else 150.0
        min_alt = 15.0 if "fixed" in agent else 35.0
        if pos[2] > max_alt or pos[2] < min_alt:
            reward -= 0.05
            
        return reward

    def _get_quad_reward(self, agent):
        """Odměna pro dron (Scout)."""
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        reward = 0
        
        # Adaptivní kamera pro odměnu (využívá stejnou logiku jako senzory)
        reward_zone = self._extract_local_fire_map(pos) 
        avg_fire_intensity = np.mean(reward_zone) # Čím víc ohně pod dronem, tím větší odměna
        
        if avg_fire_intensity > 0.001:
            reward += 1
            reward += (avg_fire_intensity * 10)
            # Penalizace za rychlost nad ohněm
            speed = np.linalg.norm(drone.get_velocity())
            reward -= speed * 0.001
            
        return reward

    def _get_fixed_reward_survival(self, agent):
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        vel = drone.get_velocity()
        reward = 0.05
 
        dist_from_center = np.linalg.norm(pos[:2])
        # 1. Koblihový bonus (všude do 500m od středu je to super)
        if dist_from_center < 500.0:
            reward += 0.15
        else:
            # Čím dál je, tím menší má plošný bonus
            reward += max(0.0, 0.15 * (1.0 - ((dist_from_center - 500.0) / (self.map_bounds - 500.0))))
            # 2. GUMOVÉ LANO (Magnet na střed) - To ho donutí zatočit!
            # Vypočítáme vektor směřující přesně do středu [0,0]
            vec_to_center = -pos[:2]
            dir_to_center = vec_to_center / dist_from_center
            # Skalární součin: Jak moc jeho aktuální rychlost míří do středu?
            # Kladné číslo = letí do bezpečí. Záporné = letí ven z mapy.
            approach_speed = np.dot(vel[:2], dir_to_center)
            # Tady je to kouzlo: Pokud točí zpět, dostane obrovskou odměnu, která přebije tresty u zdi!
            reward += approach_speed * 0.02
        return reward

    def _get_fixed_reward(self, agent):
        """
        Odměna pro letadlo (Commander).
        REŽIM MISE: Hašení + Logistika
        """
        # drone = self.sim.drones[agent]
        # pos = drone.get_position()
        
        # # Hašení
        # extinguished = self.sim.drone_extinguish_stats.get(agent, 0.0)
        # reward = min(0.1, extinguished * 0.05)
            
        # # Směrová navigace (k ohni s vodou / k bázi bez vody)
        # if drone.current_water > 0:
        #     dist = np.linalg.norm(np.array([self.fire_x, self.fire_y]) - pos[:2])
        # elif self.sim.environment.refill_zone:
        #     ref_pos = self.sim.environment.refill_zone['position']
        #     dist = np.linalg.norm(pos[:2] - ref_pos[:2])
        # else: dist = self.grid_size_m
            
        # reward += (1.0 - (dist / self.grid_size_m)) * 0.05
        # return reward
        drone = self.sim.drones[agent]
        pos = drone.get_position()
        vel = drone.get_velocity()[:2] # Rychlost v ploše XY
        
        # Určení cíle (oheň nebo báze)
        if drone.current_water > 0:
            target_pos = np.array([self.fire_x, self.fire_y])
        elif self.sim.environment.refill_zone:
            target_pos = self.sim.environment.refill_zone['position'][:2]
        else: 
            return 0.0
            
        # Vektor k cíli
        vec_to_target = target_pos - pos[:2]
        dist = np.linalg.norm(vec_to_target)
        
        if dist < 1e-5: return 0.01
        
        # Normalizovaný směr k cíli
        dir_to_target = vec_to_target / dist
        
        # RADIÁLNÍ RYCHLOST: Jak moc letím přímo k cíli (skalární součin)
        # Pokud letím přímo k cíli, hodnota je vysoká. Pokud od něj, je záporná.
        approach_speed = np.dot(vel, dir_to_target)
        
        reward = approach_speed * 0.01 # Bonus za přibližování
        reward += (1.0 - (dist / self.grid_size_m)) * 0.005 # Bonus za blízkost
        
        return reward


    def _check_death(self, agent):
        """Kontrola havárie nebo opuštění mapy."""
        if agent not in self.sim.drones:
            print(f"💥 {agent} havaroval ve stepu {self.current_step}")
            return True, -50
            
        pos = self.sim.drones[agent].get_position()
        if abs(pos[0]) > self.map_bounds or abs(pos[1]) > self.map_bounds:
            print(f"🚫 {agent} uletěl z mapy ve stepu {self.current_step}")
            self.sim._destroy_drone(agent)
            return True, -50
            
        return False, 0.0