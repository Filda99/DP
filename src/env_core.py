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
        # Pos(3) + Vel(3) + Walls(4) + WaterLvl(1) + pos_water_fill(2-[x,y]) + init fire(2) = 15
        self.fixed_self_dim = 15
        
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
        # self.map_bounds je např. 100.0. Dron je na pozici x=80.
        # Vzdálenost k východu = 100 - 80 = 20m.
        # Nasledne znormalizujeme
        dist_measurements = self._get_boundary_measurements(pos)
        
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
    
    def _get_boundary_measurements(self, pos):
        """
        Výpočet vzdáleností k okrajům mapy.
        Vrací: (dist_north, dist_south, dist_east, dist_west) - všechny normalizované 0..1
        """
        dist_north = (self.map_bounds - pos[1]) / self.grid_size_m
        dist_south = (pos[1] - (-self.map_bounds)) / self.grid_size_m
        dist_east  = (self.map_bounds - pos[0]) / self.grid_size_m
        dist_west  = (pos[0] - (-self.map_bounds)) / self.grid_size_m
        return dist_north, dist_south, dist_east, dist_west
    
    def _get_fixed_obs(self, agent_name):
        """Generuje pozorování pro Commandera (Jen fyzika + Voda)."""
        drone = self.sim.drones[agent_name]
        pos = drone.get_position()
        vel = drone.get_velocity()
        
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
        
        dist_boundaries = self._get_boundary_measurements(pos)

        self_state = np.array([
            norm_pos[0], norm_pos[1], norm_pos[2],
            norm_vel[0], norm_vel[1], norm_vel[2],
            dist_boundaries[0], dist_boundaries[1], dist_boundaries[2], dist_boundaries[3],
            water_lvl,
            rel_base_x,
            rel_base_y,
            rel_fire_x,
            rel_fire_y
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
        # Oheň spawne kdekoli ve čtverci 100x100m kolem středu
        self.fire_x = random.uniform(-400, 400)
        self.fire_y = random.uniform(-400, 400)
        self.sim.start_fire([self.fire_x, self.fire_y], intensity=0.5)
        
        # 4. Přidáme drony na náhodné startovní pozice
        for agent in self.agents:
            # Startujte agenty jen v centrálních 60 % mapy
            safe_zone = self.map_bounds * 0.6
            start_x = random.uniform(-safe_zone, safe_zone)
            start_y = random.uniform(-safe_zone, safe_zone)
            
            if "fixed" in agent:
                # Startuje výš
                self.sim.add_fixedwing(agent, position=[start_x, start_y, 60.0], water_capacity=100.0)
            else:
                # Startuje níž
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
        
        # Mapování akcí pro simulaci
        drone_controls = {}
        for agent_name, action in actions.items():
            if "fixed" in agent_name:
                # FixedWing: [Roll, Pitch, Throttle, Water]
                # NN vrací [-1, 1], my to mapujeme
                mapped_action = np.copy(action)
                mapped_action[2] = (action[2] + 1.0) / 2.0  # Throttle 0..1
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
            
            # Časová penalizace (Nutí je jednat rychle)
            step_reward = 0.0
            
            # A) Je dron zničený fyzikálně? (Tvoje simulace ho smazala nebo spadl)
            if agent not in self.sim.drones:
                terminations[agent] = True
                step_reward = -1000.0
                print(f"💥 {agent} havaroval ve stepu {self.current_step}")
                
            else:
                # Získáme aktuální data dronu
                drone = self.sim.drones[agent]
                pos = drone.get_position()
                
                # B) Zkontrolujeme hranice mapy (Out of Bounds)
                if abs(pos[0]) > self.map_bounds or abs(pos[1]) > self.map_bounds:
                    terminations[agent] = True
                    step_reward = -1000.0
                    print(f"🚫 {agent} uletěl z mapy ve stepu {self.current_step}")
                    self.sim._destroy_drone(agent)
                    
                else:
                    # === ZMĚNA STRATEGIE: UČÍME SE LÉTAT ===
                    
                    # A) Survival Bonus (Každý krok, co žije, je dobrý)
                    step_reward += 0.5

                    # B) Explorační bonus (Nutí ho to létat a hledat)
                    # grid_x = int(pos[0] / 10.0)
                    # grid_y = int(pos[1] / 10.0)
                    # cell_key = (grid_x, grid_y)
                    # if cell_key not in self.visited_cells:
                    #     self.visited_cells.add(cell_key)
                    #     step_reward += 0.1 # Odměna za objev nového 10x10 sektoru!

                    # C) Penalizace za divoké létání (Velocity Penalty)
                    # Chceme, aby létal klidně, ne jako raketa
                    # vel = drone.get_velocity()
                    # speed = np.linalg.norm(vel)
                    # if speed > 10.0:
                    #     step_reward -= 0.1

                    # D) Penalizace za výškový limit
                    max_alt = 150.0 if "fixed" in agent else 80.0
                    if pos[2] > max_alt:
                        step_reward -= 0.1

                    # E) Penazilace za přílišné přiblíženi k hranicím mapy (Boundary Proximity)
                    dist_boundaries_norm = self._get_boundary_measurements(pos)
                    dist_to_edge_norm = min(dist_boundaries_norm[0], dist_boundaries_norm[1], dist_boundaries_norm[2], dist_boundaries_norm[3])
                    if dist_to_edge_norm < 0.1:
                        # Trest roste kvadraticky: čím blíž zdi, tím brutálnější propad rewardu
                        step_reward -= 20.0 * (1.0 - dist_to_edge_norm / 0.1)**2

                    # E) Specifické úkoly
                    if "fixed" in agent:
                        # Získáme info ze simulace, kolik vody dopadlo na oheň
                        extinguished_amount = self.sim.drone_extinguish_stats.get(agent, 0.0)
                        
                        # 1. HLAVNÍ ODMĚNA: Hašení
                        if extinguished_amount > 0:
                            step_reward += extinguished_amount * 2
                        
                        # 2. POMOCNÁ ODMĚNA: Směr k ohni (když má vodu)
                        if drone.current_water > 0:
                            # Pokud letí k ohni (využijeme tvé fire_x, fire_y z resetu)
                            dist_to_fire = np.linalg.norm(np.array([self.fire_x, self.fire_y]) - pos[:2])
                            # Malý bonus za to, že je blízko ohni
                            step_reward += (1.0 - (dist_to_fire / self.grid_size_m)) * 0.05
                        
                        # 3. POMOCNÁ ODMĚNA: Směr k základně (když je prázdný)
                        else:
                            # Skutečná pozice základny
                            if self.sim.environment.refill_zone is not None:
                                refill_pos = self.sim.environment.refill_zone['position']
                                dist_to_base = np.linalg.norm(pos[:2] - refill_pos[:2])
                                
                                # Bonus za návrat pro vodu k reálné základně
                                step_reward += (1.0 - (dist_to_base / self.grid_size_m)) * 0.05

                        # 4. TREST: Plýtvání vodou
                        # Pokud letadlo "hasí" (akce[3] > 0), ale pod ním není oheň
                        # reward_zone_small = self._extract_local_fire_map(pos, window_size_m=10.0)
                        # if actions[agent][3] > 0 and np.sum(reward_zone_small) < 0.1:
                        #     step_reward -= 0.05

                        
                    else:
                        # Pro výpočet odměny si vytáhneme jen úzký okruh 30x30m kolem dronu.
                        # Do neuronové sítě dál půjde těch 200m (to řeší metoda _get_obs),
                        # ale body dostane, jen když je fyzicky přímo nad ohněm!
                        reward_zone = self._extract_local_fire_map(pos, window_size_m=30.0)
                        fire_under_drone = np.sum(reward_zone)
                        
                        if fire_under_drone > 0.1:
                        #     if not self.fire_discovered:
                        #         self.fire_discovered = True
                        #         print(f"🎯 {agent} OBJVIL OHEŇ ve stepu {self.current_step}!")
                            
                        #     # Masivní odměna za visení PŘÍMO nad ohněm
                        #     step_reward += 0.5 + (fire_under_drone * 0.005)

                            # Bonus za oheň
                            step_reward += 1.0 + (fire_under_drone * 0.05)

                            # PENALIZACE ZA RYCHLOST nad ohněm (nutí ho zastavit)
                            speed = np.linalg.norm(drone.get_velocity())
                            step_reward -= speed * 0.01  # Čím rychleji letí nad ohněm, tím míň dostane

                        # # Quad: Odměna za sledování ohně
                        # local_map = self._extract_local_fire_map(pos)
                        # fire_intensity = np.sum(local_map)
                        
                        # if fire_intensity > 0.1:
                        #     # Masivní odměna za to, že vidí oheň.
                        #     # Přebije explorační bonus, takže dron nad ohněm zůstane viset.
                        #     # Bonus roste s tím, kolik ohně vidí (aby stál přímo nad ním).
                        #     step_reward += 1.0 + (fire_intensity * 0.01)
                
            # 4. ZÁPIS VÝSLEDKŮ PRO AGENTA
            # rewards[agent] = np.clip(step_reward, -10, 10)
            rewards[agent] = step_reward
            if time_is_up:
                rewards[agent] += 500.0 # Velká odměna za "úspěšné přežití mise"
            
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