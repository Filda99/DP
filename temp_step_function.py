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
                local_map = new_obs["quads"]["local_map"]
                
                # Check if we have valid local map data
                if len(local_map) > 0:
                    # Get fire layer for first drone
                    local_fire = local_map[0, 0]  # Shape: [n_drones, channels, H, W] -> [H, W]
                    fire_intensity = np.sum(local_fire)
                    
                    if fire_intensity > 0.1:  # Vidí oheň!
                        fire_visible = True
                        
                        # Najdi konkrétní fire cells v local mapě
                        fire_cells_in_view = []
                        local_size = local_fire.shape[-1]  # Předpokládáme čtvercovou mapu
                        center = local_size // 2
                        
                        for i in range(local_size):
                            for j in range(local_size):
                                if local_fire[i, j] > 0.1:  # Burning cell
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
            
            # === 3. TIME PENALTY (-0.1 za vteřinu) ===
            time_elapsed = self.sim.simulation_time - self.episode_start_time
            reward -= time_elapsed * 0.01  # Velmi malá penalizace za čas
            
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