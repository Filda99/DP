    def step(self, actions):
        """Execute one environment step with CLEAN exploration-based reward system"""
        
        # 1. Execute actions
        if isinstance(actions, dict) and "quads" in actions and "action" in actions["quads"]:
            action_vector = actions["quads"]["action"]
            if len(self.quad_agents) > 0:
                agent_actions = {self.quad_agents[0]: action_vector}
                self.sim.step(agent_actions)
        else:
            # Invalid action format - just step with no actions
            self.sim.step({})
        
        # 2. Collect new state
        new_obs = self._get_obs()
        
        # 3. ===== CLEAN EXPLORATION-BASED REWARD SYSTÉM =====
        reward = 0.0  # Žádná základní odměna - pouze za akce!
        
        # Získej pozici prvního drona
        drone_name = self.quad_agents[0] if self.quad_agents else None
        
        if drone_name and drone_name in self.sim.drones:
            drone_pos = self.sim.drones[drone_name].get_position()
            x, y, z = drone_pos
            
            # === 1. EXPLORATION REWARD (+2 za nové políčko) ===
            current_cell = (int(x), int(y))
            if current_cell not in self.visited_cells:
                self.visited_cells.add(current_cell)
                reward += 2.0  # Odměna za exploration
            
            # === 2. FIRE DISCOVERY REWARD (+10 za first discovery) ===
            # Správný access k local_map
            if "quads" in new_obs and "local_map" in new_obs["quads"] and len(new_obs["quads"]["local_map"]) > 0:
                # new_obs["quads"]["local_map"] je array shape (1, 15, 15) - první drone
                local_fire = new_obs["quads"]["local_map"][0]  # Shape (15, 15)
                fire_intensity = np.sum(local_fire)
                
                if fire_intensity > 0.1:  # Vidí oheň!
                    # První discovery bonus
                    if not self.fire_discovered:
                        reward += 10.0  # VELKÁ odměna za první nalezení ohně
                        self.fire_discovered = True
                    
                    # === 3. FIRE MONITORING REWARD (+4 za kontinuální monitoring) ===
                    reward += 4.0  # Odměna za vidění ohně
                    
                else:
                    # === 4. NO FIRE PENALTY (-3 za nevidění ohně poté co byl objeven) ===
                    if self.fire_discovered:
                        reward -= 3.0  # Penalizace za ztrátu ohně z dohledu
                    else:
                        reward -= 0.5  # Menší penalizace když ještě nebyl nalezen
            
            # === 5. BOUNDARY CHECK ===
            if abs(x) > self.map_bounds or abs(y) > self.map_bounds:
                reward -= 5.0  # Penalizace za vylĚtení z mapy
            
            # === 6. VÝŠKA CHECK ===
            if z < 1.0:  # Příliš nízko (crash risk)
                reward -= 5.0
            elif z > 30.0:  # Příliš vysoko (neefektivní)
                reward -= 2.0
                
        else:
            # Dron neexistuje/crashnul - velká penalizace
            reward = -10.0
        
        # === 7. TIME PENALTY (motivace být efektivní) ===
        reward -= 0.1  # Malá časová penalizace

        # 4. Check termination
        terminated = self.sim.simulation_time > 120  # 2 minute timeout
        
        # 5. Ensure reward is always a valid number
        reward = float(reward) if reward is not None else -10.0
        
        return new_obs, reward, terminated, False, {}