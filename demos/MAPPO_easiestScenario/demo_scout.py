import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Importy z tvého projektu
from src.env_core import DroneFireEnv
from src.models import ScoutActor

def run_demo():
    print("🎬 Spouštím Demo: Lovec Ohně (Scout)")
    
    # 1. Konfigurace (Stejná jako při tréninku)
    N_QUADS = 2
    N_FIXED = 0
    MAX_STEPS = 500
    
    # Který model chceme načíst? (Změň číslo podle toho, který měl u tebe nejlepší Reward)
    MODEL_PATH = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/scout_solo/scout_b0210.pt"
    
    # 2. Inicializace prostředí
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=900.0, max_steps=MAX_STEPS)
    
    # Zjištění dimenzí (stejné jako v train.py)
    obs_q = env.observation_space("quad_0")
    scout_self_dim = obs_q["self_state"].shape[0]
    scout_msg_dim = 5
    scout_hidden_dim = 128
    
    # 3. Načtení sítě (sdílená pro oba scouty — MAPPO parameter sharing)
    scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim)
    
    if os.path.exists(MODEL_PATH):
        scout_actor.load_state_dict(torch.load(MODEL_PATH), strict=False)
        print(f"✅ Model {MODEL_PATH} úspěšně načten!")
    else:
        print(f"❌ CHYBA: Model {MODEL_PATH} neexistuje! Zkontroluj název souboru.")
        return
        
    scout_actor.eval()

    # 4. Spuštění Epizody
    obs, _ = env.reset(seed=103, epizode_number=20000) 
    
    quad_agents = env.quad_agents  # ["quad_0", "quad_1"]
    hidden_states = {q: torch.zeros(1, 1, scout_hidden_dim) for q in quad_agents}
    
    COLORS = ['cyan', 'lime']  # barvy pro jednotlivé scouty
    MARKERS = ['^', 's']
    
    frames = []
    total_reward = 0.0
    
    drone_paths = {q: {"x": [], "y": []} for q in quad_agents}

    # Seznamy pro logování dat (per scout)
    inputs_log = {q: [] for q in quad_agents}
    velocity_log = {q: [] for q in quad_agents}
    position_log = {q: [] for q in quad_agents}
    fire_seen_log = {q: [] for q in quad_agents}
    reward_step_log = {q: [] for q in quad_agents}

    print(f"🚁 {N_QUADS} drony startují...")
    
    for step in range(MAX_STEPS):
        actions = {}
        
        for qi, agent in enumerate(quad_agents):
            if agent not in env.agents or agent not in env.sim.drones:
                continue
            
            # Příprava dat pro síť
            local_map = torch.FloatTensor(obs[agent]["local_map"]).unsqueeze(0)
            self_state = torch.FloatTensor(obs[agent]["self_state"]).unsqueeze(0)
            neigh_s = torch.FloatTensor(obs[agent]["neighbor_states"]).unsqueeze(0)
            neigh_m = torch.BoolTensor(obs[agent]["neighbor_mask"]).unsqueeze(0)
            
            with torch.no_grad():
                dist, message, h_out = scout_actor(local_map, self_state, neigh_s, neigh_m, hidden_states[agent])
                action = dist.mean
                
            hidden_states[agent] = h_out
            actions[agent] = action.squeeze(0).numpy()
            
            # === DIAGNOSTICKÉ VÝPISY ===
            fire_pixels = np.sum(obs[agent]["local_map"])
            pos = env.sim.drones[agent].get_position()
            fire_alert = "🔥 VIDÍ OHEŇ!" if fire_pixels > 0.05 else "👀 Slepý"
            
            if step % 10 == 0:
                act_np = action.squeeze(0).numpy().round(2)
                print(f"Krok {step:03d} [{agent}] | Pozice: [X:{pos[0]:5.1f}, Y:{pos[1]:5.1f}, Z:{pos[2]:5.1f}] | Kamera: {fire_pixels:5.2f} {fire_alert} | Akce: {act_np}")
        
        if not actions:
            print("💥 Všechny drony havarovaly!")
            break
        
        # Krok prostředí
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in quad_agents:
            r = rewards.get(agent, 0.0)
            total_reward += r
            reward_step_log[agent].append(r)
            
            if agent in env.sim.drones:
                drone = env.sim.drones[agent]
                fire_pixels = np.sum(obs.get(agent, {}).get("local_map", np.zeros(1)))
                fire_seen_log[agent].append(fire_pixels)
                velocity_log[agent].append(np.linalg.norm(drone.get_velocity()))
                position_log[agent].append(drone.get_position().copy())
                inputs_log[agent].append(actions.get(agent, np.zeros(4)).copy())
            
            if terminations.get(agent, False) or truncations.get(agent, False):
                print(f"🏁 {agent} ukončen na kroku {step}")
        
        # Kontrola, jestli ještě žijí
        alive = [q for q in quad_agents if q in env.agents]
        if not alive:
            break

        # --- VYKRESLOVÁNÍ (Každý 2. krok) ---
        if step % 2 == 0:
            b = env.map_bounds
            fire_map = env.sim.environment.fire_grid.I
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_facecolor('#2b2b2b')
            
            extent = [-b, b, -b, b]
            fire_map_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
            ax.imshow(fire_map_masked, extent=extent, origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
            
            for qi, agent in enumerate(quad_agents):
                if agent not in env.sim.drones:
                    continue
                pos = env.sim.drones[agent].get_position()
                color = COLORS[qi]
                
                drone_paths[agent]["x"].append(pos[0])
                drone_paths[agent]["y"].append(pos[1])
                
                # Stopa
                ax.plot(drone_paths[agent]["x"], drone_paths[agent]["y"],
                        color=color, alpha=0.5, linestyle=':', linewidth=2)
                
                # Pozice
                ax.scatter(pos[0], pos[1], c=color, s=150, marker=MARKERS[qi],
                           edgecolors='white', label=agent)
                
                # Zorné pole
                current_z = pos[2]
                win_size = max(10.0, current_z * 1.5)
                rect = plt.Rectangle((pos[0]-win_size/2, pos[1]-win_size/2), win_size, win_size, 
                                    fill=False, edgecolor=color, linestyle='-', alpha=0.8)
                ax.add_patch(rect)
            
            ax.set_xlim(-b, b)
            ax.set_ylim(-b, b)
            ax.set_title(f"Krok: {step:03d} | Reward: {total_reward:.1f}", color='black', fontsize=14)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(color='white', alpha=0.1)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames.append(np.array(img))
            plt.close(fig)

    print(f"🏁 Hotovo! Celkový Reward: {total_reward:.2f}")
    
    # 5. Uložení jako GIF
    output_filename = "scout_demo.gif"
    print(f"💾 Ukládám video do {output_filename} (to může chvíli trvat)...")
    imageio.mimsave(output_filename, frames, fps=15, loop=0) 
    print("✅ Video úspěšně uloženo!")

    # === GENEROVÁNÍ GRAFU ANALÝZY (per scout) ===
    fig, axes = plt.subplots(5, N_QUADS, figsize=(8 * N_QUADS, 18), squeeze=False)
    
    for qi, agent in enumerate(quad_agents):
        if not inputs_log[agent]:
            continue
        steps_range = range(len(inputs_log[agent]))
        
        # 1. Vstupy (Actions)
        ax = axes[0, qi]
        inputs_np = np.array(inputs_log[agent])
        labels = ['Roll', 'Pitch', 'Yaw', 'Throttle']
        for i in range(4):
            ax.plot(steps_range, inputs_np[:, i], label=labels[i])
        ax.set_title(f"{agent} — Vstupy do dronu (Actions)")
        ax.set_ylabel("Intenzita [-1, 1]")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Rychlost
        ax = axes[1, qi]
        ax.plot(steps_range, velocity_log[agent], color='red')
        ax.set_title(f"{agent} — Rychlost")
        ax.set_ylabel("m/s")
        ax.grid(True, alpha=0.3)

        # 3. Pozice
        ax = axes[2, qi]
        pos_np = np.array(position_log[agent])
        ax.plot(steps_range, pos_np[:, 0], label='X')
        ax.plot(steps_range, pos_np[:, 1], label='Y')
        ax.plot(steps_range, pos_np[:, 2], label='Z (Altitude)')
        ax.set_title(f"{agent} — Pozice")
        ax.set_ylabel("Metry")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Detekce ohně
        ax = axes[3, qi]
        ax.fill_between(steps_range, fire_seen_log[agent], color='orange', alpha=0.5)
        ax.set_title(f"{agent} — Detekce ohně")
        ax.set_ylabel("Suma pixelů")
        ax.grid(True, alpha=0.3)

        # 5. Reward per step
        ax = axes[4, qi]
        ax.bar(steps_range, reward_step_log[agent], color='green', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f"{agent} — Step Reward")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Krok epizody")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("demo_analysis_full.png")
    print("📊 Komplexní analýza uložena jako 'demo_analysis_full.png'")

if __name__ == "__main__":
    run_demo()