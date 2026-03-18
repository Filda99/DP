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
    N_QUADS = 1
    N_FIXED = 0
    MAX_STEPS = 1000
    
    # Který model chceme načíst? (Změň číslo podle toho, který měl u tebe nejlepší Reward)
    MODEL_PATH = "/homes/eva/xj/xjahnf00/tmp/DP/results/TrainingQuad/08_QuadTrainedWithDemo/scout_ep24600.pt" 
    
    # 2. Inicializace prostředí
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=1000.0)
    
    # Zjištění dimenzí (stejné jako v train.py)
    obs_q = env.observation_space("quad_0")
    scout_self_dim = obs_q["self_state"].shape[0]
    scout_msg_dim = 5
    scout_hidden_dim = 128
    
    # 3. Načtení sítě
    scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim)
    
    if os.path.exists(MODEL_PATH):
        scout_actor.load_state_dict(torch.load(MODEL_PATH))
        print(f"✅ Model {MODEL_PATH} úspěšně načten!")
    else:
        print(f"❌ CHYBA: Model {MODEL_PATH} neexistuje! Zkontroluj název souboru.")
        return
        
    # Přepneme model do evaluačního módu (vypne dropouty apod., pokud by tam byly)
    scout_actor.eval()

    # 4. Spuštění Epizody
    # Zvolíme seed, aby oheň byl na nějakém pěkném místě pro ukázku
    obs, _ = env.reset(seed=42, epizode_number=2000) 
    hidden_state = torch.zeros(1, 1, scout_hidden_dim)
    
    frames = []
    total_reward = 0.0
    
    drone_path_x = []
    drone_path_y = []

    # Seznamy pro logování dat
    inputs_log = []      # Akce [Roll, Pitch, Yaw, Throttle]
    velocity_log = []    # Rychlost (velikost vektoru)
    position_log = []    # Pozice [X, Y, Z]
    fire_seen_log = []   # Kolik ohně dron vidí
    reward_step_log = [] # Odměna v daném kroku

    print("🚁 Dron startuje...")
    
    for step in range(MAX_STEPS):
        agent = "quad_0"
        if not env.sim.drones[agent]: 
            print("💥 Dron havaroval nebo uletěl!")
            break
            
        
        # Příprava dat pro síť
        local_map = torch.FloatTensor(obs[agent]["local_map"]).unsqueeze(0)
        self_state = torch.FloatTensor(obs[agent]["self_state"]).unsqueeze(0)
        neigh_s = torch.FloatTensor(obs[agent]["neighbor_states"]).unsqueeze(0)
        neigh_m = torch.BoolTensor(obs[agent]["neighbor_mask"]).unsqueeze(0)
        
        # Inference (BEZ gradientů)
        with torch.no_grad():
            dist, message, hidden_state = scout_actor(local_map, self_state, neigh_s, neigh_m, hidden_state)
            
            # ❗ KLÍČOVÝ ROZDÍL PROTI TRÉNINKU ❗
            # Při tréninku jsme dělali "dist.sample()" (přidává šum pro objevování).
            # Při demu chceme "čistou" naučenou dovednost, proto bereme PRŮMĚR (mean) akce.
            # Dron tak poletí krásně plynule a nebude se "třást".
            action = dist.mean
            
        actions = {agent: action.squeeze(0).numpy()}

        # === DIAGNOSTICKÉ VÝPISY ===
        # 1. Zjistíme, jestli má kamera (local_map) na vstupu nějaký oheň
        fire_pixels = np.sum(obs[agent]["local_map"])
        
        # 2. Zjistíme skutečnou pozici
        pos = env.sim.drones[agent].get_position()
        
        # 3. Vypsání do konzole
        fire_alert = "🔥 VIDÍ OHEŇ!" if fire_pixels > 0.05 else "👀 Slepý"
        
        # Vypíšeme to např. každý 5. krok, ať nám to nezahltí celou konzoli
        if step % 5 == 0:
            act_np = action.squeeze(0).numpy().round(2)
            print(f"Krok {step:03d} | Pozice: [X:{pos[0]:5.1f}, Y:{pos[1]:5.1f}, Z:{pos[2]:5.1f}] | Kamera: {fire_pixels:5.2f} {fire_alert} | Akce [Roll, Pitch, Yaw, Thr]: {act_np}")
        # ===========================
        
        # Krok prostředí
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += rewards.get(agent, 0.0)

        # Výpočet viděného ohně (již v kódu máš jako fire_pixels)
        fire_pixels = np.sum(obs[agent]["local_map"])
        
        # Uložení do historie
        fire_seen_log.append(fire_pixels)
        reward_step_log.append(rewards.get(agent, 0.0))
        
        # (Zde zůstává i to minulé logování rychlosti, pozice a vstupů)
        drone = env.sim.drones[agent]
        velocity_log.append(np.linalg.norm(drone.get_velocity()))
        position_log.append(drone.get_position().copy())
        inputs_log.append(action.squeeze(0).numpy().copy())
        
        # --- VYKRESLOVÁNÍ (Každý 2. krok, ať GIF není zbytečně obrovský) ---
        if step % 2 == 0 and agent in env.sim.drones:
            b = env.map_bounds
            drone = env.sim.drones[agent]
            pos = drone.get_position()
            
            drone_path_x.append(pos[0])
            drone_path_y.append(pos[1])
            
            # Získání mapy ohně z prostředí
            fire_map = env.sim.environment.fire_grid.I
            
            # Vykreslení
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_facecolor('#2b2b2b') # Tmavé pozadí
            
            # Vykreslíme oheň
            extent = [-b, b, -b, b]
            # Maska, aby se nuly (žádný oheň) nevykreslovaly
            fire_map_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
            ax.imshow(fire_map_masked, extent=extent, origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
            
            # Vykreslíme stopu dronu
            ax.plot(drone_path_x, drone_path_y, color='cyan', alpha=0.5, linestyle=':', linewidth=2)
            
            # Vykreslíme aktuální pozici dronu
            ax.scatter(pos[0], pos[1], c='cyan', s=150, marker='^', edgecolors='white', label='Scout')
            
            # Vykreslíme Zorné pole (Kameru) dronu (30x30 metrů)
            # rect = plt.Rectangle((pos[0]-15, pos[1]-15), 30, 30, fill=False, edgecolor='cyan', linestyle='-', alpha=0.8)
            # Vykreslíme Zorné pole (Kameru) dronu podle skutečné logiky z env_core
            current_z = env.sim.drones[agent].get_position()[2]
            win_size = max(10.0, current_z * 1.5)
            rect = plt.Rectangle((pos[0]-win_size/2, pos[1]-win_size/2), win_size, win_size, 
                                fill=False, edgecolor='cyan', linestyle='-', alpha=0.8)
            ax.add_patch(rect)
            
            ax.set_xlim(-b, b)
            ax.set_ylim(-b, b)
            ax.set_title(f"Krok: {step:03d} | Reward: {total_reward:.1f}", color='black', fontsize=14)
            ax.grid(color='white', alpha=0.1)
            
            # Uložení obrázku do paměti a následně do pole frames
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
    # Zrychlíme to na 15 FPS
    imageio.mimsave(output_filename, frames, fps=15, loop=0) 
    print("✅ Video úspěšně uloženo!")

    # === GENEROVÁNÍ GRAFU ANALÝZY ===
    steps_range = range(len(inputs_log))
    plt.figure(figsize=(12, 18)) # Vyšší graf pro 5 částí

    # 1. Graf Vstupů (Actions)
    plt.subplot(5, 1, 1)
    inputs_np = np.array(inputs_log)
    labels = ['Roll', 'Pitch', 'Yaw', 'Throttle']
    for i in range(4):
        plt.plot(steps_range, inputs_np[:, i], label=labels[i])
    plt.title("Vstupy do dronu (Actions - NN Output)")
    plt.ylabel("Intenzita [-1, 1]")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 2. Graf Rychlosti
    plt.subplot(5, 1, 2)
    plt.plot(steps_range, velocity_log, color='red', label='Rychlost')
    plt.title("Velikost vektoru pohybu (Aktuální rychlost)")
    plt.ylabel("m/s")
    plt.grid(True, alpha=0.3)

    # 3. Graf Pozice
    plt.subplot(5, 1, 3)
    pos_np = np.array(position_log)
    plt.plot(steps_range, pos_np[:, 0], label='X (North)')
    plt.plot(steps_range, pos_np[:, 1], label='Y (East)')
    plt.plot(steps_range, pos_np[:, 2], label='Z (Altitude)')
    plt.title("Pozice dronu v mapě")
    plt.ylabel("Metry")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 4. Graf Viděného ohně (Senzor)
    plt.subplot(5, 1, 4)
    plt.fill_between(steps_range, fire_seen_log, color='orange', alpha=0.5, label='Intenzita ohně')
    plt.title("Detekce ohně senzorem (Local Map Pixels)")
    plt.ylabel("Suma pixelů")
    plt.grid(True, alpha=0.3)

    # 5. Graf Odměny (Reward per Step)
    plt.subplot(5, 1, 5)
    plt.bar(steps_range, reward_step_log, color='green', alpha=0.6, label='Step Reward')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Odměna v jednotlivých krocích (Step Reward)")
    plt.ylabel("Reward")
    plt.xlabel("Krok epizody")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("demo_analysis_full.png")
    print("📊 Komplexní analýza uložena jako 'demo_analysis_full.png'")

if __name__ == "__main__":
    run_demo()