import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image
import os, sys
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Importy z tvého projektu
from src.env_core import DroneFireEnv
from src.models import ScoutActor

def run_demo():
    print("🎬 Spouštím Diagnostické Demo: Scout Komunikace")
    
    # 1. Konfigurace (Stejná jako při tréninku)
    N_QUADS = 1
    N_FIXED = 0
    MAX_STEPS = 1000
    
    # !!! ZMĚŇ CESTU K MODELU PODLE POTŘEBY !!!
    MODEL_PATH = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/scout_solo/scout_b0610.pt" 
    
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
        try:
            scout_actor.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            print(f"✅ Model {MODEL_PATH} úspěšně načten!")
        except Exception as e:
            print(f"❌ CHYBA při načítání modelu: {e}")
            return
    else:
        print(f"❌ CHYBA: Model {MODEL_PATH} neexistuje! Zkontroluj název souboru.")
        return
        
    scout_actor.eval()

    # 4. Spuštění Epizody
    # Použijeme epizode_number > 1500, aby oheň byl náhodně (těžší test pro skauta)
    seed = 42 # Pevný seed pro reprodukovatelnost diagnostiky
    obs, _ = env.reset(seed=seed, epizode_number=2000) 
    hidden_state = torch.zeros(1, 1, scout_hidden_dim)
    
    frames = []
    total_reward = 0.0
    
    drone_path_x = []
    drone_path_y = []

    # Seznamy pro logování dat
    velocity_log = [] 
    fire_seen_log = [] 
    reward_step_log = []

    # --- NOVÉ: Logování komunikace ---
    logged_target_protocol = []    # Co by měl vysílat
    logged_broadcast_protocol = [] # Co reálně vysílá
    mse_log = []                   # Chyba v každém kroku

    print(f"🚁 Dron startuje (Seed: {seed}). Oheň je na [X:{env.fire_x:.1f}, Y:{env.fire_y:.1f}]")
    print("-" * 100)
    print(f"{'Krok':<5} | {'Skutečná GPS (Norm)':<20} | {'Vysílaná zpráva':<20} | {'MSE Chyba':<10}")
    print("-" * 100)
    
    agent = "quad_0"
    mse_fn = nn.MSELoss()

    for step in range(MAX_STEPS):
        if agent not in env.agents: 
            print("💥 Dron havaroval nebo uletěl!")
            break
            
        # Příprava dat pro síť
        local_map_obs = obs[agent]["local_map"]
        self_state_obs = obs[agent]["self_state"]
        
        local_map_t = torch.FloatTensor(local_map_obs).unsqueeze(0)
        self_state_t = torch.FloatTensor(self_state_obs).unsqueeze(0)
        
        # Ostatní agenty v tomto demu ignorujeme
        neigh_s = torch.zeros(1, env.max_neighbors, 3)
        neigh_m = torch.ones(1, env.max_neighbors, dtype=torch.bool)
        
        # Inference (BEZ gradientů)
        with torch.no_grad():
            dist, message, hidden_state = scout_actor(local_map_t, self_state_t, neigh_s, neigh_m, hidden_state)
            action = dist.mean # Deterministické demo
            
        actions = {agent: action.squeeze(0).numpy()}

        # =========================================================================
        # === KOMUNIKAČNÍ DIAGNOSTIKA (Jádro pudla) ===
        # =========================================================================
        
        # 1. Vytvoříme TARGET (Cíl): Co by síť MĚLA vysílat podle nových pravidel v train.py
        # Očekáváme: [norm_pos_x, norm_pos_y, dyn_intensity]
        # Vycházíme z indexů definovaných v tvém train.py snippetu:
        target_norm_x = self_state_obs[0]     # Skautovo X (normované [-1, 1])
        target_norm_y = self_state_obs[1]     # Skautovo Y (normované [-1, 1])
        target_dyn_int = self_state_obs[14]   # Intenzita ohně pod ním (normované [0, 1])
        
        target_protocol = np.array([target_norm_x, target_norm_y, target_dyn_int], dtype=np.float32)
        logged_target_protocol.append(target_protocol)

        # 2. Vytvoříme BROADCAST (Realita): Co síť REÁLNĚ vysílá (první 3 složky zprávy)
        # msg_head má Tanh(), takže hodnoty jsou v [-1, 1].
        broadcast_msg = message.squeeze(0).cpu().numpy()[:3]
        logged_broadcast_protocol.append(broadcast_msg)

        # 3. Vypočítáme chybu (MSE)
        # Použijeme torch MSE loss pro konzistenci s tréninkem
        target_t = torch.from_numpy(target_protocol)
        broad_t = torch.from_numpy(broadcast_msg)
        step_mse = mse_fn(broad_t, target_t).item()
        mse_log.append(step_mse)

        # 4. Diagnostický výpis každých 20 kroků
        if step % 20 == 0:
            gt_str = f"[{target_norm_x:+.2f}, {target_norm_y:+.2f}, {target_dyn_int:.2f}]"
            bc_str = f"[{broadcast_msg[0]:+.2f}, {broadcast_msg[1]:+.2f}, {broadcast_msg[2]:+.2f}]"
            # Alert, pokud je chyba obrovská
            error_alert = "⚠️ ŠPATNĚ!" if step_mse > 0.1 else "✅ OK"
            print(f"{step:04d} | {gt_str:<20} | {bc_str:<20} | {step_mse:.6f} {error_alert}")
        # =========================================================================
        
        # Krok prostředí
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += rewards.get(agent, 0.0)

        # Uložení do historie pro ostatní grafy
        fire_pixels = np.sum(local_map_obs)
        fire_seen_log.append(fire_pixels)
        reward_step_log.append(rewards.get(agent, 0.0))
        
        drone = env.sim.drones[agent]
        drone_pos = drone.get_position()
        velocity_log.append(np.linalg.norm(drone.get_velocity()))
        drone_path_x.append(drone_pos[0])
        drone_path_y.append(drone_pos[1])
        
        # --- VYKRESLOVÁNÍ GIFu (Každý 5. krok pro rychlost) ---
        if step % 5 == 0:
            b = env.map_bounds
            fire_map = env.sim.environment.fire_grid.I
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_facecolor('#2b2b2b') 
            
            extent = [-b, b, -b, b]
            fire_map_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
            ax.imshow(fire_map_masked, extent=extent, origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
            
            ax.plot(drone_path_x, drone_path_y, color='cyan', alpha=0.3, linestyle='-', linewidth=1)
            ax.scatter(drone_pos[0], drone_pos[1], c='cyan', s=100, marker='^', edgecolors='white', zorder=5)
            
            current_z = drone_pos[2]
            win_size = max(10.0, current_z * 1.5)
            rect = plt.Rectangle((drone_pos[0]-win_size/2, drone_pos[1]-win_size/2), win_size, win_size, 
                                fill=False, edgecolor='cyan', linestyle='-', alpha=0.6)
            ax.add_patch(rect)
            
            ax.set_xlim(-b, b)
            ax.set_ylim(-b, b)
            # Přidáme informaci o MSE chybě do titulku
            ax.set_title(f"Krok: {step:03d} | R: {total_reward:.1f} | Comm MSE: {step_mse:.4f}", color='black', fontsize=12)
            ax.grid(color='white', alpha=0.1)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            frames.append(np.array(img))
            plt.close(fig)

    print("-" * 100)
    print(f"🏁 Hotovo! Celkový Reward: {total_reward:.2f} | Průměrná Comm MSE: {np.mean(mse_log):.6f}")
    
    # 5. Uložení jako GIF
    output_filename = "scout_diag.gif"
    print(f"💾 Ukládám GIF ({len(frames)} snímků) → {output_filename} ...")
    imageio.mimsave(output_filename, frames, fps=15, loop=0) 

    # === GENEROVÁNÍ GRAFU KOMUNIKAČNÍ ANALÝZY ===
    print("📊 Generuji graf komunikační analýzy...")
    
    steps_range = range(len(logged_target_protocol))
    target_np = np.array(logged_target_protocol)       # (S, 3)
    broadcast_np = np.array(logged_broadcast_protocol) # (S, 3)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f"Diagnostika Komunikace Scouta (Model: {os.path.basename(MODEL_PATH)})", fontsize=14)

    # 1. Porovnání X pozice (Normované [-1, 1])
    ax = axes[0]
    ax.plot(steps_range, target_np[:, 0],    label='Target (Skutečná Norm X)', color='black', linestyle='--', alpha=0.7)
    ax.plot(steps_range, broadcast_np[:, 0], label='Broadcast (Vysílaná X)',   color='cyan', alpha=0.8)
    ax.set_ylabel("Norm. Pozice X")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("GPS X: Skutečnost vs. Vysílání")

    # 2. Porovnání Y pozice (Normované [-1, 1])
    ax = axes[1]
    ax.plot(steps_range, target_np[:, 1],    label='Target (Skutečná Norm Y)', color='black', linestyle='--', alpha=0.7)
    ax.plot(steps_range, broadcast_np[:, 1], label='Broadcast (Vysílaná Y)',   color='magenta', alpha=0.8)
    ax.set_ylabel("Norm. Pozice Y")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("GPS Y: Skutečnost vs. Vysílání")

    # 3. Porovnání Intenzity (Normované [0, 1])
    # Tanh() zpráva je [-1, 1], cíl intenzity je [0, 1]. PPO by se měl naučit používat jen kladnou část Tanh().
    ax = axes[2]
    ax.plot(steps_range, target_np[:, 2],    label='Target (Skutečná Intenzita)', color='black', linestyle='--', alpha=0.7)
    ax.plot(steps_range, broadcast_np[:, 2], label='Broadcast (Vysílaná Intenzita)', color='orange', alpha=0.8)
    ax.set_ylabel("Norm. Intenzita")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Intenzita Ohně: Skutečnost vs. Vysílání")

    # 4. Historie MSE Chyby
    ax = axes[3]
    ax.plot(steps_range, mse_log, color='red', label='MSE Loss per step')
    ax.axhline(np.mean(mse_log), color='red', linestyle=':', label=f'Průměr: {np.mean(mse_log):.4f}')
    ax.set_ylabel("MSE Loss")
    ax.set_xlabel("Krok epizody")
    ax.set_yscale('log') # Logaritmická škála pro detaily
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Celková chyba protokolu (MSE)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    analysis_png = "scout_comm_analysis.png"
    plt.savefig(analysis_png, dpi=120)
    plt.close()
    print(f"📊 Analýza uložena → {analysis_png}")

if __name__ == "__main__":
    run_demo()