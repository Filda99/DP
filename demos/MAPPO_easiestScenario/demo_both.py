import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image
import os, sys
import cv2

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Importy z tvého projektu
from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor

def run_demo_both():
    print("🎬 Spouštím Komplexní Demo: Scout & Commander v akci")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Konfigurace
    N_QUADS = 1
    N_FIXED = 1
    MAX_STEPS = 1500
    GRID_SIZE = 2000.0  # Nastaveno na tvou novou velikost mapy
    
    # Cesty k modelům
    MODEL_PATH_QUAD = "saved_models/scout_ep3980.pt" 
    MODEL_PATH_FIXED = "saved_models/commander_ep3980.pt" 
    
    # 2. Inicializace prostředí
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    
    # Dimenze (podle env_core.py)
    scout_self_dim = 12
    fixed_self_dim = 19  # Včetně tvého Danger Flag
    msg_dim = 5
    hidden_dim = 128
    
    # 3. Načtení sítí
    scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=msg_dim, hidden_dim=hidden_dim).to(device)
    commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=msg_dim).to(device)
    
    if os.path.exists(MODEL_PATH_QUAD) and os.path.exists(MODEL_PATH_FIXED):
        scout_actor.load_state_dict(torch.load(MODEL_PATH_QUAD, map_location=device))
        commander_actor.load_state_dict(torch.load(MODEL_PATH_FIXED, map_location=device))
        print(f"✅ Oba modely úspěšně načteny!")
    else:
        print(f"❌ CHYBA: Některý z modelů chybí!")
        return
        
    scout_actor.eval()
    commander_actor.eval()

    # 4. Spuštění Epizody
    obs, _ = env.reset(seed=42) 
    
    # Inicializace pamětí pro GRU
    h_scout = torch.zeros(1, 1, hidden_dim).to(device)
    h_commander = torch.zeros(1, 1, hidden_dim).to(device)
    
    frames = []
    
    # Logování pro analýzu
    history = {
        "quad": {"path_x": [], "path_y": [], "reward": [], "vel": [], "actions": []},
        "fixed": {"path_x": [], "path_y": [], "reward": [], "vel": [], "actions": []}
    }

    print("🚀 Mise začíná...")
    
    for step in range(MAX_STEPS):
        if not env.agents: 
            print("💥 Tým byl vyřazen!")
            break
            
        actions_to_env = {}
        
        # --- FÁZE 1: SCOUT (Generování pohybu a zprávy) ---
        q_id = "quad_0"
        if q_id in env.agents:
            l_map = torch.FloatTensor(obs[q_id]["local_map"]).to(device).unsqueeze(0)
            s_state = torch.FloatTensor(obs[q_id]["self_state"]).to(device).unsqueeze(0)
            n_state = torch.FloatTensor(obs[q_id]["neighbor_states"]).to(device).unsqueeze(0)
            n_mask = torch.BoolTensor(obs[q_id]["neighbor_mask"]).to(device).unsqueeze(0)
            
            with torch.no_grad():
                dist_q, message_q, h_scout = scout_actor(l_map, s_state, n_state, n_mask, h_scout)
                action_q = dist_q.mean # Čistá akce
            
            actions_to_env[q_id] = action_q.squeeze(0).cpu().numpy()
            
            # Příprava zprávy pro Commandera
            current_msgs = message_q.unsqueeze(1) # [1, 1, 5]
            msg_mask = torch.BoolTensor([[False]]).to(device) # Scout žije
        else:
            current_msgs = torch.zeros(1, 1, 5).to(device)
            msg_mask = torch.BoolTensor([[True]]).to(device)

        # --- FÁZE 2: COMMANDER (Rozhodování na základě zpráv) ---
        f_id = "fixed_0"
        if f_id in env.agents:
            s_state_f = torch.FloatTensor(obs[f_id]["self_state"]).to(device).unsqueeze(0)
            
            with torch.no_grad():
                dist_f, _, h_commander = commander_actor(s_state_f, current_msgs, msg_mask, h_commander)
                action_f = dist_f.mean
            
            actions_to_env[f_id] = action_f.squeeze(0).cpu().numpy()

        # Krok prostředí
        obs, rewards, terminations, truncations, infos = env.step(actions_to_env)

        # --- LOGOVÁNÍ DAT ---
        for a_type, a_id in [("quad", q_id), ("fixed", f_id)]:
            if a_id in env.sim.drones:
                drone = env.sim.drones[a_id]
                pos = drone.get_position()
                history[a_type]["path_x"].append(pos[0])
                history[a_type]["path_y"].append(pos[1])
                history[a_type]["vel"].append(np.linalg.norm(drone.get_velocity()))
                history[a_type]["reward"].append(rewards.get(a_id, 0.0))
                history[a_type]["actions"].append(actions_to_env[a_id])

        # --- VYKRESLOVÁNÍ DO GIFU (Každý 5. krok pro úsporu RAM) ---
        if step % 5 == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_facecolor('#1a1a1a')
            
            # Vykreslení zmenšeného ohně pro stabilitu RAM
            fire_map = cv2.resize(env.sim.environment.fire_grid.I, (200, 200), interpolation=cv2.INTER_AREA)
            extent = [-GRID_SIZE/2, GRID_SIZE/2, -GRID_SIZE/2, GRID_SIZE/2]
            ax.imshow(fire_map, extent=extent, origin='lower', cmap='YlOrRd', alpha=0.9)
            
            # Vykreslení pozic
            if q_id in env.sim.drones:
                ax.scatter(history["quad"]["path_x"][-1], history["quad"]["path_y"][-1], c='cyan', s=100, marker='^', label='Scout')
            if f_id in env.sim.drones:
                ax.scatter(history["fixed"]["path_x"][-1], history["fixed"]["path_y"][-1], c='red', s=150, marker='>', label='Commander')
            
            ax.set_xlim(-GRID_SIZE/2, GRID_SIZE/2)
            ax.set_ylim(-GRID_SIZE/2, GRID_SIZE/2)
            ax.set_title(f"Mise: Krok {step} | Společný zásah")
            ax.legend(loc='upper right')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            frames.append(np.array(Image.open(buf)))
            plt.close(fig)

    # 5. ULOŽENÍ VÝSLEDKŮ
    imageio.mimsave("team_mission.gif", frames, fps=10, loop=0)
    generate_analysis_plot(history)
    print("✅ Demo dokončeno. Soubory 'team_mission.gif' a 'team_analysis.png' jsou připraveny.")

def generate_analysis_plot(history):
    plt.figure(figsize=(15, 12))
    
    # 1. Odměny
    plt.subplot(3, 1, 1)
    plt.plot(np.cumsum(history["quad"]["reward"]), label="Scout (Quad)", color='cyan')
    plt.plot(np.cumsum(history["fixed"]["reward"]), label="Commander (Fixed)", color='red')
    plt.title("Kumulativní odměna týmu")
    plt.legend(); plt.grid(True, alpha=0.3)

    # 2. Rychlosti
    plt.subplot(3, 1, 2)
    plt.plot(history["quad"]["vel"], color='cyan', alpha=0.8, label="Scout Speed")
    plt.plot(history["fixed"]["vel"], color='red', alpha=0.8, label="Commander Speed")
    plt.title("Rychlost agentů (m/s)")
    plt.legend(); plt.grid(True, alpha=0.3)

    # 3. Výška (Z osa)
    plt.subplot(3, 1, 3)
    # Tady bys mohl přidat i graf výšky, pokud bys ji logoval do history
    plt.plot(history["quad"]["path_x"], label="Scout X", color='cyan', linestyle='--')
    plt.plot(history["fixed"]["path_x"], label="Commander X", color='red', linestyle='--')
    plt.title("Pohyb po ose X")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("team_analysis.png")

if __name__ == "__main__":
    run_demo_both()