import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor

def run_animated_demo(scout_path="saved_models/scout_best.pt", commander_path="saved_models/commander_best.pt"):
    print(f"🚀 Generuji animované demo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Nastavení prostředí
    grid_size = 2000.0
    env = DroneFireEnv(num_quads=1, num_fixed=1, grid_size_m=grid_size, max_steps=400)
    
    # Načtení modelů (vycházíme z tvých nových dimenzí 12 a 19)
    scout_actor = ScoutActor(self_state_dim=12, msg_dim=5, hidden_dim=128).to(device)
    commander_actor = CommanderActor(self_state_dim=19, msg_input_dim=5).to(device)
    
    scout_actor.load_state_dict(torch.load(scout_path, map_location=device))
    commander_actor.load_state_dict(torch.load(commander_path, map_location=device))
    scout_actor.eval()
    commander_actor.eval()

    obs, _ = env.reset()
    
    # Hidden states pro GRU
    scout_hiddens = {q: torch.zeros(1, 1, 128).to(device) for q in env.quad_agents}
    commander_hiddens = {f: torch.zeros(1, 1, 128).to(device) for f in env.fixed_agents}
    
    frames_data = [] # Zde budeme ukládat data pro každý snímek
    
    print("🛸 Simulace běží a sbírá snímky...")
    for step in range(400):
        if not env.agents: break
            
        actions = {}
        scout_msgs = []
        scout_mask = []

        # 1. SCOUTI
        for q in env.quad_agents:
            if q in env.agents:
                l_map = torch.FloatTensor(obs[q]["local_map"]).to(device).unsqueeze(0)
                s_state = torch.FloatTensor(obs[q]["self_state"]).to(device).unsqueeze(0)
                n_state = torch.FloatTensor(obs[q]["neighbor_states"]).to(device).unsqueeze(0)
                n_mask = torch.BoolTensor(obs[q]["neighbor_mask"]).to(device).unsqueeze(0)
                
                with torch.no_grad():
                    dist, msg, h_out = scout_actor(l_map, s_state, n_state, n_mask, scout_hiddens[q])
                    scout_hiddens[q] = h_out
                    actions[q] = dist.loc.squeeze(0).cpu().numpy()
                    scout_msgs.append(msg)
                    scout_mask.append(False)
            else:
                scout_msgs.append(torch.zeros(1, 5).to(device))
                scout_mask.append(True)

        msgs_tensor = torch.stack(scout_msgs, dim=1)
        msgs_mask = torch.tensor(scout_mask).unsqueeze(0).to(device)

        # 2. COMMANDER
        for f in env.fixed_agents:
            if f in env.agents:
                s_state = torch.FloatTensor(obs[f]["self_state"]).to(device).unsqueeze(0)
                with torch.no_grad():
                    dist, _, h_out = commander_actor(s_state, msgs_tensor, msgs_mask, commander_hiddens[f])
                    commander_hiddens[f] = h_out
                    actions[f] = dist.loc.squeeze(0).cpu().numpy()

        # Uložení stavu pro vizualizaci před krokem (aby oheň seděl s pozicí)
        frame_info = {
            'fire': env.sim.environment.fire_grid.I.copy(),
            'drones': {name: env.sim.drones[name].get_position()[:2] for name in env.agents},
            'base': env.sim.environment.refill_zone['position'][:2] if env.sim.environment.refill_zone else None,
            'step': step
        }
        frames_data.append(frame_info)

        obs, _, _, _, _ = env.step(actions)

    # --- GENEROVÁNÍ GIFU ---
    print(f"🎬 Sestavuji GIF z {len(frames_data)} snímků...")
    fig, ax = plt.subplots(figsize=(8, 8))
    bounds = grid_size / 2.0

    def update(i):
        ax.clear()
        data = frames_data[i]
        
        # Vykreslení ohně (zadní vrstva)
        ax.imshow(data['fire'], extent=[-bounds, bounds, -bounds, bounds], origin='lower', cmap='YlOrRd', alpha=0.8)
        
        # Vykreslení základny
        if data['base'] is not None:
            ax.scatter(data['base'][0], data['base'][1], c='blue', marker='s', s=100, label='Base')
        
        # Vykreslení dronů
        for name, pos in data['drones'].items():
            if "quad" in name:
                ax.scatter(pos[0], pos[1], c='cyan', marker='^', s=80, label='Scout', edgecolors='black')
            else:
                ax.scatter(pos[0], pos[1], c='red', marker='>', s=120, label='Commander', edgecolors='black')
        
        ax.set_xlim(-bounds, bounds)
        ax.set_ylim(-bounds, bounds)
        ax.set_title(f"Krok: {data['step']:03d} | Fire Monitoring")
        ax.grid(True, alpha=0.2)
        if i == 0: ax.legend(loc='upper right')

    ani = FuncAnimation(fig, update, frames=len(frames_data), interval=50)
    ani.save("mission_demo.gif", writer='pillow')
    print("✅ mission_demo.gif byl uložen!")

if __name__ == "__main__":
    run_animated_demo()