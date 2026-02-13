import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env_core import DroneFireEnv
from models import MAPPOActor

def run_demo(model_path="saved_models/actor_ep2000.pt"):
    print(f"🚀 Spouštím Demo s modelem: {model_path}")
    
    env = DroneFireEnv(num_quads=2, grid_size_m=200.0)
    
    # Musíme zjistit rozměry pro síť
    obs_space = env.observation_space("quad_0")
    self_state_size = obs_space["self_state"].shape[0]
    action_dim = env.action_space("quad_0").shape[0]
    
    # Načteme mozek
    actor = MAPPOActor(self_state_size, action_dim)
    try:
        actor.load_state_dict(torch.load(model_path))
        print("✅ Mozek úspěšně načten!")
    except FileNotFoundError:
        print(f"❌ Chyba: Soubor {model_path} nenalezen. Nejdříve spusť train.py!")
        return
        
    actor.eval() # Přepneme síť do módu hodnocení (vypne exploraci)

    obs, _ = env.reset()
    
    # Sběr dat pro grafy
    history = {agent: {"x": [], "y": [], "z": [], "reward": []} for agent in env.possible_agents}
    
    for step in range(500):
        if not env.agents:
            break
            
        actions = {}
        for agent in env.agents:
            local_map = torch.FloatTensor(obs[agent]["local_map"]).unsqueeze(0)
            self_state = torch.FloatTensor(obs[agent]["self_state"]).unsqueeze(0)
            
            with torch.no_grad():
                # Vezmeme přímo "action_mean" (průměrnou akci) bez přidávání náhodného šumu
                cnn_feat = actor.cnn(local_map)
                mlp_feat = actor.mlp(self_state)
                action = actor.action_mean(torch.cat([cnn_feat, mlp_feat], dim=1))
                
            actions[agent] = action.squeeze(0).numpy()
            
        # Fyzikální krok
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Zápis do historie
        for agent in env.agents:
            state = obs[agent]["self_state"]
            history[agent]["x"].append(state[0])
            history[agent]["y"].append(state[1])
            history[agent]["z"].append(state[2])
            history[agent]["reward"].append(rewards[agent])
            
        obs = next_obs

    print("✅ Demo simulace dokončena. Generuji grafy...")
    generate_plots(history)

def generate_plots(history):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['b', 'r', 'g', 'm']
    
    ax1.set_title("Trajektorie dronů (Pohled shora)")
    ax1.set_xlim(-100, 100); ax1.set_ylim(-100, 100); ax1.grid(True)
    ax2.set_title("Výška dronů v čase"); ax2.grid(True)
    ax2.axhline(0.5, color='black', linestyle='--', label='Země (Crash limit)')
    ax3.set_title("Odměny v čase"); ax3.grid(True)
    
    for i, (agent, data) in enumerate(history.items()):
        if data["x"]:
            ax1.plot(data["x"], data["y"], label=agent, color=colors[i%len(colors)])
            ax1.plot(data["x"][0], data["y"][0], 'k*', markersize=10)
            ax2.plot(data["z"], label=agent, color=colors[i%len(colors)])
            ax3.plot(data["reward"], label=agent, color=colors[i%len(colors)])
            
    ax1.legend(); ax2.legend(); ax3.legend()
    plt.tight_layout()
    plt.savefig("demo_trained_agents.png", dpi=150)
    print("📊 Grafy uloženy do 'demo_trained_agents.png'")

if __name__ == "__main__":
    run_demo()