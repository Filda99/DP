import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Vynutí vykreslování do souboru (ideální pro WSL/servery)
import matplotlib.pyplot as plt

from env_core import DroneFireEnv

def run_random_test():
    print("🚀 Inicializuji testovací prostředí se 2 kvadrokoptérami (bez GUI)...")
    env = DroneFireEnv(num_quads=2, grid_size_m=200.0)
    
    obs, info = env.reset()
    print(f"✅ Reset hotov. Žijící agenti: {env.agents}")
    
    # 1. Příprava slovníku pro sběr dat
    history = {
        agent: {"x": [], "y": [], "z": [], "reward": []}
        for agent in env.possible_agents
    }
    
    for step in range(500):
        if not env.agents:
            print(f"🛑 Všichni agenti jsou vyřazeni (nabourali nebo uletěli). Konec v kroku {step}.")
            break
            
        # Generování náhodných akcí
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Krok v prostředí
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 2. Uložení dat pro grafy
        for agent, reward in rewards.items():
            # Z pozorování (self_state) vytáhneme pozici: index 0=x, 1=y, 2=z
            state = observations[agent]["self_state"]
            history[agent]["x"].append(state[0])
            history[agent]["y"].append(state[1])
            history[agent]["z"].append(state[2])
            history[agent]["reward"].append(reward)
            
        formatted_rewards = {k: round(v, 2) for k, v in rewards.items()}
        print(f"Krok {step:03d} | Agenti: {len(env.agents)} | Odměny: {formatted_rewards}")
        
    print("✅ Smyčka doběhla. Generuji grafy...")
    generate_plots(history)

def generate_plots(history):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['b', 'r', 'g', 'm']
    
    # Graf 1: Trajektorie (Pohled shora)
    ax1.set_title("Trajektorie dronů (Pohled shora)")
    ax1.set_xlabel("X (metry)")
    ax1.set_ylabel("Y (metry)")
    ax1.set_xlim(-100, 100)  # Podle naší map_bounds
    ax1.set_ylim(-100, 100)
    ax1.grid(True)
    
    for i, (agent, data) in enumerate(history.items()):
        if data["x"]:
            ax1.plot(data["x"], data["y"], label=agent, color=colors[i%len(colors)], alpha=0.7)
            ax1.plot(data["x"][0], data["y"][0], 'k*', markersize=10)  # Hvězdička pro start
            ax1.plot(data["x"][-1], data["y"][-1], 'kx', markersize=8) # Křížek pro konec
    ax1.legend()
    
    # Graf 2: Výška v čase
    ax2.set_title("Výška dronů v čase")
    ax2.set_xlabel("Krok")
    ax2.set_ylabel("Z (metry) - Výška")
    ax2.grid(True)
    
    for i, (agent, data) in enumerate(history.items()):
        if data["z"]:
            ax2.plot(data["z"], label=agent, color=colors[i%len(colors)])
    ax2.axhline(0.5, color='black', linestyle='--', label='Země (Crash limit)')
    ax2.legend()
    
    # Graf 3: Odměny v čase
    ax3.set_title("Odměny v čase")
    ax3.set_xlabel("Krok")
    ax3.set_ylabel("Odměna")
    ax3.grid(True)
    
    for i, (agent, data) in enumerate(history.items()):
        if data["reward"]:
            ax3.plot(data["reward"], label=agent, color=colors[i%len(colors)])
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("test_random_agents.png", dpi=150)
    print("📊 Grafy byly uloženy do souboru 'test_random_agents.png'!")

if __name__ == "__main__":
    run_random_test()