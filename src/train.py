import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importujeme naše prostředí a sítě
from env_core import DroneFireEnv
from models import MAPPOActor, MAPPOCritic

def train():
    print("🚀 Spouštím MAPPO Trénink (CleanRL styl)...")
    
    # 1. Nastavení hyperparametrů (Dají se měnit pro ladění)
    num_episodes = 2000      # Kolik her celkem odehrajeme (Zvýšeno pro delší trénink)
    max_steps = 300          # Maximální délka jedné hry
    learning_rate = 3e-4     # Rychlost učení
    gamma = 0.99             # Jak moc řešíme budoucí odměny
    clip_coef = 0.2          # PPO oříznutí (zabraňuje zničení mozku velkým updatem)
    update_epochs = 4        # Kolikrát projdeme nasbíraná data při jednom updatu
    episodes_per_batch = 5   # Kolik her odehrajeme před updatem sítě
    
    # 2. Inicializace prostředí a zjištění velikostí vektorů
    env = DroneFireEnv(num_quads=2, grid_size_m=200.0)
    obs_space = env.observation_space("quad_0")
    
    self_state_size = obs_space["self_state"].shape[0]
    global_state_size = env.state_space.shape[0]
    action_dim = env.action_space("quad_0").shape[0]
    
    # 3. Inicializace Sítí a Optimalizátoru
    actor = MAPPOActor(self_state_size, action_dim)
    critic = MAPPOCritic(global_state_size)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate)
    
    # Pro ukládání grafu odměn
    episode_rewards_history = []
    os.makedirs("saved_models", exist_ok=True)
    
    # Globální buffery pro sběr dat z VÍCE epizod najednou
    batch_local_maps, batch_self_states, batch_global_states = [], [], []
    batch_actions, batch_logprobs, batch_returns, batch_values = [], [], [], []
    
    # --- HLAVNÍ TRÉNOVACÍ SMYČKA ---
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        
        # Datové buffery JEN pro aktuální epizodu (kvůli výpočtu budoucnosti odzadu)
        ep_data = {
            agent: {"maps": [], "states": [], "g_states": [], "actions": [], "logprobs": [], "rewards": [], "values": []} 
            for agent in env.possible_agents
        }
        
        episode_reward = 0.0
        
        # == FÁZE 1: SBĚR DAT (ROLLOUT) ==
        for step in range(max_steps):
            if not env.agents:
                break # Všichni nabourali
                
            global_state = env.state()
            actions = {}
            
            # Získáme akci a hodnotu pro každého žijícího agenta
            for agent in env.agents:
                local_map = torch.FloatTensor(obs[agent]["local_map"]).unsqueeze(0)
                self_state = torch.FloatTensor(obs[agent]["self_state"]).unsqueeze(0)
                g_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
                
                with torch.no_grad():
                    action, logprob, _ = actor(local_map, self_state)
                    value = critic(g_state_tensor)
                
                actions[agent] = action.squeeze(0).numpy()
                
                # Uložíme data do dočasného bufferu agenta
                ep_data[agent]["maps"].append(local_map)
                ep_data[agent]["states"].append(self_state)
                ep_data[agent]["g_states"].append(g_state_tensor)
                ep_data[agent]["actions"].append(action)
                ep_data[agent]["logprobs"].append(logprob)
                ep_data[agent]["values"].append(value)
            
            # Krok v prostředí
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Uložíme odměny
            for agent in actions.keys(): # Vezmeme odměny těch, co udělali akci
                ep_data[agent]["rewards"].append(rewards[agent])
                episode_reward += rewards[agent]
                
            obs = next_obs
            
        episode_rewards_history.append(episode_reward)
        
        # == FÁZE 1.5: VÝPOČET BUDOUCNOSTI (DISCOUNTED RETURNS) ==
        # Pro každého agenta spočítáme jeho křivku odměn odzadu dopředu
        for agent, data in ep_data.items():
            if not data["rewards"]: continue
            
            discounted_sum = 0
            agent_returns = []
            
            # Počítáme od konce (poslední krok) směrem k začátku
            for r in reversed(data["rewards"]):
                discounted_sum = r + gamma * discounted_sum
                agent_returns.insert(0, discounted_sum)
                
            # Nasypeme zprocesovaná data tohoto agenta do velkého Globálního Batche
            for i in range(len(data["rewards"])):
                batch_local_maps.append(data["maps"][i])
                batch_self_states.append(data["states"][i])
                batch_global_states.append(data["g_states"][i])
                batch_actions.append(data["actions"][i])
                batch_logprobs.append(data["logprobs"][i])
                batch_returns.append(agent_returns[i])
                batch_values.append(data["values"][i])
        
        # Výpis po každé epizodě
        print(f"Epizoda {episode:03d} | Celková odměna týmu: {episode_reward:.2f}")

        # == FÁZE 2: UČENÍ (PPO UPDATE) ==
        # Pokud se nasbírala data za X epizod, jdeme updatovat sítě
        if episode % episodes_per_batch == 0 and len(batch_returns) > 0:
            print(f"🛠️ UPDATE SÍTĚ na {len(batch_returns)} krocích...")
            
            # Převedeme seznamy na PyTorch Tenzory
            b_local_maps = torch.cat(batch_local_maps)
            b_self_states = torch.cat(batch_self_states)
            b_global_states = torch.cat(batch_global_states)
            b_actions = torch.cat(batch_actions)
            b_logprobs = torch.cat(batch_logprobs)
            b_returns = torch.FloatTensor(batch_returns).unsqueeze(1)
            b_values = torch.cat(batch_values)
            
            # Výpočet výhod (Advantages) - SKUTEČNÁ VÝHODA = To co se stalo (Returns) - to co čekal Kritik
            # V profi kódu se počítá GAE (Generalized Advantage Estimation), zde pro názornost rozdíl
            advantages = b_returns - b_values.detach()
            # Normalizace výhod pomáhá stabilitě učení
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Několikrát projdeme nasbíraná data
            for epoch in range(update_epochs):
                # Znovu proženeme data sítěmi (sledujeme gradienty!)
                _, new_logprobs, entropy = actor(b_local_maps, b_self_states, b_actions)
                new_values = critic(b_global_states)
                
                # 1. Výpočet ztráty Hérce (Policy Loss)
                ratio = torch.exp(new_logprobs - b_logprobs) # Jak moc jsme změnili pravděpodobnost?
                pg_loss1 = -advantages.squeeze() * ratio
                pg_loss2 = -advantages.squeeze() * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 2. Výpočet ztráty Kritika (Value Loss)
                # Kritik se nově učí odhadovat celý b_returns (součet budoucích odměn)
                value_loss = nn.MSELoss()(new_values, b_returns)
                
                # 3. Entropie (nutí drony zkoušet nové věci a nebýt hned absolutně jistí)
                entropy_loss = entropy.mean()
                
                # Celková ztráta (Loss)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                
                # Krok optimalizátoru (Upraví váhy v sítích!)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Vyčištění globálního bufferu pro sběr dat do dalšího updatu
            batch_local_maps, batch_self_states, batch_global_states = [], [], []
            batch_actions, batch_logprobs, batch_returns, batch_values = [], [], [], []
        
        # Uložení modelu každých 50 epizod
        if episode % 50 == 0:
            torch.save(actor.state_dict(), f"saved_models/actor_ep{episode}.pt")
            
    print("✅ Trénink dokončen!")
    
    # Vykreslení grafu učení
    plt.figure(figsize=(10, 5))
    
    # Přidáno smoothování grafu pomocí klouzavého průměru
    smoothed = [np.mean(episode_rewards_history[max(0, i-10):i+1]) for i in range(len(episode_rewards_history))]
    plt.plot(episode_rewards_history, alpha=0.3, label="Hrubá odměna")
    plt.plot(smoothed, label="Trend (10 ep.)")
    
    plt.title("Průběh tréninku (Total Reward)")
    plt.xlabel("Epizoda")
    plt.ylabel("Součet odměn")
    plt.legend()
    plt.grid()
    plt.savefig("training_progress.png")
    print("📊 Graf vývoje odměn uložen jako 'training_progress.png'")

if __name__ == "__main__":
    train()