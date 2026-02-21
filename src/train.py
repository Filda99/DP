import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importy
from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor, MAPPOCritic

def train():
    print("🚀 Spouštím Heterogenní MAPPO Trénink (Communication Link)...")
    
    # 1. Hyperparametry
    num_episodes = 2000
    max_steps = 400          # Prodlouženo, aby letadlo stihlo doletět
    learning_rate = 3e-4
    gamma = 0.99
    clip_coef = 0.2
    update_epochs = 4
    episodes_per_batch = 5
    
    # Konfigurace týmu
    N_QUADS = 1
    N_FIXED = 0
    
    # 2. Inicializace prostředí
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=200.0)
    
    # Zjištění dimenzí pro sítě
    # A) SCOUT (Quad)
    obs_q = env.observation_space("quad_0")
    scout_self_dim = obs_q["self_state"].shape[0]     # 10
    scout_msg_dim = 5                                 # Velikost zprávy (vymyšleno námi)
    scout_hidden_dim = 128                            # Velikost paměti GRU
    
    # B) COMMANDER (Fixed)
    if N_FIXED > 0:
        obs_f = env.observation_space("fixed_0")
        fixed_self_dim = obs_f["self_state"].shape[0] # 11
    else:
        fixed_self_dim = 0
    
    # C) GLOBAL STATE (Critic)
    global_state_dim = env.state_space.shape[0]

    # 3. Inicializace Sítí
    # Scout má paměť (GRU), Commander zatím ne
    scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim)
    commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=scout_msg_dim)
    critic = MAPPOCritic(global_state_dim)
    
    # Společný optimalizátor pro celý "mozkový trust"
    optimizer = optim.Adam(
        list(scout_actor.parameters()) + 
        list(commander_actor.parameters()) + 
        list(critic.parameters()), 
        lr=learning_rate
    )
    
    episode_rewards_history = []
    os.makedirs("saved_models", exist_ok=True)
    
    # Globální buffery
    batch_data = {
        "maps": [], "self_states": [], "neighbor_states": [], "neighbor_masks": [], # Scout Inputs
        "fixed_states": [], "incoming_msgs": [], "msg_masks": [],                   # Commander Inputs
        "g_states": [], "actions": [], "logprobs": [], "returns": [], "values": [], # Common
        "agent_types": [], "hiddens": []                                            # Meta
    }

    # --- HLAVNÍ SMYČKA ---
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        
        # Inicializace paměti pro Scouty (Batch=1, Hidden=128)
        # Každý dron má svou vlastní paměť
        scout_hiddens = {q: torch.zeros(1, 1, scout_hidden_dim) for q in env.quad_agents}

        # Epizodní buffer
        ep_data = {agent: {"rewards": [], "values": []} for agent in env.possible_agents}
        # Dočasné úložiště pro rollout data (protože je musíme spárovat s odměnami)
        rollout_memory = [] 

        episode_reward = 0.0
        
        for step in range(max_steps):
            if not env.agents: break
            
            global_state = env.state()
            g_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
            
            actions = {}
            current_step_data = {} # Data pro tento krok pro všechny agenty
            
            # === FÁZE 1: SCOUTI (Senzory & Vysílání) ===
            # Musíme je projet první, abychom získali zprávy pro letadlo
            
            scout_messages = []   # Seznam [1, 5] tensorů
            scout_alive_mask = [] # True = Mrtvý
            
            # Projdeme fixně podle seznamu quad_agents, aby pořadí zpráv sedělo
            for q_agent in env.quad_agents:
                if q_agent in env.agents: # Dron žije
                    # Příprava dat
                    local_map = torch.FloatTensor(obs[q_agent]["local_map"]).unsqueeze(0)
                    self_state = torch.FloatTensor(obs[q_agent]["self_state"]).unsqueeze(0)
                    neigh_s = torch.FloatTensor(obs[q_agent]["neighbor_states"]).unsqueeze(0)
                    neigh_m = torch.BoolTensor(obs[q_agent]["neighbor_mask"]).unsqueeze(0)
                    hidden_in = scout_hiddens[q_agent]
                    
                    # Akce sítě
                    with torch.no_grad():
                        dist, message, hidden_out = scout_actor(local_map, self_state, neigh_s, neigh_m, hidden_in)
                        value = critic(g_state_tensor)
                        action = dist.sample()
                        logprob = dist.log_prob(action).sum(1)

                    # Uložení výstupů
                    scout_hiddens[q_agent] = hidden_out # Update paměti
                    actions[q_agent] = action.squeeze(0).numpy()
                    
                    # Zpráva do éteru (pro Commandera)
                    scout_messages.append(message)
                    scout_alive_mask.append(False) # Žije
                    
                    # Uložení do paměti pro trénink
                    current_step_data[q_agent] = {
                        "type": "scout",
                        "map": local_map, "self": self_state, "neigh_s": neigh_s, "neigh_m": neigh_m, "hidden": hidden_in,
                        "action": action, "logprob": logprob, "value": value, "g_state": g_state_tensor
                    }
                else:
                    # Dron je mrtvý -> Posílá prázdnou zprávu a je maskován
                    scout_messages.append(torch.zeros(1, scout_msg_dim))
                    scout_alive_mask.append(True) # Mrtvý (Ignorovat v Attention)

            # Sestavení komunikačního balíčku pro Commandera
            # Shape: [1, N_Scouts, Msg_Dim]
            if scout_messages:
                msgs_tensor = torch.stack(scout_messages, dim=1) 
                msgs_mask = torch.tensor(scout_alive_mask).unsqueeze(0) # [1, N_Scouts]
            else:
                # Fallback kdyby nebyly drony (nemělo by nastat)
                msgs_tensor = torch.zeros(1, N_QUADS, scout_msg_dim)
                msgs_mask = torch.ones(1, N_QUADS, dtype=torch.bool)

            # === FÁZE 2: COMMANDER (Rozhodování) ===
            for f_agent in env.fixed_agents:
                if f_agent in env.agents:
                    self_state = torch.FloatTensor(obs[f_agent]["self_state"]).unsqueeze(0)
                    
                    # Akce sítě (Dostává zprávy od scoutů!)
                    with torch.no_grad():
                        dist, _, _ = commander_actor(self_state, msgs_tensor, msgs_mask)
                        value = critic(g_state_tensor)
                        action = dist.sample()
                        logprob = dist.log_prob(action).sum(1)
                    
                    actions[f_agent] = action.squeeze(0).numpy()
                    
                    current_step_data[f_agent] = {
                        "type": "commander",
                        "self": self_state, "msgs": msgs_tensor, "msg_mask": msgs_mask,
                        "action": action, "logprob": logprob, "value": value, "g_state": g_state_tensor
                    }

            # === KROK PROSTŘEDÍ ===
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Uložení odměn
            for agent, data in current_step_data.items():
                rew = rewards.get(agent, 0.0)
                ep_data[agent]["rewards"].append(rew)
                ep_data[agent]["values"].append(data["value"])
                episode_reward += rew
                
                # Přidáme data do listu pro pozdější zpracování (až budeme znát Returns)
                rollout_memory.append({
                    "agent": agent,
                    "data": data,
                    "step_idx": len(ep_data[agent]["rewards"]) - 1
                })
            
            obs = next_obs

        episode_rewards_history.append(episode_reward)

        # == ZPRACOVÁNÍ EPIZODY (Discounted Returns) ==
        for agent in ep_data:
            rewards = ep_data[agent]["rewards"]
            if not rewards: continue
            
            discounted_sum = 0
            returns = []
            for r in reversed(rewards):
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            ep_data[agent]["returns"] = returns

        # Přesun dat z Rollout Memory do Global Batch
        for item in rollout_memory:
            agent = item["agent"]
            idx = item["step_idx"]
            d = item["data"]
            
            # Společná data
            batch_data["actions"].append(d["action"])
            batch_data["logprobs"].append(d["logprob"])
            batch_data["values"].append(d["value"])
            batch_data["g_states"].append(d["g_state"])
            batch_data["returns"].append(ep_data[agent]["returns"][idx])
            
            if d["type"] == "scout":
                batch_data["agent_types"].append(0) # 0 = Scout
                batch_data["maps"].append(d["map"])
                batch_data["self_states"].append(d["self"])
                batch_data["neighbor_states"].append(d["neigh_s"])
                batch_data["neighbor_masks"].append(d["neigh_m"])
                batch_data["hiddens"].append(d["hidden"])
                # Commander placeholders
                batch_data["fixed_states"].append(torch.zeros(1, fixed_self_dim)) 
                batch_data["incoming_msgs"].append(torch.zeros(1, N_QUADS, scout_msg_dim))
                batch_data["msg_masks"].append(torch.ones(1, N_QUADS, dtype=torch.bool))

            else:
                batch_data["agent_types"].append(1) # 1 = Commander
                batch_data["fixed_states"].append(d["self"])
                batch_data["incoming_msgs"].append(d["msgs"])
                batch_data["msg_masks"].append(d["msg_mask"])
                # Scout placeholders
                batch_data["maps"].append(torch.zeros(1, 1, 32, 32))
                batch_data["self_states"].append(torch.zeros(1, scout_self_dim))
                batch_data["neighbor_states"].append(torch.zeros(1, N_QUADS-1, 3))
                batch_data["neighbor_masks"].append(torch.ones(1, N_QUADS-1, dtype=torch.bool))
                batch_data["hiddens"].append(torch.zeros(1, 1, scout_hidden_dim))

        print(f"Epizoda {episode:03d} | Reward: {episode_reward:.2f}")

        # == PPO UPDATE ==
        if episode % episodes_per_batch == 0 and len(batch_data["returns"]) > 0:
            print(f"🛠️ UPDATE SÍTÍ ({len(batch_data['returns'])} vzorků)...")
            
            # Stackování dat
            b_actions = torch.cat(batch_data["actions"])
            b_logprobs = torch.cat(batch_data["logprobs"])
            b_returns = torch.tensor(batch_data["returns"], dtype=torch.float32).unsqueeze(1)
            b_values = torch.cat(batch_data["values"])
            b_g_states = torch.cat(batch_data["g_states"])
            b_types = torch.tensor(batch_data["agent_types"])
            
            # Advantages
            advantages = b_returns - b_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(update_epochs):
                new_logprobs = torch.zeros_like(b_logprobs)
                entropy_sum = 0
                
                # 1. SCOUT UPDATE
                idx_s = (b_types == 0)
                if idx_s.any():
                    # Forward pass s původními hidden states (Truncated BPTT window=1)
                    dist, _, _ = scout_actor(
                        torch.cat([batch_data["maps"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        torch.cat([batch_data["self_states"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        torch.cat([batch_data["neighbor_states"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        torch.cat([batch_data["neighbor_masks"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        # Funkce torch.cat spojila paměti všech dronů pod sebe a vytvořila tvar (Počet_vzorků, 1, 128).
                        # Ale GRU vrstva (která je uvnitř scout_actor) očekává tvar (1, Počet_vzorků, 128).
                        # Prostě to máme otočené o 90 stupňů.
                        torch.cat([batch_data["hiddens"][i] for i in range(len(b_types)) if b_types[i]==0]).transpose(0, 1)
                    )
                    actions_s = b_actions[idx_s]
                    new_logprobs[idx_s] = dist.log_prob(actions_s).sum(1)
                    entropy_sum += dist.entropy().sum(1).mean()

                # 2. COMMANDER UPDATE
                idx_c = (b_types == 1)
                if idx_c.any():
                    dist, _, _ = commander_actor(
                        torch.cat([batch_data["fixed_states"][i] for i in range(len(b_types)) if b_types[i]==1]),
                        torch.cat([batch_data["incoming_msgs"][i] for i in range(len(b_types)) if b_types[i]==1]),
                        torch.cat([batch_data["msg_masks"][i] for i in range(len(b_types)) if b_types[i]==1])
                    )
                    actions_c = b_actions[idx_c]
                    new_logprobs[idx_c] = dist.log_prob(actions_c).sum(1)
                    entropy_sum += dist.entropy().sum(1).mean()

                # Společný Kritik
                new_values = critic(b_g_states)
                
                # Loss
                ratio = torch.exp(new_logprobs - b_logprobs)
                pg_loss1 = -advantages.squeeze() * ratio
                pg_loss2 = -advantages.squeeze() * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                value_loss = nn.MSELoss()(new_values, b_returns)
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_sum
                
                optimizer.zero_grad()
                loss.backward()
                # === GRADIENT CLIPPING (Prevence explozí) ===
                nn.utils.clip_grad_norm_(list(scout_actor.parameters()) + 
                                         list(commander_actor.parameters()) + 
                                         list(critic.parameters()), max_norm=0.5)
                optimizer.step()
            
            # Reset bufferů
            for k in batch_data: batch_data[k] = []

        if episode % 50 == 0:
            torch.save(scout_actor.state_dict(), f"saved_models/scout_ep{episode}.pt")
            torch.save(commander_actor.state_dict(), f"saved_models/commander_ep{episode}.pt")
            
            # Graf
            plt.figure(figsize=(10, 5))
            smoothed = [np.mean(episode_rewards_history[max(0, i-10):i+1]) for i in range(len(episode_rewards_history))]
            plt.plot(episode_rewards_history, alpha=0.3)
            plt.plot(smoothed, label="Trend")
            plt.title("Heterogenní Tým (Communication Enabled)")
            plt.savefig("training_comm.png")
            plt.close()

if __name__ == "__main__":
    train()