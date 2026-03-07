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
from concurrent.futures import ProcessPoolExecutor

# ============================================================================
# WORKER FUNKCE: Běží paralelně na oddělených CPU jádrech
# ============================================================================
def collect_single_episode(scout_w, cmdr_w, critic_w, config):
    """Tohle poběží na jednom vyhrazeném CPU jádře."""
    # import cProfile
    # import os
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # 1. Každý proces si vytvoří VLASTNÍ prostředí na CPU!
    local_env = DroneFireEnv(num_quads=config['N_QUADS'], num_fixed=config['N_FIXED'], 
                             grid_size_m=config['grid_size_m'], max_steps=config['max_steps'])
    
    # 2. Inicializace lokálních sítí na CPU (Inference)
    local_scout = None
    if config['N_QUADS'] > 0:
        local_scout = ScoutActor(self_state_dim=config['scout_self_dim'], msg_dim=config['scout_msg_dim'], hidden_dim=config['scout_hidden_dim'])
        local_scout.load_state_dict(scout_w)
        local_scout.eval()

    local_commander = None
    if config['N_FIXED'] > 0:
        local_commander = CommanderActor(self_state_dim=config['fixed_self_dim'], msg_input_dim=config['scout_msg_dim'])
        local_commander.load_state_dict(cmdr_w)
        local_commander.eval()
        
    local_critic = MAPPOCritic(config['global_state_dim'])
    local_critic.load_state_dict(critic_w)
    local_critic.eval()

    # 3. Příprava proměnných pro epizodu (Přesunuto z tvé původní smyčky)
    obs, _ = local_env.reset()
    
    scout_hiddens = {q: torch.zeros(1, 1, config['scout_hidden_dim']) for q in local_env.quad_agents}
    commander_hiddens = {f: torch.zeros(1, 1, 128) for f in local_env.fixed_agents}
    critic_hiddens = {a: torch.zeros(1, 1, 128) for a in local_env.possible_agents}

    agent_lifespans = {agent: config['max_steps'] for agent in local_env.possible_agents}
    ep_data = {agent: {"rewards": [], "values": []} for agent in local_env.possible_agents}
    rollout_memory = [] 
    episode_reward = 0.0

    # 4. TVOJE PŮVODNÍ SMYČKA KROKŮ
    for step in range(config['max_steps']):
        global_state = local_env.state()
        g_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
        
        actions = {}
        current_step_data = {} 
        
        # === FÁZE 1: SCOUTI (Senzory & Vysílání) ===
        scout_messages = []   
        scout_alive_mask = [] 
        
        for q_agent in local_env.quad_agents:
            if q_agent in local_env.agents: # Dron žije
                local_map = torch.FloatTensor(obs[q_agent]["local_map"]).unsqueeze(0)
                self_state = torch.FloatTensor(obs[q_agent]["self_state"]).unsqueeze(0)
                neigh_s = torch.FloatTensor(obs[q_agent]["neighbor_states"]).unsqueeze(0)
                neigh_m = torch.BoolTensor(obs[q_agent]["neighbor_mask"]).unsqueeze(0)
                hidden_in = scout_hiddens[q_agent]
                
                with torch.no_grad():
                    dist, message, hidden_out = local_scout(local_map, self_state, neigh_s, neigh_m, hidden_in)
                    value, c_hidden_out = local_critic(g_state_tensor, critic_hiddens[q_agent])
                    action = dist.sample()
                    logprob = dist.log_prob(action).sum(1)

                scout_hiddens[q_agent] = hidden_out 
                critic_hiddens[q_agent] = c_hidden_out  
                actions[q_agent] = action.squeeze(0).numpy()

                scout_messages.append(message)
                scout_alive_mask.append(False) 
                
                current_step_data[q_agent] = {
                    "type": "scout",
                    "map": local_map, "self": self_state, "neigh_s": neigh_s, "neigh_m": neigh_m, "hidden": hidden_in,
                    "action": action, "logprob": logprob, "value": value, "g_state": g_state_tensor, "c_hidden": critic_hiddens[q_agent]
                }

            else:
                scout_messages.append(torch.zeros(1, config['scout_msg_dim']))
                scout_alive_mask.append(True) 

                current_step_data[q_agent] = {
                    "type": "scout",
                    "map": torch.zeros(1, 1, 32, 32), "self": torch.zeros(1, config['scout_self_dim']),
                    "neigh_s": torch.zeros(1, local_env.max_neighbors, 3), "neigh_m": torch.ones(1, local_env.max_neighbors, dtype=torch.bool),
                    "hidden": torch.zeros(1, 1, config['scout_hidden_dim']), "action": torch.zeros(1, 4),
                    "logprob": torch.tensor([0.0]), "value": torch.tensor([[0.0]]), "g_state": g_state_tensor, "c_hidden": torch.zeros(1, 1, 128)
                }
                actions[q_agent] = np.zeros(4) 

        if scout_messages:
            msgs_tensor = torch.stack(scout_messages, dim=1) 
            msgs_mask = torch.tensor(scout_alive_mask).unsqueeze(0) 
        else:
            msgs_tensor = torch.zeros(1, config['N_QUADS'] if config['N_QUADS']>0 else 1, config['scout_msg_dim'])
            msgs_mask = torch.ones(1, config['N_QUADS'] if config['N_QUADS']>0 else 1, dtype=torch.bool)

        # === FÁZE 2: COMMANDER (Rozhodování) ===
        for f_agent in local_env.fixed_agents:
            if f_agent in local_env.agents and local_commander is not None:
                self_state = torch.FloatTensor(obs[f_agent]["self_state"]).unsqueeze(0)
                hidden_in = commander_hiddens[f_agent]
                
                with torch.no_grad():
                    dist, _, hidden_out = local_commander(self_state, msgs_tensor, msgs_mask, hidden_in)
                    value, c_hidden_out = local_critic(g_state_tensor, critic_hiddens[f_agent])
                    action = dist.sample()
                    logprob = dist.log_prob(action).sum(1)
                
                commander_hiddens[f_agent] = hidden_out 
                critic_hiddens[f_agent] = c_hidden_out 
                actions[f_agent] = action.squeeze(0).numpy()
                
                current_step_data[f_agent] = {
                    "type": "commander",
                    "self": self_state, "msgs": msgs_tensor, "msg_mask": msgs_mask,
                    "hidden": hidden_in,
                    "action": action, "logprob": logprob, "value": value, "g_state": g_state_tensor, "c_hidden": critic_hiddens[f_agent]
                }
            else:
                current_step_data[f_agent] = {
                    "type": "commander",
                    "self": torch.zeros(1, config['fixed_self_dim'] if config['N_FIXED'] > 0 else 1),
                    "msgs": msgs_tensor, "msg_mask": msgs_mask, "hidden": torch.zeros(1, 1, 128),
                    "action": torch.zeros(1, 4), "logprob": torch.tensor([0.0]), "value": torch.tensor([[0.0]]),
                    "g_state": g_state_tensor, "c_hidden": torch.zeros(1, 1, 128)
                }
                actions[f_agent] = np.zeros(4)

        # === KROK PROSTŘEDÍ ===
        next_obs, rewards, terminations, truncations, infos = local_env.step(actions)
        
        for agent, data in current_step_data.items():
            rew = rewards.get(agent, 0.0)
            ep_data[agent]["rewards"].append(rew)
            ep_data[agent]["values"].append(data["value"])
            episode_reward += rew
            
            rollout_memory.append({
                "agent": agent, "data": data, "step_idx": len(ep_data[agent]["rewards"]) - 1
            })
        
        obs = next_obs
        for agent, terminated in terminations.items():
            if terminated and agent_lifespans[agent] == config['max_steps']:
                agent_lifespans[agent] = step

    # == TVŮJ VÝPOČET ZPRACOVÁNÍ EPIZODY (Discounted Returns) ==
    for agent in ep_data:
        rewards = ep_data[agent]["rewards"]
        if not rewards: continue
        discounted_sum = 0
        returns = []
        for r in reversed(rewards):
            discounted_sum = r + config['gamma'] * discounted_sum
            returns.insert(0, discounted_sum)
        ep_data[agent]["returns"] = returns

    # Sbalení dat tak, jak je očekává hlavní vlákno
    for item in rollout_memory:
        item["data"]["return"] = ep_data[item["agent"]]["returns"][item["step_idx"]]

    local_env.sim.stop_simulation()
    avg_lifespan = np.mean(list(agent_lifespans.values()))
    
    # Vrátíme kompletní pole dat pro batch, celkovou odměnu a průměrný věk
    final_data = [item["data"] for item in rollout_memory]

    # profiler.disable()
    # profiler.dump_stats(f"profil_delnika_{os.getpid()}.prof")

    return final_data, episode_reward, avg_lifespan


# ============================================================================
# HLAVNÍ FUNKCE TRAIN
# ============================================================================
def train():
    print("🚀 Spouštím Heterogenní MAPPO Trénink (PARALELNÍ)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Hyperparametry
    num_episodes = 8000
    max_steps = 1000
    learning_rate = 3e-4
    gamma = 0.99    # Jak moc se agent zajímá o budoucí odměny (0.99 = velmi, 0.9 = méně)
    clip_coef = 0.2 # PPO klipovací faktor (jak moc se může nová politika odchýlit od staré), aby se zabránilo příliš velkým updateům
    update_epochs = 8
    episodes_per_batch = 20

    lr_commander = learning_rate
    lr_critic = learning_rate           # Kritik musí stíhat sledovat změny
    lr_scout_fine_tune = learning_rate / 10  # Dron se jen jemně dolaďuje (1e-6)
    
    # Konfigurace týmu
    N_QUADS = 0
    N_FIXED = 1
    
    # Dočasné prostředí jen pro zjištění dimenzí sítí (pak ho smažeme)
    temp_env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=4000.0, max_steps=max_steps)
    if N_QUADS > 0:
        scout_self_dim = temp_env.observation_space("quad_0")["self_state"].shape[0]
    else:
        scout_self_dim = 12 

    scout_msg_dim = 5                                 
    scout_hidden_dim = 128                            
    
    if N_FIXED > 0:
        fixed_self_dim = temp_env.observation_space(temp_env.fixed_agents[0])["self_state"].shape[0]
    else:
        fixed_self_dim = 0
    
    global_state_dim = temp_env.state_space.shape[0]
    # temp_env.sim.stop_simulation()

    # Config dict pro workery
    worker_config = {
        'N_QUADS': N_QUADS, 'N_FIXED': N_FIXED, 'grid_size_m': 4000.0, 'max_steps': max_steps,
        'scout_self_dim': scout_self_dim, 'scout_msg_dim': scout_msg_dim, 'scout_hidden_dim': scout_hidden_dim,
        'fixed_self_dim': fixed_self_dim, 'global_state_dim': global_state_dim, 'gamma': gamma
    }

    # 3. Inicializace Sítí na GPU
    if N_QUADS > 0:
        scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim).to(device)
        path_to_old_model = "retrainModels/scout_ep3980.pt"
        if os.path.exists(path_to_old_model):
            print(f"📥 Načítám naučený model drona z {path_to_old_model}")
            scout_actor.load_state_dict(torch.load(path_to_old_model, map_location=device), strict=False)
        else:
            print(f"⚠️ Nenalezen žádný model drona, trénink začne od nuly.")
    else:
        scout_actor = None
    
    if N_FIXED > 0:
        commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=scout_msg_dim).to(device)
        path_to_old_model = "retrainModels/commander_best.pt"
        if os.path.exists(path_to_old_model):
            print(f"📥 Načítám naučený model letadla z {path_to_old_model}")
            commander_actor.load_state_dict(torch.load(path_to_old_model, map_location=device), strict=False)
        else:
            print(f"⚠️ Nenalezen žádný model letadla, trénink začne od nuly.")
    else:
        commander_actor = None
    
    critic = MAPPOCritic(global_state_dim).to(device)
    
    # Optimalizátor
    optim_groups = [{"params": critic.parameters(), "lr": lr_critic}]
    if scout_actor: optim_groups.append({"params": scout_actor.parameters(), "lr": lr_scout_fine_tune})
    if commander_actor: optim_groups.append({"params": commander_actor.parameters(), "lr": lr_commander})
    optimizer = optim.Adam(optim_groups)
    
    # Historie
    episode_rewards_history = []
    loss_history = []  
    entropy_history = []
    v_loss_history = [] 
    lifespan_history = [] 

    os.makedirs("saved_models", exist_ok=True)
    best_avg_reward = -1000.0
    episodes_played = 0
    num_batches = num_episodes // episodes_per_batch

    # --- HLAVNÍ PARALELNÍ SMYČKA ---
    # Tady tvá smyčka 'for episode' končí a mění se na Pool
    with ProcessPoolExecutor(max_workers=episodes_per_batch) as executor:
        for batch_idx in range(1, num_batches + 1):
            
            # Vytáhneme váhy na CPU, aby je přečetly thready
            scout_w = {k: v.cpu() for k, v in scout_actor.state_dict().items()} if scout_actor else None
            cmdr_w = {k: v.cpu() for k, v in commander_actor.state_dict().items()} if commander_actor else None
            critic_w = {k: v.cpu() for k, v in critic.state_dict().items()}

            # Vyšleme workery (spustí se jich naráz tolik, kolik je episodes_per_batch)
            futures = []
            for _ in range(episodes_per_batch):
                futures.append(executor.submit(collect_single_episode, scout_w, cmdr_w, critic_w, worker_config))

            # Globální buffery z tvého původního kódu
            batch_data = {
                "maps": [], "self_states": [], "neighbor_states": [], "neighbor_masks": [], 
                "fixed_states": [], "incoming_msgs": [], "msg_masks": [], 
                "g_states": [], "actions": [], "logprobs": [], "returns": [], "values": [], 
                "agent_types": [], "hiddens": [], "critic_hiddens": []                      
            }

            # Posbíráme data z workerů (počkáme, až všechny dojedou)
            batch_rewards = []
            for future in futures:
                ep_rollout, ep_reward, ep_lifespan = future.result()
                
                episodes_played += 1
                episode_rewards_history.append(ep_reward)
                batch_rewards.append(ep_reward)
                lifespan_history.append(ep_lifespan)
                
                # Přesun dat z Rollout Memory do Global Batch (Tvůj původní if/else strom)
                for d in ep_rollout:
                    batch_data["actions"].append(d["action"])
                    batch_data["logprobs"].append(d["logprob"])
                    batch_data["values"].append(d["value"])
                    batch_data["g_states"].append(d["g_state"])
                    batch_data["returns"].append(d["return"])
                    
                    if d["type"] == "scout":
                        batch_data["agent_types"].append(0) 
                        batch_data["maps"].append(d["map"])
                        batch_data["self_states"].append(d["self"])
                        batch_data["neighbor_states"].append(d["neigh_s"])
                        batch_data["neighbor_masks"].append(d["neigh_m"])
                        batch_data["hiddens"].append(d["hidden"])
                        batch_data["critic_hiddens"].append(d["c_hidden"])
                        batch_data["fixed_states"].append(torch.zeros(1, fixed_self_dim)) 
                        batch_data["incoming_msgs"].append(torch.zeros(1, max(1, N_QUADS), scout_msg_dim))
                        batch_data["msg_masks"].append(torch.ones(1, max(1, N_QUADS), dtype=torch.bool))

                    else:
                        batch_data["agent_types"].append(1) 
                        batch_data["fixed_states"].append(d["self"])
                        batch_data["incoming_msgs"].append(d["msgs"])
                        batch_data["msg_masks"].append(d["msg_mask"])
                        batch_data["maps"].append(torch.zeros(1, 1, 32, 32))
                        batch_data["self_states"].append(torch.zeros(1, scout_self_dim))
                        batch_data["neighbor_states"].append(torch.zeros(1, max(1, N_QUADS-1), 3)) 
                        batch_data["neighbor_masks"].append(torch.ones(1, max(1, N_QUADS-1), dtype=torch.bool))
                        batch_data["hiddens"].append(d["hidden"])
                        batch_data["critic_hiddens"].append(d["c_hidden"])

            # Logování za dávku (místo za epizodu)
            avg_batch = np.mean(batch_rewards)
            avg_reward = np.mean(episode_rewards_history[-15:]) if len(episode_rewards_history) >= 15 else np.mean(episode_rewards_history)
            print(f"Batch {batch_idx:04d} (Ep {episodes_played:04d}) | Avg Batch Reward: {avg_batch:.2f} | Průměr(15): {avg_reward:.2f}")

            # === UKLÁDÁNÍ ZLATÉHO STANDARDU ===
            if episodes_played >= 15 and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                if scout_actor:
                    torch.save(scout_actor.state_dict(), f"saved_models/scout_best.pt")
                if commander_actor:
                    torch.save(commander_actor.state_dict(), f"saved_models/commander_best.pt")
                print(f"⭐ Uložen nový NEJLEPŠÍ model (Průměr: {best_avg_reward:.2f})!")

            # == TVŮJ PŮVODNÍ PPO UPDATE BEZE ZMĚN ==
            if len(batch_data["returns"]) > 0:
                print(f"🛠️ UPDATE SÍTÍ ({len(batch_data['returns'])} vzorků)...")
                
                b_actions = torch.cat(batch_data["actions"]).to(device)
                b_logprobs = torch.cat(batch_data["logprobs"]).to(device)
                b_returns = torch.tensor(batch_data["returns"], dtype=torch.float32).unsqueeze(1).to(device)
                b_values = torch.cat(batch_data["values"]).to(device)
                b_g_states = torch.cat(batch_data["g_states"]).to(device)
                b_types = torch.tensor(batch_data["agent_types"]).to(device)
                
                advantages = b_returns - b_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                num_agents = N_QUADS + N_FIXED
                stride = max_steps * num_agents
                episodes = episodes_per_batch
            
                for epoch in range(update_epochs):
                    new_logprobs = torch.zeros_like(b_logprobs)
                    entropy_sum = torch.tensor(0.0, device=device)
                    steps = max_steps

                    def to_seq(data_list):
                        t = torch.cat(data_list).to(device)
                        num_trajectories = t.shape[0] // steps 
                        return t.view(num_trajectories, steps, *t.shape[1:])
                    
                    def to_seq_critic(data_list):
                        t = torch.cat(data_list).to(device)
                        t = t.view(episodes, steps, num_agents, -1)
                        t = t.transpose(1, 2).contiguous()
                        return t.view(episodes * num_agents, steps, -1)
                    
                    # 1. SCOUT UPDATE
                    idx_s = (b_types == 0)
                    if idx_s.any():
                        b_hiddens = torch.stack([batch_data["hiddens"][k * stride] for k in range(episodes)])
                        b_hiddens = b_hiddens.squeeze(2).transpose(0, 1).contiguous().to(device)

                        if scout_actor is None: raise ValueError("Nemůžu aktualizovat Scouta (N_QUADS=0).")
                        dist, _, _ = scout_actor(
                            to_seq([batch_data["maps"][i] for i in range(len(b_types)) if b_types[i]==0]),
                            to_seq([batch_data["self_states"][i] for i in range(len(b_types)) if b_types[i]==0]),
                            to_seq([batch_data["neighbor_states"][i] for i in range(len(b_types)) if b_types[i]==0]),
                            to_seq([batch_data["neighbor_masks"][i] for i in range(len(b_types)) if b_types[i]==0]),
                            b_hiddens
                        )
                        actions_s = b_actions[idx_s]
                        new_logprobs[idx_s] = dist.log_prob(actions_s).sum(1)
                        entropy_sum += dist.entropy().sum(1).mean()

                    # 2. COMMANDER UPDATE
                    idx_c = (b_types == 1)
                    if idx_c.any() and commander_actor is not None:
                        offset = 1 if N_QUADS > 0 else 0 
                        b_hiddens_c = torch.stack([batch_data["hiddens"][k * stride + offset] for k in range(episodes)])
                        b_hiddens_c = b_hiddens_c.squeeze(2).transpose(0, 1).contiguous().to(device)

                        dist, _, _ = commander_actor(
                            to_seq([batch_data["fixed_states"][i] for i in range(len(b_types)) if b_types[i]==1]),
                            to_seq([batch_data["incoming_msgs"][i] for i in range(len(b_types)) if b_types[i]==1]),
                            to_seq([batch_data["msg_masks"][i] for i in range(len(b_types)) if b_types[i]==1]),
                            b_hiddens_c
                        )
                        actions_c = b_actions[idx_c]
                        new_logprobs[idx_c] = dist.log_prob(actions_c).sum(1)
                        entropy_sum += dist.entropy().sum(1).mean()

                    # Společný Kritik
                    b_critic_hiddens = torch.stack([batch_data["critic_hiddens"][k * stride + j] for k in range(episodes) for j in range(num_agents)])
                    b_critic_hiddens = b_critic_hiddens.squeeze(2).transpose(0, 1).contiguous().to(device)

                    new_values, _ = critic(to_seq_critic(batch_data["g_states"]), b_critic_hiddens)
                    value_loss = nn.MSELoss()(new_values, b_returns)
                    
                    # Loss
                    ratio = torch.exp(new_logprobs - b_logprobs)
                    pg_loss1 = -advantages.squeeze() * ratio
                    pg_loss2 = -advantages.squeeze() * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # TVÁ ZMĚNĚNÁ ENTROPIE (0.01) JE ZDE:
                    loss = policy_loss + 0.01 * value_loss - 0.01 * entropy_sum
                    
                    loss_history.append(loss.item())
                    v_loss_history.append(value_loss.item())
                    entropy_history.append(float(entropy_sum))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    params = list(critic.parameters())
                    if scout_actor: params += list(scout_actor.parameters())
                    if commander_actor: params += list(commander_actor.parameters())
                    nn.utils.clip_grad_norm_(params, max_norm=0.5)
                    optimizer.step()

            # Vykreslování grafů (Tebou napsané, generuje se každých 10 batchů = 80 epizod)
            if batch_idx % 10 == 0:
                if scout_actor: torch.save(scout_actor.state_dict(), f"saved_models/scout_ep{episodes_played}.pt")
                if commander_actor: torch.save(commander_actor.state_dict(), f"saved_models/commander_ep{episodes_played}.pt")

                plt.figure(figsize=(20, 5))
                
                plt.subplot(1, 4, 1)
                plt.plot(episode_rewards_history, label="Reward", alpha=0.3, color='green')
                if len(episode_rewards_history) > 20:
                    plt.plot(np.convolve(episode_rewards_history, np.ones(20)/20, mode='valid'), label="MA 20", color='darkgreen')
                plt.title("Vývoj odměn")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(1, 4, 2)
                plt.plot(loss_history, label="Total Loss", color='red', alpha=0.5)
                plt.plot(v_loss_history, label="Value Loss (Critic)", color='blue', alpha=0.5)
                plt.title("Vývoj Loss (Log)")
                plt.yscale('log') 
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(1, 4, 3)
                if 'entropy_history' in locals() or 'entropy_history' in globals():
                    plt.plot(entropy_history, color='purple')
                    plt.title("Vývoj Entropie (Průzkum)")
                else:
                    plt.text(0.5, 0.5, 'Pro graf entropie\npřidej logování do update smyčky', ha='center')
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 4, 4)
                plt.plot(lifespan_history, color='orange')
                plt.title("Průměrná délka dožití")

                plt.tight_layout()
                plt.savefig("final_training_plot.png")
                plt.close()

if __name__ == "__main__":
    train()