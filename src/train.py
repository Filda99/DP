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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Hyperparametry
    num_episodes = 1000
    max_steps = 1000
    learning_rate = 1e-5
    gamma = 0.99    # Jak moc se agent zajímá o budoucí odměny (0.99 = velmi, 0.9 = méně)
    clip_coef = 0.2 # PPO klipovací faktor (jak moc se může nová politika odchýlit od staré), aby se zabránilo příliš velkým updateům
    update_epochs = 8
    episodes_per_batch = 15

    lr_commander = learning_rate        # Letadlo se učí naplno (např. 1e-5)
    lr_critic = learning_rate           # Kritik musí stíhat sledovat změny
    lr_scout_fine_tune = learning_rate / 10  # Dron se jen jemně dolaďuje (1e-6)
    
    # Konfigurace týmu
    N_QUADS = 1
    N_FIXED = 1
    
    # 2. Inicializace prostředí
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=1000.0, max_steps=max_steps)
    
    # Zjištění dimenzí pro sítě
    # A) SCOUT (Quad)
    obs_q = env.observation_space("quad_0")
    scout_self_dim = obs_q["self_state"].shape[0]
    scout_msg_dim = 5                                 # Velikost zprávy (vymyšleno námi)
    scout_hidden_dim = 128                            # Velikost paměti GRU
    
    # B) COMMANDER (Fixed)
    if N_FIXED > 0:
        obs_f = env.observation_space("fixed_0")
        fixed_self_dim = obs_f["self_state"].shape[0]
    else:
        fixed_self_dim = 0
    
    # C) GLOBAL STATE (Critic)
    global_state_dim = env.state_space.shape[0]

    # 3. Inicializace Sítí
    # Scout má paměť (GRU), Commander zatím ne
    scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim).to(device)

    # Cesta k funkčnímu modelu
    path_to_old_model = "retrainModels/scout_best.pt"
    if os.path.exists(path_to_old_model):
        print(f"📥 Načítám naučený model drona z {path_to_old_model}")
        # strict=False je jistota, pokud by se drobné detaily v architektuře lišily
        scout_actor.load_state_dict(torch.load(path_to_old_model, map_location=device), strict=False)
    else:
        print(f"⚠️ Nenalezen žádný naučený model drona na {path_to_old_model}, trénink začne od nuly.")
    
    if N_FIXED > 0:
        commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=scout_msg_dim).to(device)
    else:
        commander_actor = None # Nevytvářej síť, když není letadlo
    
    critic = MAPPOCritic(global_state_dim).to(device)
    
    # Společný optimalizátor pro celý "mozkový trust"
    # Sestavíme skupiny parametrů
    optim_groups = [
        {"params": scout_actor.parameters(), "lr": lr_scout_fine_tune},
        {"params": critic.parameters(), "lr": lr_critic}
    ]

    if commander_actor:
        optim_groups.append({"params": commander_actor.parameters(), "lr": lr_commander})
    optimizer = optim.Adam(optim_groups)
    
    # Historie pro grafy
    episode_rewards_history = []
    loss_history = []  # Celková ztráta
    entropy_history = []
    v_loss_history = [] # Ztráta kritika (Value Loss)
    lifespan_history = [] # Pro sledování, jak dlouho každý agent přežil


    os.makedirs("saved_models", exist_ok=True)
    
    # Globální buffery
    batch_data = {
        "maps": [], "self_states": [], "neighbor_states": [], "neighbor_masks": [], # Scout Inputs
        "fixed_states": [], "incoming_msgs": [], "msg_masks": [],                   # Commander Inputs
        "g_states": [], "actions": [], "logprobs": [], "returns": [], "values": [], # Common
        "agent_types": [], "hiddens": [], "critic_hiddens": []                      # Meta
    }

    best_avg_reward = -1000.0

    # --- HLAVNÍ SMYČKA ---
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        
        # Inicializace paměti pro Scouty (Batch=1, Hidden=128)
        # Každý dron má svou vlastní paměť
        scout_hiddens = {q: torch.zeros(1, 1, scout_hidden_dim) for q in env.quad_agents}
        # Inicializace paměti pro Commandera (Batch=1, Hidden=128)
        commander_hiddens = {f: torch.zeros(1, 1, 128) for f in env.fixed_agents}
        # Inicializace paměti pro Kritika (Global, Batch=1, Hidden=128)
        critic_hiddens = {a: torch.zeros(1, 1, 128) for a in env.possible_agents}

        # Pro sledování délky života agentů
        agent_lifespans = {agent: max_steps for agent in env.possible_agents}

        # Epizodní buffer
        ep_data = {agent: {"rewards": [], "values": []} for agent in env.possible_agents}
        # Dočasné úložiště pro rollout data (protože je musíme spárovat s odměnami)
        rollout_memory = [] 

        episode_reward = 0.0
        
        for step in range(max_steps):
            # if not env.agents: break
            
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
                    local_map = torch.FloatTensor(obs[q_agent]["local_map"]).to(device).unsqueeze(0)
                    self_state = torch.FloatTensor(obs[q_agent]["self_state"]).to(device).unsqueeze(0)
                    neigh_s = torch.FloatTensor(obs[q_agent]["neighbor_states"]).to(device).unsqueeze(0)
                    neigh_m = torch.BoolTensor(obs[q_agent]["neighbor_mask"]).to(device).unsqueeze(0)
                    hidden_in = scout_hiddens[q_agent].to(device)
                    
                    # Akce sítě
                    with torch.no_grad():
                        dist, message, hidden_out = scout_actor(local_map, self_state, neigh_s, neigh_m, hidden_in)
                        value, c_hidden_out = critic(g_state_tensor.to(device), critic_hiddens[q_agent].to(device))
                        action = dist.sample()
                        logprob = dist.log_prob(action).sum(1)

                    # Uložení výstupů
                    scout_hiddens[q_agent] = hidden_out.cpu() # Update paměti
                    critic_hiddens[q_agent] = c_hidden_out.cpu()  # Update paměti pro další krok
                    actions[q_agent] = action.squeeze(0).cpu().numpy()

                    # Zpráva do éteru (pro Commandera)
                    scout_messages.append(message)
                    scout_alive_mask.append(False) # Žije
                    
                    # Uložení do paměti pro trénink
                    current_step_data[q_agent] = {
                        "type": "scout",
                        "map": local_map.cpu(), "self": self_state.cpu(), "neigh_s": neigh_s.cpu(), "neigh_m": neigh_m.cpu(), "hidden": hidden_in.cpu(),
                        "action": action.cpu(), "logprob": logprob.cpu(), "value": value.cpu(), "g_state": g_state_tensor.cpu(), "c_hidden": critic_hiddens[q_agent].cpu()
                    }

                else:
                    # Dron je mrtvý -> Posílá prázdnou zprávu a je maskován
                    scout_messages.append(torch.zeros(1, scout_msg_dim))
                    scout_alive_mask.append(True) # Mrtvý (Ignorovat v Attention)

                    current_step_data[q_agent] = {
                        "type": "scout",
                        "map": torch.zeros(1, 1, 32, 32).cpu(),
                        "self": torch.zeros(1, scout_self_dim).cpu(),
                        "neigh_s": torch.zeros(1, env.max_neighbors, 3).cpu(),
                        "neigh_m": torch.ones(1, env.max_neighbors, dtype=torch.bool).cpu(),
                        "hidden": torch.zeros(1, 1, scout_hidden_dim).cpu(),
                        "action": torch.zeros(1, 4).cpu(),
                        "logprob": torch.tensor([0.0]).cpu(),
                        "value": torch.tensor([[0.0]]).cpu(),
                        "g_state": g_state_tensor.cpu(), # globální stav můžeme nechat
                        "c_hidden": torch.zeros(1, 1, 128).cpu()
                    }
                    actions[q_agent] = np.zeros(4) # prostředí ignoruje akce mrtvých

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
                if f_agent in env.agents and commander_actor is not None:
                    self_state = torch.FloatTensor(obs[f_agent]["self_state"]).to(device).unsqueeze(0)
                    hidden_in = commander_hiddens[f_agent].to(device)
                    
                    # Akce sítě (Dostává zprávy od scoutů!)
                    with torch.no_grad():
                        dist, _, hidden_out = commander_actor(self_state, msgs_tensor, msgs_mask.to(device), hidden_in)
                        value, c_hidden_out = critic(g_state_tensor.to(device), critic_hiddens[f_agent].to(device))
                        action = dist.sample()
                        logprob = dist.log_prob(action).sum(1)
                    
                    commander_hiddens[f_agent] = hidden_out.cpu() # Update paměti
                    critic_hiddens[f_agent] = c_hidden_out.cpu() # Update paměti pro další krok
                    actions[f_agent] = action.squeeze(0).cpu().numpy()
                    
                    current_step_data[f_agent] = {
                        "type": "commander",
                        "self": self_state.cpu(), "msgs": msgs_tensor.cpu(), "msg_mask": msgs_mask.cpu(),
                        "hidden": hidden_in.cpu(),
                        "action": action.cpu(), "logprob": logprob.cpu(), "value": value.cpu(), "g_state": g_state_tensor.cpu(), "c_hidden": critic_hiddens[f_agent].cpu()
                    }
                else:
                    current_step_data[f_agent] = {
                        "type": "commander",
                        "self": torch.zeros(1, fixed_self_dim if N_FIXED > 0 else 1).cpu(),
                        "msgs": msgs_tensor.cpu(), "msg_mask": msgs_mask.cpu(),
                        "hidden": torch.zeros(1, 1, 128).cpu(),
                        "action": torch.zeros(1, 4).cpu(),
                        "logprob": torch.tensor([0.0]).cpu(),
                        "value": torch.tensor([[0.0]]).cpu(),
                        "g_state": g_state_tensor.cpu(),
                        "c_hidden": torch.zeros(1, 1, 128).cpu()
                    }
                    actions[f_agent] = np.zeros(4)

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

            for agent, terminated in terminations.items():
                if terminated and agent_lifespans[agent] == max_steps:
                    agent_lifespans[agent] = step

        episode_rewards_history.append(episode_reward)

        # == ZPRACOVÁNÍ EPIZODY (Discounted Returns) ==
        # Pro každý agent spočítáme discounted returns a uložíme je do ep_data, aby byly spárovány s akcemi v rollout_memory.
        # To znamená, že pro každého agenta půjdeme jeho odměny pozpátku a spočítáme kumulativní sumu s diskontem gamma.
        # Tato hodnota nám řekne, jak dobrá byla akce v daném kroku s ohledem na budoucí odměny.
        for agent in ep_data:
            rewards = ep_data[agent]["rewards"]
            if not rewards: continue
            
            discounted_sum = 0
            returns = []
            # Jdeme pozpátku, abychom mohli kumulativně sčítat odměny s diskontem
            # což znamená, že poslední odměna v epizodě má největší váhu, a čím dál od ní jsme, tím menší váhu mají předchozí odměny.
            for r in reversed(rewards):
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            ep_data[agent]["returns"] = returns

        # Přesun dat z Rollout Memory do Global Batch
        # Tady se spárují akce, logproby, hodnoty a nyní i returns pro každý krok a každého agenta do jednoho velkého batch_data, který použijeme pro PPO update.
        # Každý záznam v rollout_memory obsahuje odkaz na agenta, index kroku v epizodě a data (akce, logprob, value, g_state, atd.).
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
                batch_data["critic_hiddens"].append(d["c_hidden"])
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
                batch_data["hiddens"].append(d["hidden"])
                # batch_data["hiddens"].append(torch.zeros(1, 1, scout_hidden_dim))
                batch_data["critic_hiddens"].append(d["c_hidden"])

        # Vypočítáme průměr za posledních 15 epizod
        if len(episode_rewards_history) >= 15:
            avg_reward = np.mean(episode_rewards_history[-15:])
        else:
            avg_reward = np.mean(episode_rewards_history)
            
        print(f"Epizoda {episode:03d} | Reward: {episode_reward:.2f} | Průměr(15): {avg_reward:.2f}")

        # Průměrná délka života v této epizodě
        avg_lifespan = np.mean(list(agent_lifespans.values()))
        lifespan_history.append(avg_lifespan)

        # === UKLÁDÁNÍ ZLATÉHO STANDARDU ===
        # Model se uloží POUZE tehdy, pokud je jeho PRŮMĚRNÁ úspěšnost za 15 her lepší než dřív.
        if episode >= 15 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(scout_actor.state_dict(), "saved_models/scout_best.pt")
            print(f"⭐ Uložen nový NEJLEPŠÍ model (Průměr: {best_avg_reward:.2f})!")

        # == PPO UPDATE ==
        # Když nasbíráme data z 15 epizod, provedeme aktualizaci sítí pomocí PPO algoritmu. To zahrnuje výpočet nových logprobs, hodnot a ztrát, a poté zpětnou propagaci a optimalizaci.
        # PPO update se provádí několikrát (update_epochs) nad stejnými daty, aby se z nich vytěžilo maximum, ale zároveň s klipováním, aby se zabránilo příliš velkým změnám v politice.
        # Po update se batch_data vyčistí, aby se začalo znovu sbírat data pro další sadu epizod.
        if episode % episodes_per_batch == 0 and len(batch_data["returns"]) > 0:
            print(f"🛠️ UPDATE SÍTÍ ({len(batch_data['returns'])} vzorků)...")
            
            # Stackování dat
            b_actions = torch.cat(batch_data["actions"]).to(device)
            b_logprobs = torch.cat(batch_data["logprobs"]).to(device)
            b_returns = torch.tensor(batch_data["returns"], dtype=torch.float32).unsqueeze(1).to(device)
            b_values = torch.cat(batch_data["values"]).to(device)
            b_g_states = torch.cat(batch_data["g_states"]).to(device)
            b_types = torch.tensor(batch_data["agent_types"]).to(device)
            
            # Advantages
            advantages = b_returns - b_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
            
	        # Výpočet stride pro více agentů
            num_agents = N_QUADS + N_FIXED
	    
            stride = max_steps * num_agents
            episodes = episodes_per_batch
	    
            for epoch in range(update_epochs):
                new_logprobs = torch.zeros_like(b_logprobs)
                entropy_sum = torch.tensor(0.0, device=device)

                steps = max_steps

                def to_seq(data_list):
                    t = torch.cat(data_list).to(device)
                    # view vytvoří [15, 200, *rozměry_vstupních_dat]
                    return t.view(episodes, steps, *t.shape[1:])
                
                # Verze pro Kritika (30 trajektorií: 15 epizod * 2 agenti)
                # Musí to být samostatná funkce, protože rozměr "Batch" je dvojnásobný
                def to_seq_critic(data_list):
                    t = torch.cat(data_list).to(device)
                    return t.view(episodes * num_agents, steps, *t.shape[1:])
                
                # 1. SCOUT UPDATE
                idx_s = (b_types == 0)
                if idx_s.any():
                    # Necháme data v pořadí, jak šla v epizodě
                    # a řekneme GRU, že jde o sekvenci.
                    # B_hiddens vezmeme jen ty ze STARTU epizod (každých 300 kroků)
                    # Oprava hiddens: bereme jen začátek každé epizody (index k*stride)
                    b_hiddens = torch.stack([batch_data["hiddens"][k * stride] for k in range(episodes)])
		    
                    # Tvar b_hiddens musí být (1, Batch_Epizod, 128)
                    b_hiddens = b_hiddens.squeeze(2).transpose(0, 1).contiguous().to(device)

                    dist, _, _ = scout_actor(
                        to_seq([batch_data["maps"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        to_seq([batch_data["self_states"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        to_seq([batch_data["neighbor_states"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        to_seq([batch_data["neighbor_masks"][i] for i in range(len(b_types)) if b_types[i]==0]),
                        b_hiddens
                    )

                    # Musíme zploštit akce, abychom spočítali log_prob pro všech 3000 vzorků
                    actions_s = b_actions[idx_s]
                    new_logprobs[idx_s] = dist.log_prob(actions_s).sum(1)
                    
                    # Přičtení entropie pro Scouta
                    entropy_sum += dist.entropy().sum(1).mean()

                # 2. COMMANDER UPDATE
                idx_c = (b_types == 1)
                if idx_c.any() and commander_actor is not None:
                    # Výběr hiddens pro letadla: stride je max_steps*num_agents. 
                    # Pokud je dron (Scout) v týmu první, letadlo je na indexu 1, 401, 801...
                    offset = 1 # Index letadla v týmu (0=Scout, 1=Commander)
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
                # 1. Připravíme hidden states ze startu epizod
                b_critic_hiddens = torch.stack([batch_data["critic_hiddens"][k * stride + j] for k in range(episodes) for j in range(num_agents)])
                b_critic_hiddens = b_critic_hiddens.squeeze(2).transpose(0, 1).contiguous().to(device)

                # 2. Forward pass přes celou sekvenci globálních stavů
                # g_states jsou [3000, GS_DIM], to_seq z nich udělá [15, 200, GS_DIM]
                new_values, _ = critic(to_seq_critic(batch_data["g_states"]), b_critic_hiddens)
                
                # 3. Výpočet MSE Loss (Pamatuj: b_returns už máš normalizované)
                value_loss = nn.MSELoss()(new_values, b_returns)
                
                # Loss
                ratio = torch.exp(new_logprobs - b_logprobs)
                pg_loss1 = -advantages.squeeze() * ratio
                pg_loss2 = -advantages.squeeze() * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                value_loss = nn.MSELoss()(new_values, b_returns)
                
                # policy_loss je hlavní ztráta pro aktualizaci politiky (Actorů), 
                # value_loss je ztráta pro aktualizaci Kritika, 
                # a entropy_sum je bonus, který podporuje průzkum (vyšší entropie znamená, že agent více experimentuje s různými akcemi).
                # celková ztráta kombinuje všechny tyto aspekty, přičemž klade největší důraz na policy_loss, menší na value_loss a 
                # přidává malý bonus za entropii, aby se zabránilo příliš brzy konvergenci na suboptimální politiku.
                loss = policy_loss + 0.01 * value_loss - 0.01 * entropy_sum
                # Uložíme si průměrné hodnoty pro graf (převod na float z tensoru)
                loss_history.append(loss.item())
                v_loss_history.append(value_loss.item())
                entropy_history.append(float(entropy_sum))
                
                optimizer.zero_grad()
                loss.backward()
                # === GRADIENT CLIPPING (Prevence explozí) ===
                params = list(scout_actor.parameters()) + list(critic.parameters())
                if commander_actor:
                    params += list(commander_actor.parameters())
                # max_norm znamena, jak moc se mohou gradienty změnit v jednom kroku. 
                # 0.2 je poměrně konzervativní hodnota, která pomáhá stabilizovat trénink a 
                # zabraňuje příliš velkým updateům, které by mohly způsobit kolaps učení.
                nn.utils.clip_grad_norm_(params, max_norm=0.2)
                optimizer.step()
            
            # Reset bufferů
            for k in batch_data: batch_data[k] = []

        if episode % 10 == 0:
            torch.save(scout_actor.state_dict(), f"saved_models/scout_ep{episode}.pt")
            if commander_actor:
                torch.save(commander_actor.state_dict(), f"saved_models/commander_ep{episode}.pt")

            plt.figure(figsize=(20, 5))
            
            # 1. Graf odměn
            plt.subplot(1, 4, 1)
            plt.plot(episode_rewards_history, label="Reward", alpha=0.3, color='green')
            if len(episode_rewards_history) > 20:
                plt.plot(np.convolve(episode_rewards_history, np.ones(20)/20, mode='valid'), label="MA 20", color='darkgreen')
            plt.title("Vývoj odměn")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 2. Graf Loss (Kritik a celková)
            plt.subplot(1, 4, 2)
            plt.plot(loss_history, label="Total Loss", color='red', alpha=0.5)
            plt.plot(v_loss_history, label="Value Loss (Critic)", color='blue', alpha=0.5)
            plt.title("Vývoj Loss (Log)")
            plt.yscale('log') 
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 3. Graf Entropie (Jak moc dron experimentuje)
            plt.subplot(1, 4, 3)
            # entropy_history si musíš přidat do train smyčky podobně jako loss_history
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