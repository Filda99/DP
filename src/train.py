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
def collect_episodes_per_worker(num_eps_to_collect, scout_w, cmdr_w, critic_w, config, batch_start_idx):
    """
        Tato funkce běží na jednom jádru a nasbírá několik epizod za sebou.
        Inputs: 
            - num_eps_to_collect: Kolik epizod má tento worker nasbírat (např. 5)
            - scout_w, cmdr_w, critic_w: Váhy sítí (předané z hlavního procesu)
            - config: Konfigurace prostředí a sítí (stejná pro všechny workery)    
            - batch_start_idx: Index, od kterého začíná tento batch epizod
    """
    # Musi být importy uvnitř funkce, protože běží v samostatném procesu
    import torch
    import numpy as np
    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, MAPPOCritic
 
    # 1. Inicializace prostředí
    local_env = DroneFireEnv(num_quads=config['N_QUADS'], num_fixed=config['N_FIXED'], 
                             grid_size_m=config['grid_size_m'], max_steps=config['max_steps'])
    # 2. Inicializace sítí na CPU
    local_scout = None
    if config['N_QUADS'] > 0:
        local_scout = ScoutActor(self_state_dim=config['scout_self_dim'], msg_dim=config['scout_msg_dim'], hidden_dim=config['scout_hidden_dim'])
        local_scout.load_state_dict(scout_w); local_scout.eval() # Načteme váhy a přepneme do eval módu (nebudeme trénovat uvnitř workerů, jen sbírat data)
 
    local_commander = None
    if config['N_FIXED'] > 0:
        local_commander = CommanderActor(self_state_dim=config['fixed_self_dim'], msg_input_dim=config['scout_msg_dim'])
        local_commander.load_state_dict(cmdr_w); local_commander.eval()
    local_critic = MAPPOCritic(config['global_state_dim'])
    local_critic.load_state_dict(critic_w); local_critic.eval()
 
    # Buffery pro data z epizod (budou posílány zpět do hlavního procesu)
    worker_aggregated = {k: [] for k in ["maps", "self_states", "neighbor_states", "neighbor_masks", 
                                         "fixed_states", "incoming_msgs", "msg_masks", "g_states", 
                                         "actions", "logprobs", "returns", "values", "agent_types", 
                                         "hiddens", "critic_hiddens"]}
    worker_total_rewards = []
    worker_lifespans = []
 
    # Pomocné dummy tensory pro padding (pro mrtvé drony)
    d_map = torch.zeros(1, 1, 32, 32)
    d_scout_self = torch.zeros(1, config['scout_self_dim'])
    d_neigh_s = torch.zeros(1, max(1, config['N_QUADS']-1), 3)
    d_neigh_m = torch.ones(1, max(1, config['N_QUADS']-1), dtype=torch.bool)
    d_cmd_self = torch.zeros(1, config['fixed_self_dim'] if config['fixed_self_dim'] > 0 else 1)
    d_msgs = torch.zeros(1, max(1, config['N_QUADS']), config['scout_msg_dim'])
    d_msg_m = torch.ones(1, max(1, config['N_QUADS']), dtype=torch.bool)
 
    # 3. Smyčka pro sběr epizod, běží do té doby, než nasbíráme požadovaný počet epizod pro tento worker
    for ep_offset in range(num_eps_to_collect):
        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_offset)
        # Reset paměti na začátku každé epizody
        scout_h = {q: torch.zeros(1, 1, config['scout_hidden_dim']) for q in local_env.quad_agents}
        cmdr_h = {f: torch.zeros(1, 1, 128) for f in local_env.fixed_agents}
        crit_h = {a: torch.zeros(1, 1, 128) for a in local_env.possible_agents}
 
        ep_rollouts = {a: [] for a in local_env.possible_agents}
        agent_lifespans = {a: config['max_steps'] for a in local_env.possible_agents}
        episode_reward = 0.0
 
        # Smyčka musí doběhnout do konce, kvůli výpočtu discounted returns, které se počítají zpětně přes všechny kroky epizody
        for step in range(config['max_steps']):
            global_state = local_env.state()
            g_tensor = torch.FloatTensor(global_state).unsqueeze(0)
            actions = {}
            step_results = {}
 
            # --- SCOUTI ---
            scout_msgs_list, scout_mask_list = [], []
            for q in local_env.quad_agents:
                if q in local_env.agents:
                    with torch.no_grad():
                        l_map = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0)
                        s_st = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0)
                        n_s, n_m = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0), torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0)
                        dist, msg, h_out = local_scout(l_map, s_st, n_s, n_m, scout_h[q])
                        val, c_h_out = local_critic(g_tensor, crit_h[q])
                        act = dist.sample()
                    step_results[q] = {"type": "scout", "map": l_map, "self": s_st, "n_s": n_s, "n_m": n_m,
                                       "h": scout_h[q], "act": act, "lp": dist.log_prob(act).sum(1), "val": val, "gs": g_tensor, "ch": crit_h[q]}
                    scout_h[q], crit_h[q] = h_out, c_h_out
                    actions[q] = act.squeeze(0).numpy()
                    scout_msgs_list.append(msg); scout_mask_list.append(False)
                else:
                    # PADDING: Dron je mrtvý, posíláme nuly
                    scout_msgs_list.append(torch.zeros(1, config['scout_msg_dim'])); scout_mask_list.append(True)
                    step_results[q] = {"type": "dead_scout", "gs": g_tensor}
 
            # --- COMMANDER ---
            msgs_t = torch.stack(scout_msgs_list, dim=1) if scout_msgs_list else d_msgs
            msgs_m = torch.tensor(scout_mask_list).unsqueeze(0) if scout_mask_list else d_msg_m
            for f in local_env.fixed_agents:
                if f in local_env.agents:
                    with torch.no_grad():
                        s_st = torch.FloatTensor(obs[f]["self_state"]).unsqueeze(0)
                        dist, _, h_out = local_commander(s_st, msgs_t, msgs_m, cmdr_h[f])
                        val, c_h_out = local_critic(g_tensor, crit_h[f])
                        act = dist.sample()
                    step_results[f] = {"type": "commander", "self": s_st, "msgs": msgs_t, "m_m": msgs_m,
                                       "h": cmdr_h[f], "act": act, "lp": dist.log_prob(act).sum(1), "val": val, "gs": g_tensor, "ch": crit_h[f]}
                    cmdr_h[f], crit_h[f] = h_out, c_h_out
                    actions[f] = act.squeeze(0).numpy()
                else:
                    step_results[f] = {"type": "dead_cmdr", "gs": g_tensor}
 
            # Krok v prostředí
            if local_env.agents:
                obs, rewards, terminations, _, _ = local_env.step(actions)
                for a in local_env.possible_agents:
                    r = rewards.get(a, 0.0)
                    episode_reward += r
                    step_results[a]["reward"] = r
                    if terminations.get(a, False) and agent_lifespans[a] == config['max_steps']: agent_lifespans[a] = step
            else:
                for a in local_env.possible_agents: step_results[a]["reward"] = 0.0
            for a in local_env.possible_agents: ep_rollouts[a].append(step_results[a])

        # print(f"Worker {os.getpid()} | Epizoda {batch_start_idx + ep_offset} | Odměna: {episode_reward:.2f} | Lifespan průměr: {np.mean(list(agent_lifespans.values())):.1f} kroků")
 
        # Výpočet Returns (Discounting) - zpětně přes celých 1000 kroků
        for a in local_env.possible_agents:
            disc_sum = 0
            for i in reversed(range(config['max_steps'])):
                ep_rollouts[a][i]["ret"] = ep_rollouts[a][i]["reward"] + config['gamma'] * disc_sum
                disc_sum = ep_rollouts[a][i]["ret"]
        # Sbalení dat do hlavního bufferu (vždy 1000 kroků na agenta!)
        for step_idx in range(config['max_steps']):
            for a_name in local_env.possible_agents:
                d = ep_rollouts[a_name][step_idx]
                worker_aggregated["g_states"].append(d["gs"])
                worker_aggregated["returns"].append(d["ret"])
                worker_aggregated["values"].append(d.get("val", torch.tensor([[0.0]])))
                if "dead" in d["type"]:
                    worker_aggregated["actions"].append(torch.zeros(1, 4))
                    worker_aggregated["logprobs"].append(torch.tensor([0.0]))
                    worker_aggregated["hiddens"].append(torch.zeros(1, 1, 128))
                    worker_aggregated["critic_hiddens"].append(torch.zeros(1, 1, 128))
                    worker_aggregated["agent_types"].append(0 if "scout" in d["type"] else 1)
                    worker_aggregated["maps"].append(d_map); worker_aggregated["self_states"].append(d_scout_self if "scout" in d["type"] else d_cmd_self)
                    worker_aggregated["neighbor_states"].append(d_neigh_s); worker_aggregated["neighbor_masks"].append(d_neigh_m)
                    worker_aggregated["fixed_states"].append(d_cmd_self); worker_aggregated["incoming_msgs"].append(d_msgs); worker_aggregated["msg_masks"].append(d_msg_m)
                else:
                    worker_aggregated["actions"].append(d["act"]); worker_aggregated["logprobs"].append(d["lp"])
                    worker_aggregated["hiddens"].append(d["h"]); worker_aggregated["critic_hiddens"].append(d["ch"])
                    if d["type"] == "scout":
                        worker_aggregated["agent_types"].append(0); worker_aggregated["maps"].append(d["map"]); worker_aggregated["self_states"].append(d["self"])
                        worker_aggregated["neighbor_states"].append(d["n_s"]); worker_aggregated["neighbor_masks"].append(d["n_m"])
                        worker_aggregated["fixed_states"].append(d_cmd_self); worker_aggregated["incoming_msgs"].append(d_msgs); worker_aggregated["msg_masks"].append(d_msg_m)
                    else:
                        worker_aggregated["agent_types"].append(1); worker_aggregated["fixed_states"].append(d["self"])
                        worker_aggregated["incoming_msgs"].append(d["msgs"]); worker_aggregated["msg_masks"].append(d["m_m"])
                        worker_aggregated["maps"].append(d_map); worker_aggregated["self_states"].append(d_scout_self)
                        worker_aggregated["neighbor_states"].append(d_neigh_s); worker_aggregated["neighbor_masks"].append(d_neigh_m)
 
        worker_total_rewards.append(episode_reward)
        worker_lifespans.append(np.mean(list(agent_lifespans.values())))
 
    local_env.sim.stop_simulation()
    for k in worker_aggregated:
        if k == "agent_types": worker_aggregated[k] = torch.tensor(worker_aggregated[k], dtype=torch.long)
        elif k == "returns": worker_aggregated[k] = torch.tensor(worker_aggregated[k], dtype=torch.float32)
        else: worker_aggregated[k] = torch.cat(worker_aggregated[k])
    return worker_aggregated, worker_total_rewards, worker_lifespans



# ============================================================================
# HLAVNÍ FUNKCE TRAIN
# ============================================================================
def train():
    print("🚀 Spouštím Heterogenní MAPPO Trénink (PARALELNÍ)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Hyperparametry
    num_episodes = 15000
    max_steps = 1000
    learning_rate = 3e-4
    gamma = 0.99    # Jak moc se agent zajímá o budoucí odměny (0.99 = velmi, 0.9 = méně)
    clip_coef = 0.2 # PPO klipovací faktor (jak moc se může nová politika odchýlit od staré), aby se zabránilo příliš velkým updateům
    update_epochs = 8
    num_workers = 4
    eps_per_worker = 5 
    episodes_per_batch = num_workers * eps_per_worker 

    lr_commander = learning_rate
    lr_critic = learning_rate
    lr_scout_fine_tune = 5e-5  # Scout je předtrénovaný — jemné doladění, nechceme přepsat naučené chování
    
    # Konfigurace týmu
    N_QUADS = 1
    N_FIXED = 1
    
    # Dočasné prostředí jen pro zjištění dimenzí sítí (pak ho smažeme)
    temp_env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=500.0, max_steps=max_steps)
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
        'N_QUADS': N_QUADS, 'N_FIXED': N_FIXED, 'grid_size_m': 500.0, 'max_steps': max_steps,
        'scout_self_dim': scout_self_dim, 'scout_msg_dim': scout_msg_dim, 'scout_hidden_dim': scout_hidden_dim,
        'fixed_self_dim': fixed_self_dim, 'global_state_dim': global_state_dim, 'gamma': gamma
    }

    # 3. Inicializace Sítí na GPU
    if N_QUADS > 0:
        scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim).to(device)
        path_to_old_model = "retrainModels/scout_ep8600.pt"
        if os.path.exists(path_to_old_model):
            print(f"📥 Načítám naučený model drona z {path_to_old_model}")
            scout_actor.load_state_dict(torch.load(path_to_old_model, map_location=device), strict=False)
        else:
            print(f"⚠️ Nenalezen žádný model drona, trénink začne od nuly.")
    else:
        scout_actor = None
    
    if N_FIXED > 0:
        commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=scout_msg_dim).to(device)
        path_to_old_model = "retrainModels/commander_ep2400.pt"
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
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_idx in range(1, num_batches + 1):
            # 1. INICIALIZÁCIA BUFFERU (Musí byť TU, aby bol pre každý batch prázdny)
            batch_data = {k: [] for k in ["maps", "self_states", "neighbor_states", "neighbor_masks", 
                                          "fixed_states", "incoming_msgs", "msg_masks", "g_states", 
                                          "actions", "logprobs", "returns", "values", "agent_types", 
                                          "hiddens", "critic_hiddens"]}
 
            # 2. Príprava váh na CPU
            scout_w = {k: v.cpu() for k, v in scout_actor.state_dict().items()} if scout_actor else None
            cmdr_w = {k: v.cpu() for k, v in commander_actor.state_dict().items()} if commander_actor else None
            critic_w = {k: v.cpu() for k, v in critic.state_dict().items()}
 
            # 3. Odoslanie úloh workerom
            futures = []
            for i in range(num_workers):
                futures.append(executor.submit(collect_episodes_per_worker, eps_per_worker,
                                              scout_w, cmdr_w, critic_w, worker_config, episodes_played))
 
            # 4. Zber výsledkov
            batch_rewards = []
            for future in futures:
                w_agg, w_rewards, w_lifespans = future.result()
                batch_rewards.extend(w_rewards)
                lifespan_history.extend(w_lifespans)
                episode_rewards_history.extend(w_rewards)
                episodes_played += len(w_rewards)
 
                # Tu sa buffer plní dátami z workerov
                for k in batch_data:
                    batch_data[k].append(w_agg[k])

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

            # === PPO UPDATE (GPU maká) ===
            if len(batch_data["returns"]) > 0:
                # --- KOUZLO ROZBALENÍ ---
                # Dělníci nám poslali 20 obřích beden. PPO kód ale čeká 20 000 malých krabiček.
                # Rozsekáme ty bedny zpět na jednotlivé kroky (ve zlomku milisekundy).
                for k in batch_data:
                    big_tensor = torch.cat(batch_data[k])
                    batch_data[k] = list(torch.split(big_tensor, 1))
 
                # Stackování dat (Upraveno na torch.cat, jelikož už to jsou tensory)
                b_actions = torch.cat(batch_data["actions"]).to(device)
                b_logprobs = torch.cat(batch_data["logprobs"]).to(device)
                b_returns = torch.cat(batch_data["returns"]).unsqueeze(1).to(device)
                b_values = torch.cat(batch_data["values"]).to(device)
                b_g_states = torch.cat(batch_data["g_states"]).to(device)
                b_types = torch.cat(batch_data["agent_types"]).to(device)
                print(f"🛠️ UPDATE SÍTÍ ({len(b_returns)} vzorků)...")
                # Advantages
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
                    
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_sum
                    
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
            if batch_idx % 1 == 0:
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