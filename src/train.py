import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend so plots can be saved without a display
import matplotlib.pyplot as plt

from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor, MAPPOCritic
from concurrent.futures import ProcessPoolExecutor

# ============================================================================
# WORKER FUNCTION  —  runs in parallel on a separate CPU core
# ============================================================================
# Overview of the parallel training design:
#   • The main process spawns N workers (ProcessPoolExecutor).
#   • Each worker runs this function: it creates its OWN copy of the env and
#     networks (CPU-only), collects several full episodes, computes discounted
#     returns, and ships pre-packaged tensors back to the main process.
#   • The main process receives all worker results, merges the tensors, and
#     runs the PPO gradient update on the GPU.
#   • No gradient computation happens inside workers — they are purely for
#     fast, parallel data collection.
# ============================================================================
def collect_episodes_per_worker(num_eps_to_collect, scout_w, cmdr_w, critic_w, config, batch_start_idx):
    """
    Collects rollout data for several episodes on a single CPU core.

    Args:
        num_eps_to_collect : how many episodes this worker should run (e.g. 3)
        scout_w            : state-dict of ScoutActor  (CPU tensors, serialisable)
        cmdr_w             : state-dict of CommanderActor
        critic_w           : state-dict of MAPPOCritic
        config             : dict with env/network dimensions and training hyper-params
        batch_start_idx    : global episode index where this batch starts
                             (used to seed the env reset deterministically)
    Returns:
        out_scout   : dict of tensors for scout PPO update
        out_cmdr    : dict of tensors for commander PPO update
        out_critic  : dict of tensors for critic PPO update
        out_init_h  : initial hidden states (one per episode) for each network
        worker_total_rewards : list[float] — total reward per episode
        worker_lifespans     : list[float] — average agent lifespan per episode
    """
    # Imports must be repeated inside the function because each worker runs in a
    # completely separate Python process (spawned by ProcessPoolExecutor).
    # The parent process's global namespace is NOT inherited.
    import torch
    import numpy as np
    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, MAPPOCritic

    # ------------------------------------------------------------------
    # 1. Create a local environment instance for this worker
    # ------------------------------------------------------------------
    local_env = DroneFireEnv(num_quads=config['N_QUADS'], num_fixed=config['N_FIXED'], 
                             grid_size_m=config['grid_size_m'], max_steps=config['max_steps'])

    # ------------------------------------------------------------------
    # 2. Rebuild networks on CPU and load the weights sent by the main process
    #    eval() disables dropout/batchnorm randomness — we only do inference here
    # ------------------------------------------------------------------
    local_scout = None
    if config['N_QUADS'] > 0:
        local_scout = ScoutActor(self_state_dim=config['scout_self_dim'], msg_dim=config['scout_msg_dim'], hidden_dim=config['scout_hidden_dim'])
        local_scout.load_state_dict(scout_w); local_scout.eval()

    local_commander = None
    if config['N_FIXED'] > 0:
        local_commander = CommanderActor(self_state_dim=config['fixed_self_dim'], msg_input_dim=config['scout_msg_dim'])
        local_commander.load_state_dict(cmdr_w); local_commander.eval()

    local_critic = MAPPOCritic(config['global_state_dim'])
    local_critic.load_state_dict(critic_w); local_critic.eval()
 
    # ------------------------------------------------------------------
    # 3. Pre-allocated data buffers  (one per network type)
    #
    # We keep scout / commander / critic data in SEPARATE buffers from the start.
    # This avoids expensive agent-type filtering on the main process later —
    # the main process just calls torch.cat() on each buffer directly.
    #
    # scout_buf  — observations and actions for the quadrotor scouts
    # cmdr_buf   — observations and actions for the fixed-wing commanders
    # critic_buf — global state + returns + baseline values (shared critic)
    # ------------------------------------------------------------------
    scout_buf = {k: [] for k in ["maps", "self_states", "neighbor_states", "neighbor_masks",
                                  "actions", "logprobs", "returns"]}
    cmdr_buf  = {k: [] for k in ["fixed_states", "incoming_msgs", "msg_masks",
                                  "actions", "logprobs", "returns"]}
    critic_buf = {k: [] for k in ["g_states", "returns", "values"]}

    # Initial hidden states — one tensor per episode (not per step).
    # The LSTM in each network is stateful across the episode, but for the PPO
    # update we re-run the full sequence starting from the episode's h_0.
    # Shape per entry: [1, 1, hidden_dim]  (batch=1, num_layers=1, hidden_dim)
    scout_init_h_list        = []
    cmdr_init_h_list         = []
    critic_init_h_scout_list = []
    critic_init_h_cmdr_list  = []

    worker_total_rewards = []   # total reward per completed episode
    worker_lifespans     = []   # average agent lifespan (steps) per episode

    # ------------------------------------------------------------------
    # Dummy (padding) tensors for dead agents
    # When an agent dies mid-episode we still need to append something to the
    # buffer so that tensor shapes stay consistent across all time steps.
    # ------------------------------------------------------------------
    d_map      = torch.zeros(1, 1, 32, 32)
    d_scout_self = torch.zeros(1, config['scout_self_dim'])
    d_neigh_s  = torch.zeros(1, max(1, config['N_QUADS']-1), 3)
    d_neigh_m  = torch.ones(1, max(1, config['N_QUADS']-1), dtype=torch.bool)   # mask=True → ignore slot
    d_cmd_self = torch.zeros(1, config['fixed_self_dim'] if config['fixed_self_dim'] > 0 else 1)
    d_msgs     = torch.zeros(1, max(1, config['N_QUADS']), config['scout_msg_dim'])
    d_msg_m    = torch.ones(1, max(1, config['N_QUADS']), dtype=torch.bool)
 
    # ======================================================================
    # EPISODE COLLECTION LOOP
    # ======================================================================
    for ep_offset in range(num_eps_to_collect):

        # Reset the environment; epizode_number seeds random fire/drone placement
        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_offset)

        # Reset LSTM hidden states at the start of every episode.
        # Hidden state shape: [num_layers=1, batch=1, hidden_dim]
        scout_h = {q: torch.zeros(1, 1, config['scout_hidden_dim']) for q in local_env.quad_agents}
        cmdr_h  = {f: torch.zeros(1, 1, 128)                        for f in local_env.fixed_agents}
        crit_h  = {a: torch.zeros(1, 1, 128)                        for a in local_env.possible_agents}

        # Save h_0 for each network — these will be used as the starting hidden
        # state when the main process re-runs the sequence during PPO update.
        if config['N_QUADS'] > 0:
            scout_init_h_list.append(scout_h[local_env.quad_agents[0]].clone())
            critic_init_h_scout_list.append(crit_h[local_env.quad_agents[0]].clone())
        if config['N_FIXED'] > 0:
            cmdr_init_h_list.append(cmdr_h[local_env.fixed_agents[0]].clone())
            critic_init_h_cmdr_list.append(crit_h[local_env.fixed_agents[0]].clone())

        # ep_rollouts stores per-agent, per-step data so we can compute
        # discounted returns backwards at the end of the episode.
        ep_rollouts     = {a: [] for a in local_env.possible_agents}
        agent_lifespans = {a: config['max_steps'] for a in local_env.possible_agents}
        episode_reward  = 0.0

        # ------------------------------------------------------------------
        # STEP LOOP  — must always run the full max_steps because discounted
        # returns are computed backwards over the entire episode length.
        # ------------------------------------------------------------------
        for step in range(config['max_steps']):
            global_state = local_env.state()                         # full global state (used by critic)
            g_tensor     = torch.FloatTensor(global_state).unsqueeze(0)
            actions      = {}   # action dict fed to env.step()
            step_results = {}   # temporary store for this timestep's data
 
            # ----------------------------------------------------------
            # SCOUTS (quadrotors)
            # Each scout observes its local map, its own state, and its
            # neighbours' states. It outputs an action distribution, a
            # message vector (broadcast to the commander), and the next
            # LSTM hidden state.
            # ----------------------------------------------------------
            scout_msgs_list, scout_mask_list = [], []
            for q in local_env.quad_agents:
                if q in local_env.agents:                     # agent is still alive
                    with torch.no_grad():
                        l_map = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0)
                        s_st  = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0)
                        n_s   = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0)
                        n_m   = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0)
                        dist, msg, h_out = local_scout(l_map, s_st, n_s, n_m, scout_h[q])
                        val, c_h_out     = local_critic(g_tensor, crit_h[q])
                        act = dist.sample()
                    # Store everything needed for the PPO update later
                    step_results[q] = {"type": "scout", "map": l_map, "self": s_st, "n_s": n_s, "n_m": n_m,
                                       "h": scout_h[q], "act": act, "lp": dist.log_prob(act).sum(1),
                                       "val": val, "gs": g_tensor, "ch": crit_h[q]}
                    scout_h[q], crit_h[q] = h_out, c_h_out   # advance hidden states
                    actions[q] = act.squeeze(0).numpy()
                    scout_msgs_list.append(msg)
                    scout_mask_list.append(False)             # False = valid message slot
                else:
                    # Agent is dead — send zero message and mark slot as padding
                    scout_msgs_list.append(torch.zeros(1, config['scout_msg_dim']))
                    scout_mask_list.append(True)              # True = masked / ignored
                    step_results[q] = {"type": "dead_scout", "gs": g_tensor}
 
            # ----------------------------------------------------------
            # COMMANDER (fixed-wing aircraft)
            # The commander receives all scout messages (via attention) and
            # its own state observation, then outputs a continuous action.
            # msgs_t shape: [1, N_QUADS, scout_msg_dim]
            # msgs_m shape: [1, N_QUADS]  — True means the slot is masked out
            # ----------------------------------------------------------
            msgs_t = torch.stack(scout_msgs_list, dim=1) if scout_msgs_list else d_msgs
            msgs_m = torch.tensor(scout_mask_list).unsqueeze(0) if scout_mask_list else d_msg_m
            for f in local_env.fixed_agents:
                if f in local_env.agents:
                    with torch.no_grad():
                        s_st = torch.FloatTensor(obs[f]["self_state"]).unsqueeze(0)
                        dist, _, h_out   = local_commander(s_st, msgs_t, msgs_m, cmdr_h[f])
                        val, c_h_out     = local_critic(g_tensor, crit_h[f])
                        act = dist.sample()
                    step_results[f] = {"type": "commander", "self": s_st, "msgs": msgs_t, "m_m": msgs_m,
                                       "h": cmdr_h[f], "act": act, "lp": dist.log_prob(act).sum(1),
                                       "val": val, "gs": g_tensor, "ch": crit_h[f]}
                    cmdr_h[f], crit_h[f] = h_out, c_h_out
                    actions[f] = act.squeeze(0).numpy()
                else:
                    step_results[f] = {"type": "dead_cmdr", "gs": g_tensor}
 
            # ----------------------------------------------------------
            # ENVIRONMENT STEP
            # ----------------------------------------------------------
            if local_env.agents:   # at least one agent is still alive
                obs, rewards, terminations, _, _ = local_env.step(actions)
                for a in local_env.possible_agents:
                    r = rewards.get(a, 0.0)
                    episode_reward += r
                    step_results[a]["reward"] = r
                    # Record lifespan = first step at which the agent terminated
                    if terminations.get(a, False) and agent_lifespans[a] == config['max_steps']:
                        agent_lifespans[a] = step
            else:
                # All agents are dead — pad remaining steps with zero reward
                for a in local_env.possible_agents:
                    step_results[a]["reward"] = 0.0

            for a in local_env.possible_agents:
                ep_rollouts[a].append(step_results[a])

        # ------------------------------------------------------------------
        # COMPUTE DISCOUNTED RETURNS  (backward pass over the episode)
        #
        # G_t = r_t + γ * G_{t+1}
        # We iterate from the last step back to the first, accumulating the
        # discounted sum.  This is the target that the critic tries to predict.
        # ------------------------------------------------------------------
        for a in local_env.possible_agents:
            disc_sum = 0
            for i in reversed(range(config['max_steps'])):
                ep_rollouts[a][i]["ret"] = ep_rollouts[a][i]["reward"] + config['gamma'] * disc_sum
                disc_sum = ep_rollouts[a][i]["ret"]

        # ------------------------------------------------------------------
        # PACK EPISODE DATA INTO PRE-SPLIT BUFFERS
        #
        # We iterate step-by-step (not agent-by-agent) so we can build
        # the interleaved critic buffer: [scout_0, cmdr_0, scout_1, cmdr_1, ...]
        # This interleaving is expected by the PPO update later.
        # ------------------------------------------------------------------
        for step_idx in range(config['max_steps']):

            # CRITIC buffer — one entry per agent per step (interleaved order)
            for a_name in local_env.possible_agents:
                d = ep_rollouts[a_name][step_idx]
                critic_buf["g_states"].append(d["gs"])
                critic_buf["returns"].append(d["ret"])
                # val may be absent for dead agents — fall back to 0
                critic_buf["values"].append(d.get("val", torch.tensor([[0.0]])))

            # SCOUT buffer — one entry per scout per step
            for q in local_env.quad_agents:
                d = ep_rollouts[q][step_idx]
                if d["type"] == "scout":        # alive: use real observations
                    scout_buf["maps"].append(d["map"])
                    scout_buf["self_states"].append(d["self"])
                    scout_buf["neighbor_states"].append(d["n_s"])
                    scout_buf["neighbor_masks"].append(d["n_m"])
                    scout_buf["actions"].append(d["act"])
                    scout_buf["logprobs"].append(d["lp"])
                else:                             # dead: fill with zero padding
                    scout_buf["maps"].append(d_map)
                    scout_buf["self_states"].append(d_scout_self)
                    scout_buf["neighbor_states"].append(d_neigh_s)
                    scout_buf["neighbor_masks"].append(d_neigh_m)
                    scout_buf["actions"].append(torch.zeros(1, 4))
                    scout_buf["logprobs"].append(torch.tensor([0.0]))
                scout_buf["returns"].append(d["ret"])   # always add return

            # COMMANDER buffer — one entry per fixed-wing per step
            for f in local_env.fixed_agents:
                d = ep_rollouts[f][step_idx]
                if d["type"] == "commander":    # alive: use real observations
                    cmdr_buf["fixed_states"].append(d["self"])
                    cmdr_buf["incoming_msgs"].append(d["msgs"])
                    cmdr_buf["msg_masks"].append(d["m_m"])
                    cmdr_buf["actions"].append(d["act"])
                    cmdr_buf["logprobs"].append(d["lp"])
                else:                             # dead: fill with zero padding
                    cmdr_buf["fixed_states"].append(d_cmd_self)
                    cmdr_buf["incoming_msgs"].append(d_msgs)
                    cmdr_buf["msg_masks"].append(d_msg_m)
                    cmdr_buf["actions"].append(torch.zeros(1, 4))
                    cmdr_buf["logprobs"].append(torch.tensor([0.0]))
                cmdr_buf["returns"].append(d["ret"])
 
        worker_total_rewards.append(episode_reward)
        worker_lifespans.append(np.mean(list(agent_lifespans.values())))

    # ======================================================================
    # FINALISE AND RETURN WORKER DATA
    # ======================================================================

    # Stop the simulator cleanly before the process exits
    local_env.sim.stop_simulation()

    # Concatenate all list-of-tensors into single tensors (done here on the
    # worker CPU so the main process can just call torch.cat on worker results).
    def cat_buf(buf):
        result = {}
        for k, v in buf.items():
            if k == "returns":
                # Returns are plain floats — wrap them in a tensor
                result[k] = torch.tensor(v, dtype=torch.float32)
            else:
                result[k] = torch.cat(v)
        return result

    out_scout  = cat_buf(scout_buf)
    out_cmdr   = cat_buf(cmdr_buf)
    out_critic = cat_buf(critic_buf)

    # Stack initial hidden states across episodes:
    # list of [1, 1, hidden_dim]  →  [num_eps, 1, hidden_dim]
    out_init_h = {
        "scout":        torch.cat(scout_init_h_list,        dim=0) if scout_init_h_list        else None,
        "cmdr":         torch.cat(cmdr_init_h_list,         dim=0) if cmdr_init_h_list         else None,
        "critic_scout": torch.cat(critic_init_h_scout_list, dim=0) if critic_init_h_scout_list else None,
        "critic_cmdr":  torch.cat(critic_init_h_cmdr_list,  dim=0) if critic_init_h_cmdr_list  else None,
    }

    return out_scout, out_cmdr, out_critic, out_init_h, worker_total_rewards, worker_lifespans



# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
# High-level algorithm: Heterogeneous MAPPO with parallel rollout collection
#
#   ┌─ Main process (GPU) ──────────────────────────────────────────────────┐
#   │  for each batch:                                                      │
#   │    1. Broadcast current network weights (CPU copies) to all workers   │
#   │    2. Workers collect episodes in parallel                            │
#   │    3. Gather & concatenate worker buffers                             │
#   │    4. Run PPO update (gradient descent) on GPU                        │
#   │    5. Save best model checkpoint if avg reward improved               │
#   └───────────────────────────────────────────────────────────────────────┘
# ============================================================================
def train():
    print("🚀 Starting Heterogeneous MAPPO Training (PARALLEL)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================================
    # 1. HYPERPARAMETERS
    # ==========================================================================
    num_episodes = 25000       # total episodes to train for
    max_steps    = 2500        # maximum timesteps per episode
    learning_rate = 3e-4       # base learning rate (Adam)
    gamma         = 0.99       # discount factor — how much future rewards matter
                               #   0.99  = cares a lot about long-term rewards
                               #   0.9   = cares mostly about near-future rewards
    clip_coef     = 0.2        # PPO clipping range — prevents the new policy from
                               # deviating too far from the old one in a single update
    update_epochs = 8          # how many gradient passes over each collected batch
    num_workers   = 20         # parallel CPU workers for data collection
    eps_per_worker = 3         # episodes collected by each worker per batch
    episodes_per_batch = num_workers * eps_per_worker   # = 60 episodes per batch

    # Separate learning rates per network:
    # Scout has already been pre-trained → fine-tune with a very small LR
    # so we don't overwrite what it has already learned.
    lr_commander       = learning_rate
    lr_critic          = learning_rate
    lr_scout_fine_tune = 5e-5

    # ==========================================================================
    # 2. TEAM CONFIGURATION & NETWORK DIMENSIONS
    # ==========================================================================
    N_QUADS = 1   # number of quadrotor scouts
    N_FIXED = 1   # number of fixed-wing commanders

    # Spin up a temporary environment just to read observation/state space sizes.
    # We cannot hard-code them because they depend on N_QUADS, N_FIXED, and
    # internal env logic.
    temp_env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=500.0, max_steps=max_steps)
    if N_QUADS > 0:
        scout_self_dim = temp_env.observation_space("quad_0")["self_state"].shape[0]
    else:
        scout_self_dim = 12   # fallback if no scouts

    scout_msg_dim    = 5    # dimension of the message vector each scout broadcasts
    scout_hidden_dim = 128  # LSTM hidden size for ScoutActor

    if N_FIXED > 0:
        fixed_self_dim = temp_env.observation_space(temp_env.fixed_agents[0])["self_state"].shape[0]
    else:
        fixed_self_dim = 0

    global_state_dim = temp_env.state_space.shape[0]   # input size for the shared critic
    # temp_env.sim.stop_simulation()  # uncomment if you want to free the sim port immediately

    # Pack all dims into a config dict — this is what gets sent to each worker
    worker_config = {
        'N_QUADS': N_QUADS, 'N_FIXED': N_FIXED, 'grid_size_m': 500.0, 'max_steps': max_steps,
        'scout_self_dim': scout_self_dim, 'scout_msg_dim': scout_msg_dim, 'scout_hidden_dim': scout_hidden_dim,
        'fixed_self_dim': fixed_self_dim, 'global_state_dim': global_state_dim, 'gamma': gamma
    }

    # ==========================================================================
    # 3. NETWORK INITIALISATION  (all networks live on GPU during training)
    # ==========================================================================
    # ScoutActor — processes local map, own state, neighbour states via CNN+LSTM
    if N_QUADS > 0:
        scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim).to(device)
        path_to_old_model = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/scout_best.pt"
        if os.path.exists(path_to_old_model):
            print(f"📥 Loading pre-trained scout model from {path_to_old_model}")
            scout_actor.load_state_dict(torch.load(path_to_old_model, map_location=device), strict=False)
        else:
            print(f"⚠️  No pre-trained scout model found — training from scratch.")
    else:
        scout_actor = None

    # CommanderActor — processes own state + scout messages (attention) via LSTM
    if N_FIXED > 0:
        commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=scout_msg_dim).to(device)
        path_to_old_model = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/commander_best.pt"
        if os.path.exists(path_to_old_model):
            print(f"📥 Loading pre-trained commander model from {path_to_old_model}")
            commander_actor.load_state_dict(torch.load(path_to_old_model, map_location=device), strict=False)
        else:
            print(f"⚠️  No pre-trained commander model found — training from scratch.")
    else:
        commander_actor = None

    # MAPPOCritic — shared value network, takes global state as input
    critic = MAPPOCritic(global_state_dim).to(device)

    # Adam optimiser with per-network learning rates
    # (using parameter groups so we can fine-tune scout at a lower LR)
    optim_groups = [{"params": critic.parameters(), "lr": lr_critic}]
    if scout_actor:     optim_groups.append({"params": scout_actor.parameters(),     "lr": lr_scout_fine_tune})
    if commander_actor: optim_groups.append({"params": commander_actor.parameters(), "lr": lr_commander})
    optimizer = optim.Adam(optim_groups)

    # ==========================================================================
    # 4. TRAINING HISTORY  (for plotting)
    # ==========================================================================
    episode_rewards_history = []   # total reward per episode
    loss_history            = []   # combined PPO loss per update epoch
    entropy_history         = []   # policy entropy (exploration measure)
    v_loss_history          = []   # critic (value) loss per update epoch
    lifespan_history        = []   # average agent lifespan per episode

    os.makedirs("saved_models", exist_ok=True)
    best_avg_reward = -1000.0
    episodes_played = 0
    num_batches = num_episodes // episodes_per_batch

    # ==========================================================================
    # 5. MAIN PARALLEL TRAINING LOOP
    # ==========================================================================
    # The ProcessPoolExecutor keeps a persistent pool of worker processes alive
    # for the entire training run — no spawn overhead per batch.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_idx in range(1, num_batches + 1):

            # --------------------------------------------------------------
            # 5a. Reset per-batch aggregation buffers
            # These must be re-created every batch to stay empty.
            # --------------------------------------------------------------
            batch_scout  = {k: [] for k in ["maps", "self_states", "neighbor_states", "neighbor_masks",
                                             "actions", "logprobs", "returns"]}
            batch_cmdr   = {k: [] for k in ["fixed_states", "incoming_msgs", "msg_masks",
                                             "actions", "logprobs", "returns"]}
            batch_critic = {k: [] for k in ["g_states", "returns", "values"]}
            batch_init_h = {"scout": [], "cmdr": [], "critic_scout": [], "critic_cmdr": []}

            # --------------------------------------------------------------
            # 5b. Snapshot current network weights as CPU tensors
            # Workers always run on CPU; GPU tensors cannot be pickled for IPC.
            # --------------------------------------------------------------
            scout_w  = {k: v.cpu() for k, v in scout_actor.state_dict().items()}      if scout_actor     else None
            cmdr_w   = {k: v.cpu() for k, v in commander_actor.state_dict().items()}  if commander_actor else None
            critic_w = {k: v.cpu() for k, v in critic.state_dict().items()}

            # --------------------------------------------------------------
            # 5c. Dispatch rollout jobs to all workers simultaneously
            # Each worker gets the same weights and config but runs independently.
            # --------------------------------------------------------------
            futures = []
            for i in range(num_workers):
                futures.append(executor.submit(collect_episodes_per_worker, eps_per_worker,
                                               scout_w, cmdr_w, critic_w, worker_config, episodes_played))

            # --------------------------------------------------------------
            # 5d. Collect results from all workers
            # future.result() blocks until the worker finishes.
            # --------------------------------------------------------------
            batch_rewards = []
            for future in futures:
                w_scout, w_cmdr, w_critic, w_init_h, w_rewards, w_lifespans = future.result()
                batch_rewards.extend(w_rewards)
                lifespan_history.extend(w_lifespans)
                episode_rewards_history.extend(w_rewards)
                episodes_played += len(w_rewards)

                # Append each worker's tensors into the batch-level lists;
                # torch.cat() is called later (once, efficiently)
                for k in batch_scout:  batch_scout[k].append(w_scout[k])
                for k in batch_cmdr:   batch_cmdr[k].append(w_cmdr[k])
                for k in batch_critic: batch_critic[k].append(w_critic[k])
                for k in batch_init_h:
                    if w_init_h[k] is not None:
                        batch_init_h[k].append(w_init_h[k])

            # Progress log (one line per batch instead of per episode)
            avg_batch  = np.mean(batch_rewards)
            avg_reward = np.mean(episode_rewards_history[-15:]) if len(episode_rewards_history) >= 15 else np.mean(episode_rewards_history)
            print(f"Batch {batch_idx:04d} (Ep {episodes_played:04d}) | Avg Batch Reward: {avg_batch:.2f} | Rolling avg (15): {avg_reward:.2f}")

            # --------------------------------------------------------------
            # 5e. Save best checkpoint
            # We only update if we have enough episodes for a stable average.
            # --------------------------------------------------------------
            if episodes_played >= 15 and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                if scout_actor:
                    torch.save(scout_actor.state_dict(), "saved_models/scout_best.pt")
                if commander_actor:
                    torch.save(commander_actor.state_dict(), "saved_models/commander_best.pt")
                print(f"⭐ New best model saved! (Rolling avg: {best_avg_reward:.2f})")

            # ==============================================================
            # 5f. PPO UPDATE  (runs on GPU)
            # ==============================================================
            num_agents = N_QUADS + N_FIXED
            episodes   = episodes_per_batch

            if len(batch_critic["returns"]) > 0:

                # --- Concatenate all worker tensors into single GPU tensors ---
                # Each batch_X[key] is a list of tensors [one per worker];
                # torch.cat merges them along the first (batch) dimension.

                # Scout tensors  [episodes * max_steps, ...]
                s_maps     = torch.cat(batch_scout["maps"]).to(device)
                s_self     = torch.cat(batch_scout["self_states"]).to(device)
                s_neigh_s  = torch.cat(batch_scout["neighbor_states"]).to(device)
                s_neigh_m  = torch.cat(batch_scout["neighbor_masks"]).to(device)
                s_actions  = torch.cat(batch_scout["actions"]).to(device)
                s_logprobs = torch.cat(batch_scout["logprobs"]).to(device)    # log π_old(a|s)
                s_returns  = torch.cat(batch_scout["returns"]).to(device)

                # Commander tensors  [episodes * max_steps, ...]
                c_fixed    = torch.cat(batch_cmdr["fixed_states"]).to(device)
                c_msgs     = torch.cat(batch_cmdr["incoming_msgs"]).to(device)
                c_msg_m    = torch.cat(batch_cmdr["msg_masks"]).to(device)
                c_actions  = torch.cat(batch_cmdr["actions"]).to(device)
                c_logprobs = torch.cat(batch_cmdr["logprobs"]).to(device)
                c_returns  = torch.cat(batch_cmdr["returns"]).to(device)

                # Critic tensors  [episodes * max_steps * num_agents, ...]
                cr_g_states = torch.cat(batch_critic["g_states"]).to(device)
                cr_returns  = torch.cat(batch_critic["returns"]).to(device)
                cr_values   = torch.cat(batch_critic["values"]).to(device)

                # --- Reconstruct LSTM initial hidden states ---
                # Workers stored per-episode h_0 as [1, 1, hidden_dim].
                # We stack all episodes and reshape to [1, episodes, hidden_dim]
                # which is the format expected by PyTorch's LSTM.
                h_scout        = torch.cat(batch_init_h["scout"],        dim=0).squeeze(1).unsqueeze(0).to(device)
                h_cmdr         = torch.cat(batch_init_h["cmdr"],         dim=0).squeeze(1).unsqueeze(0).to(device)
                h_critic_scout = torch.cat(batch_init_h["critic_scout"], dim=0).squeeze(1).unsqueeze(0).to(device)
                h_critic_cmdr  = torch.cat(batch_init_h["critic_cmdr"],  dim=0).squeeze(1).unsqueeze(0).to(device)

                # Critic hidden: interleave scout/cmdr hidden states so they
                # match the interleaved order in the critic buffer:
                # [scout_ep0, cmdr_ep0, scout_ep1, cmdr_ep1, ...]
                # Final shape: [1, episodes * num_agents, hidden_dim]
                h_critic = torch.stack([h_critic_scout.squeeze(0), h_critic_cmdr.squeeze(0)], dim=1)
                h_critic = h_critic.reshape(1, episodes * num_agents, -1)

                print(f"🛠️  Running PPO update ({len(cr_returns)} samples)...")

                # --- Compute advantages ---
                # Advantage = how much better the actual return was vs. what
                # the critic predicted.  Normalise per-batch for stability.
                cr_adv = cr_returns.unsqueeze(1) - cr_values.detach()
                cr_adv = (cr_adv - cr_adv.mean()) / (cr_adv.std() + 1e-8)

                # The critic buffer is interleaved: [scout_0, cmdr_0, scout_1, ...]
                # Slice out per-agent advantages by agent index within each step.
                s_adv = cr_adv.view(episodes, max_steps, num_agents)[:, :, 0].reshape(-1)  # scout slice
                c_adv = cr_adv.view(episodes, max_steps, num_agents)[:, :, 1].reshape(-1)  # commander slice

                def to_seq_agent(t, num_traj):
                    """Reshape flat [num_traj*steps, ...] → sequential [num_traj, steps, ...]
                    needed because the LSTM expects (batch, seq, features)."""
                    return t.view(num_traj, max_steps, *t.shape[1:])

                # --- Gradient update loop (update_epochs passes over the same data) ---
                for epoch in range(update_epochs):
                    entropy_sum = torch.tensor(0.0, device=device)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # SCOUT POLICY LOSS  (PPO-clip objective)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Re-run the scout network over the collected sequences to
                    # get NEW log-probabilities for the OLD actions.
                    if scout_actor is not None:
                        dist, _, _ = scout_actor(
                            to_seq_agent(s_maps,    episodes),
                            to_seq_agent(s_self,    episodes),
                            to_seq_agent(s_neigh_s, episodes),
                            to_seq_agent(s_neigh_m, episodes),
                            h_scout
                        )
                        flat_actions_s = s_actions.view(episodes * max_steps, -1)
                        new_lp_s       = dist.log_prob(flat_actions_s).sum(1)   # log π_new(a|s)
                        entropy_sum   += dist.entropy().sum(1).mean()           # encourage exploration

                        # ratio = π_new / π_old  (in log space for numerical stability)
                        ratio_s       = torch.exp(new_lp_s - s_logprobs)
                        pg1_s         = -s_adv * ratio_s
                        pg2_s         = -s_adv * torch.clamp(ratio_s, 1 - clip_coef, 1 + clip_coef)
                        policy_loss_s = torch.max(pg1_s, pg2_s).mean()   # pessimistic clip
                    else:
                        policy_loss_s = torch.tensor(0.0, device=device)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # COMMANDER POLICY LOSS  (same PPO-clip logic)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if commander_actor is not None:
                        dist, _, _ = commander_actor(
                            to_seq_agent(c_fixed, episodes),
                            to_seq_agent(c_msgs,  episodes),
                            to_seq_agent(c_msg_m, episodes),
                            h_cmdr
                        )
                        flat_actions_c = c_actions.view(episodes * max_steps, -1)
                        new_lp_c       = dist.log_prob(flat_actions_c).sum(1)
                        entropy_sum   += dist.entropy().sum(1).mean()

                        ratio_c       = torch.exp(new_lp_c - c_logprobs)
                        pg1_c         = -c_adv * ratio_c
                        pg2_c         = -c_adv * torch.clamp(ratio_c, 1 - clip_coef, 1 + clip_coef)
                        policy_loss_c = torch.max(pg1_c, pg2_c).mean()
                    else:
                        policy_loss_c = torch.tensor(0.0, device=device)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # CRITIC (VALUE) LOSS
                    # cr_g_states is currently flat: [eps*steps*agents, gs_dim]
                    # The critic LSTM needs shape:   [eps*agents, steps, gs_dim]
                    # Reshape: separate episodes/steps/agents, transpose steps↔agents,
                    # then merge episodes and agents into the batch dimension.
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    cr_g_seq = (cr_g_states
                                .view(episodes, max_steps, num_agents, -1)
                                .transpose(1, 2)                   # → [ep, agents, steps, gs_dim]
                                .reshape(episodes * num_agents, max_steps, -1))
                    new_values, _ = critic(cr_g_seq, h_critic)
                    value_loss    = nn.MSELoss()(new_values, cr_returns.unsqueeze(1))

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # COMBINED LOSS  (standard MAPPO objective)
                    # = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    # The entropy bonus keeps the policy from collapsing to
                    # a single deterministic action too early.
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    policy_loss = policy_loss_s + policy_loss_c
                    loss        = policy_loss + 0.5 * value_loss - 0.01 * entropy_sum

                    loss_history.append(loss.item())
                    v_loss_history.append(value_loss.item())
                    entropy_history.append(entropy_sum.detach().item())

                    # Gradient step
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients to prevent exploding gradients
                    params = list(critic.parameters())
                    if scout_actor:     params += list(scout_actor.parameters())
                    if commander_actor: params += list(commander_actor.parameters())
                    nn.utils.clip_grad_norm_(params, max_norm=0.5)
                    optimizer.step()

            # ==============================================================
            # 5g. PERIODIC SNAPSHOT & TRAINING PLOT  (every 10 batches)
            # ==============================================================
            if batch_idx % 10 == 0:
                # Save a timestamped checkpoint (useful for post-training analysis)
                if scout_actor:
                    torch.save(scout_actor.state_dict(),     f"saved_models/scout_ep{episodes_played}.pt")
                if commander_actor:
                    torch.save(commander_actor.state_dict(), f"saved_models/commander_ep{episodes_played}.pt")

                # --- 4-panel training dashboard ---
                plt.figure(figsize=(20, 5))

                # Panel 1: Episode rewards over time
                # Raw rewards are noisy; 20-episode moving average shows the trend.
                plt.subplot(1, 4, 1)
                plt.plot(episode_rewards_history, label="Reward", alpha=0.3, color='green')
                if len(episode_rewards_history) > 20:
                    plt.plot(np.convolve(episode_rewards_history, np.ones(20)/20, mode='valid'),
                             label="MA 20", color='darkgreen')
                plt.title("Reward over episodes")
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Panel 2: Loss curves (log scale because values span many orders)
                plt.subplot(1, 4, 2)
                plt.plot(loss_history,   label="Total Loss",         color='red',  alpha=0.5)
                plt.plot(v_loss_history, label="Value Loss (Critic)", color='blue', alpha=0.5)
                plt.title("Loss (log scale)")
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Panel 3: Policy entropy — measures how exploratory the policy is.
                # Should start high (random) and slowly decrease as policy converges.
                plt.subplot(1, 4, 3)
                if entropy_history:
                    plt.plot(entropy_history, color='purple')
                    plt.title("Policy Entropy (exploration)")
                else:
                    plt.text(0.5, 0.5, 'No entropy data yet', ha='center')
                plt.grid(True, alpha=0.3)

                # Panel 4: Average agent lifespan — how long agents survive per episode.
                # Increasing lifespan generally means agents are getting better.
                plt.subplot(1, 4, 4)
                plt.plot(lifespan_history, color='orange')
                plt.title("Avg agent lifespan (steps)")

                plt.tight_layout()
                plt.savefig("final_training_plot.png")
                plt.close()

if __name__ == "__main__":
    train()