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
            if len(v) == 0:
                result[k] = torch.tensor([], dtype=torch.float32)
                continue
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
