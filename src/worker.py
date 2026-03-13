# ============================================================================
# WORKER MODULE  --  parallel rollout collection on CPU
# ============================================================================
# WHY A SEPARATE WORKER PROCESS?
# --------------------------------
# Neural network training has two expensive phases:
#   1. Data collection: run the policy in the environment to gather (s, a, r)
#   2. Gradient update: backprop through the network using the collected data
#
# Phase 1 is CPU-bound (PyBullet physics simulation, environment logic).
# Phase 2 is GPU-bound (matrix multiplications, backprop).
#
# These two phases use completely different hardware, so they can overlap.
# The main process runs Phase 2 on the GPU while Phase 1 runs in parallel
# across N CPU workers.  This gives near-linear data-collection speedup.
#
# PROCESS ISOLATION:
# ProcessPoolExecutor uses *fork* or *spawn* to create fully independent
# Python interpreter processes.  Each worker has its own memory space,
# its own copy of the environment, and its own copy of the networks.
# Workers communicate with the main process only through serialised return
# values (the tensor dictionaries returned at the end of this function).
# ============================================================================


def collect_episodes_per_worker(num_eps_to_collect, scout_w, cmdr_w, critic_w, config, batch_start_idx):
    """
    Collects rollout data for several full episodes on a single CPU core.

    This function is the ENTIRE body of work for one worker process.
    It: creates its own environment, rebuilds all three networks from
    the weight snapshots sent by the main process, runs episodes, computes
    discounted returns, and returns packed tensors ready for the PPO update.

    WHY WEIGHT SNAPSHOTS (not the original network objects)?
    Python's multiprocessing cannot serialise PyTorch nn.Module objects that
    live on GPU.  Instead, the main process extracts state_dicts (plain
    Python dicts of CPU tensors), which pickle cleanly, and workers
    reconstruct identical networks on their own CPU.

    Parameters
    ----------
    num_eps_to_collect : int
        How many full episodes this worker should run (e.g. 3).
    scout_w            : dict or None
        state_dict of ScoutActor (CPU tensors).  None if N_QUADS == 0.
    cmdr_w             : dict or None
        state_dict of CommanderActor.  None if N_FIXED == 0.
    critic_w           : dict
        state_dict of MAPPOCritic (always present).
    config             : dict
        Environment dimensions and training hyperparameters:
        N_QUADS, N_FIXED, grid_size_m, max_steps, scout_self_dim,
        scout_msg_dim, scout_hidden_dim, fixed_self_dim,
        global_state_dim, gamma.
    batch_start_idx    : int
        Global episode counter at the start of this batch.  Used to seed
        the environment reset so different workers explore different
        fire/drone configurations.

    Returns
    -------
    out_scout   : dict[str, Tensor]   -- obs/actions/logprobs/returns for scouts
    out_cmdr    : dict[str, Tensor]   -- same for commanders
    out_critic  : dict[str, Tensor]   -- global states / returns / values
    out_init_h  : dict[str, Tensor]   -- per-episode initial GRU hidden states
    worker_total_rewards : list[float]  -- total reward per episode
    worker_lifespans     : list[float]  -- average agent lifespan per episode
    """
    # -------------------------------------------------------------------------
    # IMPORTS INSIDE THE FUNCTION
    # -------------------------------------------------------------------------
    # In Python's "spawn" multiprocessing mode (default on Linux with CUDA),
    # each worker process starts as a BLANK Python interpreter -- it does NOT
    # inherit the parent's globals or imported modules.  That is why every
    # import the worker needs must be repeated here, inside the function body,
    # not at the top of the file.
    import torch
    import numpy as np
    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, MAPPOCritic

    # =========================================================================
    # 1. CREATE A LOCAL ENVIRONMENT INSTANCE
    # =========================================================================
    # Each worker gets its own, completely independent simulator.
    # They never share state -- they just use the same network weights
    # (which were snapshotted by the main process before dispatching workers).
    local_env = DroneFireEnv(num_quads=config['N_QUADS'], num_fixed=config['N_FIXED'],
                             grid_size_m=config['grid_size_m'], max_steps=config['max_steps'])

    # =========================================================================
    # 2. REBUILD NETWORKS ON CPU + LOAD WEIGHTS
    # =========================================================================
    # We reconstruct the exact same network architecture as the main process
    # and fill it with the weight snapshot (state_dict) that was pickled and
    # sent over the IPC channel.
    #
    # .eval() switches the network from training mode to inference mode:
    #   - Dropout layers become pass-through (no random neuron silencing).
    #   - BatchNorm uses stored running statistics instead of batch statistics.
    # Both effects are important for consistent, reproducible action selection.
    # Workers NEVER call .backward() -- gradients are only computed on the GPU
    # in the main process.
    local_scout = None
    if config['N_QUADS'] > 0:
        local_scout = ScoutActor(self_state_dim=config['scout_self_dim'],
                                 msg_dim=config['scout_msg_dim'],
                                 hidden_dim=config['scout_hidden_dim'])
        local_scout.load_state_dict(scout_w)
        local_scout.eval()

    local_commander = None
    if config['N_FIXED'] > 0:
        local_commander = CommanderActor(self_state_dim=config['fixed_self_dim'],
                                         msg_input_dim=config['scout_msg_dim'])
        local_commander.load_state_dict(cmdr_w)
        local_commander.eval()

    local_critic = MAPPOCritic(config['global_state_dim'])
    local_critic.load_state_dict(critic_w)
    local_critic.eval()

    # =========================================================================
    # 3. PRE-ALLOCATE SEPARATE DATA BUFFERS
    # =========================================================================
    # We maintain THREE separate buffer dicts from the very start.
    # This is a deliberate design choice:
    #
    # WHY NOT ONE BIG BUFFER?
    # The PPO update for scouts, commanders, and the critic uses completely
    # different input fields.  If we mixed everything into one buffer we would
    # need to filter / split it on the main process -- expensive and error-prone.
    # Pre-splitting here is cheap (just list appends) and keeps the main process
    # simple: it just calls torch.cat() on each sub-dict directly.
    #
    # scout_buf  : per-step observations and actions for quadrotor scouts
    # cmdr_buf   : per-step observations and actions for fixed-wing commanders
    # critic_buf : global state + discounted returns + baseline values
    #              (global state is used by the SINGLE shared critic, so all
    #               agents contribute to this buffer -- see interleaving below)
    scout_buf = {k: [] for k in ["maps", "self_states", "neighbor_states", "neighbor_masks",
                                  "actions", "logprobs", "returns"]}
    cmdr_buf  = {k: [] for k in ["fixed_states", "incoming_msgs", "msg_masks",
                                  "actions", "logprobs", "returns"]}
    critic_buf = {k: [] for k in ["g_states", "returns", "values"]}

    # =========================================================================
    # 4. RECORD INITIAL HIDDEN STATES (one per episode, not per step)
    # =========================================================================
    # The GRU carries a hidden state h_t across consecutive time steps.
    # During rollout we advance h_t step-by-step (live).
    # During the PPO update on the main process, we RE-RUN the entire episode
    # from scratch through the network to get updated log-probabilities.
    # For that re-run to start correctly, we must remember the hidden state
    # at the BEGINNING of the episode (h_0), not at every step.
    # Storing h_0 per episode is very cheap: [1, 1, 128] = 128 floats.
    scout_init_h_list        = []   # h_0 for ScoutActor
    cmdr_init_h_list         = []   # h_0 for CommanderActor
    critic_init_h_scout_list = []   # h_0 for critic when processing scout episodes
    critic_init_h_cmdr_list  = []   # h_0 for critic when processing commander episodes

    worker_total_rewards = []   # total undiscounted reward per episode
    worker_lifespans     = []   # average agent lifespan across all agents in episode

    # =========================================================================
    # 5. DUMMY PADDING TENSORS FOR DEAD AGENTS
    # =========================================================================
    # When an agent dies mid-episode, the environment removes it from
    # local_env.agents (the active set).  However, our buffer tensors must have
    # the SAME shape at every time step -- otherwise torch.cat() would fail
    # because tensors with different sizes along a dimension cannot be stacked.
    #
    # Solution: for each dead agent's time step, we append a "dummy" tensor of
    # all zeros (observations) or all True (masks = ignore everything).
    # These entries will appear in the final PPO tensors but carry no gradient
    # or meaningful signal -- they are invisible to the network because:
    #   1. Dead scout entries have neighbor_mask = all True -> attention ignores them.
    #   2. Dead agent returns are 0.0 -> advantage is near-zero -> no gradient signal.
    #
    # max(1, N-1) guards against the degenerate case of 0 or 1 drone where
    # the neighbour dimension would otherwise be empty.
    d_map        = torch.zeros(1, 1, 32, 32)
    d_scout_self = torch.zeros(1, config['scout_self_dim'])
    d_neigh_s    = torch.zeros(1, max(1, config['N_QUADS']-1), 3)
    d_neigh_m    = torch.ones(1, max(1, config['N_QUADS']-1), dtype=torch.bool)   # True = ignore
    d_cmd_self   = torch.zeros(1, config['fixed_self_dim'] if config['fixed_self_dim'] > 0 else 1)
    d_msgs       = torch.zeros(1, max(1, config['N_QUADS']), config['scout_msg_dim'])
    d_msg_m      = torch.ones(1, max(1, config['N_QUADS']), dtype=torch.bool)

    # =========================================================================
    # 6. EPISODE COLLECTION LOOP
    # =========================================================================
    for ep_offset in range(num_eps_to_collect):

        # Reset the environment.
        # `epizode_number` seeds the random number generator inside DroneFireEnv
        # so that every (batch, worker, episode-offset) triple lands a unique
        # fire configuration.  Without this, all workers would explore the exact
        # same scenario, and the policy would overfit to that one configuration.
        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_offset)

        # -----------------------------------------------------------------------
        # 6a. INITIALISE HIDDEN STATES for this episode
        # -----------------------------------------------------------------------
        # GRU hidden state shape: [num_layers, batch_size, hidden_dim]
        #   num_layers = 1  (single GRU layer in all our networks)
        #   batch_size = 1  (we process one agent at a time during rollout)
        #   hidden_dim = 128 for scouts/commanders; 128 for critic
        #
        # Starting from zeros is the standard convention for episodic tasks:
        # the agent has no memory from previous episodes (each episode is
        # independent).
        scout_h = {q: torch.zeros(1, 1, config['scout_hidden_dim']) for q in local_env.quad_agents}
        cmdr_h  = {f: torch.zeros(1, 1, 128)                        for f in local_env.fixed_agents}
        crit_h  = {a: torch.zeros(1, 1, 128)                        for a in local_env.possible_agents}

        # Save h_0 NOW, before any steps are taken.
        # These clean zero-tensors will be sent back to the main process as the
        # "starting point" for re-running the sequences during PPO gradient updates.
        if config['N_QUADS'] > 0:
            scout_init_h_list.append(scout_h[local_env.quad_agents[0]].clone())
            critic_init_h_scout_list.append(crit_h[local_env.quad_agents[0]].clone())
        if config['N_FIXED'] > 0:
            cmdr_init_h_list.append(cmdr_h[local_env.fixed_agents[0]].clone())
            critic_init_h_cmdr_list.append(crit_h[local_env.fixed_agents[0]].clone())

        # Storage for this episode's per-agent, per-step data.
        # We cannot compute discounted returns until the episode is DONE
        # (we need to know future rewards), so we buffer everything first.
        ep_rollouts     = {a: [] for a in local_env.possible_agents}
        agent_lifespans = {a: config['max_steps'] for a in local_env.possible_agents}
        episode_reward  = 0.0

        # -----------------------------------------------------------------------
        # 6b. TIMESTEP LOOP
        # -----------------------------------------------------------------------
        # We always run the FULL max_steps, even if all agents die early.
        # Reason: discounted returns are computed as a backwards pass over the
        # entire episode array.  A variable-length episode would require
        # variable-length tensors, which torch.cat() cannot handle easily.
        # Dead agents simply receive 0 reward and dummy observations for the
        # remaining steps (this is harmless: zero rewards + zero advantage = no
        # gradient update for those steps).
        for step in range(config['max_steps']):
            # The GLOBAL state is used by the CENTRALISED critic.
            # It contains the full fire map + all agent positions -- information
            # that would not be available in a real deployment but is allowed
            # during training under the CTDE (Centralised Training, Decentralised
            # Execution) paradigm.
            global_state = local_env.state()
            g_tensor     = torch.FloatTensor(global_state).unsqueeze(0)  # [1, global_state_dim]
            actions      = {}   # action dict to be fed to env.step()
            step_results = {}   # stores this step's data for ALL agents

            # ------------------------------------------------------------------
            # SCOUT FORWARD PASS
            # ------------------------------------------------------------------
            # Each scout independently observes its local fire map (32x32),
            # its own proprioceptive state, and the relative positions of its
            # neighbours (for collision avoidance and coordination).
            # The network outputs:
            #   dist -- Normal distribution over (Roll, Pitch, Yaw, Throttle)
            #   msg  -- 5-d message vector broadcast to the commander
            #   h_out -- updated GRU hidden state (kept for the next step)
            #
            # IMPORTANT -- torch.no_grad():
            # During rollout we do NOT need gradients.  Gradients are only
            # needed during the PPO update on the main process.
            # torch.no_grad() disables PyTorch's autograd engine entirely for
            # the duration of the block -- this makes inference ~2x faster
            # (no computation graph is built) and uses significantly less RAM
            # (no intermediate activations are stored for backprop).
            scout_msgs_list, scout_mask_list = [], []
            for q in local_env.quad_agents:
                if q in local_env.agents:                       # scout is still alive
                    with torch.no_grad():
                        l_map = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0)   # [1, 1, 32, 32]
                        s_st  = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0) # [1, self_dim]
                        n_s   = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0)  # [1, N, 3]
                        n_m   = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0)     # [1, N]
                        dist, msg, h_out = local_scout(l_map, s_st, n_s, n_m, scout_h[q])
                        val, c_h_out     = local_critic(g_tensor, crit_h[q])

                        # dist.sample() draws ONE action from the Normal distribution:
                        #   act = mean + std * epsilon,  where epsilon ~ N(0, I)
                        # This is the STOCHASTIC exploration step.
                        # We use .sample() (not .rsample()) because we do NOT need
                        # gradients here -- .sample() is slightly faster.
                        # During the PPO update we will call .log_prob() on the
                        # stored action to get the updated log-probability.
                        act = dist.sample()    # [1, action_dim]

                    # WHAT WE STORE AND WHY:
                    #   "map", "self", "n_s", "n_m" -- the observations; needed to
                    #       re-run the network during the PPO update.
                    #   "h"   -- the GRU hidden state BEFORE this step; needed as
                    #       h_0 for sequence-by-sequence re-evaluation.
                    #   "act" -- the action that was actually taken; the PPO ratio
                    #       measures how the NEW policy rates THIS specific action.
                    #   "lp"  -- log pi_old(a|s): log-probability of the chosen
                    #       action under the policy that was active during rollout.
                    #       This is the DENOMINATOR of the PPO importance-sampling
                    #       ratio: r(theta) = pi_new(a|s) / pi_old(a|s).
                    #       .sum(1) sums over the action dimensions (the 4 control
                    #       outputs are treated as independent Normal variables).
                    #   "val" -- V_old(s): the critic's CURRENT value estimate.
                    #       Used as the baseline in advantage computation:
                    #       A(s,a) = G - V_old(s).
                    #   "gs"  -- global state tensor; goes into the critic buffer.
                    #   "ch"  -- critic's GRU hidden state before this step.
                    step_results[q] = {
                        "type": "scout",
                        "map": l_map, "self": s_st, "n_s": n_s, "n_m": n_m,
                        "h": scout_h[q],           # GRU h BEFORE this step (for re-run)
                        "act": act,
                        "lp": dist.log_prob(act).sum(1),   # log pi_old(a|s)
                        "val": val,                         # V_old(s) baseline
                        "gs": g_tensor, "ch": crit_h[q]
                    }
                    # Advance hidden states to the next timestep.
                    scout_h[q], crit_h[q] = h_out, c_h_out
                    # Convert to numpy for the environment's step() API.
                    actions[q] = act.squeeze(0).numpy()

                    # Collect this scout's message for the commander.
                    # msg shape: [1, scout_msg_dim]
                    scout_msgs_list.append(msg)
                    scout_mask_list.append(False)           # False = valid (not masked out)
                else:
                    # Scout is dead: send a zero message and mark the slot as padding.
                    # True in the mask means "ignore this slot" (PyTorch convention).
                    scout_msgs_list.append(torch.zeros(1, config['scout_msg_dim']))
                    scout_mask_list.append(True)            # True = masked / ignored
                    step_results[q] = {"type": "dead_scout", "gs": g_tensor}

            # ------------------------------------------------------------------
            # COMMANDER FORWARD PASS
            # ------------------------------------------------------------------
            # Assemble the scout messages into a single [1, N_scouts, msg_dim]
            # tensor and a [1, N_scouts] bool mask so the CommanderActor's
            # Cross-Attention module knows which message slots are valid.
            #
            # torch.stack(scout_msgs_list, dim=1):
            #   Each element of scout_msgs_list has shape [1, msg_dim].
            #   Stacking along dim=1 produces [1, N_scouts, msg_dim].
            #   (dim=0 would produce [N_scouts, 1, msg_dim] -- wrong shape.)
            msgs_t = torch.stack(scout_msgs_list, dim=1) if scout_msgs_list else d_msgs  # [1, N, msg_dim]
            msgs_m = torch.tensor(scout_mask_list).unsqueeze(0) if scout_mask_list else d_msg_m  # [1, N]

            for f in local_env.fixed_agents:
                if f in local_env.agents:                   # commander is alive
                    with torch.no_grad():
                        s_st = torch.FloatTensor(obs[f]["self_state"]).unsqueeze(0)
                        dist, _, h_out   = local_commander(s_st, msgs_t, msgs_m, cmdr_h[f])
                        val, c_h_out     = local_critic(g_tensor, crit_h[f])
                        act = dist.sample()
                    step_results[f] = {
                        "type": "commander",
                        "self": s_st, "msgs": msgs_t, "m_m": msgs_m,
                        "h": cmdr_h[f],
                        "act": act,
                        "lp": dist.log_prob(act).sum(1),
                        "val": val,
                        "gs": g_tensor, "ch": crit_h[f]
                    }
                    cmdr_h[f], crit_h[f] = h_out, c_h_out
                    actions[f] = act.squeeze(0).numpy()
                else:
                    step_results[f] = {"type": "dead_cmdr", "gs": g_tensor}

            # ------------------------------------------------------------------
            # ENVIRONMENT STEP
            # ------------------------------------------------------------------
            # local_env.step(actions) returns a standard PettingZoo tuple:
            #   obs          -- new observations for all LIVING agents
            #   rewards      -- reward dict {agent_id: float} for this step
            #   terminations -- {agent_id: True} when an agent dies (crashed
            #                   or ran out of time) -- it leaves local_env.agents
            #   truncations  -- {agent_id: True} for time-limit truncation (unused)
            #   infos        -- extra info dict (unused)
            #
            # If ALL agents have died, local_env.agents is empty and we skip
            # the step, setting reward to 0.0 for all remaining dummy steps.
            if local_env.agents:   # at least one agent is alive
                obs, rewards, terminations, _, _ = local_env.step(actions)
                for a in local_env.possible_agents:
                    r = rewards.get(a, 0.0)     # dead agents don't appear in rewards dict
                    episode_reward += r
                    step_results[a]["reward"] = r
                    # Record the step at which this agent first terminated.
                    # agent_lifespans starts at max_steps (full episode); we
                    # overwrite it only on the first True termination signal.
                    if terminations.get(a, False) and agent_lifespans[a] == config['max_steps']:
                        agent_lifespans[a] = step
            else:
                # All agents are dead -- pad remaining steps with zero reward.
                for a in local_env.possible_agents:
                    step_results[a]["reward"] = 0.0

            # Append this timestep's data for all agents to the episode buffer.
            for a in local_env.possible_agents:
                ep_rollouts[a].append(step_results[a])

        # =======================================================================
        # 7. COMPUTE DISCOUNTED RETURNS (Generalized Advantage Estimation - GAE)
        # =======================================================================
        # WHAT IS GAE?
        # The standard discounted return (Monte Carlo) has high variance because
        # it sums up hundreds of noisy, delayed rewards. GAE solves this by mixing
        # the real environmental rewards with the Critic's predictions (V(s)).
        #
        # HOW IT WORKS:
        # 1. Temporal Difference (TD) Error (delta):
        #    delta_t = r_t + (gamma * V_{t+1}) - V_t
        #    This asks: "How much better was the actual reward + next state
        #    compared to what the critic expected for this current state?"
        #
        # 2. Advantage Accumulation:
        #    A_t = delta_t + (gamma * lambda) * A_{t+1}
        #    We accumulate these TD errors backwards. The lambda parameter (0 to 1)
        #    controls the bias-variance tradeoff:
        #      lambda = 1.0  -> Pure Monte Carlo (high variance, zero bias)
        #      lambda = 0.0  -> Pure 1-step TD (low variance, high bias)
        #      lambda = 0.95 -> Standard sweet spot used in PPO/MAPPO.
        #
        # WHY BACKWARD?
        # Just like standard returns, we need the future (A_{t+1} and V_{t+1}) 
        # to compute the present (A_t).
        #
        # THE PPO TRICK:
        # The main process in train.py computes advantages simply as: 
        #    Advantage = Returns - V_t
        # To pass our computed GAE to train.py without changing its logic, we
        # store a pseudo-return calculated as:
        #    Return = GAE + V_t
        # That way, when train.py runs its formula: (GAE + V_t) - V_t = GAE!
        # This also serves as a stable TD(lambda) target for training the Critic.
        
        gamma = config.get('gamma', 0.99)
        gae_lambda = config.get('gae_lambda', 0.95)

        for a in local_env.possible_agents:
            gae = 0.0
            for i in reversed(range(config['max_steps'])):
                d = ep_rollouts[a][i]
                
                # Get the critic's value estimate for the current state.
                # If the agent is dead, 'val' will not exist, so we default to 0.0.
                # .item() extracts the float from the [1, 1] tensor.
                val_t = d.get("val", torch.tensor([[0.0]])).item()
                
                # Get the critic's value estimate for the next state.
                # For the very last step of the episode, there is no future, so V_{t+1} is 0.0.
                if i == config['max_steps'] - 1:
                    val_next = 0.0
                else:
                    val_next = ep_rollouts[a][i+1].get("val", torch.tensor([[0.0]])).item()
                    
                r_t = d["reward"]
                
                # 1. Calculate the TD error for this step
                delta = r_t + gamma * val_next - val_t
                
                # 2. Accumulate the GAE backwards
                gae = delta + gamma * gae_lambda * gae
                
                # 3. Store the pseudo-return (GAE + Value) for the main PPO loop
                d["ret"] = gae + val_t

        # =======================================================================
        # 8. PACK EPISODE DATA INTO PRE-SPLIT BUFFERS
        # =======================================================================
        # We iterate STEP-BY-STEP (outer loop) and then AGENT-BY-AGENT (inner
        # loops) so that the CRITIC buffer ends up interleaved as:
        #
        #   [step0_scout0, step0_cmdr0, step1_scout0, step1_cmdr0, ...]
        #
        # This specific ordering is what the PPO update in train.py expects.
        # It allows the critic buffer to be reshaped as:
        #   [episodes, steps, num_agents, global_state_dim]
        # and then transposed to get per-agent episode sequences for the GRU.
        # If we had iterated agent-by-agent first, that reshape would be wrong.
        for step_idx in range(config['max_steps']):

            # ------------------------------------------------------------------
            # CRITIC BUFFER  (interleaved: all agents at this step, in order)
            # ------------------------------------------------------------------
            for a_name in local_env.possible_agents:
                d = ep_rollouts[a_name][step_idx]
                critic_buf["g_states"].append(d["gs"])           # [1, global_state_dim]
                critic_buf["returns"].append(d["ret"])           # scalar G_t
                # "val" is absent for dead agents (type = "dead_scout"/"dead_cmdr").
                # We fall back to V=0 for dead steps (no real baseline needed;
                # advantage will be ~0 anyway since return is also ~0 for dead agents).
                critic_buf["values"].append(d.get("val", torch.tensor([[0.0]])))

            # ------------------------------------------------------------------
            # SCOUT BUFFER  (one entry per scout per step)
            # ------------------------------------------------------------------
            # Alive scouts: use their real observations and actions.
            # Dead scouts:  use pre-built dummy tensors (all zeros / all True).
            # Both paths append the SAME number of entries per step, keeping
            # the buffer length consistent across all steps.
            for q in local_env.quad_agents:
                d = ep_rollouts[q][step_idx]
                if d["type"] == "scout":                # scout was alive at this step
                    scout_buf["maps"].append(d["map"])
                    scout_buf["self_states"].append(d["self"])
                    scout_buf["neighbor_states"].append(d["n_s"])
                    scout_buf["neighbor_masks"].append(d["n_m"])
                    scout_buf["actions"].append(d["act"])
                    scout_buf["logprobs"].append(d["lp"])
                else:                                   # scout was dead -- pad with zeros
                    scout_buf["maps"].append(d_map)
                    scout_buf["self_states"].append(d_scout_self)
                    scout_buf["neighbor_states"].append(d_neigh_s)
                    scout_buf["neighbor_masks"].append(d_neigh_m)
                    scout_buf["actions"].append(torch.zeros(1, 4))
                    scout_buf["logprobs"].append(torch.tensor([0.0]))
                # Returns are always appended (dead agents get G_t ~ 0 from the
                # discounted returns pass -- they received 0 future rewards).
                scout_buf["returns"].append(d["ret"])

            # ------------------------------------------------------------------
            # COMMANDER BUFFER  (one entry per fixed-wing per step)
            # ------------------------------------------------------------------
            for f in local_env.fixed_agents:
                d = ep_rollouts[f][step_idx]
                if d["type"] == "commander":
                    cmdr_buf["fixed_states"].append(d["self"])
                    cmdr_buf["incoming_msgs"].append(d["msgs"])
                    cmdr_buf["msg_masks"].append(d["m_m"])
                    cmdr_buf["actions"].append(d["act"])
                    cmdr_buf["logprobs"].append(d["lp"])
                else:                                   # dead / absent commander
                    cmdr_buf["fixed_states"].append(d_cmd_self)
                    cmdr_buf["incoming_msgs"].append(d_msgs)
                    cmdr_buf["msg_masks"].append(d_msg_m)
                    cmdr_buf["actions"].append(torch.zeros(1, 4))
                    cmdr_buf["logprobs"].append(torch.tensor([0.0]))
                cmdr_buf["returns"].append(d["ret"])

        worker_total_rewards.append(episode_reward)
        worker_lifespans.append(np.mean(list(agent_lifespans.values())))

    # =========================================================================
    # 9. FINALISE AND RETURN WORKER DATA
    # =========================================================================

    # Clean up the simulator process before this worker exits.
    # Failing to call stop_simulation() would leave a zombie PyBullet process
    # consuming a TCP port, eventually exhausting system resources.
    local_env.sim.stop_simulation()

    # -----------------------------------------------------------------------
    # Concatenate lists of tensors into single contiguous tensors.
    # torch.cat() along dim=0 stacks [1, ...] tensors into [N, ...] tensors.
    # We do this HERE on the worker to minimise the amount of
    # Python-object serialisation that needs to cross the IPC channel:
    # one big tensor is much cheaper to pickle than 10000 small ones.
    # -----------------------------------------------------------------------
    def cat_buf(buf):
        result = {}
        for k, v in buf.items():
            if len(v) == 0:
                result[k] = torch.tensor([], dtype=torch.float32)
                continue
            if k == "returns":
                # Returns were stored as plain Python scalars (floats), not as
                # tensors (unlike observations and actions).
                # torch.tensor() wraps them into a 1-D float tensor [N].
                result[k] = torch.tensor(v, dtype=torch.float32)
            else:
                # All other entries are already [1, ...] tensors; torch.cat()
                # merges them along the first axis into [N, ...].
                result[k] = torch.cat(v)
        return result

    out_scout  = cat_buf(scout_buf)
    out_cmdr   = cat_buf(cmdr_buf)
    out_critic = cat_buf(critic_buf)

    # -----------------------------------------------------------------------
    # Stack per-episode initial hidden states into a single tensor.
    #
    # Each h_0 was stored as [1, 1, hidden_dim] (the GRU convention).
    # torch.cat(list, dim=0) stacks them: [num_eps, 1, hidden_dim].
    #
    # The main process then squeezes dim=1 and unsqueezes at dim=0 to
    # produce [1, num_eps, hidden_dim] -- the shape PyTorch's GRU expects
    # for the h_0 argument when batch_size = num_eps.
    # -----------------------------------------------------------------------
    out_init_h = {
        "scout":        torch.cat(scout_init_h_list,        dim=0) if scout_init_h_list        else None,
        "cmdr":         torch.cat(cmdr_init_h_list,         dim=0) if cmdr_init_h_list         else None,
        "critic_scout": torch.cat(critic_init_h_scout_list, dim=0) if critic_init_h_scout_list else None,
        "critic_cmdr":  torch.cat(critic_init_h_cmdr_list,  dim=0) if critic_init_h_cmdr_list  else None,
    }

    return out_scout, out_cmdr, out_critic, out_init_h, worker_total_rewards, worker_lifespans
