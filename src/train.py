import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import cProfile, pstats, io
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend so plots can be saved without a display
import matplotlib.pyplot as plt
import datetime, time

from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor, MAPPOCritic
from concurrent.futures import ProcessPoolExecutor
from worker import collect_episodes_per_worker

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
    profiling = False  # Set to True to profile the PPO update (runs only for the first batch)

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
    update_epochs = 4          # how many gradient passes over each collected batch
    num_workers   = 20         # parallel CPU workers for data collection
    eps_per_worker = 3         # episodes collected by each worker per batch
    episodes_per_batch = num_workers * eps_per_worker   # = 60 episodes per batch

    # Learning rates — training from scratch, all networks use full LR.
    # (Use a smaller lr_scout when loading a pre-trained scout for fine-tuning.)
    lr_commander = 3e-4
    lr_critic    = 3e-4
    lr_scout     = 3e-4

    # ==========================================================================
    # 2. TEAM CONFIGURATION & NETWORK DIMENSIONS
    # ==========================================================================
    N_QUADS = 1   # number of quadrotor scouts
    N_FIXED = 0   # 0 = train scout only first; set to 1 once scout converges

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
        path_to_old_model = "/homes/eva/xj/xjahnf00/tmp/DP/results/TrainingQuad/05_TrainingFromZero_Added3Inputs/scout_ep4800.pt"
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
        path_to_old_model = "/homes/eva/xj/xjahnf00/tmp/DP/results/TrainingTogether/none.pt"
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
    if scout_actor:     optim_groups.append({"params": scout_actor.parameters(),     "lr": lr_scout})
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
    jerk_history            = []   # Track action smoothness

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
            batch_start = time.time()
            rollout_start = time.time()

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

            rollout_time = time.time() - rollout_start

            # Progress log (one line per batch instead of per episode)
            avg_batch  = np.mean(batch_rewards)
            avg_reward = np.mean(episode_rewards_history[-15:]) if len(episode_rewards_history) >= 15 else np.mean(episode_rewards_history)
            print(f"{datetime.datetime.now()} | Batch {batch_idx:04d} (Ep {episodes_played:04d}) | Avg Batch Reward: {avg_batch:.2f} | Rolling avg (15): {avg_reward:.2f} | Rollout: {rollout_time:.1f}s")

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
                print(f"{datetime.datetime.now()} | ⭐ New best model saved! (Rolling avg: {best_avg_reward:.2f})")

            # ==============================================================
            # 5f. PPO UPDATE  (runs on GPU)
            # ==============================================================
            # WHAT IS PPO?
            # -------------
            # Proximal Policy Optimisation (PPO) is an on-policy reinforcement
            # learning algorithm.  "On-policy" means the agent must learn from
            # data it collected itself, using the policy it had AT THAT MOMENT.
            #
            # THE CORE PROBLEM PPO SOLVES:
            # Plain policy gradient (REINFORCE) updates the network weights by
            # following the gradient of E[log π(a|s) * A(s,a)].  If the learning
            # rate is too large, one bad update can destroy the policy completely
            # (it starts taking random actions and never recovers, because the bad
            # policy then collects bad data, causing even worse updates — a death
            # spiral).  PPO prevents this by *clipping* how much the policy is
            # allowed to change in a single gradient step.
            #
            # THE THREE PHASES PER BATCH:
            #   Phase 1 — Rollout (already done above, by workers):
            #     Run current policy π_old in the environment.
            #     Record: states s, actions a, log π_old(a|s), rewards r,
            #             critic predictions V(s), and discounted returns G.
            #
            #   Phase 2 — Advantage estimation (below):
            #     A(s,a) = G - V(s)   (how much better was the actual return
            #     than the critic's prediction?  Positive = better than expected,
            #     negative = worse than expected.)
            #
            #   Phase 3 — Gradient update (the loop below):
            #     Re-evaluate the CURRENT (updated) policy π_new on the
            #     SAME batch of states/actions.
            #     Compute ratio r(θ) = π_new(a|s) / π_old(a|s).
            #     If A > 0 (action was good): we want r > 1 (do it more often),
            #       but we cap the benefit at 1 + clip_coef to stay conservative.
            #     If A < 0 (action was bad): we want r < 1 (do it less often),
            #       but we cap the penalty at 1 - clip_coef.
            #     This clipping is the "Proximal" part — updates are kept
            #     close (proximal) to the old policy.
            #
            # WHY MULTIPLE EPOCHS (update_epochs=4)?
            #     Collecting data is expensive (it requires interacting with
            #     the simulator).  PPO allows us to squeeze multiple gradient
            #     steps out of the same batch without diverging, because the
            #     clip prevents any single step from going too far.  This makes
            #     PPO much more sample-efficient than plain policy gradient.
            # ==============================================================
            num_agents = N_QUADS + N_FIXED
            episodes   = episodes_per_batch

            if len(batch_critic["returns"]) > 0:

                if profiling:
                    _ppo_prof = cProfile.Profile()
                    _ppo_prof.enable()

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

                # --- Reconstruct GRU initial hidden states ---
                # Each worker stored the per-episode initial hidden state h_0 as a
                # tensor of shape [1, 1, hidden_dim]  (GRU convention: [layers, batch, dim]).
                # Here we merge all episodes from all workers into one big tensor:
                #   torch.cat(lst, dim=0)  -> [num_episodes, 1, hidden_dim]
                #   .squeeze(1)            -> [num_episodes, hidden_dim]
                #   .unsqueeze(0)          -> [1, num_episodes, hidden_dim]  ← GRU expects this
                # Passing h_0 into the GRU means each episode starts from the hidden
                # state it had at the very beginning of that episode, not from zeros.
                # This is important because our agents carry memory across timesteps;
                # if we always reset to zeros the network would never learn to use its
                # GRU memory effectively.
                # Lists may be empty when N_QUADS=0 or N_FIXED=0 — guard with None.
                def _cat_h(lst):
                    return torch.cat(lst, dim=0).squeeze(1).unsqueeze(0).to(device) if lst else None

                h_scout        = _cat_h(batch_init_h["scout"])
                h_cmdr         = _cat_h(batch_init_h["cmdr"])
                h_critic_scout = _cat_h(batch_init_h["critic_scout"])
                h_critic_cmdr  = _cat_h(batch_init_h["critic_cmdr"])

                # The critic is a SINGLE shared network that evaluates ALL agents.
                # Its GRU batch dimension interleaves agents: [scout_ep0, cmdr_ep0, scout_ep1, ...].
                # We stack the per-agent hidden states side by side (dim=1) and
                # then flatten episodes*agents into the GRU's batch axis.
                # Only agent types that actually exist are included.
                h_parts = [h for h in [h_critic_scout, h_critic_cmdr] if h is not None]
                h_critic = torch.stack([h.squeeze(0) for h in h_parts], dim=1)  # [ep, agent_types, dim]
                h_critic = h_critic.reshape(1, episodes * num_agents, -1)        # [1, ep*agents, dim]

                print(f"{datetime.datetime.now()} | 🛠️  Running PPO update ({len(cr_returns)} samples)...")

                # --- Compute advantages ---
                # WHAT IS THE ADVANTAGE A(s, a)?
                # The advantage answers: "How much better (or worse) was the
                # action I actually took compared to the average action I could
                # have taken in this state?"
                #
                #   A(s, a) = G - V(s)
                #
                # where:
                #   G    = discounted return (actual sum of future rewards,
                #          computed during rollout via gamma-discounting)
                #   V(s) = critic's prediction of how good the state is
                #          (the "baseline" or "expected" return)
                #
                # Examples:
                #   A = +5.0  -> this action led to 5 more reward than expected
                #                -> the gradient will push the policy to take it
                #                   MORE often
                #   A = -3.0  -> this action led to 3 less reward than expected
                #                -> the gradient will push the policy to take it
                #                   LESS often
                #   A ~  0.0  -> performance was exactly as expected
                #                -> no change to the policy
                #
                # .detach() on cr_values: the advantage is used to scale the
                # POLICY gradient, not the critic gradient.  Detaching stops
                # gradients from flowing back through the critic via the advantage
                # term (the critic has its own separate MSE loss below).
                #
                # Normalisation (zero mean, unit std):
                # Advantages are normalised per-batch to have mean~0, std~1.
                # This acts like an adaptive learning-rate schedule: it prevents
                # one batch with unusually high returns from causing a giant update
                # while another batch with low returns barely learns anything.
                cr_adv = cr_returns.unsqueeze(1) - cr_values.detach()  # A(s,a) = G - V(s)
                cr_adv = (cr_adv - cr_adv.mean()) / (cr_adv.std() + 1e-8)  # normalise to N(0,1)

                # The critic buffer is interleaved: [scout_ep0_t0, cmdr_ep0_t0, scout_ep0_t1, ...]
                # Reshape to [episodes, steps, num_agents] and slice out per-agent advantages.
                s_adv = cr_adv.view(episodes, max_steps, num_agents)[:, :, 0].reshape(-1)  # scout
                c_adv = cr_adv.view(episodes, max_steps, num_agents)[:, :, 1].reshape(-1) if num_agents > 1 else None  # commander

                def to_seq_agent(t, num_traj):
                    """Reshape flat [num_traj*steps, ...] → sequential [num_traj, steps, ...]
                    needed because the LSTM expects (batch, seq, features)."""
                    return t.view(num_traj, max_steps, *t.shape[1:])

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # CALCULATE ACTION JERK (Smoothness Metric)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # We measure how much the actions change between consecutive 
                # timesteps. Lower values = smoother flight.
                with torch.no_grad():
                    # Scout actions: [episodes, max_steps, 4]
                    s_seq = s_actions.view(episodes, max_steps, -1)
                    # Absolute difference between step t and t+1
                    s_diff = torch.abs(s_seq[:, 1:] - s_seq[:, :-1])
                    avg_jerk = s_diff.mean().item()
                    jerk_history.append(avg_jerk)

                # --- Gradient update loop (update_epochs passes over the same data) ---
                # We run update_epochs gradient passes over this SAME batch of data.
                # This is safe because the PPO clip (below) limits how much the
                # policy can change per step -- once the policy has moved enough that
                # ratio exceeds [1-clip, 1+clip], the loss is clamped and gradients
                # effectively become zero, so further epochs have diminishing effect.
                # Using multiple epochs squeezes more learning out of expensive
                # simulator data without risking a catastrophic policy collapse.
                for epoch in range(update_epochs):
                    entropy_sum = torch.tensor(0.0, device=device)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # SCOUT POLICY LOSS  (PPO-clip objective)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # We re-run the scout network on the SAME observations that
                    # were collected during the rollout.  This gives us the
                    # NEW log-probabilities for the actions that were actually
                    # taken, under the CURRENT (partially-updated) policy weights.
                    #
                    # The network input must be (Batch, Seq, ...) for the GRU,
                    # so we use to_seq_agent() to reshape the flat buffer.
                    if scout_actor is not None:
                        dist, _, _ = scout_actor(
                            to_seq_agent(s_maps,    episodes),
                            to_seq_agent(s_self,    episodes),
                            to_seq_agent(s_neigh_s, episodes),
                            to_seq_agent(s_neigh_m, episodes),
                            h_scout
                        )
                        flat_actions_s = s_actions.view(episodes * max_steps, -1)

                        # log π_new(a|s): log-probability of taking the stored action
                        # under the NEW policy.  .sum(1) sums over the 4 action
                        # dimensions (Roll, Pitch, Yaw, Throttle) because the
                        # Normal distribution over the action VECTOR has factored
                        # dimensions -- log P(a1,a2,a3,a4) = sum log P(ai).
                        new_lp_s = dist.log_prob(flat_actions_s).sum(1)  # log π_new(a|s)

                        # Entropy of the policy distribution.  High entropy means the
                        # policy is uncertain / exploratory (good early in training).
                        # Low entropy means it has converged to near-deterministic
                        # actions.  Adding entropy to the loss prevents premature
                        # convergence to a suboptimal deterministic strategy.
                        entropy_sum += dist.entropy().sum(1).mean()

                        # IMPORTANCE SAMPLING RATIO
                        # ratio = π_new(a|s) / π_old(a|s)
                        # We compute it in log space for numerical stability:
                        #   exp(log π_new - log π_old) = π_new / π_old
                        # Interpretation:
                        #   ratio > 1  ->  current policy takes this action MORE often
                        #   ratio < 1  ->  current policy takes this action LESS often
                        #   ratio = 1  ->  policy has not changed for this action
                        ratio_s = torch.exp(new_lp_s - s_logprobs)  # π_new / π_old

                        # PPO CLIPPED OBJECTIVE (the heart of PPO)
                        # We want to MAXIMISE the policy performance, which means
                        # MINIMISING the negative of it (gradient descent).
                        #
                        # pg1: unclipped surrogate  = -A * ratio
                        # pg2: clipped  surrogate   = -A * clip(ratio, 1-ε, 1+ε)
                        #
                        # Taking max(pg1, pg2) -- the PESSIMISTIC objective --
                        # means we pick the WORSE of the two estimates:
                        #
                        #   When A > 0 (action was good, we want to increase prob):
                        #     pg1 = -A * ratio        (gets more negative as ratio grows)
                        #     pg2 = -A * min(ratio, 1+ε)  (floored at clip boundary)
                        #     max picks the LESS negative one = -A * min(ratio, 1+ε)
                        #     Effect: we only reward the policy up to ratio=1+ε;
                        #             beyond that the gradient is zero (no more benefit).
                        #
                        #   When A < 0 (action was bad, we want to decrease prob):
                        #     pg1 = -A * ratio        (positive, gets more positive as ratio grows)
                        #     pg2 = -A * max(ratio, 1-ε)  (capped at clip boundary)
                        #     max picks the LARGER (worse) value = -A * ratio
                        #     Effect: if we've already pushed the probability down
                        #             enough (ratio < 1-ε), additional penalty is cut off.
                        #
                        # Net result: PPO clips the BENEFIT of good updates AND
                        # the PUNISHMENT of bad ones, keeping the policy close to
                        # what it was when the data was collected.
                        pg1_s = -s_adv * ratio_s
                        pg2_s = -s_adv * torch.clamp(ratio_s, 1 - clip_coef, 1 + clip_coef)
                        policy_loss_s = torch.max(pg1_s, pg2_s).mean()  # pessimistic (conservative) objective
                    else:
                        policy_loss_s = torch.tensor(0.0, device=device)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # COMMANDER POLICY LOSS  (identical PPO-clip logic)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Exact same procedure as the scout, but operating on the
                    # commander's own state + incoming scout messages.
                    # When N_FIXED=0 (training scouts only), this block is skipped
                    # and policy_loss_c is set to zero so the combined loss is
                    # computed correctly without NaN or unintended scaling.
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
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # The critic learns to predict the expected discounted return
                    # V(s) from the global state.  It is trained with plain MSE:
                    #   L_V = MSE( V_new(s),  G )
                    # where G are the pre-computed discounted returns from the rollout.
                    #
                    # A better critic makes better advantage estimates (A = G - V),
                    # which in turn produces lower-variance policy gradient updates.
                    # The 0.5 coefficient in the combined loss (below) is a standard
                    # MAPPO / PPO scaling convention.
                    #
                    # RESHAPE EXPLANATION:
                    # The raw critic buffer is FLAT and interleaved:
                    #   cr_g_states: [episodes * steps * num_agents, gs_dim]
                    # The critic GRU needs a contiguous sequence per episode/agent:
                    #   [episodes * num_agents, steps, gs_dim]
                    # Transformation:
                    #   .view(episodes, steps, agents, gs_dim)  -- restore 4-D structure
                    #   .transpose(1, 2)                        -- swap steps ↔ agents
                    #                        -> [episodes, agents, steps, gs_dim]
                    #   .reshape(episodes * agents, steps, gs_dim)  -- merge ep & agent
                    # Now each "row" in the batch is ONE agent's full episode trajectory.
                    cr_g_seq = (cr_g_states
                                .view(episodes, max_steps, num_agents, -1)
                                .transpose(1, 2)                   # -> [ep, agents, steps, gs_dim]
                                .reshape(episodes * num_agents, max_steps, -1))
                    new_values, _ = critic(cr_g_seq, h_critic)
                    value_loss    = nn.MSELoss()(new_values, cr_returns.unsqueeze(1))

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # COMBINED LOSS  (standard MAPPO objective)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # The total loss combines three terms:
                    #
                    #   L = L_policy + 0.5 * L_value - 0.1 * H
                    #
                    #   L_policy  (weight 1.0):
                    #     Sum of scout + commander PPO-clipped policy losses.
                    #     This is the PRIMARY learning signal -- it pushes the
                    #     actors to take actions that had high advantage.
                    #
                    #   0.5 * L_value (weight 0.5):
                    #     Critic MSE loss, scaled down so value learning does
                    #     not dominate the policy gradient in the shared update.
                    #     (The critic shares the optimizer with the actors.)
                    #
                    #   -0.1 * H (weight -0.1, NEGATIVE):
                    #     Entropy BONUS.  Since we are minimising the loss,
                    #     subtracting entropy MAXIMISES it.
                    #     This prevents the policy from collapsing to a single
                    #     action too early in training (premature exploitation).
                    #     0.1 is the entropy coefficient -- a hyperparameter that
                    #     controls how much exploration is encouraged.
                    policy_loss = policy_loss_s + policy_loss_c
                    loss        = policy_loss + 0.5 * value_loss - 0.005 * entropy_sum

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # COMMUNICATION AUXILIARY LOSS  (protocol grounding)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Without supervision, the scout's msg_head would learn an
                    # arbitrary "language" that the commander cannot bootstrap from.
                    # This auxiliary loss GROUNDS the first 3 components of every
                    # scout message to carry a specific, human-readable meaning:
                    #   msg[0] = relative x-offset to nearest fire
                    #   msg[1] = relative y-offset to nearest fire
                    #   msg[2] = fire intensity at that location
                    # (These values live at indices 12-14 of the scout's self_state.)
                    #
                    # By forcing the scout to communicate its CURRENT observation,
                    # the commander can immediately treat the messages as a map
                    # of fire locations, making cross-attention interpretable from
                    # the very first training batch.
                    #
                    # The weight of 0.5 is a gentle nudge: strong enough to
                    # ground the protocol, small enough not to override the
                    # primary flight / reward optimisation signal.
                    if scout_actor is not None:
                        # Re-generate messages with current network weights
                        _, s_msgs, _ = scout_actor(
                            to_seq_agent(s_maps,    episodes),
                            to_seq_agent(s_self,    episodes),
                            to_seq_agent(s_neigh_s, episodes),
                            to_seq_agent(s_neigh_m, episodes),
                            h_scout
                        )

                        # Ground truth: indices 12, 13, 14 from self_state
                        # (dyn_rel_x, dyn_rel_y, dyn_intensity)
                        target_protocol    = s_self.view(episodes * max_steps, -1)[:, 12:15]

                        # The first 3 values of the broadcast message vector
                        generated_protocol = s_msgs.reshape(episodes * max_steps, -1)[:, 0:3]

                        # MSE pushes the scout's "words" towards their intended meanings
                        comm_aux_loss = nn.MSELoss()(generated_protocol, target_protocol)
                        loss += 0.5 * comm_aux_loss

                    loss_history.append(loss.item())
                    v_loss_history.append(value_loss.item())
                    entropy_history.append(entropy_sum.detach().item())

                    # ----------------------------------------------------------
                    # GRADIENT STEP
                    # ----------------------------------------------------------
                    # 1. optimizer.zero_grad()
                    #    Clear gradients from the previous epoch.  PyTorch
                    #    accumulates gradients by default; without this call,
                    #    gradients from epoch N would be added on top of epoch N-1.
                    #
                    # 2. loss.backward()
                    #    Backpropagation: compute d(loss)/d(param) for every
                    #    parameter in all three networks (critic, scout, commander)
                    #    using the chain rule through the entire computation graph.
                    #
                    # 3. clip_grad_norm_(params, max_norm=0.5)
                    #    If the L2 norm of ALL gradients stacked together exceeds
                    #    0.5, rescale them uniformly so the total norm equals 0.5.
                    #    This prevents "exploding gradients" -- situations where
                    #    a rare large loss spike would otherwise cause a massive
                    #    parameter update that derails training.
                    #    max_norm=0.5 is a conservative value appropriate for
                    #    recurrent networks (GRU here), which are more prone to
                    #    gradient explosions than plain MLPs.
                    #
                    # 4. optimizer.step()
                    #    Apply the (clipped) gradients to update all parameters
                    #    using the Adam optimiser rule.
                    optimizer.zero_grad()
                    loss.backward()
                    params = list(critic.parameters())
                    if scout_actor:     params += list(scout_actor.parameters())
                    if commander_actor: params += list(commander_actor.parameters())
                    nn.utils.clip_grad_norm_(params, max_norm=0.5)
                    optimizer.step()

                total_time = time.time() - batch_start
                print(f"{datetime.datetime.now()} | ✅ PPO update complete. Loss: {loss.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | Entropy: {entropy_sum.item():.4f} | Total batch: {total_time:.1f}s (rollout: {rollout_time:.1f}s, PPO: {total_time - rollout_time:.1f}s)")
                
                if profiling:
                    _ppo_prof.disable()
                    _s = io.StringIO()
                    pstats.Stats(_ppo_prof, stream=_s).sort_stats('cumulative').print_stats(30)
                    print("\n" + "="*80)
                    print("cProfile — PPO update (1 batch, cumulative time)")
                    print("="*80)
                    print(_s.getvalue())
                    _ppo_prof.dump_stats('ppo_update.prof')
                    print("Profile saved → ppo_update.prof  (view with: snakeviz ppo_update.prof)")
                    break   # profile only first batch

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
                plt.subplot(1, 5, 1)
                plt.plot(episode_rewards_history, label="Reward", alpha=0.3, color='green')
                if len(episode_rewards_history) > 20:
                    plt.plot(np.convolve(episode_rewards_history, np.ones(20)/20, mode='valid'),
                             label="MA 20", color='darkgreen')
                plt.title("Reward over episodes")
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Panel 2: Loss curves (log scale because values span many orders)
                plt.subplot(1, 5, 2)
                plt.plot(loss_history,   label="Total Loss",         color='red',  alpha=0.5)
                plt.plot(v_loss_history, label="Value Loss (Critic)", color='blue', alpha=0.5)
                plt.title("Loss (log scale)")
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Panel 3: Policy entropy — measures how exploratory the policy is.
                # Should start high (random) and slowly decrease as policy converges.
                plt.subplot(1, 5, 3)
                if entropy_history:
                    plt.plot(entropy_history, color='purple')
                    plt.title("Policy Entropy (exploration)")
                else:
                    plt.text(0.5, 0.5, 'No entropy data yet', ha='center')
                plt.grid(True, alpha=0.3)

                # Panel 4: Average agent lifespan — how long agents survive per episode.
                # Increasing lifespan generally means agents are getting better.
                plt.subplot(1, 5, 4)
                plt.plot(lifespan_history, color='orange')
                plt.title("Avg agent lifespan (steps)")

                # Panel 5: Action Jerk (The new smoothness metric)
                plt.subplot(1, 5, 5)
                if jerk_history:
                    plt.plot(jerk_history, color='teal')
                    plt.title("Action Jerk (Smoothness)")
                    plt.xlabel("Batches")
                    plt.ylabel("Avg Δ Action")
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig("final_training_plot.png")
                plt.close()

if __name__ == "__main__":
    train()