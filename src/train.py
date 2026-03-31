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
    print("Starting MAPPO Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Bez GPU: PyTorch během PPO update používá vlákna pro maticové operace.
    # Workeři užívají 20 jader — PPO update necháme na 4 jádrech aby si s workery nekradli.
    if device.type == "cpu":
        torch.set_num_threads(4)
    profiling = False  # Set to True to profile the PPO update (runs only for the first batch)

    # ==========================================================================
    # 1. HYPERPARAMETERS
    # ==========================================================================
    num_episodes = 150000       # total episodes to train for
    max_steps    = 2000         # maximum timesteps per episode
    learning_rate = 3e-4       # base learning rate (Adam)
    gamma         = 0.99       # discount factor — how much future rewards matter
                               #   0.99  = cares a lot about long-term rewards
                               #   0.9   = cares mostly about near-future rewards
    clip_coef     = 0.2        # PPO clipping range — prevents the new policy from
                               # deviating too far from the old one in a single update
    update_epochs = 4          # how many gradient passes over each collected batch
    num_workers   = 20         # parallel CPU workers for data collection
    eps_per_worker = 2         # episodes collected by each worker per batch
    episodes_per_batch = num_workers * eps_per_worker   # = 40 episodes per batch

    # Learning rates — training from scratch, all networks use full LR.
    # (Use a smaller lr_scout when loading a pre-trained scout for fine-tuning.)
    lr_scout     = 1e-5
    lr_commander = 1e-4
    lr_critic    = 3e-4

    path_to_critic = ""
    path_to_scout = "/homes/eva/xj/xjahnf00/tmp/DP/results/TrainingTogether/07_evaluation/models/scout_best.pt"
    path_to_commander = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/commander_ep14800.pt"   # od nuly — pre-trained model má zakořeněné špatné chování

    episodes_played = 14800

    # ==========================================================================
    # 2. TEAM CONFIGURATION & NETWORK DIMENSIONS
    # ==========================================================================
    N_QUADS = 0   # number of quadrotor scouts
    N_FIXED = 1   # 0 = train scout only first; set to 1 once scout converges

    # Spin up a temporary environment just to read observation/state space sizes.
    # We cannot hard-code them because they depend on N_QUADS, N_FIXED, and
    # internal env logic.
    temp_env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=2000.0, max_steps=max_steps)
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
        'N_QUADS': N_QUADS, 'N_FIXED': N_FIXED, 'grid_size_m': 2000.0, 'max_steps': max_steps,
        'scout_self_dim': scout_self_dim, 'scout_msg_dim': scout_msg_dim, 'scout_hidden_dim': scout_hidden_dim,
        'fixed_self_dim': fixed_self_dim, 'global_state_dim': global_state_dim, 'gamma': gamma, 'gae_lambda': 0.95
    }

    # ==========================================================================
    # 3. NETWORK INITIALISATION  (all networks live on GPU during training)
    # ==========================================================================
    # ScoutActor — processes local map, own state, neighbour states via CNN+LSTM
    if N_QUADS > 0:
        scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim, hidden_dim=scout_hidden_dim).to(device)
        if os.path.exists(path_to_scout):
            print(f"📥 Loading pre-trained scout model from {path_to_scout}")
            ckpt = torch.load(path_to_scout, map_location=device)
            model_shapes = {k: v.shape for k, v in scout_actor.state_dict().items()}
            filtered = {k: v for k, v in ckpt.items() if k in model_shapes and v.shape == model_shapes[k]}
            skipped  = [k for k in ckpt if k not in filtered]
            scout_actor.load_state_dict(filtered, strict=False)
            if skipped:
                print(f"   ↳ Přeskočeno (nekompatibilní tvar): {skipped}")
        else:
            print(f"⚠️  No pre-trained scout model found — training from scratch.")
    else:
        scout_actor = None

    # CommanderActor — processes own state + scout messages (attention) via LSTM
    if N_FIXED > 0:
        commander_actor = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=scout_msg_dim).to(device)
        if os.path.exists(path_to_commander):
            print(f"📥 Loading pre-trained commander model from {path_to_commander}")
            ckpt = torch.load(path_to_commander, map_location=device)
            model_shapes = {k: v.shape for k, v in commander_actor.state_dict().items()}
            filtered = {k: v for k, v in ckpt.items() if k in model_shapes and v.shape == model_shapes[k]}
            skipped  = [k for k in ckpt if k not in filtered]
            commander_actor.load_state_dict(filtered, strict=False)
            if skipped:
                print(f"   ↳ Přeskočeno (nekompatibilní tvar): {skipped}")
        else:
            print(f"⚠️  No pre-trained commander model found — training from scratch.")
    else:
        commander_actor = None

    # MAPPOCritic — shared value network, takes global state as input
    critic = MAPPOCritic(global_state_dim).to(device)
    if os.path.exists(path_to_critic):
        critic.load_state_dict(torch.load(path_to_critic, map_location=device))
        print(f"📥 Loading critic from {path_to_critic}")

    # Adam optimiser with per-network learning rates
    # (using parameter groups so we can fine-tune scout at a lower LR)
    optim_groups = [{"params": critic.parameters(), "lr": lr_critic}]
    if scout_actor:     optim_groups.append({"params": scout_actor.parameters(),     "lr": lr_scout})
    if commander_actor: optim_groups.append({"params": commander_actor.parameters(), "lr": lr_commander})
    optimizer = optim.Adam(optim_groups)

    # Jednorázový reset Adam state — vymaže momentum z předchozího tréninku.
    # Pokud načítáme checkpoint, optimizer neví nic o předchozích gradientech,
    # ale pokud pokračujeme v té samé session, momentum nese "paměť kroužení".
    # Reset zachová váhy sítě, pouze vymaže m a v buffery.
    for state in optimizer.state.values():
        state.clear()
    num_batches = num_episodes // episodes_per_batch
    # CosineAnnealingWarmRestarts: LR cyklicky klesá a resetuje se každých T_0 batchí.
    # LinearLR flatlinoval na 3e-5 od batche 500 navždy — policy se přestala učit.
    # T_0=200: restart každých 200 batchí; T_mult=2: každý restart 2× delší cyklus.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-5
    )

    # ==========================================================================
    # 4. TRAINING HISTORY  (for plotting)
    # ==========================================================================
    episode_rewards_history = []   # total reward per episode
    loss_history            = []   # combined PPO loss per update epoch
    entropy_history         = []   # policy entropy (exploration measure)
    v_loss_history          = []   # critic (value) loss per update epoch
    lifespan_history        = []   # average agent lifespan per episode
    jerk_history            = []   # Track action smoothness
    cmdr_act_mean_history   = []   # per-batch mean of each commander action dim
    cmdr_act_std_history    = []   # per-batch std  of each commander action dim

    os.makedirs("saved_models", exist_ok=True)
    best_avg_reward = -1000.0

    # ==========================================================================
    # Autopilot — probability-based demonstration injection
    # ==========================================================================
    # Instead of a hard on/off, we send a per-episode probability to workers.
    # Each episode independently rolls dice — so some episodes are teacher-forced
    # and some are pure policy, always. This avoids the feedback loop where a
    # fully-triggered autopilot produces corrupted gradients for 25 batches.
    #
    # Two levels:
    #   BASELINE_PROB  — always active, keeps a trickle of demonstrations
    #                    so the commander never completely forgets how to fly
    #   BOOST_PROB     — activates when rolling reward < BOOST_THRESHOLD,
    #                    temporarily increases demonstration rate
    AUTOPILOT_BASELINE_PROB  = 0.25   # 1 in 4 episodes is a demonstration (always)
    AUTOPILOT_BOOST_PROB     = 0.75   # 3 in 4 episodes when policy is struggling
    AUTOPILOT_BOOST_THRESHOLD = 15000 # rolling reward below this = struggling
                                       # 5000 bylo příliš nízko — autopilot sám dosáhl 5000+
                                       # (dense water +2/krok) a po 3 batchích přepnul na 25 %,
                                       # ačkoli policy se ještě nic nenaučila. 15000 může
                                       # dosáhnout pouze policy která skutečně umí hasit.
    AUTOPILOT_WINDOW          = 200   # zvýšeno z 50 — širší okno zabrání přepnutí na základě
                                       # několika šťastných autopilot epizod
    worker_config['autopilot_prob'] = AUTOPILOT_BASELINE_PROB

    # ==========================================================================
    # 5. MAIN PARALLEL TRAINING LOOP
    # ==========================================================================
    # The ProcessPoolExecutor keeps a persistent pool of worker processes alive
    # for the entire training run — no spawn overhead per batch.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_idx in range(1, num_batches + 1):
            batch_start = time.time()
            rollout_start = time.time()

            # Entropy annealing: start at 0.005, decay to 0.0005 over 500 batches.
            # Sníženo z 0.01 → 0.002 — lifespan roste reálně, policy exploruje dost,
            # entropy bonus byl příliš dominantní a způsoboval stoupající logstd.
            _entropy_anneal_batches = 500
            current_entropy_coef = max(
                0.0005,
                0.005 - (0.005 - 0.0005) * min(batch_idx - 1, _entropy_anneal_batches) / _entropy_anneal_batches
            )

            # --------------------------------------------------------------
            # Adaptive autopilot decision (evaluated before each batch)
            # --------------------------------------------------------------
             # -- Autopilot probability update (evaluated once per batch) ------
            if len(episode_rewards_history) >= AUTOPILOT_WINDOW:
                recent_reward = np.mean(episode_rewards_history[-AUTOPILOT_WINDOW:])
                if recent_reward < AUTOPILOT_BOOST_THRESHOLD:
                    worker_config['autopilot_prob'] = AUTOPILOT_BOOST_PROB
                else:
                    worker_config['autopilot_prob'] = AUTOPILOT_BASELINE_PROB
            else:
                worker_config['autopilot_prob'] = AUTOPILOT_BOOST_PROB  # early training
                
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
            avg_batch    = np.mean(batch_rewards)
            avg_lifespan = np.mean(w_lifespans) if w_lifespans else 0
            max_lifespan = np.max(w_lifespans)  if w_lifespans else 0
            avg_reward   = np.mean(episode_rewards_history[-15:]) if len(episode_rewards_history) >= 15 else np.mean(episode_rewards_history)
            print(f"{datetime.datetime.now()} | Batch {batch_idx:04d} (Ep {episodes_played:04d}) | Avg Batch Reward: {avg_batch:.2f} | Rolling avg (15): {avg_reward:.2f} | Lifespan avg/max: {avg_lifespan:.0f}/{max_lifespan:.0f} | Rollout: {rollout_time:.1f}s")

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
                torch.save(critic.state_dict(), "saved_models/critic_best.pt")
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

            if len(batch_critic["returns"]) <= 0: continue

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

            print(f"{datetime.datetime.now()} | 🛠️  Running PPO update ({len(cr_returns)} samples in Minibatches)...")

            # --- Compute advantages ---
            # WHAT IS THE ADVANTAGE A(s, a)?
            # The advantage answers: "How much better (or worse) was the
            # action I actually took compared to the average action I could
            # have taken in this state?"
            #   A(s, a) = G - V(s)
            #
            # .detach() on cr_values: the advantage is used to scale the
            # POLICY gradient, not the critic gradient.  Detaching stops
            # gradients from flowing back through the critic via the advantage
            # term (the critic has its own separate MSE loss below).
            cr_adv = cr_returns.unsqueeze(1) - cr_values.detach()  # A(s,a) = G - V(s)
            cr_adv = (cr_adv - cr_adv.mean()) / (cr_adv.std() + 1e-8)  # normalise to N(0,1)

            # ==============================================================
            # PREPARE SEQUENCES FOR MINIBATCHING
            # ==============================================================
            # WHY DO WE RESHAPE EVERYTHING HERE?
            # The raw tensors are flat: [episodes * max_steps, features].
            # If we just randomly sliced this flat tensor, we would break the
            # temporal continuity that the GRU (Recurrent Neural Network) needs.
            # To slice the data into minibatches SAFELY, we reshape the tensors 
            # to 3D/4D: [episodes, max_steps, features]. 
            # This way, when we select a minibatch of indices, we grab whole, 
            # unbroken episodes (trajectories) that the GRU can process correctly.
            
            num_minibatches = 4   # 4 minibatches — better gradient estimates per batch
            mb_size = max(1, episodes // num_minibatches)
            
            # Reshape advantages: [episodes, max_steps, num_agents]
            cr_adv_seq = cr_adv.view(episodes, max_steps, num_agents)
            s_adv_seq  = cr_adv_seq[:, :, 0]
            c_adv_seq  = cr_adv_seq[:, :, 1] if num_agents > 1 else cr_adv_seq[:, :, 0]

            # Reshape Scout tensors (if scouts exist)
            if scout_actor is not None:
                s_maps_seq     = s_maps.view(episodes, max_steps, 1, 32, 32)
                s_self_seq     = s_self.view(episodes, max_steps, -1)
                s_neigh_s_seq  = s_neigh_s.view(episodes, max_steps, -1, 3)
                s_neigh_m_seq  = s_neigh_m.view(episodes, max_steps, -1)
                s_actions_seq  = s_actions.view(episodes, max_steps, -1)
                s_logprobs_seq = s_logprobs.view(episodes, max_steps)
                
                # Hidden state: [1, episodes, dim] -> [episodes, 1, dim] 
                # Transposed so we can slice it easily by episode index.
                h_scout_seq    = h_scout.transpose(0, 1)  
                
            else:
                pass  # no scouts — jerk will be computed from commander below

            # Reshape Commander tensors (if commanders exist)
            if commander_actor is not None:
                c_fixed_seq    = c_fixed.view(episodes, max_steps, -1)
                c_msgs_seq     = c_msgs.view(episodes, max_steps, c_msgs.size(-2), c_msgs.size(-1))
                c_msg_m_seq    = c_msg_m.view(episodes, max_steps, -1)
                c_actions_seq  = c_actions.view(episodes, max_steps, -1)
                c_logprobs_seq = c_logprobs.view(episodes, max_steps)
                h_cmdr_seq     = h_cmdr.transpose(0, 1)

                # ----------------------------------------------------------
                # COMMANDER DIAGNOSTICS (per-batch action stats)
                # ----------------------------------------------------------
                # Measure action smoothness (jerk) and per-dim mean/std.
                # Jerk ≈ 0 means actions barely change → policy may be stuck.
                # Std ≈ 0 for a dim means network saturated / output collapsed.
                with torch.no_grad():
                    acts_flat = c_actions_seq.reshape(-1, c_actions_seq.size(-1))
                    cmdr_act_mean_history.append(acts_flat.mean(0).cpu().numpy())
                    cmdr_act_std_history.append(acts_flat.std(0).cpu().numpy())
                    c_diff = torch.abs(c_actions_seq[:, 1:] - c_actions_seq[:, :-1])
                    jerk_history.append(c_diff.mean().item())

            # Reshape Critic tensors
            # The critic processes all agents. We transpose so the shape is
            # [episodes, agents, max_steps, dim] before slicing.
            cr_g_seq_base   = cr_g_states.view(episodes, max_steps, num_agents, -1).transpose(1, 2) 
            cr_ret_seq_base = cr_returns.view(episodes, max_steps, num_agents).transpose(1, 2)
            
            if h_critic_scout is not None: h_c_scout_seq = h_critic_scout.transpose(0, 1)
            if h_critic_cmdr is not None:  h_c_cmdr_seq  = h_critic_cmdr.transpose(0, 1)


            # ==============================================================
            # GRADIENT UPDATE LOOP (Epochs & Minibatches)
            # ==============================================================
            # We run update_epochs gradient passes over the SAME data.
            # However, instead of passing all 150,000 samples at once (which
            # destroys CPU/GPU caches and memory), we divide the episodes
            # into smaller chunks (minibatches).
            for epoch in range(update_epochs):
                # Shuffle episode indices for this epoch to prevent cyclic updates
                b_inds = np.random.permutation(episodes)
                
                # Accumulators to average metrics across minibatches for logging
                epoch_loss, epoch_vloss, epoch_entropy = 0.0, 0.0, 0.0
                
                for start in range(0, episodes, mb_size):
                    end = start + mb_size
                    mb_inds = b_inds[start:end]
                    curr_mb_size = len(mb_inds)
                    
                    entropy_sum = torch.tensor(0.0, device=device)
                    loss = torch.tensor(0.0, device=device)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # 1. SCOUT POLICY LOSS (PPO-clip objective)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if scout_actor is not None:
                        # Slice out this minibatch's sequences
                        mb_maps    = s_maps_seq[mb_inds]
                        mb_self    = s_self_seq[mb_inds]
                        mb_neigh_s = s_neigh_s_seq[mb_inds]
                        mb_neigh_m = s_neigh_m_seq[mb_inds]
                        mb_acts    = s_actions_seq[mb_inds]
                        mb_old_lp  = s_logprobs_seq[mb_inds].view(-1)
                        mb_adv     = s_adv_seq[mb_inds].reshape(-1)
                        mb_adv     = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)  # normalize
                        
                        # Restore GRU hidden state shape: [1, curr_mb_size, dim]
                        mb_h_s     = h_scout_seq[mb_inds].transpose(0, 1) 
                        
                        # Forward pass on the minibatch
                        dist, s_msgs, _ = scout_actor(mb_maps, mb_self, mb_neigh_s, mb_neigh_m, mb_h_s)
                        
                        flat_acts = mb_acts.view(-1, mb_acts.size(-1))
                        new_lp_s  = dist.log_prob(flat_acts).sum(1)
                        
                        # IMPORTANCE SAMPLING RATIO: ratio = π_new(a|s) / π_old(a|s)
                        # Clamp log-ratio to [-10, 10] before exp to prevent overflow to inf
                        # (which would cause inf loss → NaN gradients → NaN weights).
                        log_ratio_s = (new_lp_s - mb_old_lp).clamp(-10.0, 10.0)
                        ratio_s = torch.exp(log_ratio_s)
                        
                        # PPO CLIPPED OBJECTIVE
                        # Taking max(pg1, pg2) -- the PESSIMISTIC objective --
                        # Net result: PPO clips the BENEFIT of good updates AND
                        # the PUNISHMENT of bad ones, keeping the policy close to
                        # what it was when the data was collected.
                        pg1_s   = -mb_adv * ratio_s
                        pg2_s   = -mb_adv * torch.clamp(ratio_s, 1 - clip_coef, 1 + clip_coef)
                        policy_loss_s = torch.max(pg1_s, pg2_s).mean()
                        
                        loss += policy_loss_s
                        entropy_sum += dist.entropy().sum(1).mean()

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # 2. COMMANDER POLICY LOSS (identical PPO-clip logic)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if commander_actor is not None:
                        mb_fixed   = c_fixed_seq[mb_inds]
                        mb_msgs    = c_msgs_seq[mb_inds]
                        mb_msg_m   = c_msg_m_seq[mb_inds]
                        mb_acts    = c_actions_seq[mb_inds]
                        mb_old_lp  = c_logprobs_seq[mb_inds].view(-1)
                        mb_adv     = c_adv_seq[mb_inds].reshape(-1)
                        mb_adv     = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)  # normalize
                        mb_h_c     = h_cmdr_seq[mb_inds].transpose(0, 1)
                        
                        dist, _, _ = commander_actor(mb_fixed, mb_msgs, mb_msg_m, mb_h_c)
                        
                        flat_acts = mb_acts.view(-1, mb_acts.size(-1))
                        new_lp_c  = dist.log_prob(flat_acts).sum(1)
                        entropy_sum += dist.entropy().sum(1).mean()
                        
                        log_ratio_c = (new_lp_c - mb_old_lp).clamp(-10.0, 10.0)
                        ratio_c = torch.exp(log_ratio_c)
                        pg1_c   = -mb_adv * ratio_c
                        pg2_c   = -mb_adv * torch.clamp(ratio_c, 1 - clip_coef, 1 + clip_coef)
                        policy_loss_c = torch.max(pg1_c, pg2_c).mean()
                        
                        loss += policy_loss_c

                    # Entropy regularizace — odečteme jednou za oba agenty
                    loss -= current_entropy_coef * entropy_sum

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # 3. CRITIC VALUE LOSS
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # RESHAPE EXPLANATION:
                    # We sliced the critic sequence as [mb_size, num_agents, max_steps, dim].
                    # The critic GRU needs a contiguous sequence per episode/agent.
                    # We reshape it back into [mb_size * num_agents, max_steps, dim]
                    # so that each "row" in the batch is ONE agent's full trajectory.
                    mb_cr_g = cr_g_seq_base[mb_inds].reshape(curr_mb_size * num_agents, max_steps, -1)
                    mb_cr_ret = cr_ret_seq_base[mb_inds].reshape(-1, 1)  # Target returns

                    # Build the interleaved hidden state for this minibatch
                    mb_h_parts = []
                    if h_critic_scout is not None: mb_h_parts.append(h_c_scout_seq[mb_inds])
                    if h_critic_cmdr is not None:  mb_h_parts.append(h_c_cmdr_seq[mb_inds])
                    
                    # Stack along the agent dimension and reshape for GRU
                    mb_h_critic = torch.cat(mb_h_parts, dim=1).view(1, curr_mb_size * num_agents, -1)
                    
                    new_values, _ = critic(mb_cr_g, mb_h_critic)
                    value_loss = nn.MSELoss()(new_values, mb_cr_ret)
                    
                    # COMBINED LOSS (standard MAPPO objective)
                    # L = L_policy + 0.5 * L_value - <0.01; 0.0001> * H
                    loss += 0.5 * value_loss

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # 4. BACKPROPAGATION
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # 1. zero_grad(): Clear gradients from the previous minibatch.
                    # 2. backward(): Compute d(loss)/d(param) for all networks.
                    # 3. clip_grad_norm_(): Prevent "exploding gradients" by
                    #    capping the L2 norm of the gradients at 0.5.
                    # 4. step(): Apply the gradients via the Adam optimiser.
                    if not torch.isfinite(loss):
                        print(f"⚠️  Non-finite loss ({loss.item():.4e}) — skipping minibatch to prevent NaN weights")
                        optimizer.zero_grad()
                        continue
                    optimizer.zero_grad()
                    loss.backward()
                    params = list(critic.parameters())
                    if scout_actor:     params += list(scout_actor.parameters())
                    if commander_actor: params += list(commander_actor.parameters())
                    nn.utils.clip_grad_norm_(params, max_norm=0.5)
                    optimizer.step()
                    # Sanity-check: abort immediately if weights became NaN
                    if any(torch.isnan(p.data).any() for p in params):
                        raise RuntimeError("NaN weights after optimizer step — training unstable. "
                                           "Consider reducing lr or checking reward scale.")
                    
                    # Accumulate metrics for logging
                    epoch_loss += loss.item()
                    epoch_vloss += value_loss.item()
                    epoch_entropy += entropy_sum.detach().item()

                # Average metrics over the minibatches and append to history (once per epoch)
                loss_history.append(epoch_loss / num_minibatches)
                v_loss_history.append(epoch_vloss / num_minibatches)
                entropy_history.append(epoch_entropy / num_minibatches)

            scheduler.step()
            total_time = time.time() - batch_start
            
            # Získáme zprůměrované metriky z poslední epochy pro přesnější logování
            final_loss = loss_history[-1] if loss_history else 0.0
            final_vloss = v_loss_history[-1] if v_loss_history else 0.0
            final_entropy = entropy_history[-1] if entropy_history else 0.0
            current_lr = scheduler.get_last_lr()[0]
            print(f"{datetime.datetime.now()} | ✅ PPO update complete. Loss: {loss.item():.4f} | Entropy Coef: {current_entropy_coef:.4f} | LR: {current_lr:.6f} | Total batch: {total_time:.1f}s")
            
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
                torch.save(critic.state_dict(),              f"saved_models/critic_ep{episodes_played}.pt")

                # --- 2×3 training dashboard ---
                fig, axes = plt.subplots(2, 3, figsize=(21, 10))

                # (0,0) Episode rewards over time
                ax = axes[0, 0]
                ax.plot(episode_rewards_history, label="Reward", alpha=0.3, color='green')
                if len(episode_rewards_history) > 20:
                    ax.plot(np.convolve(episode_rewards_history, np.ones(20)/20, mode='valid'),
                             label="MA 20", color='darkgreen')
                ax.set_title("Reward over episodes")
                ax.grid(True, alpha=0.3)
                ax.legend()

                # (0,1) Loss curves (log scale)
                ax = axes[0, 1]
                ax.plot(loss_history,   label="Total Loss",         color='red',  alpha=0.5)
                ax.plot(v_loss_history, label="Value Loss (Critic)", color='blue', alpha=0.5)
                ax.set_title("Loss (log scale)")
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                ax.legend()

                # (0,2) Policy entropy
                ax = axes[0, 2]
                if entropy_history:
                    ax.plot(entropy_history, color='purple')
                    ax.set_title("Policy Entropy (exploration)")
                else:
                    ax.text(0.5, 0.5, 'No entropy data yet', ha='center', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)

                # (1,0) Average agent lifespan
                ax = axes[1, 0]
                ax.plot(lifespan_history, color='orange', alpha=0.3, linewidth=0.6)
                if len(lifespan_history) > 20:
                    ma_life = np.convolve(lifespan_history, np.ones(20)/20, mode='valid')
                    ax.plot(range(19, len(lifespan_history)), ma_life,
                             color='darkorange', linewidth=1.5, label="MA 20")
                    ax.legend(fontsize=8)
                ax.set_xlabel("Episodes")
                ax.set_ylabel("Steps")
                ax.set_title("Avg agent lifespan (steps)")

                # (1,1) Commander Action Jerk
                ax = axes[1, 1]
                if jerk_history:
                    ax.plot(jerk_history, color='teal')
                    ax.set_title("Cmdr Action Jerk (Δ between steps)")
                    ax.set_xlabel("Batches")
                    ax.set_ylabel("Avg |Δ action|")
                else:
                    ax.text(0.5, 0.5, 'No jerk data', ha='center', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)

                # (1,2) Commander action mean per dim  (+/- std as shaded band)
                ax = axes[1, 2]
                ACTION_LABELS = ["Roll", "Pitch", "Throttle", "Water"]
                ACTION_COLORS = ["steelblue", "tomato", "forestgreen", "goldenrod"]
                if cmdr_act_mean_history:
                    means = np.array(cmdr_act_mean_history)   # [batches, 4]
                    stds  = np.array(cmdr_act_std_history)    # [batches, 4]
                    xs    = np.arange(len(means))
                    for i, (lbl, col) in enumerate(zip(ACTION_LABELS, ACTION_COLORS)):
                        ax.plot(xs, means[:, i], label=lbl, color=col)
                        ax.fill_between(xs,
                                        means[:, i] - stds[:, i],
                                        means[:, i] + stds[:, i],
                                        alpha=0.15, color=col)
                    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
                    ax.set_ylim(-1.2, 1.2)
                    ax.set_title("Cmdr Action Mean ± Std per Dim")
                    ax.set_xlabel("Batches")
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No cmdr action data', ha='center', transform=ax.transAxes)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig("final_training_plot.png")
                plt.close()

if __name__ == "__main__":
    train()