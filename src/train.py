import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import cProfile, pstats, io
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend so plots can be saved without a display
import matplotlib.pyplot as plt

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
    num_episodes = 2500       # total episodes to train for
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