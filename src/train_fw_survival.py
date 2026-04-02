"""
train_fw_survival.py — Phase 1: Teach the fixed-wing to SURVIVE (fly without crashing)
======================================================================================

Differences from train_fw_flight.py:
  1. Uses SimpleFWActor (~6K params) instead of CommanderActor (~118K params)
  2. NO autopilot — the network discovers flight from pure RL exploration
  3. NO curriculum — simple single-phase training
  4. action_logstd init = -0.5  (std ≈ 0.6, not 0.135)
  5. env_core.py smoothing reduced from α=0.2 to α=0.5
  6. Pitch clamp REMOVED — network controls full range
  7. Option to train WITHOUT the critic (advantages = raw returns)

The ONLY reward signal:
  - +0.3 per step alive  (from env_core.py _apply_physics_shaping)
  - Boundary penalty      (quadratic, near edges)
  - Altitude penalty       (outside [40, 150] m)
  - Crash penalty = -50
  - +50 for surviving all steps
  - Patrol bonus: tiny pull toward map centre (max 0.05/step)

Max achievable reward per episode (2000 steps):
  2000 × 0.35 + 50 = 750 points

Usage:
  cd src/
  python train_fw_survival.py
  python train_fw_survival.py --no-critic     # test without critic baseline
  python train_fw_survival.py --resume-episodes 5000 --resume-actor saved_models/fw_survival/actor_best.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import time
from concurrent.futures import ProcessPoolExecutor


# =============================================================================
# WORKER FUNCTION  (runs on CPU, no scouts)
# =============================================================================

def collect_survival_worker(num_eps, actor_w, critic_w, config, batch_start_idx):
    """
    Collect rollout episodes for the fixed-wing survival task.
    No autopilot, no scouts — pure RL exploration.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import torch
    torch.set_num_threads(1)
    import numpy as np

    from env_core import DroneFireEnv
    from models import SimpleFWActor, MAPPOCritic

    max_steps = config['max_steps']
    hidden_dim = config['hidden_dim']
    use_critic = config.get('use_critic', True)

    # --- Build local networks ---
    local_actor = SimpleFWActor(
        self_state_dim=config['fixed_self_dim'],
        action_dim=3,
        hidden_dim=hidden_dim,
    )
    local_actor.load_state_dict(actor_w)
    local_actor.eval()

    local_critic = None
    if use_critic and critic_w is not None:
        local_critic = MAPPOCritic(config['fixed_self_dim'], hidden_dim=128)
        local_critic.load_state_dict(critic_w)
        local_critic.eval()

    # --- Environment ---
    local_env = DroneFireEnv(
        num_quads=0, num_fixed=1,
        grid_size_m=3000.0, max_steps=max_steps
    )

    # --- Buffers ---
    actor_buf = {k: [] for k in ["states", "actions", "logprobs", "returns", "alive"]}
    critic_buf = {k: [] for k in ["g_states", "values", "returns", "alive"]}

    actor_h0_list = []
    critic_h0_list = []
    all_rewards = []
    all_lifespans = []
    all_deaths = []

    for ep_off in range(num_eps):
        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_off)

        # Kill fire immediately — this is a flight-only task
        if local_env.sim.environment.fire_grid is not None:
            local_env.sim.environment.fire_grid.B[:] = False
            local_env.sim.environment.fire_grid.I[:] = 0.0
            local_env.sim.drone_extinguish_stats = {}

        f_agent = local_env.fixed_agents[0]

        actor_h = torch.zeros(1, 1, hidden_dim)
        critic_h = torch.zeros(1, 1, 128)

        actor_h0_list.append(actor_h.clone())
        critic_h0_list.append(critic_h.clone())

        ep_data = []
        lifespan = max_steps
        ep_reward = 0.0

        for step in range(max_steps):
            g_state = local_env.state()
            # For survival training: use only the commander self-state (17-d)
            # as critic input. The full global_state is 273-d where 256-d
            # fire map is always zero (fire killed). Feeding 93% zeros to
            # the critic wastes capacity and makes learning harder.
            g_tensor = None  # set below from self_state

            if f_agent not in local_env.agents:
                # Agent is dead — pad with zeros
                g_tensor = torch.zeros(1, config['fixed_self_dim'])
                ep_data.append({
                    "dead": True, "gs": g_tensor,
                    "val": torch.tensor([[0.0]]), "ret": 0.0, "reward": 0.0,
                })
                continue

            with torch.no_grad():
                s_st = torch.FloatTensor(obs[f_agent]["self_state"]).unsqueeze(0)
                g_tensor = s_st  # critic sees same state as actor
                dist_out, _, h_out = local_actor(s_st, None, None, actor_h)
                act = dist_out.sample()

                if use_critic and local_critic is not None:
                    val, c_h_out = local_critic(g_tensor, critic_h)
                else:
                    val = torch.tensor([[0.0]])
                    c_h_out = critic_h

            ep_data.append({
                "dead": False,
                "self": s_st,
                "h": actor_h,
                "act": act,
                "lp": dist_out.log_prob(act).sum(1),
                "val": val,
                "gs": g_tensor,
                "reward": 0.0,
            })

            actor_h = h_out
            critic_h = c_h_out

            actions = {f_agent: act.squeeze(0).numpy()}
            if local_env.agents:
                obs, rewards_env, terms, _, infos_env = local_env.step(actions)
                r = rewards_env.get(f_agent, 0.0)
                ep_data[-1]["reward"] = r
                ep_reward += r
                if terms.get(f_agent, False) and lifespan == max_steps:
                    lifespan = step
                    ep_data[-1]["death_cause"] = infos_env.get(f_agent, {}).get("death_cause", "unknown")

        # --- GAE ---
        gamma = config.get('gamma', 0.99)
        gae_lam = config.get('gae_lambda', 0.95)
        gae = 0.0
        z_val = torch.tensor([[0.0]])

        for i in reversed(range(max_steps)):
            d = ep_data[i]
            v_t = d.get("val", z_val).item()
            r_t = d.get("reward", 0.0)
            v_np = ep_data[i + 1].get("val", z_val).item() if i < max_steps - 1 else 0.0
            delta = r_t + gamma * v_np - v_t
            gae = delta + gamma * gae_lam * gae
            d["ret"] = gae + v_t

        # --- Pack buffers ---
        d_state_zero = torch.zeros(1, config['fixed_self_dim'])

        for d in ep_data:
            alive = 0.0 if d["dead"] else 1.0
            critic_buf["g_states"].append(d["gs"])
            critic_buf["values"].append(d.get("val", z_val))
            critic_buf["returns"].append(d.get("ret", 0.0))
            critic_buf["alive"].append(alive)

            if d["dead"]:
                actor_buf["states"].append(d_state_zero)
                actor_buf["actions"].append(torch.zeros(1, 3))
                actor_buf["logprobs"].append(torch.tensor([0.0]))
            else:
                actor_buf["states"].append(d["self"])
                actor_buf["actions"].append(d["act"])
                actor_buf["logprobs"].append(d["lp"])
            actor_buf["returns"].append(d.get("ret", 0.0))
            actor_buf["alive"].append(alive)

        # Find death cause for this episode
        death = "survived"
        for d in ep_data:
            if "death_cause" in d:
                death = d["death_cause"]
                break

        all_rewards.append(ep_reward)
        all_lifespans.append(float(lifespan))
        all_deaths.append(death)

    # --- Final cat ---
    def _cat(lst):
        return torch.cat(lst, dim=0)

    out_actor = {
        "states": _cat(actor_buf["states"]),
        "actions": _cat(actor_buf["actions"]),
        "logprobs": _cat(actor_buf["logprobs"]),
        "returns": torch.tensor(actor_buf["returns"], dtype=torch.float32),
        "alive": torch.tensor(actor_buf["alive"], dtype=torch.float32),
    }
    out_critic = {
        "g_states": _cat(critic_buf["g_states"]),
        "values": _cat(critic_buf["values"]),
        "returns": torch.tensor(critic_buf["returns"], dtype=torch.float32),
        "alive": torch.tensor(critic_buf["alive"], dtype=torch.float32),
    }
    out_init_h = {
        "actor": torch.cat(actor_h0_list, dim=0),
        "critic": torch.cat(critic_h0_list, dim=0),
    }

    return out_actor, out_critic, out_init_h, all_rewards, all_lifespans, all_deaths


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_fw_survival(resume_episodes=0, resume_actor="", resume_critic="",
                      use_critic=True):
    print("=" * 70)
    print("  Fixed-Wing SURVIVAL Training (SimpleFWActor)")
    print(f"  Critic: {'ON' if use_critic else 'OFF (advantages = raw returns)'}")
    if resume_episodes > 0:
        print(f"  RESUME from episode {resume_episodes}")
    print("=" * 70)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from env_core import DroneFireEnv
    from models import SimpleFWActor, MAPPOCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}\n")

    # ── Hyperparameters ──────────────────────────────────────────────────────
    num_episodes = 30_000
    max_steps = 1000      # Goldilocks zone for 3km map:
                          # - 500 was too easy (constant orbit survived 100%)
                          # - 2000 was too hard (0% survival, no positive examples)
                          # - 1000: orbit survives ~30-50% → clear gradient both ways
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    update_epochs = 4
    num_workers = 15
    eps_per_worker = 2
    episodes_per_batch = num_workers * eps_per_worker  # 30

    lr_actor = 3e-4
    lr_critic = 3e-4
    hidden_dim = 64      # SimpleFWActor hidden dim

    # Entropy: moderate — 0.02 was too aggressive and pushed std to ceiling,
    # causing the catastrophic collapse at batch ~270.  With max_std=0.5 and
    # entropy_start=0.005 the bonus encourages exploration without saturating.
    entropy_start = 0.005
    entropy_end = 0.001
    entropy_anneal_batches = 300

    # ── Dims ─────────────────────────────────────────────────────────────────
    temp_env = DroneFireEnv(num_quads=0, num_fixed=1, grid_size_m=3000.0, max_steps=max_steps)
    fixed_self_dim = temp_env.observation_space(temp_env.fixed_agents[0])["self_state"].shape[0]
    global_state_dim = temp_env.state_space.shape[0]
    if hasattr(temp_env, 'sim') and temp_env.sim is not None:
        temp_env.sim.stop_simulation()

    print(f"fixed_self_dim   = {fixed_self_dim}")
    print(f"global_state_dim = {global_state_dim} (NOT used by survival critic)")
    print(f"critic_input_dim = {fixed_self_dim} (self_state only)")
    print(f"hidden_dim       = {hidden_dim}")
    print(f"use_critic       = {use_critic}")

    worker_config = {
        'N_QUADS': 0, 'N_FIXED': 1,
        'grid_size_m': 3000.0,
        'max_steps': max_steps,
        'fixed_self_dim': fixed_self_dim,
        'scout_msg_dim': 5,
        'global_state_dim': global_state_dim,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'use_critic': use_critic,
        'hidden_dim': hidden_dim,
    }

    # ── Networks ─────────────────────────────────────────────────────────────
    actor = SimpleFWActor(
        self_state_dim=fixed_self_dim,
        action_dim=3,
        hidden_dim=hidden_dim,
    ).to(device)

    # Count params
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"SimpleFWActor params: {n_params:,}")

    if resume_actor and os.path.exists(resume_actor):
        actor.load_state_dict(torch.load(resume_actor, map_location=device))
        print(f"  Loaded actor from {resume_actor}")

    # Survival critic sees self_state only (17-d), not full 273-d global state
    critic_input_dim = fixed_self_dim
    critic = MAPPOCritic(critic_input_dim, hidden_dim=128).to(device)
    if resume_critic and os.path.exists(resume_critic):
        critic.load_state_dict(torch.load(resume_critic, map_location=device))
        print(f"  Loaded critic from {resume_critic}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Separate logstd into its own param group with higher LR so it can
    # adapt faster.  With shared LR, logstd barely moved (0.35→0.36 in 20
    # batches) because its gradient is small relative to the network weights.
    actor_main_params = [p for n, p in actor.named_parameters() if n != 'action_logstd']
    param_groups = [
        {"params": actor_main_params, "lr": lr_actor},
        {"params": [actor.action_logstd], "lr": lr_actor * 3},  # 3× faster for std
    ]
    if use_critic:
        param_groups.append({"params": critic.parameters(), "lr": lr_critic})
    optimizer = optim.Adam(param_groups)

    # ── Tracking ─────────────────────────────────────────────────────────────
    reward_history = []
    loss_history = []
    lifespan_history = []
    logstd_history = []

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_models", "fw_survival")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints → {save_dir}\n")

    best_avg = -1e9
    batches_since_best = 0
    patience = 100  # stop if no improvement for 100 batches (~3000 episodes)
    episodes_played = resume_episodes
    num_batches = num_episodes // episodes_per_batch

    # ── Main training loop ───────────────────────────────────────────────────
    for batch_idx in range(1, num_batches + 1):

        # Entropy annealing
        entropy_coef = max(
            entropy_end,
            entropy_start - (entropy_start - entropy_end)
                * min(batch_idx - 1, entropy_anneal_batches) / entropy_anneal_batches
        )

        # Snapshot weights
        actor_w = {k: v.cpu() for k, v in actor.state_dict().items()}
        critic_w = {k: v.cpu() for k, v in critic.state_dict().items()} if use_critic else None

        # --- Rollout ---
        t0 = time.time()
        batch_actor = {k: [] for k in ["states", "actions", "logprobs", "returns", "alive"]}
        batch_critic = {k: [] for k in ["g_states", "values", "returns", "alive"]}
        batch_h_actor = []
        batch_h_critic = []
        batch_rewards = []
        batch_deaths = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    collect_survival_worker,
                    eps_per_worker, actor_w, critic_w,
                    worker_config,
                    episodes_played + i * eps_per_worker
                )
                for i in range(num_workers)
            ]
            for fut in futures:
                w_actor, w_critic, w_h, w_rew, w_life, w_deaths = fut.result()
                batch_rewards.extend(w_rew)
                batch_deaths.extend(w_deaths)
                lifespan_history.extend(w_life)
                reward_history.extend(w_rew)
                episodes_played += len(w_rew)

                for k in batch_actor:
                    batch_actor[k].append(w_actor[k])
                for k in batch_critic:
                    batch_critic[k].append(w_critic[k])
                batch_h_actor.append(w_h["actor"])
                batch_h_critic.append(w_h["critic"])

        rollout_time = time.time() - t0

        avg_batch = float(np.mean(batch_rewards))
        win = min(60, len(reward_history))
        avg_roll = float(np.mean(reward_history[-win:]))

        # Log per-dimension std and alive fraction
        with torch.no_grad():
            cur_stds = torch.exp(actor.action_logstd.clamp(-3.0, 0.0)).squeeze()
            cur_std = cur_stds.mean().item()
        logstd_history.append(cur_stds.cpu().numpy().copy())

        alive_frac = torch.cat(batch_actor["alive"]).mean().item()

        # Death cause stats
        from collections import Counter
        dc = Counter(batch_deaths)
        dc_str = " ".join(f"{k}={v}" for k, v in dc.most_common())

        avg_life = float(np.mean(batch_rewards))  # approx from rewards
        recent_life = float(np.mean(lifespan_history[-win:])) if lifespan_history else 0
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"Batch {batch_idx:04d} (Ep {episodes_played:05d}) | "
              f"R: {avg_batch:+7.1f} ({avg_roll:+7.1f})  "
              f"Life: {recent_life:.0f}  "
              f"alive: {alive_frac:.0%}  "
              f"std=[{cur_stds[0]:.2f},{cur_stds[1]:.2f},{cur_stds[2]:.2f}]  "
              f"deaths: [{dc_str}]  "
              f"{rollout_time:.1f}s")

        # Save best
        if episodes_played >= 60 and avg_roll > best_avg:
            best_avg = avg_roll
            batches_since_best = 0
            torch.save(actor.state_dict(), os.path.join(save_dir, "actor_best.pt"))
            if use_critic:
                torch.save(critic.state_dict(), os.path.join(save_dir, "critic_best.pt"))
            print(f"   ⭐ New best! rolling avg = {best_avg:.1f}")
        elif episodes_played >= 60:
            batches_since_best += 1

        if batch_idx % 50 == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_b{batch_idx:04d}.pt"))

        # Early stopping check
        if batches_since_best >= patience and batch_idx >= 50:
            print(f"\n⏹  Early stopping: no improvement for {patience} batches")
            print(f"   Best rolling avg = {best_avg:.1f} at batch ~{batch_idx - batches_since_best}")
            _save_plot(reward_history, loss_history, lifespan_history,
                       logstd_history, save_dir, batch_idx, use_critic)
            break

        # ── PPO UPDATE ───────────────────────────────────────────────────
        a_states = torch.cat(batch_actor["states"]).to(device)
        a_actions = torch.cat(batch_actor["actions"]).to(device)
        a_logprobs = torch.cat(batch_actor["logprobs"]).to(device)
        a_returns = torch.cat(batch_actor["returns"]).to(device)
        a_alive = torch.cat(batch_actor["alive"]).to(device)

        cr_g = torch.cat(batch_critic["g_states"]).to(device)
        cr_vals = torch.cat(batch_critic["values"]).to(device)
        cr_rets = torch.cat(batch_critic["returns"]).to(device)
        cr_alive = torch.cat(batch_critic["alive"]).to(device)

        def _mk_h(lst, dim):
            return (torch.cat(lst, dim=0).squeeze(1).unsqueeze(0).to(device))

        h_actor = _mk_h(batch_h_actor, hidden_dim)
        h_critic = _mk_h(batch_h_critic, 128)

        # Advantages — normalize ONCE using only alive steps
        if use_critic:
            cr_adv = cr_rets.unsqueeze(1) - cr_vals.detach()
        else:
            cr_adv = cr_rets.unsqueeze(1)
        alive_flat = a_alive.view(-1)
        alive_mask_bool = alive_flat > 0.5
        if alive_mask_bool.sum() > 1:
            alive_adv = cr_adv.view(-1)[alive_mask_bool]
            cr_adv_flat = cr_adv.view(-1)
            cr_adv_flat = (cr_adv_flat - alive_adv.mean()) / (alive_adv.std() + 1e-8)
            cr_adv = cr_adv_flat.view(-1, 1)

        # Reshape to sequences
        episodes = episodes_per_batch
        a_states_seq = a_states.view(episodes, max_steps, -1)
        a_actions_seq = a_actions.view(episodes, max_steps, -1)
        a_logprobs_seq = a_logprobs.view(episodes, max_steps)
        a_adv_seq = cr_adv.view(episodes, max_steps, 1).squeeze(-1)
        a_alive_seq = a_alive.view(episodes, max_steps)

        h_actor_seq = h_actor.transpose(0, 1)  # [ep, 1, hidden]

        cr_g_seq = cr_g.view(episodes, max_steps, 1, -1).transpose(1, 2)
        cr_ret_seq = cr_rets.view(episodes, max_steps, 1).transpose(1, 2)
        cr_alive_seq = cr_alive.view(episodes, max_steps)
        h_critic_seq = h_critic.transpose(0, 1)

        # Gradient loop
        num_minibatches = 4
        mb_size = max(1, episodes // num_minibatches)
        batch_loss = 0.0

        for epoch in range(update_epochs):
            b_inds = np.random.permutation(episodes)
            ep_loss = 0.0

            for start in range(0, episodes, mb_size):
                mb = b_inds[start:start + mb_size]
                curr_mb = len(mb)

                # Actor forward
                mb_states = a_states_seq[mb]
                mb_acts = a_actions_seq[mb]
                mb_old_lp = a_logprobs_seq[mb].view(-1)
                mb_adv = a_adv_seq[mb].reshape(-1)
                mb_alive = a_alive_seq[mb].reshape(-1)
                mb_h = h_actor_seq[mb].transpose(0, 1)

                dist, _, _ = actor(mb_states, None, None, mb_h)
                flat_acts = mb_acts.view(-1, 3)
                new_lp = dist.log_prob(flat_acts).sum(1)
                entropy = dist.entropy().sum(1)

                log_ratio = (new_lp - mb_old_lp).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                surr = torch.max(pg1, pg2)

                # MASK: only alive steps contribute to policy loss
                n_alive = mb_alive.sum().clamp(min=1.0)
                policy_loss = (surr * mb_alive).sum() / n_alive
                entropy_loss = (entropy * mb_alive).sum() / n_alive

                loss = policy_loss - entropy_coef * entropy_loss

                # Critic loss — also masked
                if use_critic:
                    mb_cr_g = cr_g_seq[mb].reshape(curr_mb, max_steps, -1)
                    mb_cr_ret = cr_ret_seq[mb].reshape(-1, 1)
                    mb_cr_alive = cr_alive_seq[mb].reshape(-1)
                    mb_h_cr = h_critic_seq[mb].reshape(1, curr_mb, -1)
                    new_vals, _ = critic(mb_cr_g, mb_h_cr)
                    val_err = (new_vals - mb_cr_ret).pow(2).squeeze(1)
                    n_cr_alive = mb_cr_alive.sum().clamp(min=1.0)
                    value_loss = (val_err * mb_cr_alive).sum() / n_cr_alive
                    loss = loss + 0.5 * value_loss

                if not torch.isfinite(loss):
                    print(f"  ⚠️ Non-finite loss — skipping minibatch")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                all_params = list(actor.parameters())
                if use_critic:
                    all_params += list(critic.parameters())
                nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                optimizer.step()
                ep_loss += loss.item()

            batch_loss += ep_loss / max(1, num_minibatches)

        loss_history.append(batch_loss / update_epochs)

        # Plot every 10 batches
        if batch_idx % 10 == 0:
            _save_plot(reward_history, loss_history, lifespan_history,
                       logstd_history, save_dir, batch_idx, use_critic)

    # Final plot
    _save_plot(reward_history, loss_history, lifespan_history,
               logstd_history, save_dir, batch_idx, use_critic)

    print(f"\n✅ Training complete!")
    print(f"   Best actor: {save_dir}/actor_best.pt")


# =============================================================================
# PLOT HELPER
# =============================================================================

def _save_plot(rewards, losses, lifespans, logstds, save_dir, batch_idx, use_critic):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"FW Survival — Batch {batch_idx}  "
                 f"(critic={'ON' if use_critic else 'OFF'})", fontsize=13)

    # Reward
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.2, color='steelblue', linewidth=0.5)
    if len(rewards) >= 30:
        ma = np.convolve(rewards, np.ones(30) / 30, mode='valid')
        ax.plot(range(29, len(rewards)), ma, color='navy', linewidth=1.5, label='MA 30')
        ax.legend()
    ax.set_title("Reward per Episode")
    ax.set_xlabel("Episodes")
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[0, 1]
    ax.plot(losses, color='tomato', linewidth=1)
    ax.set_title("PPO Loss (per batch)")
    ax.set_xlabel("Batches")
    ax.grid(True, alpha=0.3)

    # Lifespan
    ax = axes[1, 0]
    ax.plot(lifespans, alpha=0.3, color='orange', linewidth=0.5)
    if len(lifespans) >= 30:
        ma2 = np.convolve(lifespans, np.ones(30) / 30, mode='valid')
        ax.plot(range(29, len(lifespans)), ma2, color='darkorange', linewidth=1.5, label='MA 30')
        ax.legend()
    ax.set_title("Avg Lifespan (steps)")
    ax.set_xlabel("Episodes")
    ax.set_ylim(0, 2100)
    ax.grid(True, alpha=0.3)

    # Std (per dimension)
    ax = axes[1, 1]
    if len(logstds) > 0 and hasattr(logstds[0], '__len__'):
        arr = np.array(logstds)
        labels = ['Heading', 'Altitude', 'Water']
        colors = ['blue', 'red', 'orange']
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            ax.plot(arr[:, i], color=col, linewidth=1, label=lbl)
        ax.legend(fontsize=8)
    else:
        ax.plot(logstds, color='purple', linewidth=1)
    ax.set_title("Action Std per Dimension")
    ax.set_xlabel("Batches")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_b{batch_idx:04d}.png"), dpi=100)
    plt.close()


# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FW Survival Training (SimpleFWActor)")
    parser.add_argument("--resume-episodes", type=int, default=0)
    parser.add_argument("--resume-actor", type=str, default="")
    parser.add_argument("--resume-critic", type=str, default="")
    parser.add_argument("--no-critic", action="store_true",
                        help="Train without critic (advantage = raw return)")
    args = parser.parse_args()

    train_fw_survival(
        resume_episodes=args.resume_episodes,
        resume_actor=args.resume_actor,
        resume_critic=args.resume_critic,
        use_critic=not args.no_critic,
    )
