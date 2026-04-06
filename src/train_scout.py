"""
train_scout.py — Standalone scout (quadcopter) training
========================================================

Same architecture as train_multi.py but with N_FIXED=0.
Scout learns to find and hover over fire without a commander.

Usage:
  python train_scout.py
  python train_scout.py --resume path/to/scout.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import time
from concurrent.futures import ProcessPoolExecutor


# =============================================================================
# WORKER FUNCTION (scout only, no commander)
# =============================================================================

def collect_scout_worker(num_eps, scout_w, critic_w, config, batch_start_idx):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import cv2
    import random

    from env_core import DroneFireEnv
    from models import ScoutActor, PrivilegedCritic

    max_steps = config['max_steps']
    hidden_dim = config['hidden_dim']
    N_QUADS = config['N_QUADS']
    scout_msg_dim = config['scout_msg_dim']
    map_size_range = config.get('map_size_range', None)

    local_scout = ScoutActor(
        self_state_dim=config['scout_self_dim'],
        msg_dim=scout_msg_dim,
        hidden_dim=hidden_dim,
    )
    local_scout.load_state_dict(scout_w)
    local_scout.eval()

    scout_priv_dim = config['scout_self_dim'] + 6
    local_critic = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim)
    local_critic.load_state_dict(critic_w)
    local_critic.eval()

    local_env = DroneFireEnv(
        num_quads=N_QUADS, num_fixed=0,
        grid_size_m=config['grid_size_m'], max_steps=max_steps
    )
    local_env.map_size_range = map_size_range

    scout_buf = {k: [] for k in [
        "maps", "self_states", "neighbor_states", "neighbor_masks",
        "actions", "logprobs", "returns", "values",
        "critic_states", "alive"
    ]}

    scout_h0_list = []
    all_rewards = []
    all_lifespans = []
    all_deaths = []

    d_map = torch.zeros(1, 1, 32, 32)
    d_scout_self = torch.zeros(1, config['scout_self_dim'])
    d_neigh_s = torch.zeros(1, max(1, N_QUADS - 1), 3)
    d_neigh_m = torch.ones(1, max(1, N_QUADS - 1), dtype=torch.bool)

    ep_max_steps = config.get('ep_max_steps', max_steps)

    for ep_off in range(num_eps):
        local_env.max_steps = ep_max_steps
        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_off)

        q_agent = local_env.quad_agents[0]
        scout_h = torch.zeros(1, 1, hidden_dim)
        critic_h = torch.zeros(1, 1, hidden_dim)

        scout_h0_list.append(scout_h.clone())

        scout_ep_data = []
        ep_reward = 0.0
        scout_alive = True
        scout_lifespan = ep_max_steps
        death_cause = "survived"

        for step in range(max_steps):
            actions = {}

            if scout_alive and q_agent in local_env.agents:
                with torch.no_grad():
                    l_map = torch.FloatTensor(obs[q_agent]["local_map"]).unsqueeze(0)
                    s_st = torch.FloatTensor(obs[q_agent]["self_state"]).unsqueeze(0)
                    n_s = torch.FloatTensor(obs[q_agent]["neighbor_states"]).unsqueeze(0)
                    n_m = torch.BoolTensor(obs[q_agent]["neighbor_mask"]).unsqueeze(0)

                    dist_s, msg_s, h_out = local_scout(l_map, s_st, n_s, n_m, scout_h)
                    act_s = dist_s.sample()

                    priv_s = torch.FloatTensor(
                        local_env.get_privileged_state(q_agent)).unsqueeze(0)
                    v_s, critic_h = local_critic(priv_s, critic_h)

                scout_ep_data.append({
                    "alive": True,
                    "map": l_map, "self": s_st, "n_s": n_s, "n_m": n_m,
                    "h": scout_h, "act": act_s,
                    "lp": dist_s.log_prob(act_s).sum(1),
                    "reward": 0.0,
                    "value": v_s.item(),
                    "critic_state": priv_s,
                })
                scout_h = h_out
                actions[q_agent] = act_s.squeeze(0).numpy()
            else:
                scout_alive = False
                scout_ep_data.append({
                    "alive": False, "reward": 0.0, "value": 0.0,
                    "critic_state": torch.zeros(1, scout_priv_dim),
                })

            # ENV STEP
            if local_env.agents:
                obs, rewards, terms, truncs, infos = local_env.step(actions)

                r = rewards.get(q_agent, 0.0)
                if scout_alive and q_agent in local_env.agents:
                    scout_ep_data[-1]["reward"] = r
                    ep_reward += r
                if terms.get(q_agent, False) or truncs.get(q_agent, False):
                    scout_alive = False
                    scout_lifespan = step + 1
                    if terms.get(q_agent, False):
                        death_cause = infos.get(q_agent, {}).get("death_cause", "unknown")
                    else:
                        death_cause = "survived"
            else:
                if scout_alive:
                    scout_alive = False
                    scout_lifespan = step + 1
                    death_cause = "env_empty"

            if not scout_alive:
                break

        # GAE
        gamma = config['gamma']
        gae_lam = config['gae_lambda']
        gae = 0.0
        for i in reversed(range(len(scout_ep_data))):
            d = scout_ep_data[i]
            r_t = d["reward"]
            v_t = d["value"]
            v_next = scout_ep_data[i + 1]["value"] if i + 1 < len(scout_ep_data) else 0.0
            delta = r_t + gamma * v_next - v_t
            gae = delta + gamma * gae_lam * gae
            d["ret"] = gae + v_t

        # Pack buffer
        for i in range(max_steps):
            if i < len(scout_ep_data):
                d = scout_ep_data[i]
                if d["alive"]:
                    scout_buf["maps"].append(d["map"])
                    scout_buf["self_states"].append(d["self"])
                    scout_buf["neighbor_states"].append(d["n_s"])
                    scout_buf["neighbor_masks"].append(d["n_m"])
                    scout_buf["actions"].append(d["act"])
                    scout_buf["logprobs"].append(d["lp"])
                    scout_buf["returns"].append(d["ret"])
                    scout_buf["values"].append(d["value"])
                    scout_buf["critic_states"].append(d["critic_state"])
                    scout_buf["alive"].append(1.0)
                    continue
            scout_buf["maps"].append(d_map)
            scout_buf["self_states"].append(d_scout_self)
            scout_buf["neighbor_states"].append(d_neigh_s)
            scout_buf["neighbor_masks"].append(d_neigh_m)
            scout_buf["actions"].append(torch.zeros(1, 4))
            scout_buf["logprobs"].append(torch.tensor([0.0]))
            scout_buf["returns"].append(0.0)
            scout_buf["values"].append(0.0)
            scout_buf["critic_states"].append(torch.zeros(1, scout_priv_dim))
            scout_buf["alive"].append(0.0)

        all_rewards.append(ep_reward)
        all_lifespans.append(scout_lifespan)
        all_deaths.append(death_cause)

    local_env.sim.stop_simulation()

    def cat_buf(buf):
        result = {}
        for k, v in buf.items():
            if len(v) == 0:
                result[k] = torch.tensor([], dtype=torch.float32)
            elif k in ("returns", "alive", "values"):
                result[k] = torch.tensor(v, dtype=torch.float32)
            else:
                result[k] = torch.cat(v)
        return result

    out_scout = cat_buf(scout_buf)
    out_h = torch.cat(scout_h0_list, dim=0) if scout_h0_list else None
    return out_scout, out_h, all_rewards, all_lifespans, all_deaths


# =============================================================================
# TRAINING
# =============================================================================

def train_scout(resume="", log_episodes=False, log_dir="/tmp/ep_logs"):
    print("=" * 70)
    print("  Standalone Scout Training (no commander)")
    print("=" * 70)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from env_core import DroneFireEnv
    from models import ScoutActor, PrivilegedCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}\n")

    # ── Hyperparameters ──────────────────────────────────────────────────
    N_QUADS = 1
    grid_size_m = 2000.0
    map_size_range = (1000, 2000)
    num_episodes = 30_000
    max_steps = 4000
    steps_range = (1500, 4000)

    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    update_epochs = 4
    num_workers = 15
    eps_per_worker = 2
    episodes_per_batch = num_workers * eps_per_worker

    lr_scout = 1e-4
    lr_critic = 3e-4
    hidden_dim = 128
    scout_msg_dim = 5
    entropy_coef = 0.003

    # ── Dims ─────────────────────────────────────────────────────────────
    temp_env = DroneFireEnv(num_quads=N_QUADS, num_fixed=0,
                            grid_size_m=grid_size_m, max_steps=max_steps)
    scout_self_dim = temp_env.observation_space(
        temp_env.quad_agents[0])["self_state"].shape[0]
    if hasattr(temp_env, 'sim') and temp_env.sim is not None:
        temp_env.sim.stop_simulation()

    print(f"scout_self_dim = {scout_self_dim}")
    print(f"max_steps      = {max_steps}")
    print(f"steps_range    = {steps_range}")
    print(f"map_size_range = {map_size_range}")

    worker_config = {
        'N_QUADS': N_QUADS,
        'grid_size_m': grid_size_m,
        'map_size_range': map_size_range,
        'max_steps': max_steps,
        'scout_self_dim': scout_self_dim,
        'scout_msg_dim': scout_msg_dim,
        'hidden_dim': hidden_dim,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'log_episodes': log_episodes,
        'log_dir': log_dir,
    }

    # ── Networks ─────────────────────────────────────────────────────────
    scout_actor = ScoutActor(
        self_state_dim=scout_self_dim,
        msg_dim=scout_msg_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    print(f"ScoutActor params: {sum(p.numel() for p in scout_actor.parameters()):,}")

    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model_shapes = {k: v.shape for k, v in scout_actor.state_dict().items()}
        filtered = {k: v for k, v in ckpt.items()
                    if k in model_shapes and v.shape == model_shapes[k]}
        skipped = [k for k in ckpt if k not in filtered]
        scout_actor.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")
        print(f"  Loaded scout from {resume}")

    optimizer_scout = optim.Adam(scout_actor.parameters(), lr=lr_scout)

    scout_priv_dim = scout_self_dim + 6
    critic = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim).to(device)
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    # ── Tracking ─────────────────────────────────────────────────────────
    reward_history = []
    reward_per_batch = []
    loss_history = []
    critic_loss_history = []
    life_pct_history = []
    death_history = []

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_models", "scout_solo")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints → {save_dir}\n")

    best_avg = -1e9
    episodes_played = 0
    num_batches = num_episodes // episodes_per_batch

    for batch_idx in range(1, num_batches + 1):
        scout_w = {k: v.cpu() for k, v in scout_actor.state_dict().items()}
        critic_w = {k: v.cpu() for k, v in critic.state_dict().items()}
        t0 = time.time()

        agg = {k: [] for k in [
            "maps", "self_states", "neighbor_states", "neighbor_masks",
            "actions", "logprobs", "returns", "values", "critic_states", "alive"
        ]}
        agg_h = []
        batch_rewards = []
        batch_lifespans = []
        batch_deaths = []

        # Pick episode length
        if steps_range is not None:
            batch_ep_max = random.randint(steps_range[0] // 100, steps_range[1] // 100) * 100
        else:
            batch_ep_max = max_steps
        worker_config['ep_max_steps'] = batch_ep_max

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    collect_scout_worker,
                    eps_per_worker, scout_w, critic_w,
                    worker_config,
                    episodes_played + i * eps_per_worker
                )
                for i in range(num_workers)
            ]
            failed = 0
            for fut in futures:
                try:
                    w_scout, w_h, w_rew, w_life, w_deaths = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"   ⚠ Worker failed: {e}")
                    continue
                batch_rewards.extend(w_rew)
                batch_lifespans.extend(w_life)
                batch_deaths.extend(w_deaths)
                reward_history.extend(w_rew)
                episodes_played += len(w_rew)

                for k in agg:
                    agg[k].append(w_scout[k])
                if w_h is not None:
                    agg_h.append(w_h)

        rollout_time = time.time() - t0

        if failed > 0:
            print(f"   ⚠ {failed}/{num_workers} workers failed this batch")
        if not batch_rewards:
            print(f"   ⚠ All workers failed, skipping batch {batch_idx}")
            continue

        avg_batch = float(np.mean(batch_rewards))
        reward_per_batch.append(avg_batch)
        win = min(60, len(reward_history))
        avg_roll = float(np.mean(reward_history[-win:]))

        life_pct = float(np.mean(batch_lifespans)) / batch_ep_max * 100
        life_pct_history.append(life_pct)

        from collections import Counter
        deaths_c = Counter(batch_deaths)
        d_str = " ".join(f"{k}={v}" for k, v in deaths_c.most_common())
        death_history.append(dict(deaths_c))

        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"Batch {batch_idx:04d} (Ep {episodes_played:05d}) | "
              f"R: {avg_batch:+8.1f} ({avg_roll:+8.1f})  "
              f"Life:{life_pct:.0f}%  "
              f"[{d_str}]  "
              f"{rollout_time:.1f}s")

        # Save best
        if episodes_played >= 60 and avg_roll > best_avg:
            best_avg = avg_roll
            torch.save(scout_actor.state_dict(),
                       os.path.join(save_dir, "scout_best.pt"))
            print(f"   ⭐ New best! rolling avg = {best_avg:.1f}")

        if batch_idx % 10 == 0:
            torch.save(scout_actor.state_dict(),
                       os.path.join(save_dir, f"scout_b{batch_idx:04d}.pt"))

        # ==============================================================
        # PPO UPDATE
        # ==============================================================
        s_maps = torch.cat(agg["maps"]).to(device)
        s_self = torch.cat(agg["self_states"]).to(device)
        s_neigh_s = torch.cat(agg["neighbor_states"]).to(device)
        s_neigh_m = torch.cat(agg["neighbor_masks"]).to(device)
        s_actions = torch.cat(agg["actions"]).to(device)
        s_logprobs = torch.cat(agg["logprobs"]).to(device)
        s_returns = torch.cat(agg["returns"]).to(device)
        s_values = torch.cat(agg["values"]).to(device)
        s_cstates = torch.cat(agg["critic_states"]).to(device)
        s_alive = torch.cat(agg["alive"]).to(device)

        h_scout = (torch.cat(agg_h, dim=0)
                   .squeeze(1).unsqueeze(0).to(device))

        s_adv = s_returns - s_values
        alive_bool = s_alive > 0.5
        if alive_bool.sum() > 1:
            alive_adv = s_adv[alive_bool]
            s_adv = (s_adv - alive_adv.mean()) / (alive_adv.std() + 1e-8)

        alive_rets = s_returns[alive_bool] if alive_bool.sum() > 1 else s_returns
        s_ret_mean = alive_rets.mean()
        s_ret_std = alive_rets.std() + 1e-8
        s_returns_norm = (s_returns - s_ret_mean) / s_ret_std

        eps = s_returns.numel() // max_steps
        s_maps_seq = s_maps.view(eps, max_steps, 1, 32, 32)
        s_self_seq = s_self.view(eps, max_steps, -1)
        s_neigh_s_seq = s_neigh_s.view(eps, max_steps, s_neigh_s.size(-2), 3)
        s_neigh_m_seq = s_neigh_m.view(eps, max_steps, -1)
        s_actions_seq = s_actions.view(eps, max_steps, -1)
        s_logprobs_seq = s_logprobs.view(eps, max_steps)
        s_adv_seq = s_adv.view(eps, max_steps)
        s_returns_norm_seq = s_returns_norm.view(eps, max_steps)
        s_alive_seq = s_alive.view(eps, max_steps)
        s_cstates_seq = s_cstates.view(eps, max_steps, -1)
        h_scout_seq = h_scout.transpose(0, 1)

        num_minibatches = 4
        mb_size = max(1, eps // num_minibatches)
        total_loss = 0.0
        total_critic_loss = 0.0

        for epoch in range(update_epochs):
            b_inds = np.random.permutation(eps)
            for start in range(0, eps, mb_size):
                mb = b_inds[start:start + mb_size]

                mb_maps = s_maps_seq[mb]
                mb_self = s_self_seq[mb]
                mb_ns = s_neigh_s_seq[mb]
                mb_nm = s_neigh_m_seq[mb]
                mb_acts = s_actions_seq[mb]
                mb_old_lp = s_logprobs_seq[mb].view(-1)
                mb_adv = s_adv_seq[mb].reshape(-1)
                mb_alive = s_alive_seq[mb].reshape(-1)
                mb_rets = s_returns_norm_seq[mb].reshape(-1)
                mb_cs = s_cstates_seq[mb]
                mb_h = h_scout_seq[mb].transpose(0, 1)

                dist, _, _ = scout_actor(mb_maps, mb_self, mb_ns, mb_nm, mb_h)
                flat_acts = mb_acts.view(-1, 4)
                new_lp = dist.log_prob(flat_acts).sum(1)
                entropy = dist.entropy().sum(1)

                log_ratio = (new_lp - mb_old_lp).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                surr = torch.max(pg1, pg2)

                n_alive = mb_alive.sum().clamp(min=1.0)
                loss = ((surr * mb_alive).sum() / n_alive
                        - entropy_coef * (entropy * mb_alive).sum() / n_alive)

                if torch.isfinite(loss):
                    optimizer_scout.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(scout_actor.parameters(), max_norm=0.5)
                    optimizer_scout.step()
                    total_loss += loss.item()

                v_pred, _ = critic(mb_cs, None)
                v_err = (v_pred - mb_rets) ** 2
                v_loss = (v_err * mb_alive).sum() / n_alive
                if torch.isfinite(v_loss):
                    optimizer_critic.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                    optimizer_critic.step()
                    total_critic_loss += v_loss.item()

        loss_history.append(total_loss / max(1, update_epochs * num_minibatches))
        critic_loss_history.append(total_critic_loss / max(1, update_epochs * num_minibatches))

        # ==============================================================
        # PERIODIC PLOT
        # ==============================================================
        if batch_idx % 10 == 0:
            _save_plot_scout(reward_per_batch, loss_history, critic_loss_history,
                             life_pct_history, death_history,
                             save_dir, batch_idx)

    _save_plot_scout(reward_per_batch, loss_history, critic_loss_history,
                     life_pct_history, death_history,
                     save_dir, batch_idx)

    print(f"\n✅ Training complete!")
    print(f"   Best: {save_dir}/scout_best.pt")


# =============================================================================
# PLOT
# =============================================================================

def _save_plot_scout(reward_batches, loss, critic_loss,
                     life_pct, deaths, save_dir, batch_idx):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Scout Solo — Batch {batch_idx}", fontsize=13)
    batches = np.arange(1, len(reward_batches) + 1)

    def _ma(data, w=10):
        if len(data) < w:
            return None, None
        ma = np.convolve(data, np.ones(w) / w, mode='valid')
        return np.arange(w, len(data) + 1), ma

    # (0,0) Reward
    ax = axes[0, 0]
    ax.bar(batches, reward_batches, color='steelblue', alpha=0.4, width=1.0)
    mx, ma = _ma(reward_batches)
    if ma is not None:
        ax.plot(mx, ma, color='navy', linewidth=2, label='MA 10')
        ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_title("Avg Reward per Batch")
    ax.set_xlabel("Batch")
    ax.grid(True, alpha=0.3)

    # (0,1) PPO Loss
    ax = axes[0, 1]
    if loss:
        ax.plot(range(1, len(loss) + 1), loss, color='green', linewidth=1)
    ax.set_title("PPO Loss")
    ax.set_xlabel("Batch")
    ax.grid(True, alpha=0.3)

    # (0,2) Critic Loss
    ax = axes[0, 2]
    if critic_loss:
        ax.plot(range(1, len(critic_loss) + 1), critic_loss, color='tomato', linewidth=1)
    ax.set_title("Critic Value Loss (MSE)")
    ax.set_xlabel("Batch")
    ax.grid(True, alpha=0.3)

    # (1,0) Lifespan %
    ax = axes[1, 0]
    ax.plot(batches, life_pct, color='green', linewidth=1, alpha=0.5)
    mx, ma = _ma(life_pct)
    if ma is not None:
        ax.plot(mx, ma, color='green', linewidth=2, label='MA 10')
    ax.axhline(100, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='100%')
    ax.set_ylim(0, 110)
    ax.set_title("Lifespan (% of max_steps)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("%")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # (1,1) Death causes
    ax = axes[1, 1]
    if deaths:
        all_causes = set()
        for d in deaths:
            all_causes.update(d.keys())
        all_causes = sorted(all_causes)
        cause_colors = {
            'boundary': '#e74c3c', 'ceiling': '#e67e22', 'ground_crash': '#8e44ad',
            'survived': '#2ecc71', 'unknown': '#95a5a6', 'env_empty': '#7f8c8d'
        }
        stacks = {c: [] for c in all_causes}
        for d in deaths:
            total = max(1, sum(d.values()))
            for c in all_causes:
                stacks[c].append(d.get(c, 0) / total * 100)
        bottoms = np.zeros(len(deaths))
        for c in all_causes:
            vals = np.array(stacks[c])
            color = cause_colors.get(c, '#bdc3c7')
            ax.bar(batches[:len(vals)], vals, bottom=bottoms[:len(vals)],
                   color=color, alpha=0.8, width=1.0, label=c)
            bottoms[:len(vals)] += vals
        ax.set_ylim(0, 105)
        ax.set_title("Death Causes (%)")
        ax.set_xlabel("Batch")
        ax.set_ylabel("%")
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (1,2) empty — reserved
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_b{batch_idx:04d}.png"), dpi=100)
    plt.close()


# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone scout training")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to scout checkpoint to resume from")
    args = parser.parse_args()
    train_scout(resume=args.resume)
