"""
train_multi.py — Multi-agent hybrid training: Scout (frame-by-frame) + Commander (waypoint)
============================================================================================

Architecture:
  Scout (ScoutActor):
    - NN called EVERY physics step → 4D action + 5D message
    - Input: local_map [1,32,32] + self_state [15] + neighbor_states [N,3]

  Commander (CommanderActor):
    - NN called every `waypoint_steps` (50) physics steps → 4D waypoint
    - Input: self_state [17] + scout messages [N_msg_slots, 5]
    - Between calls: heading-hold controller flies toward waypoint

  Separate PPO optimizers, no shared critic.
  Scout buffer: [eps × max_steps]
  Commander buffer: [eps × num_decisions_cmdr], padded with alive mask

Usage:
  python train_multi.py
  python train_multi.py --resume-scout path/to/scout.pt --resume-cmdr path/to/actor.pt
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
# HELPERS
# =============================================================================

def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# WORKER FUNCTION  (runs on CPU — hybrid scout + commander)
# =============================================================================

def collect_multi_worker(num_eps, scout_w, cmdr_w, critic_scout_w, critic_cmdr_w,
                         config, batch_start_idx):
    """
    Hybrid rollout: scout runs every step, commander runs every waypoint_steps.
    Scout messages are buffered and the best-fire + latest are sent to commander.
    Critics run in parallel to estimate V(s) for GAE.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import cv2  # pre-import to avoid repeated lazy-loading in env
    import random

    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, PrivilegedCritic

    max_steps = config['max_steps']
    waypoint_steps = config['waypoint_steps']
    waypoint_range = config['waypoint_range']
    num_decisions_cmdr = config['num_decisions_cmdr']
    hidden_dim_cmdr = config['hidden_dim_cmdr']
    hidden_dim_scout = config['hidden_dim_scout']
    N_QUADS = config['N_QUADS']
    scout_msg_dim = config['scout_msg_dim']
    map_size_range = config.get('map_size_range', None)

    map_half = config['grid_size_m'] / 2.0
    safe_limit = map_half - 250.0
    boundary_emergency = map_half - 100.0   # hard override: steer toward center
    wp_reached_dist = 30.0
    wp_timeout_penalty = -1.0

    # ── Rebuild networks on CPU ──────────────────────────────────────────
    local_scout = ScoutActor(
        self_state_dim=config['scout_self_dim'],
        msg_dim=scout_msg_dim,
        hidden_dim=hidden_dim_scout,
    )
    local_scout.load_state_dict(scout_w)
    local_scout.eval()

    local_cmdr = CommanderActor(
        self_state_dim=config['fixed_self_dim'],
        msg_input_dim=scout_msg_dim,
        action_dim=4,
        hidden_dim=hidden_dim_cmdr,
    )
    local_cmdr.load_state_dict(cmdr_w)
    local_cmdr.eval()

    # Critics (privileged — see global state)
    scout_priv_dim = config['scout_self_dim'] + 6   # 15 + 6 = 21
    cmdr_priv_dim = config['fixed_self_dim'] + 6    # 17 + 6 = 23
    local_critic_scout = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim_scout)
    local_critic_scout.load_state_dict(critic_scout_w)
    local_critic_scout.eval()
    local_critic_cmdr = PrivilegedCritic(cmdr_priv_dim, hidden_dim=hidden_dim_cmdr)
    local_critic_cmdr.load_state_dict(critic_cmdr_w)
    local_critic_cmdr.eval()

    local_env = DroneFireEnv(
        num_quads=N_QUADS, num_fixed=1,
        grid_size_m=config['grid_size_m'], max_steps=max_steps
    )
    local_env.map_size_range = map_size_range

    # ── Buffers ──────────────────────────────────────────────────────────
    scout_buf = {k: [] for k in [
        "maps", "self_states", "neighbor_states", "neighbor_masks",
        "actions", "logprobs", "returns", "values",
        "critic_states"
    ]}
    cmdr_buf = {k: [] for k in [
        "states", "messages", "msg_masks",
        "actions", "logprobs", "returns", "alive", "values",
        "critic_states"
    ]}

    scout_h0_list = []
    cmdr_h0_list = []
    all_rewards = []
    all_lifespans = []
    all_deaths = []

    # Dummy tensors for dead scout steps
    d_map = torch.zeros(1, 1, 32, 32)
    d_scout_self = torch.zeros(1, config['scout_self_dim'])
    d_neigh_s = torch.zeros(1, max(1, N_QUADS - 1), 3)
    d_neigh_m = torch.ones(1, max(1, N_QUADS - 1), dtype=torch.bool)

    ep_max_steps = config.get('ep_max_steps', max_steps)

    for ep_off in range(num_eps):
        local_env.max_steps = ep_max_steps

        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_off)

        # Recalculate safe_limit after reset (map size may have changed)
        map_half = local_env.map_bounds
        safe_limit = map_half - 250.0
        boundary_emergency = map_half - 100.0

        q_agent = local_env.quad_agents[0] if N_QUADS > 0 else None
        f_agent = local_env.fixed_agents[0]

        # Hidden states
        scout_h = torch.zeros(1, 1, hidden_dim_scout)
        cmdr_h = torch.zeros(1, 1, hidden_dim_cmdr)
        critic_scout_h = torch.zeros(1, 1, hidden_dim_scout)
        critic_cmdr_h = torch.zeros(1, 1, hidden_dim_cmdr)

        scout_h0_list.append(scout_h.clone())
        cmdr_h0_list.append(cmdr_h.clone())

        # Per-episode tracking
        scout_ep_data = []  # list of per-step dicts
        cmdr_ep_data = []   # list of per-decision dicts
        ep_reward_scout = 0.0
        ep_reward_cmdr = 0.0
        scout_alive = True
        cmdr_alive = True
        scout_death_cause = "survived"
        cmdr_death_cause = "survived"
        total_cmdr_steps = 0
        scout_lifespan = ep_max_steps
        cmdr_lifespan = ep_max_steps

        # Commander waypoint state
        target_x, target_y = 0.0, 0.0
        target_alt_raw = 0.0
        water_raw = -1.0
        steps_in_segment = 0
        wp_reached = False
        segment_reward = 0.0
        need_new_waypoint = True

        # Scout message buffer (for commander)
        msg_buffer = []  # list of 5D tensors from this segment

        # Trajectory logging (only for first episode per worker)
        log_this_ep = config.get('log_episodes', False) and ep_off == 0
        traj_scout_pos = [] if log_this_ep else None
        traj_cmdr_pos = [] if log_this_ep else None
        traj_cmdr_waypoints = [] if log_this_ep else None
        traj_rewards_scout = [] if log_this_ep else None
        traj_rewards_cmdr = [] if log_this_ep else None

        for step in range(max_steps):
            actions = {}

            # ==============================================================
            # SCOUT FORWARD PASS (every step)
            # ==============================================================
            scout_msg_tensor = torch.zeros(1, scout_msg_dim)
            scout_msg_valid = False

            if scout_alive and q_agent in local_env.agents:
                with torch.no_grad():
                    l_map = torch.FloatTensor(obs[q_agent]["local_map"]).unsqueeze(0)
                    s_st = torch.FloatTensor(obs[q_agent]["self_state"]).unsqueeze(0)
                    n_s = torch.FloatTensor(obs[q_agent]["neighbor_states"]).unsqueeze(0)
                    n_m = torch.BoolTensor(obs[q_agent]["neighbor_mask"]).unsqueeze(0)

                    dist_s, msg_s, h_out_s = local_scout(l_map, s_st, n_s, n_m, scout_h)
                    act_s = dist_s.sample()

                    # Critic: privileged state
                    priv_s = torch.FloatTensor(
                        local_env.get_privileged_state(q_agent)).unsqueeze(0)
                    v_s, critic_scout_h = local_critic_scout(priv_s, critic_scout_h)

                scout_ep_data.append({
                    "alive": True,
                    "map": l_map, "self": s_st, "n_s": n_s, "n_m": n_m,
                    "h": scout_h, "act": act_s,
                    "lp": dist_s.log_prob(act_s).sum(1),
                    "reward": 0.0,  # filled after env.step
                    "value": v_s.item(),
                    "critic_state": priv_s,
                })
                scout_h = h_out_s
                actions[q_agent] = act_s.squeeze(0).numpy()
                scout_msg_tensor = msg_s.detach()
                scout_msg_valid = True
            else:
                scout_alive = False
                scout_ep_data.append({
                    "alive": False, "reward": 0.0, "value": 0.0,
                    "critic_state": torch.zeros(1, scout_priv_dim),
                })

            # Buffer scout message for commander
            msg_buffer.append((scout_msg_tensor, scout_msg_valid))

            # ==============================================================
            # COMMANDER WAYPOINT DECISION (every waypoint_steps or on reach)
            # ==============================================================
            if cmdr_alive and f_agent in local_env.agents:
                if need_new_waypoint:
                    # Build message input for commander: [latest, best_fire] per scout
                    # From msg_buffer (messages collected since last decision)
                    msgs_for_cmdr = []
                    masks_for_cmdr = []

                    if len(msg_buffer) > 0:
                        # Latest message
                        latest_msg, latest_valid = msg_buffer[-1]
                        msgs_for_cmdr.append(latest_msg)
                        masks_for_cmdr.append(not latest_valid)

                        # Best fire message (highest intensity at index 2)
                        best_msg = latest_msg
                        best_intensity = -1.0
                        for m_tensor, m_valid in msg_buffer:
                            if m_valid and m_tensor[0, 2].item() > best_intensity:
                                best_intensity = m_tensor[0, 2].item()
                                best_msg = m_tensor
                        msgs_for_cmdr.append(best_msg)
                        masks_for_cmdr.append(not latest_valid)
                    else:
                        msgs_for_cmdr.append(torch.zeros(1, scout_msg_dim))
                        masks_for_cmdr.append(True)
                        msgs_for_cmdr.append(torch.zeros(1, scout_msg_dim))
                        masks_for_cmdr.append(True)

                    # Stack: [1, 2*N_quads, msg_dim]
                    msgs_t = torch.stack(msgs_for_cmdr, dim=1)  # [1, 2, 5]
                    msgs_m = torch.tensor([masks_for_cmdr])      # [1, 2]

                    # Forward pass
                    with torch.no_grad():
                        s_st_f = torch.FloatTensor(
                            obs[f_agent]["self_state"]).unsqueeze(0)

                        dist_c, _, h_out_c = local_cmdr(
                            s_st_f, msgs_t, msgs_m, cmdr_h)

                        act_c = dist_c.sample()

                        # Critic: privileged state
                        priv_c = torch.FloatTensor(
                            local_env.get_privileged_state(f_agent)).unsqueeze(0)
                        v_c, critic_cmdr_h = local_critic_cmdr(priv_c, critic_cmdr_h)

                    cmdr_ep_data.append({
                        "alive": True,
                        "state": s_st_f,
                        "msgs": msgs_t,
                        "msg_mask": msgs_m,
                        "h": cmdr_h,
                        "act": act_c,
                        "lp": dist_c.log_prob(act_c).sum(1),
                        "reward": 0.0,
                        "value": v_c.item(),
                        "critic_state": priv_c,
                    })
                    cmdr_h = h_out_c

                    # Parse waypoint
                    act_np = act_c.squeeze(0).numpy()
                    dx_raw = float(act_np[0])
                    dy_raw = float(act_np[1])
                    target_alt_raw = float(act_np[2])
                    water_raw = float(act_np[3])

                    drone = local_env.sim.drones.get(f_agent)
                    cur_pos = drone.get_position() if drone else np.zeros(3)
                    target_x = cur_pos[0] + dx_raw * waypoint_range
                    target_y = cur_pos[1] + dy_raw * waypoint_range
                    target_x = np.clip(target_x, -safe_limit, safe_limit)
                    target_y = np.clip(target_y, -safe_limit, safe_limit)

                    steps_in_segment = 0
                    wp_reached = False
                    segment_reward = 0.0
                    need_new_waypoint = False
                    msg_buffer = []  # reset for next segment

                    if log_this_ep:
                        traj_cmdr_waypoints.append(
                            [step, target_x, target_y, target_alt_raw, water_raw])

                # Heading controller
                drone = local_env.sim.drones.get(f_agent)
                if drone is not None:
                    pos = drone.get_position()

                    # Emergency boundary override: if near kill zone, steer to center
                    if (abs(pos[0]) > boundary_emergency or
                            abs(pos[1]) > boundary_emergency):
                        target_x = 0.0
                        target_y = 0.0
                        need_new_waypoint = True  # ask NN for new wp next step

                    dx_to = target_x - pos[0]
                    dy_to = target_y - pos[1]
                    dist_to = np.sqrt(dx_to**2 + dy_to**2)

                    if dist_to < wp_reached_dist:
                        wp_reached = True

                    if dist_to > 1.0:
                        desired_heading = np.arctan2(dy_to, dx_to)
                        cur_yaw = drone.get_orientation_rpy()[2]
                        heading_error = _wrap_angle(desired_heading - cur_yaw)
                        heading_cmd = np.clip(heading_error / np.pi, -1.0, 1.0)
                    else:
                        heading_cmd = 0.0

                    inner_action = np.array(
                        [heading_cmd, target_alt_raw, water_raw],
                        dtype=np.float32)
                    actions[f_agent] = inner_action

                steps_in_segment += 1
                total_cmdr_steps += 1
            else:
                cmdr_alive = False

            # ==============================================================
            # ENV STEP
            # ==============================================================
            r_scout = 0.0
            r_cmdr = 0.0
            if local_env.agents:
                obs, rewards, terms, _, infos = local_env.step(actions)

                # Scout reward
                r_scout = rewards.get(q_agent, 0.0) if q_agent else 0.0
                if scout_alive and q_agent in local_env.agents:
                    scout_ep_data[-1]["reward"] = r_scout
                    ep_reward_scout += r_scout
                if q_agent and terms.get(q_agent, False):
                    scout_alive = False
                    scout_lifespan = step + 1
                    scout_death_cause = infos.get(
                        q_agent, {}).get("death_cause", "unknown")

                # Commander segment reward
                r_cmdr = rewards.get(f_agent, 0.0)
                segment_reward += r_cmdr

                if terms.get(f_agent, False):
                    cmdr_alive = False
                    cmdr_lifespan = step + 1
                    cmdr_death_cause = infos.get(
                        f_agent, {}).get("death_cause", "unknown")
                    # Store segment reward for current decision
                    if cmdr_ep_data:
                        cmdr_ep_data[-1]["reward"] = segment_reward
                    ep_reward_cmdr += segment_reward
            else:
                # All dead
                if scout_alive:
                    scout_alive = False
                    scout_lifespan = step + 1
                    scout_death_cause = "env_empty"
                if cmdr_alive:
                    cmdr_alive = False
                    cmdr_lifespan = step + 1
                    cmdr_death_cause = "env_empty"

            # Early exit: no point stepping physics when both agents are dead
            if not scout_alive and not cmdr_alive:
                break

            # --- Trajectory recording ---
            if log_this_ep:
                s_drone = local_env.sim.drones.get(q_agent) if q_agent else None
                c_drone = local_env.sim.drones.get(f_agent)
                traj_scout_pos.append(
                    s_drone.get_position().copy() if s_drone else np.full(3, np.nan))
                traj_cmdr_pos.append(
                    c_drone.get_position().copy() if c_drone else np.full(3, np.nan))
                traj_rewards_scout.append(r_scout if scout_alive else 0.0)
                traj_rewards_cmdr.append(r_cmdr if cmdr_alive else 0.0)

            # ==============================================================
            # CHECK WAYPOINT SEGMENT END
            # ==============================================================
            if cmdr_alive and f_agent in local_env.agents:
                segment_done = (
                    wp_reached or
                    steps_in_segment >= waypoint_steps or
                    step == max_steps - 1
                )
                if segment_done:
                    # Timeout penalty
                    if not wp_reached and steps_in_segment >= waypoint_steps:
                        segment_reward += wp_timeout_penalty

                    if cmdr_ep_data:
                        cmdr_ep_data[-1]["reward"] = segment_reward
                    ep_reward_cmdr += segment_reward - r_cmdr  # avoid double-count of last r_cmdr
                    # Actually, let me fix this: segment_reward already includes r_cmdr from earlier
                    # We assigned segment_reward to cmdr_ep_data[-1]["reward"] correctly
                    # ep_reward_cmdr should just sum all segment rewards
                    need_new_waypoint = True

        # ==================================================================
        # Fix ep_reward_cmdr: sum from cmdr_ep_data
        # ==================================================================
        ep_reward_cmdr = sum(d["reward"] for d in cmdr_ep_data)

        # ==================================================================
        # SCOUT GAE (per step, over max_steps)
        # ==================================================================
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

        # ==================================================================
        # COMMANDER GAE (per decision, over cmdr_ep_data)
        # ==================================================================
        gamma_cmdr = config.get('gamma_cmdr', gamma)
        gae = 0.0
        n_actual_cmdr = len(cmdr_ep_data)
        for i in reversed(range(n_actual_cmdr)):
            d = cmdr_ep_data[i]
            r_t = d["reward"]
            v_t = d["value"]
            v_next = cmdr_ep_data[i + 1]["value"] if i + 1 < n_actual_cmdr else 0.0
            delta = r_t + gamma_cmdr * v_next - v_t
            gae = delta + gamma_cmdr * gae_lam * gae
            d["ret"] = gae + v_t

        # ==================================================================
        # PACK SCOUT BUFFER (max_steps entries, padded for dead steps)
        # ==================================================================
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
                    continue
            # Dead or out of range — pad
            scout_buf["maps"].append(d_map)
            scout_buf["self_states"].append(d_scout_self)
            scout_buf["neighbor_states"].append(d_neigh_s)
            scout_buf["neighbor_masks"].append(d_neigh_m)
            scout_buf["actions"].append(torch.zeros(1, 4))
            scout_buf["logprobs"].append(torch.tensor([0.0]))
            scout_buf["returns"].append(0.0)
            scout_buf["values"].append(0.0)
            scout_buf["critic_states"].append(torch.zeros(1, scout_priv_dim))

        # ==================================================================
        # PACK COMMANDER BUFFER (num_decisions_cmdr entries, padded)
        # ==================================================================
        n_msg_slots = 2 * N_QUADS  # latest + best_fire per scout
        for i in range(num_decisions_cmdr):
            if i < n_actual_cmdr:
                d = cmdr_ep_data[i]
                cmdr_buf["states"].append(d["state"])
                cmdr_buf["messages"].append(d["msgs"])
                cmdr_buf["msg_masks"].append(d["msg_mask"])
                cmdr_buf["actions"].append(d["act"])
                cmdr_buf["logprobs"].append(d["lp"])
                cmdr_buf["returns"].append(d["ret"])
                cmdr_buf["alive"].append(1.0)
                cmdr_buf["values"].append(d["value"])
                cmdr_buf["critic_states"].append(d["critic_state"])
            else:
                cmdr_buf["states"].append(
                    torch.zeros(1, config['fixed_self_dim']))
                cmdr_buf["messages"].append(
                    torch.zeros(1, n_msg_slots, scout_msg_dim))
                cmdr_buf["msg_masks"].append(
                    torch.ones(1, n_msg_slots, dtype=torch.bool))
                cmdr_buf["actions"].append(torch.zeros(1, 4))
                cmdr_buf["logprobs"].append(torch.tensor([0.0]))
                cmdr_buf["returns"].append(0.0)
                cmdr_buf["alive"].append(0.0)
                cmdr_buf["values"].append(0.0)
                cmdr_buf["critic_states"].append(
                    torch.zeros(1, cmdr_priv_dim))

        ep_reward_total = ep_reward_scout + ep_reward_cmdr
        all_rewards.append(ep_reward_total)
        avg_life = (scout_lifespan + cmdr_lifespan) / 2.0
        all_lifespans.append(avg_life)
        all_deaths.append(f"s:{scout_death_cause},c:{cmdr_death_cause}")

        # --- Episode log (for replay/analysis) ---
        if log_this_ep:
            log_dir = config.get('log_dir', '/tmp/ep_logs')
            os.makedirs(log_dir, exist_ok=True)
            ep_id = batch_start_idx + ep_off
            np.savez_compressed(
                os.path.join(log_dir, f'ep_{ep_id:06d}.npz'),
                ep_id=np.array(ep_id),
                scout_reward=np.array(ep_reward_scout),
                cmdr_reward=np.array(ep_reward_cmdr),
                scout_lifespan=np.array(scout_lifespan),
                cmdr_lifespan=np.array(cmdr_lifespan),
                scout_death=np.array(scout_death_cause),
                cmdr_death=np.array(cmdr_death_cause),
                n_cmdr_decisions=np.array(n_actual_cmdr),
                scout_pos=np.array(traj_scout_pos),
                cmdr_pos=np.array(traj_cmdr_pos),
                cmdr_waypoints=np.array(traj_cmdr_waypoints) if traj_cmdr_waypoints else np.zeros((0, 5)),
                rewards_scout=np.array(traj_rewards_scout),
                rewards_cmdr=np.array(traj_rewards_cmdr),
                map_bounds=np.array(local_env.grid_size_m / 2.0),
            )

    # ── Cleanup & return ─────────────────────────────────────────────────
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
    out_cmdr = cat_buf(cmdr_buf)

    out_init_h = {
        "scout": torch.cat(scout_h0_list, dim=0) if scout_h0_list else None,
        "cmdr": torch.cat(cmdr_h0_list, dim=0) if cmdr_h0_list else None,
    }

    return out_scout, out_cmdr, out_init_h, all_rewards, all_lifespans, all_deaths


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_multi(resume_scout="", resume_cmdr="",
                log_episodes=False, log_dir="/tmp/ep_logs"):
    print("=" * 70)
    print("  Multi-Agent Training: Scout (frame-by-frame) + Commander (waypoint)")
    print("  Commander: CommanderActor (with scout messages)")
    print("=" * 70)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, PrivilegedCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}\n")

    # ── Hyperparameters ──────────────────────────────────────────────────
    N_QUADS = 1
    N_FIXED = 1
    grid_size_m = 1000.0
    map_size_range = (600, 2000)   # random map size per episode [m]
    num_episodes = 30_000
    max_steps = 4000              # buffer size (upper bound)
    steps_range = (1500, 4000)     # actual ep length randomized per episode
    waypoint_steps = 50
    waypoint_range = 100.0
    num_decisions_cmdr = max_steps // waypoint_steps  # 80

    gamma = 0.99
    gamma_cmdr = 0.95          # shorter horizon for 40 decisions
    gae_lambda = 0.95
    clip_coef = 0.2
    vf_coef = 0.5              # value loss weight
    update_epochs = 4
    num_workers = 15
    eps_per_worker = 2
    episodes_per_batch = num_workers * eps_per_worker

    lr_scout = 1e-4
    lr_cmdr = 3e-4
    lr_critic = 3e-4
    hidden_dim_scout = 128
    hidden_dim_cmdr = 64
    scout_msg_dim = 5

    entropy_scout = 0.003
    entropy_cmdr = 0.01        # prevent premature std collapse

    # ── Dims ─────────────────────────────────────────────────────────────
    temp_env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                            grid_size_m=grid_size_m, max_steps=max_steps)
    scout_self_dim = temp_env.observation_space(
        temp_env.quad_agents[0])["self_state"].shape[0]
    fixed_self_dim = temp_env.observation_space(
        temp_env.fixed_agents[0])["self_state"].shape[0]
    if hasattr(temp_env, 'sim') and temp_env.sim is not None:
        temp_env.sim.stop_simulation()

    print(f"scout_self_dim   = {scout_self_dim}")
    print(f"fixed_self_dim   = {fixed_self_dim}")
    print(f"max_steps        = {max_steps}")
    print(f"waypoint_steps   = {waypoint_steps}")
    print(f"waypoint_range   = {waypoint_range}m")
    print(f"num_decisions    = {num_decisions_cmdr} (commander)")

    worker_config = {
        'N_QUADS': N_QUADS, 'N_FIXED': N_FIXED,
        'grid_size_m': grid_size_m,
        'map_size_range': map_size_range,
        'max_steps': max_steps,
        'steps_range': steps_range,
        'waypoint_steps': waypoint_steps,
        'waypoint_range': waypoint_range,
        'num_decisions_cmdr': num_decisions_cmdr,
        'scout_self_dim': scout_self_dim,
        'fixed_self_dim': fixed_self_dim,
        'scout_msg_dim': scout_msg_dim,
        'hidden_dim_scout': hidden_dim_scout,
        'hidden_dim_cmdr': hidden_dim_cmdr,
        'gamma': gamma,
        'gamma_cmdr': gamma_cmdr,
        'gae_lambda': gae_lambda,
        'log_episodes': log_episodes,
        'log_dir': log_dir,
    }

    # ── Networks ─────────────────────────────────────────────────────────
    scout_actor = ScoutActor(
        self_state_dim=scout_self_dim,
        msg_dim=scout_msg_dim,
        hidden_dim=hidden_dim_scout,
    ).to(device)
    print(f"ScoutActor params: {sum(p.numel() for p in scout_actor.parameters()):,}")

    if resume_scout and os.path.isfile(resume_scout):
        ckpt = torch.load(resume_scout, map_location=device)
        model_shapes = {k: v.shape for k, v in scout_actor.state_dict().items()}
        filtered = {k: v for k, v in ckpt.items()
                    if k in model_shapes and v.shape == model_shapes[k]}
        skipped = [k for k in ckpt if k not in filtered]
        scout_actor.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")
        print(f"  Loaded scout from {resume_scout}")

    cmdr_actor = CommanderActor(
        self_state_dim=fixed_self_dim,
        msg_input_dim=scout_msg_dim,
        action_dim=4,
        hidden_dim=hidden_dim_cmdr,
    ).to(device)
    print(f"Commander params: {sum(p.numel() for p in cmdr_actor.parameters()):,}")

    if resume_cmdr and os.path.isfile(resume_cmdr):
        ckpt = torch.load(resume_cmdr, map_location=device)
        cmdr_actor.load_state_dict(ckpt, strict=False)
        print(f"  Loaded commander from {resume_cmdr}")

    # ── Optimizers (fully separate) ──────────────────────────────────────
    optimizer_scout = optim.Adam(scout_actor.parameters(), lr=lr_scout)
    cmdr_main_params = [p for n, p in cmdr_actor.named_parameters()
                        if n != 'action_logstd']
    optimizer_cmdr = optim.Adam([
        {"params": cmdr_main_params, "lr": lr_cmdr},
        {"params": [cmdr_actor.action_logstd], "lr": lr_cmdr},
    ])

    # ── Critics (privileged, CTDE) ───────────────────────────────────────
    scout_priv_dim = scout_self_dim + 6   # 15 + 6 = 21
    cmdr_priv_dim = fixed_self_dim + 6    # 17 + 6 = 23

    critic_scout = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim_scout).to(device)
    critic_cmdr = PrivilegedCritic(cmdr_priv_dim, hidden_dim=hidden_dim_cmdr).to(device)
    print(f"ScoutCritic params: {sum(p.numel() for p in critic_scout.parameters()):,}")
    print(f"CmdrCritic params:  {sum(p.numel() for p in critic_cmdr.parameters()):,}")

    optimizer_critic_scout = optim.Adam(critic_scout.parameters(), lr=lr_critic)
    optimizer_critic_cmdr = optim.Adam(critic_cmdr.parameters(), lr=lr_critic)

    # ── Tracking ─────────────────────────────────────────────────────────
    reward_history = []
    loss_history_scout = []
    loss_history_cmdr = []
    critic_loss_history_scout = []
    critic_loss_history_cmdr = []
    lifespan_history = []
    ep_max_steps_per_batch = []   # avg ep_max_steps per batch
    logstd_history_cmdr = []

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_models", "multi")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints → {save_dir}\n")

    best_avg = -1e9
    episodes_played = 0
    num_batches = num_episodes // episodes_per_batch

    # ── Main training loop ───────────────────────────────────────────────
    for batch_idx in range(1, num_batches + 1):

        scout_w = {k: v.cpu() for k, v in scout_actor.state_dict().items()}
        cmdr_w = {k: v.cpu() for k, v in cmdr_actor.state_dict().items()}
        cs_w = {k: v.cpu() for k, v in critic_scout.state_dict().items()}
        cc_w = {k: v.cpu() for k, v in critic_cmdr.state_dict().items()}

        t0 = time.time()

        # Aggregate buffers
        agg_scout = {k: [] for k in [
            "maps", "self_states", "neighbor_states", "neighbor_masks",
            "actions", "logprobs", "returns", "values", "critic_states"
        ]}
        agg_cmdr = {k: [] for k in [
            "states", "messages", "msg_masks",
            "actions", "logprobs", "returns", "alive", "values", "critic_states"
        ]}
        agg_h = {"scout": [], "cmdr": []}
        batch_rewards = []
        batch_deaths = []

        # Pick episode length for this batch (same for all workers)
        if steps_range is not None:
            batch_ep_max = random.randint(steps_range[0] // 100, steps_range[1] // 100) * 100
        else:
            batch_ep_max = max_steps
        worker_config['ep_max_steps'] = batch_ep_max

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    collect_multi_worker,
                    eps_per_worker, scout_w, cmdr_w, cs_w, cc_w,
                    worker_config,
                    episodes_played + i * eps_per_worker
                )
                for i in range(num_workers)
            ]
            failed = 0
            for fut in futures:
                try:
                    w_scout, w_cmdr, w_h, w_rew, w_life, w_deaths = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"   ⚠ Worker failed: {e}")
                    continue
                batch_rewards.extend(w_rew)
                batch_deaths.extend(w_deaths)
                lifespan_history.extend(w_life)
                reward_history.extend(w_rew)
                episodes_played += len(w_rew)

                for k in agg_scout:
                    agg_scout[k].append(w_scout[k])
                for k in agg_cmdr:
                    agg_cmdr[k].append(w_cmdr[k])
                if w_h["scout"] is not None:
                    agg_h["scout"].append(w_h["scout"])
                if w_h["cmdr"] is not None:
                    agg_h["cmdr"].append(w_h["cmdr"])

        rollout_time = time.time() - t0

        if failed > 0:
            print(f"   ⚠ {failed}/{num_workers} workers failed this batch")
        if not batch_rewards:
            print(f"   ⚠ All workers failed, skipping batch {batch_idx}")
            continue

        # Logging
        avg_batch = float(np.mean(batch_rewards))
        ep_max_steps_per_batch.append(batch_ep_max)
        win = min(60, len(reward_history))
        avg_roll = float(np.mean(reward_history[-win:]))

        with torch.no_grad():
            cur_stds = torch.exp(
                cmdr_actor.action_logstd.clamp(-3.0, 0.0)).squeeze()
        logstd_history_cmdr.append(cur_stds.cpu().numpy().copy())

        from collections import Counter
        # Parse "s:cause,c:cause" death strings into separate counters
        scout_deaths_c = Counter()
        cmdr_deaths_c = Counter()
        for d in batch_deaths:
            parts = d.split(",")
            for p in parts:
                role, cause = p.split(":", 1)
                if role == "s":
                    scout_deaths_c[cause] += 1
                else:
                    cmdr_deaths_c[cause] += 1
        s_str = " ".join(f"{k}={v}" for k, v in scout_deaths_c.most_common())
        c_str = " ".join(f"{k}={v}" for k, v in cmdr_deaths_c.most_common())
        recent_life = float(np.mean(lifespan_history[-win:])) if lifespan_history else 0

        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"Batch {batch_idx:04d} (Ep {episodes_played:05d}) | "
              f"R: {avg_batch:+8.1f} ({avg_roll:+8.1f})  "
              f"Life: {recent_life:.0f}  "
              f"std=[{cur_stds[0]:.2f},{cur_stds[1]:.2f},{cur_stds[2]:.2f},{cur_stds[3]:.2f}]  "
              f"scout:[{s_str}] cmdr:[{c_str}]  "
              f"{rollout_time:.1f}s")

        # Save best
        if episodes_played >= 60 and avg_roll > best_avg:
            best_avg = avg_roll
            torch.save(scout_actor.state_dict(),
                       os.path.join(save_dir, "scout_best.pt"))
            torch.save(cmdr_actor.state_dict(),
                       os.path.join(save_dir, "cmdr_best.pt"))
            print(f"   ⭐ New best! rolling avg = {best_avg:.1f}")

        if batch_idx % 10 == 0:
            torch.save(scout_actor.state_dict(),
                       os.path.join(save_dir, f"scout_b{batch_idx:04d}.pt"))
            torch.save(cmdr_actor.state_dict(),
                       os.path.join(save_dir, f"cmdr_b{batch_idx:04d}.pt"))

        # ==============================================================
        # PPO UPDATE — SCOUT
        # ==============================================================
        s_maps = torch.cat(agg_scout["maps"]).to(device)
        s_self = torch.cat(agg_scout["self_states"]).to(device)
        s_neigh_s = torch.cat(agg_scout["neighbor_states"]).to(device)
        s_neigh_m = torch.cat(agg_scout["neighbor_masks"]).to(device)
        s_actions = torch.cat(agg_scout["actions"]).to(device)
        s_logprobs = torch.cat(agg_scout["logprobs"]).to(device)
        s_returns = torch.cat(agg_scout["returns"]).to(device)
        s_values = torch.cat(agg_scout["values"]).to(device)
        s_cstates = torch.cat(agg_scout["critic_states"]).to(device)

        h_scout = (torch.cat(agg_h["scout"], dim=0)
                   .squeeze(1).unsqueeze(0).to(device))

        # Advantages = returns - values (GAE already computed correctly)
        s_adv = s_returns - s_values
        if s_adv.numel() > 1:
            s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-8)

        # Normalize returns for critic target (prevents MSE explosion)
        s_ret_mean = s_returns.mean()
        s_ret_std = s_returns.std() + 1e-8
        s_returns_norm = (s_returns - s_ret_mean) / s_ret_std

        # Reshape: [episodes, max_steps, ...]
        eps = s_returns.numel() // max_steps  # actual eps (may be < episodes_per_batch if workers failed)
        s_maps_seq = s_maps.view(eps, max_steps, 1, 32, 32)
        s_self_seq = s_self.view(eps, max_steps, -1)
        s_neigh_s_seq = s_neigh_s.view(eps, max_steps, s_neigh_s.size(-2), 3)
        s_neigh_m_seq = s_neigh_m.view(eps, max_steps, -1)
        s_actions_seq = s_actions.view(eps, max_steps, -1)
        s_logprobs_seq = s_logprobs.view(eps, max_steps)
        s_adv_seq = s_adv.view(eps, max_steps)
        s_returns_seq = s_returns.view(eps, max_steps)
        s_returns_norm_seq = s_returns_norm.view(eps, max_steps)
        s_cstates_seq = s_cstates.view(eps, max_steps, -1)
        h_scout_seq = h_scout.transpose(0, 1)

        num_minibatches = 4
        mb_size_s = max(1, eps // num_minibatches)
        scout_loss_total = 0.0
        scout_critic_loss_total = 0.0

        for epoch in range(update_epochs):
            b_inds = np.random.permutation(eps)
            for start in range(0, eps, mb_size_s):
                mb = b_inds[start:start + mb_size_s]

                mb_maps = s_maps_seq[mb]
                mb_self = s_self_seq[mb]
                mb_ns = s_neigh_s_seq[mb]
                mb_nm = s_neigh_m_seq[mb]
                mb_acts = s_actions_seq[mb]
                mb_old_lp = s_logprobs_seq[mb].view(-1)
                mb_adv = s_adv_seq[mb].reshape(-1)
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

                loss_s = surr.mean() - entropy_scout * entropy.mean()

                if torch.isfinite(loss_s):
                    optimizer_scout.zero_grad()
                    loss_s.backward()
                    nn.utils.clip_grad_norm_(scout_actor.parameters(), max_norm=0.5)
                    optimizer_scout.step()
                    scout_loss_total += loss_s.item()

                # Critic value loss (separate optimizer)
                v_pred, _ = critic_scout(mb_cs, None)
                v_loss = F.mse_loss(v_pred, mb_rets)
                if torch.isfinite(v_loss):
                    optimizer_critic_scout.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic_scout.parameters(), max_norm=0.5)
                    optimizer_critic_scout.step()
                    scout_critic_loss_total += v_loss.item()

        loss_history_scout.append(
            scout_loss_total / max(1, update_epochs * num_minibatches))
        critic_loss_history_scout.append(
            scout_critic_loss_total / max(1, update_epochs * num_minibatches))

        # ==============================================================
        # PPO UPDATE — COMMANDER
        # ==============================================================
        c_states = torch.cat(agg_cmdr["states"]).to(device)
        c_msgs = torch.cat(agg_cmdr["messages"]).to(device)
        c_msg_m = torch.cat(agg_cmdr["msg_masks"]).to(device)
        c_actions = torch.cat(agg_cmdr["actions"]).to(device)
        c_logprobs = torch.cat(agg_cmdr["logprobs"]).to(device)
        c_returns = torch.cat(agg_cmdr["returns"]).to(device)
        cmdr_loss_total = 0.0
        cmdr_critic_loss_total = 0.0
        c_alive = torch.cat(agg_cmdr["alive"]).to(device)
        c_values = torch.cat(agg_cmdr["values"]).to(device)
        c_cstates = torch.cat(agg_cmdr["critic_states"]).to(device)

        h_cmdr = (torch.cat(agg_h["cmdr"], dim=0)
                  .squeeze(1).unsqueeze(0).to(device))

        # Advantages = returns - values (normalized over alive only)
        c_adv = c_returns - c_values
        alive_bool = c_alive > 0.5
        if alive_bool.sum() > 1:
            alive_vals = c_adv[alive_bool]
            c_adv = (c_adv - alive_vals.mean()) / (alive_vals.std() + 1e-8)

        # Normalize returns for critic target
        alive_rets = c_returns[alive_bool] if alive_bool.sum() > 1 else c_returns
        c_ret_mean = alive_rets.mean()
        c_ret_std = alive_rets.std() + 1e-8
        c_returns_norm = (c_returns - c_ret_mean) / c_ret_std

        # Reshape: [episodes, num_decisions_cmdr, ...]
        nd = num_decisions_cmdr
        c_states_seq = c_states.view(eps, nd, -1)
        c_msgs_seq = c_msgs.view(eps, nd, c_msgs.size(-2), c_msgs.size(-1))
        c_msg_m_seq = c_msg_m.view(eps, nd, c_msg_m.size(-1))
        c_actions_seq = c_actions.view(eps, nd, -1)
        c_logprobs_seq = c_logprobs.view(eps, nd)
        c_adv_seq = c_adv.view(eps, nd)
        c_alive_seq = c_alive.view(eps, nd)
        c_returns_seq = c_returns.view(eps, nd)
        c_returns_norm_seq = c_returns_norm.view(eps, nd)
        c_cstates_seq = c_cstates.view(eps, nd, -1)
        h_cmdr_seq = h_cmdr.transpose(0, 1)

        mb_size_c = max(1, eps // num_minibatches)

        for epoch in range(update_epochs):
            b_inds = np.random.permutation(eps)
            for start in range(0, eps, mb_size_c):
                mb = b_inds[start:start + mb_size_c]
                curr_mb = len(mb)

                mb_states = c_states_seq[mb]
                mb_msgs = c_msgs_seq[mb]
                mb_mm = c_msg_m_seq[mb]
                mb_acts = c_actions_seq[mb]
                mb_old_lp = c_logprobs_seq[mb].view(-1)
                mb_adv = c_adv_seq[mb].reshape(-1)
                mb_alive = c_alive_seq[mb].reshape(-1)
                mb_rets = c_returns_norm_seq[mb].reshape(-1)
                mb_cs = c_cstates_seq[mb]
                mb_h = h_cmdr_seq[mb].transpose(0, 1)

                dist, _, _ = cmdr_actor(mb_states, mb_msgs, mb_mm, mb_h)

                flat_acts = mb_acts.view(-1, 4)
                new_lp = dist.log_prob(flat_acts).sum(1)
                entropy = dist.entropy().sum(1)

                log_ratio = (new_lp - mb_old_lp).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                surr = torch.max(pg1, pg2)

                n_alive = mb_alive.sum().clamp(min=1.0)
                loss_c = ((surr * mb_alive).sum() / n_alive
                          - entropy_cmdr * (entropy * mb_alive).sum() / n_alive)

                if torch.isfinite(loss_c):
                    optimizer_cmdr.zero_grad()
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(cmdr_actor.parameters(), max_norm=0.5)
                    optimizer_cmdr.step()
                    cmdr_loss_total += loss_c.item()

                # Critic value loss (separate optimizer, masked by alive)
                v_pred, _ = critic_cmdr(mb_cs, None)
                v_err = (v_pred - mb_rets) ** 2
                v_loss_c = (v_err * mb_alive).sum() / n_alive
                if torch.isfinite(v_loss_c):
                    optimizer_critic_cmdr.zero_grad()
                    v_loss_c.backward()
                    nn.utils.clip_grad_norm_(critic_cmdr.parameters(), max_norm=0.5)
                    optimizer_critic_cmdr.step()
                    cmdr_critic_loss_total += v_loss_c.item()

        loss_history_cmdr.append(
            cmdr_loss_total / max(1, update_epochs * num_minibatches))
        critic_loss_history_cmdr.append(
            cmdr_critic_loss_total / max(1, update_epochs * num_minibatches))

        # ==============================================================
        # PERIODIC SAVE & PLOT
        # ==============================================================
        if batch_idx % 10 == 0:
            _save_plot_multi(
                reward_history, loss_history_scout, loss_history_cmdr,
                lifespan_history, logstd_history_cmdr,
                save_dir, batch_idx, ep_max_steps_per_batch,
                critic_loss_history_scout, critic_loss_history_cmdr)

    _save_plot_multi(
        reward_history, loss_history_scout, loss_history_cmdr,
        lifespan_history, logstd_history_cmdr,
        save_dir, batch_idx, ep_max_steps_per_batch,
        critic_loss_history_scout, critic_loss_history_cmdr)

    print(f"\n✅ Training complete!")
    print(f"   Best: {save_dir}/scout_best.pt + cmdr_best.pt")


# =============================================================================
# PLOT HELPER
# =============================================================================

def _save_plot_multi(rewards, loss_s, loss_c, lifespans, logstds,
                     save_dir, batch_idx, ep_max_per_batch=None,
                     critic_loss_s=None, critic_loss_c=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Multi-Agent — Batch {batch_idx}", fontsize=13)

    # Reward
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.2, color='steelblue', linewidth=0.5)
    if len(rewards) >= 30:
        ma = np.convolve(rewards, np.ones(30) / 30, mode='valid')
        ax.plot(range(29, len(rewards)), ma, color='navy', linewidth=1.5,
                label='MA 30')
        ax.legend()
    ax.set_title("Total Reward per Episode")
    ax.grid(True, alpha=0.3)

    # Scout Loss
    ax = axes[0, 1]
    ax.plot(loss_s, color='green', linewidth=1, label='Scout')
    ax.set_title("Scout PPO Loss")
    ax.grid(True, alpha=0.3)

    # Commander Loss
    ax = axes[0, 2]
    ax.plot(loss_c, color='tomato', linewidth=1, label='Commander')
    ax.set_title("Commander PPO Loss")
    ax.grid(True, alpha=0.3)

    # Lifespan
    ax = axes[1, 0]
    ax.plot(lifespans, alpha=0.3, color='orange', linewidth=0.5)
    if len(lifespans) >= 30:
        ma2 = np.convolve(lifespans, np.ones(30) / 30, mode='valid')
        ax.plot(range(29, len(lifespans)), ma2, color='darkorange',
                linewidth=1.5, label='MA 30')
        ax.legend()
    ax.set_title("Avg Lifespan (steps)")
    ax.grid(True, alpha=0.3)
    # Show avg ep_max_steps per batch as dashed line
    if ep_max_per_batch and len(ep_max_per_batch) > 0:
        # Expand per-batch to per-episode by repeating each value
        eps_per_b = max(1, len(lifespans) // max(1, len(ep_max_per_batch)))
        ep_max_expanded = []
        for v in ep_max_per_batch:
            ep_max_expanded.extend([v] * eps_per_b)
        ep_max_expanded = ep_max_expanded[:len(lifespans)]
        ax.plot(ep_max_expanded, color='gray', linewidth=1.2, linestyle='--',
                alpha=0.6, label='max steps')
        ax.legend(loc='lower right', fontsize=7)

    # Commander Std
    ax = axes[1, 1]
    if len(logstds) > 0 and hasattr(logstds[0], '__len__'):
        arr = np.array(logstds)
        for i, (lbl, col) in enumerate(zip(
                ['dx', 'dy', 'Alt', 'Water'],
                ['blue', 'green', 'red', 'orange'])):
            ax.plot(arr[:, i], color=col, linewidth=1, label=lbl)
        ax.legend(fontsize=8)
    ax.set_title("Commander Action Std")
    ax.grid(True, alpha=0.3)

    # Critic Value Loss (6th panel)
    ax = axes[1, 2]
    if critic_loss_s and len(critic_loss_s) > 0:
        ax.plot(critic_loss_s, color='green', linewidth=1, label='Scout')
    if critic_loss_c and len(critic_loss_c) > 0:
        ax.plot(critic_loss_c, color='tomato', linewidth=1, label='Commander')
    ax.set_title("Critic Value Loss (MSE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_b{batch_idx:04d}.png"),
                dpi=100)
    plt.close()


# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-agent training: Scout + Commander")
    parser.add_argument("--resume-scout", type=str, default="")
    parser.add_argument("--resume-cmdr", type=str, default="")
    parser.add_argument("--log-episodes", action="store_true",
                        help="Save trajectory logs for replay (1 ep/worker/batch)")
    parser.add_argument("--log-dir", type=str, default="/tmp/ep_logs",
                        help="Directory for episode logs")
    args = parser.parse_args()

    train_multi(
        resume_scout=args.resume_scout,
        resume_cmdr=args.resume_cmdr,
        log_episodes=args.log_episodes,
        log_dir=args.log_dir,
    )
