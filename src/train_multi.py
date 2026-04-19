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
    safe_limit = max(50.0, map_half * 0.7)           # waypoints clipped to this box (proportional)
    boundary_emergency = max(50.0, map_half * 0.6)   # hard override: steer toward center
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
    scout_priv_dim = config['scout_self_dim'] + 6   # 16 + 6 = 22
    cmdr_priv_dim = config['fixed_self_dim'] + 6    # 19 + 6 = 25
    local_critic_scout = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim_scout)
    local_critic_scout.load_state_dict(critic_scout_w)
    local_critic_scout.eval()
    local_critic_cmdr = PrivilegedCritic(cmdr_priv_dim, hidden_dim=hidden_dim_cmdr)
    local_critic_cmdr.load_state_dict(critic_cmdr_w)
    local_critic_cmdr.eval()

    local_env = DroneFireEnv(
        num_quads=N_QUADS, num_fixed=config.get('N_FIXED', 1),
        grid_size_m=config['grid_size_m'], max_steps=max_steps
    )
    local_env.map_size_range = map_size_range

    # ── Buffers ──────────────────────────────────────────────────────────
    scout_buf = {k: [] for k in [
        "maps", "self_states", "neighbor_states", "neighbor_masks",
        "actions", "logprobs", "returns", "values",
        "critic_states", "alive"
    ]}
    cmdr_buf = {k: [] for k in [
        "states", "messages", "msg_masks",
        "actions", "logprobs", "returns", "alive", "values",
        "critic_states", "aux_targets"
    ]}

    scout_h0_list = []
    cmdr_h0_list = []
    all_rewards = []
    all_scout_lifespans = []
    all_cmdr_lifespans = []
    all_deaths = []

    # Dummy tensors for dead scout steps
    d_map = torch.zeros(1, 1, 32, 32)
    d_scout_self = torch.zeros(1, config['scout_self_dim'])
    d_neigh_s = torch.zeros(1, max(1, N_QUADS - 1), 3)
    d_neigh_m = torch.ones(1, max(1, N_QUADS - 1), dtype=torch.bool)

    ep_max_steps = config.get('ep_max_steps', max_steps)
    N_FIXED = config.get('N_FIXED', 1)

    for ep_off in range(num_eps):
        local_env.max_steps = ep_max_steps

        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_off)

        # Recalculate after reset (map size may have changed)
        map_half = local_env.map_bounds
        safe_limit = max(50.0, map_half * 0.7)
        boundary_emergency = max(50.0, map_half * 0.6)

        quad_agents = local_env.quad_agents   # ["quad_0", "quad_1", ...]
        fixed_agents = local_env.fixed_agents  # ["fixed_0", ...]
        f_agent = fixed_agents[0] if fixed_agents else None

        # ── Per-scout state (mirrors train_scout.py) ─────────────────
        scout_h = {q: torch.zeros(1, 1, hidden_dim_scout) for q in quad_agents}
        critic_scout_h_dict = {q: torch.zeros(1, 1, hidden_dim_scout) for q in quad_agents}
        scout_ep_data = {q: [] for q in quad_agents}
        scout_alive = {q: True for q in quad_agents}
        scout_lifespan = {q: ep_max_steps for q in quad_agents}
        scout_death_cause = {q: "survived" for q in quad_agents}
        ep_reward_scout = {q: 0.0 for q in quad_agents}
        # Per-scout message buffer for commander (latest msg each step)
        scout_msg_tensors = {q: torch.zeros(1, scout_msg_dim) for q in quad_agents}
        scout_msg_valid = {q: False for q in quad_agents}

        for q in quad_agents:
            scout_h0_list.append(scout_h[q].clone())

        # ── Commander state ──────────────────────────────────────────
        cmdr_h = torch.zeros(1, 1, hidden_dim_cmdr)
        critic_cmdr_h = torch.zeros(1, 1, hidden_dim_cmdr)
        cmdr_h0_list.append(cmdr_h.clone())

        cmdr_ep_data = []
        ep_reward_cmdr = 0.0
        cmdr_alive = True
        cmdr_death_cause = "survived"
        total_cmdr_steps = 0
        cmdr_lifespan = ep_max_steps

        # Commander waypoint state
        target_x, target_y = 0.0, 0.0
        target_alt_raw = 0.0
        water_raw = -1.0
        steps_in_segment = 0
        wp_reached = False
        segment_reward = 0.0
        need_new_waypoint = True

        # Per-segment message buffer: list of (per_scout_msgs_dict, per_scout_valid_dict)
        msg_buffer = []

        # Trajectory logging (only for first episode per worker)
        log_this_ep = config.get('log_episodes', False) and ep_off == 0
        traj_scout_pos = {q: [] for q in quad_agents} if log_this_ep else None
        traj_cmdr_pos = [] if log_this_ep else None
        traj_cmdr_waypoints = [] if log_this_ep else None
        traj_rewards_scout = {q: [] for q in quad_agents} if log_this_ep else None
        traj_rewards_cmdr = [] if log_this_ep else None

        for step in range(max_steps):
            actions = {}

            # ==============================================================
            # SCOUT FORWARD PASS — all scouts, every step
            # ==============================================================
            for q in quad_agents:
                if scout_alive[q] and q in local_env.agents:
                    with torch.no_grad():
                        l_map = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0)
                        s_st = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0)
                        n_s = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0)
                        n_m = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0)

                        dist_s, msg_s, h_out_s = local_scout(l_map, s_st, n_s, n_m, scout_h[q])
                        act_s = dist_s.sample()

                        priv_s = torch.FloatTensor(
                            local_env.get_privileged_state(q)).unsqueeze(0)
                        v_s, c_h_out = local_critic_scout(priv_s, critic_scout_h_dict[q])

                    scout_ep_data[q].append({
                        "alive": True,
                        "map": l_map, "self": s_st, "n_s": n_s, "n_m": n_m,
                        "h": scout_h[q], "act": act_s,
                        "lp": dist_s.log_prob(act_s).sum(1),
                        "reward": 0.0,
                        "value": v_s.item(),
                        "critic_state": priv_s,
                    })
                    scout_h[q] = h_out_s
                    critic_scout_h_dict[q] = c_h_out
                    actions[q] = act_s.squeeze(0).numpy()
                    scout_msg_tensors[q] = msg_s.detach()
                    scout_msg_valid[q] = True
                else:
                    scout_alive[q] = False
                    scout_ep_data[q].append({
                        "alive": False, "reward": 0.0, "value": 0.0,
                        "critic_state": torch.zeros(1, scout_priv_dim),
                    })

            # Buffer all scout messages for commander (snapshot this step)
            msg_buffer.append({
                q: (scout_msg_tensors[q].clone(), scout_msg_valid[q])
                for q in quad_agents
            })

            # ==============================================================
            # COMMANDER WAYPOINT DECISION (every waypoint_steps or on reach)
            # ==============================================================
            if cmdr_alive and f_agent in local_env.agents:
                if need_new_waypoint:
                    # Build message input: [latest, best_fire] per scout → 2*N_QUADS slots
                    msgs_for_cmdr = []
                    masks_for_cmdr = []

                    for q in quad_agents:
                        # Latest message from this scout
                        if len(msg_buffer) > 0:
                            latest_msg, latest_valid = msg_buffer[-1][q]
                        else:
                            latest_msg = torch.zeros(1, scout_msg_dim)
                            latest_valid = False
                        msgs_for_cmdr.append(latest_msg)
                        masks_for_cmdr.append(not latest_valid)

                        # Best fire message from this scout (highest intensity at index 2)
                        best_msg = latest_msg
                        best_intensity = -1.0
                        for buf_entry in msg_buffer:
                            m_tensor, m_valid = buf_entry[q]
                            if m_valid and m_tensor[0, 2].item() > best_intensity:
                                best_intensity = m_tensor[0, 2].item()
                                best_msg = m_tensor
                        msgs_for_cmdr.append(best_msg)
                        masks_for_cmdr.append(not latest_valid)

                    # Stack: [1, 2*N_QUADS, msg_dim]
                    msgs_t = torch.stack(msgs_for_cmdr, dim=1)
                    msgs_m = torch.tensor([masks_for_cmdr])

                    # Forward pass
                    with torch.no_grad():
                        s_st_f = torch.FloatTensor(
                            obs[f_agent]["self_state"]).unsqueeze(0)

                        dist_c, _, h_out_c = local_cmdr(
                            s_st_f, msgs_t, msgs_m, cmdr_h)

                        act_c = dist_c.sample()

                        priv_c = torch.FloatTensor(
                            local_env.get_privileged_state(f_agent)).unsqueeze(0)
                        v_c, critic_cmdr_h = local_critic_cmdr(priv_c, critic_cmdr_h)

                    c_pos = local_env.sim.drones[f_agent].get_position()
                    vec_x = local_env.fire_x - c_pos[0]
                    vec_y = local_env.fire_y - c_pos[1]
                    dist_to_f = np.hypot(vec_x, vec_y)
                    true_dir = [vec_x/dist_to_f, vec_y/dist_to_f] if dist_to_f > 1.0 else [0.0, 0.0]

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
                        "aux_target": torch.FloatTensor([true_dir]),
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
                    msg_buffer = []

                    if log_this_ep:
                        traj_cmdr_waypoints.append(
                            [step, target_x, target_y, target_alt_raw, water_raw])

                # Heading controller
                fw = local_env.sim.drones.get(f_agent)
                if fw is not None:
                    pos = fw.get_position()

                    if (abs(pos[0]) > boundary_emergency or
                            abs(pos[1]) > boundary_emergency):
                        target_x = 0.0
                        target_y = 0.0
                        need_new_waypoint = True

                    dx_to = target_x - pos[0]
                    dy_to = target_y - pos[1]
                    dist_to = np.sqrt(dx_to**2 + dy_to**2)

                    if dist_to < wp_reached_dist:
                        wp_reached = True

                    if dist_to > 1.0:
                        desired_heading = np.arctan2(dy_to, dx_to)
                        cur_yaw = fw.get_orientation_rpy()[2]
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
            r_cmdr = 0.0
            if local_env.agents:
                obs, rewards, terms, truncs, infos = local_env.step(actions)

                # Scout rewards (per scout)
                for q in quad_agents:
                    r_q = rewards.get(q, 0.0)
                    if scout_alive[q] and q in local_env.agents:
                        scout_ep_data[q][-1]["reward"] = r_q
                        ep_reward_scout[q] += r_q
                    if terms.get(q, False) or truncs.get(q, False):
                        scout_alive[q] = False
                        scout_lifespan[q] = step + 1
                        if terms.get(q, False):
                            scout_death_cause[q] = infos.get(
                                q, {}).get("death_cause", "unknown")
                        else:
                            scout_death_cause[q] = "survived"

                # Commander segment reward
                r_cmdr = rewards.get(f_agent, 0.0) if f_agent else 0.0
                segment_reward += r_cmdr

                if f_agent and (terms.get(f_agent, False) or truncs.get(f_agent, False)):
                    cmdr_alive = False
                    cmdr_lifespan = step + 1
                    if terms.get(f_agent, False):
                        cmdr_death_cause = infos.get(
                            f_agent, {}).get("death_cause", "unknown")
                    else:
                        cmdr_death_cause = "survived"
                    if cmdr_ep_data:
                        cmdr_ep_data[-1]["reward"] = segment_reward
                    ep_reward_cmdr += segment_reward
            else:
                for q in quad_agents:
                    if scout_alive[q]:
                        scout_alive[q] = False
                        scout_lifespan[q] = step + 1
                        scout_death_cause[q] = "env_empty"
                if cmdr_alive:
                    cmdr_alive = False
                    cmdr_lifespan = step + 1
                    cmdr_death_cause = "env_empty"

            # Early exit
            if not any(scout_alive.values()) and not cmdr_alive:
                break

            # --- Trajectory recording ---
            if log_this_ep:
                for q in quad_agents:
                    s_drone = local_env.sim.drones.get(q)
                    traj_scout_pos[q].append(
                        s_drone.get_position().copy() if s_drone else np.full(3, np.nan))
                    traj_rewards_scout[q].append(
                        rewards.get(q, 0.0) if scout_alive[q] else 0.0)
                c_drone = local_env.sim.drones.get(f_agent) if f_agent else None
                traj_cmdr_pos.append(
                    c_drone.get_position().copy() if c_drone else np.full(3, np.nan))
                traj_rewards_cmdr.append(r_cmdr if cmdr_alive else 0.0)

            # ==============================================================
            # CHECK WAYPOINT SEGMENT END
            # ==============================================================
            if cmdr_alive and f_agent and f_agent in local_env.agents:
                segment_done = (
                    wp_reached or
                    steps_in_segment >= waypoint_steps or
                    step == max_steps - 1
                )
                if segment_done:
                    if not wp_reached and steps_in_segment >= waypoint_steps:
                        segment_reward += wp_timeout_penalty

                    if cmdr_ep_data:
                        cmdr_ep_data[-1]["reward"] = segment_reward
                    ep_reward_cmdr += segment_reward - r_cmdr
                    need_new_waypoint = True

        # ==================================================================
        # Fix ep_reward_cmdr: sum from cmdr_ep_data
        # ==================================================================
        ep_reward_cmdr = sum(d["reward"] for d in cmdr_ep_data)

        # ==================================================================
        # SCOUT GAE — per scout independently (like train_scout.py)
        # ==================================================================
        gamma = config['gamma']
        gae_lam = config['gae_lambda']

        for q in quad_agents:
            ep_data = scout_ep_data[q]
            gae = 0.0
            for i in reversed(range(len(ep_data))):
                d = ep_data[i]
                r_t = d["reward"]
                v_t = d["value"]
                v_next = ep_data[i + 1]["value"] if i + 1 < len(ep_data) else 0.0
                delta = r_t + gamma * v_next - v_t
                gae = delta + gamma * gae_lam * gae
                d["ret"] = gae + v_t

            # Pack buffer (max_steps entries per scout, padded)
            for i in range(max_steps):
                if i < len(ep_data):
                    d = ep_data[i]
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
        # PACK COMMANDER BUFFER (num_decisions_cmdr entries, padded)
        # ==================================================================
        n_msg_slots = 2 * N_QUADS
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
                cmdr_buf["aux_targets"].append(d["aux_target"])
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
                cmdr_buf["aux_targets"].append(torch.zeros(1, 2))

        # Episode totals
        avg_scout_reward = sum(ep_reward_scout.values()) / max(1, N_QUADS)
        ep_reward_total = avg_scout_reward + ep_reward_cmdr
        all_rewards.append(ep_reward_total)

        # Per-scout lifespans and deaths
        for q in quad_agents:
            all_scout_lifespans.append(scout_lifespan[q])
        all_cmdr_lifespans.append(cmdr_lifespan)

        # Death strings: one per scout + one for commander
        death_parts = []
        for q in quad_agents:
            death_parts.append(f"s:{scout_death_cause[q]}")
        death_parts.append(f"c:{cmdr_death_cause}")
        all_deaths.append(",".join(death_parts))

        # --- Episode log (for replay/analysis) ---
        if log_this_ep:
            log_dir = config.get('log_dir', '/tmp/ep_logs')
            os.makedirs(log_dir, exist_ok=True)
            ep_id = batch_start_idx + ep_off
            save_dict = dict(
                ep_id=np.array(ep_id),
                scout_reward=np.array([ep_reward_scout[q] for q in quad_agents]),
                cmdr_reward=np.array(ep_reward_cmdr),
                scout_lifespan=np.array([scout_lifespan[q] for q in quad_agents]),
                cmdr_lifespan=np.array(cmdr_lifespan),
                n_cmdr_decisions=np.array(n_actual_cmdr),
                cmdr_pos=np.array(traj_cmdr_pos),
                cmdr_waypoints=np.array(traj_cmdr_waypoints) if traj_cmdr_waypoints else np.zeros((0, 5)),
                rewards_cmdr=np.array(traj_rewards_cmdr),
                map_bounds=np.array(local_env.grid_size_m / 2.0),
            )
            for qi, q in enumerate(quad_agents):
                save_dict[f'scout_{qi}_pos'] = np.array(traj_scout_pos[q])
                save_dict[f'scout_{qi}_rewards'] = np.array(traj_rewards_scout[q])
            np.savez_compressed(
                os.path.join(log_dir, f'ep_{ep_id:06d}.npz'), **save_dict)

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

    return out_scout, out_cmdr, out_init_h, all_rewards, (all_scout_lifespans, all_cmdr_lifespans), all_deaths


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_multi(resume_scout="", resume_cmdr="",
                log_episodes=False, log_dir="/tmp/ep_logs"):
    print("=" * 70)
    print("  Multi-Agent Training: Scout (frame-by-frame) + Commander (waypoint)")
    print("  Commander: CommanderActor (with scout messages)")
    print("=" * 70)

    # Resolve resume paths BEFORE chdir so relative paths work
    if resume_scout:
        resume_scout = os.path.abspath(resume_scout)
    if resume_cmdr:
        resume_cmdr = os.path.abspath(resume_cmdr)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, PrivilegedCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}\n")

    # ── Hyperparameters ──────────────────────────────────────────────────
    N_QUADS = 2
    N_FIXED = 1
    grid_size_m = 1000.0
    map_size_range = (500, 1200)  # random map size per episode [m]
    num_episodes = 30_000
    max_steps = 1200              # buffer size — full episode length
    steps_range = (500, 1200)    # long episodes; BPTT chunked to 128 steps
    bptt_chunk = 128              # GRU backprop limit per chunk
    waypoint_steps = 30
    waypoint_range = 200.0
    num_decisions_cmdr = max_steps // waypoint_steps  # 16

    # Scout freeze: keep scout frozen for first N batches so commander
    # learns to use messages before scout policy gets destabilised.
    scout_freeze_batches = 50

    gamma = 0.99
    gamma_cmdr = 0.95          # shorter horizon for commander decisions
    gae_lambda = 0.95
    clip_coef = 0.2
    update_epochs = 4
    num_workers = 15
    eps_per_worker = 2
    episodes_per_batch = num_workers * eps_per_worker

    lr_scout = 3e-5             # very low — fine-tuning pre-trained scout, prevent policy collapse
    lr_cmdr = 5e-4
    lr_critic = 3e-4
    hidden_dim_scout = 128
    hidden_dim_cmdr = 64
    scout_msg_dim = 5

    entropy_scout = 0.01        # Phase 8: higher entropy to allow re-exploration after policy drift
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
        model_shapes = {k: v.shape for k, v in cmdr_actor.state_dict().items()}
        filtered = {k: v for k, v in ckpt.items()
                    if k in model_shapes and v.shape == model_shapes[k]}
        skipped = [k for k in ckpt if k not in filtered]
        cmdr_actor.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")
        print(f"  Loaded commander from {resume_cmdr}")

    # ── Optimizers (fully separate) ──────────────────────────────────────
    optimizer_scout = optim.Adam(scout_actor.parameters(), lr=lr_scout)
    # Start with scout frozen if requested
    scout_frozen = scout_freeze_batches > 0
    if scout_frozen:
        for p in scout_actor.parameters():
            p.requires_grad = False
        print(f"  Scout FROZEN for first {scout_freeze_batches} batches")
    cmdr_main_params = [p for n, p in cmdr_actor.named_parameters()
                        if n != 'action_logstd']
    optimizer_cmdr = optim.Adam([
        {"params": cmdr_main_params, "lr": lr_cmdr},
        {"params": [cmdr_actor.action_logstd], "lr": lr_cmdr},
    ])

    # ── Critics (privileged, CTDE) ───────────────────────────────────────
    scout_priv_dim = scout_self_dim + 6   # 16 + 6 = 22
    cmdr_priv_dim = fixed_self_dim + 6    # 19 + 6 = 25

    critic_scout = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim_scout).to(device)
    critic_cmdr = PrivilegedCritic(cmdr_priv_dim, hidden_dim=hidden_dim_cmdr).to(device)
    print(f"ScoutCritic params: {sum(p.numel() for p in critic_scout.parameters()):,}")
    print(f"CmdrCritic params:  {sum(p.numel() for p in critic_cmdr.parameters()):,}")

    optimizer_critic_scout = optim.Adam(critic_scout.parameters(), lr=lr_critic)
    optimizer_critic_cmdr = optim.Adam(critic_cmdr.parameters(), lr=lr_critic)

    # ── Tracking (all per-batch, not per-episode) ──────────────────────
    reward_history = []           # per-episode (for rolling avg)
    reward_per_batch = []         # avg reward per batch
    loss_history_scout = []
    loss_history_cmdr = []
    critic_loss_history_scout = []
    critic_loss_history_cmdr = []
    scout_life_pct_history = []   # avg scout lifespan % per batch
    cmdr_life_pct_history = []    # avg cmdr lifespan % per batch
    scout_death_history = []      # list of Counter dicts per batch

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_models", "multi")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints → {save_dir}\n")

    best_avg = -1e9
    episodes_played = 0
    num_batches = num_episodes // episodes_per_batch

    # ── Main training loop ───────────────────────────────────────────────
    for batch_idx in range(1, num_batches + 1):

        # Unfreeze scout after warmup period
        if scout_frozen and batch_idx > scout_freeze_batches:
            for p in scout_actor.parameters():
                p.requires_grad = True
            # Rebuild optimizer with all params now trainable
            optimizer_scout = optim.Adam(scout_actor.parameters(), lr=lr_scout)
            scout_frozen = False
            print(f"   🔓 Scout UNFROZEN at batch {batch_idx}")

        scout_w = {k: v.cpu() for k, v in scout_actor.state_dict().items()}
        cmdr_w = {k: v.cpu() for k, v in cmdr_actor.state_dict().items()}
        cs_w = {k: v.cpu() for k, v in critic_scout.state_dict().items()}
        cc_w = {k: v.cpu() for k, v in critic_cmdr.state_dict().items()}

        t0 = time.time()

        # Aggregate buffers
        agg_scout = {k: [] for k in [
            "maps", "self_states", "neighbor_states", "neighbor_masks",
            "actions", "logprobs", "returns", "values", "critic_states", "alive"
        ]}
        agg_cmdr = {k: [] for k in [
            "states", "messages", "msg_masks",
            "actions", "logprobs", "returns", "alive", "values", "critic_states", "aux_targets"
        ]}
        agg_h = {"scout": [], "cmdr": []}
        batch_rewards = []
        batch_deaths = []
        batch_scout_lifespans = []
        batch_cmdr_lifespans = []

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
                    w_scout, w_cmdr, w_h, w_rew, (w_scout_life, w_cmdr_life), w_deaths = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"   ⚠ Worker failed: {e}")
                    continue
                batch_rewards.extend(w_rew)
                batch_deaths.extend(w_deaths)
                batch_scout_lifespans.extend(w_scout_life)
                batch_cmdr_lifespans.extend(w_cmdr_life)
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
        reward_per_batch.append(avg_batch)
        win = min(60, len(reward_history))
        avg_roll = float(np.mean(reward_history[-win:]))

        # Per-batch lifespan as % of ep_max
        scout_life_pct = float(np.mean(batch_scout_lifespans)) / batch_ep_max * 100
        cmdr_life_pct = float(np.mean(batch_cmdr_lifespans)) / batch_ep_max * 100
        scout_life_pct_history.append(scout_life_pct)
        cmdr_life_pct_history.append(cmdr_life_pct)

        with torch.no_grad():
            cur_stds = torch.exp(
                cmdr_actor.action_logstd.clamp(-3.0, 0.0)).squeeze()

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
        scout_death_history.append(dict(scout_deaths_c))

        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"Batch {batch_idx:04d} (Ep {episodes_played:05d}) | "
              f"R: {avg_batch:+8.1f} ({avg_roll:+8.1f})  "
              f"Scout:{scout_life_pct:.0f}% Cmdr:{cmdr_life_pct:.0f}%  "
              f"scout:[{s_str}] cmdr:[{c_str}]  "
              f"{rollout_time:.1f}s")

        # Save best
        if episodes_played >= 60 and avg_roll > best_avg:
            best_avg = avg_roll
            torch.save(scout_actor.state_dict(),
                       os.path.join(save_dir, "scout_best.pt"))
            torch.save(cmdr_actor.state_dict(),
                       os.path.join(save_dir, "cmdr_best.pt"))
            torch.save(critic_scout.state_dict(),
                       os.path.join(save_dir, "critic_scout_best.pt"))
            torch.save(critic_cmdr.state_dict(),
                       os.path.join(save_dir, "critic_cmdr_best.pt"))
            print(f"   ⭐ New best! rolling avg = {best_avg:.1f}")

        if batch_idx % 10 == 0:
            torch.save(scout_actor.state_dict(),
                       os.path.join(save_dir, f"scout_b{batch_idx:04d}.pt"))
            torch.save(cmdr_actor.state_dict(),
                       os.path.join(save_dir, f"cmdr_b{batch_idx:04d}.pt"))
            torch.save(critic_scout.state_dict(),
                       os.path.join(save_dir, f"critic_scout_b{batch_idx:04d}.pt"))
            torch.save(critic_cmdr.state_dict(),
                       os.path.join(save_dir, f"critic_cmdr_b{batch_idx:04d}.pt"))

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
        s_alive = torch.cat(agg_scout["alive"]).to(device)

        h_scout = (torch.cat(agg_h["scout"], dim=0)
                   .squeeze(1).unsqueeze(0).to(device))

        # Advantages = returns - values (GAE already computed on workers)
        s_adv = s_returns - s_values
        alive_bool_s = s_alive > 0.5
        if alive_bool_s.sum() > 1:
            alive_adv = s_adv[alive_bool_s]
            s_adv = (s_adv - alive_adv.mean()) / (alive_adv.std() + 1e-8)

        # Normalize returns for critic target (prevents MSE explosion)
        alive_rets_s = s_returns[alive_bool_s] if alive_bool_s.sum() > 1 else s_returns
        s_ret_mean = alive_rets_s.mean()
        s_ret_std = alive_rets_s.std() + 1e-8
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
        s_alive_seq = s_alive.view(eps, max_steps)
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
                mb_old_lp = s_logprobs_seq[mb]
                mb_adv = s_adv_seq[mb]
                mb_alive_s = s_alive_seq[mb]
                mb_rets = s_returns_norm_seq[mb]
                mb_cs = s_cstates_seq[mb]
                mb_h = h_scout_seq[mb].transpose(0, 1)

                # ── Chunked BPTT: process sequence in bptt_chunk-size windows ──
                # GRU hidden state is detached between chunks so gradients
                # only flow back ~bptt_chunk steps (not the full episode).
                chunk_new_lps = []
                chunk_entropies = []
                h_chunk = mb_h.detach()
                T = mb_maps.size(1)  # actual sequence length (= max_steps)

                for t0 in range(0, T, bptt_chunk):
                    t1 = min(t0 + bptt_chunk, T)
                    c_maps = mb_maps[:, t0:t1]
                    c_self = mb_self[:, t0:t1]
                    c_ns = mb_ns[:, t0:t1]
                    c_nm = mb_nm[:, t0:t1]
                    c_acts = mb_acts[:, t0:t1].reshape(-1, 4)

                    dist_c, _, h_chunk = scout_actor(c_maps, c_self, c_ns, c_nm, h_chunk)
                    chunk_new_lps.append(dist_c.log_prob(c_acts).sum(1))
                    chunk_entropies.append(dist_c.entropy().sum(1))
                    h_chunk = h_chunk.detach()  # cut gradient flow between chunks

                new_lp = torch.cat(chunk_new_lps)
                entropy = torch.cat(chunk_entropies)
                flat_old_lp = mb_old_lp.reshape(-1)
                flat_adv = mb_adv.reshape(-1)
                flat_alive_s = mb_alive_s.reshape(-1)
                flat_rets = mb_rets.reshape(-1)

                log_ratio = (new_lp - flat_old_lp).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg1 = -flat_adv * ratio
                pg2 = -flat_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                surr = torch.max(pg1, pg2)

                n_alive_s = flat_alive_s.sum().clamp(min=1.0)
                loss_s = ((surr * flat_alive_s).sum() / n_alive_s
                          - entropy_scout * (entropy * flat_alive_s).sum() / n_alive_s)

                if torch.isfinite(loss_s):
                    if loss_s.requires_grad:
                        optimizer_scout.zero_grad()
                        loss_s.backward()
                        nn.utils.clip_grad_norm_(scout_actor.parameters(), max_norm=0.5)
                        optimizer_scout.step()
                    scout_loss_total += loss_s.item()

                # Critic value loss (separate optimizer, masked by alive)
                v_pred, _ = critic_scout(mb_cs, None)
                v_err = (v_pred - flat_rets) ** 2
                v_loss = (v_err * flat_alive_s).sum() / n_alive_s
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
        c_aux_targets = torch.cat(agg_cmdr["aux_targets"]).to(device)

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
        eps_cmdr = c_returns.numel() // nd  # 1 commander per real episode
        c_states_seq = c_states.view(eps_cmdr, nd, -1)
        c_msgs_seq = c_msgs.view(eps_cmdr, nd, c_msgs.size(-2), c_msgs.size(-1))
        c_msg_m_seq = c_msg_m.view(eps_cmdr, nd, c_msg_m.size(-1))
        c_actions_seq = c_actions.view(eps_cmdr, nd, -1)
        c_logprobs_seq = c_logprobs.view(eps_cmdr, nd)
        c_adv_seq = c_adv.view(eps_cmdr, nd)
        c_alive_seq = c_alive.view(eps_cmdr, nd)
        c_returns_seq = c_returns.view(eps_cmdr, nd)
        c_returns_norm_seq = c_returns_norm.view(eps_cmdr, nd)
        c_cstates_seq = c_cstates.view(eps_cmdr, nd, -1)
        h_cmdr_seq = h_cmdr.transpose(0, 1)
        c_aux_targets_seq = c_aux_targets.view(eps_cmdr, nd, -1)

        mb_size_c = max(1, eps_cmdr // num_minibatches)

        for epoch in range(update_epochs):
            b_inds = np.random.permutation(eps_cmdr)
            for start in range(0, eps_cmdr, mb_size_c):
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
                mb_aux_targets = c_aux_targets_seq[mb].reshape(-1, 2)

                # dist, _, _ = cmdr_actor(mb_states, mb_msgs, mb_mm, mb_h)
                dist, mb_aux_pred, _ = cmdr_actor(mb_states, mb_msgs, mb_mm, mb_h)

                flat_acts = mb_acts.view(-1, 4)
                # Phase 1
                # new_lp = dist.log_prob(flat_acts)[:, :2].sum(1)
                # entropy = dist.entropy()[:, :2].sum(1)
                # Phase 2
                # new_lp = dist.log_prob(flat_acts)[:, :3].sum(1)
                # entropy = dist.entropy()[:, :3].sum(1)
                # Phase 3 (full)
                new_lp = dist.log_prob(flat_acts).sum(1)
                entropy = dist.entropy().sum(1)
                
                log_ratio = (new_lp - mb_old_lp).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                surr = torch.max(pg1, pg2)

                n_alive = mb_alive.sum().clamp(min=1.0)
                policy_loss = (surr * mb_alive).sum() / n_alive
                entropy_loss = (entropy * mb_alive).sum() / n_alive

                # Výpočet Auxiliary Loss (Chyba predikce ohně)
                aux_error = (mb_aux_pred - mb_aux_targets)**2 # MSE
                # Průměrujeme jen přes živé kroky
                aux_loss = (aux_error.sum(dim=-1) * mb_alive).sum() / n_alive 

                # Přičteme aux_loss k celkové ztrátě (s váhou např. 0.5, aby to nepřebilo PPO)
                loss_c = policy_loss - entropy_cmdr * entropy_loss + 0.5 * aux_loss

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
                reward_per_batch, loss_history_scout, loss_history_cmdr,
                scout_life_pct_history, cmdr_life_pct_history,
                scout_death_history,
                save_dir, batch_idx,
                critic_loss_history_scout, critic_loss_history_cmdr)

    _save_plot_multi(
        reward_per_batch, loss_history_scout, loss_history_cmdr,
        scout_life_pct_history, cmdr_life_pct_history,
        scout_death_history,
        save_dir, batch_idx,
        critic_loss_history_scout, critic_loss_history_cmdr)

    print(f"\n✅ Training complete!")
    print(f"   Best: {save_dir}/scout_best.pt + cmdr_best.pt")


# =============================================================================
# PLOT HELPER
# =============================================================================

def _save_plot_multi(reward_batches, loss_s, loss_c,
                     scout_life_pct, cmdr_life_pct, scout_deaths,
                     save_dir, batch_idx,
                     critic_loss_s=None, critic_loss_c=None):
    """All data is per-batch (x-axis = batch number)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Multi-Agent — Batch {batch_idx}", fontsize=13)
    batches = np.arange(1, len(reward_batches) + 1)

    def _ma(data, w=10):
        if len(data) < w:
            return None, None
        ma = np.convolve(data, np.ones(w) / w, mode='valid')
        return np.arange(w, len(data) + 1), ma

    # (0,0) Reward per Batch
    ax = axes[0, 0]
    ax.bar(batches, reward_batches, color='steelblue', alpha=0.4, width=1.0)
    mx, ma = _ma(reward_batches)
    if ma is not None:
        ax.plot(mx, ma, color='navy', linewidth=2, label='MA 10')
        ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    ax.set_title("Avg Reward per Batch")
    ax.set_xlabel("Batch")
    ax.grid(True, alpha=0.3)

    # (0,1) Scout PPO Loss
    ax = axes[0, 1]
    if loss_s:
        ax.plot(range(1, len(loss_s) + 1), loss_s, color='green', linewidth=1)
    ax.set_title("Scout PPO Loss")
    ax.set_xlabel("Batch")
    ax.grid(True, alpha=0.3)

    # (0,2) Commander PPO Loss
    ax = axes[0, 2]
    if loss_c:
        ax.plot(range(1, len(loss_c) + 1), loss_c, color='tomato', linewidth=1)
    ax.set_title("Commander PPO Loss")
    ax.set_xlabel("Batch")
    ax.grid(True, alpha=0.3)

    # (1,0) Lifespan % (scout + commander)
    ax = axes[1, 0]
    ax.plot(batches, scout_life_pct, color='green', linewidth=1, alpha=0.5)
    ax.plot(batches, cmdr_life_pct, color='tomato', linewidth=1, alpha=0.5)
    mx_s, ma_s = _ma(scout_life_pct)
    mx_c, ma_c = _ma(cmdr_life_pct)
    if ma_s is not None:
        ax.plot(mx_s, ma_s, color='green', linewidth=2, label='Scout MA10')
    if ma_c is not None:
        ax.plot(mx_c, ma_c, color='tomato', linewidth=2, label='Cmdr MA10')
    ax.axhline(100, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='100%')
    ax.set_ylim(0, 110)
    ax.set_title("Lifespan (% of max_steps)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("%")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # (1,1) Scout Death Causes (stacked area)
    ax = axes[1, 1]
    if scout_deaths:
        all_causes = set()
        for d in scout_deaths:
            all_causes.update(d.keys())
        all_causes = sorted(all_causes)
        cause_colors = {
            'boundary': '#e74c3c', 'ceiling': '#e67e22', 'ground_crash': '#8e44ad',
            'survived': '#2ecc71', 'unknown': '#95a5a6', 'env_empty': '#7f8c8d'
        }
        stacks = {c: [] for c in all_causes}
        for d in scout_deaths:
            total = max(1, sum(d.values()))
            for c in all_causes:
                stacks[c].append(d.get(c, 0) / total * 100)
        bottoms = np.zeros(len(scout_deaths))
        for c in all_causes:
            vals = np.array(stacks[c])
            color = cause_colors.get(c, '#bdc3c7')
            ax.bar(batches[:len(vals)], vals, bottom=bottoms[:len(vals)],
                   color=color, alpha=0.8, width=1.0, label=c)
            bottoms[:len(vals)] += vals
        ax.set_ylim(0, 105)
        ax.set_title("Scout Death Causes (%)")
        ax.set_xlabel("Batch")
        ax.set_ylabel("%")
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (1,2) Critic Value Loss
    ax = axes[1, 2]
    if critic_loss_s and len(critic_loss_s) > 0:
        ax.plot(range(1, len(critic_loss_s) + 1), critic_loss_s,
                color='green', linewidth=1, label='Scout')
    if critic_loss_c and len(critic_loss_c) > 0:
        ax.plot(range(1, len(critic_loss_c) + 1), critic_loss_c,
                color='tomato', linewidth=1, label='Commander')
    ax.set_title("Critic Value Loss (MSE)")
    ax.set_xlabel("Batch")
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
