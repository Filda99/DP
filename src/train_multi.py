"""
train_multi.py  --  Multi-Agent Hybrid Training: Scout + Commander
===================================================================

Architecture
------------
  Scout (ScoutActor)
    - Called EVERY physics step  ->  4-D action  +  5-D message
    - Input: local_map [1,32,32] + self_state [15] + neighbor_states [N,3]

  Commander (CommanderActor)
    - Called every `waypoint_steps` physics steps  ->  4-D waypoint [dx, dy, alt, water]
    - Input: self_state [19] + scout messages [N_quads, 5]
    - Between calls a PD heading controller flies toward the waypoint
    - When water < 30 % a scripted autopilot flies to the refill zone
      (these steps are NOT added to the PPO buffer)

  Separate PPO optimizers, separate privileged critics (CTDE).
  Scout buffer:  [eps x max_steps],           padded with alive mask
  Cmdr  buffer:  [eps x num_decisions_cmdr],  padded with alive mask

Usage
-----
  python train_multi.py
  python train_multi.py --resume-scout scout.pt --resume-cmdr cmdr.pt
  python train_multi.py --resume-scout scout.pt --reset-cmdr-critic --start-ep 0
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


# =====================================================================
#  Helpers
# =====================================================================


# =====================================================================
#  Rollout Worker  (runs on CPU, one per ProcessPoolExecutor slot)
# =====================================================================

def collect_multi_worker(num_eps, scout_w, cmdr_w, critic_scout_w, critic_cmdr_w,
                         config, batch_start_idx):
    """
    Run `num_eps` episodes and return packed experience buffers for both
    scout and commander, together with per-episode statistics.
    """
    # -----------------------------------------------------------------
    #  Imports & thread isolation
    # -----------------------------------------------------------------
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import cv2          # pre-import so env doesn't lazy-load it every reset
    import random

    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, PrivilegedCritic
    from commander_control import CommanderController

    # -----------------------------------------------------------------
    #  Unpack config
    # -----------------------------------------------------------------
    max_steps          = config['max_steps']
    waypoint_steps     = config['waypoint_steps']
    waypoint_range     = config['waypoint_range']
    num_decisions_cmdr = config['num_decisions_cmdr']
    hidden_dim_cmdr    = config['hidden_dim_cmdr']
    hidden_dim_scout   = config['hidden_dim_scout']
    N_QUADS            = config['N_QUADS']
    scout_msg_dim      = config['scout_msg_dim']
    map_size_range     = config.get('map_size_range', None)

    map_half           = config['grid_size_m'] / 2.0
    wp_reached_dist    = 30.0
    wp_timeout_penalty = -1.0

    ep_max_steps = config.get('ep_max_steps', max_steps)
    N_FIXED      = config.get('N_FIXED', 1)

    # Curriculum (read once, applied before each reset)
    curr_phase     = config.get('curriculum_phase', None)
    curr_map_range = config.get('curriculum_map_range', config.get('map_size_range'))
    curr_waste_pen = config.get('curriculum_waste_penalty', None)

    # -----------------------------------------------------------------
    #  Rebuild networks on CPU
    # -----------------------------------------------------------------
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

    scout_priv_dim = config['scout_self_dim'] + 6   # 16 + 6 = 22
    cmdr_priv_dim  = config['fixed_self_dim'] + 6   # 19 + 6 = 25

    local_critic_scout = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim_scout)
    local_critic_scout.load_state_dict(critic_scout_w)
    local_critic_scout.eval()

    local_critic_cmdr = PrivilegedCritic(cmdr_priv_dim, hidden_dim=hidden_dim_cmdr)
    local_critic_cmdr.load_state_dict(critic_cmdr_w)
    local_critic_cmdr.eval()

    # -----------------------------------------------------------------
    #  Environment
    # -----------------------------------------------------------------
    local_env = DroneFireEnv(
        num_quads=N_QUADS, num_fixed=N_FIXED,
        grid_size_m=config['grid_size_m'], max_steps=max_steps,
        n_fires_range=config.get('n_fires_range', (1, 1))
    )
    local_env.map_size_range = map_size_range

    # -----------------------------------------------------------------
    #  Experience buffers
    # -----------------------------------------------------------------
    scout_buf = {k: [] for k in [
        "maps", "self_states", "neighbor_states", "neighbor_masks",
        "actions", "logprobs", "returns", "values", "critic_states", "alive"
    ]}
    cmdr_buf = {k: [] for k in [
        "states", "messages", "msg_masks",
        "fw_neigh", "fw_neigh_masks",
        "actions", "logprobs", "returns", "alive", "values",
        "critic_states", "aux_targets", "expert_acts"
    ]}

    scout_h0_list       = []
    cmdr_h0_list        = []
    all_rewards         = []
    all_cmdr_rewards    = []
    all_refill_hits     = []
    all_scout_lifespans = []
    all_cmdr_lifespans  = []
    all_deaths          = []
    all_fire_stats      = []   # (peak_burning, final_burning, total_extinguish)
    all_cmdr_rd         = []   # per-episode commander reward diagnostics

    # Dummy tensors for dead-scout padding
    d_map       = torch.zeros(1, 1, 32, 32)
    d_scout_self = torch.zeros(1, config['scout_self_dim'])
    d_neigh_s   = torch.zeros(1, max(1, N_QUADS - 1), 3)
    d_neigh_m   = torch.ones(1, max(1, N_QUADS - 1), dtype=torch.bool)
    n_fw_neigh  = max(1, N_FIXED - 1)
    d_fw_neigh  = torch.zeros(1, n_fw_neigh, 3)
    d_fw_neigh_m = torch.ones(1, n_fw_neigh, dtype=torch.bool)

    worker_state = {}

    # =================================================================
    #  Episode loop
    # =================================================================
    for ep_off in range(num_eps):
        worker_state.clear()

        # -- Apply curriculum before reset ----------------------------
        local_env.curriculum_phase = curr_phase
        local_env.waste_penalty_override = curr_waste_pen
        if curr_map_range is not None:
            local_env.map_size_range = curr_map_range
        local_env.max_steps = ep_max_steps

        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_off)

        # -- Water-level initialisation --------------------------------
        #    FW spawns randomly on the map with a FULL tank.
        #    The NN must learn to navigate to fire and drop water.
        #    Scripted refill autopilot handles the refill cycle.
        for a in local_env.fixed_agents:
            if a in local_env.sim.drones:
                d = local_env.sim.drones[a]
                if d.water_capacity > 0:
                    d.current_water = d.water_capacity
                    local_env._prev_fw_water[a] = 1.0

        # Recalculate limits (map size may have changed via curriculum)
        map_half           = local_env.map_bounds

        quad_agents  = local_env.quad_agents
        fixed_agents = local_env.fixed_agents

        # -- Per-scout state ------------------------------------------
        scout_h              = {q: torch.zeros(1, 1, hidden_dim_scout) for q in quad_agents}
        critic_scout_h_dict  = {q: torch.zeros(1, 1, hidden_dim_scout) for q in quad_agents}
        scout_ep_data        = {q: [] for q in quad_agents}
        scout_alive          = {q: True for q in quad_agents}
        scout_lifespan       = {q: ep_max_steps for q in quad_agents}
        scout_death_cause    = {q: "survived" for q in quad_agents}
        ep_reward_scout      = {q: 0.0 for q in quad_agents}
        scout_msg_tensors    = {q: torch.zeros(1, scout_msg_dim) for q in quad_agents}
        scout_msg_valid      = {q: False for q in quad_agents}

        for q in quad_agents:
            scout_h0_list.append(scout_h[q].clone())

        # -- Commander state (per FW) ---------------------------------
        cmdr_h         = {f: torch.zeros(1, 1, hidden_dim_cmdr) for f in fixed_agents}
        critic_cmdr_h  = {f: torch.zeros(1, 1, hidden_dim_cmdr) for f in fixed_agents}
        for f in fixed_agents:
            cmdr_h0_list.append(cmdr_h[f].clone())

        cmdr_ep_data       = {f: [] for f in fixed_agents}
        ep_reward_cmdr     = {f: 0.0 for f in fixed_agents}
        cmdr_alive         = {f: True for f in fixed_agents}
        cmdr_death_cause   = {f: "survived" for f in fixed_agents}
        total_cmdr_steps   = {f: 0 for f in fixed_agents}
        cmdr_lifespan      = {f: ep_max_steps for f in fixed_agents}

        # Waypoint tracking (per FW)
        cmdr_ctrl          = {f: CommanderController(waypoint_range, waypoint_steps, wp_reached_dist)
                              for f in fixed_agents}
        for f in fixed_agents:
            cmdr_ctrl[f].reset(map_half)
        segment_reward       = {f: 0.0 for f in fixed_agents}
        scripted_segment     = {f: False for f in fixed_agents}
        scripted_refill_count = 0
        ep_peak_burning   = 0
        ep_total_extinguish = 0.0
        ep_cmdr_rd = {"r_extinguish": 0.0, "r_water_waste": 0.0,
                      "r_water_near": 0.0, "r_fire_out": 0.0, "r_spread": 0.0}
        ep_water_drops = []  # (alt, dist_to_fire, eff, water_left)

        msg_buffer = {f: [] for f in fixed_agents}

        # Trajectory logging (first episode per worker only)
        log_this_ep        = config.get('log_episodes', False) and ep_off == 0
        traj_scout_pos     = {q: [] for q in quad_agents} if log_this_ep else None
        traj_cmdr_pos      = {f: [] for f in fixed_agents} if log_this_ep else None
        traj_cmdr_waypoints = {f: [] for f in fixed_agents} if log_this_ep else None
        traj_rewards_scout = {q: [] for q in quad_agents} if log_this_ep else None
        traj_rewards_cmdr  = {f: [] for f in fixed_agents} if log_this_ep else None

        # =============================================================
        #  Step loop
        # =============================================================
        for step in range(max_steps):
            actions = {}

            # ---------------------------------------------------------
            #  1) SCOUT forward pass (every step, all alive scouts)
            # ---------------------------------------------------------
            for q in quad_agents:
                if scout_alive[q] and q in local_env.agents:
                    with torch.no_grad():
                        l_map = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0)
                        s_st  = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0)
                        n_s   = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0)
                        n_m   = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0)

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

            # Snapshot scout messages for commander (shared across all FW)
            scout_msg_snapshot = {
                q: (scout_msg_tensors[q].clone(), scout_msg_valid[q])
                for q in quad_agents
            }
            for f in fixed_agents:
                msg_buffer[f].append(scout_msg_snapshot)

            # ---------------------------------------------------------
            #  2) COMMANDER waypoint decision (per FW)
            # ---------------------------------------------------------
            for f_agent in fixed_agents:
              if cmdr_alive[f_agent] and f_agent in local_env.agents:

                fw_drone = local_env.sim.drones.get(f_agent)
                in_boundary_emergency = (
                    fw_drone is not None and
                    cmdr_ctrl[f_agent].check_boundary_emergency(fw_drone.get_position()))

                if in_boundary_emergency:
                    cmdr_ctrl[f_agent].need_new_waypoint = True

                if cmdr_ctrl[f_agent].need_new_waypoint:
                    # Build scout messages
                    msgs_for_cmdr = []
                    masks_for_cmdr = []
                    for q in quad_agents:
                        latest_msg, latest_valid = (
                            msg_buffer[f_agent][-1][q] if msg_buffer[f_agent]
                            else (torch.zeros(1, scout_msg_dim), False))
                        msgs_for_cmdr.append(latest_msg)
                        masks_for_cmdr.append(not latest_valid)
                    msgs_t = torch.stack(msgs_for_cmdr, dim=1)
                    msgs_m = torch.tensor([masks_for_cmdr])

                    # Build FW neighbor states (relative pos of other FW)
                    fw_neigh_list = []
                    fw_mask_list  = []
                    my_pos = fw_drone.get_position()
                    for other_f in fixed_agents:
                        if other_f == f_agent:
                            continue
                        if cmdr_alive[other_f] and other_f in local_env.sim.drones:
                            op = local_env.sim.drones[other_f].get_position()
                            fw_neigh_list.append([
                                (op[0] - my_pos[0]) / map_half,
                                (op[1] - my_pos[1]) / map_half,
                                (op[2] - my_pos[2]) / 100.0])
                            fw_mask_list.append(False)
                        else:
                            fw_neigh_list.append([0.0, 0.0, 0.0])
                            fw_mask_list.append(True)
                    if not fw_neigh_list:          # N_FIXED == 1
                        fw_neigh_list = [[0.0, 0.0, 0.0]]
                        fw_mask_list  = [True]
                    fw_neigh_t = torch.FloatTensor([fw_neigh_list])   # (1, N, 3)
                    fw_mask_t  = torch.BoolTensor([fw_mask_list])     # (1, N)

                    h_before = cmdr_h[f_agent].clone()
                    cmdr_h[f_agent], wp_info = cmdr_ctrl[f_agent].decide_waypoint(
                        fw_drone, obs[f_agent]["self_state"], local_env,
                        local_cmdr, cmdr_h[f_agent], msgs_t, msgs_m,
                        deterministic=False,
                        fw_neighbor_states=fw_neigh_t,
                        fw_neighbor_mask=fw_mask_t,
                        in_emergency=in_boundary_emergency)

                    if wp_info['scripted']:
                        segment_reward[f_agent]   = 0.0
                        scripted_segment[f_agent] = True
                        scripted_refill_count += 1
                        msg_buffer[f_agent]       = []
                    else:
                        # ── PPO buffer for NN decision ──
                        dist_c = wp_info['nn_dist']
                        act_c  = wp_info['nn_act']
                        s_st_f = wp_info['nn_state']

                        with torch.no_grad():
                            priv_c = torch.FloatTensor(
                                local_env.get_privileged_state(f_agent)).unsqueeze(0)
                            v_c, critic_cmdr_h[f_agent] = local_critic_cmdr(
                                priv_c, critic_cmdr_h[f_agent])

                        # Aux target: true direction to fire
                        c_pos = fw_drone.get_position()
                        vec_x = local_env.fire_x - c_pos[0]
                        vec_y = local_env.fire_y - c_pos[1]
                        dist_to_f = np.hypot(vec_x, vec_y)
                        true_dir = ([vec_x / dist_to_f, vec_y / dist_to_f]
                                    if dist_to_f > 1.0 else [0.0, 0.0])

                        # BC expert: fly to nearest live scout
                        c_pos_e    = fw_drone.get_position()
                        expert_act = np.zeros(4, dtype=np.float32)
                        live_sq = [q for q in quad_agents
                                   if scout_alive[q] and q in local_env.sim.drones]
                        if live_sq:
                            dists_sq = [
                                (np.hypot(c_pos_e[0] - local_env.sim.drones[q].get_position()[0],
                                          c_pos_e[1] - local_env.sim.drones[q].get_position()[1]), q)
                                for q in live_sq]
                            min_d, closest_q = min(dists_sq, key=lambda x: x[0])
                            sq_pos = local_env.sim.drones[closest_q].get_position()
                            expert_act[0] = np.clip((sq_pos[0] - c_pos_e[0]) / waypoint_range, -1.0, 1.0)
                            expert_act[1] = np.clip((sq_pos[1] - c_pos_e[1]) / waypoint_range, -1.0, 1.0)
                            expert_act[2] = -0.5   # target ~75m (was 0.0 → 110m)
                            expert_act[3] = 1.0 if min_d < 200.0 else -1.0
                        else:
                            expert_act[3] = -1.0

                        cmdr_ep_data[f_agent].append({
                            "alive": True,
                            "state": s_st_f,
                            "msgs": msgs_t,
                            "msg_mask": msgs_m,
                            "fw_neigh": fw_neigh_t,
                            "fw_neigh_mask": fw_mask_t,
                            "h": h_before,
                            "act": act_c,
                            "lp": dist_c.log_prob(act_c).sum(1),
                            "reward": 0.0,
                            "value": v_c.item(),
                            "critic_state": priv_c,
                            "aux_target": torch.FloatTensor([true_dir]),
                            "expert_act": torch.from_numpy(expert_act).unsqueeze(0),
                        })

                        segment_reward[f_agent]   = 0.0
                        scripted_segment[f_agent] = False
                        msg_buffer[f_agent]       = []

                    if log_this_ep:
                        traj_cmdr_waypoints[f_agent].append(
                            [step, cmdr_ctrl[f_agent].target_x, cmdr_ctrl[f_agent].target_y,
                             cmdr_ctrl[f_agent].target_alt_raw, cmdr_ctrl[f_agent].water_raw])

                # -- PD heading controller (every step) ---------------
                if fw_drone is not None:
                    # Update cached scout messages for valve logic
                    latest_msgs = []
                    latest_masks = []
                    for q in quad_agents:
                        latest_msg, latest_valid = (
                            msg_buffer[f_agent][-1][q] if msg_buffer[f_agent]
                            else (torch.zeros(1, scout_msg_dim), False))
                        latest_msgs.append(latest_msg)
                        latest_masks.append(not latest_valid)
                    cmdr_ctrl[f_agent].last_scout_msgs = torch.stack(latest_msgs, dim=1)
                    cmdr_ctrl[f_agent].last_scout_mask = torch.tensor([latest_masks])

                    actions[f_agent] = cmdr_ctrl[f_agent].heading_action(fw_drone, env=local_env)

                total_cmdr_steps[f_agent] += 1
              else:
                cmdr_alive[f_agent] = False

            # ---------------------------------------------------------
            #  3) Environment step
            # ---------------------------------------------------------
            r_cmdr_per_fw = {}
            if local_env.agents:
                obs, rewards, terms, truncs, infos = local_env.step(actions)

                # Scout rewards
                for q in quad_agents:
                    r_q = rewards.get(q, 0.0)
                    if scout_alive[q] and q in local_env.agents:
                        scout_ep_data[q][-1]["reward"] = r_q
                        ep_reward_scout[q] += r_q
                    if terms.get(q, False) or truncs.get(q, False):
                        scout_alive[q] = False
                        scout_lifespan[q] = step + 1
                        if terms.get(q, False):
                            scout_death_cause[q] = infos.get(q, {}).get("death_cause", "unknown")
                        else:
                            scout_death_cause[q] = "survived"

                # Commander segment reward accumulation (per FW)
                for f_agent in fixed_agents:
                    r_f = rewards.get(f_agent, 0.0)
                    r_cmdr_per_fw[f_agent] = r_f
                    segment_reward[f_agent] += r_f
                    # Accumulate reward diagnostics
                    fi = infos.get(f_agent, {})
                    for rk in ep_cmdr_rd:
                        ep_cmdr_rd[rk] += fi.get(rk, 0.0)
                    # Water-drop diagnostics
                    if "wd_alt" in fi:
                        ep_water_drops.append((fi["wd_alt"], fi["wd_dist"],
                                               fi["wd_eff"], fi["wd_water"]))

                    # Commander death handling
                    if terms.get(f_agent, False) or truncs.get(f_agent, False):
                        cmdr_alive[f_agent] = False
                        cmdr_lifespan[f_agent] = step + 1
                        if terms.get(f_agent, False):
                            cmdr_death_cause[f_agent] = infos.get(f_agent, {}).get("death_cause", "unknown")
                        else:
                            cmdr_death_cause[f_agent] = "survived"
                        if not scripted_segment[f_agent] and cmdr_ep_data[f_agent]:
                            cmdr_ep_data[f_agent][-1]["reward"] = segment_reward[f_agent]
                        ep_reward_cmdr[f_agent] += segment_reward[f_agent]
            else:
                for q in quad_agents:
                    if scout_alive[q]:
                        scout_alive[q] = False
                        scout_lifespan[q] = step + 1
                        scout_death_cause[q] = "env_empty"
                for f_agent in fixed_agents:
                    if cmdr_alive[f_agent]:
                        cmdr_alive[f_agent] = False
                        cmdr_lifespan[f_agent] = step + 1
                        cmdr_death_cause[f_agent] = "env_empty"

            # Early exit
            if not any(scout_alive.values()) and not any(cmdr_alive.values()):
                break

            # Trajectory recording
            if log_this_ep:
                for q in quad_agents:
                    s_drone = local_env.sim.drones.get(q)
                    traj_scout_pos[q].append(
                        s_drone.get_position().copy() if s_drone else np.full(3, np.nan))
                    traj_rewards_scout[q].append(
                        rewards.get(q, 0.0) if scout_alive[q] else 0.0)
                for f_agent in fixed_agents:
                    c_drone = local_env.sim.drones.get(f_agent)
                    traj_cmdr_pos[f_agent].append(
                        c_drone.get_position().copy() if c_drone else np.full(3, np.nan))
                    traj_rewards_cmdr[f_agent].append(
                        r_cmdr_per_fw.get(f_agent, 0.0) if cmdr_alive[f_agent] else 0.0)

            # Fire stats tracking
            fg = local_env.sim.environment.fire_grid
            if fg is not None:
                burning_now = int(np.sum(fg.B))
                ep_peak_burning = max(ep_peak_burning, burning_now)
            for dname, eff in local_env.sim.drone_extinguish_stats.items():
                ep_total_extinguish += eff

            # ---------------------------------------------------------
            #  4) Check waypoint segment end (per FW)
            # ---------------------------------------------------------
            for f_agent in fixed_agents:
              if cmdr_alive[f_agent] and f_agent in local_env.agents:
                segment_done = (
                    cmdr_ctrl[f_agent].check_segment_end()
                    or step == max_steps - 1
                )
                if segment_done:
                    if (not cmdr_ctrl[f_agent].wp_reached
                            and cmdr_ctrl[f_agent].steps_in_segment >= waypoint_steps):
                        segment_reward[f_agent] += wp_timeout_penalty
                    if not scripted_segment[f_agent] and cmdr_ep_data[f_agent]:
                        cmdr_ep_data[f_agent][-1]["reward"] = segment_reward[f_agent]
                    ep_reward_cmdr[f_agent] += segment_reward[f_agent] - r_cmdr_per_fw.get(f_agent, 0.0)
                    cmdr_ctrl[f_agent].need_new_waypoint = True

        # =============================================================
        #  Post-episode: fix final segment & compute GAE
        # =============================================================

        # Handle incomplete final segment (per FW)
        for f_agent in fixed_agents:
            if cmdr_ep_data[f_agent] and segment_reward[f_agent] != 0.0 and not scripted_segment[f_agent]:
                last = cmdr_ep_data[f_agent][-1]
                if last.get("reward", 0.0) == 0.0:
                    last["reward"] = segment_reward[f_agent]
            ep_reward_cmdr[f_agent] = sum(d["reward"] for d in cmdr_ep_data[f_agent])

        # -- Scout GAE (per scout, independently) ---------------------
        gamma   = config['gamma']
        gae_lam = config['gae_lambda']

        for q in quad_agents:
            ep_data = scout_ep_data[q]
            gae = 0.0
            for i in reversed(range(len(ep_data))):
                d = ep_data[i]
                v_next = ep_data[i + 1]["value"] if i + 1 < len(ep_data) else 0.0
                delta = d["reward"] + gamma * v_next - d["value"]
                gae = delta + gamma * gae_lam * gae
                d["ret"] = gae + d["value"]

            # Pack into buffer (padded to max_steps)
            for i in range(max_steps):
                if i < len(ep_data) and ep_data[i]["alive"]:
                    d = ep_data[i]
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
                else:
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

        # -- Commander GAE (per FW, then all packed together) ---------
        gamma_cmdr    = config.get('gamma_cmdr', gamma)
        n_msg_slots = N_QUADS
        for f_agent in fixed_agents:
            f_ep_data = cmdr_ep_data[f_agent]
            n_actual_cmdr = len(f_ep_data)
            gae = 0.0
            for i in reversed(range(n_actual_cmdr)):
                d = f_ep_data[i]
                v_next = f_ep_data[i + 1]["value"] if i + 1 < n_actual_cmdr else 0.0
                delta = d["reward"] + gamma_cmdr * v_next - d["value"]
                gae = delta + gamma_cmdr * gae_lam * gae
                d["ret"] = gae + d["value"]

            # Pack commander buffer (padded to num_decisions_cmdr)
            for i in range(num_decisions_cmdr):
                if i < n_actual_cmdr:
                    d = f_ep_data[i]
                    cmdr_buf["states"].append(d["state"])
                    cmdr_buf["messages"].append(d["msgs"])
                    cmdr_buf["msg_masks"].append(d["msg_mask"])
                    cmdr_buf["fw_neigh"].append(d["fw_neigh"])
                    cmdr_buf["fw_neigh_masks"].append(d["fw_neigh_mask"])
                    cmdr_buf["actions"].append(d["act"])
                    cmdr_buf["logprobs"].append(d["lp"])
                    cmdr_buf["returns"].append(d["ret"])
                    cmdr_buf["alive"].append(1.0)
                    cmdr_buf["values"].append(d["value"])
                    cmdr_buf["critic_states"].append(d["critic_state"])
                    cmdr_buf["aux_targets"].append(d["aux_target"])
                    cmdr_buf["expert_acts"].append(d.get("expert_act", torch.zeros(1, 4)))
                else:
                    cmdr_buf["states"].append(torch.zeros(1, config['fixed_self_dim']))
                    cmdr_buf["messages"].append(torch.zeros(1, n_msg_slots, scout_msg_dim))
                    cmdr_buf["msg_masks"].append(torch.ones(1, n_msg_slots, dtype=torch.bool))
                    cmdr_buf["fw_neigh"].append(d_fw_neigh)
                    cmdr_buf["fw_neigh_masks"].append(d_fw_neigh_m)
                    cmdr_buf["actions"].append(torch.zeros(1, 4))
                    cmdr_buf["logprobs"].append(torch.tensor([0.0]))
                    cmdr_buf["returns"].append(0.0)
                    cmdr_buf["alive"].append(0.0)
                    cmdr_buf["values"].append(0.0)
                    cmdr_buf["critic_states"].append(torch.zeros(1, cmdr_priv_dim))
                    cmdr_buf["aux_targets"].append(torch.zeros(1, 2))
                    cmdr_buf["expert_acts"].append(torch.zeros(1, 4))

        # -- Episode statistics ---------------------------------------
        total_ep_cmdr_reward = sum(ep_reward_cmdr.values())
        avg_scout_reward = sum(ep_reward_scout.values()) / max(1, N_QUADS)
        all_rewards.append(avg_scout_reward + total_ep_cmdr_reward / max(1, N_FIXED))
        all_cmdr_rewards.append(total_ep_cmdr_reward / max(1, N_FIXED))

        all_refill_hits.append(scripted_refill_count)

        fg_end = local_env.sim.environment.fire_grid
        ep_final_burning = int(np.sum(fg_end.B)) if fg_end is not None else 0
        all_fire_stats.append((ep_peak_burning, ep_final_burning, round(ep_total_extinguish, 1)))
        all_cmdr_rd.append(ep_cmdr_rd)
        all_cmdr_rd[-1]["_water_drops"] = ep_water_drops

        for q in quad_agents:
            all_scout_lifespans.append(scout_lifespan[q])
        for f in fixed_agents:
            all_cmdr_lifespans.append(cmdr_lifespan[f])

        death_parts = [f"s:{scout_death_cause[q]}" for q in quad_agents]
        for f in fixed_agents:
            death_parts.append(f"c:{cmdr_death_cause[f]}")
        all_deaths.append(",".join(death_parts))

        # -- Episode trajectory log -----------------------------------
        if log_this_ep:
            log_dir = config.get('log_dir', '/tmp/ep_logs')
            os.makedirs(log_dir, exist_ok=True)
            ep_id = batch_start_idx + ep_off
            f0 = fixed_agents[0] if fixed_agents else None
            save_dict = dict(
                ep_id=np.array(ep_id),
                scout_reward=np.array([ep_reward_scout[q] for q in quad_agents]),
                cmdr_reward=np.array(total_ep_cmdr_reward),
                scout_lifespan=np.array([scout_lifespan[q] for q in quad_agents]),
                cmdr_lifespan=np.array([cmdr_lifespan[f] for f in fixed_agents]),
                n_cmdr_decisions=np.array([len(cmdr_ep_data[f]) for f in fixed_agents]),
                cmdr_pos=np.array(traj_cmdr_pos[f0] if f0 else []),
                cmdr_waypoints=(np.array(traj_cmdr_waypoints[f0])
                                if f0 and traj_cmdr_waypoints[f0] else np.zeros((0, 5))),
                rewards_cmdr=np.array(traj_rewards_cmdr[f0] if f0 else []),
                map_bounds=np.array(local_env.grid_size_m / 2.0),
            )
            for qi, q in enumerate(quad_agents):
                save_dict[f'scout_{qi}_pos'] = np.array(traj_scout_pos[q])
                save_dict[f'scout_{qi}_rewards'] = np.array(traj_rewards_scout[q])
            np.savez_compressed(os.path.join(log_dir, f'ep_{ep_id:06d}.npz'), **save_dict)

    # =================================================================
    #  Cleanup & return
    # =================================================================
    local_env.sim.stop_simulation()

    def cat_buf(buf):
        result = {}
        for k, v in buf.items():
            if not v:
                result[k] = torch.tensor([], dtype=torch.float32)
            elif k in ("returns", "alive", "values"):
                result[k] = torch.tensor(v, dtype=torch.float32)
            else:
                result[k] = torch.cat(v)
        return result

    out_scout = cat_buf(scout_buf)
    out_cmdr  = cat_buf(cmdr_buf)
    out_init_h = {
        "scout": torch.cat(scout_h0_list, dim=0) if scout_h0_list else None,
        "cmdr":  torch.cat(cmdr_h0_list,  dim=0) if cmdr_h0_list  else None,
    }

    return (out_scout, out_cmdr, out_init_h,
            all_rewards, all_cmdr_rewards, all_refill_hits,
            (all_scout_lifespans, all_cmdr_lifespans), all_deaths,
            all_fire_stats, all_cmdr_rd)


# =====================================================================
#  Training Loop
# =====================================================================

def train_multi(resume_scout="", resume_cmdr="",
                log_episodes=False, log_dir="/tmp/ep_logs",
                episodes_played=0, reset_cmdr_critic=False):
    print("=" * 70)
    print("  Multi-Agent Training: Scout (frame-by-frame) + Commander (waypoint)")
    print("=" * 70)

    # Resolve paths before chdir
    if resume_scout:
        resume_scout = os.path.abspath(resume_scout)
    if resume_cmdr:
        resume_cmdr = os.path.abspath(resume_cmdr)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor, PrivilegedCritic
    from commander_control import CommanderController

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------
    #  Hyperparameters
    # -----------------------------------------------------------------
    N_QUADS            = 4
    N_FIXED            = 3
    grid_size_m        = 1000.0
    map_size_range     = (800, 1500)  # domain randomisation like train_scout
    n_fires_range      = (1, 3)      # 1-3 fires per episode
    num_episodes       = 30_000
    max_steps          = 1000        # longer episodes — FW needs time for refill cycles
    steps_range        = (1000,1500)        # fixed length
    bptt_chunk         = 128
    waypoint_steps     = 30
    waypoint_range     = 200.0       # metres
    num_decisions_cmdr = max_steps // waypoint_steps   # 26

    scout_freeze_batches = 999999     # scouts FROZEN — focus on commander training

    gamma              = 0.99
    gamma_cmdr         = 0.96        # short horizon for 16 decisions
    gae_lambda         = 0.95
    clip_coef          = 0.2
    update_epochs      = 4
    num_workers        = 15
    eps_per_worker     = 2
    episodes_per_batch = num_workers * eps_per_worker

    lr_scout           = 3e-5        # gentle finetuning rate
    lr_cmdr            = 5e-4
    lr_critic          = 5e-4
    hidden_dim_scout   = 128
    hidden_dim_cmdr    = 64
    scout_msg_dim      = 5

    entropy_scout      = 0.002
    entropy_cmdr       = 0.02
    critic_epochs_cmdr = 4           # same as update_epochs

    # -----------------------------------------------------------------
    #  Curriculum phases (commander) — firefighting-focused
    # -----------------------------------------------------------------
    #  With scripted refill autopilot the NN never controls during
    #  low-water flight, so the curriculum trains ONLY firefighting:
    #
    #  Phase 1  (batch 1-40):    easy firefighting — fire close, no
    #                             spread penalty, approach shaping ON
    #  Phase 2  (batch 41-100):  medium — fire further away, mild
    #                             spread penalty, approach shaping ON
    #  Phase 3  (batch 101-180): full mission on fixed map, full
    #                             waste penalty, spread penalty ON
    #  Phase 4+ (batch 181+):    full difficulty, domain randomisation
    # -----------------------------------------------------------------
    CURR_PHASE1_END = 0     # skip early phases — models are pre-trained
    CURR_PHASE2_END = 0
    CURR_PHASE3_END = 0

    # Behavioural cloning coefficient (decays, floor = 0.03)
    bc_coef = 0.5

    # -----------------------------------------------------------------
    #  Observation dimensions (from a temp env)
    # -----------------------------------------------------------------
    temp_env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                            grid_size_m=grid_size_m, max_steps=max_steps)
    scout_self_dim = temp_env.observation_space(temp_env.quad_agents[0])["self_state"].shape[0]
    fixed_self_dim = temp_env.observation_space(temp_env.fixed_agents[0])["self_state"].shape[0]
    if hasattr(temp_env, 'sim') and temp_env.sim is not None:
        temp_env.sim.stop_simulation()

    print(f"scout_self_dim   = {scout_self_dim}")
    print(f"fixed_self_dim   = {fixed_self_dim}")
    print(f"max_steps        = {max_steps}")
    print(f"waypoint_steps   = {waypoint_steps}")
    print(f"waypoint_range   = {waypoint_range}m")
    print(f"num_decisions    = {num_decisions_cmdr} (commander)\n")

    worker_config = {
        'N_QUADS': N_QUADS, 'N_FIXED': N_FIXED,
        'grid_size_m': grid_size_m,
        'max_steps': max_steps,
        'map_size_range': map_size_range,
        'n_fires_range': n_fires_range,
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

    # -----------------------------------------------------------------
    #  Networks
    # -----------------------------------------------------------------
    scout_actor = ScoutActor(
        self_state_dim=scout_self_dim, msg_dim=scout_msg_dim,
        hidden_dim=hidden_dim_scout).to(device)
    print(f"ScoutActor params:  {sum(p.numel() for p in scout_actor.parameters()):,}")

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
        self_state_dim=fixed_self_dim, msg_input_dim=scout_msg_dim,
        action_dim=4, hidden_dim=hidden_dim_cmdr).to(device)
    print(f"Commander params:   {sum(p.numel() for p in cmdr_actor.parameters()):,}")

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

        with torch.no_grad():
            cmdr_actor.action_logstd.fill_(-0.5)
            print("  Reset action_logstd for fresh exploration")
            # Re-init encoder weights for refill compass dims (11-12)
            # which were always zero during old training
            if hasattr(cmdr_actor, 'encoder') and hasattr(cmdr_actor.encoder, '0'):
                nn.init.normal_(cmdr_actor.encoder[0].weight[:, 11:13], std=0.01)
                print("  Re-init encoder weights for refill compass dims (11-12)")

    # -----------------------------------------------------------------
    #  Optimizers
    # -----------------------------------------------------------------
    optimizer_scout = optim.Adam(scout_actor.parameters(), lr=lr_scout)

    scout_frozen = scout_freeze_batches > 0
    if scout_frozen:
        for p in scout_actor.parameters():
            p.requires_grad = False
        print(f"  Scout FROZEN for first {scout_freeze_batches} batches")

    cmdr_main_params = [p for n, p in cmdr_actor.named_parameters() if n != 'action_logstd']
    optimizer_cmdr = optim.Adam([
        {"params": cmdr_main_params,            "lr": lr_cmdr},
        {"params": [cmdr_actor.action_logstd],  "lr": lr_cmdr},
    ])

    # -----------------------------------------------------------------
    #  Critics (privileged, CTDE)
    # -----------------------------------------------------------------
    scout_priv_dim = scout_self_dim + 6
    cmdr_priv_dim  = fixed_self_dim + 6

    critic_scout = PrivilegedCritic(scout_priv_dim, hidden_dim=hidden_dim_scout).to(device)
    critic_cmdr  = PrivilegedCritic(cmdr_priv_dim,  hidden_dim=hidden_dim_cmdr).to(device)
    print(f"ScoutCritic params: {sum(p.numel() for p in critic_scout.parameters()):,}")
    print(f"CmdrCritic params:  {sum(p.numel() for p in critic_cmdr.parameters()):,}")

    optimizer_critic_scout = optim.Adam(critic_scout.parameters(), lr=lr_critic)
    optimizer_critic_cmdr  = optim.Adam(critic_cmdr.parameters(),  lr=lr_critic)

    if reset_cmdr_critic:
        def _reinit(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        critic_cmdr.apply(_reinit)
        print("  CmdrCritic weights re-initialised (--reset-cmdr-critic)")

    # -----------------------------------------------------------------
    #  Tracking
    # -----------------------------------------------------------------
    reward_history           = []
    reward_per_batch         = []
    loss_history_scout       = []
    loss_history_cmdr        = []
    critic_loss_history_scout = []
    critic_loss_history_cmdr = []
    loss_components_scout    = []
    loss_components_cmdr     = []
    scout_life_pct_history   = []
    cmdr_life_pct_history    = []
    scout_death_history      = []
    fire_stats_history       = []   # (peak, final, extinguish, suppressed) per batch
    refill_history           = []   # avg refills per episode per batch
    cmdr_reward_history      = []
    scout_reward_history     = []

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_models", "finetune")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints -> {save_dir}\n")

    best_avg       = -1e9
    episodes_played = episodes_played
    num_batches    = num_episodes // episodes_per_batch

    # =================================================================
    #  Main training loop
    # =================================================================
    for batch_idx in range(1, num_batches + 1):

        # Unfreeze scouts if needed
        if scout_frozen and batch_idx > scout_freeze_batches:
            for p in scout_actor.parameters():
                p.requires_grad = True
            optimizer_scout = optim.Adam(scout_actor.parameters(), lr=lr_scout)
            scout_frozen = False
            print(f"   Scout UNFROZEN at batch {batch_idx}")

        scout_w = {k: v.cpu() for k, v in scout_actor.state_dict().items()}
        cmdr_w  = {k: v.cpu() for k, v in cmdr_actor.state_dict().items()}
        cs_w    = {k: v.cpu() for k, v in critic_scout.state_dict().items()}
        cc_w    = {k: v.cpu() for k, v in critic_cmdr.state_dict().items()}

        t0 = time.time()

        # Aggregate buffers
        agg_scout = {k: [] for k in [
            "maps", "self_states", "neighbor_states", "neighbor_masks",
            "actions", "logprobs", "returns", "values", "critic_states", "alive"]}
        agg_cmdr = {k: [] for k in [
            "states", "messages", "msg_masks",
            "fw_neigh", "fw_neigh_masks",
            "actions", "logprobs", "returns", "alive", "values",
            "critic_states", "aux_targets", "expert_acts"]}
        agg_h = {"scout": [], "cmdr": []}
        batch_rewards         = []
        batch_cmdr_rewards    = []
        batch_refill_hits     = []
        batch_deaths          = []
        batch_scout_lifespans = []
        batch_cmdr_lifespans  = []
        batch_fire_stats      = []
        batch_cmdr_rd         = []

        batch_ep_max = max_steps
        worker_config['ep_max_steps'] = batch_ep_max

        # -- Curriculum phase (firefighting-focused) ------------------
        if batch_idx <= CURR_PHASE1_END:
            curr_phase     = 1          # easy firefighting
            curr_map_range = (1200.0, 1200.0)
            curr_waste_pen = 0.0        # no waste penalty yet
        elif batch_idx <= CURR_PHASE2_END:
            curr_phase     = 2          # medium difficulty
            curr_map_range = (1200.0, 1200.0)
            curr_waste_pen = 0.0        # still no waste penalty
        elif batch_idx <= CURR_PHASE3_END:
            curr_phase     = 3          # full mission, fixed map
            curr_map_range = (1200.0, 1200.0)
            curr_waste_pen = 0.3        # mild waste penalty
        else:
            curr_phase     = None       # full difficulty
            curr_map_range = map_size_range
            curr_waste_pen = None       # use reward_config default

        if worker_config.get('_prev_phase') != curr_phase:
            print(f"   [CURRICULUM] Batch {batch_idx}: phase {curr_phase} "
                  f"(map={curr_map_range}, waste_pen={curr_waste_pen})")
            worker_config['_prev_phase'] = curr_phase
        worker_config['curriculum_phase']        = curr_phase
        worker_config['curriculum_map_range']     = curr_map_range
        worker_config['curriculum_waste_penalty'] = curr_waste_pen

        # -- Parallel rollout -----------------------------------------
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    collect_multi_worker,
                    eps_per_worker, scout_w, cmdr_w, cs_w, cc_w,
                    worker_config,
                    episodes_played + i * eps_per_worker)
                for i in range(num_workers)]

            failed = 0
            for fut in futures:
                try:
                    (w_scout, w_cmdr, w_h, w_rew, w_cmdr_rew,
                     w_refill, (w_scout_life, w_cmdr_life), w_deaths,
                     w_fire_stats, w_cmdr_rd) = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"   Worker failed: {e}")
                    continue

                batch_rewards.extend(w_rew)
                batch_cmdr_rewards.extend(w_cmdr_rew)
                batch_deaths.extend(w_deaths)
                batch_scout_lifespans.extend(w_scout_life)
                batch_cmdr_lifespans.extend(w_cmdr_life)
                batch_refill_hits.extend(w_refill)
                batch_fire_stats.extend(w_fire_stats)
                batch_cmdr_rd.extend(w_cmdr_rd)
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
            print(f"   {failed}/{num_workers} workers failed this batch")
        if not batch_rewards:
            print(f"   All workers failed, skipping batch {batch_idx}")
            continue

        # -- Logging --------------------------------------------------
        avg_batch   = float(np.mean(batch_rewards))
        avg_cmdr_r  = float(np.mean(batch_cmdr_rewards)) if batch_cmdr_rewards else 0.0
        avg_scout_r = avg_batch - avg_cmdr_r   # scout-only average
        refill_rate = int(sum(batch_refill_hits))
        reward_per_batch.append(avg_batch)
        win      = min(60, len(reward_history))
        avg_roll = float(np.mean(reward_history[-win:]))

        scout_life_pct = float(np.mean(batch_scout_lifespans)) / batch_ep_max * 100
        cmdr_life_pct  = float(np.mean(batch_cmdr_lifespans))  / batch_ep_max * 100
        scout_life_pct_history.append(scout_life_pct)
        cmdr_life_pct_history.append(cmdr_life_pct)

        from collections import Counter
        scout_deaths_c = Counter()
        cmdr_deaths_c  = Counter()
        for d in batch_deaths:
            for p in d.split(","):
                role, cause = p.split(":", 1)
                (scout_deaths_c if role == "s" else cmdr_deaths_c)[cause] += 1
        s_str = " ".join(f"{k}={v}" for k, v in scout_deaths_c.most_common())
        c_str = " ".join(f"{k}={v}" for k, v in cmdr_deaths_c.most_common())
        scout_death_history.append(dict(scout_deaths_c))

        nn_decisions = int(sum(float(a) for al in agg_cmdr["alive"] for a in al))
        total_slots  = sum(len(al) for al in agg_cmdr["alive"])

        # Fire stats aggregation
        if batch_fire_stats:
            avg_peak   = float(np.mean([s[0] for s in batch_fire_stats]))
            avg_final  = float(np.mean([s[1] for s in batch_fire_stats]))
            avg_ext    = float(np.mean([s[2] for s in batch_fire_stats]))
            suppressed = avg_peak - avg_final
        else:
            avg_peak = avg_final = avg_ext = suppressed = 0.0
        fire_stats_history.append((avg_peak, avg_final, avg_ext, suppressed))
        refill_history.append(refill_rate / max(1, len(batch_refill_hits)))
        cmdr_reward_history.append(avg_cmdr_r)
        scout_reward_history.append(avg_scout_r)

        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"Batch {batch_idx:04d} (Ep {episodes_played:05d}) | "
              f"R: {avg_batch:+8.1f} ({avg_roll:+8.1f})  "
              f"Scout_R: {avg_scout_r:+7.1f}  Cmdr_R: {avg_cmdr_r:+7.1f}  "
              f"Scout:{scout_life_pct:.0f}% Cmdr:{cmdr_life_pct:.0f}%  "
              f"scout:[{s_str}] cmdr:[{c_str}] refills={refill_rate}  "
              f"NN_dec={nn_decisions}/{total_slots}  "
              f"fire: peak={avg_peak:.0f} final={avg_final:.0f} supp={suppressed:.0f} ext={avg_ext:.1f}  "
              f"{rollout_time:.1f}s")

        # Commander reward diagnostics
        if batch_cmdr_rd:
            n_rd = len(batch_cmdr_rd)
            rd_avg = {k: sum(d[k] for d in batch_cmdr_rd) / n_rd
                      for k in batch_cmdr_rd[0] if not k.startswith("_")}
            rd_str = "  ".join(f"{k}={v:+.3f}" for k, v in rd_avg.items())
            print(f"   Cmdr reward breakdown: {rd_str}")

            # Water-drop diagnostics
            all_wd = []
            for d in batch_cmdr_rd:
                all_wd.extend(d.get("_water_drops", []))
            if all_wd:
                alts   = [w[0] for w in all_wd]
                dists  = [w[1] for w in all_wd]
                effs   = [w[2] for w in all_wd]
                n_hit  = sum(1 for e in effs if e > 0)
                n_miss = len(effs) - n_hit
                import statistics
                print(f"   Water drops: {len(all_wd)} total, {n_hit} hit, {n_miss} miss "
                      f"({100*n_hit/len(all_wd):.0f}% accuracy)  "
                      f"alt={statistics.mean(alts):.0f}m [{min(alts):.0f}-{max(alts):.0f}]  "
                      f"dist={statistics.mean(dists):.0f}m [{min(dists):.0f}-{max(dists):.0f}]")

        # -- Checkpoint: best & periodic ------------------------------
        if episodes_played >= 60 and avg_roll > best_avg:
            best_avg = avg_roll
            for name, net in [("scout", scout_actor), ("cmdr", cmdr_actor),
                              ("critic_scout", critic_scout), ("critic_cmdr", critic_cmdr)]:
                torch.save(net.state_dict(), os.path.join(save_dir, f"{name}_best.pt"))
            print(f"   New best! rolling avg = {best_avg:.1f}")

        if batch_idx % 10 == 0:
            for name, net in [("scout", scout_actor), ("cmdr", cmdr_actor),
                              ("critic_scout", critic_scout), ("critic_cmdr", critic_cmdr)]:
                torch.save(net.state_dict(), os.path.join(save_dir, f"{name}_b{batch_idx:04d}.pt"))

        # =============================================================
        #  PPO Update — Scout
        # =============================================================
        s_maps     = torch.cat(agg_scout["maps"]).to(device)
        s_self     = torch.cat(agg_scout["self_states"]).to(device)
        s_neigh_s  = torch.cat(agg_scout["neighbor_states"]).to(device)
        s_neigh_m  = torch.cat(agg_scout["neighbor_masks"]).to(device)
        s_actions  = torch.cat(agg_scout["actions"]).to(device)
        s_logprobs = torch.cat(agg_scout["logprobs"]).to(device)
        s_returns  = torch.cat(agg_scout["returns"]).to(device)
        s_values   = torch.cat(agg_scout["values"]).to(device)
        s_cstates  = torch.cat(agg_scout["critic_states"]).to(device)
        s_alive    = torch.cat(agg_scout["alive"]).to(device)

        h_scout = torch.cat(agg_h["scout"], dim=0).squeeze(1).unsqueeze(0).to(device)

        # Advantages (normalised over alive steps)
        s_adv = s_returns - s_values
        alive_bool_s = s_alive > 0.5
        if alive_bool_s.sum() > 1:
            alive_adv = s_adv[alive_bool_s]
            s_adv = (s_adv - alive_adv.mean()) / (alive_adv.std() + 1e-8)

        # Normalised returns for critic target
        alive_rets_s = s_returns[alive_bool_s] if alive_bool_s.sum() > 1 else s_returns
        s_returns_norm = (s_returns - alive_rets_s.mean()) / (alive_rets_s.std() + 1e-8)

        # Reshape to [episodes, max_steps, ...]
        eps = s_returns.numel() // max_steps
        s_maps_seq         = s_maps.view(eps, max_steps, 1, 32, 32)
        s_self_seq         = s_self.view(eps, max_steps, -1)
        s_neigh_s_seq      = s_neigh_s.view(eps, max_steps, s_neigh_s.size(-2), 3)
        s_neigh_m_seq      = s_neigh_m.view(eps, max_steps, -1)
        s_actions_seq      = s_actions.view(eps, max_steps, -1)
        s_logprobs_seq     = s_logprobs.view(eps, max_steps)
        s_adv_seq          = s_adv.view(eps, max_steps)
        s_returns_norm_seq = s_returns_norm.view(eps, max_steps)
        s_alive_seq        = s_alive.view(eps, max_steps)
        s_cstates_seq      = s_cstates.view(eps, max_steps, -1)
        h_scout_seq        = h_scout.transpose(0, 1)

        num_minibatches = 4
        mb_size_s = max(1, eps // num_minibatches)
        scout_loss_total        = 0.0
        scout_critic_loss_total = 0.0
        scout_policy_loss_total = 0.0
        scout_entropy_total     = 0.0

        for epoch in range(update_epochs):
            b_inds = np.random.permutation(eps)
            for start in range(0, eps, mb_size_s):
                mb = b_inds[start:start + mb_size_s]

                mb_maps    = s_maps_seq[mb]
                mb_self    = s_self_seq[mb]
                mb_ns      = s_neigh_s_seq[mb]
                mb_nm      = s_neigh_m_seq[mb]
                mb_acts    = s_actions_seq[mb]
                mb_old_lp  = s_logprobs_seq[mb]
                mb_adv     = s_adv_seq[mb]
                mb_alive_s = s_alive_seq[mb]
                mb_rets    = s_returns_norm_seq[mb]
                mb_cs      = s_cstates_seq[mb]
                mb_h       = h_scout_seq[mb].transpose(0, 1)

                # Chunked BPTT (gradient cut every bptt_chunk steps)
                chunk_new_lps = []
                chunk_entropies = []
                h_chunk = mb_h.detach()
                T = mb_maps.size(1)
                curr_mb = mb_maps.size(0)

                for t0 in range(0, T, bptt_chunk):
                    t1 = min(t0 + bptt_chunk, T)
                    chunk_len = t1 - t0
                    c_acts = mb_acts[:, t0:t1].reshape(-1, 4)
                    dist_c, _, h_chunk = scout_actor(
                        mb_maps[:, t0:t1], mb_self[:, t0:t1],
                        mb_ns[:, t0:t1], mb_nm[:, t0:t1], h_chunk)
                    chunk_new_lps.append(dist_c.log_prob(c_acts).sum(1).view(curr_mb, chunk_len))
                    chunk_entropies.append(dist_c.entropy().sum(1).view(curr_mb, chunk_len))
                    h_chunk = h_chunk.detach()

                new_lp      = torch.cat(chunk_new_lps, dim=1).reshape(-1)
                entropy     = torch.cat(chunk_entropies, dim=1).reshape(-1)
                flat_old_lp = mb_old_lp.reshape(-1)
                flat_adv    = mb_adv.reshape(-1)
                flat_alive  = mb_alive_s.reshape(-1)
                flat_rets   = mb_rets.reshape(-1)

                log_ratio = (new_lp - flat_old_lp).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)

                # Diagnostic on last epoch, first minibatch (shows post-update ratio)
                if epoch == update_epochs - 1 and start == 0:
                    with torch.no_grad():
                        alive_mask_d = flat_alive > 0.5
                        if alive_mask_d.sum() > 0:
                            r_alive = ratio[alive_mask_d]
                            a_alive = flat_adv[alive_mask_d]
                            print(f"   [SCOUT DIAG] ratio: {r_alive.mean():.4f}+/-{r_alive.std():.4f} "
                                  f"[{r_alive.min():.4f},{r_alive.max():.4f}] | "
                                  f"adv: {a_alive.mean():.4f}+/-{a_alive.std():.4f} | "
                                  f"alive={alive_mask_d.sum().item()}/{flat_alive.numel()}")

                # Clipped surrogate
                pg1 = -flat_adv * ratio
                pg2 = -flat_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                surr = torch.max(pg1, pg2)

                n_alive_s = flat_alive.sum().clamp(min=1.0)
                loss_s = ((surr * flat_alive).sum() / n_alive_s
                          - entropy_scout * (entropy * flat_alive).sum() / n_alive_s)

                if torch.isfinite(loss_s):
                    if loss_s.requires_grad and not scout_frozen:
                        optimizer_scout.zero_grad()
                        loss_s.backward()
                        nn.utils.clip_grad_norm_(scout_actor.parameters(), max_norm=0.5)
                        optimizer_scout.step()
                    scout_loss_total += loss_s.item()
                    scout_policy_loss_total += ((surr * flat_alive).sum() / n_alive_s).item()
                    scout_entropy_total += ((entropy * flat_alive).sum() / n_alive_s).item()

                # Critic update
                v_pred, _ = critic_scout(mb_cs, None)
                v_loss = ((v_pred - flat_rets) ** 2 * flat_alive).sum() / n_alive_s
                if torch.isfinite(v_loss):
                    optimizer_critic_scout.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic_scout.parameters(), max_norm=0.5)
                    optimizer_critic_scout.step()
                    scout_critic_loss_total += v_loss.item()

        n_updates_s = max(1, update_epochs * num_minibatches)
        loss_history_scout.append(scout_loss_total / n_updates_s)
        critic_loss_history_scout.append(scout_critic_loss_total / n_updates_s)
        loss_components_scout.append({
            "policy":  scout_policy_loss_total / n_updates_s,
            "entropy": scout_entropy_total     / n_updates_s,
            "value":   scout_critic_loss_total / n_updates_s,
        })

        # =============================================================
        #  PPO Update — Commander
        # =============================================================
        c_states      = torch.cat(agg_cmdr["states"]).to(device)
        c_msgs        = torch.cat(agg_cmdr["messages"]).to(device)
        c_msg_m       = torch.cat(agg_cmdr["msg_masks"]).to(device)
        c_fw_neigh    = torch.cat(agg_cmdr["fw_neigh"]).to(device)
        c_fw_neigh_m  = torch.cat(agg_cmdr["fw_neigh_masks"]).to(device)
        c_actions     = torch.cat(agg_cmdr["actions"]).to(device)
        c_logprobs    = torch.cat(agg_cmdr["logprobs"]).to(device)
        c_returns     = torch.cat(agg_cmdr["returns"]).to(device)
        c_alive       = torch.cat(agg_cmdr["alive"]).to(device)
        c_values      = torch.cat(agg_cmdr["values"]).to(device)
        c_cstates     = torch.cat(agg_cmdr["critic_states"]).to(device)
        c_aux_targets = torch.cat(agg_cmdr["aux_targets"]).to(device)
        c_expert_acts = torch.cat(agg_cmdr["expert_acts"]).to(device)

        h_cmdr = torch.cat(agg_h["cmdr"], dim=0).squeeze(1).unsqueeze(0).to(device)

        # Advantages (normalised over alive)
        c_adv = c_returns - c_values
        alive_bool = c_alive > 0.5
        if alive_bool.sum() > 1:
            alive_vals = c_adv[alive_bool]
            c_adv = (c_adv - alive_vals.mean()) / (alive_vals.std() + 1e-8)

        # Normalised returns for critic
        alive_rets = c_returns[alive_bool] if alive_bool.sum() > 1 else c_returns
        c_returns_norm = (c_returns - alive_rets.mean()) / (alive_rets.std() + 1e-8)

        # Reshape to [episodes, num_decisions_cmdr, ...]
        nd       = num_decisions_cmdr
        eps_cmdr = c_returns.numel() // nd

        c_states_seq       = c_states.view(eps_cmdr, nd, -1)
        c_msgs_seq         = c_msgs.view(eps_cmdr, nd, c_msgs.size(-2), c_msgs.size(-1))
        c_msg_m_seq        = c_msg_m.view(eps_cmdr, nd, c_msg_m.size(-1))
        c_fw_neigh_seq     = c_fw_neigh.view(eps_cmdr, nd, c_fw_neigh.size(-2), c_fw_neigh.size(-1))
        c_fw_neigh_m_seq   = c_fw_neigh_m.view(eps_cmdr, nd, c_fw_neigh_m.size(-1))
        c_actions_seq      = c_actions.view(eps_cmdr, nd, -1)
        c_logprobs_seq     = c_logprobs.view(eps_cmdr, nd)
        c_adv_seq          = c_adv.view(eps_cmdr, nd)
        c_alive_seq        = c_alive.view(eps_cmdr, nd)
        c_returns_norm_seq = c_returns_norm.view(eps_cmdr, nd)
        c_cstates_seq      = c_cstates.view(eps_cmdr, nd, -1)
        h_cmdr_seq         = h_cmdr.transpose(0, 1)
        c_aux_targets_seq  = c_aux_targets.view(eps_cmdr, nd, -1)
        c_expert_acts_seq  = c_expert_acts.view(eps_cmdr, nd, -1)

        mb_size_c = max(1, eps_cmdr // num_minibatches)

        cmdr_loss_total        = 0.0
        cmdr_critic_loss_total = 0.0
        cmdr_policy_loss_total = 0.0
        cmdr_entropy_total     = 0.0
        cmdr_aux_loss_total    = 0.0

        for epoch in range(critic_epochs_cmdr):
            b_inds = np.random.permutation(eps_cmdr)
            for start in range(0, eps_cmdr, mb_size_c):
                mb = b_inds[start:start + mb_size_c]

                mb_states      = c_states_seq[mb]
                mb_msgs        = c_msgs_seq[mb]
                mb_mm          = c_msg_m_seq[mb]
                mb_fw_n        = c_fw_neigh_seq[mb]
                mb_fw_nm       = c_fw_neigh_m_seq[mb]
                mb_acts        = c_actions_seq[mb]
                mb_old_lp      = c_logprobs_seq[mb].view(-1)
                mb_adv         = c_adv_seq[mb].reshape(-1)
                mb_alive       = c_alive_seq[mb].reshape(-1)
                mb_rets        = c_returns_norm_seq[mb].reshape(-1)
                mb_cs          = c_cstates_seq[mb]
                mb_h           = h_cmdr_seq[mb].transpose(0, 1)
                mb_aux_targets = c_aux_targets_seq[mb].reshape(-1, 2)
                mb_expert      = c_expert_acts_seq[mb].reshape(-1, 4)

                # Actor update only for first update_epochs epochs
                # (critic runs all critic_epochs_cmdr)
                if epoch < update_epochs:
                    dist, mb_aux_pred, _ = cmdr_actor(
                        mb_states, mb_msgs, mb_mm, mb_h,
                        mb_fw_n, mb_fw_nm)
                    flat_acts = mb_acts.view(-1, 4)
                    new_lp  = dist.log_prob(flat_acts).sum(1)
                    entropy = dist.entropy().sum(1)

                    log_ratio = (new_lp - mb_old_lp).clamp(-10.0, 10.0)
                    ratio = torch.exp(log_ratio)

                    # BC coefficient (computed early so diagnostic can print it)
                    bc_coef_now = max(0.0, bc_coef * (1.0 - batch_idx / max(1, CURR_PHASE3_END)))

                    # Diagnostic on last epoch, first minibatch
                    if epoch == update_epochs - 1 and start == 0:
                        with torch.no_grad():
                            alive_d = mb_alive > 0.5
                            if alive_d.sum() > 0:
                                r_a = ratio[alive_d]
                                a_a = mb_adv[alive_d]
                                print(f"   [CMDR  DIAG] ratio: {r_a.mean():.4f}+/-{r_a.std():.4f} "
                                      f"[{r_a.min():.4f},{r_a.max():.4f}] | "
                                      f"adv: {a_a.mean():.4f}+/-{a_a.std():.4f} | "
                                      f"alive={alive_d.sum().item()}/{mb_alive.numel()} | "
                                      f"bc_coef={bc_coef_now:.3f}")

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    surr = torch.max(pg1, pg2)

                    n_alive      = mb_alive.sum().clamp(min=1.0)
                    policy_loss  = (surr * mb_alive).sum() / n_alive
                    entropy_loss = (entropy * mb_alive).sum() / n_alive

                    # Auxiliary loss: predict fire direction (GRU regulariser)
                    aux_loss = ((mb_aux_pred - mb_aux_targets)**2).sum(dim=-1)
                    aux_loss = (aux_loss * mb_alive).sum() / n_alive

                    # BC loss: MSE between policy mean and expert action
                    # Decays linearly, floor = 0.03
                    bc_loss = ((dist.mean - mb_expert)**2).sum(dim=-1)
                    bc_loss = (bc_loss * mb_alive).sum() / n_alive

                    loss_c = (policy_loss
                              - entropy_cmdr * entropy_loss
                              + 0.5 * aux_loss
                              + bc_coef_now * bc_loss)

                    if torch.isfinite(loss_c):
                        optimizer_cmdr.zero_grad()
                        loss_c.backward()
                        nn.utils.clip_grad_norm_(cmdr_actor.parameters(), max_norm=0.5)
                        optimizer_cmdr.step()
                        cmdr_loss_total        += loss_c.item()
                        cmdr_policy_loss_total += policy_loss.item()
                        cmdr_entropy_total     += entropy_loss.item()
                        cmdr_aux_loss_total    += aux_loss.item()

                # Critic update (every epoch)
                v_pred, _ = critic_cmdr(mb_cs, None)
                n_alive_c = mb_alive.sum().clamp(min=1.0)
                v_loss_c  = ((v_pred - mb_rets)**2 * mb_alive).sum() / n_alive_c
                if torch.isfinite(v_loss_c):
                    optimizer_critic_cmdr.zero_grad()
                    v_loss_c.backward()
                    nn.utils.clip_grad_norm_(critic_cmdr.parameters(), max_norm=0.5)
                    optimizer_critic_cmdr.step()
                    cmdr_critic_loss_total += v_loss_c.item()

        n_updates_c = max(1, update_epochs * num_minibatches)
        loss_history_cmdr.append(cmdr_loss_total / n_updates_c)
        critic_loss_history_cmdr.append(cmdr_critic_loss_total / n_updates_c)
        loss_components_cmdr.append({
            "policy":  cmdr_policy_loss_total / n_updates_c,
            "entropy": cmdr_entropy_total     / n_updates_c,
            "aux":     cmdr_aux_loss_total    / n_updates_c,
            "value":   cmdr_critic_loss_total / n_updates_c,
        })

        # =============================================================
        #  Periodic save & plot
        # =============================================================
        if batch_idx % 10 == 0:
            _save_plot_multi(
                reward_per_batch, loss_history_scout, loss_history_cmdr,
                scout_life_pct_history, cmdr_life_pct_history,
                scout_death_history, save_dir, batch_idx,
                critic_loss_history_scout, critic_loss_history_cmdr,
                loss_components_scout, loss_components_cmdr,
                fire_stats_history, refill_history,
                cmdr_reward_history, scout_reward_history)

    # Final save
    _save_plot_multi(
        reward_per_batch, loss_history_scout, loss_history_cmdr,
        scout_life_pct_history, cmdr_life_pct_history,
        scout_death_history, save_dir, batch_idx,
        critic_loss_history_scout, critic_loss_history_cmdr,
        loss_components_scout, loss_components_cmdr,
        fire_stats_history, refill_history,
        cmdr_reward_history, scout_reward_history)

    print(f"\nTraining complete!")
    print(f"   Best: {save_dir}/scout_best.pt + cmdr_best.pt")


# =====================================================================
#  Plot Helper
# =====================================================================

def _save_plot_multi(reward_batches, loss_s, loss_c,
                     scout_life_pct, cmdr_life_pct, scout_deaths,
                     save_dir, batch_idx,
                     critic_loss_s=None, critic_loss_c=None,
                     loss_comp_s=None, loss_comp_c=None,
                     fire_stats=None, refill_hist=None,
                     cmdr_reward_hist=None, scout_reward_hist=None):
    """Generate training progress plots.  All data is per-batch."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"Multi-Agent — Batch {batch_idx}", fontsize=13)
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
    ax.set_title("Avg Reward per Batch"); ax.set_xlabel("Batch"); ax.grid(True, alpha=0.3)

    # (0,1) Scout PPO loss
    ax = axes[0, 1]
    if loss_s:
        ax.plot(range(1, len(loss_s) + 1), loss_s, color='green', linewidth=1)
    ax.set_title("Scout PPO Loss"); ax.set_xlabel("Batch"); ax.grid(True, alpha=0.3)

    # (0,2) Commander PPO loss
    ax = axes[0, 2]
    if loss_c:
        ax.plot(range(1, len(loss_c) + 1), loss_c, color='tomato', linewidth=1)
    ax.set_title("Commander PPO Loss"); ax.set_xlabel("Batch"); ax.grid(True, alpha=0.3)

    # (1,0) Lifespan %
    ax = axes[1, 0]
    ax.plot(batches, scout_life_pct, color='green', linewidth=1, alpha=0.5)
    ax.plot(batches, cmdr_life_pct, color='tomato', linewidth=1, alpha=0.5)
    mx_s, ma_s = _ma(scout_life_pct)
    mx_c, ma_c = _ma(cmdr_life_pct)
    if ma_s is not None:
        ax.plot(mx_s, ma_s, color='green', linewidth=2, label='Scout MA10')
    if ma_c is not None:
        ax.plot(mx_c, ma_c, color='tomato', linewidth=2, label='Cmdr MA10')
    ax.axhline(100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 110)
    ax.set_title("Lifespan (% of max_steps)"); ax.set_xlabel("Batch"); ax.set_ylabel("%")
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

    # (1,1) Scout death causes
    ax = axes[1, 1]
    if scout_deaths:
        all_causes = sorted({c for d in scout_deaths for c in d})
        cause_colors = {
            'boundary': '#e74c3c', 'ceiling': '#e67e22', 'ground_crash': '#8e44ad',
            'survived': '#2ecc71', 'unknown': '#95a5a6', 'env_empty': '#7f8c8d'}
        bottoms = np.zeros(len(scout_deaths))
        for c in all_causes:
            vals = np.array([d.get(c, 0) / max(1, sum(d.values())) * 100 for d in scout_deaths])
            ax.bar(batches[:len(vals)], vals, bottom=bottoms[:len(vals)],
                   color=cause_colors.get(c, '#bdc3c7'), alpha=0.8, width=1.0, label=c)
            bottoms[:len(vals)] += vals
        ax.set_ylim(0, 105)
        ax.set_title("Scout Death Causes (%)"); ax.set_xlabel("Batch"); ax.set_ylabel("%")
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (1,2) Critic losses
    ax = axes[1, 2]
    if critic_loss_s:
        ax.plot(range(1, len(critic_loss_s) + 1), critic_loss_s, color='green', linewidth=1, label='Scout')
    if critic_loss_c:
        ax.plot(range(1, len(critic_loss_c) + 1), critic_loss_c, color='tomato', linewidth=1, label='Commander')
    ax.set_title("Critic Value Loss (MSE)"); ax.set_xlabel("Batch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Row 3: Operational metrics ──────────────────────────────────

    # (2,0) Scout R vs Cmdr R breakdown
    ax = axes[2, 0]
    if scout_reward_hist and cmdr_reward_hist:
        bx = np.arange(1, len(scout_reward_hist) + 1)
        ax.bar(bx, scout_reward_hist, color='green', alpha=0.4, width=1.0, label='Scout R')
        ax.bar(bx, cmdr_reward_hist, color='tomato', alpha=0.4, width=1.0, bottom=0, label='Cmdr R')
        mx_s2, ma_s2 = _ma(scout_reward_hist)
        mx_c2, ma_c2 = _ma(cmdr_reward_hist)
        if ma_s2 is not None:
            ax.plot(mx_s2, ma_s2, color='darkgreen', linewidth=2)
        if ma_c2 is not None:
            ax.plot(mx_c2, ma_c2, color='darkred', linewidth=2)
        ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_title("Scout vs Commander Reward"); ax.set_xlabel("Batch"); ax.grid(True, alpha=0.3)

    # (2,1) Fire: peak vs final burning + suppressed
    ax = axes[2, 1]
    if fire_stats and len(fire_stats) > 0:
        bx = np.arange(1, len(fire_stats) + 1)
        peaks = [s[0] for s in fire_stats]
        finals = [s[1] for s in fire_stats]
        supps = [s[3] for s in fire_stats]
        ax.fill_between(bx, 0, peaks, color='#e74c3c', alpha=0.3, label='Peak burn')
        ax.fill_between(bx, 0, finals, color='#e67e22', alpha=0.5, label='Final burn')
        ax.plot(bx, supps, color='#2ecc71', linewidth=1.5, label='Suppressed')
        mx_p, ma_p = _ma(supps)
        if ma_p is not None:
            ax.plot(mx_p, ma_p, color='darkgreen', linewidth=2.5)
        ax.legend(fontsize=8)
    ax.set_title("Fire: Peak / Final / Suppressed [cells]"); ax.set_xlabel("Batch")
    ax.set_ylabel("Cells"); ax.grid(True, alpha=0.3)

    # (2,2) Refills per episode + extinguish effectiveness
    ax = axes[2, 2]
    if refill_hist and len(refill_hist) > 0:
        bx = np.arange(1, len(refill_hist) + 1)
        ax.bar(bx, refill_hist, color='#3498db', alpha=0.6, width=1.0, label='Refills/ep')
        mx_r, ma_r = _ma(refill_hist)
        if ma_r is not None:
            ax.plot(mx_r, ma_r, color='navy', linewidth=2)
        ax.legend(fontsize=8, loc='upper left')
    if fire_stats and len(fire_stats) > 0:
        ax2 = ax.twinx()
        exts = [s[2] for s in fire_stats]
        ax2.plot(np.arange(1, len(exts) + 1), exts, color='#e67e22', linewidth=1.5, alpha=0.8, label='Extinguish')
        mx_e, ma_e = _ma(exts)
        if ma_e is not None:
            ax2.plot(mx_e, ma_e, color='darkorange', linewidth=2.5)
        ax2.set_ylabel("Extinguish eff.", color='#e67e22')
        ax2.legend(fontsize=8, loc='upper right')
    ax.set_title("Refills/ep + Extinguish Effectiveness"); ax.set_xlabel("Batch")
    ax.set_ylabel("Refills/ep"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_b{batch_idx:04d}.png"), dpi=100)
    plt.close()

    # -- Loss breakdown plot ------------------------------------------
    if loss_comp_s and loss_comp_c:
        fig2, ax2 = plt.subplots(2, 3, figsize=(18, 8))
        fig2.suptitle(f"Loss Breakdown — Batch {batch_idx}", fontsize=13)
        xs   = np.arange(1, len(loss_comp_s) + 1)
        xs_c = np.arange(1, len(loss_comp_c) + 1)

        def _plot_comp(ax, data, xs, key, color, label):
            vals = [d[key] for d in data]
            ax.plot(xs, vals, color=color, linewidth=1.2, label=label)
            if len(vals) >= 10:
                mx = np.arange(10, len(vals) + 1)
                ma_ = np.convolve(vals, np.ones(10) / 10, mode='valid')
                ax.plot(mx, ma_, color=color, linewidth=2.2, alpha=0.6)

        _plot_comp(ax2[0,0], loss_comp_s, xs, "policy", "steelblue", "policy")
        ax2[0,0].set_title("Scout: Policy Loss"); ax2[0,0].grid(alpha=0.3)
        _plot_comp(ax2[0,1], loss_comp_s, xs, "entropy", "green", "entropy")
        ax2[0,1].set_title("Scout: Entropy"); ax2[0,1].grid(alpha=0.3)
        _plot_comp(ax2[0,2], loss_comp_s, xs, "value", "orange", "value")
        ax2[0,2].set_title("Scout: Critic MSE"); ax2[0,2].grid(alpha=0.3)

        _plot_comp(ax2[1,0], loss_comp_c, xs_c, "policy", "tomato", "policy")
        ax2[1,0].set_title("Cmdr: Policy Loss"); ax2[1,0].grid(alpha=0.3)
        _plot_comp(ax2[1,1], loss_comp_c, xs_c, "entropy", "darkorange", "entropy")
        ax2_aux = ax2[1,1].twinx()
        _plot_comp(ax2_aux, loss_comp_c, xs_c, "aux", "purple", "aux")
        ax2[1,1].set_title("Cmdr: Entropy + Aux"); ax2[1,1].grid(alpha=0.3)
        _plot_comp(ax2[1,2], loss_comp_c, xs_c, "value", "saddlebrown", "value")
        ax2[1,2].set_title("Cmdr: Critic MSE"); ax2[1,2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"loss_breakdown_b{batch_idx:04d}.png"), dpi=100)
        plt.close()


# =====================================================================
#  Entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent training: Scout + Commander")
    parser.add_argument("--resume-scout",      type=str, default="")
    parser.add_argument("--resume-cmdr",        type=str, default="")
    parser.add_argument("--reset-cmdr-critic",  action="store_true",
                        help="Re-init CmdrCritic weights (use after reward changes)")
    parser.add_argument("--log-episodes",       action="store_true",
                        help="Save trajectory logs (1 ep/worker/batch)")
    parser.add_argument("--log-dir",            type=str, default="/tmp/ep_logs")
    parser.add_argument("--start-ep",           type=int, default=0)
    args = parser.parse_args()

    train_multi(
        resume_scout=args.resume_scout,
        resume_cmdr=args.resume_cmdr,
        log_episodes=args.log_episodes,
        log_dir=args.log_dir,
        episodes_played=args.start_ep,
        reset_cmdr_critic=args.reset_cmdr_critic,
    )
