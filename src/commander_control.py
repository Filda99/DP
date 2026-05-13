"""
commander_control.py — Shared waypoint controller for the FW commander
=======================================================================

Encapsulates the logic shared between training, demo, and evaluation:
  - Scripted refill autopilot    (water < 10 % → fly to nearest refill zone)
  - NN waypoint decision         (water >= 10 % → network decides)
  - PD heading controller        (every physics step)
  - Rule-based valve             (open over estimated fire position)

Design decisions & problems solved
-----------------------------------
1. **Scripted refill vs NN refill**
   The refill flight is handled by a deterministic autopilot, not the NN.
   Early experiments showed that the NN struggled to learn both firefighting
   and refill navigation simultaneously — it would either ignore the refill
   zone or fly there permanently.  Scripting the low-water regime lets the
   NN focus entirely on fire suppression while guaranteeing water resupply.

2. **Multiple refill zones**
   A single refill zone placed far from the fire caused long return trips
   (up to 500 m), during which the fire grew unchecked.  Three refill zones
   placed 120° apart around the fire (clipped to 0.6× map boundary) ensure
   the nearest zone is always within ~150 m, reducing dead time.

3. **Boundary handling — reward shaping vs hard override**
   The original design used a hard emergency override that forcibly steered
   the FW to map centre when it approached the boundary.  This caused two
   problems: (a) the override consumed up to 94 % of control steps for some
   FW agents, leaving almost no NN training data, and (b) the NN never
   learned boundary avoidance because the override always rescued it.
   Solution: remove the override entirely and rely on a quadratic boundary
   penalty (reward_config.py) plus crash penalty at 2× map bounds.  The NN
   learns to stay inside through gradient signal alone.

4. **Valve logic — scout proximity vs fire position**
   The initial valve opened when the FW was within 50 m of a scout reporting
   fire.  Since scouts hover *near* fire but not exactly *over* it, this
   caused ~50 % water misses — water fell 30-50 m from the actual blaze.
   The fix reconstructs the fire's world position from the scout's message:
     fire_pos = scout_pos + dyn_offset × (FOV / 2)
   where dyn_offset (msg[3:5]) is the fire centroid in the scout's local
   camera frame and FOV = max(10, scout_alt × 1.5).  The valve now opens
   only when the FW is within 30 m of this estimated fire position.

5. **Reward isolation for scripted segments**
   During scripted refill, the FW flies away from scouts, accumulating
   negative fire-approach shaping reward.  This reward was originally
   counted in the episode total, making logged rewards appear much worse
   than the NN's actual performance.  Fix: exclude scripted segment
   rewards from ep_reward_cmdr — only NN-controlled segments contribute.

Usage (demo / eval — full convenience):
    ctrl = CommanderController()
    ctrl.reset(map_half=600.0)
    action, h_cmdr, info = ctrl.step(
        drone, obs_self_state, env, cmdr_actor, h_cmdr,
        scout_msgs_t, scout_mask_t, deterministic=True)

Usage (training — manual segment-end management):
    if ctrl.need_new_waypoint:
        h_cmdr, wp_info = ctrl.decide_waypoint(...)
    action = ctrl.heading_action(drone)
    ...
    if ctrl.check_segment_end() or last_step:
        ctrl.need_new_waypoint = True
"""

import numpy as np
import torch


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class CommanderController:
    """Waypoint-based FW commander with scripted refill autopilot.

    Scripted refill: when water < 10 %, waypoint → refill zone, water_raw = -1.
    NN firefighting: when water >= 10 %, NN produces [dx, dy, alt, water].
    Every physics step: PD heading controller tracks the active waypoint.
    """

    SAFE_LIMIT_FRAC = 0.55
    BOUNDARY_EMERGENCY_FRAC = 0.65
    REFILL_LIMIT_FRAC = 0.90  # wider limit for scripted refill navigation

    def __init__(self, waypoint_range=50.0, waypoint_steps=30,
                 wp_reached_dist=30.0):
        self.WP_RANGE = waypoint_range
        self.WP_STEPS = waypoint_steps
        self.WP_REACHED = wp_reached_dist

    def reset(self, map_half):
        """Reset state for a new episode."""
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_alt_raw = 0.0
        self.water_raw = -1.0
        self.steps_in_segment = 0
        self.need_new_waypoint = True
        self.wp_reached = False
        self.in_emergency = False
        self.safe_limit = max(50.0, map_half * self.SAFE_LIMIT_FRAC)
        self.refill_limit = max(50.0, map_half * self.REFILL_LIMIT_FRAC)
        self.boundary_emergency = max(50.0, map_half * self.BOUNDARY_EMERGENCY_FRAC)
        self.last_scout_msgs = None   # (num_scouts, msg_dim) tensor
        self.last_scout_mask = None   # (num_scouts,) bool tensor

    def update_limits(self, map_half):
        """Recalculate limits after map size change."""
        self.safe_limit = max(50.0, map_half * self.SAFE_LIMIT_FRAC)
        self.refill_limit = max(50.0, map_half * self.REFILL_LIMIT_FRAC)
        self.boundary_emergency = max(50.0, map_half * self.BOUNDARY_EMERGENCY_FRAC)

    # ------------------------------------------------------------------
    #  Low-level methods (used individually by the training worker)
    # ------------------------------------------------------------------

    def check_boundary_emergency(self, pos):
        """Disabled — NN learns boundaries through reward shaping.

        The boundary penalty in reward_config.py (quadratic, threshold=300m)
        plus crash_penalty at 4×map_bounds teaches the NN to stay inside.
        No hard override needed.
        """
        self.in_emergency = False
        return False

    def check_segment_end(self):
        """True if the current waypoint segment is done."""
        return self.wp_reached or self.steps_in_segment >= self.WP_STEPS

    def decide_waypoint(self, drone, obs_self_state, env, cmdr_actor, h_cmdr,
                        scout_msgs_t, scout_mask_t, *, deterministic=False,
                        fw_neighbor_states=None, fw_neighbor_mask=None,
                        in_emergency=False):
        """Compute a new waypoint (emergency / scripted refill / NN).

        Only call when ``self.need_new_waypoint`` is True.

        Returns
        -------
        h_cmdr_new : tensor
        info : dict
            'scripted'  – bool (True for refill AND emergency)
            'nn_dist'   – Distribution or None
            'nn_act'    – tensor or None
            'nn_state'  – tensor or None
            'nn_aux'    – tensor or None
        """
        pos = drone.get_position()
        water_frac = (drone.current_water / drone.water_capacity
                      if drone.water_capacity > 0 else 1.0)
        rz = env.sim.environment.refill_zone
        zones = getattr(env.sim.environment, 'refill_zones', [])
        use_scripted = (water_frac <= 0.10 and (rz is not None or zones))

        s_st = (torch.FloatTensor(obs_self_state).unsqueeze(0)
                if not isinstance(obs_self_state, torch.Tensor)
                else obs_self_state)
        dev = scout_msgs_t.device
        s_st = s_st.to(dev)

        info = {'scripted': False, 'nn_dist': None, 'nn_act': None,
                'nn_state': None, 'nn_aux': None}

        if in_emergency:
            # ── Boundary emergency: fly to center at safe altitude ────
            self.target_x, self.target_y = 0.0, 0.0
            self.target_alt_raw = 0.0   # ~145 m
            self.water_raw = -1.0       # close valve
            # Dummy forward pass keeps GRU hidden state fresh
            with torch.no_grad():
                _, _, h_cmdr = cmdr_actor(
                    s_st, scout_msgs_t, scout_mask_t, h_cmdr,
                    fw_neighbor_states, fw_neighbor_mask)
            info['scripted'] = True
        elif use_scripted:
            # ── Scripted refill: waypoint → nearest refill zone ───────
            # Pick the closest zone to minimise travel time
            if zones:
                best_zone = min(zones, key=lambda z:
                    (z['position'][0]-pos[0])**2 + (z['position'][1]-pos[1])**2)
                rz_pos = best_zone['position']
            else:
                rz_pos = rz['position']
            dx_r = float(np.clip((rz_pos[0] - pos[0]) / self.WP_RANGE, -1, 1))
            dy_r = float(np.clip((rz_pos[1] - pos[1]) / self.WP_RANGE, -1, 1))
            self.target_alt_raw = 0.0
            self.water_raw = -1.0
            self.target_x = float(np.clip(
                pos[0] + dx_r * self.WP_RANGE,
                -self.refill_limit, self.refill_limit))
            self.target_y = float(np.clip(
                pos[1] + dy_r * self.WP_RANGE,
                -self.refill_limit, self.refill_limit))
            # Dummy forward pass keeps GRU hidden state fresh
            with torch.no_grad():
                _, _, h_cmdr = cmdr_actor(
                    s_st, scout_msgs_t, scout_mask_t, h_cmdr,
                    fw_neighbor_states, fw_neighbor_mask)
            info['scripted'] = True
        else:
            # ── NN firefighting decision ──────────────────────────────
            with torch.no_grad():
                dist, aux_pred, h_cmdr = cmdr_actor(
                    s_st, scout_msgs_t, scout_mask_t, h_cmdr,
                    fw_neighbor_states, fw_neighbor_mask)
            act = dist.mean if deterministic else dist.sample()
            act_np = act.squeeze(0).detach().cpu().numpy()
            self.target_alt_raw = float(act_np[2])
            self.water_raw = float(act_np[3])
            self.target_x = float(np.clip(
                pos[0] + act_np[0] * self.WP_RANGE,
                -self.safe_limit, self.safe_limit))
            self.target_y = float(np.clip(
                pos[1] + act_np[1] * self.WP_RANGE,
                -self.safe_limit, self.safe_limit))
            info['nn_dist'] = dist
            info['nn_act'] = act
            info['nn_state'] = s_st
            info['nn_aux'] = aux_pred

        self.steps_in_segment = 0
        self.wp_reached = False
        self.need_new_waypoint = False
        return h_cmdr, info

    def heading_action(self, drone, env=None):
        """PD heading controller.  Call every physics step.

        Updates ``wp_reached`` and ``steps_in_segment``.
        Valve is rule-based: open when FW flies over estimated fire position.

        Returns
        -------
        action : np.ndarray  [heading_cmd, target_alt_raw, water_raw]
        """
        pos = drone.get_position()
        dx_to = self.target_x - pos[0]
        dy_to = self.target_y - pos[1]
        dist_to = np.hypot(dx_to, dy_to)

        if dist_to < self.WP_REACHED:
            self.wp_reached = True

        if dist_to > 1.0:
            desired = np.arctan2(dy_to, dx_to)
            cur_yaw = drone.get_orientation_rpy()[2]
            heading_cmd = float(np.clip(
                _wrap_angle(desired - cur_yaw) / np.pi, -1, 1))
        else:
            heading_cmd = 0.0

        # ── Rule-based valve ─────────────────────────────────────
        # Opens when FW is within 50 m of the scout's estimated fire
        # position AND heading toward it (dot product > 0), or within
        # 15 m regardless of heading.  The fire position is reconstructed
        # from the scout camera message:
        #   fire_pos = scout_pos + dyn_offset × (FOV / 2)
        valve = -1.0
        self._valve_debug = None
        if env is not None and pos[2] < 80.0 and drone.current_water > 0:
            msgs = self.last_scout_msgs
            mask = self.last_scout_mask
            any_scout_sees_fire = False
            best_est_d = float('inf')
            best_est_x, best_est_y = 0.0, 0.0

            if msgs is not None:
                msg_flat = msgs.view(-1, msgs.size(-1))
                for i, q in enumerate(env.quad_agents):
                    if q not in env.sim.drones:
                        continue
                    if mask is not None and mask.dim() >= 1 and i < mask.size(-1) and mask.view(-1)[i]:
                        continue
                    if msg_flat[i, 2].item() <= 0:
                        continue
                    any_scout_sees_fire = True
                    # Reconstruct fire world position from scout camera
                    sq_pos = env.sim.drones[q].get_position()
                    dyn_x = msg_flat[i, 3].item()
                    dyn_y = msg_flat[i, 4].item()
                    fov_half = max(10.0, sq_pos[2] * 1.5) / 2.0
                    est_x = sq_pos[0] + dyn_x * fov_half
                    est_y = sq_pos[1] + dyn_y * fov_half
                    d_est = np.hypot(pos[0] - est_x, pos[1] - est_y)
                    if d_est < best_est_d:
                        best_est_d = d_est
                        best_est_x, best_est_y = est_x, est_y

            # Valve opens when FW is within 50 m of scout fire estimate
            # AND heading toward it (or very close < 15 m).
            opened = False
            if any_scout_sees_fire and best_est_d < 50.0:
                vel = drone.get_velocity()
                spd = np.hypot(vel[0], vel[1])
                if spd > 1.0:
                    to_fire = np.array([best_est_x - pos[0], best_est_y - pos[1]])
                    dot = vel[0] * to_fire[0] + vel[1] * to_fire[1]
                    if dot > 0 or best_est_d < 15.0:
                        opened = True
                else:
                    opened = True

            if opened:
                valve = 1.0

            self._valve_debug = {
                'scout_sees': any_scout_sees_fire,
                'opened': opened,
                'd_est': best_est_d if any_scout_sees_fire else None,
                'reason': f'est={best_est_d:.0f}m' if opened else (
                    'no_scout_fire' if not any_scout_sees_fire else
                    f'est={best_est_d:.0f}m,too_far_or_wrong_dir'),
            }

        self.steps_in_segment += 1
        return np.array(
            [heading_cmd, self.target_alt_raw, valve],
            dtype=np.float32)

    # ------------------------------------------------------------------
    #  High-level convenience  (demo / eval — single call per step)
    # ------------------------------------------------------------------

    def step(self, drone, obs_self_state, env, cmdr_actor, h_cmdr,
             scout_msgs_t, scout_mask_t, *, deterministic=False,
             fw_neighbor_states=None, fw_neighbor_mask=None):
        """Full commander physics step.

        Combines: boundary check → segment-end check → waypoint
        decision → PD heading controller.

        Returns
        -------
        action    : np.ndarray
        h_cmdr    : tensor  (updated GRU hidden state)
        info      : dict    (superset of decide_waypoint info, plus
                             'in_emergency' and 'new_waypoint' flags)
        """
        pos = drone.get_position()
        info = {'in_emergency': False, 'new_waypoint': False,
                'scripted': False, 'nn_dist': None, 'nn_act': None,
                'nn_state': None, 'nn_aux': None}

        in_emergency = self.check_boundary_emergency(pos)
        info['in_emergency'] = in_emergency

        if in_emergency:
            # Force new waypoint → decide_waypoint handles emergency
            self.need_new_waypoint = True

        if self.check_segment_end():
            self.need_new_waypoint = True

        if self.need_new_waypoint:
            h_cmdr, wp_info = self.decide_waypoint(
                drone, obs_self_state, env, cmdr_actor, h_cmdr,
                scout_msgs_t, scout_mask_t,
                deterministic=deterministic,
                fw_neighbor_states=fw_neighbor_states,
                fw_neighbor_mask=fw_neighbor_mask,
                in_emergency=in_emergency)
            info.update(wp_info)
            info['new_waypoint'] = True

        # Cache scout messages for valve logic in heading_action
        self.last_scout_msgs = scout_msgs_t
        self.last_scout_mask = scout_mask_t

        action = self.heading_action(drone, env=env)
        return action, h_cmdr, info
