"""
commander_control.py — Shared waypoint controller for the FW commander
=======================================================================

Encapsulates the logic shared between training, demo, and evaluation:
  - Boundary emergency override  (fly to map center)
  - Scripted refill autopilot    (water < 30 % → fly to refill zone)
  - NN waypoint decision         (water >= 30 % → network decides)
  - PD heading controller        (every physics step)

Usage (demo / eval — full convenience):
    ctrl = CommanderController()
    ctrl.reset(safe_limit=420.0, boundary_emergency=360.0)
    action, h_cmdr, info = ctrl.step(
        drone, obs_self_state, env, cmdr_actor, h_cmdr,
        scout_msgs_t, scout_mask_t, deterministic=True)

Usage (training — manual segment-end management):
    in_emergency = ctrl.check_boundary_emergency(pos)
    if ctrl.need_new_waypoint and not in_emergency:
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

    Scripted refill: when water < 30 %, waypoint → refill zone, water_raw = -1.
    NN firefighting: when water >= 30 %, NN produces [dx, dy, alt, water].
    Every physics step: PD heading controller tracks the active waypoint.
    """

    def __init__(self, waypoint_range=200.0, waypoint_steps=30,
                 wp_reached_dist=30.0):
        self.WP_RANGE = waypoint_range
        self.WP_STEPS = waypoint_steps
        self.WP_REACHED = wp_reached_dist

    def reset(self, safe_limit=420.0, boundary_emergency=360.0):
        """Reset state for a new episode."""
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_alt_raw = 0.0
        self.water_raw = -1.0
        self.steps_in_segment = 0
        self.need_new_waypoint = True
        self.wp_reached = False
        self.safe_limit = safe_limit
        self.boundary_emergency = boundary_emergency

    def update_limits(self, map_half):
        """Recalculate limits after map size change."""
        self.safe_limit = max(50.0, map_half * 0.7)
        self.boundary_emergency = max(50.0, map_half * 0.6)

    # ------------------------------------------------------------------
    #  Low-level methods (used individually by the training worker)
    # ------------------------------------------------------------------

    def check_boundary_emergency(self, pos):
        """Check and handle boundary emergency.  Returns True if active."""
        if (abs(pos[0]) > self.boundary_emergency or
                abs(pos[1]) > self.boundary_emergency):
            self.target_x, self.target_y = 0.0, 0.0
            return True
        return False

    def check_segment_end(self):
        """True if the current waypoint segment is done."""
        return self.wp_reached or self.steps_in_segment >= self.WP_STEPS

    def decide_waypoint(self, drone, obs_self_state, env, cmdr_actor, h_cmdr,
                        scout_msgs_t, scout_mask_t, *, deterministic=False):
        """Compute a new waypoint (scripted refill or NN).

        Only call when ``self.need_new_waypoint`` is True.

        Returns
        -------
        h_cmdr_new : tensor
        info : dict
            'scripted'  – bool
            'nn_dist'   – Distribution or None
            'nn_act'    – tensor or None
            'nn_state'  – tensor or None
            'nn_aux'    – tensor or None
        """
        pos = drone.get_position()
        water_frac = (drone.current_water / drone.water_capacity
                      if drone.water_capacity > 0 else 1.0)
        rz = env.sim.environment.refill_zone
        use_scripted = (water_frac < 0.30 and rz is not None)

        s_st = (torch.FloatTensor(obs_self_state).unsqueeze(0)
                if not isinstance(obs_self_state, torch.Tensor)
                else obs_self_state)
        dev = scout_msgs_t.device
        s_st = s_st.to(dev)

        info = {'scripted': False, 'nn_dist': None, 'nn_act': None,
                'nn_state': None, 'nn_aux': None}

        if use_scripted:
            # ── Scripted refill: waypoint → refill zone ───────────────
            rz_pos = rz['position']
            dx_r = float(np.clip((rz_pos[0] - pos[0]) / self.WP_RANGE, -1, 1))
            dy_r = float(np.clip((rz_pos[1] - pos[1]) / self.WP_RANGE, -1, 1))
            self.target_alt_raw = 0.0
            self.water_raw = -1.0
            self.target_x = float(np.clip(
                pos[0] + dx_r * self.WP_RANGE,
                -self.safe_limit, self.safe_limit))
            self.target_y = float(np.clip(
                pos[1] + dy_r * self.WP_RANGE,
                -self.safe_limit, self.safe_limit))
            # Dummy forward pass keeps GRU hidden state fresh
            with torch.no_grad():
                _, _, h_cmdr = cmdr_actor(
                    s_st, scout_msgs_t, scout_mask_t, h_cmdr)
            info['scripted'] = True
        else:
            # ── NN firefighting decision ──────────────────────────────
            with torch.no_grad():
                dist, aux_pred, h_cmdr = cmdr_actor(
                    s_st, scout_msgs_t, scout_mask_t, h_cmdr)
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

    def heading_action(self, drone):
        """PD heading controller.  Call every physics step.

        Updates ``wp_reached`` and ``steps_in_segment``.

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

        self.steps_in_segment += 1
        return np.array(
            [heading_cmd, self.target_alt_raw, self.water_raw],
            dtype=np.float32)

    # ------------------------------------------------------------------
    #  High-level convenience  (demo / eval — single call per step)
    # ------------------------------------------------------------------

    def step(self, drone, obs_self_state, env, cmdr_actor, h_cmdr,
             scout_msgs_t, scout_mask_t, *, deterministic=False):
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

        if not in_emergency:
            if self.check_segment_end():
                self.need_new_waypoint = True
            if self.need_new_waypoint:
                h_cmdr, wp_info = self.decide_waypoint(
                    drone, obs_self_state, env, cmdr_actor, h_cmdr,
                    scout_msgs_t, scout_mask_t,
                    deterministic=deterministic)
                info.update(wp_info)
                info['new_waypoint'] = True

        action = self.heading_action(drone)
        return action, h_cmdr, info
