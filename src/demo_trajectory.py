"""
demo_trajectory.py — Visualize drone trajectory, waypoints, and boundary deaths.

Runs the trained SimpleFWActor for a few episodes and produces a top-down plot
showing the flight path, waypoint targets (green=reached, red=timeout, X=death),
and the map boundary / safe zone.

Usage:
  python demo_trajectory.py                        # random policy (no model)
  python demo_trajectory.py --actor saved_models/fw_survival/actor_best.pt
"""

import os, sys, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_core import DroneFireEnv
from models import SimpleFWActor

def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def run_episode(env, actor, device, hidden_dim, waypoint_steps, waypoint_range,
                safe_limit, wp_reached_dist, max_steps, num_decisions):
    """Run one episode, returning trajectory data."""
    obs, _ = env.reset()
    f_agent = env.fixed_agents[0]

    actor_h = torch.zeros(1, 1, hidden_dim).to(device)

    trajectory = []    # list of (x, y, z) per physics step
    waypoints = []     # list of dict: {target_x, target_y, reached, pos_at_decision, death}
    total_inner_steps = 0
    agent_dead = False
    death_pos = None
    death_cause = None

    for dec_idx in range(num_decisions):
        if agent_dead or f_agent not in env.agents:
            break

        remaining = max_steps - total_inner_steps
        if remaining <= 0:
            break

        # --- NN decision ---
        drone = env.sim.drones.get(f_agent)
        if drone is None:
            break
        cur_pos = drone.get_position()

        with torch.no_grad():
            s_st = torch.FloatTensor(obs[f_agent]["self_state"]).unsqueeze(0).to(device)
            dist_out, _, h_out = actor(s_st, None, None, actor_h)
            act = dist_out.sample()

        actor_h = h_out
        act_np = act.squeeze(0).cpu().numpy()

        dx_raw = float(act_np[0])
        dy_raw = float(act_np[1])
        target_alt_raw = float(act_np[2])
        water_raw = float(act_np[3])

        target_x = cur_pos[0] + dx_raw * waypoint_range
        target_y = cur_pos[1] + dy_raw * waypoint_range
        target_x = np.clip(target_x, -safe_limit, safe_limit)
        target_y = np.clip(target_y, -safe_limit, safe_limit)

        wp_info = {
            "target_x": target_x, "target_y": target_y,
            "start_x": cur_pos[0], "start_y": cur_pos[1],
            "dx_raw": dx_raw, "dy_raw": dy_raw,
            "reached": False, "death": False
        }

        # --- Fly toward waypoint ---
        wp_reached = False
        steps_this_segment = min(waypoint_steps, remaining)

        for inner in range(steps_this_segment):
            if f_agent not in env.agents:
                break
            drone = env.sim.drones.get(f_agent)
            if drone is None:
                break

            pos = drone.get_position()
            trajectory.append((pos[0], pos[1], pos[2]))

            dx_to_target = target_x - pos[0]
            dy_to_target = target_y - pos[1]
            dist_to_target = np.sqrt(dx_to_target**2 + dy_to_target**2)

            if dist_to_target < wp_reached_dist:
                wp_reached = True

            if dist_to_target > 1.0:
                desired_heading = np.arctan2(dy_to_target, dx_to_target)
                cur_yaw = drone.get_orientation_rpy()[2]
                heading_error = _wrap_angle(desired_heading - cur_yaw)
                heading_cmd = np.clip(heading_error / np.pi, -1.0, 1.0)
            else:
                heading_cmd = 0.0

            inner_action = np.array([heading_cmd, target_alt_raw, water_raw],
                                    dtype=np.float32)
            obs, rewards_env, terms, _, infos_env = env.step({f_agent: inner_action})
            total_inner_steps += 1

            if terms.get(f_agent, False):
                death_pos = pos
                death_cause = infos_env.get(f_agent, {}).get("death_cause", "unknown")
                wp_info["death"] = True
                agent_dead = True
                break

            if wp_reached:
                break

        wp_info["reached"] = wp_reached
        waypoints.append(wp_info)

    return trajectory, waypoints, death_pos, death_cause, total_inner_steps


def plot_episodes(all_episodes, save_path, map_bounds, safe_limit):
    n_eps = len(all_episodes)
    cols = min(3, n_eps)
    rows = (n_eps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 7*rows))
    if n_eps == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for i, (traj, wps, death_pos, death_cause, steps) in enumerate(all_episodes):
        ax = axes[i]

        # Map boundary
        rect_bound = plt.Rectangle((-map_bounds, -map_bounds), 2*map_bounds, 2*map_bounds,
                                   fill=False, edgecolor='red', linewidth=2, linestyle='-',
                                   label=f'Boundary (±{map_bounds}m)')
        ax.add_patch(rect_bound)

        # Safe zone
        rect_safe = plt.Rectangle((-safe_limit, -safe_limit), 2*safe_limit, 2*safe_limit,
                                  fill=False, edgecolor='orange', linewidth=1.5, linestyle='--',
                                  label=f'Safe zone (±{safe_limit}m)')
        ax.add_patch(rect_safe)

        # Trajectory
        if traj:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.6, label='Flight path')
            ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')

        # Waypoints with arrows from decision position to target
        for j, wp in enumerate(wps):
            color = 'green' if wp["reached"] else ('red' if not wp["death"] else 'black')
            marker = 'o' if not wp["death"] else 'X'
            ax.plot(wp["target_x"], wp["target_y"], marker=marker, color=color,
                    markersize=8, zorder=5)
            # Arrow from decision position to target
            ax.annotate('', xy=(wp["target_x"], wp["target_y"]),
                       xytext=(wp["start_x"], wp["start_y"]),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1))
            ax.text(wp["target_x"]+15, wp["target_y"]+15, str(j+1),
                    fontsize=7, color=color)

        # Death marker
        if death_pos is not None:
            ax.plot(death_pos[0], death_pos[1], 'rx', markersize=15, markeredgewidth=3,
                    label=f'Death: {death_cause}')
            # Print distance from boundary
            dx_bound = map_bounds - abs(death_pos[0])
            dy_bound = map_bounds - abs(death_pos[1])
            min_bound = min(dx_bound, dy_bound)
            ax.set_title(f'Ep {i+1}: {steps} steps, {death_cause}\n'
                        f'Death@({death_pos[0]:.0f},{death_pos[1]:.0f}) '
                        f'dist_to_boundary={min_bound:.0f}m',
                        fontsize=10)
        else:
            ax.set_title(f'Ep {i+1}: {steps} steps, SURVIVED', fontsize=10)

        # Formatting
        margin = 200
        ax.set_xlim(-map_bounds - margin, map_bounds + margin)
        ax.set_ylim(-map_bounds - margin, map_bounds + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

        # Stats text
        n_reached = sum(1 for w in wps if w["reached"])
        n_timeout = sum(1 for w in wps if not w["reached"] and not w["death"])
        stats = f"WP: {len(wps)} total, {n_reached} reached, {n_timeout} timeout"
        ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("FW Waypoint Trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=str, default="",
                        help="Path to actor .pt file (empty = random init)")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    max_steps = 1000
    waypoint_steps = 50
    waypoint_range = 200.0
    num_decisions = max_steps // waypoint_steps
    wp_reached_dist = 50.0
    hidden_dim = 64
    grid_size_m = 3000.0
    map_bounds = grid_size_m / 2.0   # 1500
    safe_limit = map_bounds - 200.0  # 1300

    device = torch.device("cpu")

    env = DroneFireEnv(num_quads=0, num_fixed=1, grid_size_m=grid_size_m,
                       max_steps=max_steps)
    fixed_self_dim = env.observation_space(env.fixed_agents[0])["self_state"].shape[0]

    actor = SimpleFWActor(self_state_dim=fixed_self_dim, hidden_dim=hidden_dim, action_dim=4)
    if args.actor and os.path.isfile(args.actor):
        actor.load_state_dict(torch.load(args.actor, map_location=device))
        print(f"Loaded actor: {args.actor}")
    else:
        print("Using random actor (no checkpoint loaded)")
    actor.eval()

    all_episodes = []
    for ep in range(args.episodes):
        print(f"  Running episode {ep+1}/{args.episodes}...", end=" ")
        traj, wps, death_pos, death_cause, steps = run_episode(
            env, actor, device, hidden_dim, waypoint_steps, waypoint_range,
            safe_limit, wp_reached_dist, max_steps, num_decisions)
        status = f"DIED ({death_cause})" if death_pos is not None else "SURVIVED"
        n_reached = sum(1 for w in wps if w["reached"])
        print(f"{steps} steps, {len(wps)} WP ({n_reached} reached), {status}")
        all_episodes.append((traj, wps, death_pos, death_cause, steps))

    if args.output:
        save_path = args.output
    else:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "output")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "trajectory_debug.png")

    plot_episodes(all_episodes, save_path, map_bounds, safe_limit)

    if hasattr(env, 'sim') and env.sim is not None:
        env.sim.stop_simulation()


if __name__ == "__main__":
    main()
