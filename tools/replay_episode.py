#!/usr/bin/env python3
"""Replay an episode from saved .npz log, producing trajectory plot + reward graphs."""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_episode(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def plot_episode(data, out_path=None, show=False):
    scout_pos = data.get("scout_pos", None)
    cmdr_pos = data.get("cmdr_pos", None)
    waypoints = data.get("cmdr_waypoints", None)
    r_scout = data.get("rewards_scout", None)
    r_cmdr = data.get("rewards_cmdr", None)
    map_bounds = float(data.get("map_bounds", 1500))

    ep_id = int(data.get("ep_id", 0))
    s_rew = float(data.get("scout_reward", 0))
    c_rew = float(data.get("cmdr_reward", 0))
    s_life = int(data.get("scout_lifespan", 0))
    c_life = int(data.get("cmdr_lifespan", 0))
    s_death = str(data.get("scout_death", "?"))
    c_death = str(data.get("cmdr_death", "?"))

    has_traj = scout_pos is not None and len(scout_pos) > 0
    has_rew = r_scout is not None and len(r_scout) > 0

    n_plots = 1 + int(has_rew) + int(has_traj)  # traj + rewards + altitude
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    title = (f"Episode {ep_id}  |  Scout R={s_rew:.0f} life={s_life} ({s_death})  |  "
             f"Cmdr R={c_rew:.0f} life={c_life} ({c_death})")
    fig.suptitle(title, fontsize=11, y=0.98)

    ax_idx = 0

    # --- Plot 1: Top-down trajectory ---
    if has_traj:
        ax = axes[ax_idx]; ax_idx += 1
        b = map_bounds
        safe = map_bounds - 200
        ax.add_patch(mpatches.Rectangle((-b, -b), 2*b, 2*b,
                     fill=False, edgecolor='red', linewidth=2, linestyle='--', label='boundary'))
        ax.add_patch(mpatches.Rectangle((-safe, -safe), 2*safe, 2*safe,
                     fill=False, edgecolor='orange', linewidth=1, linestyle=':', label='safe zone'))

        if scout_pos is not None and len(scout_pos) > 0:
            sp = np.array(scout_pos)
            valid = ~np.isnan(sp[:, 0])
            ax.plot(sp[valid, 0], sp[valid, 1], 'b-', alpha=0.5, linewidth=0.8, label='scout')
            if valid.any():
                ax.plot(sp[valid][0, 0], sp[valid][0, 1], 'bo', markersize=8)
                last = np.where(valid)[0][-1]
                ax.plot(sp[last, 0], sp[last, 1], 'bx', markersize=10, markeredgewidth=2)

        if cmdr_pos is not None and len(cmdr_pos) > 0:
            cp = np.array(cmdr_pos)
            valid = ~np.isnan(cp[:, 0])
            ax.plot(cp[valid, 0], cp[valid, 1], 'r-', alpha=0.5, linewidth=0.8, label='commander')
            if valid.any():
                ax.plot(cp[valid][0, 0], cp[valid][0, 1], 'ro', markersize=8)
                last = np.where(valid)[0][-1]
                ax.plot(cp[last, 0], cp[last, 1], 'rx', markersize=10, markeredgewidth=2)

        if waypoints is not None and len(waypoints) > 0:
            wp = np.array(waypoints)
            ax.scatter(wp[:, 1], wp[:, 2], c='green', marker='^', s=40,
                      zorder=5, label='waypoints')
            # Color by water flag
            water_mask = wp[:, 4] > 0.5
            if water_mask.any():
                ax.scatter(wp[water_mask, 1], wp[water_mask, 2],
                          c='cyan', marker='v', s=60, zorder=6, label='water drop')

        ax.set_xlim(-b * 1.05, b * 1.05)
        ax.set_ylim(-b * 1.05, b * 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Top-down trajectory')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    # --- Plot 2: Per-step rewards ---
    if has_rew:
        ax = axes[ax_idx]; ax_idx += 1
        steps = np.arange(len(r_scout))
        # Cumulative rewards
        cum_scout = np.cumsum(r_scout)
        cum_cmdr = np.cumsum(r_cmdr)
        ax.plot(steps, cum_scout, 'b-', alpha=0.8, label=f'scout (total={s_rew:.0f})')
        ax.plot(steps, cum_cmdr, 'r-', alpha=0.8, label=f'cmdr (total={c_rew:.0f})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative reward')
        ax.set_title('Cumulative rewards')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Plot 3: Altitude profile ---
    if has_traj:
        ax = axes[ax_idx]; ax_idx += 1
        if scout_pos is not None and len(scout_pos) > 0:
            sp = np.array(scout_pos)
            valid = ~np.isnan(sp[:, 2])
            ax.plot(np.where(valid)[0], sp[valid, 2], 'b-', alpha=0.8, label='scout alt')
        if cmdr_pos is not None and len(cmdr_pos) > 0:
            cp = np.array(cmdr_pos)
            valid = ~np.isnan(cp[:, 2])
            ax.plot(np.where(valid)[0], cp[valid, 2], 'r-', alpha=0.8, label='cmdr alt')
        ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='alt min (40m)')
        ax.axhline(y=150, color='orange', linestyle=':', alpha=0.5, label='alt max (150m)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Altitude profile')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replay episode logs")
    parser.add_argument("path", help="Path to .npz file or directory of .npz files")
    parser.add_argument("--out-dir", default=None, help="Output directory for plots (default: same as input)")
    parser.add_argument("--last", type=int, default=5, help="When path is a directory, plot last N episodes")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "ep_*.npz")))
        if not files:
            print(f"No episode files found in {args.path}")
            sys.exit(1)
        files = files[-args.last:]
        print(f"Found {len(files)} episode files, plotting last {len(files)}")
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)

    out_dir = args.out_dir or (os.path.dirname(files[0]) if os.path.isfile(args.path) else args.path)
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        data = load_episode(f)
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, f"{base}.png")
        plot_episode(data, out_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
