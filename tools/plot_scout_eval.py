#!/usr/bin/env python3
"""
Generate clean, thesis-quality plots from scout evaluation CSV.

Usage:
    python tools/plot_scout_eval.py results/scout_eval_thesis/eval_scout.csv \
           --out-dir results/scout_eval_thesis
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

SCOUT_COLORS = {1: '#e74c3c', 2: '#e67e22', 3: '#2ecc71', 4: '#3498db', 5: '#9b59b6'}
SCOUT_MARKERS = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'seed': int(r['seed']),
                'n_scouts': int(r['n_scouts']),
                'map_m': int(r['map_m']),
                'surv_pct': float(r['surv_pct']),
                'disc': int(r['disc']),
                't_disc': float(r['t_disc']),
                'dwell_pct': float(r['dwell_pct']),
                'dwell_mean': float(r['dwell_mean']),
                'sep_avg_m': float(r['sep_avg_m']),
                'R_per_scout': float(r['R_per_scout']),
            })
    return rows


def aggregate(rows):
    """Group by (n_scouts, map_m), compute mean±std."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r['n_scouts'], r['map_m'])].append(r)
    agg = {}
    for (ns, gs), sub in sorted(groups.items()):
        agg[(ns, gs)] = {
            'n': len(sub),
            'surv_mean': np.mean([r['surv_pct'] for r in sub]),
            'surv_std': np.std([r['surv_pct'] for r in sub]),
            'disc_pct': np.mean([r['disc'] for r in sub]) * 100,
            'dwell_mean': np.mean([r['dwell_pct'] for r in sub]),
            'dwell_std': np.std([r['dwell_pct'] for r in sub]),
            'dwell_per_scout_mean': np.mean([r['dwell_mean'] for r in sub]),
            'dwell_per_scout_std': np.std([r['dwell_mean'] for r in sub]),
            't_disc_mean': np.mean([r['t_disc'] for r in sub if r['t_disc'] >= 0]),
            't_disc_std': np.std([r['t_disc'] for r in sub if r['t_disc'] >= 0]),
            'R_mean': np.mean([r['R_per_scout'] for r in sub]),
            'R_std': np.std([r['R_per_scout'] for r in sub]),
        }
    return agg


def plot_lines(agg, scout_counts, grid_sizes, out_dir):
    """
    Figure 1: Line plots — each metric as a function of map size,
    one line per scout count. Clear, comparable, thesis-quality.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Scout Team Evaluation", fontsize=15, fontweight='bold', y=0.99)

    metrics = [
        ('dwell_mean', 'dwell_std', 'Team Fire Visibility [%]', (0, 110)),
        ('dwell_per_scout_mean', 'dwell_per_scout_std', 'Per-Scout Mean Dwell [%]', (0, 110)),
        ('disc_pct', None, 'Fire Discovery Rate [%]', (0, 110)),
        ('t_disc_mean', 't_disc_std', 'Discovery Time [steps]', None),
        ('R_mean', 'R_std', 'Reward per Scout', None),
        ('surv_mean', 'surv_std', 'Survival Rate [%]', (90, 101)),
    ]

    for idx, (key_mean, key_std, ylabel, ylim) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        for ns in scout_counts:
            xs, ys, es = [], [], []
            for gs in grid_sizes:
                if (ns, gs) in agg:
                    a = agg[(ns, gs)]
                    xs.append(gs)
                    ys.append(a[key_mean])
                    es.append(a.get(key_std, 0) if key_std else 0)
            if xs:
                color = SCOUT_COLORS.get(ns, 'gray')
                marker = SCOUT_MARKERS.get(ns, 'o')
                label = f'{ns} scout{"s" if ns > 1 else ""}'
                ax.plot(xs, ys, marker=marker, color=color, linewidth=2,
                        markersize=6, label=label, zorder=3)
                if key_std:
                    ys_arr = np.array(ys)
                    es_arr = np.array(es)
                    lo = ys_arr - es_arr
                    hi = ys_arr + es_arr
                    # Clamp lower bound for non-negative metrics
                    if key_mean != 'R_mean':
                        lo = np.maximum(lo, 0)
                    ax.fill_between(xs, lo, hi,
                                    color=color, alpha=0.12, zorder=1)
        ax.set_xlabel('Map Size [m]')
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'scout_eval_lines.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  → {path}')
    plt.close(fig)


def plot_heatmaps(agg, scout_counts, grid_sizes, out_dir):
    """
    Figure 2: Heatmaps — compact 2D view (scouts × map size) for each metric.
    """
    metrics = [
        ('dwell_mean', 'Team Fire Visibility [%]', 'YlGn', (0, 100)),
        ('disc_pct', 'Fire Discovery Rate [%]', 'YlGn', (0, 100)),
        ('t_disc_mean', 'Discovery Time [steps]', 'YlOrRd', None),
        ('R_mean', 'Reward per Scout', 'RdYlGn', None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    fig.suptitle("Scout Evaluation — Heatmaps", fontsize=14, fontweight='bold', y=1.01)

    for idx, (key, title, cmap_name, vlim) in enumerate(metrics):
        ax = axes[idx]
        mat = np.full((len(scout_counts), len(grid_sizes)), np.nan)
        for i, ns in enumerate(scout_counts):
            for j, gs in enumerate(grid_sizes):
                if (ns, gs) in agg:
                    mat[i, j] = agg[(ns, gs)][key]

        vmin = vlim[0] if vlim else np.nanmin(mat)
        vmax = vlim[1] if vlim else np.nanmax(mat)
        im = ax.imshow(mat, aspect='auto', cmap=cmap_name, vmin=vmin, vmax=vmax,
                       origin='lower')
        # Annotate cells
        for i in range(len(scout_counts)):
            for j in range(len(grid_sizes)):
                val = mat[i, j]
                if not np.isnan(val):
                    txt = f'{val:.0f}' if abs(val) >= 10 else f'{val:.1f}'
                    color = 'white' if (val - vmin) / max(1, vmax - vmin) > 0.65 else 'black'
                    if cmap_name == 'YlOrRd':
                        color = 'white' if (val - vmin) / max(1, vmax - vmin) > 0.55 else 'black'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                            fontweight='bold', color=color)

        ax.set_xticks(range(len(grid_sizes)))
        ax.set_xticklabels([f'{g}' for g in grid_sizes], fontsize=9)
        ax.set_yticks(range(len(scout_counts)))
        ax.set_yticklabels([f'{s}S' for s in scout_counts], fontsize=10)
        ax.set_xlabel('Map Size [m]')
        if idx == 0:
            ax.set_ylabel('Scout Count')
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'scout_eval_heatmaps.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  → {path}')
    plt.close(fig)


def plot_discovery(agg, scout_counts, grid_sizes, out_dir):
    """
    Figure 3: Discovery time — dedicated clean plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for ns in scout_counts:
        xs, ys, es = [], [], []
        for gs in grid_sizes:
            if (ns, gs) in agg:
                a = agg[(ns, gs)]
                xs.append(gs)
                ys.append(a['t_disc_mean'])
                es.append(a['t_disc_std'])
        if xs:
            color = SCOUT_COLORS.get(ns, 'gray')
            marker = SCOUT_MARKERS.get(ns, 'o')
            ax.errorbar(xs, ys, yerr=es, marker=marker, color=color,
                        linewidth=2, capsize=5, markersize=7,
                        label=f'{ns} scout{"s" if ns > 1 else ""}')

    ax.set_xlabel('Map Size [m]')
    ax.set_ylabel('Discovery Time [steps]')
    ax.set_title('Fire Discovery Time vs. Map Size', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(400, grid_sizes[-1] + 100)

    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'scout_eval_discovery.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  → {path}')
    plt.close(fig)


def plot_dwell_bars(agg, scout_counts, grid_sizes, out_dir):
    """
    Figure 4: Grouped bar chart — dwell % by map size, bars grouped by scout count.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    n_groups = len(grid_sizes)
    n_bars = len(scout_counts)
    bar_w = 0.8 / n_bars
    x_base = np.arange(n_groups)

    for k, ns in enumerate(scout_counts):
        vals, errs = [], []
        for gs in grid_sizes:
            if (ns, gs) in agg:
                vals.append(agg[(ns, gs)]['dwell_mean'])
                errs.append(agg[(ns, gs)]['dwell_std'])
            else:
                vals.append(0)
                errs.append(0)
        offset = (k - n_bars / 2 + 0.5) * bar_w
        color = SCOUT_COLORS.get(ns, 'gray')
        ax.bar(x_base + offset, vals, bar_w * 0.9, yerr=errs,
               color=color, alpha=0.85, capsize=3,
               label=f'{ns} scout{"s" if ns > 1 else ""}')

    ax.set_xticks(x_base)
    ax.set_xticklabels([f'{g}m' for g in grid_sizes])
    ax.set_xlabel('Map Size')
    ax.set_ylabel('Team Fire Visibility [%]')
    ax.set_title('Fire Visibility by Map Size and Scout Count', fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(framealpha=0.9)

    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'scout_eval_dwell_bars.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  → {path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to eval_scout.csv')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: same as CSV)')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.csv_path)
    os.makedirs(out_dir, exist_ok=True)

    rows = load_csv(args.csv_path)
    agg = aggregate(rows)

    scout_counts = sorted(set(r['n_scouts'] for r in rows))
    grid_sizes = sorted(set(r['map_m'] for r in rows))

    print(f"Loaded {len(rows)} episodes, {len(agg)} configs")
    print(f"  Scouts: {scout_counts}")
    print(f"  Maps:   {grid_sizes}")

    plot_lines(agg, scout_counts, grid_sizes, out_dir)
    plot_heatmaps(agg, scout_counts, grid_sizes, out_dir)
    plot_discovery(agg, scout_counts, grid_sizes, out_dir)
    plot_dwell_bars(agg, scout_counts, grid_sizes, out_dir)
    print("Done.")
