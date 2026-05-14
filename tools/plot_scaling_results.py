"""
plot_scaling_results.py — Publication-ready plots for scaling evaluation
========================================================================
Reads results/eval_scaling/results.csv and generates compact figures
suitable for a diploma thesis.

Usage:
    python tools/plot_scaling_results.py
"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(PROJECT, "results", "eval_scaling", "results.csv")
OUT = os.path.join(PROJECT, "results", "eval_scaling", "plots")
os.makedirs(OUT, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, 5))

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df = df[df["error"].isna() | (df["error"] == "")].copy()
df["success"] = df["time_full_suppression"] > 0
print(f"Loaded {len(df)} successful episodes")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Success Rate heatmap — Scouts × FW
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 4))
pivot = df.pivot_table(values="success", index="n_scouts", columns="n_fw",
                       aggfunc="mean") * 100
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns.astype(int))
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index.astype(int))
ax.set_xlabel("Number of fixed-wing aircraft")
ax.set_ylabel("Number of scout UAVs")
ax.set_title("Fire suppression success rate [%]")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        color = "white" if v < 50 else "black"
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
fig.colorbar(im, ax=ax, label="%", shrink=0.8)
fig.savefig(os.path.join(OUT, "fig1_success_heatmap_scouts_fw.png"))
fig.savefig(os.path.join(OUT, "fig1_success_heatmap_scouts_fw.pdf"))
plt.close(fig)
print("  fig1 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Success Rate by FW count — one line per map size
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4))
for i, ms in enumerate([700, 1100, 2000, 4000, 6000]):
    sub = df[df["map_size_m"] == ms]
    rates = sub.groupby("n_fw")["success"].mean() * 100
    ax.plot(rates.index, rates.values, "o-", color=COLORS[i],
            label=f"{ms} m", linewidth=2, markersize=5)
ax.set_xlabel("Number of fixed-wing aircraft")
ax.set_ylabel("Success rate [%]")
ax.set_title("Suppression success vs. team size by map area")
ax.set_xticks([1, 3, 5, 7, 10])
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.legend(title="Map size", loc="lower right")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT, "fig2_success_vs_fw_by_map.png"))
fig.savefig(os.path.join(OUT, "fig2_success_vs_fw_by_map.pdf"))
plt.close(fig)
print("  fig2 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Success Rate heatmap — FW × Map Size
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4))
pivot3 = df.pivot_table(values="success", index="n_fw", columns="map_size_m",
                        aggfunc="mean") * 100
im = ax.imshow(pivot3.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(pivot3.columns)))
ax.set_xticklabels([f"{int(c)}" for c in pivot3.columns])
ax.set_yticks(range(len(pivot3.index)))
ax.set_yticklabels(pivot3.index.astype(int))
ax.set_xlabel("Map size [m]")
ax.set_ylabel("Number of fixed-wing aircraft")
ax.set_title("Suppression success rate [%]")
for i in range(len(pivot3.index)):
    for j in range(len(pivot3.columns)):
        v = pivot3.values[i, j]
        color = "white" if v < 50 else "black"
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
fig.colorbar(im, ax=ax, label="%", shrink=0.8)
fig.savefig(os.path.join(OUT, "fig3_success_heatmap_fw_map.png"))
fig.savefig(os.path.join(OUT, "fig3_success_heatmap_fw_map.pdf"))
plt.close(fig)
print("  fig3 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Suppression time vs FW count (successful episodes only)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4))
succ = df[df["success"]].copy()
for i, ms in enumerate([700, 1100, 2000, 4000, 6000]):
    sub = succ[succ["map_size_m"] == ms]
    times = sub.groupby("n_fw")["time_full_suppression"].agg(["mean", "std"])
    ax.errorbar(times.index, times["mean"], yerr=times["std"],
                fmt="o-", color=COLORS[i], label=f"{ms} m",
                linewidth=2, markersize=5, capsize=3)
ax.set_xlabel("Number of fixed-wing aircraft")
ax.set_ylabel("Steps to full suppression")
ax.set_title("Time to suppress fire (successful episodes)")
ax.set_xticks([1, 3, 5, 7, 10])
ax.legend(title="Map size", loc="upper right")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT, "fig4_suppression_time_vs_fw.png"))
fig.savefig(os.path.join(OUT, "fig4_suppression_time_vs_fw.pdf"))
plt.close(fig)
print("  fig4 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Total burned area vs FW count by map size
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4))
for i, ms in enumerate([700, 1100, 2000, 4000, 6000]):
    sub = df[df["map_size_m"] == ms]
    burned = sub.groupby("n_fw")["total_burned_cells"].agg(["mean", "std"])
    ax.errorbar(burned.index, burned["mean"], yerr=burned["std"],
                fmt="s-", color=COLORS[i], label=f"{ms} m",
                linewidth=2, markersize=5, capsize=3)
ax.set_xlabel("Number of fixed-wing aircraft")
ax.set_ylabel("Total burned cells (5×5 m each)")
ax.set_title("Fire damage vs. team size")
ax.set_xticks([1, 3, 5, 7, 10])
ax.legend(title="Map size", loc="upper right")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT, "fig5_burned_area_vs_fw.png"))
fig.savefig(os.path.join(OUT, "fig5_burned_area_vs_fw.pdf"))
plt.close(fig)
print("  fig5 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Compact 2×2 overview — the "money figure"
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# (a) Success heatmap: FW × Map
ax = axes[0, 0]
im = ax.imshow(pivot3.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(pivot3.columns)))
ax.set_xticklabels([f"{int(c)}" for c in pivot3.columns])
ax.set_yticks(range(len(pivot3.index)))
ax.set_yticklabels(pivot3.index.astype(int))
ax.set_xlabel("Map size [m]")
ax.set_ylabel("Fixed-wing count")
ax.set_title("(a) Suppression success rate [%]")
for ii in range(len(pivot3.index)):
    for jj in range(len(pivot3.columns)):
        v = pivot3.values[ii, jj]
        color = "white" if v < 50 else "black"
        ax.text(jj, ii, f"{v:.0f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

# (b) Success lines by FW per map
ax = axes[0, 1]
for i, ms in enumerate([700, 1100, 2000, 4000, 6000]):
    sub = df[df["map_size_m"] == ms]
    rates = sub.groupby("n_fw")["success"].mean() * 100
    ax.plot(rates.index, rates.values, "o-", color=COLORS[i],
            label=f"{ms} m", linewidth=2, markersize=4)
ax.set_xlabel("Fixed-wing count")
ax.set_ylabel("Success rate [%]")
ax.set_title("(b) Success rate by team size")
ax.set_xticks([1, 3, 5, 7, 10])
ax.set_ylim(0, 105)
ax.legend(title="Map size", fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# (c) Suppression time
ax = axes[1, 0]
for i, ms in enumerate([700, 1100, 2000, 4000, 6000]):
    sub = succ[succ["map_size_m"] == ms]
    times = sub.groupby("n_fw")["time_full_suppression"].mean()
    ax.plot(times.index, times.values, "o-", color=COLORS[i],
            label=f"{ms} m", linewidth=2, markersize=4)
ax.set_xlabel("Fixed-wing count")
ax.set_ylabel("Steps to suppression")
ax.set_title("(c) Time to suppress (successful only)")
ax.set_xticks([1, 3, 5, 7, 10])
ax.legend(title="Map size", fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# (d) Burned area
ax = axes[1, 1]
for i, ms in enumerate([700, 1100, 2000, 4000, 6000]):
    sub = df[df["map_size_m"] == ms]
    burned = sub.groupby("n_fw")["total_burned_cells"].mean()
    ax.plot(burned.index, burned.values, "s-", color=COLORS[i],
            label=f"{ms} m", linewidth=2, markersize=4)
ax.set_xlabel("Fixed-wing count")
ax.set_ylabel("Burned cells (5×5 m)")
ax.set_title("(d) Total fire damage")
ax.set_xticks([1, 3, 5, 7, 10])
ax.legend(title="Map size", fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

fig.suptitle("Multi-Agent Firefighting: Scaling Evaluation (4949 episodes)", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_overview_2x2.png"))
fig.savefig(os.path.join(OUT, "fig6_overview_2x2.pdf"))
plt.close(fig)
print("  fig6 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Scout scaling effect — success by n_scouts, fixed FW counts
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# (a) Success vs scouts, lines per FW count
for i, nf in enumerate([1, 3, 5, 7, 10]):
    sub = df[df["n_fw"] == nf]
    rates = sub.groupby("n_scouts")["success"].mean() * 100
    ax1.plot(rates.index, rates.values, "o-", color=COLORS[i],
             label=f"{nf} FW", linewidth=2, markersize=5)
ax1.set_xlabel("Number of scout UAVs")
ax1.set_ylabel("Success rate [%]")
ax1.set_title("(a) Effect of scout count on suppression")
ax1.set_xticks([1, 3, 5, 7, 10])
ax1.set_ylim(0, 105)
ax1.legend(title="FW count", loc="lower left")
ax1.grid(alpha=0.3)

# (b) Scout deaths & altitude vs n_scouts
ax2b = ax2.twinx()
alt_data = df.groupby("n_scouts")["scout_mean_altitude_m"].mean()
death_data = df.groupby("n_scouts")["scout_deaths"].mean()
l1, = ax2.plot(death_data.index, death_data.values, "D-", color="red",
               linewidth=2, markersize=6, label="Deaths/episode")
l2, = ax2b.plot(alt_data.index, alt_data.values, "^-", color="blue",
                linewidth=2, markersize=6, label="Mean altitude")
ax2.set_xlabel("Number of scout UAVs")
ax2.set_ylabel("Scout deaths per episode", color="red")
ax2b.set_ylabel("Mean scout altitude [m]", color="blue")
ax2.set_title("(b) Scout survival & altitude")
ax2.set_xticks([1, 3, 5, 7, 10])
ax2.legend(handles=[l1, l2], loc="center right")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_scout_scaling.png"))
fig.savefig(os.path.join(OUT, "fig7_scout_scaling.pdf"))
plt.close(fig)
print("  fig7 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Water efficiency
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# (a) Water accuracy vs FW count by map
for i, ms in enumerate([700, 1100, 2000, 4000, 6000]):
    sub = df[df["map_size_m"] == ms]
    wacc = sub.groupby("n_fw")["water_accuracy_pct"].mean()
    ax1.plot(wacc.index, wacc.values, "o-", color=COLORS[i],
             label=f"{ms} m", linewidth=2, markersize=4)
ax1.set_xlabel("Fixed-wing count")
ax1.set_ylabel("Water accuracy [%]")
ax1.set_title("(a) Water drop accuracy")
ax1.set_xticks([1, 3, 5, 7, 10])
ax1.set_ylim(0, 100)
ax1.legend(title="Map size", fontsize=8)
ax1.grid(alpha=0.3)

# (b) Drop distance & altitude vs FW count
dist_data = df.groupby("n_fw")["fw_mean_drop_distance_m"].mean()
alt_data = df.groupby("n_fw")["fw_mean_drop_altitude_m"].mean()
ax2.bar(np.array([1, 3, 5, 7, 10]) - 0.3, dist_data.values, width=0.6,
        label="Distance to fire", color=COLORS[1], alpha=0.8)
ax2.bar(np.array([1, 3, 5, 7, 10]) + 0.3, alt_data.values, width=0.6,
        label="Drop altitude", color=COLORS[3], alpha=0.8)
ax2.set_xlabel("Fixed-wing count")
ax2.set_ylabel("Distance / Altitude [m]")
ax2.set_title("(b) Water drop position")
ax2.set_xticks([1, 3, 5, 7, 10])
ax2.legend()
ax2.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_water_efficiency.png"))
fig.savefig(os.path.join(OUT, "fig8_water_efficiency.pdf"))
plt.close(fig)
print("  fig8 done")


print(f"\nAll figures saved to: {OUT}")
