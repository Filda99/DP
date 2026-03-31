"""
plot_survivor_trajectories.py — Vizualizace posledních N celých epizod commandera
==================================================================================
Načte .npz soubory z output/survivor_trajectories/ a vykreslí:
  - Top-down XY trajektorie (každá epizoda jiná barva)
  - Výška (Z) v čase
  - Celkový reward a délka epizody v názvu

Použití:
  python tools/plot_survivor_trajectories.py           # posledních 10
  python tools/plot_survivor_trajectories.py --n 5     # posledních 5
  python tools/plot_survivor_trajectories.py --all     # všechny
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ─── Cesta k výstupní složce ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ_DIR     = os.path.join(PROJECT_ROOT, "output", "survivor_trajectories")
GRID_BOUND   = 1000.0   # map_bounds = grid_size_m / 2


def load_trajectories(n: int | None) -> list[dict]:
    """Načte a seřadí .npz soubory, vrátí posledních n."""
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "ep_*.npz")))
    if not files:
        print(f"❌  Žádné soubory v {TRAJ_DIR}")
        print("    Trénink ještě nevygeneroval žádnou celou epizodu.")
        sys.exit(0)

    if n is not None:
        files = files[-n:]

    trajs = []
    for f in files:
        d = np.load(f)
        trajs.append({
            "file":        os.path.basename(f),
            "positions":   d["positions"],        # [steps, 3]
            "total_reward": float(d["total_reward"]),
            "max_steps":   int(d["max_steps"]),
            "steps":       len(d["positions"]),
        })
    return trajs


def plot(trajs: list[dict], save_path: str | None = None):
    n = len(trajs)
    colors = cm.tab10(np.linspace(0, 1, n)) if n <= 10 else cm.tab20(np.linspace(0, 1, n))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Survivor Trajectories — posledních {n} celých epizod", fontsize=14, fontweight="bold")

    ax_xy = axes[0]
    ax_z  = axes[1]

    # ── Hranice mapy ──────────────────────────────────────────────────────────
    b = GRID_BOUND
    for ax in (ax_xy,):
        ax.axhline( b, color="red", linewidth=0.8, linestyle=":", alpha=0.6, label=f"±{b:.0f}m hranice")
        ax.axhline(-b, color="red", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axvline( b, color="red", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axvline(-b, color="red", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)

    # ── Trajektorie ───────────────────────────────────────────────────────────
    for i, traj in enumerate(trajs):
        pos = traj["positions"]   # [steps, 3]
        ep  = traj["file"].replace("ep_", "").replace(".npz", "")
        lbl = f"ep {int(ep):,}  R={traj['total_reward']:.0f}"
        c   = colors[i]

        # XY top-down
        ax_xy.plot(pos[:, 0], pos[:, 1], color=c, linewidth=1.2, alpha=0.85, label=lbl)
        ax_xy.scatter(pos[0, 0],  pos[0, 1],  color=c, marker="o", s=50, zorder=5)   # start
        ax_xy.scatter(pos[-1, 0], pos[-1, 1], color=c, marker="*", s=80, zorder=5)   # end

        # Z (výška) v čase
        ax_z.plot(pos[:, 2], color=c, linewidth=1.0, alpha=0.85, label=lbl)

    # ── Dekorace ──────────────────────────────────────────────────────────────
    ax_xy.set_title("Trajektorie XY (pohled shora)")
    ax_xy.set_xlabel("X — East [m]")
    ax_xy.set_ylabel("Y — North [m]")
    ax_xy.set_xlim(-b * 1.08, b * 1.08)
    ax_xy.set_ylim(-b * 1.08, b * 1.08)
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="upper left", fontsize=7, ncol=1)

    ax_z.set_title("Výška (Z / Altitude) v čase")
    ax_z.set_xlabel("Krok epizody")
    ax_z.set_ylabel("Výška [m]")
    ax_z.axhline(40,  color="green",   linestyle="--", alpha=0.5, linewidth=0.8, label="Min ideál 40m")
    ax_z.axhline(150, color="orange",  linestyle="--", alpha=0.5, linewidth=0.8, label="Max ideál 150m")
    ax_z.axhline(450, color="darkred", linestyle=":",  alpha=0.5, linewidth=0.8, label="Ceiling 450m")
    ax_z.fill_between(range(trajs[0]["max_steps"]), 40, 150, color="green", alpha=0.05)
    ax_z.grid(True, alpha=0.25)
    ax_z.legend(loc="upper right", fontsize=7)

    plt.tight_layout()

    out = save_path or os.path.join(PROJECT_ROOT, "output", "survivor_trajectories_plot.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"✅  Graf uložen: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vizualizace survivor trajektorií")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--n",   type=int, default=10, help="Počet posledních epizod (default: 10)")
    group.add_argument("--all", action="store_true",  help="Zobraz všechny dostupné epizody")
    parser.add_argument("--save", type=str, default=None, help="Cesta pro uložení PNG (volitelné)")
    args = parser.parse_args()

    n = None if args.all else args.n
    trajs = load_trajectories(n)
    print(f"📂  Načteno {len(trajs)} trajektorií z {TRAJ_DIR}")
    for t in trajs:
        print(f"    {t['file']}  →  {t['steps']} kroků  |  reward: {t['total_reward']:.1f}")

    plot(trajs, save_path=args.save)
