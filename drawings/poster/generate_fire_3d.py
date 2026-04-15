#!/usr/bin/env python3
"""
Generate 3D fire-spread + updraft visualisation for paper/poster.

Pipeline:
  1. Build fuel grid from OSM (reuses generate_osm_vs_fuel logic)
  2. Run standalone FireGrid simulation until fire is well-developed
  3. Compute updraft (vertical convection) field above fire
  4. Render a publication-quality 3D figure:
       - ground terrain coloured by fuel type
       - fire intensity overlay (red/orange)
       - updraft arrows / streamlines rising above the flames

Usage:
    python drawings/poster/generate_fire_3d.py
    python drawings/poster/generate_fire_3d.py --location "49.226870, 16.596895" --size 400
    python drawings/poster/generate_fire_3d.py --lat 49.2269 --lon 16.5969 --size 400 --steps 600
"""

import sys, os, argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – registers 3D projection
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# project root on path so we can import src.*
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.fire_grid import FireGrid

# Reuse fuel-grid builder from the sister script
from drawings.poster.generate_osm_vs_fuel import build_fuel_grid   # noqa

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_LAT = 49.226870
DEFAULT_LON = 16.596895
DEFAULT_SIZE = 200          # metres
DEFAULT_CELL = 2.0          # metres  (1 m is too fine for 400 m 3D)
DEFAULT_STEPS = 600         # fire sim steps
FIRE_DT = 0.1               # seconds per step

# Updraft model params (matching simulation.py)
AIRFLOW_H = 100.0           # metres – max height of fire influence
CONVECTION_GAIN = 8.0
PLUME_RADIUS_FACTOR = 2.0

OUTPUT_DIR = SCRIPT_DIR

# ── helpers ─────────────────────────────────────────────────────────────────

def _build_fire_grid(fuel, cell_size, dt=FIRE_DT):
    """Wrap raw fuel array into a FireGrid, copying fuel values."""
    H, W = fuel.shape
    fg = FireGrid(H=H, W=W, dt=dt)
    fg.F[:] = fuel

    # Burn rates (same logic as Environment.rasterize_terrain_layers)
    cell_area = cell_size * cell_size
    BURN_GRASS   = 0.01 / cell_area
    BURN_FOREST  = 0.0067 / cell_area
    BURN_BUILDING = 0.0015 / cell_area

    fg.fuel_burn_rate[:] = BURN_GRASS                              # default
    fg.fuel_burn_rate[fuel >= 0.75] = BURN_FOREST                  # forest
    fg.fuel_burn_rate[(fuel >= 0.85) & (fuel <= 0.95)] = BURN_BUILDING  # buildings
    fg.fuel_burn_rate[fuel < 0.05] = 0.0                            # water/roads

    # Spread rate
    PHYSICAL_SPREAD_SPEED = 0.1  # m/s (matches environment.py)
    fg.l_base[:] = PHYSICAL_SPREAD_SPEED / cell_size

    return fg


def _ignite_center(fg, cell_size, radius_m=15.0):
    """Ignite a patch around the grid centre (where there is fuel)."""
    ci, cj = fg.H // 2, fg.W // 2
    r_cells = int(radius_m / cell_size)
    ignited = 0
    for i in range(ci - r_cells, ci + r_cells + 1):
        for j in range(cj - r_cells, cj + r_cells + 1):
            if 0 <= i < fg.H and 0 <= j < fg.W and fg.F[i, j] > 0:
                fg.B[i, j] = True
                fg.I[i, j] = min(1.0, fg.F[i, j] * 0.5)
                ignited += 1
    return ignited


def _compute_updraft_columns(fg, cell_size, half, n_z=8, subsample=6):
    """
    Compute updraft columns **directly above burning cells**.

    Clusters burning cells into groups of *subsample × subsample*,
    takes their centroid, sums their intensities, and builds a vertical
    column of arrows there.

    Returns arrays: x, y, z, wz  (1-D, one entry per arrow).
    """
    origin_x = -half
    origin_y = -half

    burn_i, burn_j = np.where(fg.B & (fg.I > 0.05))
    if len(burn_i) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    fire_x = origin_x + (burn_j + 0.5) * cell_size
    fire_y = origin_y + (burn_i + 0.5) * cell_size
    fire_I = fg.I[burn_i, burn_j]

    # Debug: print fire centroid
    print(f"   Fire centroid: X={fire_x.mean():.1f}, Y={fire_y.mean():.1f}")
    print(f"   Fire X range: [{fire_x.min():.1f}, {fire_x.max():.1f}]")
    print(f"   Fire Y range: [{fire_y.min():.1f}, {fire_y.max():.1f}]")

    # ── cluster fire cells on a coarse grid ───────────────────────────
    block_i = burn_i // subsample
    block_j = burn_j // subsample

    blocks = {}
    for idx in range(len(burn_i)):
        key = (int(block_i[idx]), int(block_j[idx]))
        if key not in blocks:
            blocks[key] = {"sum_x": 0.0, "sum_y": 0.0, "sum_I": 0.0, "n": 0}
        blocks[key]["sum_x"] += fire_x[idx]
        blocks[key]["sum_y"] += fire_y[idx]
        blocks[key]["sum_I"] += fire_I[idx]
        blocks[key]["n"] += 1

    # ── build per-cluster columns ─────────────────────────────────────
    MAX_Z = 60.0  # metres – keep columns moderate height
    zs = np.linspace(0.5, MAX_Z, n_z)   # start from ground level

    all_x, all_y, all_z, all_w = [], [], [], []

    for blk in blocks.values():
        cx = blk["sum_x"] / blk["n"]
        cy = blk["sum_y"] / blk["n"]
        avg_I = blk["sum_I"] / blk["n"]

        for z in zs:
            z_norm = z / AIRFLOW_H
            height_taper = z_norm / 0.3 if z_norm < 0.3 else (1.0 - z_norm) / 0.7
            w = avg_I * CONVECTION_GAIN * height_taper
            all_x.append(cx)
            all_y.append(cy)
            all_z.append(z)
            all_w.append(w)

    up_x = np.array(all_x)
    up_y = np.array(all_y)
    print(f"   Updraft centroid: X={up_x.mean():.1f}, Y={up_y.mean():.1f}")

    return (up_x, np.array(all_y),
            np.array(all_z), np.array(all_w))


def _compute_updraft_at(x, y, z, fg, cell_size, half):
    """
    Compute total vertical updraft w [m/s] at world position (x, y, z)
    using the same Gaussian plume model as simulation.py.
    """
    if z >= AIRFLOW_H or z < 0:
        return 0.0

    origin_x, origin_y = -half, -half
    z_norm = z / AIRFLOW_H
    height_taper = z_norm / 0.3 if z_norm < 0.3 else (1.0 - z_norm) / 0.7

    plume_radius = cell_size * PLUME_RADIUS_FACTOR
    ci = int((y - origin_y) / cell_size)
    cj = int((x - origin_x) / cell_size)

    w_total = 0.0
    for di in range(-8, 9):
        for dj in range(-8, 9):
            i, j = ci + di, cj + dj
            if not (0 <= i < fg.H and 0 <= j < fg.W):
                continue
            intensity = fg.I[i, j]
            if intensity <= 0:
                continue
            fx = origin_x + (j + 0.5) * cell_size
            fy = origin_y + (i + 0.5) * cell_size
            r = np.hypot(x - fx, y - fy)
            w_total += (intensity * CONVECTION_GAIN * height_taper *
                        np.exp(-0.5 * (r / plume_radius) ** 2))
    return w_total


def _simulate_drone_flythrough(fg, cell_size, half, drone_alt=25.0,
                               drone_speed=8.0, y_path=None):
    """
    Simulate a quadcopter flying a straight line across the map.

    The drone flies at constant horizontal speed along X at altitude
    *drone_alt*.  It has a simple vertical-velocity model with inertia
    and a spring-like restoration towards *drone_alt* (mimicking a very
    weak altitude-hold that is overwhelmed by an updraft).

    Returns: xs, ys, zs  (1-D arrays – the 3-D trajectory)
    """
    if y_path is None:
        # Fly through the fire centroid Y
        burn_i = np.where(fg.B & (fg.I > 0.05))[0]
        if len(burn_i) > 0:
            y_path = -half + (np.mean(burn_i) + 0.5) * cell_size
        else:
            y_path = 0.0

    x_start = -half + 5
    x_end = half - 5
    n_pts = 600
    dt = (x_end - x_start) / (drone_speed * n_pts)  # time per step

    # Vertical dynamics parameters
    DRAG = 3.0          # 1/s – how fast vz decays (air drag + motor damping)
    RESTORE = 0.5       # 1/s² – spring constant toward target altitude
    UPDRAFT_COUPLING = 0.25  # fraction of updraft that actually accelerates drone

    xs, ys, zs = [x_start], [y_path], [drone_alt]
    x, z, vz = x_start, drone_alt, 0.0

    for _ in range(n_pts):
        w = _compute_updraft_at(x, y_path, z, fg, cell_size, half)
        # Force balance: updraft pushes up, drag opposes, spring restores
        az = UPDRAFT_COUPLING * w - DRAG * vz - RESTORE * (z - drone_alt)
        vz += az * dt
        z += vz * dt
        # Clamp: drone can't sink below target (motors hold baseline)
        if z < drone_alt:
            z = drone_alt
            vz = max(vz, 0.0)
        x += drone_speed * dt
        if x > x_end:
            break
        xs.append(x)
        ys.append(y_path)
        zs.append(z)

    return np.array(xs), np.array(ys), np.array(zs)


# ── render ──────────────────────────────────────────────────────────────────

def render_3d(fuel, fg, cell_size, size_m, output_path):
    """Render a 3D figure with terrain, fire, updraft arrows, and drone."""
    try:
        plt.style.use(["science", "ieee"])
    except Exception:
        plt.rcParams.update({
            "font.family": "serif", "font.size": 10,
            "axes.labelsize": 11, "axes.titlesize": 13,
        })

    half = size_m / 2
    H, W = fuel.shape
    origin_x, origin_y = -half, -half

    # ── 1. Ground surface coloured by fuel type ───────────────────────
    xs_c = origin_x + (np.arange(W) + 0.5) * cell_size
    ys_c = origin_y + (np.arange(H) + 0.5) * cell_size
    Xg, Yg = np.meshgrid(xs_c, ys_c)
    Zg = np.zeros_like(Xg)

    # Colour map for terrain (RGBA per cell)
    terrain_rgba = np.zeros((H, W, 4))
    mask_wr = fuel < 0.05
    terrain_rgba[mask_wr] = [0.23, 0.23, 0.23, 1.0]
    mask_gr = (fuel >= 0.05) & (fuel < 0.5)
    terrain_rgba[mask_gr] = [0.78, 0.85, 0.44, 1.0]
    mask_fo = (fuel >= 0.5) & (fuel < 0.85)
    terrain_rgba[mask_fo] = [0.29, 0.49, 0.35, 1.0]
    mask_bl = fuel >= 0.85
    terrain_rgba[mask_bl] = [0.55, 0.55, 0.55, 1.0]

    # ── overlay fire on terrain with orange-red ───────────────────────
    fire_mask = fg.B & (fg.I > 0.01)
    intensity = fg.I.copy()
    intensity[~fire_mask] = 0.0
    for i in range(H):
        for j in range(W):
            t = intensity[i, j]
            if t > 0.01:
                fire_col = np.array([1.0, 0.3 * (1.0 - t), 0.0, 1.0])
                terrain_rgba[i, j] = (1 - t) * terrain_rgba[i, j] + t * fire_col

    # ── 2. Compute updraft columns directly above fire ──────────────
    print("   Computing updraft columns above fire...")
    up_x, up_y, up_z, up_w = _compute_updraft_columns(
        fg, cell_size, half, n_z=10, subsample=3)

    # ── 3. Simulate quadcopter fly-through ────────────────────────────
    print("   Simulating quadcopter fly-through...")
    drone_x, drone_y, drone_z = _simulate_drone_flythrough(
        fg, cell_size, half, drone_alt=25.0, drone_speed=8.0)
    print(f"   Drone Z range: [{drone_z.min():.1f}, {drone_z.max():.1f}] m  "
          f"(target 25 m, peak deviation +{drone_z.max()-25:.1f} m)")

    # ── 4. Plot ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False  # use manual zorder (ground behind everything)

    # Ground surface — single call at lowest zorder so it never covers
    # fire/updraft/drone.  Push it slightly below z=0 to stay behind.
    Zg_low = np.full_like(Xg, -0.1)   # tiny negative so depth < everything
    ax.plot_surface(
        Xg, Yg, Zg_low,
        facecolors=terrain_rgba,
        rstride=1, cstride=1,
        shade=False, antialiased=False, zorder=0,
    )

    # Fire scatter on terrain
    if np.any(fire_mask):
        fi, fj = np.where(fire_mask)
        fx = origin_x + (fj + 0.5) * cell_size
        fy = origin_y + (fi + 0.5) * cell_size
        fz = np.full_like(fx, 0.3)
        f_int = fg.I[fi, fj]
        ax.scatter(
            fx, fy, fz,
            c=f_int, cmap=plt.cm.hot, norm=Normalize(vmin=0, vmax=1.0),
            s=8, alpha=0.9, edgecolors="none", zorder=2,
        )

    # ── Updraft pillars ───────────────────────────────────────────────
    if len(up_w) > 0 and up_w.max() > 0:
        seen = {}
        for i in range(len(up_x)):
            key = (round(up_x[i], 2), round(up_y[i], 2))
            if key not in seen:
                seen[key] = {"zs": [], "ws": []}
            seen[key]["zs"].append(up_z[i])
            seen[key]["ws"].append(up_w[i])

        max_w_global = up_w.max()
        cmap_up = plt.cm.YlOrRd
        norm_up = Normalize(vmin=0, vmax=max_w_global)

        for (cx, cy), col_data in seen.items():
            zs_col = np.array(col_data["zs"])
            ws_col = np.array(col_data["ws"])
            sort_idx = np.argsort(zs_col)
            zs_col = zs_col[sort_idx]
            ws_col = ws_col[sort_idx]

            for k in range(len(zs_col) - 1):
                z0, z1 = zs_col[k], zs_col[k + 1]
                w_avg = (ws_col[k] + ws_col[k + 1]) / 2
                color = cmap_up(norm_up(w_avg))
                lw = 1.5 + 3.0 * (w_avg / max_w_global)
                ax.plot([cx, cx], [cy, cy], [z0, z1],
                        color=color, linewidth=lw, alpha=0.8, zorder=3)

            # Arrow at top
            top_z, top_w = zs_col[-1], ws_col[-1]
            arrow_h = 3.0 + 5.0 * (top_w / max_w_global)
            ax.quiver(cx, cy, top_z, 0, 0, arrow_h,
                      color=cmap_up(norm_up(top_w)), alpha=0.85,
                      arrow_length_ratio=0.4, linewidth=1.5, zorder=4)

        # Ground-level dots to anchor updraft columns visually
        base_xs = [k[0] for k in seen.keys()]
        base_ys = [k[1] for k in seen.keys()]
        ax.scatter(base_xs, base_ys, [0.5]*len(base_xs),
                   marker=".", s=15, c="orange", alpha=0.7,
                   edgecolors="none", zorder=2)

    # ── Drone trajectory ──────────────────────────────────────────────
    # Target altitude reference line (dashed, thin)
    ax.plot([drone_x[0], drone_x[-1]],
            [drone_y[0], drone_y[-1]],
            [25.0, 25.0],
            color="steelblue", linestyle="--", linewidth=0.8,
            alpha=0.6, zorder=5, label="_nolegend_")

    # Actual trajectory (solid, coloured by altitude deviation)
    # Draw segment-by-segment coloured by altitude deviation
    alt_dev = drone_z - 25.0
    max_dev = max(alt_dev.max(), 1.0)
    for k in range(len(drone_x) - 1):
        frac = np.clip(alt_dev[k] / max_dev, 0, 1)
        # blue→red gradient: more deviation = more red
        col = (frac, 0.2, 1.0 - frac, 1.0)
        ax.plot(drone_x[k:k+2], drone_y[k:k+2], drone_z[k:k+2],
                color=col, linewidth=2.0, zorder=6)

    # Drone icon at a few positions (small marker)
    n_icons = 5
    idx_icons = np.linspace(0, len(drone_x) - 1, n_icons, dtype=int)
    ax.scatter(drone_x[idx_icons], drone_y[idx_icons], drone_z[idx_icons],
               marker="o", s=40, c="dodgerblue", edgecolors="navy",
               linewidths=0.8, zorder=7)

    # Vertical drop-lines from drone to terrain (shows altitude visually)
    for idx in idx_icons:
        ax.plot([drone_x[idx], drone_x[idx]],
                [drone_y[idx], drone_y[idx]],
                [0, drone_z[idx]],
                color="steelblue", linewidth=0.5, alpha=0.4, zorder=5)

    # ── axes & view ───────────────────────────────────────────────────
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    z_max = max(70, drone_z.max() + 10) if len(drone_z) > 0 else 70
    ax.set_zlim(0, z_max)
    ax.set_title("Fire spread with thermal updrafts", pad=12)
    ax.view_init(elev=35, azim=-140)   # look from opposite corner, higher up

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_items = [
        Patch(facecolor="#C8D96F", label="Grass"),
        Patch(facecolor="#4A7C59", label="Forest"),
        Patch(facecolor="#8C8C8C", label="Buildings"),
        Patch(facecolor="#3A3A3A", label="Road / Water"),
        Patch(facecolor="#FF4400", label="Active fire"),
        Line2D([0], [0], color="#FF6600", marker="^", linestyle="None",
               markersize=7, label="Updraft"),
        Line2D([0], [0], color="dodgerblue", marker="o", linewidth=2,
               markersize=5, label="Quadcopter (no ctrl)"),
        Line2D([0], [0], color="steelblue", linestyle="--", linewidth=0.8,
               label="Target altitude"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=6.5,
              framealpha=0.85)

    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="png")
    print(f"✅ Saved: {output_path}")

    base, _ = os.path.splitext(output_path)
    pdf_path = base + ".pdf"
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight",
                    format="pdf", backend="pdf")
    print(f"✅ Saved: {pdf_path}")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3D fire-spread + updraft visualisation for paper."
    )
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--location", type=str, default=None,
                        help="'lat, lon' string or place name to geocode")
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE,
                        help="Side length in metres (default: 400)")
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL,
                        help="Cell size in metres (default: 2)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Number of fire simulation steps (default: 600)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # ── resolve coordinates ───────────────────────────────────────────
    if args.location:
        # Accept "lat, lon" directly or a place name
        parts = args.location.split(",")
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
            except ValueError:
                from drawings.poster.generate_osm_vs_fuel import geocode_location
                lat, lon = geocode_location(args.location)
        else:
            from drawings.poster.generate_osm_vs_fuel import geocode_location
            lat, lon = geocode_location(args.location)
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        lat, lon = DEFAULT_LAT, DEFAULT_LON

    size_m = args.size
    cell_size = args.cell
    n_steps = args.steps
    output_path = args.output or os.path.join(OUTPUT_DIR, "fire_3d.png")

    print(f"🗺️  Centre: ({lat:.6f}, {lon:.6f})")
    print(f"📐 Area: {size_m}×{size_m} m, cell: {cell_size} m")
    print(f"🔥 Sim steps: {n_steps} (dt={FIRE_DT}s → {n_steps*FIRE_DT:.0f}s)")

    # ── Step 1: fuel grid ─────────────────────────────────────────────
    print("\n── Step 1: Building fuel grid from OSM ──")
    fuel, fuel_ext = build_fuel_grid(lat, lon, size_m, cell_size)
    print(f"   Fuel grid: {fuel.shape}")

    # ── Step 2: fire simulation ───────────────────────────────────────
    print("\n── Step 2: Running fire simulation ──")
    fg = _build_fire_grid(fuel, cell_size)
    n_ign = _ignite_center(fg, cell_size, radius_m=15.0)
    print(f"   Ignited {n_ign} cells at grid centre")

    best_step = 0
    best_burn = 0
    for step in range(n_steps):
        fg.step()
        n_burn = int(np.sum(fg.B))
        if n_burn > best_burn:
            best_burn = n_burn
            best_step = step
        if step % 100 == 0 or step == n_steps - 1:
            stats = fg.get_stats()
            print(f"   step {step:>4d}: burning={stats['burning_cells']:>5d} "
                  f"({stats['burn_percentage']:.1f}%)  "
                  f"avg_I={stats['avg_intensity']:.3f}")

    print(f"   Peak fire at step {best_step} ({best_burn} cells)")

    # ── Step 3: render ────────────────────────────────────────────────
    print("\n── Step 3: Rendering 3D figure ──")
    render_3d(fuel, fg, cell_size, size_m, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
