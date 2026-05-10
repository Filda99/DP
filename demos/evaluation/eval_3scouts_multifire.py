"""
eval_3scouts_multifire.py
─────────────────────────
Evaluation: 3 scouts + 1 FW commander on 3-4 fires.
Runs 5 seeds, produces GIF + PNG analysis for each.

Usage:
    python demos/evaluation/eval_3scouts_multifire.py
"""

import torch
import numpy as np
import os, sys, glob, argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import imageio
import io, tqdm
from PIL import Image

import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor
from src.commander_control import CommanderController

# ── Config ──────────────────────────────────────────────────
MODEL_SCOUT = os.path.join(
    project_root,
    "results/TrainingQuad/050626_TrainingMultipleScoutsOnMultipleFires/scout_b0120.pt")
MODEL_CMDR = os.path.join(
    project_root,
    "results/TrainingTogether/09_finalTraining/2on1/01/cmdr_b0180.pt")

N_QUADS       = 3
N_FIXED       = 1
N_FIRES_RANGE = (3, 4)
MAX_STEPS     = 1000
GRID_SIZE     = 1200.0
GIF_EVERY     = 3
GIF_FPS       = 15
SEEDS         = [42, 111, 256, 777, 1234]

USE_OSM    = True
OSM_LAT    = 49.35
OSM_LON    = 16.42
OSM_CACHE  = os.path.join(project_root, "data")

WAYPOINT_RANGE   = 200.0
WAYPOINT_STEPS   = 30
WP_REACHED_DIST  = 30.0

OUTPUT_DIR = os.path.join(project_root, "output", "eval_3scouts_multifire")

# ── Terrain colours ─────────────────────────────────────────
CLR_GRASS    = '#f0eed8'
CLR_BUILDING = '#b0b0b0'
CLR_FOREST   = '#6abf69'
CLR_WATER    = '#7ec8e3'
CLR_SCOUT    = '#00bfff'
CLR_CMDR     = '#ff3333'


# ── OSM helpers (copied from demo_both) ────────────────────
def _load_terrain_gdfs():
    categories = ['building', 'landuse', 'natural', 'waterway']
    prefix = f"{OSM_LAT}_{OSM_LON}"
    gdfs_raw = {}
    for cat in categories:
        matches = glob.glob(os.path.join(OSM_CACHE, f"{prefix}_{cat}_*.gpkg"))
        if matches:
            try:
                gdfs_raw[cat] = gpd.read_file(matches[0])
            except Exception:
                pass
    if not gdfs_raw:
        return {}
    utm_zone = int((OSM_LON + 180) / 6) + 1
    utm_crs = f"EPSG:326{utm_zone:02d}" if OSM_LAT >= 0 else f"EPSG:327{utm_zone:02d}"
    center_proj = gpd.GeoSeries([Point(OSM_LON, OSM_LAT)],
                                crs='EPSG:4326').to_crs(utm_crs).iloc[0]
    result = {}
    for cat, gdf in gdfs_raw.items():
        if gdf.empty:
            continue
        gdf_p = gdf.to_crs(utm_crs)
        gdf_p = gdf_p[gdf_p.distance(center_proj) <= GRID_SIZE / 2.0]
        if len(gdf_p) == 0:
            continue
        gdf_p['geometry'] = gdf_p.translate(xoff=-center_proj.x, yoff=-center_proj.y)
        result[cat] = gdf_p
    return result


def _classify_features(gdfs):
    water, buildings, forests = [], [], []
    for gdf in gdfs.values():
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            nat = row.get('natural', None)
            ww  = row.get('waterway', None)
            lu  = row.get('landuse', None)
            bld = row.get('building', None)
            if nat in ('water','wetland') or ww in ('river','canal','dock','riverbank') or lu in ('reservoir','basin'):
                water.append(geom)
            elif (bld is not None and bld != 'no') or lu in ('residential','commercial','industrial','retail'):
                buildings.append(geom)
            elif lu in ('forest','orchard','vineyard','wood') or nat in ('wood','scrub','heath'):
                forests.append(geom)
    return water, buildings, forests


def _geom_patches(geom, **kw):
    out = []
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out.extend(_geom_patches(p, **kw))
    elif isinstance(geom, Polygon):
        xs, ys = geom.exterior.coords.xy
        out.append(plt.Polygon(list(zip(xs, ys)), closed=True, **kw))
    return out


def _build_terrain_collections(water, buildings, forests):
    colls = []
    fp = [p for g in forests for p in _geom_patches(g)]
    if fp:
        colls.append(PatchCollection(fp, facecolor=CLR_FOREST, edgecolor='none',
                                     alpha=0.7, zorder=1))
    bp = [p for g in buildings for p in _geom_patches(g)]
    if bp:
        colls.append(PatchCollection(bp, facecolor=CLR_BUILDING, edgecolor='#999',
                                     linewidth=0.3, alpha=0.85, zorder=2))
    wp = [p for g in water for p in _geom_patches(g)]
    if wp:
        colls.append(PatchCollection(wp, facecolor=CLR_WATER, edgecolor='none',
                                     alpha=0.85, zorder=3))
    return colls


def _add_terrain_to_ax(ax, terrain_collections):
    for coll in terrain_collections:
        new_p = [plt.Polygon(p.vertices, closed=True) for p in coll.get_paths()]
        alpha = coll.get_alpha()
        alpha = alpha[0] if hasattr(alpha, '__len__') else alpha
        zo = coll.get_zorder()
        zo = zo[0] if hasattr(zo, '__len__') else zo
        nc = PatchCollection(new_p, facecolor=coll.get_facecolor(),
                             edgecolor=coll.get_edgecolor(),
                             linewidth=coll.get_linewidth(),
                             alpha=alpha, zorder=zo)
        ax.add_collection(nc)


# ── Models ──────────────────────────────────────────────────
def _load_models(device):
    env_tmp = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                           grid_size_m=GRID_SIZE, max_steps=MAX_STEPS,
                           n_fires_range=N_FIRES_RANGE)
    scout_self_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0]
    fixed_self_dim = env_tmp.observation_space(env_tmp.fixed_agents[0])["self_state"].shape[0]
    del env_tmp

    scout = ScoutActor(self_state_dim=scout_self_dim, msg_dim=5,
                       hidden_dim=128).to(device)
    cmdr  = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=5,
                           action_dim=4, hidden_dim=64).to(device)

    for path, name, model in [(MODEL_SCOUT, "Scout", scout),
                               (MODEL_CMDR, "Commander", cmdr)]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device),
                                  strict=False)
            print(f"  ✅ {name}: {os.path.basename(path)}")
        else:
            print(f"  ⚠️  {name}: NOT FOUND ({path})")

    scout.eval(); cmdr.eval()
    return scout, cmdr


# ── Frame renderer ──────────────────────────────────────────
def _render_frame(step, fire_map, b,
                  q_paths, q_positions, q_alive_map,
                  f_path_x, f_path_y, f_pos, f_water_pct,
                  refill_pos,
                  local_map_np,
                  total_reward_q, total_reward_f,
                  fire_seen_sum, f_alive, n_quads,
                  terrain_collections=None):

    fig = plt.figure(figsize=(12, 6), facecolor='white')
    gs  = gridspec.GridSpec(2, 2, width_ratios=[2, 1], hspace=0.35, wspace=0.25,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Panel 1: Global map ─────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(CLR_GRASS)
    if terrain_collections:
        _add_terrain_to_ax(ax_map, terrain_collections)

    extent = [-b, b, -b, b]
    fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
    ax_map.imshow(fire_masked, extent=extent, origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.75, zorder=4)

    if refill_pos is not None:
        circle = plt.Circle((refill_pos[0], refill_pos[1]), 150,
                            color='deepskyblue', fill=False, linestyle='--',
                            linewidth=1.5, alpha=0.6, zorder=5)
        ax_map.add_patch(circle)
        ax_map.text(refill_pos[0], refill_pos[1] + 60, "REFILL",
                    color='deepskyblue', fontsize=8, ha='center',
                    fontweight='bold', alpha=0.8)

    # Trails
    for q_name, paths in q_paths.items():
        if len(paths["x"]) > 1:
            ax_map.plot(paths["x"], paths["y"], color=CLR_SCOUT, alpha=0.4,
                        linewidth=1.2, linestyle=':', zorder=5)
    if len(f_path_x) > 1:
        ax_map.plot(f_path_x, f_path_y, color=CLR_CMDR, alpha=0.4,
                    linewidth=1.2, linestyle=':', zorder=5)

    # Agent positions
    for q_name in q_positions:
        q_pos = q_positions[q_name]
        if q_alive_map.get(q_name, False) and q_pos is not None:
            q_alt = q_pos[2]
            q_fov = max(10.0, q_alt * 1.5)
            rect = plt.Rectangle((q_pos[0] - q_fov/2, q_pos[1] - q_fov/2),
                                 q_fov, q_fov,
                                 fill=False, edgecolor=CLR_SCOUT, linewidth=1.0,
                                 alpha=0.7, zorder=6)
            ax_map.add_patch(rect)
            ax_map.scatter(q_pos[0], q_pos[1], c=CLR_SCOUT, s=100, marker='^',
                           edgecolors='black', linewidths=0.6, zorder=7)

    if f_alive and f_pos is not None:
        ax_map.scatter(f_pos[0], f_pos[1], c=CLR_CMDR, s=130, marker='>',
                       edgecolors='black', linewidths=0.6, zorder=7)

    ax_map.set_xlim(-b, b); ax_map.set_ylim(-b, b)
    ax_map.set_aspect('equal')
    ax_map.set_title(f"Step {step:04d}  |  3 Scouts + 1 FW  |  Multi-fire",
                     fontsize=11, fontweight='bold')
    ax_map.set_xlabel('X [m]', fontsize=8)
    ax_map.set_ylabel('Y [m]', fontsize=8)
    ax_map.tick_params(labelsize=7)

    leg_h = []
    if terrain_collections:
        leg_h.append(mpatches.Patch(color=CLR_FOREST,   label='Forest'))
        leg_h.append(mpatches.Patch(color=CLR_BUILDING, label='Building'))
        leg_h.append(mpatches.Patch(color=CLR_WATER,    label='Water'))
    leg_h.append(plt.Line2D([0],[0], marker='^', color='w',
                            markerfacecolor=CLR_SCOUT, markersize=7, label='Scout'))
    leg_h.append(plt.Line2D([0],[0], marker='>', color='w',
                            markerfacecolor=CLR_CMDR, markersize=7, label='Commander'))
    ax_map.legend(handles=leg_h, loc='upper right', fontsize=6.5,
                  framealpha=0.85, edgecolor='#ccc')

    # ── Panel 2: Scout camera ──────────────────────────────
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_cam.set_facecolor('#fafafa')
    if local_map_np is not None:
        ax_cam.imshow(local_map_np, origin='lower', cmap='YlOrRd',
                      vmin=0, vmax=1.0)
    ax_cam.set_title("Best scout camera (32x32)", fontsize=9, fontweight='bold')
    ax_cam.set_xticks([]); ax_cam.set_yticks([])

    # ── Panel 3: Stats ─────────────────────────────────────
    ax_st = fig.add_subplot(gs[1, 1])
    ax_st.set_facecolor('#fafafa')
    ax_st.set_xticks([]); ax_st.set_yticks([])
    ax_st.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.08,
                    color='#dddddd', transform=ax_st.transAxes))
    ax_st.add_patch(plt.Rectangle((0.1, 0.15), 0.8 * f_water_pct, 0.08,
                    color='deepskyblue', transform=ax_st.transAxes))
    status = ('TANK FULL' if f_water_pct > 0.9 else
              'REFILL NEEDED' if f_water_pct < 0.2 else 'OPERATIONAL')
    txt = (f"Scouts reward (sum): {total_reward_q:+.1f}\n"
           f"Commander reward:    {total_reward_f:+.1f}\n\n"
           f"Fire visibility:     {fire_seen_sum:.2f}\n"
           f"Water level:         {f_water_pct*100:3.1f}%\n\n"
           f"Status: {status}")
    ax_st.text(0.08, 0.9, txt, color='#222222', fontsize=8.5,
               va='top', transform=ax_st.transAxes, fontfamily='monospace')
    ax_st.set_title("Mission stats", fontsize=9, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


# ── Analysis PNG ────────────────────────────────────────────
def _save_analysis(hist, out_path):
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    fig.patch.set_facecolor('white')

    axes[0].plot(np.cumsum(hist["f_r"]), label="Commander", color=CLR_CMDR)
    for q_name in hist["q_r"]:
        axes[0].plot(np.cumsum(hist["q_r"][q_name]), label=q_name,
                     color=CLR_SCOUT, alpha=0.7)
    axes[0].set_title("Cumulative reward", fontweight='bold')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for q_name in hist["q_alt"]:
        axes[1].plot(hist["q_alt"][q_name], color=CLR_SCOUT, alpha=0.7,
                     label=q_name)
    axes[1].plot(hist["f_alt"], color=CLR_CMDR, label="Commander")
    axes[1].set_title("Altitude [m]", fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].fill_between(range(len(hist["fire"])), hist["fire"],
                         color='orange', alpha=0.5)
    axes[2].set_title("Fire intensity (best scout)", fontweight='bold')
    axes[2].grid(alpha=0.3)

    axes[3].plot(hist["water"], color='deepskyblue', linewidth=2)
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_title("Commander water level", fontweight='bold')
    axes[3].grid(alpha=0.3)

    axes[4].set_title("Scout reward per step", fontweight='bold')
    axes[4].grid(alpha=0.3)
    for q_name in hist["q_r"]:
        axes[4].plot(hist["q_r"][q_name], color=CLR_SCOUT, alpha=0.3)

    axes[5].plot(hist["f_r"], color=CLR_CMDR, alpha=0.4)
    axes[5].set_title("Commander reward per step", fontweight='bold')
    axes[5].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Step', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"    Analysis → {out_path}")


# ── Run one episode ─────────────────────────────────────────
def run_episode(env, scout_actor, cmdr_actor, seed, device,
                terrain_collections=None):
    quad_names = [f"quad_{i}" for i in range(N_QUADS)]

    obs, _ = env.reset(seed=seed, epizode_number=30000)

    refill_info = env.sim.environment.refill_zone
    refill_pos = refill_info['position'] if refill_info else None

    safe_limit = max(50.0, env.map_bounds * 0.7)
    boundary_emergency = max(50.0, env.map_bounds * 0.6)

    h_scout = {q: torch.zeros(1, 1, 128).to(device) for q in quad_names}
    h_cmdr  = torch.zeros(1, 1, 64).to(device)

    cmdr_ctrl = CommanderController(WAYPOINT_RANGE, WAYPOINT_STEPS,
                                    WP_REACHED_DIST)
    cmdr_ctrl.reset(safe_limit, boundary_emergency)

    scout_msgs = {q: {"latest": torch.zeros(1, 5).to(device),
                      "valid": False}
                  for q in quad_names}

    hist = {"q_r": {q: [] for q in quad_names},
            "f_r": [], "q_alt": {q: [] for q in quad_names},
            "f_alt": [], "fire": [], "water": []}
    frames = []
    total_rf = 0.0
    total_rq = {q: 0.0 for q in quad_names}
    q_paths = {q: {"x": [], "y": []} for q in quad_names}
    f_path_x, f_path_y = [], []
    last_local_maps = {q: None for q in quad_names}
    fire_cells_peak = 0

    for step in tqdm.tqdm(range(MAX_STEPS), desc=f"seed={seed}", leave=False):
        if not env.agents:
            break

        if env.sim.environment.fire_grid is not None:
            fire_cells_peak = max(fire_cells_peak,
                                  int(np.sum(env.sim.environment.fire_grid.B)))
        actions = {}

        # ── Scouts ──────────────────────────────────────────
        for qi in range(N_QUADS):
            q = f"quad_{qi}"
            if q in env.agents:
                l_map = torch.FloatTensor(obs[q]["local_map"]).to(device).unsqueeze(0)
                s_st  = torch.FloatTensor(obs[q]["self_state"]).to(device).unsqueeze(0)
                n_st  = torch.FloatTensor(obs[q]["neighbor_states"]).to(device).unsqueeze(0)
                n_m   = torch.BoolTensor(obs[q]["neighbor_mask"]).to(device).unsqueeze(0)
                with torch.no_grad():
                    dist, scout_msg, h_scout[q] = scout_actor(
                        l_map, s_st, n_st, n_m, h_scout[q])
                actions[q] = dist.sample().squeeze(0).cpu().numpy()
                last_local_maps[q] = obs[q]["local_map"][0]
                scout_msgs[q]["latest"] = scout_msg
                scout_msgs[q]["valid"]  = True

        # ── Commander ───────────────────────────────────────
        if "fixed_0" in env.agents:
            drone = env.sim.drones.get("fixed_0")
            if drone is not None:
                msgs_t = torch.stack(
                    [scout_msgs[f"quad_{qi}"]["latest"]
                     for qi in range(N_QUADS)], dim=1).to(device)
                msgs_m = torch.tensor(
                    [[not scout_msgs[f"quad_{qi}"]["valid"]
                      for qi in range(N_QUADS)]],
                    dtype=torch.bool).to(device)
                action, h_cmdr, _ = cmdr_ctrl.step(
                    drone, obs["fixed_0"]["self_state"], env,
                    cmdr_actor, h_cmdr, msgs_t, msgs_m,
                    deterministic=True)
                actions["fixed_0"] = action

        obs, rewards, _, _, infos = env.step(actions)

        # ── Tracking ────────────────────────────────────────
        for qi in range(N_QUADS):
            q = f"quad_{qi}"
            total_rq[q] += rewards.get(q, 0.0)
            hist["q_r"][q].append(rewards.get(q, 0.0))
            if q in env.sim.drones:
                pos = env.sim.drones[q].get_position()
                q_paths[q]["x"].append(pos[0])
                q_paths[q]["y"].append(pos[1])
                hist["q_alt"][q].append(pos[2])

        total_rf += rewards.get("fixed_0", 0.0)
        hist["f_r"].append(rewards.get("fixed_0", 0.0))

        f_alive = "fixed_0" in env.sim.drones
        f_pos = None
        f_water_pct = 0.0
        if f_alive:
            f_pos = env.sim.drones["fixed_0"].get_position()
            f_path_x.append(f_pos[0]); f_path_y.append(f_pos[1])
            f_water_pct = (env.sim.drones["fixed_0"].current_water /
                           env.sim.drones["fixed_0"].water_capacity)
            hist["f_alt"].append(f_pos[2])
        hist["water"].append(f_water_pct)

        best_local_map = None
        best_fire_sum = -1
        for qi in range(N_QUADS):
            lm = last_local_maps[f"quad_{qi}"]
            if lm is not None:
                s = float(np.sum(lm))
                if s > best_fire_sum:
                    best_fire_sum = s
                    best_local_map = lm
        fire_seen = max(0.0, best_fire_sum)
        hist["fire"].append(fire_seen)

        # ── GIF frame ──────────────────────────────────────
        if step % GIF_EVERY == 0:
            q_positions = {}
            q_alive_map = {}
            for qi in range(N_QUADS):
                q = f"quad_{qi}"
                q_alive_map[q] = q in env.sim.drones
                q_positions[q] = (env.sim.drones[q].get_position()
                                  if q_alive_map[q] else None)
            frame = _render_frame(
                step,
                env.sim.environment.fire_grid.I.copy(),
                env.map_bounds,
                q_paths, q_positions, q_alive_map,
                list(f_path_x), list(f_path_y), f_pos, f_water_pct,
                refill_pos,
                best_local_map.copy() if best_local_map is not None else None,
                sum(total_rq.values()), total_rf, fire_seen, f_alive,
                N_QUADS,
                terrain_collections=terrain_collections)
            frames.append(frame)

    # ── Save outputs ────────────────────────────────────────
    gif_path = os.path.join(OUTPUT_DIR, f"seed_{seed}.gif")
    imageio.mimsave(gif_path, frames, fps=GIF_FPS, loop=0)
    print(f"    GIF → {gif_path}")

    png_path = os.path.join(OUTPUT_DIR, f"seed_{seed}_analysis.png")
    _save_analysis(hist, png_path)

    end_cells = (int(np.sum(env.sim.environment.fire_grid.B))
                 if env.sim.environment.fire_grid is not None else 0)
    supp_pct = ((1.0 - end_cells / fire_cells_peak) * 100.0
                if fire_cells_peak > 0 else 100.0)

    return {
        "seed": seed,
        "peak_cells": fire_cells_peak,
        "end_cells": end_cells,
        "supp_pct": supp_pct,
        "fw_survived": "fixed_0" in env.sim.drones,
        "scouts_survived": {f"quad_{i}": f"quad_{i}" in env.sim.drones
                            for i in range(N_QUADS)},
        "total_rq": total_rq,
        "total_rf": total_rf,
    }


# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  3-Scout + 1-FW Evaluation  |  3-4 fires  |  5 seeds")
    print("=" * 70)
    print(f"  Scout  : {MODEL_SCOUT}")
    print(f"  Cmdr   : {MODEL_CMDR}")
    print(f"  Output : {OUTPUT_DIR}")
    print()

    scout_actor, cmdr_actor = _load_models(device)

    terrain_collections = None
    if USE_OSM:
        print("  Loading OSM terrain...")
        gdfs = _load_terrain_gdfs()
        if gdfs:
            water, buildings, forests = _classify_features(gdfs)
            terrain_collections = _build_terrain_collections(
                water, buildings, forests)

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                       grid_size_m=GRID_SIZE, max_steps=MAX_STEPS,
                       use_osm=USE_OSM, osm_lat=OSM_LAT,
                       osm_lon=OSM_LON, osm_cache_dir=OSM_CACHE,
                       n_fires_range=N_FIRES_RANGE)

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n  [{i+1}/{len(SEEDS)}] Seed {seed}")
        m = run_episode(env, scout_actor, cmdr_actor, seed, device,
                        terrain_collections=terrain_collections)
        results.append(m)
        fw_icon = "✓" if m["fw_survived"] else "✗"
        s_icons = " ".join("✓" if v else "✗"
                           for v in m["scouts_survived"].values())
        print(f"    S={s_icons}  FW={fw_icon}  |  "
              f"peak={m['peak_cells']:4d}  end={m['end_cells']:4d}  "
              f"supp={m['supp_pct']:5.1f}%  |  R_cmdr={m['total_rf']:+.1f}")

    env.sim.stop_simulation()

    # ── Summary ─────────────────────────────────────────────
    n = len(results)
    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({n} episodes)")
    print(f"{'='*70}")
    fw_surv = sum(1 for r in results if r["fw_survived"])
    all_s = sum(1 for r in results if all(r["scouts_survived"].values()))
    avg_supp = float(np.mean([r["supp_pct"] for r in results]))
    avg_rf   = float(np.mean([r["total_rf"] for r in results]))
    avg_peak = float(np.mean([r["peak_cells"] for r in results]))
    print(f"  FW survived        : {fw_surv}/{n}")
    print(f"  All scouts survived: {all_s}/{n}")
    print(f"  Avg suppression    : {avg_supp:.1f}%")
    print(f"  Avg peak cells     : {avg_peak:.0f}")
    print(f"  Avg R_cmdr         : {avg_rf:+.1f}")
    print(f"\n  Output dir: {OUTPUT_DIR}")
