"""
demo_scout.py
─────────────
Visualisation demo for 2-scout MAPPO team (no commander).
Uses real OSM terrain (buildings, forests, water) as map background.

Layout:
┌──────────────────────────┬──────────────────┐
│                          │ Scout 0 camera   │
│   Global map + fire      │ (32×32 view)     │
│   OSM terrain + trails   ├──────────────────┤
│   FoV rectangles         │ Scout 1 camera   │
│                          ├──────────────────┤
│                          │ Stats            │
└──────────────────────────┴──────────────────┘

Output: scout_demo.gif  +  scout_demo_analysis.png
"""

import torch
import numpy as np
import os, sys, glob
import matplotlib
matplotlib.use('Agg')
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
from src.models import ScoutActor

# ── Terrain colours ─────────────────────────────────────────
CLR_GRASS    = '#f0eed8'
CLR_BUILDING = '#b0b0b0'
CLR_FOREST   = '#6abf69'
CLR_WATER    = '#7ec8e3'

SCOUT_COLORS  = ['#00bfff', '#00e676']   # cyan, green
SCOUT_MARKERS = ['^', 's']
SCOUT_LABELS  = ['Scout 0', 'Scout 1']

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH  = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/scout_solo/scout_b0710.pt"

N_QUADS      = 2
N_FIXED      = 0
MAX_STEPS    = 1000
GRID_SIZE    = 1000.0
GIF_EVERY    = 2
GIF_FPS      = 15
EPISODE_SEED = 103
USE_OSM      = True
OSM_LAT      = 49.35
OSM_LON      = 16.42
OSM_CACHE    = os.path.join(project_root, "data")
# ============================================================


# ── OSM terrain helpers ─────────────────────────────────────

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
    center_proj = gpd.GeoSeries([Point(OSM_LON, OSM_LAT)], crs='EPSG:4326').to_crs(utm_crs).iloc[0]
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
    fp = []
    for g in forests:
        fp.extend(_geom_patches(g))
    if fp:
        colls.append(PatchCollection(fp, facecolor=CLR_FOREST, edgecolor='none', alpha=0.7, zorder=1))
    bp = []
    for g in buildings:
        bp.extend(_geom_patches(g))
    if bp:
        colls.append(PatchCollection(bp, facecolor=CLR_BUILDING, edgecolor='#999', linewidth=0.3, alpha=0.85, zorder=2))
    wp = []
    for g in water:
        wp.extend(_geom_patches(g))
    if wp:
        colls.append(PatchCollection(wp, facecolor=CLR_WATER, edgecolor='none', alpha=0.85, zorder=3))
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


# ── Render one frame ────────────────────────────────────────

def _render_frame(step, fire_map, b,
                  scout_paths, scout_positions, scout_fovs, scout_alive,
                  local_maps, total_rewards, fire_sums,
                  terrain_collections=None):

    fig = plt.figure(figsize=(12, 7), facecolor='white')
    gs  = gridspec.GridSpec(3, 2, width_ratios=[2, 1], hspace=0.35, wspace=0.25,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Panel 1: Global map ─────────────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(CLR_GRASS)

    if terrain_collections:
        _add_terrain_to_ax(ax_map, terrain_collections)

    extent = [-b, b, -b, b]
    fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
    ax_map.imshow(fire_masked, extent=extent, origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.75, zorder=4)

    for qi in range(N_QUADS):
        clr = SCOUT_COLORS[qi]
        mk  = SCOUT_MARKERS[qi]
        px, py = scout_paths[qi]

        if len(px) > 1:
            ax_map.plot(px, py, color=clr, alpha=0.4,
                        linewidth=1.2, linestyle=':', zorder=5)

        if scout_alive[qi] and scout_positions[qi] is not None:
            pos = scout_positions[qi]
            fov = scout_fovs[qi]
            rect = plt.Rectangle((pos[0] - fov/2, pos[1] - fov/2), fov, fov,
                                  fill=False, edgecolor=clr, linewidth=1.0,
                                  alpha=0.7, zorder=6)
            ax_map.add_patch(rect)
            ax_map.scatter(pos[0], pos[1], c=clr, s=100, marker=mk,
                           edgecolors='black', linewidths=0.6, zorder=7)

    ax_map.set_xlim(-b, b); ax_map.set_ylim(-b, b)
    ax_map.set_aspect('equal')
    ax_map.set_title(f"Step {step:04d}  |  Global overview", fontsize=11, fontweight='bold')
    ax_map.set_xlabel('X [m]', fontsize=8); ax_map.set_ylabel('Y [m]', fontsize=8)
    ax_map.tick_params(labelsize=7)

    leg_h = []
    if terrain_collections:
        leg_h.append(mpatches.Patch(color=CLR_FOREST,   label='Forest'))
        leg_h.append(mpatches.Patch(color=CLR_BUILDING, label='Building'))
        leg_h.append(mpatches.Patch(color=CLR_WATER,    label='Water'))
    for qi in range(N_QUADS):
        leg_h.append(plt.Line2D([0],[0], marker=SCOUT_MARKERS[qi], color='w',
                                markerfacecolor=SCOUT_COLORS[qi],
                                markersize=7, label=SCOUT_LABELS[qi]))
    ax_map.legend(handles=leg_h, loc='upper right', fontsize=6.5,
                  framealpha=0.85, edgecolor='#ccc')

    # ── Panel 2: Scout 0 camera ─────────────────────────────────────
    ax_cam0 = fig.add_subplot(gs[0, 1])
    ax_cam0.set_facecolor('#fafafa')
    if local_maps[0] is not None:
        ax_cam0.imshow(local_maps[0], origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
    ax_cam0.set_title("Scout 0 camera (32x32)", fontsize=9, fontweight='bold',
                       color=SCOUT_COLORS[0])
    ax_cam0.set_xticks([]); ax_cam0.set_yticks([])

    # ── Panel 3: Scout 1 camera ─────────────────────────────────────
    ax_cam1 = fig.add_subplot(gs[1, 1])
    ax_cam1.set_facecolor('#fafafa')
    if local_maps[1] is not None:
        ax_cam1.imshow(local_maps[1], origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
    ax_cam1.set_title("Scout 1 camera (32x32)", fontsize=9, fontweight='bold',
                       color=SCOUT_COLORS[1])
    ax_cam1.set_xticks([]); ax_cam1.set_yticks([])

    # ── Panel 4: Stats ──────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.set_facecolor('#fafafa')
    ax_stats.set_xticks([]); ax_stats.set_yticks([])

    alt_strs = []
    for qi in range(N_QUADS):
        if scout_alive[qi] and scout_positions[qi] is not None:
            alt_strs.append(f"{scout_positions[qi][2]:.0f} m")
        else:
            alt_strs.append("DEAD")

    stats_text = (
        f"Scout 0 reward:  {total_rewards[0]:+.1f}   alt: {alt_strs[0]}\n"
        f"Scout 1 reward:  {total_rewards[1]:+.1f}   alt: {alt_strs[1]}\n\n"
        f"Fire under S0:   {fire_sums[0]:.2f}\n"
        f"Fire under S1:   {fire_sums[1]:.2f}\n\n"
        f"Step: {step} / {MAX_STEPS}"
    )
    ax_stats.text(0.08, 0.92, stats_text, color='#222222', fontsize=8.5,
                  va='top', transform=ax_stats.transAxes, fontfamily='monospace')
    ax_stats.set_title("Mission stats", fontsize=9, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


def run_demo():
    print("Demo: 2-Scout MAPPO team")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load environment & model ────────────────────────────────────
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=GRID_SIZE,
                       max_steps=MAX_STEPS, use_osm=USE_OSM, osm_lat=OSM_LAT,
                       osm_lon=OSM_LON, osm_cache_dir=OSM_CACHE)

    obs_space = env.observation_space("quad_0")
    scout_self_dim = obs_space["self_state"].shape[0]
    scout_msg_dim = 5
    hidden_dim = 128

    scout_actor = ScoutActor(self_state_dim=scout_self_dim,
                             msg_dim=scout_msg_dim, hidden_dim=hidden_dim).to(device)
    if os.path.exists(MODEL_PATH):
        scout_actor.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return
    scout_actor.eval()

    # ── Load OSM terrain ────────────────────────────────────────────
    terrain_collections = None
    if USE_OSM:
        print("Loading OSM terrain...")
        gdfs = _load_terrain_gdfs()
        if gdfs:
            water, buildings, forests = _classify_features(gdfs)
            terrain_collections = _build_terrain_collections(water, buildings, forests)
            print(f"  Water: {len(water)}  Buildings: {len(buildings)}  Forest: {len(forests)}")

    # ── Run episode ─────────────────────────────────────────────────
    obs, _ = env.reset(seed=EPISODE_SEED, epizode_number=20000)

    # Override: bigger fire + scouts further away for a nicer demo
    # 1) Enlarge the fire — ignite a 60m radius patch (much larger than default 5m)
    fire_x, fire_y = env.fire_x, env.fire_y
    env.sim.environment.start_fire_at_position(
        [fire_x, fire_y], intensity=0.8, radius_m=6)
    # Let fire spread a few ticks so it looks established
    for _ in range(10):
        env.sim.environment.update_fire_simulation(real_dt=0.1)

    # 2) Respawn scouts at least 200m from fire, with 50m mutual separation
    import random as _rnd
    import pybullet as _p
    for agent in env.quad_agents:
        if agent not in env.sim.drones:
            continue
        for _try in range(50):
            angle = _rnd.uniform(0, 2 * np.pi)
            dist = _rnd.uniform(100.0, 150.0)
            sx = fire_x + dist * np.cos(angle)
            sy = fire_y + dist * np.sin(angle)
            if abs(sx) > env.map_bounds * 0.9 or abs(sy) > env.map_bounds * 0.9:
                continue
            # Check separation from other scouts
            ok = True
            for other in env.quad_agents:
                if other == agent or other not in env.sim.drones:
                    continue
                op = env.sim.drones[other].get_position()
                if np.hypot(sx - op[0], sy - op[1]) < 50.0:
                    ok = False
                    break
            if ok:
                break
        drone = env.sim.drones[agent]
        new_z = _rnd.uniform(60.0, 80.0)
        orn = drone.get_orientation_quaternion()
        _p.resetBasePositionAndOrientation(drone.drone_id, [sx, sy, new_z], orn)
        _p.resetBaseVelocity(drone.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    # Re-observe after override
    obs = {agent: env._get_obs(agent) for agent in env.agents}

    quad_agents = env.quad_agents

    hidden_states = {q: torch.zeros(1, 1, hidden_dim).to(device) for q in quad_agents}
    scout_paths = {qi: ([], []) for qi in range(N_QUADS)}
    local_maps = [None] * N_QUADS
    total_rewards = [0.0] * N_QUADS
    fire_sums = [0.0] * N_QUADS
    frames = []

    hist = {qi: {"reward": [], "alt": [], "fire": []} for qi in range(N_QUADS)}

    print("Starting mission...")
    for step in tqdm.tqdm(range(MAX_STEPS)):
        if not env.agents:
            break
        actions = {}

        for qi, agent in enumerate(quad_agents):
            if agent not in env.agents or agent not in env.sim.drones:
                continue

            l_map = torch.FloatTensor(obs[agent]["local_map"]).to(device).unsqueeze(0)
            s_st  = torch.FloatTensor(obs[agent]["self_state"]).to(device).unsqueeze(0)
            n_st  = torch.FloatTensor(obs[agent]["neighbor_states"]).to(device).unsqueeze(0)
            n_m   = torch.BoolTensor(obs[agent]["neighbor_mask"]).to(device).unsqueeze(0)

            with torch.no_grad():
                dist, msg, h_out = scout_actor(l_map, s_st, n_st, n_m, hidden_states[agent])
            hidden_states[agent] = h_out
            actions[agent] = dist.mean.squeeze(0).cpu().numpy()
            local_maps[qi] = obs[agent]["local_map"][0]

        if not actions:
            break

        obs, rewards, terminations, truncations, infos = env.step(actions)

        scout_positions = [None] * N_QUADS
        scout_fovs = [40.0] * N_QUADS
        scout_alive = [False] * N_QUADS

        for qi, agent in enumerate(quad_agents):
            r = rewards.get(agent, 0.0)
            total_rewards[qi] += r
            hist[qi]["reward"].append(r)

            if agent in env.sim.drones:
                scout_alive[qi] = True
                pos = env.sim.drones[agent].get_position()
                scout_positions[qi] = pos
                scout_fovs[qi] = max(10.0, pos[2] * 1.5)
                scout_paths[qi][0].append(pos[0])
                scout_paths[qi][1].append(pos[1])
                hist[qi]["alt"].append(pos[2])
                fire_sums[qi] = float(np.sum(local_maps[qi])) if local_maps[qi] is not None else 0.0
                hist[qi]["fire"].append(fire_sums[qi])
            else:
                hist[qi]["alt"].append(0.0)
                hist[qi]["fire"].append(0.0)

        if step % GIF_EVERY == 0:
            frame = _render_frame(
                step, env.sim.environment.fire_grid.I.copy(), env.map_bounds,
                {qi: (list(scout_paths[qi][0]), list(scout_paths[qi][1])) for qi in range(N_QUADS)},
                scout_positions, scout_fovs, scout_alive,
                [m.copy() if m is not None else None for m in local_maps],
                list(total_rewards), list(fire_sums),
                terrain_collections=terrain_collections,
            )
            frames.append(frame)

    out_gif = os.path.join(project_root, "scout_demo.gif")
    imageio.mimsave(out_gif, frames, fps=GIF_FPS, loop=0)
    print(f"GIF saved: {out_gif}")

    _save_analysis(hist, project_root)


def _save_analysis(hist, project_root):
    fig, axes = plt.subplots(3, N_QUADS, figsize=(7 * N_QUADS, 10), squeeze=False)
    fig.patch.set_facecolor('white')
    fig.suptitle("Scout Demo - Analysis", fontsize=14, fontweight='bold')

    for qi in range(N_QUADS):
        clr = SCOUT_COLORS[qi]

        axes[0, qi].plot(np.cumsum(hist[qi]["reward"]), color=clr, linewidth=1.5)
        axes[0, qi].set_title(f"{SCOUT_LABELS[qi]} - Cumulative reward", fontweight='bold')
        axes[0, qi].grid(alpha=0.3)

        axes[1, qi].plot(hist[qi]["alt"], color=clr, linewidth=1.0)
        axes[1, qi].set_title(f"{SCOUT_LABELS[qi]} - Altitude [m]", fontweight='bold')
        axes[1, qi].grid(alpha=0.3)

        axes[2, qi].fill_between(range(len(hist[qi]["fire"])), hist[qi]["fire"],
                                  color='orange', alpha=0.5)
        axes[2, qi].set_title(f"{SCOUT_LABELS[qi]} - Fire under camera", fontweight='bold')
        axes[2, qi].set_xlabel("Step")
        axes[2, qi].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(project_root, "scout_demo_analysis.png")
    plt.savefig(out, dpi=120, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Analysis saved: {out}")


if __name__ == "__main__":
    run_demo()