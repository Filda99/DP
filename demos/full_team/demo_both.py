"""
demo_both_training.py
─────────────────────
Visualisation demo for heterogeneous MAPPO team (Scout + Commander).
Uses real OSM terrain (buildings, forests, water) as map background.

Layout:
┌──────────────────────────┬──────────────────┐
│                          │  Scout camera    │
│   Global map + fire      │  (32×32 view)    │
│   OSM terrain + trails   ├──────────────────┤
│   FoV rectangle          │  Stats           │
└──────────────────────────┴──────────────────┘

Output: demo_training.gif  +  demo_training_analysis.png
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
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor
from src.commander_control import CommanderController

# ── Terrain colours ─────────────────────────────────────────
CLR_GRASS    = '#f0eed8'
CLR_BUILDING = '#b0b0b0'
CLR_FOREST   = '#6abf69'
CLR_WATER    = '#7ec8e3'
CLR_SCOUT    = '#00bfff'
CLR_CMDR     = '#ff3333'

# ============================================================
# KONFIGURACE
# ============================================================
MODEL_SCOUT     = "/homes/eva/xj/xjahnf00/tmp/DP/results/TrainingQuad/050626_TrainingMultipleScoutsOnMultipleFires/scout_b0120.pt"
MODEL_COMMANDER = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/v6_shortWP/cmdr_best.pt"

N_QUADS    = 4
N_FIXED    = 3
MAX_STEPS  = 1000
GRID_SIZE  = 1000.0
GIF_EVERY  = 3
GIF_FPS    = 15
EPISODE_SEED = 111
USE_OSM      = True
OSM_LAT      = 49.35
OSM_LON      = 16.42
OSM_CACHE    = os.path.join(project_root, "data")

# Commander waypoint parameters (must match training)
WAYPOINT_RANGE  = 50.0    # metres per unit of dx/dy
WAYPOINT_STEPS  = 30      # physics steps per waypoint segment
WP_REACHED_DIST = 30.0    # metres
# ============================================================

# ── OSM terrain helpers ─────────────────────────────────────────

def _load_terrain_gdfs():
    """Load cached .gpkg files, project to UTM, centre on (0,0)."""
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
    """Split into water / building / forest geometry lists."""
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
    """Pre-build PatchCollections for terrain layers."""
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
    """Re-add terrain PatchCollections to a fresh axes (deep-copy paths)."""
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


def _load_models(device):
    env_tmp = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                           grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    scout_self_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0]
    fixed_self_dim = env_tmp.observation_space(env_tmp.fixed_agents[0])["self_state"].shape[0]

    scout = ScoutActor(self_state_dim=scout_self_dim, msg_dim=5, hidden_dim=128).to(device)
    cmdr  = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=5,
                           action_dim=4, hidden_dim=64).to(device)

    for path, name, model in [(MODEL_SCOUT, "Scout", scout),
                               (MODEL_COMMANDER, "Commander", cmdr)]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            print(f"✅ {name}: {path}")
        else:
            print(f"⚠️ {name}: model nenalezen ({path})")
    
    scout.eval()
    cmdr.eval()
    return scout, cmdr, scout_self_dim, fixed_self_dim

def _render_frame(step, fire_map, b,
                  q_paths, q_positions, q_alive_map,
                  f_paths, f_positions, f_alive_map, f_water_pcts,
                  refill_zones_list, refill_size,
                  local_maps_dict,
                  total_reward_q, total_reward_f,
                  fire_seen_sum, any_f_alive,
                  terrain_collections=None,
                  moisture_map=None,
                  f_valve_open=None):
    """Render one animation frame.

    f_paths      : dict  {f_name: {"x":[], "y":[]}}
    f_positions  : dict  {f_name: pos or None}
    f_alive_map  : dict  {f_name: bool}
    f_water_pcts : dict  {f_name: float}
    local_maps_dict : dict {q_name: np.ndarray or None}
    """

    n_scouts = len(q_positions)
    n_fw = len(f_positions)
    # Dynamic right-panel rows: scouts on top, stats on bottom
    right_rows = max(n_scouts, 1) + 1  # +1 for stats

    fig = plt.figure(figsize=(14, 7), facecolor='white')
    gs  = gridspec.GridSpec(right_rows, 2, width_ratios=[2.2, 1],
                            hspace=0.4, wspace=0.25,
                            left=0.05, right=0.97, top=0.93, bottom=0.05)

    # ── Panel 1: Global map ─────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(CLR_GRASS)

    # OSM terrain background
    if terrain_collections:
        _add_terrain_to_ax(ax_map, terrain_collections)

    # Fire
    extent = [-b, b, -b, b]
    # Moisture (blue overlay — shows where water was dropped)
    if moisture_map is not None:
        moist_masked = np.ma.masked_where(moisture_map < 0.01, moisture_map)
        ax_map.imshow(moist_masked, extent=extent, origin='lower',
                      cmap='Blues', vmin=0, vmax=1.0, alpha=0.55, zorder=3)

    fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
    ax_map.imshow(fire_masked, extent=extent, origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.75, zorder=4)

    # Refill zones (all of them)
    if refill_zones_list:
        for rz in refill_zones_list:
            rp = rz['position'] if isinstance(rz, dict) else rz
            circle = plt.Circle((rp[0], rp[1]), 20,
                                color='deepskyblue', fill=False, linestyle='--',
                                linewidth=1.5, alpha=0.6, zorder=5)
            ax_map.add_patch(circle)
            ax_map.text(rp[0], rp[1] + 25, "REFILL",
                        color='deepskyblue', fontsize=6, ha='center',
                        fontweight='bold', alpha=0.7)

    # Trails — scouts
    for q_name, paths in q_paths.items():
        if len(paths["x"]) > 1:
            ax_map.plot(paths["x"], paths["y"], color=CLR_SCOUT, alpha=0.4,
                        linewidth=1.2, linestyle=':', zorder=5)
    # Trails — each FW separately
    fw_colors = ['#ff3333', '#ff9933', '#cc33ff', '#33ccff', '#33ff99', '#ff6699']
    for idx, (f_name, fpaths) in enumerate(f_paths.items()):
        if len(fpaths["x"]) > 1:
            c = fw_colors[idx % len(fw_colors)]
            ax_map.plot(fpaths["x"], fpaths["y"], color=c, alpha=0.4,
                        linewidth=1.2, linestyle=':', zorder=5)

    # Agent positions — scouts
    for q_name in q_positions:
        q_pos = q_positions[q_name]
        if q_alive_map.get(q_name, False) and q_pos is not None:
            q_alt = q_pos[2]
            q_fov = max(10.0, q_alt * 1.5)
            rect = plt.Rectangle((q_pos[0] - q_fov/2, q_pos[1] - q_fov/2), q_fov, q_fov,
                                  fill=False, edgecolor=CLR_SCOUT, linewidth=1.0,
                                  alpha=0.7, zorder=6)
            ax_map.add_patch(rect)
            ax_map.scatter(q_pos[0], q_pos[1], c=CLR_SCOUT, s=100, marker='^',
                           edgecolors='black', linewidths=0.6, zorder=7)

    # Agent positions — each FW
    for idx, f_name in enumerate(f_positions):
        if f_alive_map.get(f_name, False) and f_positions[f_name] is not None:
            fp = f_positions[f_name]
            c = fw_colors[idx % len(fw_colors)]
            ax_map.scatter(fp[0], fp[1], c=c, s=130, marker='>',
                           edgecolors='black', linewidths=0.6, zorder=7)
            # Valve-open indicator: blue water-drop ring around FW
            if f_valve_open and f_valve_open.get(f_name, False):
                ring = plt.Circle((fp[0], fp[1]), 12,
                                  color='deepskyblue', fill=False,
                                  linewidth=2.5, alpha=0.9, zorder=8)
                ax_map.add_patch(ring)
                ax_map.text(fp[0], fp[1] - 18, 'W', fontsize=7,
                            ha='center', va='top', zorder=8,
                            color='deepskyblue', fontweight='bold')

    ax_map.set_xlim(-b, b); ax_map.set_ylim(-b, b)
    ax_map.set_aspect('equal')
    ax_map.set_title(f"Step {step:04d}  |  Global overview", fontsize=11, fontweight='bold')
    ax_map.set_xlabel('X [m]', fontsize=8); ax_map.set_ylabel('Y [m]', fontsize=8)
    ax_map.tick_params(labelsize=7)

    # Map legend
    leg_h = []
    if terrain_collections:
        leg_h.append(mpatches.Patch(color=CLR_FOREST,   label='Forest'))
        leg_h.append(mpatches.Patch(color=CLR_BUILDING, label='Building'))
        leg_h.append(mpatches.Patch(color=CLR_WATER,    label='Water'))
    leg_h.append(plt.Line2D([0],[0], marker='^', color='w', markerfacecolor=CLR_SCOUT,
                            markersize=7, label='Scout'))
    leg_h.append(plt.Line2D([0],[0], marker='>', color='w', markerfacecolor=CLR_CMDR,
                            markersize=7, label='Commander'))
    ax_map.legend(handles=leg_h, loc='upper right', fontsize=6.5,
                  framealpha=0.85, edgecolor='#ccc')

    # ── Panel 2: Scout cameras (one per scout) ────────────────────────────
    scout_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#bcbd22', '#e377c2']
    for qi, q_name in enumerate(sorted(q_positions.keys())):
        ax_cam = fig.add_subplot(gs[qi, 1])
        ax_cam.set_facecolor('#fafafa')
        lm = local_maps_dict.get(q_name) if local_maps_dict else None
        if lm is not None:
            ax_cam.imshow(lm, origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
        alive_str = "" if q_alive_map.get(q_name, False) else " [DEAD]"
        sc = scout_colors[qi % len(scout_colors)]
        ax_cam.set_title(f"{q_name}{alive_str}", fontsize=8, fontweight='bold', color=sc)
        ax_cam.set_xticks([]); ax_cam.set_yticks([])

    # ── Panel 3: Stats + Per-FW Water Bars ──────────────────────────────
    ax_stats = fig.add_subplot(gs[n_scouts:, 1])
    ax_stats.set_facecolor('#fafafa')
    ax_stats.set_xticks([]); ax_stats.set_yticks([])

    # Per-FW water bars
    bar_h = 0.045
    bar_gap = 0.06
    bar_top = 0.38
    for fi, f_name in enumerate(sorted(f_water_pcts.keys())):
        alive = f_alive_map.get(f_name, False)
        wpct = f_water_pcts[f_name] if alive else 0.0
        y_pos = bar_top - fi * bar_gap
        c = fw_colors[fi % len(fw_colors)]
        # Background
        ax_stats.add_patch(plt.Rectangle((0.22, y_pos), 0.7, bar_h,
                           color='#dddddd', transform=ax_stats.transAxes))
        # Fill
        ax_stats.add_patch(plt.Rectangle((0.22, y_pos), 0.7 * wpct, bar_h,
                           color=c, alpha=0.8, transform=ax_stats.transAxes))
        label = f_name if alive else f"{f_name} ✗"
        ax_stats.text(0.04, y_pos + bar_h / 2, label, fontsize=7, va='center',
                      transform=ax_stats.transAxes, fontfamily='monospace',
                      color=c if alive else '#999999')
        ax_stats.text(0.93, y_pos + bar_h / 2, f"{wpct*100:.0f}%", fontsize=7,
                      va='center', transform=ax_stats.transAxes, fontfamily='monospace')

    n_fw_alive = sum(1 for v in f_alive_map.values() if v)
    status = f'{n_fw_alive} FW alive' if n_fw_alive > 0 else 'ALL FW DEAD'
    stats_text = (
        f"R_scout: {total_reward_q:+.1f}  R_cmdr: {total_reward_f:+.1f}\n"
        f"Fire vis: {fire_seen_sum:.2f}   Status: {status}"
    )
    ax_stats.text(0.04, 0.95, stats_text, color='#222222', fontsize=8,
                  va='top', transform=ax_stats.transAxes, fontfamily='monospace')
    ax_stats.set_title("Mission stats", fontsize=9, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr

def run_episode(env, scout_actor, commander_actor, seed, ep_num, device,
                terrain_collections=None, save_gif=False, gif_path=None,
                scout_dist=None, fw_dist=None, init_water=None):
    """Spustí jednu epizodu. Vrací dict metrik."""
    quad_names = [f"quad_{i}" for i in range(N_QUADS)]

    obs, _ = env.reset(seed=seed, epizode_number=ep_num)

    # ── Optional: override spawn distances ────────────────────────────
    if scout_dist is not None or fw_dist is not None:
        import pybullet as _p
        fire_x, fire_y = env.fire_x, env.fire_y
        rng = np.random.default_rng(seed)
        if scout_dist is not None:
            n_q = len([q for q in quad_names if q in env.sim.drones])
            for qi, q in enumerate(quad_names):
                if q in env.sim.drones:
                    # Spread scouts evenly around fire + small random jitter
                    base_angle = 2 * np.pi * qi / max(n_q, 1)
                    angle = base_angle + rng.uniform(-0.3, 0.3)
                    nx = fire_x + scout_dist * np.cos(angle)
                    ny = fire_y + scout_dist * np.sin(angle)
                    nx = float(np.clip(nx, -env.map_bounds * 0.85, env.map_bounds * 0.85))
                    ny = float(np.clip(ny, -env.map_bounds * 0.85, env.map_bounds * 0.85))
                    d = env.sim.drones[q]
                    _p.resetBasePositionAndOrientation(
                        d.drone_id, [nx, ny, 80.0], [0, 0, 0, 1])
        if fw_dist is not None:
            for fi in range(N_FIXED):
                fname = f"fixed_{fi}"
                if fname in env.sim.drones:
                    angle = rng.uniform(0, 2 * np.pi)
                    nx = fire_x + fw_dist * np.cos(angle)
                    ny = fire_y + fw_dist * np.sin(angle)
                    nx = float(np.clip(nx, -env.map_bounds * 0.85, env.map_bounds * 0.85))
                    ny = float(np.clip(ny, -env.map_bounds * 0.85, env.map_bounds * 0.85))
                    d = env.sim.drones[fname]
                    d.state_pos[0] = nx
                    d.state_pos[1] = ny
        # Re-read obs after teleport so first frame is correct
        obs = {a: env._get_obs(a) for a in env.agents}

    # ── FW initial water level ────────────────────────────────────────
    # Training uses empty tank (0 L) so the FW must refill first.
    # Mirror that default here; override with --init-water if needed.
    water_val = init_water if init_water is not None else 0.0
    for a in env.fixed_agents:
        if a in env.sim.drones:
            d = env.sim.drones[a]
            if d.water_capacity > 0:
                d.current_water = float(np.clip(water_val, 0.0, d.water_capacity))
                env._prev_fw_water[a] = d.current_water / d.water_capacity
    obs = {a: env._get_obs(a) for a in env.agents}
    # ──────────────────────────────────────────────────
    # Support multiple refill zones (v4+)
    refill_zones = getattr(env.sim.environment, 'refill_zones', [])
    refill_info = env.sim.environment.refill_zone
    if not refill_zones and refill_info:
        refill_zones = [refill_info]
    refill_pos = refill_zones[0]['position'] if refill_zones else None
    refill_size = refill_zones[0].get('size', 20.0) if refill_zones else 20.0

    print(f"📐 map_bounds={env.map_bounds}")

    h_scout = {q: torch.zeros(1, 1, 128).to(device) for q in quad_names}
    fixed_names = [f"fixed_{i}" for i in range(N_FIXED)]
    h_cmdr  = {f: torch.zeros(1, 1, 64).to(device) for f in fixed_names}

    hist = { "q_r": {q: [] for q in quad_names},
             "f_r": {f: [] for f in fixed_names},
             "q_alt": {q: [] for q in quad_names},
             "fire": [], "water": {f: [] for f in fixed_names},
             "fw": {f: {"x": [], "y": [], "z": [], "v": []} for f in fixed_names},
             "q_pos": {q: {"x": [], "y": [], "z": [], "v": []} for q in quad_names},
             # Per-agent reward component breakdown
             "q_diag": {q: {} for q in quad_names},   # keys added dynamically
             "f_diag": {f: {} for f in fixed_names},
             }
    frames, total_rf = [], 0.0
    total_rq = {q: 0.0 for q in quad_names}
    q_paths = {q: {"x": [], "y": []} for q in quad_names}
    f_paths = {f: {"x": [], "y": []} for f in fixed_names}
    last_local_maps = {q: None for q in quad_names}

    # Commander controller per FW (waypoint + scripted refill + heading)
    cmdr_ctrl = {f: CommanderController(WAYPOINT_RANGE, WAYPOINT_STEPS, WP_REACHED_DIST)
                 for f in fixed_names}
    for f in fixed_names:
        cmdr_ctrl[f].reset(env.map_bounds)
    # Per-scout message tracking: latest + best-fire
    scout_msgs = {q: {"latest": torch.zeros(1, 5).to(device),
                      "best": torch.zeros(1, 5).to(device),
                      "valid": False, "best_intensity": -1.0}
                  for q in quad_names}

    print(f"  Seed {seed} | 🚀 Mise začíná...  fire=({env.fire_x:.0f}, {env.fire_y:.0f})  "
          f"water={water_val:.0f}L")
    print(f"  Refill zones: {len(refill_zones)}")
    for rz in refill_zones:
        rp = rz['position']
        print(f"    zone at ({rp[0]:.0f}, {rp[1]:.0f})")
    fire_cells_peak = 0
    total_water_drops = 0
    total_water_hits = 0
    for step in tqdm.tqdm(range(MAX_STEPS), desc=f"seed={seed}", leave=False):
        if not env.agents: break
        # Peak fire
        if env.sim.environment.fire_grid is not None:
            fire_cells_peak = max(fire_cells_peak, int(np.sum(env.sim.environment.fire_grid.B)))
        actions = {}

        # ── Scouts ───────────────────────────────────────────────────────
        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            if q_name in env.agents:
                l_map = torch.FloatTensor(obs[q_name]["local_map"]).to(device).unsqueeze(0)
                s_st  = torch.FloatTensor(obs[q_name]["self_state"]).to(device).unsqueeze(0)
                n_st  = torch.FloatTensor(obs[q_name]["neighbor_states"]).to(device).unsqueeze(0)
                n_m   = torch.BoolTensor(obs[q_name]["neighbor_mask"]).to(device).unsqueeze(0)
                with torch.no_grad():
                    dist, scout_msg, h_scout[q_name] = scout_actor(
                        l_map, s_st, n_st, n_m, h_scout[q_name])
                actions[q_name] = dist.sample().squeeze(0).cpu().numpy()
                last_local_maps[q_name] = obs[q_name]["local_map"][0]
                # Update message tracking
                sm = scout_msgs[q_name]
                sm["latest"] = scout_msg  # [1, 5]
                sm["valid"] = True
                intensity = scout_msg[0, 2].item()
                if intensity > sm["best_intensity"]:
                    sm["best_intensity"] = intensity
                    sm["best"] = scout_msg

        # ── Commander (via shared CommanderController, per FW) ────────
        for f_name in fixed_names:
          if f_name in env.agents:
            drone = env.sim.drones.get(f_name)
            if drone is not None:
                msgs_t = torch.stack(
                    [scout_msgs[f"quad_{qi}"]["latest"] for qi in range(N_QUADS)],
                    dim=1).to(device)
                msgs_m = torch.tensor(
                    [[not scout_msgs[f"quad_{qi}"]["valid"] for qi in range(N_QUADS)]],
                    dtype=torch.bool).to(device)
                # Build FW neighbor states
                my_pos = drone.get_position()
                fw_nl, fw_ml = [], []
                for of in fixed_names:
                    if of == f_name:
                        continue
                    if of in env.sim.drones:
                        op = env.sim.drones[of].get_position()
                        fw_nl.append([(op[0]-my_pos[0])/env.map_bounds,
                                      (op[1]-my_pos[1])/env.map_bounds,
                                      (op[2]-my_pos[2])/100.0])
                        fw_ml.append(False)
                    else:
                        fw_nl.append([0.,0.,0.]); fw_ml.append(True)
                if not fw_nl:
                    fw_nl = [[0.,0.,0.]]; fw_ml = [True]
                fw_n_t = torch.FloatTensor([fw_nl]).to(device)
                fw_nm_t = torch.BoolTensor([fw_ml]).to(device)
                action, h_cmdr[f_name], _ = cmdr_ctrl[f_name].step(
                    drone, obs[f_name]["self_state"], env,
                    commander_actor, h_cmdr[f_name], msgs_t, msgs_m,
                    deterministic=True,
                    fw_neighbor_states=fw_n_t,
                    fw_neighbor_mask=fw_nm_t)
                actions[f_name] = action

        obs, rewards, _, _, infos = env.step(actions)

        # Water drop tracking
        for f_name in fixed_names:
            fi = infos.get(f_name, {})
            if "wd_alt" in fi:
                total_water_drops += 1
                hit = fi.get("wd_eff", 0) > 0
                if hit:
                    total_water_hits += 1
                tag = "HIT" if hit else "MISS"
                print(f"  💧 step={step:4d} {f_name} [{tag}] alt={fi['wd_alt']:.0f}m "
                      f"dist_fire={fi.get('wd_dist',0):.0f}m eff={fi.get('wd_eff',0):.3f} "
                      f"water_left={fi.get('wd_water',0):.0f}L")

        # Commander death diagnostic
        for f_name in fixed_names:
          if f_name in infos and infos[f_name].get("death_cause", ""):
            dc = infos[f_name]["death_cause"]
            lx = f_paths[f_name]["x"][-1] if f_paths[f_name]["x"] else "?"
            ly = f_paths[f_name]["y"][-1] if f_paths[f_name]["y"] else "?"
            ctrl = cmdr_ctrl[f_name]
            print(f"\n💀 {f_name} died at step {step}: cause={dc}, "
                  f"last_pos=({lx:.1f}, {ly:.1f}), "
                  f"map_bounds={env.map_bounds}, "
                  f"safe_limit={ctrl.safe_limit:.1f}, boundary_emergency={ctrl.boundary_emergency:.1f}")

        # Scout death diagnostic
        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            if q_name in infos and infos[q_name].get("death_cause", ""):
                dc = infos[q_name]["death_cause"]
                lx = q_paths[q_name]["x"][-1] if q_paths[q_name]["x"] else "?"
                ly = q_paths[q_name]["y"][-1] if q_paths[q_name]["y"] else "?"
                print(f"  💀 {q_name} died at step {step}: cause={dc}, "
                      f"last_pos=({lx:.1f}, {ly:.1f})")

        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            total_rq[q_name] += rewards.get(q_name, 0.0)
        for f_name in fixed_names:
            total_rf += rewards.get(f_name, 0.0)
        f_alive = any(f in env.sim.drones for f in fixed_names)
        f_pos = None

        # Average water across all alive FW for history
        alive_fw_water = []
        for f_name in fixed_names:
            if f_name in env.sim.drones:
                d = env.sim.drones[f_name]
                alive_fw_water.append(d.current_water / d.water_capacity if d.water_capacity > 0 else 0)
        f_water_pct = sum(alive_fw_water) / max(1, len(alive_fw_water)) if alive_fw_water else 0.0

        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            if q_name in env.sim.drones:
                q_pos = env.sim.drones[q_name].get_position()
                q_vel = env.sim.drones[q_name].get_velocity()
                q_paths[q_name]["x"].append(q_pos[0])
                q_paths[q_name]["y"].append(q_pos[1])
                hist["q_alt"][q_name].append(q_pos[2])
                hist["q_pos"][q_name]["x"].append(q_pos[0])
                hist["q_pos"][q_name]["y"].append(q_pos[1])
                hist["q_pos"][q_name]["z"].append(q_pos[2])
                hist["q_pos"][q_name]["v"].append(np.linalg.norm(q_vel[:2]))
            hist["q_r"][q_name].append(rewards.get(q_name, 0.0))
            # Collect reward component diagnostics
            qi_info = infos.get(q_name, {})
            for dk in ["r_survival", "r_boundary", "r_alt_pen", "r_sweet", "r_ground",
                        "r_approach", "r_compass", "r_fire", "r_abandon",
                        "r_separation", "r_exploration", "dist_to_fire", "fire_intensity"]:
                if dk not in hist["q_diag"][q_name]:
                    hist["q_diag"][q_name][dk] = []
                hist["q_diag"][q_name][dk].append(qi_info.get(dk, 0.0))

        # Track all FW positions, velocity, water
        for f_name in fixed_names:
            if f_name in env.sim.drones:
                fd = env.sim.drones[f_name]
                fp = fd.get_position()
                f_paths[f_name]["x"].append(fp[0])
                f_paths[f_name]["y"].append(fp[1])
                hist["fw"][f_name]["x"].append(fp[0])
                hist["fw"][f_name]["y"].append(fp[1])
                hist["fw"][f_name]["z"].append(fp[2])
                hist["fw"][f_name]["v"].append(getattr(fd, 'state_va', fd.get_speed()))
                hist["water"][f_name].append(fd.current_water / fd.water_capacity if fd.water_capacity > 0 else 0)
                hist["f_r"][f_name].append(rewards.get(f_name, 0.0))
                if f_pos is None:
                    f_pos = fp
            # Collect FW reward component diagnostics
            fi_info = infos.get(f_name, {})
            for dk in ["r_survival", "r_boundary", "r_alt_pen",
                        "r_approach", "r_alt_shape", "fw_dist_scout",
                        "r_spread"]:
                if dk not in hist["f_diag"][f_name]:
                    hist["f_diag"][f_name][dk] = []
                hist["f_diag"][f_name][dk].append(fi_info.get(dk, 0.0))

        # Collect local maps per scout for display
        fire_seen_vals = []
        for qi in range(N_QUADS):
            lm = last_local_maps[f"quad_{qi}"]
            if lm is not None:
                fire_seen_vals.append(float(np.sum(lm)))
        fire_seen = max(fire_seen_vals) if fire_seen_vals else 0.0
        hist["fire"].append(fire_seen)

        if step % GIF_EVERY == 0:
            if save_gif:
                # Gather scout positions
                q_positions = {}
                q_alive_map = {}
                for qi in range(N_QUADS):
                    q_name = f"quad_{qi}"
                    q_alive_map[q_name] = q_name in env.sim.drones
                    q_positions[q_name] = env.sim.drones[q_name].get_position() if q_alive_map[q_name] else None

                # Per-FW positions and water
                f_positions = {}
                f_alive_map_frame = {}
                f_water_pcts = {}
                for f_name in fixed_names:
                    alive = f_name in env.sim.drones
                    f_alive_map_frame[f_name] = alive
                    if alive:
                        f_positions[f_name] = env.sim.drones[f_name].get_position()
                        f_water_pcts[f_name] = env.sim.drones[f_name].current_water / env.sim.drones[f_name].water_capacity
                    else:
                        f_positions[f_name] = None
                        f_water_pcts[f_name] = 0.0

                any_f_alive = any(f_alive_map_frame.values())
                moisture = env.sim.environment.fire_grid.M.copy() \
                    if env.sim.environment.fire_grid is not None else None
                # Valve open status per FW
                f_valve_open = {}
                for f_name in fixed_names:
                    ctrl = cmdr_ctrl.get(f_name)
                    if ctrl is not None:
                        vd = getattr(ctrl, '_valve_debug', None)
                        f_valve_open[f_name] = (vd is not None and vd.get('opened', False))
                    else:
                        f_valve_open[f_name] = False
                frame = _render_frame(
                    step, env.sim.environment.fire_grid.I.copy(), env.map_bounds,
                    q_paths, q_positions, q_alive_map,
                    f_paths, f_positions, f_alive_map_frame, f_water_pcts,
                    refill_zones, refill_size,
                    {q: (last_local_maps[q].copy() if last_local_maps[q] is not None else None)
                     for q in last_local_maps},
                    sum(total_rq.values()), total_rf, fire_seen, any_f_alive,
                    terrain_collections=terrain_collections,
                    moisture_map=moisture,
                    f_valve_open=f_valve_open,
                )
                frames.append(frame)

    if save_gif and frames:
        out_gif = gif_path or os.path.join(project_root, f"demo_seed{seed}.gif")
        imageio.mimsave(out_gif, frames, fps=GIF_FPS, loop=0)
        print(f"  GIF → {out_gif}")
        _save_analysis(hist, project_root, suffix=f"_seed{seed}")

    # Metriky
    end_cells = int(np.sum(env.sim.environment.fire_grid.B)) \
                if env.sim.environment.fire_grid is not None else 0
    supp_pct = (1.0 - end_cells / fire_cells_peak) * 100.0 if fire_cells_peak > 0 else 100.0
    fw_survived = any(f in env.sim.drones for f in fixed_names)
    scouts_survived = {q: q in env.sim.drones for q in quad_names}

    w_acc = (total_water_hits / total_water_drops * 100) if total_water_drops > 0 else 0
    print(f"  Water: {total_water_drops} drops, {total_water_hits} hits ({w_acc:.0f}% accuracy)")
    print(f"  Fire: peak={fire_cells_peak}, end={end_cells}, suppressed={supp_pct:.1f}%")

    return {
        "seed": seed,
        "peak_cells": fire_cells_peak,
        "end_cells": end_cells,
        "supp_pct": supp_pct,
        "fw_survived": fw_survived,
        "scouts_survived": scouts_survived,
        "total_rq": total_rq,
        "total_rf": total_rf,
        "water_used_pct": 0.0,  # per-FW water tracked in hist["water"]
        "water_drops": total_water_drops,
        "water_hits": total_water_hits,
    }

def _save_analysis(hist, project_root, suffix=""):
    _save_scout_pdf(hist, project_root, suffix)
    _save_fw_pdf(hist, project_root, suffix)
    print("Analysis PDFs saved.")


def _save_scout_pdf(hist, project_root, suffix=""):
    """Multi-page PDF: one page per scout with full diagnostics."""
    q_names = sorted(hist["q_r"].keys())
    q_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
    path = os.path.join(project_root, f"demo_scouts{suffix}.pdf")

    with PdfPages(path) as pdf:
        # ── Page 0: Overview — all scouts together ────────────────
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.patch.set_facecolor('white')
        fig.suptitle("SCOUT OVERVIEW", fontsize=14, fontweight='bold')

        # Cumulative reward
        for i, q in enumerate(q_names):
            axes[0].plot(np.cumsum(hist["q_r"][q]), label=q,
                         color=q_colors[i % len(q_colors)], linewidth=1.5)
        axes[0].set_title("Cumulative reward"); axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3); axes[0].set_xlabel("Step")

        # Altitude
        for i, q in enumerate(q_names):
            if hist["q_pos"][q]["z"]:
                axes[1].plot(hist["q_pos"][q]["z"], label=q,
                             color=q_colors[i % len(q_colors)], alpha=0.8)
        axes[1].axhline(60, ls='--', color='gray', alpha=0.5, label='alt_ideal_min')
        axes[1].axhline(120, ls='--', color='gray', alpha=0.5, label='alt_ideal_max')
        axes[1].set_title("Altitude [m]"); axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3); axes[1].set_xlabel("Step")

        # XY speed
        for i, q in enumerate(q_names):
            if hist["q_pos"][q]["v"]:
                axes[2].plot(hist["q_pos"][q]["v"], label=q,
                             color=q_colors[i % len(q_colors)], alpha=0.7)
        axes[2].set_title("XY speed [m/s]"); axes[2].legend(fontsize=7)
        axes[2].grid(alpha=0.3); axes[2].set_xlabel("Step")

        # Fire intensity
        axes[3].fill_between(range(len(hist["fire"])), hist["fire"],
                             color='orange', alpha=0.5)
        axes[3].set_title("Fire intensity under best scout")
        axes[3].grid(alpha=0.3); axes[3].set_xlabel("Step")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ── Per-scout pages ───────────────────────────────────────
        for qi, q in enumerate(q_names):
            c = q_colors[qi % len(q_colors)]
            diag = hist["q_diag"][q]
            fig, axes = plt.subplots(5, 2, figsize=(16, 20))
            fig.patch.set_facecolor('white')
            fig.suptitle(f"{q} — detailed diagnostics", fontsize=14, fontweight='bold')

            # Row 0L: X, Y position
            ax = axes[0, 0]
            if hist["q_pos"][q]["x"]:
                ax.plot(hist["q_pos"][q]["x"], color=c, alpha=0.8, label='x')
                ax.plot(hist["q_pos"][q]["y"], color=c, alpha=0.4, ls='--', label='y')
            ax.set_title("Position X, Y [m]"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            # Row 0R: Altitude
            ax = axes[0, 1]
            if hist["q_pos"][q]["z"]:
                ax.plot(hist["q_pos"][q]["z"], color=c)
                ax.axhline(60, ls=':', color='red', alpha=0.5, label='ideal_min=60')
                ax.axhline(120, ls=':', color='red', alpha=0.5, label='ideal_max=120')
            ax.set_title("Altitude [m]"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            # Row 1L: Speed
            ax = axes[1, 0]
            if hist["q_pos"][q]["v"]:
                ax.plot(hist["q_pos"][q]["v"], color=c, alpha=0.8)
            ax.set_title("XY speed [m/s]"); ax.grid(alpha=0.3)

            # Row 1R: Total reward per step
            ax = axes[1, 1]
            ax.plot(hist["q_r"][q], color=c, alpha=0.5, linewidth=0.8)
            # Smoothed
            if len(hist["q_r"][q]) > 20:
                w = min(50, len(hist["q_r"][q]) // 5)
                smoothed = np.convolve(hist["q_r"][q], np.ones(w)/w, mode='valid')
                ax.plot(range(w-1, w-1+len(smoothed)), smoothed, color='black', linewidth=1.5, label=f'MA-{w}')
                ax.legend(fontsize=7)
            ax.set_title("Total reward / step"); ax.grid(alpha=0.3)

            # Row 2L: Physics rewards stacked
            ax = axes[2, 0]
            physics_keys = ["r_survival", "r_boundary", "r_alt_pen", "r_sweet", "r_ground"]
            physics_colors = ['green', 'red', 'orange', 'cyan', 'brown']
            for pk, pc in zip(physics_keys, physics_colors):
                if pk in diag and diag[pk]:
                    ax.plot(diag[pk], label=pk, color=pc, alpha=0.7, linewidth=0.8)
            ax.set_title("Physics reward components"); ax.legend(fontsize=6); ax.grid(alpha=0.3)

            # Row 2R: Mission rewards
            ax = axes[2, 1]
            mission_keys = ["r_approach", "r_compass", "r_fire", "r_abandon", "r_separation", "r_exploration"]
            mission_colors = ['blue', 'teal', 'orange', 'red', 'purple', 'green']
            for mk, mc in zip(mission_keys, mission_colors):
                if mk in diag and diag[mk]:
                    ax.plot(diag[mk], label=mk, color=mc, alpha=0.7, linewidth=0.8)
            ax.set_title("Mission reward components"); ax.legend(fontsize=6); ax.grid(alpha=0.3)

            # Row 3L: Cumulative reward breakdown
            ax = axes[3, 0]
            all_keys = physics_keys + mission_keys
            all_colors = physics_colors + mission_colors
            for ak, ac in zip(all_keys, all_colors):
                if ak in diag and diag[ak]:
                    ax.plot(np.cumsum(diag[ak]), label=ak, color=ac, alpha=0.7, linewidth=1)
            ax.set_title("Cumulative reward per component"); ax.legend(fontsize=5); ax.grid(alpha=0.3)

            # Row 3R: Distance to fire + fire intensity
            ax = axes[3, 1]
            if "dist_to_fire" in diag and diag["dist_to_fire"]:
                ax.plot(diag["dist_to_fire"], color='red', alpha=0.7, label='dist_to_fire')
                ax.set_ylabel("Distance [m]", color='red')
            ax2 = ax.twinx()
            if "fire_intensity" in diag and diag["fire_intensity"]:
                ax2.plot(diag["fire_intensity"], color='orange', alpha=0.7, label='fire_intensity')
                ax2.set_ylabel("Intensity", color='orange')
            ax.set_title("Distance to fire & intensity"); ax.legend(loc='upper left', fontsize=7)
            ax2.legend(loc='upper right', fontsize=7); ax.grid(alpha=0.3)

            # Row 4: XY trajectory
            ax = axes[4, 0]
            if hist["q_pos"][q]["x"] and hist["q_pos"][q]["y"]:
                ax.plot(hist["q_pos"][q]["x"], hist["q_pos"][q]["y"], color=c, alpha=0.5, linewidth=0.8)
                ax.scatter(hist["q_pos"][q]["x"][0], hist["q_pos"][q]["y"][0],
                           c='green', s=80, marker='o', zorder=5, label='start')
                ax.scatter(hist["q_pos"][q]["x"][-1], hist["q_pos"][q]["y"][-1],
                           c='red', s=80, marker='x', zorder=5, label='end')
            ax.set_aspect('equal'); ax.set_title("XY trajectory"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            axes[4, 1].axis('off')  # empty

            for row in axes:
                for a in row:
                    a.set_xlabel("Step", fontsize=7)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)

    print(f"  Scout PDF → {path}")


def _save_fw_pdf(hist, project_root, suffix=""):
    """Multi-page PDF: one page per FW with full diagnostics."""
    fw_names = sorted(hist["fw"].keys())
    fw_colors = ['#ff3333', '#ff9933', '#cc33ff']
    path = os.path.join(project_root, f"demo_fw{suffix}.pdf")

    with PdfPages(path) as pdf:
        # ── Page 0: Overview — all FW together ────────────────────
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.patch.set_facecolor('white')
        fig.suptitle("FIXED-WING OVERVIEW", fontsize=14, fontweight='bold')

        # Cumulative reward
        for i, f in enumerate(fw_names):
            if hist["f_r"][f]:
                axes[0].plot(np.cumsum(hist["f_r"][f]), label=f,
                             color=fw_colors[i % len(fw_colors)], linewidth=1.5)
        axes[0].set_title("Cumulative reward"); axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3); axes[0].set_xlabel("Step")

        # Altitude
        for i, f in enumerate(fw_names):
            if hist["fw"][f]["z"]:
                axes[1].plot(hist["fw"][f]["z"], label=f,
                             color=fw_colors[i % len(fw_colors)], alpha=0.8)
        axes[1].axhline(30, ls='--', color='gray', alpha=0.5, label='alt_min=30')
        axes[1].axhline(80, ls='--', color='gray', alpha=0.5, label='alt_max=80')
        axes[1].set_title("Altitude [m]"); axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3); axes[1].set_xlabel("Step")

        # Airspeed
        for i, f in enumerate(fw_names):
            if hist["fw"][f]["v"]:
                axes[2].plot(hist["fw"][f]["v"], label=f,
                             color=fw_colors[i % len(fw_colors)], alpha=0.8)
        axes[2].set_title("Airspeed [m/s]"); axes[2].legend(fontsize=7)
        axes[2].grid(alpha=0.3); axes[2].set_xlabel("Step")

        # Water level
        for i, f in enumerate(fw_names):
            if hist["water"][f]:
                axes[3].plot(hist["water"][f], label=f,
                             color=fw_colors[i % len(fw_colors)], linewidth=1.5)
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].set_title("Water level"); axes[3].legend(fontsize=7)
        axes[3].grid(alpha=0.3); axes[3].set_xlabel("Step")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ── Per-FW pages ──────────────────────────────────────────
        for fi, f in enumerate(fw_names):
            c = fw_colors[fi % len(fw_colors)]
            fw = hist["fw"][f]
            diag = hist["f_diag"][f]

            fig, axes = plt.subplots(5, 2, figsize=(16, 20))
            fig.patch.set_facecolor('white')
            fig.suptitle(f"{f} — detailed diagnostics", fontsize=14, fontweight='bold')

            # Row 0L: X, Y position
            ax = axes[0, 0]
            if fw["x"]:
                ax.plot(fw["x"], color=c, alpha=0.8, label='x')
                ax.plot(fw["y"], color=c, alpha=0.4, ls='--', label='y')
            ax.set_title("Position X, Y [m]"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            # Row 0R: Altitude
            ax = axes[0, 1]
            if fw["z"]:
                ax.plot(fw["z"], color=c)
                ax.axhline(30, ls=':', color='red', alpha=0.5, label='min=30')
                ax.axhline(80, ls=':', color='red', alpha=0.5, label='max=80')
            ax.set_title("Altitude [m]"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            # Row 1L: Airspeed
            ax = axes[1, 0]
            if fw["v"]:
                ax.plot(fw["v"], color=c, alpha=0.8)
            ax.set_title("Airspeed [m/s]"); ax.grid(alpha=0.3)

            # Row 1R: Total reward per step
            ax = axes[1, 1]
            if hist["f_r"][f]:
                ax.plot(hist["f_r"][f], color=c, alpha=0.5, linewidth=0.8)
                if len(hist["f_r"][f]) > 20:
                    w = min(50, len(hist["f_r"][f]) // 5)
                    sm = np.convolve(hist["f_r"][f], np.ones(w)/w, mode='valid')
                    ax.plot(range(w-1, w-1+len(sm)), sm, color='black', linewidth=1.5, label=f'MA-{w}')
                    ax.legend(fontsize=7)
            ax.set_title("Total reward / step"); ax.grid(alpha=0.3)

            # Row 2L: Physics rewards
            ax = axes[2, 0]
            physics_keys = ["r_survival", "r_boundary", "r_alt_pen"]
            physics_colors_l = ['green', 'red', 'orange']
            for pk, pc in zip(physics_keys, physics_colors_l):
                if pk in diag and diag[pk]:
                    ax.plot(diag[pk], label=pk, color=pc, alpha=0.7, linewidth=0.8)
            ax.set_title("Physics reward components"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            # Row 2R: Mission rewards
            ax = axes[2, 1]
            mission_keys = ["r_approach", "r_alt_shape", "r_spread"]
            mission_colors_l = ['blue', 'cyan', 'red']
            for mk, mc in zip(mission_keys, mission_colors_l):
                if mk in diag and diag[mk]:
                    ax.plot(diag[mk], label=mk, color=mc, alpha=0.7, linewidth=0.8)
            ax.set_title("Mission reward components"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            # Row 3L: Cumulative reward breakdown
            ax = axes[3, 0]
            all_k = physics_keys + mission_keys
            all_c = physics_colors_l + mission_colors_l
            for ak, ac in zip(all_k, all_c):
                if ak in diag and diag[ak]:
                    ax.plot(np.cumsum(diag[ak]), label=ak, color=ac, alpha=0.7, linewidth=1)
            ax.set_title("Cumulative reward per component"); ax.legend(fontsize=6); ax.grid(alpha=0.3)

            # Row 3R: Distance to scout + water level
            ax = axes[3, 1]
            if "fw_dist_scout" in diag and diag["fw_dist_scout"]:
                ax.plot(diag["fw_dist_scout"], color='red', alpha=0.7, label='dist_to_scout')
                ax.set_ylabel("Distance [m]", color='red')
            ax2r = ax.twinx()
            if hist["water"][f]:
                ax2r.plot(hist["water"][f], color='deepskyblue', alpha=0.7, label='water')
                ax2r.set_ylabel("Water frac", color='deepskyblue')
                ax2r.set_ylim(-0.05, 1.05)
            ax.set_title("Distance to scout & water"); ax.legend(loc='upper left', fontsize=7)
            ax2r.legend(loc='upper right', fontsize=7); ax.grid(alpha=0.3)

            # Row 4L: XY trajectory
            ax = axes[4, 0]
            if fw["x"] and fw["y"]:
                ax.plot(fw["x"], fw["y"], color=c, alpha=0.5, linewidth=0.8)
                ax.scatter(fw["x"][0], fw["y"][0], c='green', s=80, marker='o', zorder=5, label='start')
                ax.scatter(fw["x"][-1], fw["y"][-1], c='red', s=80, marker='x', zorder=5, label='end')
            ax.set_aspect('equal'); ax.set_title("XY trajectory"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

            # Row 4R: Water level
            ax = axes[4, 1]
            if hist["water"][f]:
                ax.fill_between(range(len(hist["water"][f])), hist["water"][f],
                                color='deepskyblue', alpha=0.5)
                ax.plot(hist["water"][f], color='deepskyblue', linewidth=1.5)
                ax.set_ylim(-0.05, 1.05)
            ax.set_title("Water level"); ax.grid(alpha=0.3)

            for row in axes:
                for a in row:
                    a.set_xlabel("Step", fontsize=7)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)

    print(f"  FW PDF → {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scout",      default=MODEL_SCOUT)
    parser.add_argument("--commander",  default=MODEL_COMMANDER)
    parser.add_argument("--episodes",   type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=EPISODE_SEED)
    parser.add_argument("--ep-num",     type=int, default=30000,
                        help="Curriculum episode number (30000 = full difficulty)")
    parser.add_argument("--gif-all",    action="store_true",
                        help="Uložit GIF pro všechny epizody (default: jen první)")
    parser.add_argument("--scout-dist", type=float, default=10.0,
                        help="Spawn scouts this many metres from fire (default: 10, matching training)")
    parser.add_argument("--fw-dist",    type=float, default=None,
                        help="Spawn FW this many metres from fire (default: env random)")
    parser.add_argument("--init-water",  type=float, default=None,
                        help="Override FW initial water [L] (default: 0 = empty, matching training)")
    parser.add_argument("--grid-size",   type=float, default=None,
                        help="Override map size in metres (e.g. 300 for 300x300m)")
    parser.add_argument("--n-quads",     type=int, default=N_QUADS,
                        help="Number of scout quadcopters (default: 3)")
    parser.add_argument("--n-fixed",     type=int, default=N_FIXED,
                        help="Number of fixed-wing commanders (default: 2)")
    parser.add_argument("--n-fires",     type=int, default=1,
                        help="Number of fires to place (default: 1)")
    args = parser.parse_args()

    # Přepis cestám z argumentu
    MODEL_SCOUT     = args.scout
    MODEL_COMMANDER = args.commander
    N_QUADS         = args.n_quads
    N_FIXED         = args.n_fixed
    grid_size_demo  = args.grid_size if args.grid_size is not None else GRID_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scout_actor, commander_actor, _, _ = _load_models(device)

    # Terrain (načte se jednou)
    terrain_collections = None
    if USE_OSM:
        print("Loading OSM terrain...")
        gdfs = _load_terrain_gdfs()
        if gdfs:
            water, buildings, forests = _classify_features(gdfs)
            terrain_collections = _build_terrain_collections(water, buildings, forests)

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=grid_size_demo,
                       max_steps=MAX_STEPS, use_osm=USE_OSM, osm_lat=OSM_LAT,
                       osm_lon=OSM_LON, osm_cache_dir=OSM_CACHE,
                       n_fires_range=(args.n_fires, args.n_fires))

    print(f"\nSpouštím {args.episodes} epizod, seed {args.seed_start}–"
          f"{args.seed_start + args.episodes - 1}\n")

    results = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        save_gif = (i == 0) or args.gif_all
        gif_path = os.path.join(project_root, f"demo_seed{seed}.gif") if save_gif else None
        m = run_episode(env, scout_actor, commander_actor, seed, args.ep_num,
                        device, terrain_collections=terrain_collections,
                        save_gif=save_gif, gif_path=gif_path,
                        scout_dist=args.scout_dist, fw_dist=args.fw_dist,
                        init_water=args.init_water)
        results.append(m)
        fw_icon  = "✓" if m["fw_survived"] else "✗"
        s_icons  = " ".join("✓" if v else "✗" for v in m["scouts_survived"].values())
        print(f"  [{i+1:2d}/{args.episodes}] seed={seed:3d} | "
              f"S={s_icons} FW={fw_icon} | "
              f"peak={m['peak_cells']:4d} end={m['end_cells']:4d} "
              f"supp={m['supp_pct']:5.1f}% | "
              f"R_cmdr={m['total_rf']:+.1f}")

    env.sim.stop_simulation()

    # Souhrnná tabulka
    n = len(results)
    print(f"\n{'='*70}")
    print(f"  SOUHRN  ({n} epizod, seeds {args.seed_start}–{args.seed_start+n-1})")
    print(f"{'='*70}")
    fw_surv = sum(1 for r in results if r["fw_survived"])
    all_s_surv = sum(1 for r in results if all(r["scouts_survived"].values()))
    avg_supp = float(np.mean([r["supp_pct"] for r in results]))
    avg_rf   = float(np.mean([r["total_rf"] for r in results]))
    avg_peak = float(np.mean([r["peak_cells"] for r in results]))
    print(f"  Přežil FW           : {fw_surv}/{n} ({fw_surv/n*100:.0f}%)")
    print(f"  Přežili oba scouti : {all_s_surv}/{n} ({all_s_surv/n*100:.0f}%)")
    print(f"  Prům. potlačení   : {avg_supp:.1f}%")
    print(f"  Prům. peak buněk  : {avg_peak:.0f}")
    print(f"  Prům. R_cmdr      : {avg_rf:+.1f}")