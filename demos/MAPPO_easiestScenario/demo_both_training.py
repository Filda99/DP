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
import os, sys, glob
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
MODEL_SCOUT     = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/multi/scout_b0670.pt"
MODEL_COMMANDER = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/multi/cmdr_b0670.pt"

N_QUADS    = 2
N_FIXED    = 1
MAX_STEPS  = 1000
GRID_SIZE  = 1200.0
GIF_EVERY  = 3
GIF_FPS    = 15
EPISODE_SEED = 111
USE_OSM      = True
OSM_LAT      = 49.35
OSM_LON      = 16.42
OSM_CACHE    = os.path.join(project_root, "data")

# Commander waypoint parameters (must match training)
WAYPOINT_RANGE  = 200.0   # metres per unit of dx/dy
WAYPOINT_STEPS  = 30      # physics steps per waypoint segment
WP_REACHED_DIST = 30.0    # metres
# ============================================================


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


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
                  f_path_x, f_path_y, f_pos, f_water_pct,
                  refill_pos, refill_size,
                  local_map_np,
                  total_reward_q, total_reward_f,
                  fire_seen_sum, f_alive,
                  terrain_collections=None):

    fig = plt.figure(figsize=(12, 6), facecolor='white')
    gs  = gridspec.GridSpec(2, 2, width_ratios=[2, 1], hspace=0.35, wspace=0.25,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Panel 1: Global map ─────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(CLR_GRASS)

    # OSM terrain background
    if terrain_collections:
        _add_terrain_to_ax(ax_map, terrain_collections)

    # Fire
    extent = [-b, b, -b, b]
    fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
    ax_map.imshow(fire_masked, extent=extent, origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.75, zorder=4)

    # Refill zone
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
            rect = plt.Rectangle((q_pos[0] - q_fov/2, q_pos[1] - q_fov/2), q_fov, q_fov,
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

    # ── Panel 2: Scout camera ───────────────────────────────────────────
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_cam.set_facecolor('#fafafa')
    if local_map_np is not None:
        ax_cam.imshow(local_map_np, origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
    ax_cam.set_title("Scout camera (32×32)", fontsize=9, fontweight='bold')
    ax_cam.set_xticks([]); ax_cam.set_yticks([])

    # ── Panel 3: Stats + Water Bar ──────────────────────────────────────
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.set_facecolor('#fafafa')
    ax_stats.set_xticks([]); ax_stats.set_yticks([])

    # Water bar
    ax_stats.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.08,
                       color='#dddddd', transform=ax_stats.transAxes))
    ax_stats.add_patch(plt.Rectangle((0.1, 0.15), 0.8 * f_water_pct, 0.08,
                       color='deepskyblue', transform=ax_stats.transAxes))

    status = 'TANK FULL' if f_water_pct > 0.9 else 'REFILL NEEDED' if f_water_pct < 0.2 else 'OPERATIONAL'
    stats_text = (
        f"Scout reward:      {total_reward_q:+.1f}\n"
        f"Commander reward:  {total_reward_f:+.1f}\n\n"
        f"Fire visibility:   {fire_seen_sum:.2f}\n"
        f"Water level:       {f_water_pct*100:3.1f}%\n\n"
        f"Status: {status}"
    )
    ax_stats.text(0.08, 0.9, stats_text, color='#222222', fontsize=8.5,
                  va='top', transform=ax_stats.transAxes, fontfamily='monospace')
    ax_stats.set_title("Mission stats", fontsize=9, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr

def run_demo():
    print("Demo: Heterogeneous MAPPO team (Scout + Commander)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scout_actor, commander_actor, _, _ = _load_models(device)

    # Load OSM terrain for visualisation
    terrain_collections = None
    if USE_OSM:
        print("Loading OSM terrain...")
        gdfs = _load_terrain_gdfs()
        if gdfs:
            water, buildings, forests = _classify_features(gdfs)
            terrain_collections = _build_terrain_collections(water, buildings, forests)
            print(f"  Water: {len(water)}  Buildings: {len(buildings)}  Forest: {len(forests)}")

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=GRID_SIZE,
                       max_steps=MAX_STEPS, use_osm=USE_OSM, osm_lat=OSM_LAT,
                       osm_lon=OSM_LON, osm_cache_dir=OSM_CACHE)
    obs, _ = env.reset(seed=EPISODE_SEED, epizode_number=30000)
    
    refill_info = env.sim.environment.refill_zone
    refill_pos = refill_info['position'] if refill_info else None
    refill_size = refill_info['size'] if refill_info else 20.0

    safe_limit = max(50.0, env.map_bounds * 0.7)           # must match training
    boundary_emergency = max(50.0, env.map_bounds * 0.6)   # must match training
    print(f"📐 map_bounds={env.map_bounds}, safe_limit={safe_limit}, "
          f"boundary_emergency={boundary_emergency}")

    h_scout = {f"quad_{i}": torch.zeros(1, 1, 128).to(device) for i in range(N_QUADS)}
    h_cmdr  = torch.zeros(1, 1, 64).to(device)

    hist = { "q_r": {f"quad_{i}": [] for i in range(N_QUADS)},
             "f_r": [], "q_alt": {f"quad_{i}": [] for i in range(N_QUADS)},
             "f_alt": [], "fire": [], "water": [] }
    frames, total_rf = [], 0.0
    total_rq = {f"quad_{i}": 0.0 for i in range(N_QUADS)}
    q_paths = {f"quad_{i}": {"x": [], "y": []} for i in range(N_QUADS)}
    f_path_x, f_path_y = [], []
    last_local_maps = {f"quad_{i}": None for i in range(N_QUADS)}

    # Commander waypoint state
    need_new_waypoint = True
    target_x, target_y = 0.0, 0.0
    target_alt_raw, water_raw = 0.0, -0.5
    steps_in_segment = 0
    # Per-scout message tracking: latest + best-fire
    scout_msgs = {f"quad_{i}": {"latest": torch.zeros(1, 5).to(device),
                                 "best": torch.zeros(1, 5).to(device),
                                 "valid": False, "best_intensity": -1.0}
                  for i in range(N_QUADS)}

    print("🚀 Mise začíná...")
    for step in tqdm.tqdm(range(MAX_STEPS)):
        if not env.agents: break
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

        # ── Commander (waypoint mode) ────────────────────────────────────
        if "fixed_0" in env.agents:
            drone = env.sim.drones.get("fixed_0")

            # Boundary emergency check FIRST — overrides NN
            in_boundary_emergency = False
            if drone is not None:
                pos = drone.get_position()
                if abs(pos[0]) > boundary_emergency or abs(pos[1]) > boundary_emergency:
                    target_x = 0.0
                    target_y = 0.0
                    in_boundary_emergency = True

            if not in_boundary_emergency:
                if drone is not None:
                    dx_to = target_x - pos[0]
                    dy_to = target_y - pos[1]
                    dist_to = np.sqrt(dx_to**2 + dy_to**2)
                    if dist_to < WP_REACHED_DIST:
                        need_new_waypoint = True

                if steps_in_segment >= WAYPOINT_STEPS:
                    need_new_waypoint = True

                if need_new_waypoint:
                    s_st_f = torch.FloatTensor(obs["fixed_0"]["self_state"]).to(device).unsqueeze(0)

                    # Build 4-slot message tensor: [latest_q0, best_q0, latest_q1, best_q1]
                    msgs_for_cmdr = []
                    masks_for_cmdr = []
                    for qi in range(N_QUADS):
                        sm = scout_msgs[f"quad_{qi}"]
                        msgs_for_cmdr.append(sm["latest"])   # each is [1, 5]
                        masks_for_cmdr.append(not sm["valid"])
                        # msgs_for_cmdr.append(sm["best"])     # each is [1, 5]
                        # masks_for_cmdr.append(not sm["valid"])

                    # msgs_t = torch.stack(msgs_for_cmdr, dim=1)    # [1, 2*N_QUADS, 5]
                    # msgs_m = torch.tensor([masks_for_cmdr])       # [1, 2*N_QUADS]
                    msgs_t = torch.stack(msgs_for_cmdr, dim=1).to(device) 
                    msgs_m = torch.tensor([masks_for_cmdr], dtype=torch.bool).to(device)

                    with torch.no_grad():
                        dist_c, aux_pred, h_cmdr = commander_actor(s_st_f, msgs_t, msgs_m, h_cmdr)
                    act_np = dist_c.mean.squeeze(0).cpu().numpy()
                    std_np = dist_c.stddev.squeeze(0).cpu().numpy()
                    dx_raw = float(act_np[0])
                    dy_raw = float(act_np[1])
                    target_alt_raw = float(act_np[2])
                    water_raw = float(act_np[3])

                    cur_pos = drone.get_position() if drone else np.zeros(3)
                    target_x = np.clip(cur_pos[0] + dx_raw * WAYPOINT_RANGE, -safe_limit, safe_limit)
                    target_y = np.clip(cur_pos[1] + dy_raw * WAYPOINT_RANGE, -safe_limit, safe_limit)
                    # print(f"  [WP step={step}] pos=({cur_pos[0]:.0f},{cur_pos[1]:.0f},{cur_pos[2]:.0f}) "
                    #         f"mean=({act_np}) std=({std_np}) "
                    #         f"aux={aux_pred.squeeze().cpu().numpy()} "
                    #         f"→ target=({target_x:.0f},{target_y:.0f})")
                    steps_in_segment = 0
                    need_new_waypoint = False

            # Heading controller → physical action
            if drone is not None:
                pos = drone.get_position()
                dx_to = target_x - pos[0]
                dy_to = target_y - pos[1]
                dist_to = np.sqrt(dx_to**2 + dy_to**2)
                if dist_to > 1.0:
                    desired_heading = np.arctan2(dy_to, dx_to)
                    cur_yaw = drone.get_orientation_rpy()[2]
                    heading_error = _wrap_angle(desired_heading - cur_yaw)
                    heading_cmd = np.clip(heading_error / np.pi, -1.0, 1.0)
                else:
                    heading_cmd = 0.0

                actions["fixed_0"] = np.array(
                    [heading_cmd, target_alt_raw, water_raw], dtype=np.float32)

            steps_in_segment += 1

        obs, rewards, _, _, infos = env.step(actions)

        # Commander death diagnostic
        if "fixed_0" in infos and infos["fixed_0"].get("death_cause", ""):
            dc = infos["fixed_0"]["death_cause"]
            last_pos = f_path_x[-1] if f_path_x else "?"
            last_pos_y = f_path_y[-1] if f_path_y else "?"
            print(f"\n💀 Commander died at step {step}: cause={dc}, "
                  f"last_pos=({last_pos:.1f}, {last_pos_y:.1f}), "
                  f"map_bounds={env.map_bounds}, "
                  f"safe_limit={safe_limit:.1f}, boundary_emergency={boundary_emergency:.1f}")

        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            total_rq[q_name] += rewards.get(q_name, 0.0)
        total_rf += rewards.get("fixed_0", 0.0)

        # Sběr dat pro vizualizaci
        f_water_pct = 0.0
        f_alive = "fixed_0" in env.sim.drones
        f_pos = None

        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            if q_name in env.sim.drones:
                q_pos = env.sim.drones[q_name].get_position()
                q_paths[q_name]["x"].append(q_pos[0])
                q_paths[q_name]["y"].append(q_pos[1])
                hist["q_alt"][q_name].append(q_pos[2])
            hist["q_r"][q_name].append(rewards.get(q_name, 0.0))

        if f_alive:
            f_pos = env.sim.drones["fixed_0"].get_position()
            f_path_x.append(f_pos[0]); f_path_y.append(f_pos[1])
            f_water_pct = env.sim.drones["fixed_0"].current_water / env.sim.drones["fixed_0"].water_capacity

        # Use best local map from any scout for display
        best_local_map = None
        best_fire_sum = -1
        for qi in range(N_QUADS):
            lm = last_local_maps[f"quad_{qi}"]
            if lm is not None:
                s = float(np.sum(lm))
                if s > best_fire_sum:
                    best_fire_sum = s
                    best_local_map = lm

        fire_seen = best_fire_sum if best_fire_sum >= 0 else 0.0
        hist["f_r"].append(rewards.get("fixed_0", 0.0))
        hist["fire"].append(fire_seen)
        hist["water"].append(f_water_pct)
        if f_alive: hist["f_alt"].append(f_pos[2])

        if step % GIF_EVERY == 0:
            # Gather scout positions
            q_positions = {}
            q_alive_map = {}
            for qi in range(N_QUADS):
                q_name = f"quad_{qi}"
                q_alive_map[q_name] = q_name in env.sim.drones
                q_positions[q_name] = env.sim.drones[q_name].get_position() if q_alive_map[q_name] else None

            frame = _render_frame(
                step, env.sim.environment.fire_grid.I.copy(), env.map_bounds,
                q_paths, q_positions, q_alive_map,
                list(f_path_x), list(f_path_y), f_pos, f_water_pct,
                refill_pos, refill_size,
                best_local_map.copy() if best_local_map is not None else None,
                sum(total_rq.values()), total_rf, fire_seen, f_alive,
                terrain_collections=terrain_collections,
            )
            frames.append(frame)

    imageio.mimsave(os.path.join(project_root, "demo_training.gif"), frames, fps=GIF_FPS, loop=0)
    _save_analysis(hist, project_root)

def _save_analysis(hist, project_root):
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    fig.patch.set_facecolor('white')

    axes[0].plot(np.cumsum(hist["f_r"]), label="Commander", color=CLR_CMDR)
    for q_name in hist["q_r"]:
        axes[0].plot(np.cumsum(hist["q_r"][q_name]), label=q_name, color=CLR_SCOUT, alpha=0.7)
    axes[0].set_title("Cumulative reward", fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.3)

    for q_name in hist["q_alt"]:
        axes[1].plot(hist["q_alt"][q_name], color=CLR_SCOUT, alpha=0.7, label=q_name)
    axes[1].plot(hist["f_alt"], color=CLR_CMDR, label="Commander")
    axes[1].set_title("Altitude [m]", fontweight='bold'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].fill_between(range(len(hist["fire"])), hist["fire"], color='orange', alpha=0.5)
    axes[2].set_title("Fire intensity under scout", fontweight='bold'); axes[2].grid(alpha=0.3)

    axes[3].plot(hist["water"], color='deepskyblue', linewidth=2)
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_title("Commander water level", fontweight='bold'); axes[3].grid(alpha=0.3)

    axes[4].set_title("Scout reward per step", fontweight='bold'); axes[4].grid(alpha=0.3)
    for q_name in hist["q_r"]:
        axes[4].plot(hist["q_r"][q_name], color=CLR_SCOUT, alpha=0.3)

    axes[5].plot(hist["f_r"], color=CLR_CMDR, alpha=0.4)
    axes[5].set_title("Commander reward per step", fontweight='bold'); axes[5].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Step', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "demo_training_analysis.png"), dpi=120,
                facecolor='white', edgecolor='none')
    print("Analysis saved.")

if __name__ == "__main__":
    run_demo()