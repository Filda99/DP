"""
demo_final_collage.py
─────────────────────
Run full-team demo (scouts + FW) and save individual frames at checkpoint steps
plus a final collage image. For thesis results.

Usage:
    python demos/full_team/demo_final_collage.py
"""

import torch
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from PIL import Image
import tqdm, io

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
MODEL_SCOUT     = os.path.join(project_root, "saved_models/v10_finetune/scout_best.pt")
MODEL_COMMANDER = os.path.join(project_root, "saved_models/v10_finetune/cmdr_best.pt")

N_QUADS    = 2
N_FIXED    = 3
MAX_STEPS  = 2000
GRID_SIZE  = 1500.0
N_FIRES    = 2
SCOUT_SPAWN_DIST = 200.0   # metres from fire — close but outside FOV (~120m at alt 80)
EPISODE_SEEDS = [213]

USE_OSM    = True
OSM_LAT    = 49.35
OSM_LON    = 16.42
OSM_CACHE  = os.path.join(project_root, "data")

# Commander waypoint parameters (must match training)
WAYPOINT_RANGE  = 50.0
WAYPOINT_STEPS  = 30
WP_REACHED_DIST = 30.0

# Checkpoint steps to save frames (every 250 steps, 8 total)
CHECKPOINT_STEPS = [0, 249, 499, 749, 999, 1249, 1499, 1749, 1999]

OUTPUT_BASE = os.path.join(project_root, "results/TrainingResultsFinal/both_2s3fw_1500m")
# ============================================================


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

    n_scouts = len(q_positions)
    n_fw = len(f_positions)
    right_rows = max(n_scouts, 1) + 1

    fig = plt.figure(figsize=(14, 7), facecolor='white')
    gs  = gridspec.GridSpec(right_rows, 2, width_ratios=[2.2, 1],
                            hspace=0.4, wspace=0.25,
                            left=0.05, right=0.97, top=0.93, bottom=0.05)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(CLR_GRASS)

    if terrain_collections:
        _add_terrain_to_ax(ax_map, terrain_collections)

    extent = [-b, b, -b, b]
    if moisture_map is not None:
        moist_masked = np.ma.masked_where(moisture_map < 0.01, moisture_map)
        ax_map.imshow(moist_masked, extent=extent, origin='lower',
                      cmap='Blues', vmin=0, vmax=1.0, alpha=0.55, zorder=3)

    fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
    ax_map.imshow(fire_masked, extent=extent, origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.75, zorder=4)

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

    # Agent positions — FW
    for idx, f_name in enumerate(f_positions):
        if f_alive_map.get(f_name, False) and f_positions[f_name] is not None:
            fp = f_positions[f_name]
            c = fw_colors[idx % len(fw_colors)]
            ax_map.scatter(fp[0], fp[1], c=c, s=130, marker='>',
                           edgecolors='black', linewidths=0.6, zorder=7)
            if f_valve_open and f_valve_open.get(f_name, False):
                ring = plt.Circle((fp[0], fp[1]), 12,
                                  color='deepskyblue', fill=False,
                                  linewidth=2.5, alpha=0.9, zorder=8)
                ax_map.add_patch(ring)

    ax_map.set_xlim(-b, b); ax_map.set_ylim(-b, b)
    ax_map.set_aspect('equal')
    ax_map.set_title(f"Step {step:04d}  |  {N_QUADS} Scouts + {N_FIXED} FW  |  {int(GRID_SIZE)}×{int(GRID_SIZE)} m",
                     fontsize=11, fontweight='bold')
    ax_map.set_xlabel('X [m]', fontsize=8); ax_map.set_ylabel('Y [m]', fontsize=8)
    ax_map.tick_params(labelsize=7)

    leg_h = []
    if terrain_collections:
        leg_h.append(mpatches.Patch(color=CLR_FOREST,   label='Forest'))
        leg_h.append(mpatches.Patch(color=CLR_BUILDING, label='Building'))
        leg_h.append(mpatches.Patch(color=CLR_WATER,    label='Water'))
    leg_h.append(plt.Line2D([0],[0], marker='^', color='w', markerfacecolor=CLR_SCOUT,
                            markersize=7, label='Scout'))
    leg_h.append(plt.Line2D([0],[0], marker='>', color='w', markerfacecolor=CLR_CMDR,
                            markersize=7, label='FW Commander'))
    ax_map.legend(handles=leg_h, loc='upper right', fontsize=6.5,
                  framealpha=0.85, edgecolor='#ccc')

    # Scout cameras
    scout_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#bcbd22']
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

    # Stats panel
    ax_stats = fig.add_subplot(gs[n_scouts:, 1])
    ax_stats.set_facecolor('#fafafa')
    ax_stats.set_xticks([]); ax_stats.set_yticks([])

    bar_h = 0.04
    bar_gap = 0.055
    bar_top = 0.38
    for fi, f_name in enumerate(sorted(f_water_pcts.keys())):
        alive = f_alive_map.get(f_name, False)
        wpct = f_water_pcts[f_name] if alive else 0.0
        y_pos = bar_top - fi * bar_gap
        c = fw_colors[fi % len(fw_colors)]
        ax_stats.add_patch(plt.Rectangle((0.22, y_pos), 0.7, bar_h,
                           color='#dddddd', transform=ax_stats.transAxes))
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
    plt.savefig(buf, format='png', dpi=120, facecolor='white')
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


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
            print(f"  {name}: {path}")
        else:
            print(f"  {name}: NOT FOUND ({path})")

    scout.eval()
    cmdr.eval()
    return scout, cmdr


def run_seed(seed, device, scout_actor, cmdr_actor, terrain_collections, output_dir):
    """Run one seed and save checkpoint frames + collage."""
    os.makedirs(output_dir, exist_ok=True)

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=GRID_SIZE,
                       max_steps=MAX_STEPS, use_osm=USE_OSM, osm_lat=OSM_LAT,
                       osm_lon=OSM_LON, osm_cache_dir=OSM_CACHE,
                       n_fires_range=(N_FIRES, N_FIRES))

    quad_names = [f"quad_{i}" for i in range(N_QUADS)]
    fixed_names = [f"fixed_{i}" for i in range(N_FIXED)]

    obs, _ = env.reset(seed=seed, epizode_number=30000)

    # FW start FULL (matching training: d.current_water = d.water_capacity)
    for a in env.fixed_agents:
        if a in env.sim.drones:
            d = env.sim.drones[a]
            if d.water_capacity > 0:
                d.current_water = d.water_capacity
                env._prev_fw_water[a] = 1.0

    # Teleport scouts close to fire (but outside FOV so frame 0 shows no fire)
    import pybullet as _p
    fire_x, fire_y = env.fire_x, env.fire_y
    rng = np.random.default_rng(seed + 1000)
    for qi in range(N_QUADS):
        q_name = f"quad_{qi}"
        if q_name in env.sim.drones:
            angle = 2 * np.pi * qi / N_QUADS + rng.uniform(-0.3, 0.3)
            nx = fire_x + SCOUT_SPAWN_DIST * np.cos(angle)
            ny = fire_y + SCOUT_SPAWN_DIST * np.sin(angle)
            nx = float(np.clip(nx, -env.map_bounds * 0.85, env.map_bounds * 0.85))
            ny = float(np.clip(ny, -env.map_bounds * 0.85, env.map_bounds * 0.85))
            d = env.sim.drones[q_name]
            _p.resetBasePositionAndOrientation(
                d.drone_id, [nx, ny, 80.0], [0, 0, 0, 1])

    obs = {a: env._get_obs(a) for a in env.agents}

    refill_zones = getattr(env.sim.environment, 'refill_zones', [])
    refill_info = env.sim.environment.refill_zone
    if not refill_zones and refill_info:
        refill_zones = [refill_info]
    refill_size = refill_zones[0].get('size', 20.0) if refill_zones else 20.0

    print(f"  map_bounds={env.map_bounds}")
    print(f"  fire=({env.fire_x:.0f}, {env.fire_y:.0f})")
    print(f"  Refill zones: {len(refill_zones)}")

    h_scout = {q: torch.zeros(1, 1, 128).to(device) for q in quad_names}
    h_cmdr  = {f: torch.zeros(1, 1, 64).to(device) for f in fixed_names}

    total_rq = {q: 0.0 for q in quad_names}
    total_rf = 0.0
    q_paths = {q: {"x": [], "y": []} for q in quad_names}
    f_paths = {f: {"x": [], "y": []} for f in fixed_names}
    last_local_maps = {q: None for q in quad_names}

    cmdr_ctrl = {f: CommanderController(WAYPOINT_RANGE, WAYPOINT_STEPS, WP_REACHED_DIST)
                 for f in fixed_names}
    for f in fixed_names:
        cmdr_ctrl[f].reset(env.map_bounds)

    scout_msgs = {q: {"latest": torch.zeros(1, 5).to(device),
                      "best": torch.zeros(1, 5).to(device),
                      "valid": False, "best_intensity": -1.0}
                  for q in quad_names}

    saved_frames = {}  # step -> numpy array

    print(f"\n  Running {MAX_STEPS} steps...")
    for step in tqdm.tqdm(range(MAX_STEPS), desc="Episode"):
        if not env.agents:
            break

        actions = {}

        # ── Scouts ───────────────────────────────────────────
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
                sm = scout_msgs[q_name]
                sm["latest"] = scout_msg
                sm["valid"] = True
                intensity = scout_msg[0, 2].item()
                if intensity > sm["best_intensity"]:
                    sm["best_intensity"] = intensity
                    sm["best"] = scout_msg

        # ── Commander FW ─────────────────────────────────────
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
                        cmdr_actor, h_cmdr[f_name], msgs_t, msgs_m,
                        deterministic=True,
                        fw_neighbor_states=fw_n_t,
                        fw_neighbor_mask=fw_nm_t)
                    actions[f_name] = action

        obs, rewards, _, _, infos = env.step(actions)

        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            total_rq[q_name] += rewards.get(q_name, 0.0)
        for f_name in fixed_names:
            total_rf += rewards.get(f_name, 0.0)

        # Track paths
        for qi in range(N_QUADS):
            q_name = f"quad_{qi}"
            if q_name in env.sim.drones:
                q_pos = env.sim.drones[q_name].get_position()
                q_paths[q_name]["x"].append(q_pos[0])
                q_paths[q_name]["y"].append(q_pos[1])

        for f_name in fixed_names:
            if f_name in env.sim.drones:
                fp = env.sim.drones[f_name].get_position()
                f_paths[f_name]["x"].append(fp[0])
                f_paths[f_name]["y"].append(fp[1])

        # ── Save frame at checkpoint steps ───────────────────
        if step in CHECKPOINT_STEPS:
            q_positions = {}
            q_alive_map = {}
            for qi in range(N_QUADS):
                q_name = f"quad_{qi}"
                q_alive_map[q_name] = q_name in env.sim.drones
                q_positions[q_name] = env.sim.drones[q_name].get_position() if q_alive_map[q_name] else None

            f_positions = {}
            f_alive_map = {}
            f_water_pcts = {}
            for f_name in fixed_names:
                alive = f_name in env.sim.drones
                f_alive_map[f_name] = alive
                if alive:
                    f_positions[f_name] = env.sim.drones[f_name].get_position()
                    f_water_pcts[f_name] = env.sim.drones[f_name].current_water / env.sim.drones[f_name].water_capacity
                else:
                    f_positions[f_name] = None
                    f_water_pcts[f_name] = 0.0

            fire_seen_vals = []
            for qi in range(N_QUADS):
                lm = last_local_maps[f"quad_{qi}"]
                if lm is not None:
                    fire_seen_vals.append(float(np.sum(lm)))
            fire_seen = max(fire_seen_vals) if fire_seen_vals else 0.0

            any_f_alive = any(f_alive_map.values())
            moisture = env.sim.environment.fire_grid.M.copy() \
                if env.sim.environment.fire_grid is not None else None

            f_valve_open = {}
            for f_name in fixed_names:
                ctrl = cmdr_ctrl.get(f_name)
                if ctrl is not None:
                    vd = getattr(ctrl, '_valve_debug', None)
                    f_valve_open[f_name] = (vd is not None and vd.get('opened', False))
                else:
                    f_valve_open[f_name] = False

            frame = _render_frame(
                step + 1, env.sim.environment.fire_grid.I.copy(), env.map_bounds,
                q_paths, q_positions, q_alive_map,
                f_paths, f_positions, f_alive_map, f_water_pcts,
                refill_zones, refill_size,
                {q: (last_local_maps[q].copy() if last_local_maps[q] is not None else None)
                 for q in last_local_maps},
                sum(total_rq.values()), total_rf, fire_seen, any_f_alive,
                terrain_collections=terrain_collections,
                moisture_map=moisture,
                f_valve_open=f_valve_open,
            )
            saved_frames[step] = frame
            # Save individual PNG
            step_label = step + 1  # human-readable
            png_path = os.path.join(output_dir, f"step_{step_label:04d}.png")
            Image.fromarray(frame).save(png_path)
            print(f"    Saved frame: {png_path}")

    env.sim.stop_simulation()

    # ── Create collage ───────────────────────────────────────
    n_frames = len(saved_frames)
    if n_frames > 0:
        sorted_steps = sorted(saved_frames.keys())
        imgs = [Image.fromarray(saved_frames[s]) for s in sorted_steps]

        w, h = imgs[0].size
        collage = Image.new('RGB', (w * n_frames, h), 'white')
        for i, img in enumerate(imgs):
            collage.paste(img, (i * w, 0))

        collage_path = os.path.join(output_dir, f"collage_seed{seed}.png")
        collage.save(collage_path)
        print(f"  Collage PNG → {collage_path}")

        collage_pdf = os.path.join(output_dir, f"collage_seed{seed}.pdf")
        collage.save(collage_pdf, "PDF", resolution=120.0)
        print(f"  Collage PDF → {collage_pdf}")

    # Summary
    end_cells = int(np.sum(env.sim.environment.fire_grid.B)) \
                if env.sim.environment.fire_grid is not None else 0
    print(f"  Seed {seed} done! End fire cells: {end_cells}")
    print(f"  Total scout reward: {sum(total_rq.values()):+.1f}")
    print(f"  Total FW reward: {total_rf:+.1f}")
    return {"seed": seed, "end_cells": end_cells, "total_rf": total_rf,
            "total_rq": sum(total_rq.values())}


def run_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {N_QUADS} scouts, {N_FIXED} FW, {N_FIRES} fires, {int(GRID_SIZE)}m map, {MAX_STEPS} steps")
    print(f"Seeds to try: {EPISODE_SEEDS}")
    print(f"Checkpoints at steps: {[s+1 for s in CHECKPOINT_STEPS]}")

    scout_actor, cmdr_actor = _load_models(device)

    # Terrain (load once)
    terrain_collections = None
    if USE_OSM:
        print("Loading OSM terrain...")
        gdfs = _load_terrain_gdfs()
        if gdfs:
            water, buildings, forests = _classify_features(gdfs)
            terrain_collections = _build_terrain_collections(water, buildings, forests)
            print(f"  Water:{len(water)}  Buildings:{len(buildings)}  Forests:{len(forests)}")

    results = []
    for seed in EPISODE_SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        output_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
        r = run_seed(seed, device, scout_actor, cmdr_actor, terrain_collections, output_dir)
        results.append(r)

    print(f"\n\n{'='*60}")
    print("  SUMMARY — pick the best seed for thesis")
    print(f"{'='*60}")
    for r in results:
        print(f"  seed={r['seed']:3d}  end_fire={r['end_cells']:4d}  "
              f"R_fw={r['total_rf']:+.1f}  R_scout={r['total_rq']:+.1f}")


if __name__ == "__main__":
    run_and_save()
