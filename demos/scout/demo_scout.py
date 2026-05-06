"""
demo_scout.py
─────────────
Multi-seed visualisation demo for N-scout MAPPO team (no commander).
Uses real OSM terrain as map background. Light theme throughout.

Layout (dynamic, scales with N scouts):
┌──────────────────────────┬──────────────────┐
│                          │ Scout 0 camera   │
│   Global map + fire      ├──────────────────┤
│   OSM terrain + trails   │ Scout 1 camera   │
│   FoV rectangles         ├──────────────────┤
│                          │ ...              │
│                          ├──────────────────┤
│                          │ Stats            │
└──────────────────────────┴──────────────────┘

Usage:
    python demo_scout.py
    python demo_scout.py --scouts 3 --seeds 100 101 102 103 104
    python demo_scout.py --scouts 2 --seed-start 200 --episodes 5 --no-gif
    python demo_scout.py --model path/scout.pt

Output per seed: scout_s{seed}_n{N}.gif + scout_s{seed}_n{N}_analysis.png
Summary:         scout_summary.csv
"""

import argparse
import torch
import numpy as np
import os, sys, glob, csv
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

# ── Light-theme colours ──────────────────────────────────────
CLR_GRASS    = '#f0eed8'
CLR_MEADOW   = '#d4e8a0'  # light yellow-green for open/meadow areas
CLR_BUILDING = '#b0b0b0'
CLR_FOREST   = '#6abf69'
CLR_WATER    = '#7ec8e3'
CLR_BG       = '#ffffff'
CLR_GRID     = '#e5e5e5'

# Up to 5 scout colours (distinct on white)
_ALL_SCOUT_COLORS  = ['#1e90ff', '#e05c00', '#1aaa1a', '#9b30d9', '#d41f1f']
_ALL_SCOUT_MARKERS = ['^',       'o',       's',        'D',       'P']

OSM_LAT   = 49.35
OSM_LON   = 16.42
OSM_CACHE = os.path.join(project_root, "data")


# =============================================================================
# OSM terrain helpers
# =============================================================================

def _load_terrain_gdfs(grid_size_m=1000.0):
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
        gdf_p = gdf_p[gdf_p.distance(center_proj) <= grid_size_m / 2.0]
        if len(gdf_p) == 0:
            continue
        gdf_p['geometry'] = gdf_p.translate(xoff=-center_proj.x, yoff=-center_proj.y)
        result[cat] = gdf_p
    return result


# Area threshold: skip water polygons larger than this (river-basin / watershed
# artifacts in OSM data that would flood the entire map with blue).
_MAX_WATER_AREA_M2 = 5_000_000  # 5 km²


def _classify_features(gdfs):
    water, buildings, forests, meadows = [], [], [], []
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
                # Skip anomalously large polygons (river-basin / watershed OSM
                # relations whose extent covers the entire visible map area)
                try:
                    area = geom.area if geom.is_valid else geom.buffer(0).area
                except Exception:
                    area = float('inf')
                if area < _MAX_WATER_AREA_M2:
                    water.append(geom)
            elif (bld is not None and bld != 'no') or lu in ('residential','commercial','industrial','retail'):
                buildings.append(geom)
            elif lu in ('forest','orchard','vineyard','wood') or nat in ('wood','scrub','heath'):
                forests.append(geom)
            elif lu in ('grass','meadow','farmland','farmyard','allotments') or nat in ('grassland','fell','heath'):
                meadows.append(geom)
    return water, buildings, forests, meadows


def _geom_patches(geom, **kw):
    # Fix invalid geometries before rendering
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except Exception:
            return []
    out = []
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out.extend(_geom_patches(p, **kw))
    elif isinstance(geom, Polygon):
        xs, ys = geom.exterior.coords.xy
        out.append(plt.Polygon(list(zip(xs, ys)), closed=True, **kw))
    return out


def _build_terrain_collections(water, buildings, forests, meadows=None):
    colls = []
    mp = [p for g in (meadows or []) for p in _geom_patches(g)]
    if mp:
        colls.append(PatchCollection(mp, facecolor=CLR_MEADOW, edgecolor='none',
                                     alpha=0.75, zorder=1))
    fp = [p for g in forests for p in _geom_patches(g)]
    if fp:
        colls.append(PatchCollection(fp, facecolor=CLR_FOREST, edgecolor='none',
                                     alpha=0.7, zorder=2))
    bp = [p for g in buildings for p in _geom_patches(g)]
    if bp:
        colls.append(PatchCollection(bp, facecolor=CLR_BUILDING, edgecolor='#999',
                                     linewidth=0.3, alpha=0.85, zorder=3))
    wp = [p for g in water for p in _geom_patches(g)]
    if wp:
        colls.append(PatchCollection(wp, facecolor=CLR_WATER, edgecolor='none',
                                     alpha=0.85, zorder=4))
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

def _render_frame(step, fire_map, b, n_quads,
                  scout_paths, scout_positions, scout_fovs, scout_alive,
                  local_maps, total_rewards, fire_sums,
                  scout_colors, scout_markers, scout_labels,
                  terrain_collections=None):

    n_rows = n_quads + 1                      # N camera rows + 1 stats row
    fig_h  = max(7.0, 2.3 * n_rows)

    fig = plt.figure(figsize=(12, fig_h), facecolor=CLR_BG)
    # Two separate GridSpecs so the right column can start lower than the map title
    gs_left  = gridspec.GridSpec(1, 1,
                                 left=0.07, right=0.62, top=0.93, bottom=0.07)
    gs_right = gridspec.GridSpec(n_rows, 1,
                                 left=0.67, right=0.97, top=0.82, bottom=0.07,
                                 hspace=0.45)

    # ── Global map ──────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs_left[0, 0])
    ax_map.set_facecolor(CLR_GRASS)
    if terrain_collections:
        _add_terrain_to_ax(ax_map, terrain_collections)

    extent = [-b, b, -b, b]
    fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
    ax_map.imshow(fire_masked, extent=extent, origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.78, zorder=4)

    for qi in range(n_quads):
        clr = scout_colors[qi]
        mk  = scout_markers[qi]
        px, py = scout_paths[qi]
        if len(px) > 1:
            ax_map.plot(px, py, color=clr, alpha=0.90, linewidth=4.0,
                        linestyle='-', zorder=5)
        if scout_alive[qi] and scout_positions[qi] is not None:
            pos = scout_positions[qi]
            fov = scout_fovs[qi]
            rect = plt.Rectangle((pos[0] - fov / 2, pos[1] - fov / 2), fov, fov,
                                  fill=False, edgecolor=clr, linewidth=1.5,
                                  alpha=0.7, zorder=7)
            ax_map.add_patch(rect)
            # White halo for visibility on any background
            ax_map.scatter(pos[0], pos[1], c=clr, s=160, marker=mk,
                           edgecolors='#111111', linewidths=0.8, zorder=9)
            ax_map.annotate(f' S{qi}', (pos[0], pos[1]),
                            textcoords='offset points', xytext=(6, 6),
                            fontsize=10, color=clr, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec=clr, alpha=0.80, linewidth=1.0),
                            zorder=10)
    ax_map.set_xlim(-b, b); ax_map.set_ylim(-b, b)
    ax_map.set_aspect('equal')
    ax_map.set_facecolor(CLR_GRASS)
    for spine in ax_map.spines.values():
        spine.set_edgecolor(CLR_GRID)
    ax_map.tick_params(labelsize=12, colors='#555555')
    ax_map.set_xlabel('X [m]', fontsize=13, color='#555555')
    ax_map.set_ylabel('Y [m]', fontsize=13, color='#555555')
    ax_map.grid(True, color=CLR_GRID, linewidth=0.4, zorder=0)

    leg_h = []
    if terrain_collections:
        leg_h += [mpatches.Patch(color=CLR_MEADOW,   label='Meadow/Grass'),
                  mpatches.Patch(color=CLR_FOREST,   label='Forest'),
                  mpatches.Patch(color=CLR_BUILDING, label='Building'),
                  mpatches.Patch(color=CLR_WATER,    label='Water')]
    for qi in range(n_quads):
        leg_h.append(plt.Line2D([0], [0], marker=scout_markers[qi], color='w',
                                markerfacecolor=scout_colors[qi],
                                markeredgecolor='#333', markersize=7,
                                label=scout_labels[qi]))
    ax_map.legend(handles=leg_h, loc='upper right', fontsize=11.5,
                  framealpha=0.92, edgecolor=CLR_GRID, facecolor=CLR_BG)

    # ── Camera panels ────────────────────────────────────────────────
    for qi in range(n_quads):
        ax_c = fig.add_subplot(gs_right[qi, 0])
        ax_c.set_facecolor(CLR_BG)
        for sp in ax_c.spines.values():
            sp.set_edgecolor(scout_colors[qi]); sp.set_linewidth(1.2)
        if local_maps[qi] is not None:
            ax_c.imshow(local_maps[qi], origin='lower', cmap='YlOrRd',
                        vmin=0, vmax=1.0)
        else:
            ax_c.text(0.5, 0.5, 'DEAD', ha='center', va='center',
                      fontsize=15, color='#999999', transform=ax_c.transAxes)
        fire_str = f"  fire={fire_sums[qi]:.2f}" if fire_sums[qi] > 0 else ""
        ax_c.set_title(f"{scout_labels[qi]}  (32\u00d732){fire_str}",
                       fontsize=13.5, fontweight='bold', color='#222222', pad=3)
        ax_c.set_xticks([]); ax_c.set_yticks([])

    # ── Stats panel ──────────────────────────────────────────────────
    ax_st = fig.add_subplot(gs_right[n_quads, 0])
    ax_st.set_facecolor('none')
    for sp in ax_st.spines.values():
        sp.set_visible(False)
    ax_st.set_xticks([]); ax_st.set_yticks([])

    # Global fire stats from fire_map
    total_fire_cells = int(np.sum(fire_map > 0.01))
    total_fire_area  = total_fire_cells * 25  # cell_size=5m → 25 m²/cell

    lines = []
    for qi in range(n_quads):
        alive_str = "alive" if scout_alive[qi] else "DEAD"
        pos = scout_positions[qi]
        if scout_alive[qi] and pos is not None:
            alt_str = f"{pos[2]:.0f} m"
            xy_str  = f"({pos[0]:+.0f}, {pos[1]:+.0f})"
        else:
            alt_str = "—"
            xy_str  = "—"

        # Distance travelled
        px, py = scout_paths[qi]
        if len(px) > 1:
            diffs = np.hypot(np.diff(px), np.diff(py))
            dist_m = float(np.sum(diffs))
            dist_str = f"{dist_m/1000:.2f} km" if dist_m >= 1000 else f"{dist_m:.0f} m"
        else:
            dist_str = "0 m"

        fire_str = f"{fire_sums[qi]:.1f}" if fire_sums[qi] > 0 else "0"

        sep = "─" * 16
        lines += [
            f"S{qi}  [{alive_str}]",
            f"  reward   {total_rewards[qi]:+.1f}",
            f"  alt      {alt_str}",
            f"  pos      {xy_str}",
            f"  fire↓    {fire_str}",
            f"  dist     {dist_str}",
            sep,
        ]

    lines += [
        f"step     {step}",
        f"fire↑    {total_fire_cells} cells",
        f"         ({total_fire_area/1e4:.1f} ha)",
    ]
    ax_st.text(0.06, 0.98, "\n".join(lines), color='#222222', fontsize=11,
               va='top', transform=ax_st.transAxes, fontfamily='monospace')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=CLR_BG)
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


# =============================================================================
# Analysis plot
# =============================================================================

def _save_analysis(hist, n_quads, scout_colors, scout_labels, out_path, seed, ep):
    fig, axes = plt.subplots(3, n_quads, figsize=(5 * n_quads, 9), squeeze=False)
    fig.patch.set_facecolor(CLR_BG)
    fig.suptitle(f"Scout Demo — seed={seed} ep={ep}  ({n_quads} scouts)",
                 fontsize=13, fontweight='bold', color='#222222')

    for qi in range(n_quads):
        clr = scout_colors[qi]
        axes[0, qi].plot(np.cumsum(hist[qi]["reward"]), color=clr, linewidth=1.5)
        axes[0, qi].set_title(f"{scout_labels[qi]} — Cumulative R",
                              fontweight='bold', color=clr)
        axes[0, qi].set_facecolor(CLR_BG)
        axes[0, qi].grid(color=CLR_GRID, linewidth=0.5)

        axes[1, qi].plot(hist[qi]["alt"], color=clr, linewidth=1.0)
        axes[1, qi].set_title(f"{scout_labels[qi]} — Altitude [m]",
                              fontweight='bold', color=clr)
        axes[1, qi].set_facecolor(CLR_BG)
        axes[1, qi].grid(color=CLR_GRID, linewidth=0.5)

        axes[2, qi].fill_between(range(len(hist[qi]["fire"])), hist[qi]["fire"],
                                  color='#e07020', alpha=0.55)
        axes[2, qi].set_title(f"{scout_labels[qi]} — Fire under camera",
                              fontweight='bold', color=clr)
        axes[2, qi].set_xlabel("Step", color='#555555')
        axes[2, qi].set_facecolor(CLR_BG)
        axes[2, qi].grid(color=CLR_GRID, linewidth=0.5)

    for ax in axes.flat:
        ax.tick_params(colors='#555555')
        for sp in ax.spines.values():
            sp.set_edgecolor(CLR_GRID)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor=CLR_BG, edgecolor='none')
    plt.close(fig)
    print(f"  Analysis → {out_path}")


# =============================================================================
# Episode runner
# =============================================================================

def run_episode(env, scout_actor, seed, ep_num, n_quads, device,
                scout_colors, scout_markers, scout_labels,
                hidden_dim=128, max_steps=1000,
                gif_every=2, gif_fps=15,
                terrain_collections=None, save_gif=True, gif_path=None,
                analysis_path=None, n_fires=3, fixed_fire=False):
    """Run one episode. Returns metrics dict."""
    import random as _rnd
    import pybullet as _p

    quad_agents = [f"quad_{i}" for i in range(n_quads)]

    obs, _ = env.reset(seed=seed, epizode_number=ep_num)

    # ── Place fires ─────────────────────────────────────────────────
    _rnd.seed(seed * 7 + 13)           # reproducible per seed
    if fixed_fire:
        # 1 fire at map center, fixed intensity
        fire_positions = [(0.0, 0.0)]
        env.sim.environment.start_fire_at_position([0.0, 0.0], intensity=0.80, radius_m=8.0)
    else:
        b = env.map_bounds * 0.65
        fire_positions = []
        attempts = 0
        while len(fire_positions) < n_fires and attempts < 200:
            attempts += 1
            fx = _rnd.uniform(-b, b)
            fy = _rnd.uniform(-b, b)
            if any(np.hypot(fx - px, fy - py) < 150.0 for px, py in fire_positions):
                continue
            fire_positions.append((fx, fy))
        for fx, fy in fire_positions:
            env.sim.environment.start_fire_at_position(
                [fx, fy], intensity=_rnd.uniform(0.6, 0.9), radius_m=_rnd.uniform(6.0, 12.0))

    for _ in range(15):
        env.sim.environment.update_fire_simulation(real_dt=0.1)

    # ── Respawn scouts ─────────────────────────────────────────────
    if fixed_fire:
        # Each scout placed 100–150 m from center fire at evenly-spaced angles
        base_angle = _rnd.uniform(0, 2 * np.pi)
        for qi, agent in enumerate(quad_agents):
            if agent not in env.sim.drones:
                continue
            angle = base_angle + qi * (2 * np.pi / n_quads)
            dist  = _rnd.uniform(200.0, 250.0)
            sx = dist * np.cos(angle)
            sy = dist * np.sin(angle)
            drone = env.sim.drones[agent]
            orn   = drone.get_orientation_quaternion()
            _p.resetBasePositionAndOrientation(drone.drone_id, [sx, sy, 75.0], orn)
            _p.resetBaseVelocity(drone.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    else:
        for qi, agent in enumerate(quad_agents):
            if agent not in env.sim.drones:
                continue
            target_fx, target_fy = fire_positions[qi % len(fire_positions)]
            for _try in range(60):
                angle = _rnd.uniform(0, 2 * np.pi)
                dist  = _rnd.uniform(180.0, 280.0)
                sx = target_fx + dist * np.cos(angle)
                sy = target_fy + dist * np.sin(angle)
                if abs(sx) > env.map_bounds * 0.85 or abs(sy) > env.map_bounds * 0.85:
                    continue
                ok = True
                for other in quad_agents[:qi]:
                    if other not in env.sim.drones:
                        continue
                    op = env.sim.drones[other].get_position()
                    if np.hypot(sx - op[0], sy - op[1]) < 60.0:
                        ok = False; break
                if ok:
                    break
            drone = env.sim.drones[agent]
            orn   = drone.get_orientation_quaternion()
            _p.resetBasePositionAndOrientation(drone.drone_id, [sx, sy, _rnd.uniform(65.0, 90.0)], orn)
            _p.resetBaseVelocity(drone.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    obs = {ag: env._get_obs(ag) for ag in env.agents}

    hidden_states = {ag: torch.zeros(1, 1, hidden_dim).to(device) for ag in quad_agents}
    scout_paths   = {qi: ([], []) for qi in range(n_quads)}
    local_maps    = [None] * n_quads
    total_rewards = [0.0] * n_quads
    fire_sums     = [0.0] * n_quads
    hist          = {qi: {"reward": [], "alt": [], "fire": []} for qi in range(n_quads)}
    frames        = []
    fire_cells_peak = 0

    for step in tqdm.tqdm(range(max_steps), desc=f"seed={seed}", leave=False):
        if not env.agents:
            break

        if env.sim.environment.fire_grid is not None:
            fire_cells_peak = max(fire_cells_peak,
                                  int(np.sum(env.sim.environment.fire_grid.B)))

        actions = {}
        for qi, ag in enumerate(quad_agents):
            if ag not in env.agents or ag not in env.sim.drones:
                continue
            l_map = torch.FloatTensor(obs[ag]["local_map"]).to(device).unsqueeze(0)
            s_st  = torch.FloatTensor(obs[ag]["self_state"]).to(device).unsqueeze(0)
            n_st  = torch.FloatTensor(obs[ag]["neighbor_states"]).to(device).unsqueeze(0)
            n_m   = torch.BoolTensor(obs[ag]["neighbor_mask"]).to(device).unsqueeze(0)
            with torch.no_grad():
                dist, _msg, h_out = scout_actor(l_map, s_st, n_st, n_m, hidden_states[ag])
            hidden_states[ag] = h_out
            actions[ag] = dist.mean.squeeze(0).cpu().numpy()
            local_maps[qi] = obs[ag]["local_map"][0]

        if not actions:
            break

        obs, rewards, _, _, _ = env.step(actions)

        scout_positions = [None] * n_quads
        scout_fovs      = [40.0] * n_quads
        scout_alive     = [False] * n_quads

        for qi, ag in enumerate(quad_agents):
            r = rewards.get(ag, 0.0)
            total_rewards[qi] += r
            hist[qi]["reward"].append(r)
            if ag in env.sim.drones:
                scout_alive[qi] = True
                pos = env.sim.drones[ag].get_position()
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

        if save_gif and step % gif_every == 0:
            frame = _render_frame(
                step, env.sim.environment.fire_grid.I.copy(), env.map_bounds, n_quads,
                {qi: (list(scout_paths[qi][0]), list(scout_paths[qi][1]))
                 for qi in range(n_quads)},
                scout_positions, scout_fovs, scout_alive,
                [m.copy() if m is not None else None for m in local_maps],
                list(total_rewards), list(fire_sums),
                scout_colors, scout_markers, scout_labels,
                terrain_collections=terrain_collections,
            )
            frames.append(frame)

    if save_gif and frames:
        out = gif_path or os.path.join(project_root, f"scout_s{seed}_n{n_quads}.gif")
        imageio.mimsave(out, frames, fps=gif_fps, loop=0)
        print(f"  GIF → {out}")        # Auto-extract 3 presentation frames
        # if fixed_fire:
        n_f = len(frames)
        for lbl, pct in [('1_approach', 0.05),
                            ('2_arrival',  0.25),
                            ('3_hover',    0.8)]:
            idx  = min(int(pct * n_f), n_f - 1)
            base = out.replace('.gif', f'_{lbl}.pdf')
            Image.fromarray(frames[idx]).save(base)
            print(f"  Frame → {base}")
    if analysis_path:
        _save_analysis(hist, n_quads, scout_colors, scout_labels,
                       analysis_path, seed, ep_num)

    scouts_survived = sum(1 for ag in quad_agents if ag in env.sim.drones)
    total_r = sum(total_rewards)

    # Scout-relevant metrics (scouts cannot suppress fire — no water)
    end_cells = (int(np.sum(env.sim.environment.fire_grid.B))
                 if env.sim.environment.fire_grid is not None else 0)
    # fire_discovered: did any scout ever see fire in their camera?
    fire_discovered = any(
        max(hist[qi]["fire"]) > 0.1 for qi in range(n_quads))
    # avg_fire_seen: mean fire signal across all scouts and all steps
    all_fire_vals = [v for qi in range(n_quads) for v in hist[qi]["fire"] if v > 0]
    avg_fire_seen = float(np.mean(all_fire_vals)) if all_fire_vals else 0.0
    # coverage_pct: fraction of steps where at least one scout saw fire
    steps_with_fire = sum(
        1 for s in range(len(hist[0]["fire"]))
        if any(hist[qi]["fire"][s] > 0.1 for qi in range(n_quads) if s < len(hist[qi]["fire"])))
    total_steps = len(hist[0]["fire"]) or 1
    coverage_pct = round(steps_with_fire / total_steps * 100.0, 1)

    return {
        "seed":             seed,
        "n_quads":          n_quads,
        "peak_cells":       fire_cells_peak,
        "end_cells":        end_cells,
        "fire_discovered":  fire_discovered,
        "coverage_pct":     coverage_pct,
        "avg_fire_seen":    round(avg_fire_seen, 3),
        "scouts_survived":  scouts_survived,
        "total_r":          round(total_r, 2),
        "avg_r_per_scout":  round(total_r / n_quads, 2),
    }


# =============================================================================
# Main
# =============================================================================

def _default_model():
    candidates = sorted(glob.glob(
        os.path.join(project_root, "saved_models", "multi", "scout_b*.pt")))
    if candidates:
        return candidates[-1]
    candidates = sorted(glob.glob(
        os.path.join(project_root, "saved_models", "**", "scout*.pt"), recursive=True))
    return candidates[-1] if candidates else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scout MAPPO demo (multi-seed)")
    parser.add_argument("--model",       type=str,   default=None)
    parser.add_argument("--scouts",      type=int,   default=2, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds",       type=int,   nargs="+", default=None)
    parser.add_argument("--seed-start",  type=int,   default=100)
    parser.add_argument("--episodes",    type=int,   default=5)
    parser.add_argument("--max-steps",   type=int,   default=1000)
    parser.add_argument("--gif-every",   type=int,   default=2)
    parser.add_argument("--gif-fps",     type=int,   default=15)
    parser.add_argument("--no-gif",      action="store_true")
    parser.add_argument("--grid-size",   type=float, default=None,
                        help="Single map size in metres (overrides --grid-sizes)")
    parser.add_argument("--grid-sizes",  type=float, nargs="+", default=[1000.0],
                        help="One or more map sizes to run in sequence (default: 1000)")
    parser.add_argument("--no-osm",      action="store_true")
    parser.add_argument("--fires",       type=int,   default=3,
                        help="Number of fire sources per episode (default: 3)")
    parser.add_argument("--fixed-fire",  action="store_true",
                        help="1 fire at center, scouts 100-150 m away; auto-saves 3 PNG frames")
    args = parser.parse_args()

    # --grid-size overrides --grid-sizes for backwards compatibility
    grid_sizes = [args.grid_size] if args.grid_size is not None else args.grid_sizes

    seeds   = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.episodes))
    n_quads = args.scouts

    scout_colors  = _ALL_SCOUT_COLORS[:n_quads]
    scout_markers = _ALL_SCOUT_MARKERS[:n_quads]
    scout_labels  = [f"Scout {i}" for i in range(n_quads)]

    model_path = args.model or _default_model()
    if model_path is None:
        raise FileNotFoundError("No scout checkpoint found. Use --model to specify.")
    print(f"Model:  {model_path}")
    print(f"Scouts: {n_quads}  |  Fires: {args.fires}  |  Seeds: {seeds}")
    print(f"Map sizes: {grid_sizes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load actor once — shared across all map sizes (obs shape is independent of map size)
    _tmp_env  = DroneFireEnv(num_quads=n_quads, num_fixed=0, grid_size_m=1000.0, max_steps=10)
    obs_space = _tmp_env.observation_space("quad_0")
    # sim is only created after reset() — skip stop_simulation() here

    scout_actor = ScoutActor(self_state_dim=obs_space["self_state"].shape[0],
                             msg_dim=5, hidden_dim=128).to(device)
    scout_actor.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    scout_actor.eval()
    print(f"Loaded: {model_path}")

    all_results = []

    for grid_size in grid_sizes:
        size_tag = f"{int(grid_size)}m"
        print(f"\n{'='*60}")
        print(f"  Map size: {grid_size} × {grid_size} m  ({size_tag})")
        print(f"{'='*60}")

        env = DroneFireEnv(num_quads=n_quads, num_fixed=0, grid_size_m=grid_size,
                           max_steps=args.max_steps, use_osm=not args.no_osm,
                           osm_lat=OSM_LAT, osm_lon=OSM_LON, osm_cache_dir=OSM_CACHE)

        terrain_collections = None
        if not args.no_osm:
            print("Loading OSM terrain...")
            gdfs = _load_terrain_gdfs(grid_size)
            if gdfs:
                water, buildings, forests, meadows = _classify_features(gdfs)
                terrain_collections = _build_terrain_collections(water, buildings, forests, meadows)
                print(f"  Water:{len(water)}  Buildings:{len(buildings)}  Forests:{len(forests)}  Meadows:{len(meadows)}")

        for ep_idx, seed in enumerate(seeds):
            print(f"\n[{ep_idx+1}/{len(seeds)}] seed={seed}  map={size_tag}")
            gif_path      = os.path.join(project_root, f"scout_s{seed}_n{n_quads}_{size_tag}.gif")
            analysis_path = os.path.join(project_root, f"scout_s{seed}_n{n_quads}_{size_tag}_analysis.png")
            result = run_episode(
                env, scout_actor, seed, ep_idx, n_quads, device,
                scout_colors, scout_markers, scout_labels,
                hidden_dim=128, max_steps=args.max_steps,
                gif_every=args.gif_every, gif_fps=args.gif_fps,
                terrain_collections=terrain_collections,
                save_gif=not args.no_gif, gif_path=gif_path,
                analysis_path=analysis_path,
                n_fires=args.fires,
                fixed_fire=args.fixed_fire,
            )
            result["grid_size"] = grid_size
            all_results.append(result)
            disc_str = "YES" if result['fire_discovered'] else "no "
            print(f"  fire_disc={disc_str}  coverage={result['coverage_pct']:.1f}%  "
                  f"avg_seen={result['avg_fire_seen']:.3f}  "
                  f"survived={result['scouts_survived']}/{n_quads}  "
                  f"R={result['total_r']:+.1f}")

        env.sim.stop_simulation()

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'─'*82}")
    print(f"{'Seed':>6}  {'Map':>6}  {'N':>2}  {'Disc':>4}  {'Cover%':>7}  {'AvgSeen':>8}  "
          f"{'Surv':>5}  {'R/scout':>8}")
    print(f"{'─'*82}")
    for r in all_results:
        disc_str = "YES" if r['fire_discovered'] else "no "
        print(f"{r['seed']:>6}  {int(r['grid_size']):>5}m  {r['n_quads']:>2}  {disc_str:>4}  "
              f"{r['coverage_pct']:>7.1f}  {r['avg_fire_seen']:>8.3f}  "
              f"{r['scouts_survived']:>2}/{r['n_quads']:<2}  "
              f"{r['avg_r_per_scout']:>+8.1f}")
    print(f"{'─'*82}")

    # Per-map-size averages
    for gs in grid_sizes:
        sub = [r for r in all_results if r['grid_size'] == gs]
        avg_cov  = np.mean([r['coverage_pct'] for r in sub])
        avg_seen = np.mean([r['avg_fire_seen'] for r in sub])
        avg_r    = np.mean([r['avg_r_per_scout'] for r in sub])
        n_disc   = sum(1 for r in sub if r['fire_discovered'])
        print(f"  {int(gs):>5}m AVG  disc={n_disc}/{len(sub)}  cover={avg_cov:.1f}%  "
              f"seen={avg_seen:.3f}  R/scout={avg_r:+.1f}")

    csv_path = os.path.join(project_root, "scout_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)
    print(f"\nCSV → {csv_path}")

