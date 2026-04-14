#!/usr/bin/env python3
"""
Generate paper figure: OSM satellite image vs. internal fuel representation.

Creates two side-by-side images:
  (a) Real aerial/satellite tile (Esri World Imagery via contextily)
  (b) Internal fuel grid after OSM → rasterization pipeline

The script downloads OSM vector features directly by coordinates (no geocoding),
projects them to UTM, centres at (0,0), and rasterises using the same
Painter's Algorithm as the simulation (Environment.rasterize_terrain_layers).

Usage:
    python drawings/poster/generate_osm_vs_fuel.py
    python drawings/poster/generate_osm_vs_fuel.py --lat 49.35 --lon 16.42 --size 80
    python drawings/poster/generate_osm_vs_fuel.py --location "Tišnov, Czech Republic" --size 80
"""

import sys, os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
import osmnx as ox
import contextily as ctx
from shapely.geometry import box, Point
from matplotlib.path import Path as MplPath

# ── Configuration ───────────────────────────────────────────────────────────
DEFAULT_LAT = 49.35        # Tišnov area – mixed forest/buildings/water
DEFAULT_LON = 16.42
DEFAULT_SIZE_M = 80        # 80 × 80 metres
CELL_SIZE_M = 1.0          # 1 m resolution for sharp poster image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

# Satellite tile providers – tried in order; first that returns real
# imagery wins.  Esri World Imagery has gaps in Czech Republic at high zoom.
TILE_PROVIDERS = [
    ("Esri.WorldImagery",  ctx.providers.Esri.WorldImagery),
    ("OpenStreetMap",      ctx.providers.OpenStreetMap.Mapnik),
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _utm_crs_for(lat, lon):
    """Return EPSG code string for the UTM zone covering (lat, lon)."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}" if lat >= 0 else f"EPSG:327{zone:02d}"


def geocode_location(location_str: str):
    """Geocode a location string to (lat, lon) via osmnx."""
    point = ox.geocode(location_str)
    return point[0], point[1]


# ── Step 1: Satellite tile ──────────────────────────────────────────────────

def _tile_is_blank(img: np.ndarray, threshold: float = 0.92) -> bool:
    """Heuristic: if > *threshold* of pixels are the same colour the tile is
    a placeholder / 'not available' stub."""
    if img.ndim == 3 and img.shape[2] == 4:      # RGBA → RGB
        img = img[:, :, :3]
    flat = img.reshape(-1, img.shape[-1]) if img.ndim == 3 else img.ravel()
    # most-common colour
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    return counts.max() / counts.sum() > threshold


def _crop_tile_to_bounds(img, tile_ext, req_bounds):
    """
    Crop a tile image so it covers exactly *req_bounds*.

    tile_ext : (left, right, bottom, top) – extent returned by contextily
    req_bounds : (minx, miny, maxx, maxy) – the bounds we actually asked for
    Returns  : cropped image (numpy array)
    """
    t_left, t_right, t_bottom, t_top = tile_ext
    r_minx, r_miny, r_maxx, r_maxy = req_bounds

    h, w = img.shape[:2]

    # Pixel per metre in each axis
    ppx = w / (t_right - t_left)
    ppy = h / (t_top - t_bottom)

    # Pixel coordinates of the requested sub-rectangle
    # (origin of image is top-left, y increases downward)
    col0 = int(round((r_minx - t_left) * ppx))
    col1 = int(round((r_maxx - t_left) * ppx))
    row0 = int(round((t_top - r_maxy) * ppy))   # top edge
    row1 = int(round((t_top - r_miny) * ppy))   # bottom edge

    col0 = max(0, min(col0, w))
    col1 = max(0, min(col1, w))
    row0 = max(0, min(row0, h))
    row1 = max(0, min(row1, h))

    return img[row0:row1, col0:col1]


def download_satellite_tile(lat: float, lon: float, size_m: float):
    """
    Download a map/satellite tile covering *size_m × size_m* metres centred on
    (lat, lon).  Tries each provider in TILE_PROVIDERS until one returns a
    non-blank image.

    Returns
    -------
    img : ndarray  – cropped to the requested area
    req_bounds : tuple  – (minx, miny, maxx, maxy) in EPSG:3857
    """
    half = size_m / 2
    utm_crs = _utm_crs_for(lat, lon)

    center_utm = (gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                  .to_crs(utm_crs))
    cx, cy = center_utm.iloc[0].x, center_utm.iloc[0].y

    # Bounding box in Web Mercator (needed by contextily)
    bbox_utm = gpd.GeoSeries(
        [box(cx - half, cy - half, cx + half, cy + half)], crs=utm_crs
    )
    bbox_wm = bbox_utm.to_crs(epsg=3857)
    minx, miny, maxx, maxy = bbox_wm.total_bounds
    req_bounds = (minx, miny, maxx, maxy)

    for prov_name, prov in TILE_PROVIDERS:
        try:
            img, ext = ctx.bounds2img(
                minx, miny, maxx, maxy,
                source=prov, zoom=18, ll=False,
            )
            if _tile_is_blank(img):
                print(f"   ⚠ {prov_name}: blank/placeholder tile, trying next…")
                continue
            print(f"   ✔ Using provider: {prov_name}")
            img = _crop_tile_to_bounds(img, ext, req_bounds)
            return img, req_bounds
        except Exception as e:
            print(f"   ⚠ {prov_name} failed: {e}")

    # Last resort
    print("   ⚠ All providers returned blank; using last result anyway.")
    img, ext = ctx.bounds2img(
        minx, miny, maxx, maxy,
        source=TILE_PROVIDERS[0][1], zoom=17, ll=False,
    )
    img = _crop_tile_to_bounds(img, ext, req_bounds)
    return img, req_bounds


# ── Step 2: Fuel grid (standalone, no PyBullet) ────────────────────────────

OSM_TAGS = {
    "landuse": ["residential", "commercial", "industrial",
                "forest", "grass", "meadow", "reservoir"],
    "natural": ["wood", "water", "wetland", "scrub"],
    "waterway": ["river", "stream", "canal", "drain"],
    "building": True,
    "highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service", "pedestrian",
                "motorway_link", "trunk_link", "primary_link",
                "secondary_link", "tertiary_link"],
}

# Fuel values matching Environment.rasterize_terrain_layers
FUEL_WATER   = 0.0
FUEL_ROAD    = 0.0   # asphalt / concrete – non-flammable
FUEL_GRASS   = 0.3
FUEL_FOREST  = 0.8
FUEL_BUILDING = 0.9

# Approximate road half-widths in metres (used when buffering LineStrings)
ROAD_WIDTHS = {
    "motorway": 7, "trunk": 6, "primary": 5, "secondary": 4,
    "tertiary": 3.5, "unclassified": 3, "residential": 3,
    "service": 2.5, "pedestrian": 2,
    "motorway_link": 4, "trunk_link": 4,
    "primary_link": 4, "secondary_link": 3.5, "tertiary_link": 3,
}
DEFAULT_ROAD_WIDTH = 3  # metres half-width fallback


def _rasterise_polygons(fuel, origin_x, origin_y, cell_size, H, W,
                        gdf, fuel_val):
    """Burn *gdf* polygons into the fuel grid (same logic as env_core)."""
    if gdf is None or gdf.empty:
        return
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = (list(geom.geoms) if geom.geom_type == "MultiPolygon"
                 else [geom] if geom.geom_type == "Polygon" else [])
        for poly in polys:
            mnx, mny, mxx, mxy = poly.bounds
            j0 = max(0, int((mnx - origin_x) / cell_size))
            i0 = max(0, int((mny - origin_y) / cell_size))
            j1 = min(W, int(np.ceil((mxx - origin_x) / cell_size)))
            i1 = min(H, int(np.ceil((mxy - origin_y) / cell_size)))
            if j0 >= j1 or i0 >= i1:
                continue
            xs = origin_x + (np.arange(j0, j1) + 0.5) * cell_size
            ys = origin_y + (np.arange(i0, i1) + 0.5) * cell_size
            xv, yv = np.meshgrid(xs, ys)
            pts = np.column_stack((xv.ravel(), yv.ravel()))
            mask = MplPath(list(poly.exterior.coords)).contains_points(pts)
            fuel[i0:i1, j0:j1][mask.reshape(i1 - i0, j1 - j0)] = fuel_val


def _buffer_roads(gdf_roads):
    """
    Buffer road LineStrings / MultiLineStrings into Polygons using
    per-class road widths.  Returns a GeoDataFrame of polygons.
    """
    polys = []
    for _, row in gdf_roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        hw = row.get("highway", None)
        half_w = ROAD_WIDTHS.get(hw, DEFAULT_ROAD_WIDTH)
        buffered = geom.buffer(half_w, cap_style=2)  # flat cap
        if not buffered.is_empty:
            polys.append(buffered)
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_roads.crs)
    return gpd.GeoDataFrame(geometry=polys, crs=gdf_roads.crs)


def build_fuel_grid(lat: float, lon: float, size_m: float, cell_size: float):
    """
    Download OSM features around *(lat, lon)*, project to UTM, centre at
    (0, 0), and rasterise into a fuel grid using the Painter's Algorithm
    (grass → forest → buildings → roads → water).
    """
    utm_crs = _utm_crs_for(lat, lon)

    # Project centre to UTM
    center_utm = (gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                  .to_crs(utm_crs))
    cx, cy = center_utm.iloc[0].x, center_utm.iloc[0].y

    # Download from OSM (use a generous radius for border features)
    download_dist = max(size_m * 1.5, 200)
    print(f"   Downloading OSM features (dist={download_dist:.0f} m)...")
    gdf = ox.features_from_point((lat, lon), tags=OSM_TAGS, dist=download_dist)
    print(f"   Downloaded {len(gdf)} features.")

    # Project and centre
    gdf_proj = gdf.to_crs(utm_crs)
    gdf_proj["geometry"] = gdf_proj.geometry.translate(xoff=-cx, yoff=-cy)

    # Ensure needed columns exist
    for col in ("natural", "waterway", "landuse", "building", "highway"):
        if col not in gdf_proj.columns:
            gdf_proj[col] = None

    # Filter categories (same masks as map_importer._process_osm_features)
    mask_water = (
        gdf_proj["natural"].isin(["water", "wetland"])
        | gdf_proj["waterway"].isin(["river", "canal", "dock", "riverbank"])
        | gdf_proj["landuse"].isin(["reservoir", "basin"])
    )
    mask_building = (
        (gdf_proj["building"].notna() & (gdf_proj["building"] != "no"))
        | gdf_proj["landuse"].isin(["residential", "commercial", "industrial", "retail"])
    )
    mask_forest = (
        gdf_proj["landuse"].isin(["forest", "orchard", "vineyard", "wood"])
        | gdf_proj["natural"].isin(["wood", "scrub", "heath"])
    )
    mask_road = gdf_proj["highway"].notna() & (gdf_proj["highway"] != "no")

    gdf_water = gdf_proj[mask_water]
    gdf_building = gdf_proj[mask_building]
    gdf_forest = gdf_proj[mask_forest]
    gdf_road = gdf_proj[mask_road]

    # Buffer road lines → polygons
    gdf_road_poly = _buffer_roads(gdf_road)

    print(f"   Water: {len(gdf_water)}, Buildings: {len(gdf_building)}, "
          f"Forest: {len(gdf_forest)}, Roads: {len(gdf_road)}")

    # Grid dimensions
    half = size_m / 2
    H = int(np.ceil(size_m / cell_size))
    W = H
    origin_x, origin_y = -half, -half

    # 1. Base layer: grass
    fuel = np.full((H, W), FUEL_GRASS, dtype=float)

    # 2. Painter's order: forest → buildings → roads → water
    _rasterise_polygons(fuel, origin_x, origin_y, cell_size, H, W,
                        gdf_forest, FUEL_FOREST)
    _rasterise_polygons(fuel, origin_x, origin_y, cell_size, H, W,
                        gdf_building, FUEL_BUILDING)
    _rasterise_polygons(fuel, origin_x, origin_y, cell_size, H, W,
                        gdf_road_poly, FUEL_ROAD)
    _rasterise_polygons(fuel, origin_x, origin_y, cell_size, H, W,
                        gdf_water, FUEL_WATER)

    extent = (origin_x, origin_x + W * cell_size,
              origin_y, origin_y + H * cell_size)
    return fuel, extent


# ── Step 3: Render ──────────────────────────────────────────────────────────

def render_figure(sat_img, sat_ext, fuel, fuel_ext, output_path, size_m):
    """Create a publication-quality two-panel figure."""
    try:
        plt.style.use(["science", "ieee"])
    except Exception:
        plt.rcParams.update({
            "font.family": "serif", "font.size": 10,
            "axes.labelsize": 11, "axes.titlesize": 12,
        })

    half = size_m / 2
    local_ext = [-half, half, -half, half]   # same for both panels

    fig, (ax_sat, ax_fuel) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    # ── (a) Satellite image ────────────────────────────────────────────
    # Display with local-metre extent so it visually matches the fuel grid.
    # The image was already cropped to exactly the requested geographic area.
    ax_sat.imshow(sat_img, extent=local_ext, origin="upper", aspect="equal")
    ax_sat.set_title("(a) Aerial imagery (Esri)")
    ax_sat.set_xlabel("X [m]")
    ax_sat.set_ylabel("Y [m]")

    # ── (b) Fuel grid ──────────────────────────────────────────────────
    # Fuel = 0.0 covers both water AND roads.  We use a 5-colour map that
    # splits the 0.0 bin visually via a tiny epsilon so both share the same
    # non-flammable value but get distinct colours in the map.
    # However, since water and roads both have fuel=0.0 and we can't
    # distinguish them from the grid alone, we use a single "Non-flammable"
    # colour (dark grey / asphalt) for the zero bin.
    cmap_fuel = mcolors.ListedColormap([
        "#3A3A3A",   # non-flammable: roads + water  (0.0)
        "#C8D96F",   # grass                         (0.3)
        "#4A7C59",   # forest                        (0.8)
        "#8C8C8C",   # buildings                     (0.9)
    ])
    bounds = [0.0, 0.05, 0.5, 0.85, 1.0]
    norm_fuel = mcolors.BoundaryNorm(bounds, cmap_fuel.N)

    ax_fuel.imshow(fuel, extent=local_ext,
                   origin="lower", cmap=cmap_fuel, norm=norm_fuel,
                   interpolation="nearest", aspect="equal")
    ax_fuel.set_title(f"(b) Internal fuel map ({int(size_m)}$\\times${int(size_m)} m)")
    ax_fuel.set_xlabel("X [m]")
    ax_fuel.set_ylabel("Y [m]")

    legend_patches = [
        mpatches.Patch(color="#3A3A3A", label="Road / Water (0.0)"),
        mpatches.Patch(color="#C8D96F", label="Grass (0.3)"),
        mpatches.Patch(color="#4A7C59", label="Forest (0.8)"),
        mpatches.Patch(color="#8C8C8C", label="Buildings (0.9)"),
    ]
    ax_fuel.legend(handles=legend_patches, loc="lower right",
                   fontsize=6, framealpha=0.85)

    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="png")
    print(f"✅ Saved: {output_path}")

    # PDF – embed fonts as TrueType (Type 42) for correct rendering
    base, _ = os.path.splitext(output_path)
    pdf_path = base + ".pdf"
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight",
                    format="pdf", backend="pdf")
    print(f"✅ Saved: {pdf_path}")

    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate OSM satellite vs. fuel-map figure for paper."
    )
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--location", type=str, default=None,
                        help="Location string to geocode (alternative to --lat/--lon)")
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE_M,
                        help="Side length in metres (default: 80)")
    parser.add_argument("--cell", type=float, default=CELL_SIZE_M,
                        help="Fuel grid cell size in metres (default: 1)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: drawings/poster/osm_vs_fuel.png)")
    args = parser.parse_args()

    # Resolve coordinates
    if args.location:
        print(f"📍 Geocoding '{args.location}'...")
        lat, lon = geocode_location(args.location)
        print(f"   → ({lat:.5f}, {lon:.5f})")
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        lat, lon = DEFAULT_LAT, DEFAULT_LON

    size_m = args.size
    cell_size = args.cell
    output_path = args.output or os.path.join(OUTPUT_DIR, "osm_vs_fuel.png")

    print(f"🗺️  Centre: ({lat:.5f}, {lon:.5f})")
    print(f"📐 Area: {size_m}×{size_m} m, cell size: {cell_size} m")

    # Step 1 – Satellite tile
    print("\n── Step 1: Downloading satellite tile ──")
    sat_img, sat_ext = download_satellite_tile(lat, lon, size_m)
    print(f"   Tile shape: {sat_img.shape}")

    # Step 2 – Fuel grid (standalone, no PyBullet needed)
    print("\n── Step 2: Building internal fuel grid ──")
    fuel, fuel_ext = build_fuel_grid(lat, lon, size_m, cell_size)
    print(f"   Fuel grid shape: {fuel.shape}")

    # Brief stats
    bins = [0.0, 0.05, 0.5, 0.85, 1.01]
    labels = ["Water", "Grass", "Forest", "Buildings"]
    counts = np.histogram(fuel, bins=bins)[0]
    total = fuel.size
    for lbl, cnt in zip(labels, counts):
        print(f"   {lbl:>10s}: {cnt:>6d} cells ({100 * cnt / total:.1f}%)")

    # Step 3 – Render figure
    print("\n── Step 3: Rendering figure ──")
    render_figure(sat_img, sat_ext, fuel, fuel_ext, output_path, size_m)
    print("\nDone.")


if __name__ == "__main__":
    main()
