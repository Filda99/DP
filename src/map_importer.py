"""
Map Importer from OpenStreetMap (OSM)

This module uses osmnx to download real-world map data and populates 
the simulation Environment.

UPDATED LOGIC:
1. Filters data into categories (Water, Buildings, Forest).
2. Sends raw polygons to Environment for precise rasterization (Painter's Algorithm).
3. Creates simplified visual proxies for PyBullet 3D view.
"""

import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPolygon

# Try to import the Environment class
try:
    from .environment import Environment
except ImportError:
    from src.environment import Environment

def _approximate_polygon_with_circles(geom, min_radius=5.0, max_circles=10):
    """
    Helper for VISUALIZATION ONLY. Approximates irregular polygons with circles
    so they can be drawn in PyBullet. Does not affect fire physics.
    """
    if isinstance(geom, MultiPolygon):
        circles = []
        for poly in geom.geoms:
            circles.extend(_approximate_polygon_with_circles(poly, min_radius, max_circles))
        return circles[:max_circles]
    
    bounds = geom.bounds
    area = geom.area
    avg_radius = np.sqrt(area / np.pi) / 2
    radius = max(min_radius, avg_radius)
    
    step = radius * 1.5
    circles = []
    
    x = bounds[0]
    while x <= bounds[2] and len(circles) < max_circles:
        y = bounds[1]
        while y <= bounds[3] and len(circles) < max_circles:
            center = Point(x, y)
            if geom.contains(center) or geom.distance(center) < radius * 0.5:
                circles.append((x, y, radius))
            y += step
        x += step
        
    if not circles:
        centroid = geom.centroid
        circles.append((centroid.x, centroid.y, max(min_radius, np.sqrt(area / np.pi))))
    
    return circles

def _process_osm_features(environment, gdf_proj, default_height_m=10.0, distance_m=2000):
    """
    Process OSM features:
    1. Send exact geometry to Environment for Fire Grid rasterization.
    2. Create visual objects for PyBullet.
    """
    if gdf_proj is None or len(gdf_proj) == 0:
        print("⚠️  No features to process")
        return
    
    print("    Processing map data...")

    # --- 1. DATA PREPARATION & FILTERING (For Fire Physics) ---
    
    # A. WATER (High priority - Firebreak)
    # Filter: Natural water, waterways (rivers/canals), reservoirs.
    # EXCLUDE: Small streams (ditches, drains, streams)
    mask_water = (
        (gdf_proj['natural'].isin(['water', 'wetland'])) |
        (gdf_proj['waterway'].isin(['river', 'canal', 'dock', 'riverbank'])) |
        (gdf_proj['landuse'].isin(['reservoir', 'basin']))
    )
    if 'waterway' in gdf_proj.columns:
        # Explicitly remove small streams
        mask_water = mask_water & (~gdf_proj['waterway'].isin(['stream', 'ditch', 'drain']))
        
    gdf_water = gdf_proj[mask_water].copy()

    # B. BUILDINGS (Medium priority - Slow burn)
    mask_buildings = (
        (gdf_proj['building'].notna() & (gdf_proj['building'] != 'no')) |
        (gdf_proj['landuse'].isin(['residential', 'commercial', 'industrial', 'retail']))
    )
    gdf_buildings = gdf_proj[mask_buildings].copy()

    # C. FOREST (Low priority - Fuel)
    mask_forest = (
        (gdf_proj['landuse'].isin(['forest', 'orchard', 'vineyard', 'wood'])) |
        (gdf_proj['natural'].isin(['wood', 'scrub', 'heath']))
    )
    gdf_forest = gdf_proj[mask_forest].copy()

    print(f"    Found for physics engine:")
    print(f"      💧 Water bodies (Rivers/Lakes): {len(gdf_water)}")
    print(f"      🏢 Buildings/Urban: {len(gdf_buildings)}")
    print(f"      🌲 Forests: {len(gdf_forest)}")

    # --- 2. RASTERIZATION (Send to Environment) ---
    # This creates the internal representation for fire spread (Painter's Algo)
    environment.rasterize_terrain_layers(gdf_water, gdf_buildings, gdf_forest)

    # --- 3. VISUALIZATION (Create PyBullet Objects) ---
    # This loop is purely for the 3D view, it does NOT affect fire logic anymore.
    print("    Creating 3D visual objects...")
    
    # Visuals: Buildings
    for _, row in gdf_buildings.iterrows():
        geom = row.geometry
        if geom is None: continue
        
        centroid = geom.centroid
        pos_xy = [centroid.x, centroid.y]
        
        # Determine height
        height = default_height_m
        if 'height' in row and row['height'] is not None:
            try: height = float(str(row['height']).replace('m', ''))
            except: pass
        elif 'building:levels' in row and row['building:levels'] is not None:
            try: height = int(row['building:levels']) * 3.0
            except: pass
            
        min_x, min_y, max_x, max_y = geom.bounds
        size = [max_x - min_x, max_y - min_y, height]
        environment.add_city_block(pos_xy, size)

    # Visuals: Forests (Approximated as circles)
    for _, row in gdf_forest.iterrows():
        geom = row.geometry
        if geom is None: continue
        circles = _approximate_polygon_with_circles(geom, min_radius=10.0, max_circles=5)
        for cx, cy, rad in circles:
            if rad > 5.0:
                # Add simplified forest visual
                tree_count = int((np.pi * rad**2) / 100)
                environment.add_forest_area([cx, cy], rad, tree_count=max(3, min(tree_count, 15)))

    # Visuals: Lakes (Approximated as circles or rectangles)
    for _, row in gdf_water.iterrows():
        geom = row.geometry
        if geom is None: continue
        
        # Heuristic: If it's a long linestring/polygon (river), use rectangles
        # For simplicity in visualizer, we often approximate everything with circles 
        # or just add the main bodies.
        circles = _approximate_polygon_with_circles(geom, min_radius=10.0, max_circles=5)
        for cx, cy, rad in circles:
            if rad > 8.0:
                environment.add_lake([cx, cy], rad)

    print(f"✅ Map processing complete.")

def load_environment_from_osm_cache(environment: Environment, 
                                   cache_dir: str,
                                   region_prefix: str,
                                   center_lat: float,
                                   center_lon: float,
                                   radius_m: float = 1500,
                                   default_height_m: float = 10.0):
    """
    Load OSM data from pre-downloaded cache files.
    """
    import glob
    
    print(f"📂 Loading from cache: {cache_dir}/{region_prefix}_*")
    
    cache_files = {
        'building': f"{cache_dir}/{region_prefix}_building_*.gpkg",
        'landuse': f"{cache_dir}/{region_prefix}_landuse_*.gpkg",
        'natural': f"{cache_dir}/{region_prefix}_natural_*.gpkg",
        'waterway': f"{cache_dir}/{region_prefix}_waterway_*.gpkg"
    }
    
    gdfs = {}
    for category, pattern in cache_files.items():
        matches = glob.glob(pattern)
        if matches:
            try:
                gdf = gpd.read_file(matches[0])
                gdfs[category] = gdf
            except Exception as e:
                print(f"   ⚠️  Could not load {category}: {e}")
    
    if not gdfs:
        raise FileNotFoundError(f"No cache files found in {cache_dir}")
    
    # Projection setup
    center_point = Point(center_lon, center_lat)
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = f'EPSG:326{utm_zone:02d}' if center_lat >= 0 else f'EPSG:327{utm_zone:02d}'
    
    center_gdf = gpd.GeoSeries([center_point], crs='EPSG:4326')
    center_proj = center_gdf.to_crs(utm_crs).iloc[0]
    
    filtered_gdfs = {}
    for category, gdf in gdfs.items():
        gdf_proj = gdf.to_crs(utm_crs)
        gdf_proj = gdf_proj[gdf_proj.distance(center_proj) <= radius_m]
        gdf_proj['geometry'] = gdf_proj.translate(xoff=-center_proj.x, yoff=-center_proj.y)
        filtered_gdfs[category] = gdf_proj
    
    combined_gdf = gpd.GeoDataFrame(
        pd.concat([gdf for gdf in filtered_gdfs.values() if len(gdf) > 0], ignore_index=True)
    )
    
    _process_osm_features(environment, combined_gdf, default_height_m, radius_m)

def load_environment_from_osm(environment: Environment, location: str, default_height_m: float = 10.0, radius_m: float = 1500):
    """
    Downloads map data from OSM and populates environment.
    """
    print(f"🌍 Downloading map data for '{location}'...")
    
    try:
        center_point = ox.geocode(location)
        center_lat, center_lon = center_point
    except Exception as e:
        print(f"❌ Failed to geocode location: {e}")
        return
    
    tags = {
        'landuse': ['residential', 'commercial', 'industrial', 'forest', 'grass', 'meadow', 'reservoir'],
        'natural': ['wood', 'water', 'wetland', 'scrub'],
        'waterway': ['river', 'stream', 'canal', 'drain'],
        'building': True
    }

    try:
        try:
            gdf = ox.features_from_point(center_point, tags=tags, dist=radius_m)
        except AttributeError:
            gdf = ox.geometries_from_point(center_point, tags=tags, dist=radius_m)
    except Exception as e:
        print(f"❌ Failed to download OSM data: {e}")
        return
    
    # Projection and centering
    try:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    except AttributeError:
        gdf_proj = ox.project_geometries(gdf, to_crs=gdf.estimate_utm_crs())
    
    center_geom = Point(center_lon, center_lat)
    gdf_center = gpd.GeoDataFrame([{'geometry': center_geom}], crs='EPSG:4326')
    gdf_center_proj = gdf_center.to_crs(gdf_proj.crs)
    
    center_x = gdf_center_proj.geometry.iloc[0].x
    center_y = gdf_center_proj.geometry.iloc[0].y
    
    gdf_proj['geometry'] = gdf_proj['geometry'].translate(xoff=-center_x, yoff=-center_y)
    
    _process_osm_features(environment, gdf_proj, default_height_m, radius_m)