"""
Map Importer from OpenStreetMap (OSM)

This module uses osmnx to download real-world map data (buildings,
forests, water) and populates the simulation Environment.

The conversion process:
1. Buildings → Rectangular blocks (using bounding box)
2. Forests → Multiple circular approximations for irregular shapes
3. Water → Multiple circular approximations for irregular shapes
4. Grass/meadow/other → Treated as default burnable terrain
"""

import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
import random
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

# Try to import the Environment class, handling potential path issues
try:
    from .environment import Environment
except ImportError:
    from src.environment import Environment

def _approximate_polygon_with_circles(geom, min_radius=5.0, max_circles=10):
    """
    Approximate an irregular polygon with multiple overlapping circles.
    This gives a better representation than a single circle.
    
    Args:
        geom: Shapely geometry (Polygon or MultiPolygon)
        min_radius: Minimum radius for circles (meters)
        max_circles: Maximum number of circles to use
    
    Returns:
        List of (center_x, center_y, radius) tuples
    """
    if isinstance(geom, MultiPolygon):
        # Handle multiple polygons separately
        circles = []
        for poly in geom.geoms:
            circles.extend(_approximate_polygon_with_circles(poly, min_radius, max_circles))
        return circles[:max_circles]  # Limit total circles
    
    # For simple polygon, use a grid-based approach
    bounds = geom.bounds  # (minx, miny, maxx, maxy)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    # Calculate appropriate circle size based on polygon size
    area = geom.area
    avg_radius = np.sqrt(area / np.pi) / 2  # Smaller circles for better coverage
    radius = max(min_radius, avg_radius)
    
    # Create a grid of potential circle centers
    step = radius * 1.5  # Overlap circles slightly
    circles = []
    
    x = bounds[0]
    while x <= bounds[2] and len(circles) < max_circles:
        y = bounds[1]
        while y <= bounds[3] and len(circles) < max_circles:
            center = Point(x, y)
            # Only add circle if center is inside or very close to the polygon
            if geom.contains(center) or geom.distance(center) < radius * 0.5:
                circles.append((x, y, radius))
            y += step
        x += step
    
    # If no circles were added (very small polygon), use centroid
    if not circles:
        centroid = geom.centroid
        circles.append((centroid.x, centroid.y, max(min_radius, np.sqrt(area / np.pi))))
    
    return circles


def _process_osm_features(environment, gdf_proj,
                          default_height_m=10.0, use_city_boundaries=True, distance_m=2000):
    """
    Process OSM GeoDataFrame and populate the environment.
    
    This function handles the common logic for both cache and live downloads.
    
    Args:
        environment: Environment instance to populate
        gdf_proj: GeoDataFrame with all OSM features (already projected to UTM and centered)
        default_height_m: Default building height
        use_city_boundaries: Whether to merge buildings into boundaries
        distance_m: Max distance for filtering (waterways)
    """
    if gdf_proj is None or len(gdf_proj) == 0:
        print("⚠️  No features to process")
        return
    
    # Counters
    buildings_added = 0
    forests_added = 0
    lakes_added = 0
    grass_areas_added = 0
    urban_areas_added = 0
    
    print("🏗️  Converting OSM data to simulation environment...")
    print(f"   Total features to process: {len(gdf_proj)}")
    
    # Debug: Check what types of features we have
    if 'landuse' in gdf_proj.columns:
        landuse_types = gdf_proj['landuse'].value_counts()
        print(f"   Landuse types: {dict(landuse_types)}")
    if 'natural' in gdf_proj.columns:
        natural_types = gdf_proj['natural'].value_counts()
        print(f"   Natural types: {dict(natural_types)}")
    if 'waterway' in gdf_proj.columns:
        waterway_types = gdf_proj['waterway'].value_counts()
        print(f"   Waterway types: {dict(waterway_types)}")
    if 'building' in gdf_proj.columns:
        building_count = gdf_proj['building'].notna().sum()
        print(f"   Buildings: {building_count}")
    
    # Process each feature
    for _, row in gdf_proj.iterrows():
        geom = row.geometry
        if geom is None or not geom.is_valid:
            continue

        # --- Handle Urban Areas (residential/commercial/industrial) ---
        if 'landuse' in row and pd.notna(row.get('landuse')) and \
           row['landuse'] in ['residential', 'commercial', 'industrial']:
            
            centroid = geom.centroid
            pos_xy = [centroid.x, centroid.y]
            height = default_height_m
            
            min_x, min_y, max_x, max_y = geom.bounds
            size = [max_x - min_x, max_y - min_y, height]
            
            environment.add_city_block(pos_xy, size)
            urban_areas_added += 1
        
        # --- Handle Individual Buildings ---
        elif 'building' in row and pd.notna(row['building']) and row['building']:
            centroid = geom.centroid
            pos_xy = [centroid.x, centroid.y]
            
            height = default_height_m
            if 'height' in row and row['height'] is not None:
                try:
                    height = float(str(row['height']).replace('m', ''))
                except ValueError:
                    pass
            elif 'building:levels' in row and row['building:levels'] is not None:
                try:
                    height = int(row['building:levels']) * 3.0
                except ValueError:
                    pass
            
            min_x, min_y, max_x, max_y = geom.bounds
            size = [max_x - min_x, max_y - min_y, height]
            
            environment.add_city_block(pos_xy, size)
            buildings_added += 1
        
        # --- Handle Forests ---
        elif ('landuse' in row and pd.notna(row.get('landuse')) and row['landuse'] in ['forest']) or \
             ('natural' in row and pd.notna(row.get('natural')) and row['natural'] in ['wood']):
            
            circles = _approximate_polygon_with_circles(geom, min_radius=10.0, max_circles=8)
            
            for center_x, center_y, radius in circles:
                if radius > 5.0:
                    circle_area = np.pi * radius * radius
                    tree_count = int(circle_area / 50)
                    environment.add_forest_area([center_x, center_y], radius, 
                                               tree_count=max(5, min(tree_count, 30)))
                    forests_added += 1
        
        # --- Handle Water (Lakes) ---
        elif 'natural' in row and pd.notna(row.get('natural')) and row['natural'] in ['water', 'wetland']:
            circles = _approximate_polygon_with_circles(geom, min_radius=15.0, max_circles=5)
            
            for center_x, center_y, radius in circles:
                if radius > 10.0:
                    environment.add_lake([center_x, center_y], radius)
                    lakes_added += 1
        
        # --- Handle Waterways (Rivers/Streams) ---
        elif 'waterway' in row and pd.notna(row.get('waterway')) and row['waterway'] in ['river', 'stream', 'canal']:
            from shapely.geometry import LineString, MultiLineString
            
            if isinstance(geom, (LineString, MultiLineString)):
                if row['waterway'] == 'river':
                    river_width = 40.0
                elif row['waterway'] == 'canal':
                    river_width = 20.0
                else:
                    river_width = 10.0
                
                if isinstance(geom, MultiLineString):
                    lines = list(geom.geoms)
                else:
                    lines = [geom]
                
                rectangles_added = 0
                max_distance = distance_m * 1.2
                
                for line in lines:
                    coords = list(line.coords)
                    
                    for i in range(len(coords) - 1):
                        x1, y1 = coords[i]
                        x2, y2 = coords[i + 1]
                        
                        dist1 = np.sqrt(x1**2 + y1**2)
                        dist2 = np.sqrt(x2**2 + y2**2)
                        
                        if dist1 > max_distance and dist2 > max_distance:
                            continue
                        
                        dx = x2 - x1
                        dy = y2 - y1
                        segment_length = np.sqrt(dx**2 + dy**2)
                        
                        if segment_length < 1.0:
                            continue
                        
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        angle = np.arctan2(dy, dx)
                        
                        environment.add_water_rectangle(
                            center=[center_x, center_y],
                            length=segment_length,
                            width=river_width,
                            angle=angle
                        )
                        lakes_added += 1
                        rectangles_added += 1
                
                if rectangles_added > 0:
                    print(f"   🌊 Added {row['waterway']}: {rectangles_added} rectangles, width={river_width}m")
        
        # --- Handle Grass/Meadow ---
        elif 'landuse' in row and pd.notna(row.get('landuse')) and row['landuse'] in ['grass', 'meadow']:
            grass_areas_added += 1
    
    print(f"✅ Map data loaded successfully!")
    if use_city_boundaries:
        print(f"   - Added {urban_areas_added} urban areas (residential/commercial/industrial)")
        if buildings_added > 0:
            print(f"   - Added {buildings_added} individual buildings (outside urban zones)")
    else:
        print(f"   - Added {buildings_added} individual buildings (non-burnable obstacles)")
    print(f"   - Added {forests_added} forest circles (high fuel areas)")
    print(f"   - Added {lakes_added} water body rectangles/circles (fire breaks)")
    print(f"   - Found {grass_areas_added} grass/meadow areas (default terrain)")


def load_environment_from_osm_cache(environment: Environment, 
                                   cache_dir: str,
                                   region_prefix: str,
                                   center_lat: float,
                                   center_lon: float,
                                   radius_m: float = 1500,
                                   default_height_m: float = 10.0,
                                   use_city_boundaries: bool = True):
    """
    Load OSM data from pre-downloaded cache files and populate environment.
    
    This is MUCH faster than downloading, and works offline!
    
    Args:
        environment: Environment instance to populate
        cache_dir: Directory containing .gpkg cache files (e.g., "data")
        region_prefix: Prefix of cached files (e.g., "Jihomoravský_kraj_Czechia")
        center_lat: Latitude of simulation center
        center_lon: Longitude of simulation center
        radius_m: Radius around center to extract (meters)
        default_height_m: Default building height
        use_city_boundaries: Whether to merge buildings into boundaries
    
    Example:
        >>> load_environment_from_osm_cache(
        ...     environment=env,
        ...     cache_dir="data",
        ...     region_prefix="Jihomoravský_kraj_Czechia",
        ...     center_lat=49.1951,
        ...     center_lon=16.6068,  # Brno
        ...     radius_m=1500
        ... )
    """
    import os
    import glob
    
    print(f"📂 Loading from cache: {cache_dir}/{region_prefix}_*")
    print(f"   Center: ({center_lat:.4f}°N, {center_lon:.4f}°E)")
    print(f"   Radius: {radius_m}m")
    
    # Find cache files
    cache_files = {
        'building': f"{cache_dir}/{region_prefix}_building_*.gpkg",
        'landuse': f"{cache_dir}/{region_prefix}_landuse_*.gpkg",
        'natural': f"{cache_dir}/{region_prefix}_natural_*.gpkg",
        'waterway': f"{cache_dir}/{region_prefix}_waterway_*.gpkg"
    }
    
    # Load each category
    gdfs = {}
    for category, pattern in cache_files.items():
        matches = glob.glob(pattern)
        if matches:
            try:
                gdf = gpd.read_file(matches[0])
                print(f"   ✅ Loaded {category}: {len(gdf)} features")
                gdfs[category] = gdf
            except Exception as e:
                print(f"   ⚠️  Could not load {category}: {e}")
        else:
            print(f"   ⚠️  No cache file for {category} (pattern: {pattern})")
    
    if not gdfs:
        raise FileNotFoundError(f"No cache files found in {cache_dir} with prefix {region_prefix}")
    
    # Create point for center
    center_point = Point(center_lon, center_lat)
    
    # Project to UTM for metric calculations
    # Determine UTM zone from longitude
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = f'EPSG:326{utm_zone:02d}' if center_lat >= 0 else f'EPSG:327{utm_zone:02d}'
    
    print(f"   🗺️  Projecting to {utm_crs}...")
    
    # Project center point
    center_gdf = gpd.GeoSeries([center_point], crs='EPSG:4326')
    center_proj = center_gdf.to_crs(utm_crs).iloc[0]
    
    # Filter and project each GeoDataFrame to extract only features within radius
    filtered_gdfs = {}
    for category, gdf in gdfs.items():
        # Project to UTM
        gdf_proj = gdf.to_crs(utm_crs)
        
        # Filter by distance from center
        gdf_proj = gdf_proj[gdf_proj.distance(center_proj) <= radius_m]
        
        # Translate so center is at origin (0, 0)
        gdf_proj['geometry'] = gdf_proj.translate(
            xoff=-center_proj.x,
            yoff=-center_proj.y
        )
        
        filtered_gdfs[category] = gdf_proj
        print(f"   📍 Filtered {category}: {len(gdf_proj)} features within {radius_m}m")
    
    # Combine all filtered GeoDataFrames into one
    combined_gdf = gpd.GeoDataFrame(
        pd.concat([gdf for gdf in filtered_gdfs.values() if len(gdf) > 0], 
                  ignore_index=True)
    )
    
    # Now call the existing processing logic
    _process_osm_features(
        environment=environment,
        gdf_proj=combined_gdf,
        default_height_m=default_height_m,
        use_city_boundaries=use_city_boundaries,
        distance_m=radius_m
    )


def load_environment_from_osm(environment: Environment, location_query: str, default_height_m: float = 10.0, 
                            distance_m: float = 2000, use_city_boundaries: bool = True,
                            regional_cache_file: str = None):
    """
    Downloads map data from OpenStreetMap for a given location and populates
    the simulation environment with obstacles and terrain.

    Args:
        environment: An instance of the Environment class to populate.
        location_query: The name of the location (e.g., "Tišnov, Czech Republic", "Manhattan, New York City").
        default_height_m: The height to assign buildings that don't have height data.
        distance_m: Radius in meters around the location center to download (default: 2000m = 2km).
        use_city_boundaries: If True, use city/residential boundaries instead of individual buildings (much faster).
        regional_cache_file: Optional path to pre-downloaded regional .gpkg file (faster, offline).
    """
    if regional_cache_file:
        print(f"🗂️  Loading from regional cache: {regional_cache_file}")
        print(f"   Location: '{location_query}', Radius: {distance_m}m")
    else:
        print(f"🌍 Downloading map data for '{location_query}'...")
        print(f"   Radius: {distance_m}m ({distance_m/1000:.1f} km) around center")
    print(f"   Mode: {'City boundaries' if use_city_boundaries else 'Individual buildings'}")
    
    # --- 1. Download or Load Data ---
    if regional_cache_file:
        # Load from pre-downloaded regional file
        from src.regional_cache import RegionalMapCache
        
        cache = RegionalMapCache(regional_cache_file)
        gdf = cache.get_subregion_by_location_name(location_query, radius_m=distance_m)
        
        # Get center point for coordinate system
        center_point = ox.geocode(location_query)
        print(f"📍 Center coordinates: {center_point}")
        
    else:
        # Original: download from OSM API
        # Choose between detailed buildings or city boundaries
        if use_city_boundaries:
            # Download city/residential boundaries instead of individual buildings
            tags = {
                'landuse': ['residential', 'commercial', 'industrial', 'forest', 'grass', 'meadow'],
                'natural': ['wood', 'water', 'wetland'],
                'waterway': ['river', 'stream', 'canal']  # Add rivers/streams!
            }
        else:
            # Original: download individual buildings
            tags = {
                'building': True,
                'landuse': ['forest', 'grass', 'meadow'],
                'natural': ['wood', 'water', 'wetland'],
                'waterway': ['river', 'stream', 'canal']  # Add rivers/streams!
            }
        
        try:
            # First, geocode the location to get coordinates
            try:
                from shapely.geometry import Point
                center_point = ox.geocode(location_query)
                print(f"📍 Center coordinates: {center_point}")
            except Exception as e:
                print(f"⚠️  Could not geocode location: {e}")
                print(f"   Trying with place boundary instead...")
                center_point = None
            
            # Download geometries (polygons) for the specified tags
            # Note: osmnx v1.x+ uses 'features_from_place' instead of 'geometries_from_place'
            try:
                if center_point:
                    # Download using point + distance (better for getting surrounding areas)
                    gdf = ox.features_from_point(center_point, tags=tags, dist=distance_m)
                else:
                    # Fallback to place boundary
                    gdf = ox.features_from_place(location_query, tags=tags)
            except AttributeError:
                # Fallback for older osmnx versions
                if center_point:
                    gdf = ox.geometries_from_point(center_point, tags=tags, dist=distance_m)
                else:
                    gdf = ox.geometries_from_place(location_query, tags=tags)
        except Exception as e:
            print(f"❌ Failed to download OSM data: {e}")
            print("   -> Please check your internet connection and location query.")
            return
    
    # Common processing for both cache and live download
    try:
        # --- 2. Project to Meters (UTM) ---
        # Convert from Lat/Lon (degrees) to a local UTM projection (meters)
        try:
            # Try newer osmnx API
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
        except AttributeError:
            # Fallback for older versions
            gdf_proj = ox.project_geometries(gdf, to_crs=gdf.estimate_utm_crs())
        
        # --- 3. Center the Map at (0, 0) ---
        # Use the QUERIED LOCATION as center, not the geometric centroid of features
        # (otherwise large forests can shift the center away from the city!)
        if center_point:
            # Project the center point to UTM
            from shapely.geometry import Point
            center_geom = Point(center_point[1], center_point[0])  # lon, lat
            gdf_center = gpd.GeoDataFrame([{'geometry': center_geom}], crs='EPSG:4326')
            gdf_center_proj = gdf_center.to_crs(gdf_proj.crs)
            center_x = gdf_center_proj.geometry.iloc[0].x
            center_y = gdf_center_proj.geometry.iloc[0].y
        else:
            # Fallback: use geometric center of all features
            map_center = gdf_proj.unary_union.centroid
            center_x, center_y = map_center.x, map_center.y
        
        print(f"📍 Map center (UTM): ({center_x:.2f}, {center_y:.2f}). Centering at (0,0) for simulation.")
        
        # Translate all geometries so the map center is at (0, 0)
        gdf_proj['geometry'] = gdf_proj['geometry'].translate(xoff=-center_x, yoff=-center_y)
        
        # --- 4. Populate Environment ---
        # Use the shared processing function
        _process_osm_features(
            environment=environment,
            gdf_proj=gdf_proj,
            default_height_m=default_height_m,
            use_city_boundaries=use_city_boundaries,
            distance_m=distance_m
        )

    except Exception as e:
        print(f"❌ Failed to download or process map data: {e}")
        print("   -> Please check your internet connection and location query.")