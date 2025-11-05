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

def load_environment_from_osm(environment: Environment, location_query: str, default_height_m: float = 10.0, 
                            distance_m: float = 2000, use_city_boundaries: bool = True):
    """
    Downloads map data from OpenStreetMap for a given location and populates
    the simulation environment with obstacles and terrain.

    Args:
        environment: An instance of the Environment class to populate.
        location_query: The name of the location (e.g., "Tišnov, Czech Republic", "Manhattan, New York City").
        default_height_m: The height to assign buildings that don't have height data.
        distance_m: Radius in meters around the location center to download (default: 2000m = 2km).
        use_city_boundaries: If True, use city/residential boundaries instead of individual buildings (much faster).
    """
    print(f"🌍 Downloading map data for '{location_query}'...")
    print(f"   Radius: {distance_m}m ({distance_m/1000:.1f} km) around center")
    print(f"   Mode: {'City boundaries' if use_city_boundaries else 'Individual buildings'}")
    
    # --- 1. Download Data ---
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
        buildings_added = 0
        forests_added = 0
        lakes_added = 0
        grass_areas_added = 0
        urban_areas_added = 0
        
        print("🏗️  Converting OSM data to simulation environment...")
        print(f"   Total features downloaded: {len(gdf_proj)}")
        
        # Debug: Check what types of features we have
        if 'landuse' in gdf_proj.columns:
            landuse_types = gdf_proj['landuse'].value_counts()
            print(f"   Landuse types found: {dict(landuse_types)}")
        if 'natural' in gdf_proj.columns:
            natural_types = gdf_proj['natural'].value_counts()
            print(f"   Natural types found: {dict(natural_types)}")
        if 'waterway' in gdf_proj.columns:
            waterway_types = gdf_proj['waterway'].value_counts()
            print(f"   Waterway types found: {dict(waterway_types)}")
        
        for _, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom is None or not geom.is_valid:
                continue

            # --- Handle Urban Areas (residential/commercial/industrial) ---
            # Process these as large non-burnable blocks (instead of individual buildings)
            if 'landuse' in row and pd.notna(row.get('landuse')) and \
               row['landuse'] in ['residential', 'commercial', 'industrial']:
                
                # Get centroid for position
                centroid = geom.centroid
                pos_xy = [centroid.x, centroid.y]
                
                # Use average building height for urban areas
                height = default_height_m
                
                # Get size (bounding box) - treat entire area as one large block
                min_x, min_y, max_x, max_y = geom.bounds
                size = [max_x - min_x, max_y - min_y, height]
                
                # Add the urban area as a large city block
                environment.add_city_block(pos_xy, size)
                urban_areas_added += 1
            
            # --- Handle Individual Buildings (only if not using city boundaries) ---
            # Check if 'building' column exists AND has a non-null value
            elif 'building' in row and pd.notna(row['building']) and row['building']:
                # Get centroid for position
                centroid = geom.centroid
                pos_xy = [centroid.x, centroid.y]
                
                # Get height
                height = default_height_m
                if 'height' in row and row['height'] is not None:
                    try:
                        # Handle values like '15m'
                        height = float(str(row['height']).replace('m', ''))
                    except ValueError:
                        pass
                elif 'building:levels' in row and row['building:levels'] is not None:
                    try:
                        # Estimate height at 3m per level
                        height = int(row['building:levels']) * 3.0
                    except ValueError:
                        pass
                
                # Get size (bounding box) - buildings use rectangular approximation
                min_x, min_y, max_x, max_y = geom.bounds
                size = [max_x - min_x, max_y - min_y, height]
                
                # Add the building to the environment
                environment.add_city_block(pos_xy, size)
                buildings_added += 1
            
            # --- Handle Forests (use multiple circles for better shape approximation) ---
            elif ('landuse' in row and pd.notna(row.get('landuse')) and row['landuse'] in ['forest']) or \
                 ('natural' in row and pd.notna(row.get('natural')) and row['natural'] in ['wood']):
                
                # Get multiple circles to approximate the irregular forest shape
                circles = _approximate_polygon_with_circles(geom, min_radius=10.0, max_circles=8)
                
                for center_x, center_y, radius in circles:
                    if radius > 5.0:  # Ignore very small circles
                        # Calculate tree count based on circle area
                        circle_area = np.pi * radius * radius
                        tree_count = int(circle_area / 50)  # 1 tree per 50 sq. m
                        environment.add_forest_area([center_x, center_y], radius, 
                                                   tree_count=max(5, min(tree_count, 30)))
                        forests_added += 1
            
            # --- Handle Water (Lakes - use multiple circles for better shape) ---
            elif 'natural' in row and pd.notna(row.get('natural')) and row['natural'] in ['water', 'wetland']:
                # Get multiple circles to approximate the irregular water body shape
                circles = _approximate_polygon_with_circles(geom, min_radius=15.0, max_circles=5)
                
                for center_x, center_y, radius in circles:
                    if radius > 10.0:  # Ignore very small water bodies
                        environment.add_lake([center_x, center_y], radius)
                        lakes_added += 1
            
            # --- Handle Waterways (Rivers/Streams - linear features) ---
            elif 'waterway' in row and pd.notna(row.get('waterway')) and row['waterway'] in ['river', 'stream', 'canal']:
                # Rivers are linear features (LineString or MultiLineString)
                # We'll approximate them as a series of circles along the path
                from shapely.geometry import LineString, MultiLineString
                
                # Debug: Check geometry type
                geom_type = type(geom).__name__
                
                if isinstance(geom, (LineString, MultiLineString)):
                    # For rivers, create circles along the line
                    # Determine river width based on type
                    if row['waterway'] == 'river':
                        river_width = 20.0  # 20m radius for rivers
                    elif row['waterway'] == 'canal':
                        river_width = 10.0  # 10m radius for canals
                    else:  # stream
                        river_width = 5.0   # 5m radius for streams
                    
                    # Sample points along the waterway
                    if isinstance(geom, MultiLineString):
                        lines = list(geom.geoms)
                    else:
                        lines = [geom]
                    
                    circles_added = 0
                    for line in lines:
                        # Sample points every 30m along the waterway
                        length = line.length
                        num_points = max(2, int(length / 30))
                        
                        for i in range(num_points):
                            fraction = i / (num_points - 1) if num_points > 1 else 0.5
                            point = line.interpolate(fraction, normalized=True)
                            environment.add_lake([point.x, point.y], river_width)
                            lakes_added += 1
                            circles_added += 1
                    
                    # Debug output for each waterway processed
                    if circles_added > 0:
                        print(f"   🌊 Added {row['waterway']}: {circles_added} circles, width={river_width}m")
                else:
                    print(f"   ⚠️  Waterway {row['waterway']} has unexpected geometry type: {geom_type}")
            
            # --- Handle Grass/Meadow (noted but not explicitly added - default terrain) ---
            elif 'landuse' in row and pd.notna(row.get('landuse')) and row['landuse'] in ['grass', 'meadow']:
                # Grass/meadow areas are treated as default burnable terrain
                # No explicit geometry needed - fire grid will handle this
                grass_areas_added += 1
        
        print(f"✅ Map data loaded successfully!")
        if use_city_boundaries:
            print(f"   - Added {urban_areas_added} urban areas (residential/commercial/industrial)")
            if buildings_added > 0:
                print(f"   - Added {buildings_added} individual buildings (outside urban zones)")
        else:
            print(f"   - Added {buildings_added} individual buildings (non-burnable obstacles)")
        print(f"   - Added {forests_added} forest circles (high fuel areas)")
        print(f"   - Added {lakes_added} water body circles (fire breaks)")
        print(f"   - Found {grass_areas_added} grass/meadow areas (default terrain)")
        print(f"   ℹ️  Note: Forests and water use multiple circles to approximate real shapes")

    except Exception as e:
        print(f"❌ Failed to download or process map data: {e}")
        print("   -> Please check your internet connection and location query.")