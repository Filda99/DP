#!/usr/bin/env python3
"""
Download and display actual OpenStreetMap imagery for the simulation area.
This shows the REAL map view, not our internal representation.
"""

import matplotlib
matplotlib.use('Agg')  # Headless mode - no GUI

import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import Point, box

# Configuration - should match your demo parameters
LOCATION = "Tišnov, Czech Republic"
RADIUS_M = 1000  # meters

print("=" * 70)
print("🗺️  DOWNLOADING REAL OPENSTREETMAP IMAGERY")
print("=" * 70)

# Get the location coordinates
print(f"\n📍 Geocoding location: {LOCATION}")
try:
    location = ox.geocode(LOCATION)
    print(f"   Coordinates: {location}")
except Exception as e:
    print(f"❌ Error geocoding: {e}")
    exit(1)

# Create a GeoDataFrame with the center point
point = Point(location[1], location[0])  # longitude, latitude
gdf_point = gpd.GeoDataFrame([{'geometry': point}], crs='EPSG:4326')

# Create a bounding box around the point
print(f"\n📦 Creating {RADIUS_M}m radius bounding box...")
# Convert to UTM for metric calculations
gdf_point_utm = gdf_point.to_crs(gdf_point.estimate_utm_crs())
point_utm = gdf_point_utm.geometry.iloc[0]

# Create box around point (±radius)
bbox_utm = box(
    point_utm.x - RADIUS_M,
    point_utm.y - RADIUS_M,
    point_utm.x + RADIUS_M,
    point_utm.y + RADIUS_M
)
gdf_bbox_utm = gpd.GeoDataFrame([{'geometry': bbox_utm}], crs=gdf_point_utm.crs)

# Convert back to WGS84 for display
gdf_bbox = gdf_bbox_utm.to_crs('EPSG:4326')

# Get bounds
bounds = gdf_bbox.total_bounds
print(f"   Bounds: {bounds}")

print("\n🌍 Downloading OpenStreetMap features for context...")
try:
    # Download features for overlay - simplified to just key features
    tags = {
        'landuse': ['forest', 'residential', 'commercial', 'industrial'],
        'natural': ['wood', 'water']
    }
    
    gdf_features = ox.features_from_point(
        (location[0], location[1]),
        tags=tags,
        dist=RADIUS_M
    )
    
    # Convert to Web Mercator for contextily
    gdf_features_web = gdf_features.to_crs('EPSG:3857')
    gdf_bbox_web = gdf_bbox.to_crs('EPSG:3857')
    gdf_point_web = gdf_point.to_crs('EPSG:3857')
    
    print(f"   Downloaded {len(gdf_features)} features")
    
except Exception as e:
    print(f"⚠️  Could not download features: {e}")
    gdf_features_web = None
    gdf_bbox_web = gdf_bbox.to_crs('EPSG:3857')
    gdf_point_web = gdf_point.to_crs('EPSG:3857')

print("\n🖼️  Creating map visualization...")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# LEFT: Satellite imagery
ax1.set_title(f"Real Satellite Imagery\n{LOCATION}\n(±{RADIUS_M}m radius)", 
              fontsize=14, fontweight='bold')

# Plot bounding box
gdf_bbox_web.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=3)

# Plot center point
gdf_point_web.plot(ax=ax1, color='red', markersize=200, marker='x', 
                   label='Center (0,0 in simulation)', zorder=5)

# Add basemap - satellite imagery
try:
    ctx.add_basemap(ax1, source=ctx.providers.Esri.WorldImagery, zoom=15)
    print("   ✅ Satellite imagery loaded")
except Exception as e:
    print(f"   ⚠️  Could not load satellite imagery: {e}")
    print("   Trying alternative source...")
    try:
        ctx.add_basemap(ax1, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15)
        print("   ✅ OpenStreetMap loaded")
    except Exception as e2:
        print(f"   ❌ Could not load basemap: {e2}")

ax1.legend(fontsize=12)
ax1.set_xlabel("Longitude (Web Mercator)", fontsize=10)
ax1.set_ylabel("Latitude (Web Mercator)", fontsize=10)

# RIGHT: OpenStreetMap standard view with feature overlay
ax2.set_title(f"OpenStreetMap Features\n{LOCATION}\n(±{RADIUS_M}m radius)", 
              fontsize=14, fontweight='bold')

# Plot bounding box
gdf_bbox_web.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=3)

# Plot features if available - simplified for speed
if gdf_features_web is not None and len(gdf_features_web) > 0:
    print(f"   Plotting {len(gdf_features_web)} features...")
    
    # Group by type for faster plotting
    for feature_type in ['forest', 'wood', 'water', 'residential', 'commercial', 'industrial']:
        if feature_type in ['forest']:
            mask = (gdf_features_web.get('landuse') == 'forest')
            color = 'darkgreen'
        elif feature_type == 'wood':
            mask = (gdf_features_web.get('natural') == 'wood')
            color = 'green'
        elif feature_type == 'water':
            mask = (gdf_features_web.get('natural') == 'water')
            color = 'blue'
        elif feature_type in ['residential', 'commercial', 'industrial']:
            mask = (gdf_features_web.get('landuse') == feature_type)
            color = 'gray'
        else:
            continue
        
        subset = gdf_features_web[mask]
        if len(subset) > 0:
            subset.plot(ax=ax2, color=color, alpha=0.6, edgecolor='none')
            print(f"      {feature_type}: {len(subset)} features")

# Plot center point
gdf_point_web.plot(ax=ax2, color='red', markersize=200, marker='x', 
                   label='Center (0,0)', zorder=5)

# Add basemap - standard OSM
try:
    ctx.add_basemap(ax2, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15)
    print("   ✅ OpenStreetMap loaded")
except Exception as e:
    print(f"   ⚠️  Could not load OpenStreetMap: {e}")

ax2.legend(fontsize=12)
ax2.set_xlabel("Longitude (Web Mercator)", fontsize=10)
ax2.set_ylabel("Latitude (Web Mercator)", fontsize=10)

# Save figure
output_path = "output/real_osm_map.png"
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Map saved to: {output_path}")

print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"Location: {LOCATION}")
print(f"Center coordinates: {location}")
print(f"Radius: {RADIUS_M}m")
print(f"Bounding box (lat/lon): {bounds}")
if gdf_features_web is not None:
    print(f"Features downloaded: {len(gdf_features)}")
print(f"\nRed 'X' marks the center (0,0) in your simulation")
print(f"Red box shows the ±{RADIUS_M}m simulation area")
print("=" * 70)

# Don't show - headless mode
# plt.show()
