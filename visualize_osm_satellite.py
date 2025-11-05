#!/usr/bin/env python3
"""
Download and display actual OpenStreetMap satellite imagery for the simulation area.
This shows the REAL satellite view, not our internal representation.
"""

import matplotlib
matplotlib.use('Agg')  # Headless mode - no GUI

import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point, box

# Configuration - should match your demo parameters
LOCATION = "Tišnov, Czech Republic"
RADIUS_M = 1000  # meters

print("=" * 70)
print("🗺️  DOWNLOADING REAL OPENSTREETMAP SATELLITE IMAGERY")
print("=" * 70)

# Get the location coordinates
print(f"\n📍 Geocoding location: {LOCATION}")
try:
    location = ox.geocode(LOCATION)
    print(f"   Coordinates: (lat={location[0]}, lon={location[1]})")
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

# Convert to Web Mercator for contextily
gdf_bbox_web = gdf_bbox_utm.to_crs('EPSG:3857')
gdf_point_web = gdf_point.to_crs('EPSG:3857')

# Get bounds
bounds = gdf_bbox_utm.total_bounds
print(f"   Bounds (UTM): {bounds}")
print(f"   Size: {bounds[2]-bounds[0]:.0f}m x {bounds[3]-bounds[1]:.0f}m")

print("\n🖼️  Creating satellite imagery map...")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

ax.set_title(f"Real Satellite Imagery\n{LOCATION}\n(±{RADIUS_M}m radius)", 
             fontsize=16, fontweight='bold', pad=20)

# Plot bounding box (red square)
gdf_bbox_web.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=4, 
                  label=f'Simulation area (±{RADIUS_M}m)')

# Plot center point (red X)
gdf_point_web.plot(ax=ax, color='red', markersize=400, marker='x', 
                   linewidths=5, label='Center (0,0 in simulation)', zorder=10)

# Add satellite basemap
print("   Downloading satellite imagery tiles...")
try:
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom='auto')
    print("   ✅ Esri WorldImagery loaded")
except Exception as e:
    print(f"   ⚠️  Esri failed: {e}")
    print("   Trying Google Satellite...")
    try:
        ctx.add_basemap(ax, source=ctx.providers.Google.Satellite, zoom='auto')
        print("   ✅ Google Satellite loaded")
    except Exception as e2:
        print(f"   ⚠️  Google failed: {e2}")
        print("   Trying OpenStreetMap...")
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto')
            print("   ✅ OpenStreetMap loaded (no satellite)")
        except Exception as e3:
            print(f"   ❌ All basemaps failed: {e3}")

ax.legend(fontsize=14, loc='upper right', framealpha=0.9)
ax.set_xlabel("Easting (Web Mercator)", fontsize=12)
ax.set_ylabel("Northing (Web Mercator)", fontsize=12)

# Add scale bar text
ax.text(0.02, 0.02, f"Scale: ±{RADIUS_M}m from center",
        transform=ax.transAxes, fontsize=12, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save figure
output_path = "output/real_osm_satellite.png"
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\n✅ Satellite map saved to: {output_path}")

print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"Location: {LOCATION}")
print(f"Center: (lat={location[0]}, lon={location[1]})")
print(f"Simulation area: ±{RADIUS_M}m (red square)")
print(f"Center marker: Red X marks (0,0) in your simulation")
print(f"\n💡 TIP: Compare this with output/demo_01_environment.png")
print("   to see how your simulation represents the real terrain")
print("=" * 70)
