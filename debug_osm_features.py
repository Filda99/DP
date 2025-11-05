#!/usr/bin/env python3
"""
Debug: Show exactly where each OSM feature is located
"""

import matplotlib
matplotlib.use('Agg')

import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

LOCATION = "Tišnov, Czech Republic"
RADIUS_M = 1000

print("=" * 70)
print("🔍 DEBUGGING OSM FEATURE LOCATIONS")
print("=" * 70)

# Download data
print(f"\n📍 Downloading data for: {LOCATION}")
center = ox.geocode(LOCATION)
print(f"   Center: {center}")

tags = {
    'landuse': ['residential', 'commercial', 'industrial', 'forest', 'grass', 'meadow'],
    'natural': ['wood', 'water']
}

gdf = ox.features_from_point(center, tags=tags, dist=RADIUS_M)
print(f"   Downloaded {len(gdf)} features")

# Project to UTM
gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

# Center at (0,0)
map_center = gdf_proj.unary_union.centroid
gdf_proj['geometry'] = gdf_proj['geometry'].translate(xoff=-map_center.x, yoff=-map_center.y)

# Categorize features
gdf_proj['category'] = 'other'
for idx, row in gdf_proj.iterrows():
    if 'landuse' in row and pd.notna(row.get('landuse')):
        if row['landuse'] in ['residential', 'commercial', 'industrial']:
            gdf_proj.at[idx, 'category'] = 'urban'
        elif row['landuse'] == 'forest':
            gdf_proj.at[idx, 'category'] = 'forest'
        elif row['landuse'] in ['grass', 'meadow']:
            gdf_proj.at[idx, 'category'] = 'grass'
    
    if 'natural' in row and pd.notna(row.get('natural')):
        if row['natural'] == 'wood':
            gdf_proj.at[idx, 'category'] = 'forest'
        elif row['natural'] == 'water':
            gdf_proj.at[idx, 'category'] = 'water'

# Print summary with locations
print("\n📊 FEATURE LOCATIONS:")
print("-" * 70)

for category in ['urban', 'forest', 'water', 'grass']:
    subset = gdf_proj[gdf_proj['category'] == category]
    if len(subset) > 0:
        print(f"\n{category.upper()}: {len(subset)} features")
        for idx, row in subset.iterrows():
            centroid = row.geometry.centroid
            bounds = row.geometry.bounds
            size_x = bounds[2] - bounds[0]
            size_y = bounds[3] - bounds[1]
            
            tag_info = ""
            if 'landuse' in row and pd.notna(row.get('landuse')):
                tag_info += f"landuse={row['landuse']} "
            if 'natural' in row and pd.notna(row.get('natural')):
                tag_info += f"natural={row['natural']} "
            
            print(f"  - Center: ({centroid.x:6.0f}, {centroid.y:6.0f})  "
                  f"Size: {size_x:5.0f}×{size_y:5.0f}m  {tag_info}")

# Create visualization
print("\n🖼️  Creating visualization...")
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# Plot by category with different colors
colors = {
    'urban': 'gray',
    'forest': 'green',
    'water': 'blue',
    'grass': 'lightgreen',
    'other': 'orange'
}

for category, color in colors.items():
    subset = gdf_proj[gdf_proj['category'] == category]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, alpha=0.6, label=f'{category} ({len(subset)})')

# Mark center
ax.plot(0, 0, 'rx', markersize=20, markeredgewidth=3, label='Center (0,0)')

# Draw ±1000m box
from matplotlib.patches import Rectangle
rect = Rectangle((-RADIUS_M, -RADIUS_M), 2*RADIUS_M, 2*RADIUS_M, 
                 linewidth=2, edgecolor='red', facecolor='none', 
                 label='Simulation area')
ax.add_patch(rect)

ax.set_xlim(-RADIUS_M*1.1, RADIUS_M*1.1)
ax.set_ylim(-RADIUS_M*1.1, RADIUS_M*1.1)
ax.set_xlabel("X (meters)", fontsize=12)
ax.set_ylabel("Y (meters)", fontsize=12)
ax.set_title(f"OSM Features by Category\n{LOCATION}", fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

output_path = 'output/debug_osm_features.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

print("\n" + "=" * 70)
