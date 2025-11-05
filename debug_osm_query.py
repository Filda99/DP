#!/usr/bin/env python3
"""
Debug: What OSM features did we actually download for Křivoklát?
"""
import osmnx as ox

LOCATION = "Křivoklát, Czech Republic"
RADIUS_M = 1000

print("=" * 70)
print("🔍 DEBUGGING OSM DATA FOR KŘIVOKLÁT")
print("=" * 70)

center = ox.geocode(LOCATION)
print(f"\n📍 Location: {LOCATION}")
print(f"   Center: {center}")

# What we're currently downloading
tags_current = {
    'landuse': ['residential', 'commercial', 'industrial', 'forest', 'grass', 'meadow'],
    'natural': ['wood', 'water', 'wetland']
}

print(f"\n📥 Current query tags:")
for key, values in tags_current.items():
    print(f"   {key}: {values}")

gdf = ox.features_from_point(center, tags=tags_current, dist=RADIUS_M)
print(f"\n✅ Downloaded {len(gdf)} features")

# Count by type
print("\n📊 Feature breakdown:")
if 'landuse' in gdf.columns:
    print(f"\nLanduse tags:")
    for value, count in gdf['landuse'].value_counts().items():
        print(f"   {value}: {count}")

if 'natural' in gdf.columns:
    print(f"\nNatural tags:")
    for value, count in gdf['natural'].value_counts().items():
        print(f"   {value}: {count}")

if 'waterway' in gdf.columns:
    print(f"\nWaterway tags:")
    for value, count in gdf['waterway'].value_counts().items():
        print(f"   {value}: {count}")
else:
    print(f"\n⚠️  NO WATERWAY DATA! This is why the river is missing!")

# Now try querying for waterways separately
print("\n" + "=" * 70)
print("🌊 TRYING TO DOWNLOAD WATERWAYS (rivers, streams)")
print("=" * 70)

try:
    waterway_tags = {'waterway': True}
    gdf_water = ox.features_from_point(center, tags=waterway_tags, dist=RADIUS_M)
    print(f"✅ Found {len(gdf_water)} waterway features!")
    
    if 'waterway' in gdf_water.columns:
        print(f"\nWaterway types:")
        for value, count in gdf_water['waterway'].value_counts().items():
            print(f"   {value}: {count}")
except Exception as e:
    print(f"❌ Error downloading waterways: {e}")

print("\n" + "=" * 70)
print("💡 CONCLUSION")
print("=" * 70)
print("The problem is likely:")
print("1. We're NOT querying for 'waterway' tag → rivers missing!")
print("2. OSM may not have detailed forest polygons in this area")
print("3. We need to add 'waterway': ['river', 'stream'] to our query")
print("=" * 70)
