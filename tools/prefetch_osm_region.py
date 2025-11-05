#!/usr/bin/env python3
"""
Prefetch OSM Region Data

Downloads and caches large OSM regions (e.g., entire Moravia) for offline use.
Run this ONCE to download data, then your simulations can use the cached data.

Usage:
    python tools/prefetch_osm_region.py --region "Jihomoravský kraj, Czechia"
    python tools/prefetch_osm_region.py --bbox 49.5,48.5,18.0,15.5
"""

import osmnx as ox
import geopandas as gpd
import argparse
import os
from pathlib import Path


def prefetch_region(region_name=None, bbox=None, output_dir="data"):
    """
    Download OSM data for a large region and save to GeoPackage files.
    
    Args:
        region_name: Name of region (e.g., "Jihomoravský kraj, Czechia")
        bbox: Tuple of (north, south, east, west) coordinates
        output_dir: Directory to save data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define tags to download
    tags = {
        'building': True,
        'landuse': ['forest', 'residential', 'commercial', 'industrial'],
        'natural': ['wood'],
        'waterway': ['river', 'stream', 'canal']
    }
    
    print("=" * 70)
    print("🗺️  OSM Region Prefetch Tool")
    print("=" * 70)
    
    # Determine query method
    if region_name:
        print(f"📍 Region: {region_name}")
        query_str = region_name.replace(" ", "_").replace(",", "")
    elif bbox:
        north, south, east, west = bbox
        print(f"📦 Bounding Box: N={north}, S={south}, E={east}, W={west}")
        query_str = f"bbox_{north}_{south}_{east}_{west}"
    else:
        raise ValueError("Must provide either region_name or bbox")
    
    # Download each feature type separately (more efficient)
    for tag_key, tag_value in tags.items():
        print(f"\n🔍 Downloading: {tag_key} = {tag_value}")
        
        try:
            if region_name:
                gdf = ox.features_from_place(region_name, tags={tag_key: tag_value})
            else:
                gdf = ox.features_from_bbox(
                    bbox=(north, south, east, west),
                    tags={tag_key: tag_value}
                )
            
            # Save to GeoPackage
            if isinstance(tag_value, list):
                tag_str = "_".join(tag_value)
            else:
                tag_str = str(tag_value)
            
            filename = f"{output_dir}/{query_str}_{tag_key}_{tag_str}.gpkg"
            gdf.to_file(filename, driver="GPKG")
            
            print(f"   ✅ Saved {len(gdf)} features to {filename}")
            print(f"   📊 Size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not download {tag_key}: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Prefetch complete!")
    print(f"📁 Data saved to: {output_dir}/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download OSM data for large regions"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        help="Region name (e.g., 'Jihomoravský kraj, Czechia')"
    )
    
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as 'north,south,east,west' (e.g., '49.5,48.5,18.0,15.5')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for cached data (default: data)"
    )
    
    args = parser.parse_args()
    
    # Parse bbox if provided
    bbox = None
    if args.bbox:
        try:
            bbox = tuple(map(float, args.bbox.split(',')))
            if len(bbox) != 4:
                raise ValueError("Bbox must have 4 values")
        except Exception as e:
            print(f"❌ Error parsing bbox: {e}")
            print("   Format: 'north,south,east,west'")
            return
    
    # Validate input
    if not args.region and not bbox:
        print("❌ Error: Must provide either --region or --bbox")
        parser.print_help()
        return
    
    # Run prefetch
    prefetch_region(
        region_name=args.region,
        bbox=bbox,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
