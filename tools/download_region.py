"""
Download Large OSM Region

Downloads entire region (e.g., Moravia, South Moravia) from OSM
and saves it as GeoPackage for offline use.

Usage:
    python tools/download_region.py --region "Jihomoravský kraj, Czechia"
    python tools/download_region.py --bbox 49.5 48.5 18.0 15.5  # north south east west
"""

import osmnx as ox
import geopandas as gpd
import argparse
import os
from pathlib import Path

# Tags to download
TAGS = {
    'building': True,
    'landuse': ['forest', 'residential', 'commercial', 'industrial', 'grass', 'meadow'],
    'natural': ['wood', 'water'],
    'waterway': ['river', 'stream', 'canal']
}

def download_by_place(place_name: str, output_dir: str = "data/regions"):
    """Download OSM data for a named place (e.g., kraj, city)."""
    print(f"📥 Downloading OSM data for: {place_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize filename
    safe_name = place_name.replace(" ", "_").replace(",", "").replace(".", "")
    output_file = os.path.join(output_dir, f"{safe_name}.gpkg")
    
    print(f"   Fetching features...")
    gdf = ox.features_from_place(place_name, tags=TAGS)
    
    print(f"   Downloaded {len(gdf)} features")
    print(f"   Saving to: {output_file}")
    
    # Save as GeoPackage (efficient format)
    gdf.to_file(output_file, driver="GPKG")
    
    print(f"✅ Saved {len(gdf)} features to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    return output_file

def download_by_bbox(north: float, south: float, east: float, west: float, 
                     name: str = "custom_region", output_dir: str = "data/regions"):
    """Download OSM data for a bounding box."""
    print(f"📥 Downloading OSM data for bounding box:")
    print(f"   North: {north}, South: {south}, East: {east}, West: {west}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{name}.gpkg")
    
    print(f"   Fetching features...")
    gdf = ox.features_from_bbox(bbox=(north, south, east, west), tags=TAGS)
    
    print(f"   Downloaded {len(gdf)} features")
    print(f"   Saving to: {output_file}")
    
    # Save as GeoPackage
    gdf.to_file(output_file, driver="GPKG")
    
    print(f"✅ Saved {len(gdf)} features to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    return output_file

def load_region(gpkg_file: str):
    """Load previously downloaded region from GeoPackage."""
    print(f"📂 Loading region from: {gpkg_file}")
    gdf = gpd.read_file(gpkg_file)
    print(f"   Loaded {len(gdf)} features")
    return gdf

def extract_subregion(gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float, 
                     radius_m: float = 1500):
    """
    Extract a circular subregion from the large GeoDataFrame.
    
    Args:
        gdf: Full region GeoDataFrame
        center_lat, center_lon: Center point (GPS coordinates)
        radius_m: Radius in meters
    
    Returns:
        Filtered GeoDataFrame containing only features in the subregion
    """
    print(f"🔍 Extracting subregion:")
    print(f"   Center: ({center_lat}, {center_lon})")
    print(f"   Radius: {radius_m}m")
    
    # Create point and buffer (circle)
    from shapely.geometry import Point
    
    # Convert to appropriate CRS for buffering (need meters)
    center_point = gpd.GeoSeries([Point(center_lon, center_lat)], crs='EPSG:4326')
    
    # Reproject to UTM for accurate distance (determine zone from lon/lat)
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'north' if center_lat >= 0 else 'south'
    epsg_code = f"326{utm_zone}" if hemisphere == 'north' else f"327{utm_zone}"
    
    center_utm = center_point.to_crs(f'EPSG:{epsg_code}')
    circle = center_utm.buffer(radius_m)
    
    # Reproject GeoDataFrame to same CRS
    gdf_utm = gdf.to_crs(f'EPSG:{epsg_code}')
    
    # Filter features that intersect with circle
    mask = gdf_utm.intersects(circle.iloc[0])
    filtered_gdf = gdf[mask].copy()
    
    print(f"   Found {len(filtered_gdf)} features in subregion")
    print(f"   Reduction: {len(gdf)} → {len(filtered_gdf)} ({100*len(filtered_gdf)/len(gdf):.1f}%)")
    
    return filtered_gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OSM region data")
    parser.add_argument("--region", type=str, help="Region name (e.g., 'Jihomoravský kraj, Czechia')")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=('N', 'S', 'E', 'W'),
                       help="Bounding box: north south east west")
    parser.add_argument("--name", type=str, default="custom_region", 
                       help="Output filename (without extension)")
    parser.add_argument("--output-dir", type=str, default="data/regions",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.region:
        download_by_place(args.region, args.output_dir)
    elif args.bbox:
        n, s, e, w = args.bbox
        download_by_bbox(n, s, e, w, args.name, args.output_dir)
    else:
        print("❌ Error: Must specify either --region or --bbox")
        parser.print_help()
