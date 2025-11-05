"""
Regional Map Cache

Loads OSM data from pre-downloaded regional files (GeoPackage)
instead of downloading from OSM API each time.

This allows:
1. Offline operation
2. Faster loading (no network delay)
3. Consistent data across runs
4. Extract any subregion from large dataset
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pathlib import Path

class RegionalMapCache:
    """Manages pre-downloaded OSM regional data."""
    
    def __init__(self, gpkg_file: str):
        """
        Initialize cache from GeoPackage file.
        
        Args:
            gpkg_file: Path to .gpkg file (created by download_region.py)
        """
        self.gpkg_file = gpkg_file
        self.gdf = None
        self.loaded = False
        
    def load(self):
        """Load GeoDataFrame from file (lazy loading)."""
        if not self.loaded:
            print(f"📂 Loading regional cache: {self.gpkg_file}")
            self.gdf = gpd.read_file(self.gpkg_file)
            print(f"   ✅ Loaded {len(self.gdf)} features")
            self.loaded = True
        return self.gdf
    
    def get_subregion_by_point(self, lat: float, lon: float, radius_m: float = 1500):
        """
        Extract circular subregion around a point.
        
        Args:
            lat, lon: Center point (GPS coordinates)
            radius_m: Radius in meters
            
        Returns:
            GeoDataFrame with features in subregion
        """
        if not self.loaded:
            self.load()
        
        print(f"🔍 Extracting subregion: ({lat}, {lon}), radius={radius_m}m")
        
        # Create circle buffer
        center_point = gpd.GeoSeries([Point(lon, lat)], crs='EPSG:4326')
        
        # Determine UTM zone for accurate buffering
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = 'north' if lat >= 0 else 'south'
        epsg_code = f"326{utm_zone}" if hemisphere == 'north' else f"327{utm_zone}"
        
        # Reproject and buffer
        center_utm = center_point.to_crs(f'EPSG:{epsg_code}')
        circle = center_utm.buffer(radius_m).iloc[0]
        
        # Reproject GeoDataFrame
        if self.gdf.crs != f'EPSG:{epsg_code}':
            gdf_utm = self.gdf.to_crs(f'EPSG:{epsg_code}')
        else:
            gdf_utm = self.gdf
        
        # Filter by intersection
        mask = gdf_utm.intersects(circle)
        subregion = self.gdf[mask].copy()
        
        print(f"   Found {len(subregion)} features ({100*len(subregion)/len(self.gdf):.2f}%)")
        
        return subregion
    
    def get_subregion_by_bbox(self, north: float, south: float, east: float, west: float):
        """
        Extract rectangular subregion by bounding box.
        
        Args:
            north, south, east, west: GPS coordinates
            
        Returns:
            GeoDataFrame with features in bbox
        """
        if not self.loaded:
            self.load()
        
        print(f"🔍 Extracting bbox subregion: ({north},{south},{east},{west})")
        
        # Filter by bounds
        mask = (
            (self.gdf.geometry.bounds['minx'] <= east) &
            (self.gdf.geometry.bounds['maxx'] >= west) &
            (self.gdf.geometry.bounds['miny'] <= north) &
            (self.gdf.geometry.bounds['maxy'] >= south)
        )
        
        subregion = self.gdf[mask].copy()
        
        print(f"   Found {len(subregion)} features ({100*len(subregion)/len(self.gdf):.2f}%)")
        
        return subregion
    
    def get_subregion_by_location_name(self, location_name: str, radius_m: float = 1500):
        """
        Extract subregion by geocoding a location name.
        
        Args:
            location_name: Name of location (e.g., "Brno, Czechia")
            radius_m: Radius around location
            
        Returns:
            GeoDataFrame with features in subregion
        """
        import osmnx as ox
        
        # Geocode location
        lat, lon = ox.geocode(location_name)
        print(f"   '{location_name}' → ({lat}, {lon})")
        
        return self.get_subregion_by_point(lat, lon, radius_m)

# Example usage functions

def example_download_south_moravia():
    """Example: Download entire South Moravian Region."""
    from tools.download_region import download_by_place
    
    # Download entire region (may take several minutes)
    gpkg_file = download_by_place("Jihomoravský kraj, Czechia", output_dir="data/regions")
    
    return gpkg_file

def example_extract_brno():
    """Example: Extract Brno from pre-downloaded region."""
    
    # Assume we already downloaded South Moravia
    cache = RegionalMapCache("data/regions/Jihomoravský_krajCzechia.gpkg")
    
    # Extract Brno center (1.5km radius)
    brno_data = cache.get_subregion_by_location_name("Brno, Czechia", radius_m=1500)
    
    return brno_data

def example_extract_custom_area():
    """Example: Extract custom area from region."""
    
    cache = RegionalMapCache("data/regions/Jihomoravský_krajCzechia.gpkg")
    
    # Extract by exact coordinates
    lat, lon = 49.1951, 16.6068  # Brno coordinates
    data = cache.get_subregion_by_point(lat, lon, radius_m=2000)
    
    return data
