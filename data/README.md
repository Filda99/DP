# 🗺️ OSM Data Prefetch

This directory stores pre-downloaded OpenStreetMap data for faster simulations.

## 📋 Workflow

### **Step 1: Download Region (ONE-TIME)**
```bash
# Download entire Moravia
python tools/prefetch_osm_region.py --region "Jihomoravský kraj, Czechia"

# Or download by bounding box
python tools/prefetch_osm_region.py --bbox 49.5,48.5,18.0,15.5

# Or download smaller area (faster for testing)
python tools/prefetch_osm_region.py --region "Křivoklát, Czechia"
```

### **Step 2: Use Cached Data**
```python
# In your simulation code
from src.map_importer import load_environment_from_osm_cache

# Load from cache instead of downloading
load_environment_from_osm_cache(
    environment=env,
    cache_dir="data",
    region_prefix="Jihomoravský_kraj",  # Matches downloaded files
    center_lat=49.2,
    center_lon=16.6,
    radius_m=1500,
    ...
)
```

## 📦 File Format

Files are saved as GeoPackage (`.gpkg`) format:
- `{region}_building_True.gpkg` - Buildings
- `{region}_landuse_*.gpkg` - Land use (forests, residential, etc.)
- `{region}_natural_*.gpkg` - Natural features (woods)
- `{region}_waterway_*.gpkg` - Waterways (rivers, streams)

## 🎯 Benefits

✅ **Offline work** - No internet needed after prefetch
✅ **Faster** - No repeated downloads for nearby locations  
✅ **Consistent** - Same data for all simulations in region
✅ **Large areas** - Download entire regions once

## 📊 Storage

Typical sizes:
- **Small town** (Křivoklát): ~0.8 MB
- **City** (Brno): ~20-50 MB
- **Region** (Jihomoravský kraj): ~200-500 MB

## ⚠️ Note

These files are in `.gitignore` - they won't be committed to git due to size.
Each user should run prefetch for their regions of interest.
