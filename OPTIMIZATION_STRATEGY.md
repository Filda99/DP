# Fire Simulation Optimization Strategy

## Current Performance Issues

### Problem 1: Too Many Objects (5,749 buildings!)
- **Current**: Download 3km radius → 5,749 buildings + 406 forests + 42 lakes
- **Solution**: Focus on smaller area or use city boundaries

### Problem 2: Large Fire Grid (300×300 = 90,000 cells)
- **Current**: Every cell checked every timestep
- **Solution**: Only process cells near active fires

### Problem 3: Nested Loops Everywhere
- **Current**: O(n²) complexity in multiple places
- **Solution**: Spatial indexing and vectorization

---

## Optimization Options

### Option A: Focus on Single Forest Area (RECOMMENDED for testing)

Instead of downloading entire city, focus on one forest:

```python
# Current (slow):
sim.setup_osm_environment("Tišnov, Czech Republic", distance_m=3000)
# → 5,749 buildings, 406 forests

# Optimized (fast):
sim.setup_osm_environment("Klucanina, Tišnov", distance_m=500)
# → ~50 buildings, ~10 forests
```

**Benefits:**
- 100x fewer buildings to process
- Smaller fire grid (e.g., 1000m × 1000m instead of 6000m)
- 10x faster initialization
- Still realistic terrain!

### Option B: Use City Boundaries Instead of Individual Buildings

Query OSM for city/district boundaries:

```python
tags = {
    'boundary': ['administrative'],  # City boundaries
    'landuse': ['forest', 'residential'],  # Forest + residential areas (not individual buildings)
    'natural': ['wood', 'water']
}
```

**Benefits:**
- 1 city boundary polygon instead of 5,749 buildings
- Mark entire residential areas as non-burnable
- Much faster processing

### Option C: Adaptive Grid Resolution

Different cell sizes for different terrain:

```
Cities/Lakes:     100m × 100m cells (coarse)
Open areas:       20m × 20m cells (medium)
Forests:          5m × 5m cells (fine - where fire spreads)
```

**Benefits:**
- Total cells: ~10,000 instead of 90,000
- Most cells are coarse (fast to process)
- Fine resolution only where needed (forests)

### Option D: Only Process Active Fire Regions

Currently processes all 90,000 cells every timestep. Instead:

```python
# Only check cells within 10 cells of any burning cell
active_region_mask = expand_mask(fire_grid.B, radius=10)
cells_to_check = np.where(active_region_mask)
# Process only ~500-1000 cells instead of 90,000!
```

**Benefits:**
- 100x speedup for fire spread calculation
- Early in simulation: few burning cells → very fast
- Late in simulation: many burning cells → still faster than full grid

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Do First!)

1. **Reduce download area**
   ```python
   distance_m=500  # Instead of 3000
   ```

2. **Smaller fire grid**
   ```python
   grid_width_m=1000   # Instead of 6000
   cell_size_m=10.0     # Instead of 20.0
   ```

3. **Use city boundaries instead of buildings**
   - Change OSM tags to get city polygons
   - Mark entire residential areas as non-burnable

### Phase 2: Fire Grid Optimization

1. **Process only active regions**
   - Track bounding box of burning cells
   - Only check cells within 10-20 cells of fire
   - Skip 90% of grid calculations

2. **Vectorize terrain checks**
   - Pre-compute terrain type for each cell (building/forest/water/grass)
   - Use NumPy boolean arrays instead of loops

### Phase 3: Advanced (If Needed)

1. **Adaptive mesh refinement**
   - Implement multi-resolution grid
   - Fine cells for forests, coarse for cities

2. **Spatial indexing**
   - Use quadtree or R-tree for fast object lookups
   - O(log n) instead of O(n) for terrain queries

---

## Example: Optimized Configuration

### For Testing Fire Spread in Forest

```python
# Focus on specific forest area
sim.setup_osm_environment(
    "Forest near Tišnov, Czech Republic",
    distance_m=500,  # Just 500m radius
    default_building_height=8.0
)

# Smaller, higher-resolution grid for forest details
sim.enable_fire_simulation(
    grid_width_m=1000,   # 1km × 1km
    grid_height_m=1000,
    cell_size_m=5.0       # 5m cells for detail
)
# Result: 200×200 = 40,000 cells (instead of 90,000)
```

### For City Fire Simulation

```python
# Use city boundaries
sim.setup_osm_environment(
    "Tišnov, Czech Republic",
    distance_m=1000,
    use_building_clusters=True  # NEW: cluster buildings into districts
)

# Coarse grid for urban area
sim.enable_fire_simulation(
    grid_width_m=2000,
    grid_height_m=2000,
    cell_size_m=20.0  # Coarse cells
)
# Result: 100×100 = 10,000 cells
```

---

## Performance Comparison

| Configuration | Buildings | Grid Cells | Init Time | Step Time | Total (10s sim) |
|--------------|-----------|------------|-----------|-----------|-----------------|
| **Current** (Tišnov 3km) | 5,749 | 90,000 | ~60s | ~2s | **TIMEOUT** |
| **Option A** (Forest 500m) | ~50 | 40,000 | ~5s | ~0.5s | **~10s** ✅ |
| **Option B** (City bounds) | ~10 | 10,000 | ~2s | ~0.2s | **~4s** ✅ |
| **Option D** (Active region) | 5,749 | 90,000 | ~60s | ~0.1s | **~61s** ⚠️ |

---

## Next Steps

1. **Try Option A first** (focus on forest) - easiest and fastest
2. **Implement active region processing** (Option D) - biggest speedup
3. **Consider city boundaries** (Option B) if you need urban areas
4. **Adaptive grid** (Option C) is advanced but powerful

Would you like me to implement any of these optimizations?
