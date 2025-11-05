# Environment Visualization Guide

## Overview

The simulation now supports **real-world map data from OpenStreetMap (OSM)**. This guide shows you how to visualize and verify the environment before running fire simulations.

## Files Created

### 1. Environment Visualization Tools

- **`visualize_environment.py`** - Comprehensive environment analysis tool
  - Shows buildings, forests, water bodies
  - Displays fuel distribution
  - Suggests fire starting locations
  - Creates detailed multi-panel analysis

- **`test_osm_conversion.py`** - Quick test of OSM data loading

### 2. Updated Demos

All three demos now save environment maps automatically:

- **`demos/demo_01_fire_spread.py`** → Creates `output/demo_01_environment.png`
- **`demos/demo_02_water_suppression.py`** → Creates `output/demo_02_environment.png`
- **`demos/demo_03_physics.py`** → Creates `output/demo_03_environment.png`

### 3. New Environment Method

Added to `src/environment.py`:

```python
environment.save_environment_map(filename, show_fire_grid=True)
```

This creates a top-down visualization showing:
- **Gray rectangles** = Buildings (non-burnable)
- **Green circles** = Forests (high fuel)
- **Blue circles** = Water bodies (fire breaks)
- **Color overlay** = Fire grid fuel levels

## How to Use

### Basic Visualization

```bash
# Visualize Tišnov environment
python visualize_environment.py
```

Output files:
- `output/env_Tišnov_Czech_Republic.png` - Map view
- `output/env_Tišnov_Czech_Republic_detailed.png` - Fuel analysis

### Customize Location

Edit `visualize_environment.py`:

```python
LOCATION = "Brno, Czech Republic"  # Change this
GRID_SIZE = 1000  # Larger area
CELL_SIZE = 10.0   # Bigger cells
```

### In Your Code

```python
from src.simulation import Simulation

sim = Simulation(gui=False)
sim.start_simulation()

# Load real map data
sim.setup_osm_environment("Your Location Here")

# Enable fire grid
sim.enable_fire_simulation(grid_width_m=500, grid_height_m=500, cell_size_m=5.0)

# Save visualization
sim.environment.save_environment_map("my_map.png", show_fire_grid=True)
```

## What the Visualization Shows

### Environment Map
- **Buildings**: Exact locations and sizes from OSM
- **Forests**: Approximated with overlapping circles
- **Water**: Approximated with overlapping circles
- **Fire Grid**: Overlay showing fuel levels at each cell

### Statistics Panel
- Number of buildings, forests, water bodies
- Fire grid size and resolution
- Fuel distribution percentages
- Suggested fire starting locations

### Detailed Analysis (visualize_environment.py)
- **Panel 1**: Fuel burn rate heatmap
- **Panel 2**: Binary burnable/non-burnable map
- **Panel 3**: Fuel distribution histogram

## Example Output

Running `python visualize_environment.py` for **Tišnov, Czech Republic**:

```
🏗️  Physical Objects:
   Buildings: 3,741
   Forest areas: 0
   Water bodies: 0

🔥 Fire Grid (100×100 = 10,000 cells):
   No fuel (buildings/water): 8,731 cells (87.3%)
   Low fuel (grass/open):     0 cells (0.0%)
   High fuel (forest):        1,269 cells (12.7%)

🎯 Suggested Fire Starting Locations:
   1. (  37.5,  102.5) - forest (fuel=0.080)
   2. ( 227.5,  127.5) - forest (fuel=0.080)
   ...
```

## Workflow

1. **Load OSM data** for your location
2. **Visualize environment** to see what was loaded
3. **Check fuel distribution** to verify forests/buildings
4. **Find fire locations** from suggested coordinates
5. **Run simulation** with realistic fire starting points

## Tips

- **Urban areas** (like Tišnov center) have mostly buildings
- **Parks/forests** have better fire spread potential
- Try locations like:
  - "Golden Gate Park, San Francisco" (lots of forests)
  - "Yosemite National Park" (wilderness)
  - "Central Park, New York" (mixed urban/forest)

## Next Steps

After visualizing the environment:
1. Note the suggested fire starting locations
2. Update your demo scripts with those coordinates
3. Run the simulation and watch realistic fire spread!

The fire will:
- ✅ Spread faster in forests
- ✅ Stop at buildings
- ✅ Stop at water bodies  
- ✅ Spread slower in open/grass areas
